"""Screen candidate v1 vector codec designs (subgroup scales, 2D vs 4D, 100% residual) on blocks 0, 20, 41."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

VERSION_V1 = "hades_vq_v1_screen"


def nearest(points: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    norms = table.square().sum(1)
    return torch.cat([(norms - 2 * part @ table.T).argmin(1) for part in points.split(16384)])


def fit_codebook(points: torch.Tensor, dim: int, iters: int = 10) -> torch.Tensor:
    table = points[:256].clone()
    pts_cpu = points.cpu()
    for _ in range(iters):
        indices = nearest(points, table).cpu()
        sums = torch.zeros(256, dim).index_add_(0, indices, pts_cpu)
        counts = torch.bincount(indices, minlength=256)
        active = counts > 0
        updated = table.cpu()
        updated[active] = sums[active] / counts[active, None]
        table = updated.cuda()
    return table.half().float()


def top_mask(score: torch.Tensor) -> torch.Tensor:
    indices = score.flatten().topk(score.numel() // 4, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.flatten()[indices] = True
    return mask


def encode_matrix(
    w: torch.Tensor,
    cal_x: torch.Tensor | None,
    group_size: int,
    dim: int,
    residual_mode: str,
    rng: torch.Generator,
):
    """Encode one 2D matrix with subgroup RMS scales and 1- or 2-stage VQ."""
    rows, width = w.shape
    if width % group_size != 0 or group_size % dim != 0:
        raise ValueError(f"Incompatible shapes: width={width}, group_size={group_size}, dim={dim}")

    # Disjoint dictionary training sample (256 rows) vs full matrix encoding.
    train_idx = torch.randperm(rows, generator=rng)[: min(256, rows)]
    w_train = w[train_idx].cuda().float()

    def normalize_subgroups(x: torch.Tensor):
        xg = x.reshape(x.shape[0], width // group_size, group_size)
        sc = xg.square().mean(-1).sqrt().half().float().clamp_min(1e-7)
        normed = (xg / sc[..., None]).reshape(x.shape[0], width // dim, dim)
        return normed, sc

    train_norm, _ = normalize_subgroups(w_train)
    pts = train_norm.reshape(-1, dim)
    perm = torch.randperm(len(pts), generator=rng)[: min(16384, len(pts))].cuda()
    pts = pts[perm]

    base = fit_codebook(pts, dim)
    residual = None
    if residual_mode in ("r25_act", "r100"):
        residual = fit_codebook(pts - base[nearest(pts, base)], dim)

    w_gpu = w.cuda().float()
    normed, scales = normalize_subgroups(w_gpu)
    vecs_per_sub = group_size // dim
    scales_exp = scales.repeat_interleave(vecs_per_sub, dim=1)[..., None]

    main_chunks, res_chunks, gain_chunks = [], [], []
    if residual_mode == "r25_act":
        assert cal_x is not None
        x_in = cal_x.cuda().float().reshape(-1, width // dim, dim)
        cov = torch.einsum("ngi,ngj->gij", x_in, x_in) / len(x_in)

    for start in range(0, rows, 256):
        end = min(start + 256, rows)
        t_chunk = normed[start:end]
        sc_chunk = scales_exp[start:end]
        w_chunk = w_gpu[start:end].reshape(end - start, width // dim, dim)

        c_idx = nearest(t_chunk.reshape(-1, dim), base).reshape(t_chunk.shape[:2])
        main_chunks.append(c_idx.byte())

        if residual_mode == "none":
            continue

        r_idx = nearest((t_chunk - base[c_idx]).reshape(-1, dim), residual).reshape_as(c_idx)
        res_chunks.append(r_idx.byte())

        if residual_mode == "r25_act":
            e0 = w_chunk - (base[c_idx] * sc_chunk).bfloat16().float()
            e1 = w_chunk - ((base[c_idx] + residual[r_idx]) * sc_chunk).bfloat16().float()
            imp0 = torch.einsum("rgi,gij,rgj->rg", e0, cov, e0)
            imp1 = torch.einsum("rgi,gij,rgj->rg", e1, cov, e1)
            gain_chunks.append(imp0 - imp1)

    codes = torch.cat(main_chunks, dim=0)
    fields = {"base": base.half().cpu(), "scales": scales.half().cpu(), "codes": codes.cpu()}
    if residual_mode == "r100":
        fields["residual"] = residual.half().cpu()
        fields["correction"] = torch.cat(res_chunks, dim=0).cpu()
    elif residual_mode == "r25_act":
        fields["residual"] = residual.half().cpu()
        res_all = torch.cat(res_chunks, dim=0)
        sel = top_mask(torch.cat(gain_chunks, dim=0))
        flat = sel.flatten()
        mask = torch.nn.functional.pad(flat, (0, (-flat.numel()) % 8)).reshape(-1, 8)
        bits = (mask.to(torch.int32) << torch.arange(8, device=flat.device)).sum(1).byte()
        fields["mask"] = bits.cpu()
        fields["correction"] = res_all.flatten()[flat].byte().cpu()

    return {k: v.contiguous() for k, v in fields.items()}


def decode_matrix(fields: dict[str, torch.Tensor], width: int, group_size: int, dim: int, residual_mode: str, device="cuda") -> torch.Tensor:
    codes = fields["codes"].to(device)
    rows = codes.shape[0]
    base = fields["base"].to(device).float()
    scales = fields["scales"].to(device).float()
    vecs_per_sub = group_size // dim
    scales_exp = scales.repeat_interleave(vecs_per_sub, dim=1)[..., None]

    v = base[codes.long()]
    if residual_mode == "r100":
        residual = fields["residual"].to(device).float()
        corr = fields["correction"].to(device).long()
        v = v + residual[corr]
    elif residual_mode == "r25_act":
        residual = fields["residual"].to(device).float()
        mask_bytes = fields["mask"].to(device)
        corr = fields["correction"].to(device).long()
        count = codes.numel()
        marked = ((mask_bytes[:, None].long() >> torch.arange(8, device=device)) & 1).flatten().bool()[:count]
        v_flat = v.reshape(-1, dim)
        v_flat[marked] = v_flat[marked] + residual[corr]
        v = v_flat.reshape(rows, width // dim, dim)

    return (v * scales_exp).reshape(rows, width).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "source", "w6", "captures", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False

    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import comfy.ops
    from comfy.ldm.moss_image.model import Block
    from comfy_kitchen.tensor import w4a8_int8

    cal = [load_file(str(p)) for p in sorted(args.captures.glob("cal_*.safetensors"))]
    if len(cal) != 2:
        raise ValueError("Expected two calibration scenes")

    blocks = (0, 20, 41)
    families = ("qkv", "o", "ffn_down", "ffn_up_gate")
    configs = [
        ("v1_g64_4d_r25", 64, 4, "r25_act"),
        ("v1_g64_4d_r100", 64, 4, "r100"),
        ("v1_g64_2d_r100", 64, 2, "none"),
        ("v1_g16_2d_r100", 16, 2, "none"),
    ]

    size_summary = {}
    decoded_cache = {}
    with safe_open(args.source, framework="pt") as src:
        for cfg_name, gsize, dim, rmode in configs:
            rng = torch.Generator().manual_seed(1121)
            tensors = {}
            decoded_cache[cfg_name] = {}
            widths = {}
            for idx in blocks:
                for fam in families:
                    mat_name = f"layers.{idx}.proj.{fam}"
                    w = src.get_tensor(mat_name + ".weight")
                    widths[mat_name] = int(w.shape[1])
                    cal_x = torch.cat([d[mat_name] for d in cal]) if rmode == "r25_act" else None
                    fields = encode_matrix(w, cal_x, gsize, dim, rmode, rng)
                    for fk, fv in fields.items():
                        tensors[f"{mat_name}.{fk}"] = fv
                    recon = decode_matrix(fields, widths[mat_name], gsize, dim, rmode, device="cuda")
                    if not torch.isfinite(recon).all():
                        raise ValueError(f"Non-finite decode: {cfg_name} {mat_name}")
                    decoded_cache[cfg_name][mat_name] = recon.cpu()
            meta = {
                "version": VERSION_V1,
                "config": json.dumps({"group_size": gsize, "dim": dim, "residual_mode": rmode}),
                "columns": json.dumps(widths, sort_keys=True),
            }
            arc_path = args.out / f"{cfg_name}.safetensors"
            tmp_path = arc_path.with_suffix(".tmp")
            save_file(tensors, str(tmp_path), metadata=meta)
            tmp_path.rename(arc_path)
            tbytes = sum(t.numel() * t.element_size() for t in tensors.values())
            fbytes = arc_path.stat().st_size
            size_summary[cfg_name] = {
                "file_bytes": fbytes,
                "tensor_bytes": tbytes,
                "header_bytes": fbytes - tbytes,
                "extrapolated_42_blocks_bytes": tbytes * 14 + 651935808,
            }
            print("encoded", cfg_name, fbytes, flush=True)

    (args.out / "v1_size_summary.json").write_text(json.dumps(size_summary, indent=2), encoding="utf-8")

    native_calls = 0
    original = w4a8_int8.w4a8_int8_linear

    def counted(*a, **kw):
        nonlocal native_calls
        y = original(*a, **kw)
        native_calls += 1
        return y

    w4a8_int8.w4a8_int8_linear = counted
    results = []
    variants = ["bf16", "w6"] + [c[0] for c in configs]
    for idx in blocks:
        prefix = f"layers.{idx}."
        for variant in variants:
            native = variant == "w6"
            ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16) if native else comfy.ops.disable_weight_init
            with safe_open(args.w6 if native else args.source, framework="pt") as f:
                state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
            if variant in decoded_cache:
                for fam in families:
                    mat_name = f"layers.{idx}.proj.{fam}"
                    state[f"proj.{fam}.weight"] = decoded_cache[variant][mat_name]
            block = Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, device="cuda", operations=ops)
            block.load_state_dict(state, strict=True)
            block.eval()
            cases = sorted(args.inputs.glob(f"*_{idx:02}_*.safetensors"))
            if len(cases) != 8:
                raise RuntimeError("Expected eight recorded calls per block")
            for case in cases:
                data = load_file(str(case), device="cuda")
                x = data["x"]
                count = native_calls
                y = block(x.clone(), tuple(data["mod"]), data["pe"], {"block_index": idx}, mask=data.get("mask"))
                ref = data["y"].float()
                diff = (y.float() - ref).norm()
                actual_native = native_calls - count
                if not torch.isfinite(diff) or actual_native != (4 if native else 0):
                    raise RuntimeError(f"Invalid result/dispatch: {case.stem} {variant}")
                if variant == "bf16" and not torch.equal(y, data["y"]):
                    raise RuntimeError(f"BF16 mismatch: {case.stem}")
                results.append(dict(
                    block=idx,
                    variant=variant,
                    case=case.stem,
                    output_error=float(diff / ref.norm()),
                    update_error=float(diff / (ref - x.float()).norm()),
                    native_calls=actual_native,
                ))
            print(idx, variant, flush=True)
            del block, state

    with (args.out / "v1_block_errors.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
