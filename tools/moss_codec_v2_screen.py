"""Screen Hadamard-rotated (FWHT-64) vector and 4/5/6-bit scalar codec designs on blocks 0, 20, 41."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

VERSION_V2 = "hades_fwht_v2_screen"


def hadamard_matrix(n: int, device="cuda") -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (n ** 0.5)


def nearest(points: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    norms = table.square().sum(1)
    return torch.cat([(norms - 2 * part @ table.T).argmin(1) for part in points.split(16384)])


def fit_codebook(points: torch.Tensor, dim: int = 4, iters: int = 10) -> torch.Tensor:
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


def pack_nbits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned uint8 values in [0, 2^bits - 1] along last dim (width divisible by 8)."""
    if bits == 4:
        pairs = q.reshape(*q.shape[:-1], -1, 2)
        return (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()
    if bits == 5:
        octs = q.reshape(*q.shape[:-1], -1, 8).to(torch.int64)
        packed_val = torch.zeros(octs.shape[:-1], dtype=torch.int64, device=q.device)
        for i in range(8):
            packed_val |= octs[..., i] << (5 * i)
        out = torch.empty(*octs.shape[:-1], 5, dtype=torch.uint8, device=q.device)
        for b in range(5):
            out[..., b] = ((packed_val >> (8 * b)) & 0xFF).to(torch.uint8)
        return out.reshape(*q.shape[:-1], -1).contiguous()
    if bits == 6:
        quads = q.reshape(*q.shape[:-1], -1, 4).to(torch.int32)
        val = quads[..., 0] | (quads[..., 1] << 6) | (quads[..., 2] << 12) | (quads[..., 3] << 18)
        out = torch.empty(*quads.shape[:-1], 3, dtype=torch.uint8, device=q.device)
        out[..., 0] = (val & 0xFF).to(torch.uint8)
        out[..., 1] = ((val >> 8) & 0xFF).to(torch.uint8)
        out[..., 2] = ((val >> 16) & 0xFF).to(torch.uint8)
        return out.reshape(*q.shape[:-1], -1).contiguous()
    raise ValueError(f"Unsupported bits: {bits}")


def unpack_nbits(packed: torch.Tensor, width: int, bits: int) -> torch.Tensor:
    if bits == 4:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], width)
    if bits == 5:
        b = packed.reshape(*packed.shape[:-1], -1, 5).to(torch.int64)
        val = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24) | (b[..., 4] << 32)
        out = torch.empty(*b.shape[:-1], 8, dtype=torch.uint8, device=packed.device)
        for i in range(8):
            out[..., i] = ((val >> (5 * i)) & 0x1F).to(torch.uint8)
        return out.reshape(*packed.shape[:-1], width)
    if bits == 6:
        b = packed.reshape(*packed.shape[:-1], -1, 3).to(torch.int32)
        val = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
        out = torch.empty(*b.shape[:-1], 4, dtype=torch.uint8, device=packed.device)
        for i in range(4):
            out[..., i] = ((val >> (6 * i)) & 0x3F).to(torch.uint8)
        return out.reshape(*packed.shape[:-1], width)
    raise ValueError(f"Unsupported bits: {bits}")


def encode_fwht_rvq(w: torch.Tensor, h64: torch.Tensor, rng: torch.Generator):
    rows, width = w.shape
    w_rot = (w.cuda().float().reshape(rows, width // 64, 64) @ h64).reshape(rows, width)

    train_idx = torch.randperm(rows, generator=rng)[: min(256, rows)]
    wt = w_rot[train_idx].reshape(-1, width // 64, 64)
    sc_t = wt.square().mean(-1).sqrt().half().float().clamp_min(1e-7)
    pts = (wt / sc_t[..., None]).reshape(-1, 4)
    pts = pts[torch.randperm(len(pts), generator=rng)[: min(16384, len(pts))].cuda()]
    base = fit_codebook(pts, 4)
    residual = fit_codebook(pts - base[nearest(pts, base)], 4)

    wg = w_rot.reshape(rows, width // 64, 64)
    scales = wg.square().mean(-1).sqrt().half().float().clamp_min(1e-7)
    normed = (wg / scales[..., None]).reshape(rows, width // 4, 4)
    c_chunks, r_chunks = [], []
    for s in range(0, rows, 256):
        tc = normed[s : s + 256]
        ci = nearest(tc.reshape(-1, 4), base).reshape(tc.shape[:2])
        ri = nearest((tc - base[ci]).reshape(-1, 4), residual).reshape_as(ci)
        c_chunks.append(ci.byte())
        r_chunks.append(ri.byte())
    return {
        "base": base.half().cpu().contiguous(),
        "residual": residual.half().cpu().contiguous(),
        "scales": scales.half().cpu().contiguous(),
        "codes": torch.cat(c_chunks, 0).cpu().contiguous(),
        "correction": torch.cat(r_chunks, 0).cpu().contiguous(),
    }


def decode_fwht_rvq(fields: dict[str, torch.Tensor], width: int, h64: torch.Tensor) -> torch.Tensor:
    codes = fields["codes"].cuda().long()
    corr = fields["correction"].cuda().long()
    base = fields["base"].cuda().float()
    residual = fields["residual"].cuda().float()
    scales = fields["scales"].cuda().float().repeat_interleave(16, dim=1)[..., None]
    v = (base[codes] + residual[corr]) * scales
    v_rot = v.reshape(codes.shape[0], width // 64, 64)
    return (v_rot @ h64).reshape(codes.shape[0], width).bfloat16()


def encode_fwht_scalar(w: torch.Tensor, h64: torch.Tensor, bits: int):
    rows, width = w.shape
    w_rot = (w.cuda().float().reshape(rows, width // 64, 64) @ h64)
    qmax = (1 << (bits - 1)) - 1
    scales = (w_rot.abs().amax(dim=-1) / qmax).half().float().clamp_min(1e-7)
    q_signed = torch.round(w_rot / scales[..., None]).clamp(-qmax, qmax).to(torch.int32)
    q_unsigned = (q_signed + qmax).to(torch.uint8).reshape(rows, width)
    packed = pack_nbits(q_unsigned, bits)
    return {
        "packed": packed.cpu().contiguous(),
        "scales": scales.half().cpu().contiguous(),
    }


def decode_fwht_scalar(fields: dict[str, torch.Tensor], width: int, bits: int, h64: torch.Tensor) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    q_unsigned = unpack_nbits(fields["packed"].cuda(), width, bits).to(torch.int32)
    q_signed = (q_unsigned - qmax).float().reshape(q_unsigned.shape[0], width // 64, 64)
    scales = fields["scales"].cuda().float()[..., None]
    return ((q_signed * scales) @ h64).reshape(q_unsigned.shape[0], width).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "source", "w6", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replay-from", type=Path, default=None, help="Load already-encoded v2 safetensors from this directory instead of re-encoding")
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

    h64 = hadamard_matrix(64, device="cuda")
    blocks = (0, 20, 41)
    families = ("qkv", "o", "ffn_down", "ffn_up_gate")
    configs = [
        ("v2_fwht64_4d_s2", "rvq", 0),
        ("v2_fwht64_q4", "scalar", 4),
        ("v2_fwht64_q5", "scalar", 5),
        ("v2_fwht64_q6", "scalar", 6),
    ]

    size_summary = {}
    decoded_cache = {}
    if args.replay_from is not None:
        configs = [("v2_fwht64_q5", "scalar", 5), ("v2_fwht64_q6", "scalar", 6)]
        for cfg_name, mode, bits in configs:
            arc_path = args.replay_from / f"{cfg_name}.safetensors"
            decoded_cache[cfg_name] = {}
            with safe_open(arc_path, framework="pt") as af:
                widths = json.loads((af.metadata() or {})["columns"])
                for mat_name, width in widths.items():
                    sub = mat_name + "."
                    fields = {k[len(sub):]: af.get_tensor(k) for k in af.keys() if k.startswith(sub)}
                    recon = decode_fwht_scalar(fields, width, bits, h64)
                    if not torch.isfinite(recon).all():
                        raise ValueError(f"Non-finite decode: {cfg_name} {mat_name}")
                    decoded_cache[cfg_name][mat_name] = recon.cpu()
            print("loaded", cfg_name, flush=True)
    else:
        with safe_open(args.source, framework="pt") as src:
            for cfg_name, mode, bits in configs:
                rng = torch.Generator().manual_seed(1121)
                tensors = {}
                decoded_cache[cfg_name] = {}
                widths = {}
                for idx in blocks:
                    for fam in families:
                        mat_name = f"layers.{idx}.proj.{fam}"
                        w = src.get_tensor(mat_name + ".weight")
                        width = int(w.shape[1])
                        widths[mat_name] = width
                        if mode == "rvq":
                            fields = encode_fwht_rvq(w, h64, rng)
                            recon = decode_fwht_rvq(fields, width, h64)
                        else:
                            fields = encode_fwht_scalar(w, h64, bits)
                            recon = decode_fwht_scalar(fields, width, bits, h64)
                        if not torch.isfinite(recon).all():
                            raise ValueError(f"Non-finite decode: {cfg_name} {mat_name}")
                        for fk, fv in fields.items():
                            tensors[f"{mat_name}.{fk}"] = fv
                        decoded_cache[cfg_name][mat_name] = recon.cpu()
                meta = {"version": VERSION_V2, "mode": mode, "bits": str(bits), "columns": json.dumps(widths, sort_keys=True)}
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

        (args.out / "v2_size_summary.json").write_text(json.dumps(size_summary, indent=2), encoding="utf-8")

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
            if not cases or len(cases) % 4 != 0:
                raise RuntimeError(f"Expected multiple of four recorded calls per block, got {len(cases)}")
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

    with (args.out / "v2_block_errors.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
