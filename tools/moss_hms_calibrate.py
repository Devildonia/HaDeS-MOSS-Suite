"""Calibrate ffn30 projection values and verify every packed chunk against eager dequant."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from comfy_kitchen.backends.eager import w4a8_int8 as core
from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight as rotate
from comfy_kitchen.tensor.w4a8_packed import _pack_codes


def fit_table(weight, diagonal):
    """Seeded sampling over individual weights, normalized with their 16-column groups."""
    rows, cols = weight.shape
    group = weight.view(rows, -1, 16)
    rng = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(rows * cols, (300000,), generator=rng, device="cuda")
    x = weight.flatten()[idx] / group.abs().amax(-1).flatten()[idx // 16].clamp_min(1e-8)
    d = diagonal[idx % cols]
    table = torch.quantile(x, torch.linspace(0, 1, 16, device="cuda"))
    for _ in range(25):
        code = torch.bucketize(x, (table[:-1] + table[1:]) / 2)
        # Deterministic reduction order for this fixed sample.
        for j in range(16):
            mass = torch.where(code == j, d, 0.)
            total = mass.sum()
            table[j] = torch.where(total > 0, (x * mass).sum() / total, table[j])
    return table


def calibrate(weight, baseline_scale, channel, diagonal, bits):
    rows, cols = weight.shape
    rotated = rotate(weight.float().cuda(), 256)
    d = diagonal.cuda().view(1, -1, 16)
    table = fit_table(rotated, d.flatten()) if bits == 4 else torch.arange(-31, 32, device="cuda").float()
    packed_parts, scale_parts = [], []
    for start in range(0, rows, 128):
        group = rotated[start:start+128].view(-1, cols//16, 16)
        ch = channel[start:start+128].cuda()
        base = baseline_scale[start:start+128].cuda()
        raw = base.view(torch.uint8).int()
        target = group / ch.view(-1, 1, 1)
        maximum = target.abs().amax(-1) / (31 if bits == 6 else 1)
        candidates = [base, (raw-1).clamp(1,126).byte().view(base.dtype),
                      (raw+1).clamp(1,126).byte().view(base.dtype)]
        candidates.extend((maximum * factor).clamp(0,448).to(base.dtype)
                          for factor in (0.8, 0.85, 0.9, 0.95, 1.0))
        best_err = torch.full_like(maximum, float("inf"))
        best_scale = torch.zeros_like(maximum)
        best_idx = torch.zeros_like(group, dtype=torch.int32)
        for scale in candidates:
            levels = core._grid_levels(table, scale)
            idx = core._assign_grid(group, levels, ch, target=target)
            err = ((levels.gather(2, idx.long()) - target).square() * d).sum(-1)
            better = err < best_err
            best_idx = torch.where(better.unsqueeze(-1), idx, best_idx)
            best_scale = torch.where(better, scale.float(), best_scale)
            best_err = torch.minimum(best_err, err)
        scale = best_scale.to(base.dtype)
        packed = _pack_codes((best_idx + int(bits == 6)).view(-1, cols), bits)
        lut = table if bits == 4 else None
        expected = core._grid_levels(table, scale).gather(2, best_idx.long()) * ch.view(-1, 1, 1)
        expected = rotate(expected.reshape(-1, cols).bfloat16(), 256)
        decoded = core.dequantize_w4a8_int8_weight(packed, scale, ch, lut)
        if not torch.equal(decoded, expected):
            raise ValueError("Core dequant differs from independent reconstruction")
        packed_parts.append(packed.cpu())
        scale_parts.append(scale.cpu())
    out = {"weight": torch.cat(packed_parts), "weight_s_rel": torch.cat(scale_parts)}
    if bits == 4:
        out["weight_codebook"] = table.cpu()
    return out


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "ffn30", "captures", "out"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=list(range(42)))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.captures.glob("cal_*.safetensors"))
    if len(files) != 4:
        raise ValueError("Expected four completed calibration scenes")
    samples = [(load_file(str(f)), json.loads(f.with_suffix(".json").read_text())) for f in files]
    with safe_open(args.source, framework="pt") as src, safe_open(args.ffn30, framework="pt") as ref:
        for block in args.blocks:
            path = args.out / f"block_{block:02}.safetensors"
            if path.exists():
                raise FileExistsError(path)
            state = {}
            for fam in ("qkv", "o", "ffn_up_gate", "ffn_down"):
                name = f"layers.{block}.proj.{fam}"
                rows = [m["projections"][name]["rows"] for _,m in samples]
                diagonal = sum(s[name]*n for (s,_),n in zip(samples,rows)) / sum(rows)
                prefix = name+"."
                bits = 4 if prefix+"weight_codebook" in ref.keys() else 6
                result = calibrate(src.get_tensor(prefix+"weight"), ref.get_tensor(prefix+"weight_s_rel"),
                                   ref.get_tensor(prefix+"weight_s_channel"), diagonal, bits)
                for suffix, value in result.items():
                    key = prefix+suffix
                    old = ref.get_slice(key)
                    if list(value.shape) != old.get_shape() or value.dtype != ref.get_tensor(key).dtype:
                        raise ValueError(key)
                    state[key] = value
            tmp = path.with_suffix(".partial")
            save_file(state, str(tmp))
            tmp.rename(path)
            print("verified", path, flush=True)


if __name__ == "__main__":
    main()
