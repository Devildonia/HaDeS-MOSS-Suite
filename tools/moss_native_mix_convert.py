"""Build a mixed-precision native W4A8/W6A8 MOSS checkpoint with hades/recipe metadata."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ATTN_FAMILIES = ("qkv", "o")
FFN_FAMILIES = ("ffn_up_gate", "ffn_down")
RECIPE_VERSION = "hades_native_mix_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_block_set(spec: str) -> set[int]:
    if not spec.strip():
        return set()
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="BF16 source checkpoint (standalone quantization)")
    parser.add_argument("--w4", type=Path, help="Pre-quantized W4A8 checkpoint (fast merge with --w6)")
    parser.add_argument("--w6", type=Path, help="Pre-quantized W6A8 checkpoint (fast merge with --w4)")
    parser.add_argument("--protected-ffn-blocks", type=str, default="38-41", help="Blocks whose FFN stays in W6A8")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists() or args.audit_out.exists():
        raise FileExistsError("Refusing to overwrite existing output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)

    protected_ffn = sorted(parse_block_set(args.protected_ffn_blocks))
    protected_set = set(protected_ffn)
    t0 = time.perf_counter()

    def target_format(block_idx: int, fam: str) -> str:
        if fam in ATTN_FAMILIES or block_idx in protected_set:
            return "w6a8_int8"
        return "asym_w4a8_int8"

    state = {}
    tensor_formats = {}
    counts = {"w6a8_int8": 0, "asym_w4a8_int8": 0, "literal": 0}
    byte_breakdown = {"w6a8_int8": 0, "asym_w4a8_int8": 0, "literal": 0}

    if args.w4 and args.w6:
        with safe_open(args.w4, framework="pt") as f4, safe_open(args.w6, framework="pt") as f6:
            orig_meta = dict(f6.metadata() or {})
            all_keys = set(f6.keys())
            assigned = set()
            for idx in range(42):
                for fam in ATTN_FAMILIES + FFN_FAMILIES:
                    fmt = target_format(idx, fam)
                    src_f = f6 if fmt == "w6a8_int8" else f4
                    prefix = f"layers.{idx}.proj.{fam}."
                    matched = [k for k in src_f.keys() if k.startswith(prefix)]
                    if not matched:
                        raise KeyError(f"Missing {prefix} in {fmt}")
                    for k in matched:
                        t = src_f.get_tensor(k).cpu().contiguous()
                        state[k] = t
                        byte_breakdown[fmt] += t.numel() * t.element_size()
                        assigned.add(k)
                    tensor_formats[f"layers.{idx}.proj.{fam}"] = fmt
                    counts[fmt] += 1
            for k in sorted(all_keys - assigned):
                if ".proj." in k and k.startswith("layers."):
                    continue
                t = f6.get_tensor(k).cpu().contiguous()
                state[k] = t
                byte_breakdown["literal"] += t.numel() * t.element_size()
                counts["literal"] += 1
    elif args.source:
        sys.path.insert(0, str(args.core.resolve()))
        from comfy.quant_ops import QuantizedTensor

        with safe_open(args.source, framework="pt") as src:
            orig_meta = dict(src.metadata() or {})
            for key in src.keys():
                val = src.get_tensor(key)
                if key.startswith("layers.") and ".proj." in key and key.endswith(".weight"):
                    parts = key.split(".")
                    idx = int(parts[1])
                    fam = parts[3]
                    fmt = target_format(idx, fam)
                    opts = {"bits": 6, "codebook": False} if fmt == "w6a8_int8" else {}
                    packed = QuantizedTensor.from_float(val.cuda(), "AsymW4A8Int8Layout", **opts)
                    for sk, sv in packed.state_dict(key).items():
                        sv_cpu = sv.cpu().contiguous()
                        state[sk] = sv_cpu
                        byte_breakdown[fmt] += sv_cpu.numel() * sv_cpu.element_size()
                    marker = json.dumps({"format": fmt}, separators=(",", ":")).encode()
                    mk = key.removesuffix("weight") + "comfy_quant"
                    mt = torch.tensor(list(marker), dtype=torch.uint8)
                    state[mk] = mt
                    byte_breakdown[fmt] += mt.numel() * mt.element_size()
                    tensor_formats[f"layers.{idx}.proj.{fam}"] = fmt
                    counts[fmt] += 1
                else:
                    t = val.cpu().contiguous()
                    state[key] = t
                    byte_breakdown["literal"] += t.numel() * t.element_size()
                    counts["literal"] += 1
    else:
        raise ValueError("Provide either (--w4 and --w6) or --source")

    recipe = {
        "version": RECIPE_VERSION,
        "method": "native_mixed_w4a8_w6a8",
        "attention_format": "w6a8_int8",
        "default_ffn_format": "asym_w4a8_int8",
        "protected_ffn_blocks": protected_ffn,
        "rationale": "Native W4A8/W6A8 sensitivity screen showed attention (qkv, o) is more sensitive than FFN in AsymW4A8Int8Layout; late FFN blocks 38-41 protected in W6A8 to match ~3.80 GB target.",
        "counts": counts,
    }
    meta = dict(orig_meta)
    meta["hades/recipe"] = json.dumps(recipe, separators=(",", ":"))

    with TemporaryDirectory(dir=args.out.parent) as tmpdir:
        tmp_path = Path(tmpdir) / "model.safetensors"
        save_file(state, str(tmp_path), metadata=meta)
        with safe_open(tmp_path, framework="pt") as saved:
            if set(saved.keys()) != set(state):
                raise RuntimeError("Serialized tensor inventory mismatch")
        tmp_path.rename(args.out)

    total_bytes = args.out.stat().st_size
    tensor_bytes = sum(byte_breakdown.values())
    audit = {
        "out": str(args.out),
        "out_sha256": sha256_file(args.out),
        "total_bytes": total_bytes,
        "tensor_bytes": tensor_bytes,
        "header_and_metadata_bytes": total_bytes - tensor_bytes,
        "byte_breakdown": byte_breakdown,
        "counts": counts,
        "recipe": recipe,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
