"""Screen native W4A8 vs W6A8 sensitivity and mixed-precision dispatch on frozen MOSS blocks."""
import argparse
import csv
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

FAMILIES = ("qkv", "o", "ffn_up_gate", "ffn_down")


def load_mixed_block(idx: int, w4_fams: set[str], source_path: Path, w4_path: Path, w6_path: Path):
    prefix = f"layers.{idx}."
    with safe_open(source_path, framework="pt") as sf:
        state = {
            k[len(prefix):]: sf.get_tensor(k)
            for k in sf.keys()
            if k.startswith(prefix) and not k.startswith(prefix + "proj.")
        }
    with safe_open(w4_path, framework="pt") as f4, safe_open(w6_path, framework="pt") as f6:
        for fam in FAMILIES:
            sub = prefix + "proj." + fam + "."
            src_f = f4 if fam in w4_fams else f6
            for k in src_f.keys():
                if k.startswith(sub):
                    state[k[len(prefix):]] = src_f.get_tensor(k)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "source", "w4", "w6", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import comfy.ops
    from comfy.ldm.moss_image.model import Block
    from comfy_kitchen.tensor import w4a8_int8

    native_calls = 0
    original = w4a8_int8.w4a8_int8_linear

    def counted(*a, **kw):
        nonlocal native_calls
        y = original(*a, **kw)
        native_calls += 1
        return y

    w4a8_int8.w4a8_int8_linear = counted

    variants = [
        ("bf16", set(), False),
        ("w6", set(), True),
        ("w4", set(FAMILIES), True),
        ("w4_attn_w6_ffn", {"qkv", "o"}, True),
        ("w6_attn_w4_ffn", {"ffn_up_gate", "ffn_down"}, True),
    ]
    isolation_41 = [
        ("b41_only_qkv_w4", {"qkv"}, True),
        ("b41_only_o_w4", {"o"}, True),
        ("b41_only_up_gate_w4", {"ffn_up_gate"}, True),
        ("b41_only_down_w4", {"ffn_down"}, True),
    ]

    results = []
    for idx in (0, 20, 41):
        cases = sorted(args.inputs.glob(f"*_{idx:02}_*.safetensors"))
        if not cases:
            raise RuntimeError(f"No inputs found for block {idx}")
        block_variants = variants + (isolation_41 if idx == 41 else [])
        for vname, w4_fams, is_native in block_variants:
            ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16) if is_native else comfy.ops.disable_weight_init
            if not is_native:
                prefix = f"layers.{idx}."
                with safe_open(args.source, framework="pt") as sf:
                    state = {k[len(prefix):]: sf.get_tensor(k) for k in sf.keys() if k.startswith(prefix)}
            else:
                state = load_mixed_block(idx, w4_fams, args.source, args.w4, args.w6)

            block = Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, device="cuda", operations=ops)
            block.load_state_dict(state, strict=True)
            block.eval()

            for case in cases:
                data = load_file(str(case), device="cuda")
                x = data["x"]
                c0 = native_calls
                y = block(x.clone(), tuple(data["mod"]), data["pe"], {"block_index": idx}, mask=data.get("mask"))
                ref = data["y"].float()
                diff = (y.float() - ref).norm()
                actual_native = native_calls - c0
                expected_native = 4 if is_native else 0
                if not torch.isfinite(diff) or actual_native != expected_native:
                    raise RuntimeError(f"Invalid result/dispatch: {case.stem} {vname} {actual_native} != {expected_native}")
                if vname == "bf16" and not torch.equal(y, data["y"]):
                    raise RuntimeError(f"BF16 mismatch: {case.stem}")
                results.append(dict(
                    block=idx,
                    variant=vname,
                    case=case.stem,
                    output_error=float(diff / ref.norm()),
                    update_error=float(diff / (ref - x.float()).norm()),
                    native_calls=actual_native,
                ))
            print(idx, vname, flush=True)
            del block, state

    with (args.out / "native_mix_errors.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
