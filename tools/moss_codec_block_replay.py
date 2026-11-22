"""Replay complete stock MOSS blocks on frozen inputs and score the block output and residual update."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from moss_vector_codec import VERSION, read_matrix


HADES_FIELDS = ("base", "residual", "codes", "mask", "correction", "scales")
ATTN_FAMILIES = ("qkv", "o")
FFN_FAMILIES = ("ffn_down", "ffn_up_gate")
ALL_FAMILIES = ATTN_FAMILIES + FFN_FAMILIES


def serialize_mixed_blocks(encoded_path: Path, w6_path: Path, out_dir: Path, blocks=(0, 20, 41)):
    """Serialize attention (HaDeS v0) + FFN (W6A8) for selected blocks and return byte breakdown."""
    tensors = {}
    attn_bytes = 0
    ffn_bytes = 0
    with safe_open(encoded_path, framework="pt") as hf, safe_open(w6_path, framework="pt") as wf:
        hades_meta = hf.metadata() or {}
        if hades_meta.get("version") != VERSION:
            raise ValueError("Unsupported vector codec version")
        all_cols = json.loads(hades_meta["columns"])
        mixed_cols = {}
        for idx in blocks:
            prefix = f"layers.{idx}."
            for fam in ATTN_FAMILIES:
                name = prefix + "proj." + fam
                mixed_cols[name] = all_cols[name]
                for field in HADES_FIELDS:
                    k = f"{name}.{field}"
                    t = hf.get_tensor(k).cpu().contiguous()
                    tensors[k] = t
                    attn_bytes += t.numel() * t.element_size()
            for fam in FFN_FAMILIES:
                sub = f"{prefix}proj.{fam}."
                matched = [k for k in wf.keys() if k.startswith(sub)]
                if not matched:
                    raise KeyError(f"Missing W6A8 tensors for {sub}")
                for k in sorted(matched):
                    t = wf.get_tensor(k).cpu().contiguous()
                    tensors[k] = t
                    ffn_bytes += t.numel() * t.element_size()

    meta = {"version": VERSION, "columns": json.dumps(mixed_cols, sort_keys=True)}
    archive_path = out_dir / "mixed_blocks.safetensors"
    tmp_path = archive_path.with_suffix(".tmp")
    save_file(tensors, str(tmp_path), metadata=meta)
    for name in mixed_cols:
        data = read_matrix(tmp_path, name, "cpu")
        if not torch.isfinite(data).all():
            raise ValueError(f"Non-finite reconstructed matrix: {name}")
    if archive_path.exists():
        archive_path.unlink()
    tmp_path.rename(archive_path)

    total_bytes = archive_path.stat().st_size
    tensor_bytes = attn_bytes + ffn_bytes
    return {
        "archive": str(archive_path),
        "blocks": list(blocks),
        "total_bytes": total_bytes,
        "tensor_bytes": tensor_bytes,
        "header_and_metadata_bytes": total_bytes - tensor_bytes,
        "attn_hades_bytes": attn_bytes,
        "ffn_w6a8_bytes": ffn_bytes,
        "metadata": meta,
    }


def load_block_state(idx: int, variant: str, source_path: Path, w6_path: Path, encoded_dir: Path):
    """Load a block state_dict and return (state, ops, expected_native_calls)."""
    import comfy.ops

    prefix = f"layers.{idx}."
    if variant == "w6":
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
        with safe_open(w6_path, framework="pt") as f:
            state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        return state, ops, 4

    if variant == "mixed_attn_hades_ffn_w6":
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
        with safe_open(source_path, framework="pt") as sf:
            state = {k[len(prefix):]: sf.get_tensor(k) for k in sf.keys() if k.startswith(prefix)}
        for fam in ATTN_FAMILIES:
            name = "proj." + fam
            state[name + ".weight"] = read_matrix(encoded_dir / "activations.safetensors", prefix + name)
        with safe_open(w6_path, framework="pt") as wf:
            for fam in FFN_FAMILIES:
                sub = "proj." + fam + "."
                state.pop(sub + "weight", None)
                full_sub = prefix + sub
                for k in wf.keys():
                    if k.startswith(full_sub):
                        state[k[len(prefix):]] = wf.get_tensor(k)
        return state, ops, 2

    ops = comfy.ops.disable_weight_init
    with safe_open(source_path, framework="pt") as f:
        state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    if variant in ("weights", "activations") or variant in ALL_FAMILIES:
        archive = "activations" if variant in ALL_FAMILIES else variant
        fams = (variant,) if variant in ALL_FAMILIES else ALL_FAMILIES
        for fam in fams:
            name = "proj." + fam
            state[name + ".weight"] = read_matrix(encoded_dir / (archive + ".safetensors"), prefix + name)
    return state, ops, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "source", "w6", "encoded", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--isolate-final", action="store_true", help="Quantize one projection at a time in block 41")
    mode.add_argument("--mixed", action="store_true", help="Test HaDeS v0 attention (activations) + native W6A8 FFN")
    mode.add_argument("--hms", type=Path, help="C1/C2 checkpoint or verified block shards; --encoded is the ffn30 control")
    parser.add_argument("--serialize-mixed-dir", type=Path, default=None, help="Optional directory to write mixed_blocks.safetensors and size_breakdown.json")
    args = parser.parse_args()
    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    from comfy.ldm.moss_image.model import Block
    from comfy_kitchen.tensor import w4a8_int8

    if args.serialize_mixed_dir is not None:
        args.serialize_mixed_dir.mkdir(parents=True, exist_ok=True)
        breakdown = serialize_mixed_blocks(args.encoded / "activations.safetensors", args.w6, args.serialize_mixed_dir)
        (args.serialize_mixed_dir / "size_breakdown.json").write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
        print("serialized_mixed_bytes", breakdown["total_bytes"], flush=True)

    native_calls = 0
    original = w4a8_int8.w4a8_int8_linear

    def counted(*a, **kw):
        nonlocal native_calls
        y = original(*a, **kw)
        native_calls += 1
        return y

    w4a8_int8.w4a8_int8_linear = counted
    results = []
    if args.hms:
        blocks = (0, 20, 41)
        variants = ("bf16", "w6", "ffn30", "calibrated")
    elif args.isolate_final:
        blocks = (41,)
        variants = ALL_FAMILIES
    elif args.mixed:
        blocks = (0, 20, 41)
        variants = ("bf16", "w6", "mixed_attn_hades_ffn_w6")
    else:
        blocks = (0, 20, 41)
        variants = ("bf16", "w6", "weights", "activations")

    for idx in blocks:
        for variant in variants:
            if variant in ("ffn30", "calibrated"):
                checkpoint = args.hms if variant == "calibrated" and args.hms.is_file() else args.encoded
                state, ops, expected_native = load_block_state(idx, "w6", args.source, checkpoint, args.encoded)
                if variant == "calibrated" and args.hms.is_dir():
                    prefix = f"layers.{idx}."
                    updates = load_file(str(args.hms / f"block_{idx:02}.safetensors"))
                    state.update({k.removeprefix(prefix): v for k, v in updates.items()})
            else:
                state, ops, expected_native = load_block_state(idx, variant, args.source, args.w6, args.encoded)
            block = Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, device="cuda", operations=ops)
            block.load_state_dict(state, strict=True)
            block.eval()
            cases = sorted(args.inputs.glob(f"*_{idx:02}_*.safetensors"))
            expected_cases = 12 if args.hms else 8
            if len(cases) != expected_cases:
                raise RuntimeError(f"Expected {expected_cases} recorded calls per block")
            for case in cases:
                data = load_file(str(case), device="cuda")
                x = data["x"]
                count = native_calls
                y = block(x.clone(), tuple(data["mod"]), data["pe"], {"block_index": idx}, mask=data.get("mask"))
                ref = data["y"].float()
                difference = (y.float() - ref).norm()
                actual_native = native_calls - count
                if not torch.isfinite(difference) or actual_native != expected_native:
                    raise RuntimeError(f"Invalid result/dispatch: {case.stem}, {variant}, {actual_native} != {expected_native}")
                if variant == "bf16" and not torch.equal(y, data["y"]):
                    raise RuntimeError(f"BF16 replay mismatch {case.stem}: relative {difference/ref.norm()}")
                results.append(dict(
                    block=idx,
                    variant=variant,
                    case=case.stem,
                    output_error=float(difference / ref.norm()),
                    update_error=float(difference / (ref - x.float()).norm()),
                    native_calls=actual_native,
                ))
            print(idx, variant, flush=True)
            del block, state
    with args.out.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
