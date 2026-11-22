"""Export the four MOSS Base projection families with a stock Comfy quantization layout."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    layouts = {
        "nvfp4": "TensorCoreNVFP4Layout",
        "convrot_w4a4": "TensorCoreConvRotW4A4Layout",
        "asym_w4a8_int8": "AsymW4A8Int8Layout",
        "w6a8_int8": "AsymW4A8Int8Layout",
    }
    parser.add_argument("--format", choices=layouts, default="nvfp4")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core.resolve()))
    from comfy.quant_ops import QuantizedTensor

    if args.output.exists():
        raise FileExistsError(args.output)
    expected = {f"layers.{i}.proj.{name}.weight" for i in range(42)
                for name in ("qkv", "o", "ffn_up_gate", "ffn_down")}
    state = {}
    options = {"bits": 6, "codebook": False} if args.format == "w6a8_int8" else {}
    with safe_open(args.source, framework="pt") as original:
        if not expected.issubset(original.keys()):
            raise ValueError("Missing MOSS Base projections")
        metadata = original.metadata()
        for key in original.keys():
            value = original.get_tensor(key)
            if key in expected:
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"Expected a BF16 matrix at {key}")
                packed = QuantizedTensor.from_float(value.cuda(), layouts[args.format], **options)
                state.update({k: v.cpu() for k, v in packed.state_dict(key).items()})
                marker = json.dumps({"format": args.format}, separators=(",", ":")).encode()
                state[key.removesuffix("weight") + "comfy_quant"] = torch.tensor(list(marker), dtype=torch.uint8)
            else:
                state[key] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        temp = Path(directory) / "model.safetensors"
        save_file(state, temp, metadata=metadata)
        with safe_open(temp, framework="pt") as saved:
            if set(saved.keys()) != set(state):
                raise ValueError("Serialized tensor inventory differs")
        temp.rename(args.output)
    print(f"{len(expected)} matrices, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
