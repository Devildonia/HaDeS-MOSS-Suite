"""Measure stock UNETLoader return time in a fresh process; inference/device upload is excluded."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import torch
    import folder_paths

    path = args.model.resolve(strict=True)
    folder_paths.add_model_folder_path("diffusion_models", str(path.parent), is_default=True)
    if Path(folder_paths.get_full_path_or_raise("diffusion_models", path.name)).resolve() != path:
        raise RuntimeError("Unexpected model path")
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()
        model, = nodes.UNETLoader().load_unet(path.name, "default")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    with args.out.open("x") as out:
        json.dump(dict(model=str(path), loader_seconds=elapsed), out, indent=2)


if __name__ == "__main__":
    main()
