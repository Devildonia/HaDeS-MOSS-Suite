"""Render a list of prompts across multiple seeds with a local DiT checkpoint and frozen graph."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("core", "model", "graph", "prompts", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--seeds", type=str, default="1001,1002,1003")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    import folder_paths
    from comfy_kitchen.tensor import w4a8_int8
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    prompts = [p.strip() for p in args.prompts.read_text(encoding="utf-8-sig").splitlines() if p.strip()]
    inputs = {key: node["inputs"] for key, node in graph.items()}
    folder_paths.add_model_folder_path("diffusion_models", str(args.model.resolve().parent), is_default=True)
    inputs["unet"]["unet_name"] = args.model.name
    if Path(folder_paths.get_full_path_or_raise("diffusion_models", args.model.name)).resolve() != args.model.resolve():
        raise RuntimeError("Model folder resolved to a different checkpoint")

    w4a8_calls = 0
    orig_w4a8 = w4a8_int8.w4a8_int8_linear

    def counted_w4a8(*a, **kw):
        nonlocal w4a8_calls
        out = orig_w4a8(*a, **kw)
        w4a8_calls += 1
        return out

    w4a8_int8.w4a8_int8_linear = counted_w4a8

    with torch.inference_mode():
        model, = nodes.UNETLoader().load_unet(**inputs["unet"])
        clip, = nodes.CLIPLoader().load_clip(**inputs["clip"])
        vae, = nodes.VAELoader().load_vae(**inputs["vae"])
        negative, = nodes.CLIPTextEncode().encode(clip, inputs["neg"]["text"])
        patched = SkipLayerGuidanceDiTSimple.execute(**dict(inputs["slg"], model=model)).result[0]
        latent = EmptyFlux2LatentImage.execute(**inputs["latent"]).result[0]

        pos_conds = [nodes.CLIPTextEncode().encode(clip, p)[0] for p in prompts]
        rows = []
        for p_idx, positive in enumerate(pos_conds):
            for seed in seeds:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                c0 = w4a8_calls
                start = time.perf_counter()
                ks_args = dict(inputs["ks"], model=patched, positive=positive, negative=negative, latent_image=latent, seed=seed)
                samples, = nodes.KSampler().sample(**ks_args)
                decoded, = nodes.VAEDecode().decode(vae, samples)
                if not torch.isfinite(decoded).all().item():
                    raise ValueError(f"Non-finite decoded output for p{p_idx} s{seed}")
                pixels = (decoded[0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                elapsed = time.perf_counter() - start
                fname = f"p{p_idx:02d}_s{seed}.png"
                Image.fromarray(pixels).save(root / fname)
                rows.append(dict(
                    prompt_index=p_idx,
                    seed=seed,
                    file=fname,
                    seconds=elapsed,
                    w4a8_mm_completed=w4a8_calls - c0,
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                ))
        (root / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Rendered {len(rows)} images to {root}, total w4a8 calls={w4a8_calls}", flush=True)


if __name__ == "__main__":
    main()
