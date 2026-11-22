"""Repeat a MOSS Base workflow using stock ComfyUI nodes directly in an isolated process."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from moss_eval import build_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage

    graph = build_graph(
        SimpleNamespace(clip="qwen3.5_9b_w4a8_hybrid.safetensors", vae="flux2-vae.safetensors",
                        size=1024, steps=25, cfg=2.5, sampler="er_sde", scheduler="simple"),
        "mossImage_base_bf16.safetensors", True,
        args.prompts.read_text(encoding="utf-8-sig").splitlines()[0], 1001,
    )
    (root / "graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    inputs = {key: node["inputs"] for key, node in graph.items()}
    with torch.inference_mode():
        start = time.perf_counter()
        model, = nodes.UNETLoader().load_unet(**inputs["unet"])
        clip, = nodes.CLIPLoader().load_clip(**inputs["clip"])
        vae, = nodes.VAELoader().load_vae(**inputs["vae"])
        pos, neg = [nodes.CLIPTextEncode().encode(clip, inputs[key]["text"])[0] for key in ("pos", "neg")]
        slg = dict(inputs["slg"], model=model)
        patched = SkipLayerGuidanceDiTSimple.execute(**slg).result[0]
        latent = EmptyFlux2LatentImage.execute(**inputs["latent"]).result[0]
        settings = dict(inputs["ks"], model=patched, positive=pos, negative=neg, latent_image=latent)
        torch.cuda.synchronize()
        setup = time.perf_counter() - start
        runs, images = [], []
        for i in range(2):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            sampled, = nodes.KSampler().sample(**settings)
            decoded, = nodes.VAEDecode().decode(vae, sampled)
            pixels = (decoded[0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            runs.append(dict(seconds=time.perf_counter()-start, peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                             pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
            Image.fromarray(pixels).save(root / f"repeat_{i}.png")
            images.append(pixels.astype(np.int16))
        delta = np.abs(images[0] - images[1])
        result = dict(setup_seconds=setup, runs=runs, mean_abs_rgb=float(delta.mean()),
                      max_abs_rgb=int(delta.max()), changed_pixels=int(delta.any(axis=2).sum()),
                      torch=torch.__version__, gpu=torch.cuda.get_device_name())
        (root / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
