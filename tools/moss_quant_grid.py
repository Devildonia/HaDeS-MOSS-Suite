"""Render the ten historical prompts with the saved Base pilot graph and a local DiT checkpoint."""
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
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    import folder_paths
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import convrot_w4a4, w4a8_int8
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    prompts = [p.strip() for p in args.prompts.read_text(encoding="utf-8-sig").splitlines() if p.strip()]
    inputs = {key: node["inputs"] for key, node in graph.items()}
    folder_paths.add_model_folder_path("diffusion_models", str(args.model.resolve().parent), is_default=True)
    inputs["unet"]["unet_name"] = args.model.name
    if Path(folder_paths.get_full_path_or_raise("diffusion_models", args.model.name)).resolve() != args.model.resolve():
        raise RuntimeError("Model folder resolved to a different checkpoint")

    # Count completed calls only. The stock layout can fall back after a failed dispatch.
    calls = {}

    def count_operator(module, name):
        original = getattr(module, name)
        calls[name] = 0

        def counted(*values, **kwargs):
            output = original(*values, **kwargs)
            calls[name] += 1
            return output

        setattr(module, name, counted)

    count_operator(ck, "scaled_mm_nvfp4")
    count_operator(convrot_w4a4, "convrot_w4a4_linear")
    count_operator(w4a8_int8, "w4a8_int8_linear")
    with torch.inference_mode():
        model, = nodes.UNETLoader().load_unet(**inputs["unet"])
        clip, = nodes.CLIPLoader().load_clip(**inputs["clip"])
        vae, = nodes.VAELoader().load_vae(**inputs["vae"])
        negative, = nodes.CLIPTextEncode().encode(clip, inputs["neg"]["text"])
        patched = SkipLayerGuidanceDiTSimple.execute(**dict(inputs["slg"], model=model)).result[0]
        latent = EmptyFlux2LatentImage.execute(**inputs["latent"]).result[0]
        rows = []
        for index in (0, 6, 10, 14, 20, 25, 27, 30, 34, 35):
            inputs["pos"]["text"] = prompts[index]
            (root / f"{index:02}.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
            positive, = nodes.CLIPTextEncode().encode(clip, prompts[index])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = calls.copy()
            start = time.perf_counter()
            settings = dict(inputs["ks"], model=patched, positive=positive, negative=negative, latent_image=latent)
            samples, = nodes.KSampler().sample(**settings)
            decoded, = nodes.VAEDecode().decode(vae, samples)
            if not torch.isfinite(decoded).all().item():
                raise ValueError(f"Non-finite decoded output for prompt {index}")
            pixels = (decoded[0].float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            rows.append(dict(index=index, seconds=time.perf_counter()-start,
                             nvfp4_mm_completed=calls["scaled_mm_nvfp4"]-before["scaled_mm_nvfp4"],
                             convrot_mm_completed=calls["convrot_w4a4_linear"]-before["convrot_w4a4_linear"],
                             w4a8_mm_completed=calls["w4a8_int8_linear"]-before["w4a8_int8_linear"],
                             peak_allocated=torch.cuda.max_memory_allocated(),
                             rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
            Image.fromarray(pixels).save(root / f"{index:02}.png")
            (root / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
