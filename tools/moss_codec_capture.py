"""Capture sampled linear inputs from early, middle and late BF16 MOSS blocks for the codec pilot."""
import argparse
import json
import sys
from pathlib import Path

from safetensors.torch import save_file

BLOCKS = (0, 20, 41)
FAMILIES = ("qkv", "o", "ffn_up_gate", "ffn_down")
# Disjoint scenes for calibration and validation; the final acceptance set stays unused.
SCENES = {"cal_street": 1, "cal_watercolor": 21, "val_wildlife": 8, "val_pixel": 32}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("core", "graph", "prompts", "out"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--blocks", action="store_true", help="Save full validation block calls and reference outputs")
    p.add_argument("--hms-diagonal", action="store_true", help="Collect calibration-only ConvRot energies for all blocks")
    p.add_argument("--hms-rows", type=Path, help="Collect the same calibration rows using prior diagonal JSON manifests")
    p.add_argument("--block-ids", type=int, nargs="+", help="Limit an HMS row capture to these blocks")
    p.add_argument("--scenes", type=str, default=None, help="Optional comma-separated name=idx pairs overriding SCENES")
    args = p.parse_args()
    scenes = SCENES
    if args.scenes:
        scenes = {k.strip(): int(v) for item in args.scenes.split(",") for k, v in [item.split("=")]}
    if (args.hms_rows or args.hms_diagonal) and (args.blocks or any(not k.startswith("cal_") for k in scenes)):
        p.error("--hms-diagonal requires only calibration scenes and projection hooks")
    if args.hms_rows and args.hms_diagonal or args.block_ids and not args.hms_rows:
        p.error("Choose one HMS mode; block IDs apply to rows only")
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    inputs = {k:n["inputs"] for k,n in graph.items()}
    prompts = [s.strip() for s in args.prompts.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    with torch.inference_mode():
        model, = nodes.UNETLoader().load_unet(**inputs["unet"])
        clip, = nodes.CLIPLoader().load_clip(**inputs["clip"])
        negative, = nodes.CLIPTextEncode().encode(clip, inputs["neg"]["text"])
        patched = SkipLayerGuidanceDiTSimple.execute(**dict(inputs["slg"], model=model)).result[0]
        latent = EmptyFlux2LatentImage.execute(**inputs["latent"]).result[0]
        def hook(name):
            def capture(module, values):
                i = counts[name]
                counts[name] += 1
                if args.hms_rows:
                    if i in manifest[name]["indices"]:
                        x = values[0]
                        positions = torch.linspace(0, 4095, 48, device=x.device).long()
                        captures[name].append(x[:, positions].reshape(-1,x.shape[-1]).cpu().contiguous())
                    return
                if args.hms_diagonal:
                    from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight
                    x = values[0]
                    positions = torch.linspace(0, 4095, 48, device=x.device).long()
                    rows = x[:, positions].reshape(-1, x.shape[-1]).float()
                    rotated = rotate_int8_convrot_weight(rows.contiguous(), 256)
                    captures[name].append((rotated.square().sum(0), len(rows)))
                    return
                if i in (0, 16, 32, 48):
                    x = values[0]
                    # Evenly spaced image tokens from every batch item, excluding appended text tokens.
                    positions = torch.linspace(0, 4095, 48, device=x.device).long()
                    captures[name].append(x[:, positions].reshape(-1,x.shape[-1]).cpu().contiguous())
            return capture
        handles = []
        names = []
        def before_block(i):
            def capture(module, values, kwargs):
                count = counts[i]
                counts[i] += 1
                if count in (0, 16, 32, 48):
                    x, mod, pe, options, *refs = values
                    if any(r is not None for r in refs):
                        raise ValueError("Only text-to-image replay is supported")
                    batch = dict(x=x.cpu(), mod=torch.stack(mod).cpu(), pe=pe.cpu())
                    if kwargs.get("mask") is not None:
                        batch["mask"] = kwargs["mask"].cpu()
                    pending[i] = batch
            return capture

        def after_block(i):
            def capture(module, values, output):
                if i in pending:
                    batch = pending.pop(i)
                    batch["y"] = output.cpu()
                    path = args.out / f"{scene}_{i:02}_{counts[i]-1:02}.safetensors"
                    save_file({k:v.contiguous() for k,v in batch.items()}, str(path))
            return capture

        for i in (args.block_ids or (range(42) if args.hms_diagonal or args.hms_rows else BLOCKS)):
            if args.blocks:
                block = model.model.diffusion_model.layers[i]
                handles.append(block.register_forward_pre_hook(before_block(i), with_kwargs=True))
                handles.append(block.register_forward_hook(after_block(i)))
                names.append(i)
                continue
            for f in FAMILIES:
                name = f"layers.{i}.proj.{f}"
                layer = model.model.diffusion_model.layers[i].proj[f]
                layer.comfy_force_forward = True
                handles.append(layer.register_forward_pre_hook(hook(name)))
                names.append(name)
        try:
            for scene, idx in scenes.items():
                if args.blocks and scene.startswith("cal_"):
                    continue
                pending = {}
                captures = {k:[] for k in names}
                counts = dict.fromkeys(names,0)
                if args.hms_rows:
                    prior = json.loads((args.hms_rows / (scene+".json")).read_text(encoding="utf-8"))
                    if prior["prompt"] != prompts[idx]:
                        raise ValueError("Prompt mismatch against diagonal manifest")
                    manifest = prior["projections"]
                positive, = nodes.CLIPTextEncode().encode(clip, prompts[idx])
                nodes.KSampler().sample(**dict(inputs["ks"], model=patched, positive=positive,
                                               negative=negative, latent_image=latent))
                if args.hms_diagonal:
                    diagonals, provenance = {}, {}
                    for name, calls in captures.items():
                        total = len(calls)
                        if total < 4:
                            raise ValueError(f"{name} only received {total} calls")
                        indices = [round(t * (total - 1) / 49) for t in (0, 16, 32, 48)]
                        rows = sum(calls[j][1] for j in indices)
                        energy = sum(calls[j][0] for j in indices) / rows
                        if not energy.isfinite().all() or energy.max() <= 0:
                            raise ValueError(f"Invalid energy for {name}")
                        diagonals[name] = energy.cpu()
                        provenance[name] = dict(total=total, indices=indices, rows=rows)
                    save_file(diagonals, str(args.out / (scene+".safetensors")))
                    (args.out / (scene+".json")).write_text(json.dumps(dict(
                        index=idx, prompt=prompts[idx], projections=provenance)), encoding="utf-8")
                    print(scene, "saved", len(diagonals), "diagonals", flush=True)
                    continue
                if args.blocks:
                    if len(list(args.out.glob(scene+"_*.safetensors"))) != 12:
                        raise RuntimeError(f"Incomplete block capture: {counts}")
                    continue
                if any(len(parts)!=4 for parts in captures.values()):
                    raise RuntimeError(f"Incomplete activation capture: {counts}")
                if args.hms_rows and any(counts[k] != manifest[k]["total"] for k in names):
                    raise ValueError("Invocation counts differ from diagonal capture")
                save_file({k:torch.cat(v) for k,v in captures.items()}, str(args.out/(scene+".safetensors")))
                print(scene, counts, flush=True)
        finally:
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    main()
