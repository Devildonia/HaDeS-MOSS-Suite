"""LoRA atlas: what each layer group of a LoRA does on MOSS-Image, and a few block-weight recipes checked on other prompts.

1. Calibration: the moss_eval.py LoRA ablation (no LoRA, full LoRA, each group alone, each group left out) on --calib prompts.
   Per group: its share of the full LoRA's change in style (texture, color, edges, averaged) and in composition (layout).
2. Recipes, built from those numbers only:
     full        every group at 1
     style       only the two groups with the largest style share over layout share
     keep_comp   full, minus the group whose removal takes away the most layout change
     half        every group at 0.5: plain strength reduction, the baseline a block recipe has to beat
3. Validation on --valid prompts (not used to build the recipes): per recipe, the share of the full LoRA's style change kept
   and the share of its composition change kept. A good recipe keeps style and drops composition change.

The numbers belong to this LoRA, this model and these prompts; they are not rules about the layers. The metrics are low-level
image statistics (moss_eval.features): they see texture, color, edges and layout, not identity or concepts.

  python tools/moss_lora_atlas.py --lora MOSS/x.safetensors --out I:/atlas/x [--lora-model mossImage_base_bf16.safetensors]
  (needs a running ComfyUI with this Suite; run from anywhere)"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moss_eval as me  # noqa: E402

GROUPS = me.LORA_GROUPS[:-1]
STYLE = ("texture", "color", "edges")
CALIB = {
    "portrait": "A photograph of a woman with short dark hair in a denim jacket, standing by a window in soft daylight.",
    "street": "An illustration of a narrow street at dusk with shop signs, a cyclist and warm window lights.",
    "still_life": "A still life of a teapot, two cups and lemons on a wooden table, side light.",
}
VALID = {
    "knight": "A knight in silver armor standing in a castle courtyard, banners in the wind.",
    "cafe": "A barista in an apron pouring coffee in a small cafe, morning light through the window.",
    "forest": "A young man sitting on a mossy rock by a forest stream, reading a book.",
}


def graph(args, weights, prompt, seed):
    g = me.build_graph(args, {"lora": weights}, False, prompt, seed)
    if args.turbo:  # few-step LoRA on the model before the atlas LoRA, at full strength in every config
        g["turbo"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["sft", 0], "lora_name": args.turbo, "strength_model": 1.0}}
        g["unet"]["inputs"]["model"] = ["turbo", 0]
    return g


def render(args, name, weights, prompts, log):
    """{sample key: features} of one config over prompts x seeds; PNGs cached in --out/<name>/."""
    feats = {}
    for pname, prompt in prompts.items():
        for seed in range(args.seed0, args.seed0 + args.seeds):
            path = os.path.join(args.out, name, f"{pname}_{seed}.png")
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                me.run_graph(args.server, graph(args, weights, prompt, seed)).save(path)
            feats[f"{pname}_{seed}"] = me.features(Image.open(path).convert("RGB"))
            log(name, pname, seed)
    return feats


def shares(feats, a, b, x):
    """Per metric: distance(a, x) / distance(a, b), summed over samples (1 = as far from a as the full LoRA is)."""
    keys = list(feats[a])
    tot = lambda p, q: {m: sum(me.distance(feats[p][k], feats[q][k])[m] for k in keys) for m in me.METRICS}  # noqa: E731
    ab, ax = tot(a, b), tot(a, x)
    return {m: ax[m] / ab[m] if ab[m] > 0 else 0.0 for m in me.METRICS}


def style_of(s):
    return float(np.mean([s[m] for m in STYLE]))


def recipes(profile):
    full = dict.fromkeys(GROUPS, 1.0)
    gap = {g: profile[g]["alone_style"] - profile[g]["alone_layout"] for g in GROUPS}
    style = sorted(GROUPS, key=lambda g: -gap[g])[:2]  # the two groups that move style most over layout
    worst = max(GROUPS, key=lambda g: profile[g]["removed_layout"])
    return {
        "full": full,
        "style": {g: float(g in style) for g in GROUPS},
        "keep_comp": {**full, worst: 0.0},
        "half": dict.fromkeys(GROUPS, 0.5),
    }


def sheet(args, names, prompts, path):
    t = 256
    fnt = ImageFont.load_default(14)
    keys = [f"{p}_{s}" for p in prompts for s in range(args.seed0, args.seed0 + args.seeds)]
    img = Image.new("RGB", (t * len(names), (t + 18) * len(keys) + 18), "white")
    d = ImageDraw.Draw(img)
    for j, n in enumerate(names):
        d.text((j * t + 4, 2), n, fill="black", font=fnt)
        for i, k in enumerate(keys):
            img.paste(Image.open(os.path.join(args.out, n, f"{k}.png")).convert("RGB").resize((t, t)), (j * t, 18 + i * (t + 18) + 18))
            if j == 0:
                d.text((4, 18 + i * (t + 18) + 2), k, fill="gray", font=fnt)
    img.save(path, quality=88)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora", required=True, help="models/loras name of the LoRA")
    p.add_argument("--lora-model", default="mossImage_base_bf16.safetensors", help="diffusion_models file the LoRA is applied to")
    p.add_argument("--turbo", help="models/loras name of a few-step LoRA applied under every config (then use its steps / cfg)")
    p.add_argument("--clip", default="qwen3.5_9b_int8_convrot.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--out", required=True)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=2.5)
    p.add_argument("--sampler", default="er_sde")
    p.add_argument("--scheduler", default="simple")
    args = p.parse_args(argv)
    args.sft = args.lora_model  # moss_eval's LoRA graph reads the model from --sft's node name
    os.makedirs(args.out, exist_ok=True)
    t0, n = time.time(), [0]

    def log(name, pname, seed):
        n[0] += 1
        print(f"[{n[0]}] {name} {pname} {seed} ({time.time() - t0:.0f}s)", flush=True)

    # 1. calibration
    full = dict.fromkeys(GROUPS, 1.0)
    configs = {"none": {}, "full": full, **{f"in_{g}": {g: 1.0} for g in GROUPS}, **{f"out_{g}": {**full, g: 0.0} for g in GROUPS}}
    feats = {c: render(args, f"calib/{c}", w, CALIB, log) for c, w in configs.items()}
    profile = {}
    for g in GROUPS:
        alone = shares(feats, "none", "full", f"in_{g}")
        removed = shares(feats, "none", "full", f"out_{g}")
        profile[g] = {"alone_style": round(style_of(alone), 3), "alone_layout": round(alone["layout"], 3),
                      # what the full LoRA loses without this group: 1 - share kept
                      "removed_style": round(1 - style_of(removed), 3), "removed_layout": round(1 - removed["layout"], 3)}

    # 2. recipes, 3. validation on other prompts
    rec = recipes(profile)
    vfeats = {"none": render(args, "valid/none", {}, VALID, log)}
    for r, w in rec.items():
        vfeats[r] = render(args, f"valid/{r}", w, VALID, log)
    result = {}
    for r in rec:
        s = shares(vfeats, "none", "full", r)
        result[r] = {"weights": rec[r], "style_kept": round(style_of(s), 3), "layout_kept": round(s["layout"], 3)}

    report = {"lora": args.lora, "lora_model": args.lora_model, "turbo": args.turbo, "steps": args.steps, "cfg": args.cfg,
              "sampler": args.sampler, "scheduler": args.scheduler, "size": args.size, "seeds": args.seeds,
              "calib_prompts": CALIB, "valid_prompts": VALID, "profile": profile, "recipes": result}
    with open(os.path.join(args.out, "atlas.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    lines = [f"# LoRA atlas: {args.lora} on {args.lora_model}", "",
             "Share of the full LoRA's change, on the calibration prompts (style = texture, color, edges).", "",
             "| Group | alone: style | alone: layout | removed: style lost | removed: layout lost |", "|---|---|---|---|---|"]
    lines += [f"| {g} | {v['alone_style']:.2f} | {v['alone_layout']:.2f} | {v['removed_style']:.2f} | {v['removed_layout']:.2f} |"
              for g, v in profile.items()]
    lines += ["", "Recipes on the validation prompts (share of the full LoRA's change kept; full = 1.00 / 1.00).", "",
              "| Recipe | style kept | layout kept | groups at 1 (0.5 for half) |", "|---|---|---|---|"]
    lines += [f"| {r} | {v['style_kept']:.2f} | {v['layout_kept']:.2f} | "
              f"{', '.join(g for g, w in v['weights'].items() if w > 0) or '-'} |" for r, v in result.items()]
    lines += ["", "Valid for this LoRA, this model and these prompts only; low-level statistics, not identity or concepts."]
    with open(os.path.join(args.out, "atlas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sheet(args, ["valid/none"] + [f"valid/{r}" for r in rec], VALID, os.path.join(args.out, "valid_sheet.jpg"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
