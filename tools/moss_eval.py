"""Phase 0 checks for MOSS-Image: does the model earn a suite? (dev tool, not a node)

Queues fixed prompt x seed grids on a running local ComfyUI through its HTTP API with stock nodes only
(UNETLoader + CLIPLoader moss_image + VAELoader), downloads the results and builds contact sheets to judge by eye.

Sampling follows the official ComfyUI templates (image_moss_image_t2i / image_moss_image_base_t2i): 25 steps, CFG 2.5,
er_sde / simple, model shift 5.0 (the model default), the templates' negative prompt; the Base template enables
SkipLayerGuidanceDiTSimple on single block 22 for the first half of the steps, the SFT template does not.

Modes:
  bias      A neutral prompt over many seeds, then explicit prompts that ask for other ethnicities, on SFT and Base.
            Count the results by hand on the sheets.
  compare   SFT and Base on the same prompts and seeds (photo, illustration, text, line breaks, layout JSON).
  groups    Which layer groups carry the SFT -> Base change, through HaDeS MOSS Model Block Merger (this suite must be
            installed): swap-in = SFT with one group from Base (reference SFT), leave-out = Base with one group from SFT
            (reference Base). share = distance(variant, reference) / distance(Base, SFT) per image metric:
              layout   - RMSE of a 32x32 lightness thumbnail (composition, silhouettes, light masses)
              edges    - 1 - correlation of Sobel edge maps at 128 px (line placement / structure)
              texture  - distance of the high-frequency log power spectrum (line weight, grain, detail)
              color    - distance of Lab mean / std (palette and contrast)
            Every config samples without SLG, so the only difference between them is the merge.
  bias_merge  The two neutral prompts (café woman, fisherman) over --neutral-seeds seeds on SFT and Base with and without the
            Base template's SLG, and on SFT with its later layer groups taken from Base. Count the results by hand.

Example:
  python tools/moss_eval.py --mode bias --out H:/moss_eval/bias

Re-running with the same --out reuses downloaded images, so an interrupted run resumes.
"""
import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

BIAS_NEUTRAL = {
    "neutral_woman": "photograph of a woman sitting in a café",
}
BIAS_EXPLICIT = {
    "scandinavian_blonde": "photograph of a blonde Scandinavian woman with pale skin and blue eyes sitting in a café",
    "black_man_beard": "photograph of a Black man with a full beard sitting in a café",
    "older_latina": "photograph of an older Latina woman in her sixties sitting in a café",
}

# The layout prompt is built like the core MOSSImageLayoutPrompt node does it: pixel boxes centred on the canvas, yxyx.
LAYOUT = json.dumps({
    "summary": {"caption": "Minimalist concert poster on off-white paper, a red sun behind a black mountain silhouette, bold sans-serif type."},
    "items": [
        {"label": "object", "caption": "large red sun behind a black mountain silhouette", "bbox": [-352, -352, 96, 352]},
        {"label": "text", "caption": "huge bold black condensed title", "text": "NORTHERN\nLIGHTS", "bbox": [160, -416, 352, 416]},
        {"label": "text", "caption": "small black date line", "text": "14 MARCH 2027 · OSLO", "bbox": [384, -256, 448, 256]},
    ],
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
}, ensure_ascii=False, separators=(",", ":"))

COMPARE = {
    "photo_portrait": "Studio photograph of an elderly fisherman with a weathered face and a grey knitted cap, soft window light from the left, shallow depth of field, 85mm lens, muted colors.",
    "photo_street": "Wide photograph of a busy European market street at golden hour, people walking between fruit stalls, long shadows, warm light, 35mm film look.",
    "illustration": "Watercolor illustration of a fox reading a book under a mushroom in a rainy forest, soft pastel palette, visible paper texture.",
    "text_sign": "Photograph of a hand-painted wooden sign outside a bakery that reads \"FRESH BREAD DAILY\" in large white letters and \"open 7am - 3pm\" in smaller letters below, morning light.",
    "text_linebreaks": "Flat graphic poster on a mustard yellow background with black text in three lines:\n\"THE QUIET\"\n\"HOURS OF\"\n\"THE CITY\"\nsmall black line drawing of a skyline at the bottom.",
    "layout_json": LAYOUT,
}

GROUP_PROMPTS = {
    "neutral_woman": BIAS_NEUTRAL["neutral_woman"],
    "photo_portrait": COMPARE["photo_portrait"],
    "illustration": COMPARE["illustration"],
}
MERGE_GROUPS = [f"layers_{g * 6:02d}_{g * 6 + 5:02d}" for g in range(7)] + ["text_embedder", "time_modulation", "io"]
METRICS = ("layout", "edges", "texture", "color")

MODELS = {
    "sft": {"slg": False},
    "base": {"slg": True},
}


# ------------------------------------------------------------------------------------------------
# ComfyUI API
# ------------------------------------------------------------------------------------------------
def _request(server, path, data=None):
    req = urllib.request.Request(f"{server}{path}", data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def run_graph(server, graph):
    prompt_id = json.loads(_request(server, "/prompt", json.dumps({"prompt": graph}).encode()))["prompt_id"]
    while True:
        hist = json.loads(_request(server, f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed on prompt {prompt_id}: {entry['status'].get('messages')}")
            img = entry["outputs"]["out"]["images"][0]
            query = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            return Image.open(io.BytesIO(_request(server, f"/view?{query}"))).convert("RGB")
        time.sleep(0.5)


def merge_node(ratios):
    """HaDeS MOSS Model Block Merger of --sft (model1) and --base (model2); `ratios` maps MERGE_GROUPS names to ratios."""
    return {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["sft", 0], "model2": ["base", 0], "recipe_override": "", **ratios}}


def build_graph(args, unet, slg, prompt, seed):
    """`unet` is a diffusion_models file name, or {"ratios": ...} for a merge of --sft and --base."""
    g = {
        "unet": merge_node(unet["ratios"]) if isinstance(unet, dict) else {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": "moss_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": args.size, "height": args.size, "batch_size": 1}},
        "ks": {"class_type": "KSampler", "inputs": {
            "model": ["slg" if slg else "unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": args.steps, "cfg": args.cfg, "sampler_name": args.sampler, "scheduler": args.scheduler, "denoise": 1.0,
        }},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "out": {"class_type": "PreviewImage", "inputs": {"images": ["dec", 0]}},
    }
    if isinstance(unet, dict):
        g["sft"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.sft, "weight_dtype": "default"}}
        g["base"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.base, "weight_dtype": "default"}}
    if slg:
        g["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
            "model": ["unet", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 0.5}}
    return g


def jobs_for(args):
    """[(prompt name, prompt, seeds)]"""
    if args.mode == "groups":
        return [(n, p, list(range(args.seed0, args.seed0 + args.seeds))) for n, p in GROUP_PROMPTS.items()]
    if args.mode == "bias_merge":
        return [(n, GROUP_PROMPTS[n], list(range(args.seed0, args.seed0 + args.neutral_seeds))) for n in ("neutral_woman", "photo_portrait")]
    if args.mode == "bias":
        return [(n, p, list(range(args.seed0, args.seed0 + args.neutral_seeds))) for n, p in BIAS_NEUTRAL.items()] + \
               [(n, p, list(range(args.seed0, args.seed0 + args.seeds))) for n, p in BIAS_EXPLICIT.items()]
    return [(n, p, list(range(args.seed0, args.seed0 + args.seeds))) for n, p in COMPARE.items()]


def group_configs():
    """config name -> {"ratios": ..., "slg": ...}: SFT, Base, and every group swapped in / left out."""
    cfg = {"A": {"ratios": dict.fromkeys(MERGE_GROUPS, 0.0), "slg": False}, "B": {"ratios": dict.fromkeys(MERGE_GROUPS, 1.0), "slg": False}}
    for g in MERGE_GROUPS:
        cfg[f"in_{g}"] = {"ratios": {k: float(k == g) for k in MERGE_GROUPS}, "slg": False}
        cfg[f"out_{g}"] = {"ratios": {k: float(k != g) for k in MERGE_GROUPS}, "slg": False}
    return cfg


def bias_merge_configs():
    """SFT and Base with and without the Base template's SLG, and SFT with its later layers taken from Base."""
    def sft_with(groups):
        return {"ratios": {k: float(k in groups) for k in MERGE_GROUPS}, "slg": False}
    return {
        "A": sft_with(()),
        "A_slg": {**sft_with(()), "slg": True},
        "B": sft_with(MERGE_GROUPS),
        "B_slg": {**sft_with(MERGE_GROUPS), "slg": True},
        "A_base_24_35": sft_with(("layers_24_29", "layers_30_35")),
        "A_base_18_41": sft_with(("layers_18_23", "layers_24_29", "layers_30_35", "layers_36_41")),
        "A_base_all_but_io": sft_with([g for g in MERGE_GROUPS if g != "io"]),
    }


# ------------------------------------------------------------------------------------------------
# Image statistics (same as tools/anima_ablation.py in HaDeS Anima Suite)
# ------------------------------------------------------------------------------------------------
def srgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def features(img):
    lab = srgb_to_lab(np.asarray(img.resize((512, 512), Image.LANCZOS), dtype=np.float64) / 255.0)
    L = lab[..., 0]
    thumb = np.asarray(Image.fromarray(L.astype(np.float32)).resize((32, 32), Image.BOX))
    small = np.asarray(Image.fromarray(L.astype(np.float32)).resize((128, 128), Image.BOX))
    gx = small[1:-1, 2:] - small[1:-1, :-2] + 0.5 * (small[:-2, 2:] - small[:-2, :-2] + small[2:, 2:] - small[2:, :-2])
    gy = small[2:, 1:-1] - small[:-2, 1:-1] + 0.5 * (small[2:, :-2] - small[:-2, :-2] + small[2:, 2:] - small[:-2, 2:])
    # Radially averaged log power spectrum, upper half of the frequency range only (line / grain / detail)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(L - L.mean()))) ** 2
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - 256, xx - 256).astype(int)
    radial = np.bincount(r.ravel(), spec.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    texture = np.log10(np.array([radial[b].mean() for b in np.array_split(np.arange(64, 256), 16)]) + 1e-9)
    color = np.concatenate([lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)])
    return {"layout": thumb, "edges": np.hypot(gx, gy).ravel(), "texture": texture, "color": color}


def distance(fa, fb):
    return {
        "layout": float(np.sqrt(np.mean((fa["layout"] - fb["layout"]) ** 2))),
        "edges": float(1.0 - np.corrcoef(fa["edges"], fb["edges"])[0, 1]),
        "texture": float(np.linalg.norm(fa["texture"] - fb["texture"])),
        "color": float(np.linalg.norm(fa["color"] - fb["color"])),
    }


def analyze_groups(feats, samples):
    def total(name, ref):
        return {m: sum(distance(feats[name][s], feats[ref][s])[m] for s in samples) for m in METRICS}
    ab = total("A", "B")
    report = {"samples": len(samples), "A_to_B_mean_distance": {m: v / len(samples) for m, v in ab.items()}}
    for kind, prefix, ref in (("swap_in", "in", "A"), ("leave_out", "out", "B")):
        report[kind] = {g: {m: round(v / ab[m], 3) for m, v in total(f"{prefix}_{g}", ref).items()} for g in MERGE_GROUPS}
    return report


# ------------------------------------------------------------------------------------------------
# Contact sheets
# ------------------------------------------------------------------------------------------------
def contact_sheet(out_dir, name, prompt, seeds, models, cols=8, tile=256):
    """One sheet per prompt: a block of rows per model, seeds left to right."""
    rows_per_model = -(-len(seeds) // cols)
    width = tile * min(cols, len(seeds))
    head = 40
    sheet = Image.new("RGB", (width, head + len(models) * rows_per_model * (tile + 18)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), f"{name}: {prompt[:150]}", fill=(230, 230, 230))
    draw.text((6, 22), f"{prompt[150:300]}", fill=(230, 230, 230))
    y0 = head
    for m in models:
        for i, seed in enumerate(seeds):
            x, y = (i % cols) * tile, y0 + (i // cols) * (tile + 18)
            sheet.paste(Image.open(os.path.join(out_dir, m, name, f"{seed}.png")).resize((tile, tile), Image.LANCZOS), (x, y + 18))
            draw.text((x + 4, y + 3), f"{m} seed {seed}", fill=(220, 220, 220))
        y0 += rows_per_model * (tile + 18)
    path = os.path.join(out_dir, f"sheet_{name}.jpg")
    sheet.save(path, quality=88)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("bias", "compare", "groups", "bias_merge"), default="bias")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--sft", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--base", default="mossImage_base_bf16.safetensors")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--clip", default="mossImage_base_txt_int4.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--out", required=True)
    p.add_argument("--seed0", type=int, default=1000, help="first seed; seeds are consecutive")
    p.add_argument("--neutral-seeds", type=int, default=16, help="seeds for the neutral bias prompt")
    p.add_argument("--seeds", type=int, default=6, help="seeds for every other prompt")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=2.5)
    p.add_argument("--sampler", default="er_sde")
    p.add_argument("--scheduler", default="simple")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = jobs_for(args)
    if args.mode in ("groups", "bias_merge"):
        run_merges(args, jobs)
        return
    total, done = len(args.models) * sum(len(s) for _, _, s in jobs), 0
    t0 = time.time()
    # one model at a time, so ComfyUI swaps the diffusion model once per model
    for m in args.models:
        unet = getattr(args, m)
        for name, prompt, seeds in jobs:
            os.makedirs(os.path.join(args.out, m, name), exist_ok=True)
            for seed in seeds:
                path = os.path.join(args.out, m, name, f"{seed}.png")
                if not os.path.exists(path):
                    run_graph(args.server, build_graph(args, unet, MODELS[m]["slg"], prompt, seed)).save(path)
                done += 1
                print(f"[{done}/{total}] {m} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)

    settings = {k: getattr(args, k) for k in ("mode", "sft", "base", "clip", "vae", "size", "steps", "cfg", "sampler", "scheduler")}
    settings.update(negative=NEGATIVE, slg={m: MODELS[m]["slg"] for m in args.models}, prompts={n: p for n, p, _ in jobs})
    with open(os.path.join(args.out, f"settings_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, args.models))


def run_merges(args, jobs):
    configs = group_configs() if args.mode == "groups" else bias_merge_configs()
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for c, unet in configs.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                run_graph(args.server, build_graph(args, unet, unet["slg"], prompt, seed)).save(path)
            if args.mode == "groups":
                feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)

    if args.mode == "bias_merge":
        for name, prompt, seeds in jobs:
            print(contact_sheet(args.out, name, prompt, seeds, list(configs)))
        return
    report = analyze_groups(feats, list(feats["A"]))
    report.update(sft=args.sft, base=args.base, steps=args.steps, cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler, slg=False)
    with open(os.path.join(args.out, "report_groups.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, list(configs), cols=len(seeds), tile=160))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
