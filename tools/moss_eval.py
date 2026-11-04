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
  lora      Like groups, for one LoRA (--lora) on --lora-model through HaDeS MOSS LoRA Block Weight: A = the model without
            the LoRA, B = the full LoRA, each layer group swapped in / left out.
  shift     --sft at several square sizes and flow shifts (SHIFT_GRID, ModelSamplingAuraFlow). Reports the low-frequency spectral
            slope (composition scale, compared with 1024 at the default shift 5.0), native-resolution detail and the layout change
            against shift 5.0 at the same size.
  slg       SFT and Base (--models) without SLG and with SkipLayerGuidanceDiTSimple on single block 22 (first half / all steps),
            24 and 26 (first half), on text, line-break, layout and portrait prompts. Count text errors by eye on the sheets.
  schedulers  --sft with er_sde / euler x simple / beta / normal x 15 / 25 steps against a 60-step euler / simple reference.
  attention  --attn-models at ATTENTION_SIZES with the attention backend the server was started with, named by --tag (the server's
            argv is saved next to the images). Seconds per image count only runs where the model and both prompts came from
            ComfyUI's cache (sampling + decode). The cache is cleared first. --compare TAG [TAG ...] then reports every tag
            against the first: mean absolute pixel difference (0-255), the image metrics and median seconds per image.
  progressive  --sft sampled in two stages through stock KSamplerAdvanced: the first --switch steps at --low-size, the latent
            upscaled (bilinear) to --size, the remaining steps there. renoise = stage 1 ends with the clean-image prediction and
            stage 2 adds fresh noise at that step's sigma; leftover = the noisy latent is upscaled and continued as is. Control:
            renoise at full size (no resize). --resume K:B backtracks: K steps at --low-size, then --size re-noised from step B.
            Reports seconds per image, distance to the one-stage render of the same seed and the
            change in fine detail; contact sheets to judge by eye.
  turbo     Few-step LoRAs (TURBO_LORAS) on --sft at their cards' settings (euler / simple, CFG 1), each loaded the usual way
            (LoraLoaderModelOnly: the delta is added to the weights and rounded to their dtype) and in bypass mode
            (LoraLoaderBypassModelOnly: the weights stay as they are and the LoRA runs beside them), plus --sft alone at the same
            few-step settings. Reports seconds per image and distances to the --steps / --cfg / --sampler render of the same seed.
  turbo_groups  Each TURBO_LORAS LoRA at its few-step settings with only some layer groups kept (TURBO_GROUP_SETS, through HaDeS MOSS
            LoRA Block Weight, the usual weight patch); distances to the full LoRA's render of the same seed. Judge broken / clean by eye.
  turbo_files  --turbo-files NAME=FILE ...: models/loras files derived from a TURBO_LORAS LoRA (NAME starts with its key, which sets
            the step count), rendered with --loader; distances to that LoRA's full render in --ref-dir/<key>_all (a turbo_groups run),
            or, without --ref-dir, to the NAME <key>_orig of the same run. --prompts picks COMPARE prompts instead of the default three.
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
LORA_GROUPS = MERGE_GROUPS[:7] + ["other_layers"]
# square side -> flow shifts tried there; 5.0 is the model default (supported_models.MOSSImage)
SHIFT_GRID = {768: (3.0, 5.0), 1024: (3.0, 5.0, 7.0), 1536: (5.0, 7.5), 2048: (5.0, 7.5, 10.0)}
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


def run_graph(server, graph, info=None):
    """The output image; `info`, when given, receives the run's seconds and the node ids ComfyUI served from its cache."""
    prompt_id = json.loads(_request(server, "/prompt", json.dumps({"prompt": graph}).encode()))["prompt_id"]
    while True:
        hist = json.loads(_request(server, f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed on prompt {prompt_id}: {entry['status'].get('messages')}")
            if info is not None:
                msgs = dict(entry["status"]["messages"])
                info["seconds"] = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000.0
                info["cached"] = msgs.get("execution_cached", {}).get("nodes", [])
            img = entry["outputs"]["out"]["images"][0]
            query = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            return Image.open(io.BytesIO(_request(server, f"/view?{query}"))).convert("RGB")
        time.sleep(0.5)


def merge_node(ratios):
    """HaDeS MOSS Model Block Merger of --sft (model1) and --base (model2); `ratios` maps MERGE_GROUPS names to ratios."""
    return {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["sft", 0], "model2": ["base", 0], "recipe_override": "", **ratios}}


def lora_node(args, weights):
    """HaDeS MOSS LoRA Block Weight of --lora on --lora-model; `weights` maps LORA_GROUPS names to multipliers (missing = 0)."""
    return {"class_type": "HaDeSMossLoraBlockWeight", "inputs": {"model": ["sft", 0], "lora_name": args.lora, "strength_model": 1.0,
                                                                 **{g: float(weights.get(g, 0.0)) for g in LORA_GROUPS}}}


def shift_graph(args, size, shift, prompt, seed):
    """--sft at size x size with ModelSamplingAuraFlow (multiplier 1, as the model's own sampling) at the given shift."""
    g = build_graph(args, args.sft, False, prompt, seed)
    g["ms"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": shift}}
    g["ks"]["inputs"]["model"] = ["ms", 0]
    g["latent"]["inputs"].update(width=size, height=size)
    return g


def build_graph(args, unet, slg, prompt, seed):
    """`unet` is a diffusion_models file name, {"ratios": ...} for a merge of --sft and --base, or {"lora": ...} for
    --lora on --lora-model through HaDeS MOSS LoRA Block Weight."""
    if isinstance(unet, dict) and "lora" in unet:
        model_node = lora_node(args, unet["lora"])
    elif isinstance(unet, dict):
        model_node = merge_node(unet["ratios"])
    else:
        model_node = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    g = {
        "unet": model_node,
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
    if isinstance(unet, dict) and "lora" in unet:
        g["sft"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.lora_model, "weight_dtype": "default"}}
    elif isinstance(unet, dict):
        g["sft"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.sft, "weight_dtype": "default"}}
        g["base"] = {"class_type": "UNETLoader", "inputs": {"unet_name": args.base, "weight_dtype": "default"}}
    if slg:
        # True = the official Base template: single block 22 skipped on the negative pass for the first half of the steps
        spec = {"layers": "22", "start": 0.0, "end": 0.5} if slg is True else slg
        g["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
            "model": ["unet", 0], "double_layers": "", "single_layers": spec["layers"], "start_percent": spec["start"], "end_percent": spec["end"]}}
    return g


SLG_CONFIGS = {
    "none": None,
    "22_half": {"layers": "22", "start": 0.0, "end": 0.5},
    "22_full": {"layers": "22", "start": 0.0, "end": 1.0},
    "24_half": {"layers": "24", "start": 0.0, "end": 0.5},
    "26_half": {"layers": "26", "start": 0.0, "end": 0.5},
}
SLG_PROMPTS = ("text_sign", "text_linebreaks", "layout_json", "photo_portrait")
# (sampler, scheduler, steps); the reference is euler / simple at 60 steps
SCHEDULER_CONFIGS = [(s, sch, st) for s in ("er_sde", "euler") for sch in ("simple", "beta", "normal") for st in (15, 25)]
ATTENTION_SIZES = (1024, 2048)
# name -> (models/loras file, steps, the same LoRA with fused proj.qkv); euler / simple / CFG 1 as their model cards give.
# The core bypass loader skips split q/k/v on a fused model, so bypass runs load the fused copy, written losslessly by
# HaDeS MOSS LoRA Resize at rank 512 on --sft.
TURBO_LORAS = {
    "dmd2_8": ("MOSS\\lodestones-8-steps-dmd2-lora.safetensors", 8, "HaDeS_test_dmd2_8_fused.safetensors"),
    "rapid_4": ("MOSS\\moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors", 4, "HaDeS_test_rapid_4_fused.safetensors"),
    # official akhaliq/MOSS-Image-SFT-4-steps files (rapid_4 above is a resize of the 4-step one); no fused copies
    "akh4_4": ("MOSS\\moss_image_sft_4step_v2.0.safetensors", 4, None),
    "akh3_3": ("MOSS\\moss_image_sft_3step_v1.0.safetensors", 3, None),
    "akh2_2": ("MOSS\\moss_image_sft_2step_v0.1.safetensors", 2, None),
}
# name -> LORA_GROUPS kept at 1.0 (the rest at 0)
TURBO_GROUP_SETS = {
    "all": LORA_GROUPS,
    "00_23": LORA_GROUPS[:4],
    "24_41": LORA_GROUPS[4:7],
    "30_41": LORA_GROUPS[5:7],
    "36_41": LORA_GROUPS[6:7],
}


def jobs_for(args):
    """[(prompt name, prompt, seeds)]"""
    if args.mode == "turbo_files" and args.prompts:
        return [(n, COMPARE[n], list(range(args.seed0, args.seed0 + args.seeds))) for n in args.prompts]
    if args.mode in ("groups", "lora", "shift", "schedulers", "attention", "progressive", "turbo", "turbo_groups", "turbo_files"):
        return [(n, p, list(range(args.seed0, args.seed0 + args.seeds))) for n, p in GROUP_PROMPTS.items()]
    if args.mode == "slg":
        return [(n, COMPARE[n], list(range(args.seed0, args.seed0 + args.seeds))) for n in SLG_PROMPTS]
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


def lora_configs():
    """A = --lora-model without the LoRA, B = the full LoRA, and every layer group swapped in / left out.
    other_layers is left out of the ablation: no community MOSS LoRA measured so far trains outside the layers."""
    full = dict.fromkeys(LORA_GROUPS, 1.0)
    cfg = {"A": {"lora": {}, "slg": False}, "B": {"lora": full, "slg": False}}
    for g in LORA_GROUPS[:-1]:
        cfg[f"in_{g}"] = {"lora": {g: 1.0}, "slg": False}
        cfg[f"out_{g}"] = {"lora": {**full, g: 0.0}, "slg": False}
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
    # Slope of log power over 2-32 cycles per image: steeper = larger masses, flatter = busier / repeated layout
    r_lo = np.arange(2, 33)
    slope = np.polyfit(np.log(r_lo), np.log(radial[r_lo] + 1e-9), 1)[0]
    # Upper-half log power of a native-resolution 512 px center crop: fine detail per pixel, comparable at one size only
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    crop = gray[cy - 256:cy + 256, cx - 256:cx + 256]
    cspec = np.abs(np.fft.fftshift(np.fft.fft2(crop - crop.mean()))) ** 2
    detail = np.log10(cspec[(r >= 128) & (r < 256)].mean() + 1e-9)
    return {"layout": thumb, "edges": np.hypot(gx, gy).ravel(), "texture": texture, "color": color, "slope": slope, "detail": detail}


def distance(fa, fb):
    return {
        "layout": float(np.sqrt(np.mean((fa["layout"] - fb["layout"]) ** 2))),
        "edges": float(1.0 - np.corrcoef(fa["edges"], fb["edges"])[0, 1]),
        "texture": float(np.linalg.norm(fa["texture"] - fb["texture"])),
        "color": float(np.linalg.norm(fa["color"] - fb["color"])),
    }


def analyze_groups(feats, samples, groups):
    def total(name, ref):
        return {m: sum(distance(feats[name][s], feats[ref][s])[m] for s in samples) for m in METRICS}
    ab = total("A", "B")
    report = {"samples": len(samples), "A_to_B_mean_distance": {m: v / len(samples) for m, v in ab.items()}}
    for kind, prefix, ref in (("swap_in", "in", "A"), ("leave_out", "out", "B")):
        report[kind] = {g: {m: round(v / ab[m], 3) for m, v in total(f"{prefix}_{g}", ref).items()} for g in groups}
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
    p.add_argument("--mode", choices=("bias", "compare", "groups", "lora", "bias_merge", "shift", "slg", "schedulers", "attention", "progressive",
                                      "turbo", "turbo_groups", "turbo_files"),
                   default="bias")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--sft", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--base", default="mossImage_base_bf16.safetensors")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--lora", help="models/loras name of the LoRA (lora mode)")
    p.add_argument("--lora-model", default="mossImage_sft_bf16.safetensors", help="model the LoRA is applied to (lora mode)")
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
    p.add_argument("--attn-models", nargs="+", default=["mossImage_sft_bf16.safetensors"], help="diffusion_models files (attention mode)")
    p.add_argument("--tag", help="name of the attention backend the server runs (attention mode)")
    p.add_argument("--compare", nargs="+", metavar="TAG", help="compare attention runs, the first tag is the reference")
    p.add_argument("--low-size", type=int, default=512, help="first-stage square side (progressive mode)")
    p.add_argument("--switch", type=int, nargs="+", default=[5, 8, 10, 13], help="steps run at --low-size (progressive mode)")
    p.add_argument("--methods", nargs="+", choices=("renoise", "leftover"), default=["renoise", "leftover"], help="progressive mode")
    p.add_argument("--resume", nargs="+", metavar="K:B", help="progressive mode: K steps at --low-size, then --size from step B <= K (re-noised)")
    p.add_argument("--turbo-files", nargs="+", metavar="NAME=FILE", help="turbo_files mode")
    p.add_argument("--loader", default="LoraLoaderModelOnly", help="turbo_files mode: LoraLoaderModelOnly or LoraLoaderBypassModelOnly")
    p.add_argument("--ref-dir", help="turbo_files mode: a turbo_groups --out")
    p.add_argument("--prompts", nargs="+", choices=list(COMPARE), help="turbo_files mode: COMPARE prompts to render")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = jobs_for(args)
    if args.mode in ("groups", "lora", "bias_merge"):
        run_merges(args, jobs)
        return
    if args.mode == "shift":
        run_shift(args, jobs)
        return
    if args.mode == "slg":
        run_slg(args, jobs)
        return
    if args.mode == "schedulers":
        run_schedulers(args, jobs)
        return
    if args.mode == "attention":
        compare_attention(args, jobs) if args.compare else run_attention(args, jobs)
        return
    if args.mode == "progressive":
        run_progressive(args, jobs)
        return
    if args.mode == "turbo":
        run_turbo(args, jobs)
        return
    if args.mode == "turbo_groups":
        run_turbo_groups(args, jobs)
        return
    if args.mode == "turbo_files":
        run_turbo_files(args, jobs)
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
    configs = {"groups": group_configs, "lora": lora_configs, "bias_merge": bias_merge_configs}[args.mode]()
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for c, unet in configs.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                run_graph(args.server, build_graph(args, unet, unet["slg"], prompt, seed)).save(path)
            if args.mode in ("groups", "lora"):
                feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)

    if args.mode == "bias_merge":
        for name, prompt, seeds in jobs:
            print(contact_sheet(args.out, name, prompt, seeds, list(configs)))
        return
    report = analyze_groups(feats, list(feats["A"]), MERGE_GROUPS if args.mode == "groups" else LORA_GROUPS[:-1])
    if args.mode == "lora":
        report.update(lora=args.lora, lora_model=args.lora_model)
    else:
        report.update(sft=args.sft, base=args.base)
    report.update(steps=args.steps, cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler, slg=False)
    with open(os.path.join(args.out, f"report_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, list(configs), cols=len(seeds), tile=160))
    print(json.dumps(report, indent=2))


def run_shift(args, jobs):
    configs = [(size, shift) for size, shifts in SHIFT_GRID.items() for shift in shifts]
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for size, shift in configs:
        c = f"{size}_s{shift:g}"
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                run_graph(args.server, shift_graph(args, size, shift, prompt, seed)).save(path)
            feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)

    keys = list(feats["1024_s5"])
    mean = lambda vals: round(float(np.mean(vals)), 3)
    rows = {}
    for c in feats:
        ref = f"{c.split('_')[0]}_s5"
        rows[c] = {
            "slope": mean([feats[c][s]["slope"] for s in keys]),
            "slope_vs_1024": mean([feats[c][s]["slope"] - feats["1024_s5"][s]["slope"] for s in keys]),
            "detail_vs_shift5": mean([feats[c][s]["detail"] - feats[ref][s]["detail"] for s in keys]),
            "layout_vs_shift5": round(float(np.mean([distance(feats[c][s], feats[ref][s])["layout"] for s in keys])), 2),
        }
    report = {"samples": len(keys), "sft": args.sft, "steps": args.steps, "cfg": args.cfg, "sampler": args.sampler, "scheduler": args.scheduler,
              "by_size_and_shift": rows}
    with open(os.path.join(args.out, "report_shift.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, [f"{s}_s{sh:g}" for s, sh in configs], cols=len(seeds), tile=256))
    print(json.dumps(report, indent=2))


def run_slg(args, jobs):
    """SFT and Base with each SLG_CONFIGS entry; contact sheets per prompt to count text errors by eye."""
    configs = [(m, name) for m in args.models for name in SLG_CONFIGS]
    total, done, t0 = len(configs) * sum(len(s) for _, _, s in jobs), 0, time.time()
    for m, name in configs:
        c = f"{m}_{name}"
        for pname, prompt, seeds in jobs:
            os.makedirs(os.path.join(args.out, c, pname), exist_ok=True)
            for seed in seeds:
                path = os.path.join(args.out, c, pname, f"{seed}.png")
                if not os.path.exists(path):
                    run_graph(args.server, build_graph(args, getattr(args, m), SLG_CONFIGS[name], prompt, seed)).save(path)
                done += 1
                print(f"[{done}/{total}] {c} {pname} {seed} ({time.time() - t0:.0f}s)", flush=True)
    for pname, prompt, seeds in jobs:
        print(contact_sheet(args.out, pname, prompt, seeds, [f"{m}_{n}" for m, n in configs], cols=len(seeds), tile=256))


def run_schedulers(args, jobs):
    """--sft with SCHEDULER_CONFIGS against a 60-step euler / simple reference (lower distance = closer to the converged image)."""
    configs = {"ref": ("euler", "simple", 60)} | {f"{s}_{sch}_{st}": (s, sch, st) for s, sch, st in SCHEDULER_CONFIGS}
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for c, (sampler, scheduler, steps) in configs.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                g = build_graph(args, args.sft, False, prompt, seed)
                g["ks"]["inputs"].update(sampler_name=sampler, scheduler=scheduler, steps=steps)
                run_graph(args.server, g).save(path)
            feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)
    keys = list(feats["ref"])
    rows = {c: {m: round(float(np.mean([distance(feats[c][s], feats["ref"][s])[m] for s in keys])), 3) for m in METRICS}
            for c in configs if c != "ref"}
    report = {"samples": len(keys), "sft": args.sft, "cfg": args.cfg, "reference": "euler / simple / 60 steps", "distance_to_reference": rows}
    with open(os.path.join(args.out, "report_schedulers.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def attention_configs(args):
    return [(os.path.splitext(m)[0], m, size) for m in args.attn_models for size in ATTENTION_SIZES]


def run_attention(args, jobs):
    """Every --attn-models file at every ATTENTION_SIZES side under the server's current attention; images and timings per --tag."""
    if not args.tag:
        raise SystemExit("--tag is required: name the attention backend the server was started with")
    tag_dir = os.path.join(args.out, args.tag)
    os.makedirs(tag_dir, exist_ok=True)
    argv = json.loads(_request(args.server, "/system_stats"))["system"].get("argv")
    print(f"server argv: {argv}", flush=True)
    with open(os.path.join(tag_dir, "server.json"), "w", encoding="utf-8") as f:
        json.dump({"argv": argv}, f, indent=2)
    # a cached graph would not sample again, so timings would read zero
    _request(args.server, "/free", json.dumps({"unload_models": True, "free_memory": True}).encode())
    timings_path = os.path.join(tag_dir, "timings.json")
    timings = json.load(open(timings_path, encoding="utf-8")) if os.path.exists(timings_path) else {}
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    total, done, t0 = len(attention_configs(args)) * len(samples), 0, time.time()
    for c, unet, size in attention_configs(args):
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(tag_dir, f"{c}_{size}", name), exist_ok=True)
            path = os.path.join(tag_dir, f"{c}_{size}", name, f"{seed}.png")
            if not os.path.exists(path):
                g = build_graph(args, unet, False, prompt, seed)
                g["latent"]["inputs"].update(width=size, height=size)
                info = {}
                run_graph(args.server, g, info).save(path)
                clean = {"unet", "pos", "neg"} <= set(info["cached"])
                timings[f"{c}_{size}/{name}/{seed}"] = {"seconds": info["seconds"], "sampling_only": clean}
                with open(timings_path, "w", encoding="utf-8") as f:
                    json.dump(timings, f, indent=1)
            done += 1
            print(f"[{done}/{total}] {args.tag} {c}_{size} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)


def compare_attention(args, jobs):
    ref, others = args.compare[0], args.compare[1:]
    keys = [f"{name}/{seed}" for name, _, seeds in jobs for seed in seeds]
    timings = {t: json.load(open(os.path.join(args.out, t, "timings.json"), encoding="utf-8")) for t in args.compare}
    report = {"reference": ref, "samples": len(keys), "steps": args.steps, "sampler": args.sampler, "scheduler": args.scheduler, "configs": {}}
    for c, _, size in attention_configs(args):
        cfg = f"{c}_{size}"
        row = {}
        for t in args.compare:
            secs = [v["seconds"] for k, v in timings[t].items() if k.startswith(cfg + "/") and v["sampling_only"]]
            row[t] = {"median_seconds": round(float(np.median(secs)), 2) if secs else None, "timed_images": len(secs)}
        for t in others:
            pixel, feats = [], []
            for k in keys:
                a = Image.open(os.path.join(args.out, ref, cfg, f"{k}.png")).convert("RGB")
                b = Image.open(os.path.join(args.out, t, cfg, f"{k}.png")).convert("RGB")
                pixel.append(float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean()))
                feats.append(distance(features(a), features(b)))
            row[t].update(pixel_diff_mean=round(float(np.mean(pixel)), 2), pixel_diff_max=round(float(np.max(pixel)), 2),
                          **{m: round(float(np.mean([f[m] for f in feats])), 3) for m in METRICS})
        report["configs"][cfg] = row
    with open(os.path.join(args.out, "report_attention.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def progressive_graph(args, prompt, seed, switch, method, low, resume=None):
    """Two KSamplerAdvanced stages of one --steps schedule: steps [0, switch) on a `low` x `low` latent, [resume, steps) at --size.
    resume (default switch) earlier than switch backtracks: the full-size stage re-noises to that earlier step's sigma."""
    resume = switch if resume is None else resume
    g = build_graph(args, args.sft, False, prompt, seed)
    common = {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "noise_seed": seed, "steps": args.steps,
              "cfg": args.cfg, "sampler_name": args.sampler, "scheduler": args.scheduler}
    g["latent"]["inputs"].update(width=low, height=low)
    g["ks"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "latent_image": ["latent", 0], "add_noise": "enable",
                                                            "start_at_step": 0, "end_at_step": switch,
                                                            "return_with_leftover_noise": "disable" if method == "renoise" else "enable"}}
    # LatentUpscaleBy scales the latent itself; LatentUpscale takes pixels and assumes a x8 VAE (MOSS is x16)
    g["up"] = {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["ks", 0], "upscale_method": "bilinear", "scale_by": args.size / low}}
    g["ks2"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "latent_image": ["up", 0],
                                                             "add_noise": "enable" if method == "renoise" else "disable",
                                                             "start_at_step": resume, "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
    g["dec"]["inputs"]["samples"] = ["ks2", 0]
    return g


def run_progressive(args, jobs):
    if args.resume:
        pairs = [tuple(int(v) for v in p.split(":")) for p in args.resume]
        configs = {"ref": None} | {f"renoise_{k}_from_{b}": ("renoise", k, args.low_size, b) for k, b in pairs}
    else:
        configs = {"ref": None, f"renoise_full_{args.switch[len(args.switch) // 2]}": ("renoise", args.switch[len(args.switch) // 2], args.size)}
        configs |= {f"{m}_{k}": (m, k, args.low_size) for m in args.methods for k in args.switch}
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    _request(args.server, "/free", json.dumps({"unload_models": True, "free_memory": True}).encode())
    argv = json.loads(_request(args.server, "/system_stats"))["system"].get("argv")
    print(f"server argv: {argv}", flush=True)
    timings_path = os.path.join(args.out, "timings.json")
    timings = json.load(open(timings_path, encoding="utf-8")) if os.path.exists(timings_path) else {}
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for c, spec in configs.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                if spec is None:
                    g = build_graph(args, args.sft, False, prompt, seed)
                else:
                    g = progressive_graph(args, prompt, seed, spec[1], spec[0], spec[2], *spec[3:])
                info = {}
                img = run_graph(args.server, g, info)
                if img.size != (args.size, args.size):
                    raise RuntimeError(f"{c}: rendered {img.size[0]}x{img.size[1]}, expected {args.size}x{args.size}")
                img.save(path)
                timings[f"{c}/{name}/{seed}"] = {"seconds": info["seconds"], "sampling_only": {"unet", "pos", "neg"} <= set(info["cached"])}
                with open(timings_path, "w", encoding="utf-8") as f:
                    json.dump(timings, f, indent=1)
            feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)
    keys = list(feats["ref"])
    rows = {}
    for c in configs:
        secs = [v["seconds"] for k, v in timings.items() if k.startswith(c + "/") and v["sampling_only"]]
        rows[c] = {"median_seconds": round(float(np.median(secs)), 2) if secs else None,
                   "detail_vs_ref": round(float(np.mean([feats[c][s]["detail"] - feats["ref"][s]["detail"] for s in keys])), 3)}
        if c != "ref":
            rows[c].update({m: round(float(np.mean([distance(feats[c][s], feats["ref"][s])[m] for s in keys])), 3) for m in METRICS})
    report = {"samples": len(keys), "sft": args.sft, "size": args.size, "low_size": args.low_size, "steps": args.steps, "cfg": args.cfg,
              "sampler": args.sampler, "scheduler": args.scheduler, "server_argv": argv, "configs": rows}
    with open(os.path.join(args.out, "report_progressive.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, list(configs), cols=len(seeds), tile=256))
    print(json.dumps(report, indent=2))


def turbo_graph(args, prompt, seed, lora, loader, steps):
    """--sft (with `lora` through `loader`, if given) at euler / simple / CFG 1 and `steps`."""
    g = build_graph(args, args.sft, False, prompt, seed)
    if lora:
        g["lora"] = {"class_type": loader, "inputs": {"model": ["unet", 0], "lora_name": lora, "strength_model": 1.0}}
        g["ks"]["inputs"]["model"] = ["lora", 0]
    g["ks"]["inputs"].update(steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple")
    return g


def run_turbo(args, jobs):
    configs = {"ref": None}
    for name, (lora, steps, fused) in TURBO_LORAS.items():
        configs[f"{name}_weights"] = (lora, "LoraLoaderModelOnly", steps)
        if fused:
            configs[f"{name}_bypass"] = (fused, "LoraLoaderBypassModelOnly", steps)
        configs[f"sft_alone_{steps}"] = (None, None, steps)
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    _request(args.server, "/free", json.dumps({"unload_models": True, "free_memory": True}).encode())
    argv = json.loads(_request(args.server, "/system_stats"))["system"].get("argv")
    print(f"server argv: {argv}", flush=True)
    timings_path = os.path.join(args.out, "timings.json")
    timings = json.load(open(timings_path, encoding="utf-8")) if os.path.exists(timings_path) else {}
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for c, spec in configs.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                g = build_graph(args, args.sft, False, prompt, seed) if spec is None else turbo_graph(args, prompt, seed, *spec)
                info = {}
                run_graph(args.server, g, info).save(path)
                cached = {"unet", "pos", "neg"} | ({"lora"} if spec and spec[0] else set())
                timings[f"{c}/{name}/{seed}"] = {"seconds": info["seconds"], "sampling_only": cached <= set(info["cached"])}
                with open(timings_path, "w", encoding="utf-8") as f:
                    json.dump(timings, f, indent=1)
            feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)
    keys = list(feats["ref"])
    mean_dist = lambda a, b: {m: round(float(np.mean([distance(feats[a][s], feats[b][s])[m] for s in keys])), 3) for m in METRICS}
    rows = {}
    for c in configs:
        secs = [v["seconds"] for k, v in timings.items() if k.startswith(c + "/") and v["sampling_only"]]
        rows[c] = {"median_seconds": round(float(np.median(secs)), 2) if secs else None, "timed_images": len(secs),
                   "detail_vs_ref": round(float(np.mean([feats[c][s]["detail"] - feats["ref"][s]["detail"] for s in keys])), 3)}
        if c != "ref":
            rows[c]["vs_ref"] = mean_dist(c, "ref")
    for name in TURBO_LORAS:
        if f"{name}_bypass" in configs:
            rows[f"{name}_weights"]["vs_bypass"] = mean_dist(f"{name}_weights", f"{name}_bypass")
    report = {"samples": len(keys), "sft": args.sft, "reference": f"{args.sampler} / {args.scheduler} / {args.steps} steps / CFG {args.cfg}",
              "server_argv": argv, "configs": rows}
    with open(os.path.join(args.out, "report_turbo.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, list(configs), cols=len(seeds), tile=256))
    print(json.dumps(report, indent=2))


def run_turbo_groups(args, jobs):
    configs = {}
    for name, (lora, steps, _) in TURBO_LORAS.items():
        for gname, kept in TURBO_GROUP_SETS.items():
            configs[f"{name}_{gname}"] = (lora, steps, {g: 1.0 if g in kept else 0.0 for g in LORA_GROUPS})
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    argv = json.loads(_request(args.server, "/system_stats"))["system"].get("argv")
    print(f"server argv: {argv}", flush=True)
    feats, total, done, t0 = {}, len(configs) * len(samples), 0, time.time()
    for c, (lora, steps, weights) in configs.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                g = turbo_graph(args, prompt, seed, None, None, steps)
                g["lora"] = {"class_type": "HaDeSMossLoraBlockWeight", "inputs": {"model": ["unet", 0], "lora_name": lora, "strength_model": 1.0, **weights}}
                g["ks"]["inputs"]["model"] = ["lora", 0]
                run_graph(args.server, g).save(path)
            feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)
    rows = {}
    for name in TURBO_LORAS:
        full = f"{name}_all"
        keys = list(feats[full])
        for gname in TURBO_GROUP_SETS:
            c = f"{name}_{gname}"
            rows[c] = {"detail_vs_full": round(float(np.mean([feats[c][s]["detail"] - feats[full][s]["detail"] for s in keys])), 3)}
            if c != full:
                rows[c].update({m: round(float(np.mean([distance(feats[c][s], feats[full][s])[m] for s in keys])), 3) for m in METRICS})
    report = {"samples": len(samples), "sft": args.sft, "server_argv": argv, "reference": "the same LoRA with every group", "configs": rows}
    with open(os.path.join(args.out, "report_turbo_groups.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, list(configs), cols=len(seeds), tile=200))
    print(json.dumps(report, indent=2))


def run_turbo_files(args, jobs):
    files = dict(item.split("=", 1) for item in args.turbo_files)
    key_of = {name: next(k for k in TURBO_LORAS if name.startswith(k)) for name in files}
    samples = [(name, prompt, seed) for name, prompt, seeds in jobs for seed in seeds]
    argv = json.loads(_request(args.server, "/system_stats"))["system"].get("argv")
    print(f"server argv: {argv}", flush=True)
    timings_path = os.path.join(args.out, "timings.json")
    timings = json.load(open(timings_path, encoding="utf-8")) if os.path.exists(timings_path) else {}
    feats, total, done, t0 = {}, len(files) * len(samples), 0, time.time()
    for c, lora in files.items():
        feats[c] = {}
        for name, prompt, seed in samples:
            os.makedirs(os.path.join(args.out, c, name), exist_ok=True)
            path = os.path.join(args.out, c, name, f"{seed}.png")
            if not os.path.exists(path):
                info = {}
                g = turbo_graph(args, prompt, seed, lora, args.loader, TURBO_LORAS[key_of[c]][1])
                g["latent"]["inputs"].update(width=args.size, height=args.size)
                run_graph(args.server, g, info).save(path)
                timings[f"{c}/{name}/{seed}"] = {"seconds": info["seconds"], "sampling_only": {"unet", "pos", "neg", "lora"} <= set(info["cached"])}
                with open(timings_path, "w", encoding="utf-8") as f:
                    json.dump(timings, f, indent=1)
            feats[c][f"{name}_{seed}"] = features(Image.open(path).convert("RGB"))
            done += 1
            print(f"[{done}/{total}] {c} {name} {seed} ({time.time() - t0:.0f}s)", flush=True)
    rows = {}
    for c in files:
        ref_dir = os.path.join(args.ref_dir, f"{key_of[c]}_all") if args.ref_dir else os.path.join(args.out, f"{key_of[c]}_orig")
        ref = {f"{n}_{s}": features(Image.open(os.path.join(ref_dir, n, f"{s}.png")).convert("RGB")) for n, _, s in samples}
        secs = [v["seconds"] for k, v in timings.items() if k.startswith(c + "/") and v["sampling_only"]]
        rows[c] = {"median_seconds": round(float(np.median(secs)), 2) if secs else None,
                   "detail_vs_full": round(float(np.mean([feats[c][s]["detail"] - ref[s]["detail"] for s in ref])), 3),
                   **{m: round(float(np.mean([distance(feats[c][s], ref[s])[m] for s in ref])), 3) for m in METRICS}}
    report = {"samples": len(samples), "sft": args.sft, "loader": args.loader, "server_argv": argv, "files": files, "configs": rows}
    with open(os.path.join(args.out, "report_turbo_files.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for name, prompt, seeds in jobs:
        print(contact_sheet(args.out, name, prompt, seeds, list(files), cols=len(seeds), tile=200))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
