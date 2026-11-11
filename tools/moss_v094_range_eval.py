"""HaDeS MOSS v0.9.4 Turbo outside its 8-step / CFG 1 reference: a step sweep, a CFG sweep and the café bias count.

    python tools/moss_v094_range_eval.py --out <folder> [--server http://127.0.0.1:8188]

- steps: the eight held-out scenes (tools/moss_v094_sft_heldout.py, their seeds) at 6, 8, 12, 20, 30 and 50 steps, CFG 1.
- cfg: the first two person and two non-person scenes at 20 steps, CFG 1.0, 1.5, 2.0, 2.5 and 3.0.
- bias: "photograph of a woman sitting in a café", seeds 0-15, 1024 x 1024, 8 steps, CFG 1, the protocol of the 2026-11-09 counts
  (Base 25 steps and the BeTa Turbo v0.9 LoRA on Base), for the people to be counted by eye.
All runs: euler / simple, shift 5, denoise 1, SLG off, the held-out negative, 1152 x 1536 except bias. Waits for an empty queue
before each job and reuses finished images on a rerun. Writes per-image JSON (graph, timing, metrics), metrics.csv, one contact
sheet per scene and a bias grid.
"""
import argparse
import csv
import io
import json
import time
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from moss_baked_turbo_eval import CLIP, MODEL, VAE, request, save_json, sha
from moss_v094_sft_heldout import CASES, NEGATIVE

STEPS = (6, 8, 12, 20, 30, 50)
CFGS = (1.0, 1.5, 2.0, 2.5, 3.0)
CFG_CASES = ("watchmaker", "trumpeter", "watch_macro", "rainy_alley")
BIAS_PROMPT = "photograph of a woman sitting in a café"
BIAS_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
BIAS_SEEDS = range(16)


def graph(positive, negative, seed, steps, cfg, size):
    return {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "moss_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "sampler": {"class_type": "KSampler", "inputs": {
            "model": ["model", 0], "positive": ["positive", 0], "negative": ["negative", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "out": {"class_type": "PreviewImage", "inputs": {"images": ["decode", 0]}},
    }


def metrics(raw):
    """Appearance numbers on 8-bit RGB: luma, HSV saturation, near-clipped pixels, Laplacian sharpness and the share of spectral
    energy above a quarter of Nyquist (fine detail, or noise / burn when it jumps)."""
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    lap = luma[1:-1, 1:-1] * 4 - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]
    power = np.abs(np.fft.rfft2(luma - luma.mean())) ** 2
    fy = np.abs(np.fft.fftfreq(luma.shape[0]))[:, None]
    fx = np.fft.rfftfreq(luma.shape[1])[None, :]
    high = np.maximum(fy, fx) > 0.125
    return {"mean_luma_255": float(luma.mean()), "mean_hsv_saturation_255": float(sat.mean() * 255),
            "near_clip_pct": float(100 * np.mean((rgb <= 1).any(axis=2) | (rgb >= 254).any(axis=2))),
            "laplacian_std": float(lap.std()), "high_freq_energy_pct": float(100 * power[high].sum() / power.sum())}


def run(server, out, stem, g):
    image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
    fingerprint = sha(json.dumps(g, sort_keys=True).encode())
    if image_path.exists() and record_path.exists():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old["fingerprint"] == fingerprint and old["image_sha256"] == sha(image_path.read_bytes()):
            return old
        raise RuntimeError(f"Prior result changed: {stem}")
    while True:  # wait for an empty queue: never interrupt the user's own jobs
        q = request(server, "/queue")
        if not q.get("queue_running") and not q.get("queue_pending"):
            break
        time.sleep(5)
    prompt_id = request(server, "/prompt", {"prompt": g, "client_id": "hades_moss_v094_range"})["prompt_id"]
    started = time.monotonic()
    while True:
        history = request(server, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            break
        if time.monotonic() - started > 900:
            raise TimeoutError(f"ComfyUI job {prompt_id} timed out; it was not cancelled")
        time.sleep(0.5)
    if entry.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"ComfyUI job {prompt_id} failed: {entry.get('status')}")
    raw = request(server, "/view?" + urllib.parse.urlencode(entry["outputs"]["out"]["images"][0]))
    stamps = {k: d.get("timestamp") for k, d in entry["status"].get("messages", []) if "timestamp" in d}
    seconds = (stamps["execution_success"] - stamps["execution_start"]) / 1000 if {"execution_start", "execution_success"} <= set(stamps) else None
    record = {"fingerprint": fingerprint, "graph": g, "prompt_id": prompt_id, "image_sha256": sha(raw), "seconds": seconds, "metrics": metrics(raw)}
    image_path.write_bytes(raw)
    save_json(record_path, record)
    print("saved", stem, f"{seconds}s", flush=True)
    return record


def sheet(paths, labels, target, cell=(384, 512)):
    img = Image.new("RGB", (cell[0] * len(paths), cell[1] + 24), "#17191b")
    draw = ImageDraw.Draw(img)
    for i, (p, label) in enumerate(zip(paths, labels)):
        with Image.open(p) as im:
            img.paste(im.convert("RGB").resize(cell, Image.Resampling.LANCZOS), (i * cell[0], 24))
        draw.text((i * cell[0] + 8, 5), label, fill="white")
    img.save(target, quality=92)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    size = (1152, 1536)
    for case in CASES:
        stems = []
        for steps in STEPS:
            stem = f"steps_{case['label']}_{steps:02d}"
            rec = run(args.server, args.out, stem, graph(case["positive"], NEGATIVE, case["seed"], steps, 1.0, size))
            rows.append({"part": "steps", "scene": case["label"], "steps": steps, "cfg": 1.0, "seconds": rec["seconds"], **rec["metrics"]})
            stems.append(stem)
        sheet([args.out / f"{s}.png" for s in stems], [f"{n} steps" for n in STEPS], args.out / f"sheet_steps_{case['label']}.jpg")
    for case in (c for c in CASES if c["label"] in CFG_CASES):
        stems = []
        for cfg in CFGS:
            stem = f"cfg_{case['label']}_{cfg:.1f}"
            rec = run(args.server, args.out, stem, graph(case["positive"], NEGATIVE, case["seed"], 20, cfg, size))
            rows.append({"part": "cfg", "scene": case["label"], "steps": 20, "cfg": cfg, "seconds": rec["seconds"], **rec["metrics"]})
            stems.append(stem)
        sheet([args.out / f"{s}.png" for s in stems], [f"20 steps CFG {c:g}" for c in CFGS], args.out / f"sheet_cfg_{case['label']}.jpg")
    bias = []
    for seed in BIAS_SEEDS:
        stem = f"bias_cafe_{seed:02d}"
        rec = run(args.server, args.out, stem, graph(BIAS_PROMPT, BIAS_NEGATIVE, seed, 8, 1.0, (1024, 1024)))
        rows.append({"part": "bias", "scene": "cafe", "steps": 8, "cfg": 1.0, "seconds": rec["seconds"], **rec["metrics"]})
        bias.append(args.out / f"{stem}.png")
    grid = Image.new("RGB", (4 * 256, 4 * 256))
    for i, path in enumerate(bias):
        with Image.open(path) as im:
            grid.paste(im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), ((i % 4) * 256, (i // 4) * 256))
    grid.save(args.out / "bias_cafe_grid.jpg", quality=92)
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_json(args.out / "manifest.json", {"model": MODEL, "clip": CLIP, "vae": VAE, "steps": STEPS, "cfgs": CFGS, "cfg_cases": CFG_CASES,
                                           "bias_prompt": BIAS_PROMPT, "bias_seeds": list(BIAS_SEEDS), "negative": NEGATIVE})
    print("done", len(rows), "images", flush=True)


if __name__ == "__main__":
    main()
