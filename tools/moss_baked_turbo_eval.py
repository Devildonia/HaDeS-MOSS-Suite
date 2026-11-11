"""Paired CFG / step screen for the baked-in HaDeS MOSS v0.9.4 Turbo model.

Reads four existing PNGs only for their resolved prompts and one seed each.
Writes generated images and exact ComfyUI requests under --out; never rewrites
the source PNGs or model. A second seed (1200) checks each scene independently.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


MODEL = "HaDeS_MOSS_v0.9.4_Turbo.safetensors"
CLIP = "qwen3.5_9b_int8_convrot.safetensors"
VAE = "flux2-vae.safetensors"
SOURCE_DIR = Path("H:/Comfy_UI_V61/ComfyUI/output/HaDeS/HaDeS_MOSS")
SOURCES = {
    "airlock": "HaDeS_MOSS 902666576933789_00001_.png",
    "trawler_portrait": "HaDeS_MOSS 920407339656580_00001_.png",
    "ceramic_texture": "HaDeS_MOSS 902658975663233_00001_.png",
    "comic_text": "HaDeS_MOSS 292476657386784_00001_.png",
}
CONFIGS = (
    ("s08_cfg1", 8, 1.0),
    ("s08_cfg1p2", 8, 1.2),
    ("s08_cfg1p5", 8, 1.5),
    ("s08_cfg2", 8, 2.0),
    ("s12_cfg1", 12, 1.0),
    ("s16_cfg1", 16, 1.0),
    ("s12_cfg1p2", 12, 1.2),
    ("s12_cfg1p5", 12, 1.5),
)
SIZE = (1152, 1536)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def request(server, path, payload=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(server + path, body, {"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = response.read()
    return json.loads(raw) if path != "/view" and not path.startswith("/view?") else raw


def source_case(label, filename):
    path = SOURCE_DIR / filename
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != SIZE:
            raise ValueError(f"{path}: source must be {SIZE}")
        parameters = image.info["parameters"]
        graph = json.loads(image.info["prompt"])
    loaders = [n["inputs"] for n in graph.values() if n["class_type"] == "HaDeSMossLoader"]
    samplers = [n["inputs"] for n in graph.values() if n["class_type"] == "HaDeSMossKSampler"]
    if len(loaders) != 1 or len(samplers) != 1:
        raise ValueError(f"{path}: expected one Suite loader and sampler")
    loader, sampler = loaders[0], samplers[0]
    if (loader["unet_name"], loader["clip_name"], loader["vae_name"]) != (MODEL, CLIP, VAE):
        raise ValueError(f"{path}: source uses another model, text encoder or VAE")
    if (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"],
            sampler["flow_shift"], sampler["denoise"], sampler["slg"]) != (8, 1.2, "euler", "simple", 5.0, 1.0, "off"):
        raise ValueError(f"{path}: unexpected source sampling settings")
    positive, tail = parameters.rsplit("\nNegative prompt: ", 1)
    negative = tail.split("\nSteps: ", 1)[0]
    seed = int(filename.rsplit(" ", 1)[1].split("_", 1)[0])
    return {"label": label, "source": str(path), "source_sha256": sha(raw),
            "positive": positive, "negative": negative, "seeds": [seed, 1200]}


def graph_for(case, seed, steps, cfg):
    return {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "moss_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": case["positive"], "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": case["negative"], "clip": ["clip", 0]}},
        "latent": {"class_type": "HaDeSMossLatent", "inputs": {
            "aspect_ratio": "custom", "megapixels": 1.0, "custom_width": SIZE[0], "custom_height": SIZE[1],
            "batch_size": 1, "seed": seed}},
        "sampler": {"class_type": "HaDeSMossKSampler", "inputs": {
            "model": ["model", 0], "positive": ["positive", 0], "negative": ["negative", 0],
            "latent_image": ["latent", 0], "vae": ["vae", 0], "seed": ["latent", 4],
            "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0, "flow_shift": 5.0, "slg": "off"}},
        "out": {"class_type": "PreviewImage", "inputs": {"images": ["sampler", 1]}},
    }


def metrics(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != SIZE:
            raise ValueError(f"ComfyUI returned {image.size}, expected {SIZE}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return {"mean_luma_255": float(luma.mean()), "luma_std_255": float(luma.std()),
            "mean_hsv_saturation_255": float(saturation.mean() * 255),
            "saturation_p95_255": float(np.percentile(saturation, 95) * 255),
            "dark_luma_pct": float(100 * np.mean(luma <= 16)),
            "bright_luma_pct": float(100 * np.mean(luma >= 240)),
            "any_channel_near_clip_pct": float(100 * np.mean((rgb <= 1).any(axis=2) | (rgb >= 254).any(axis=2)))}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(server, out, case, seed, config):
    name, steps, cfg = config
    stem = f"{case['label']}_{seed}_{name}"
    image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
    graph = graph_for(case, seed, steps, cfg)
    fingerprint = sha(json.dumps({"graph": graph, "source_sha256": case["source_sha256"]}, sort_keys=True).encode())
    if record_path.exists() and not image_path.exists():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("state") == "failed":
            record_path.rename(out / f"{stem}.failed.json")
    if image_path.exists() or record_path.exists():
        if not image_path.exists() or not record_path.exists():
            raise RuntimeError(f"Incomplete prior result: {stem}")
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint or old["image_sha256"] != sha(image_path.read_bytes()):
            raise RuntimeError(f"Prior result changed: {stem}; use another output directory")
        print("reuse", stem, flush=True)
        return old
    queue = request(server, "/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI queue is occupied; no existing work was interrupted")
    prompt_id = request(server, "/prompt", {"prompt": graph, "client_id": "hades_moss_baked_turbo_eval"})["prompt_id"]
    print("submitted", stem, prompt_id, flush=True)
    started = time.monotonic()
    while True:
        history = request(server, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            break
        if time.monotonic() - started > 600:
            raise TimeoutError(f"ComfyUI job {prompt_id} timed out; it was not cancelled")
        time.sleep(0.5)
    if entry.get("status", {}).get("status_str") != "success":
        save_json(record_path, {"fingerprint": fingerprint, "graph": graph, "prompt_id": prompt_id,
                                "status": entry.get("status"), "state": "failed"})
        raise RuntimeError(f"ComfyUI job {prompt_id} failed: {entry.get('status')}")
    output = entry["outputs"]["out"]["images"][0]
    raw = request(server, "/view?" + urllib.parse.urlencode(output))
    values = metrics(raw)
    stamps = {kind: data.get("timestamp") for kind, data in entry["status"].get("messages", []) if "timestamp" in data}
    seconds = ((stamps["execution_success"] - stamps["execution_start"]) / 1000
               if "execution_start" in stamps and "execution_success" in stamps else None)
    record = {"fingerprint": fingerprint, "state": "success", "source": case["source"],
              "source_sha256": case["source_sha256"], "graph": graph, "prompt_id": prompt_id,
              "image_sha256": sha(raw), "output": output, "seconds": seconds,
              "cached_nodes": next((data.get("nodes", []) for kind, data in entry["status"].get("messages", [])
                                    if kind == "execution_cached"), []), "metrics": values}
    image_path.write_bytes(raw)
    save_json(record_path, record)
    print("saved", stem, f"{seconds}s", flush=True)
    return record


def summaries(out, cases):
    rows = []
    paired = []
    blind_key = {}
    blind_dir = out / "blind"
    for case in cases:
        for seed in case["seeds"]:
            for name, steps, cfg in CONFIGS:
                path = out / f"{case['label']}_{seed}_{name}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("state") == "success":
                    rows.append({"scene": case["label"], "seed": seed, "config": name,
                                 "steps": steps, "cfg": cfg, "seconds": record["seconds"], **record["metrics"]})
    if rows:
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for case in cases:
        for seed in case["seeds"]:
            files = [(name, out / f"{case['label']}_{seed}_{name}.png") for name, _, _ in CONFIGS]
            if not all(path.exists() for _, path in files):
                continue
            with Image.open(files[0][1]) as image:
                baseline = np.asarray(image.convert("RGB"), dtype=np.int16)
            for name, path in files[1:]:
                with Image.open(path) as image:
                    candidate = np.asarray(image.convert("RGB"), dtype=np.int16)
                difference = np.abs(candidate - baseline)
                paired.append({"scene": case["label"], "seed": seed, "config": name,
                               "mae_255_vs_8_cfg1": float(difference.mean()),
                               "changed_percent_vs_8_cfg1": float(100 * np.mean(difference.max(axis=2) > 2))})
            cols, cell_width, cell_height, preview_height = 4, 288, 407, 384
            sheet = Image.new("RGB", (cols * cell_width, 2 * cell_height), "#17191b")
            draw = ImageDraw.Draw(sheet)
            for index, (name, path) in enumerate(files):
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((cell_width, preview_height), Image.Resampling.LANCZOS)
                x, y = (index % cols) * cell_width, (index // cols) * cell_height
                sheet.paste(thumb, (x, y + 23))
                draw.text((x + 8, y + 5), name, fill="white")
            sheet.save(out / f"sheet_{case['label']}_{seed}.jpg", quality=92)
            blind_dir.mkdir(exist_ok=True)
            shuffled = list(files)
            random.Random(int(sha(f"{case['label']}:{seed}".encode()), 16)).shuffle(shuffled)
            blind_key[f"{case['label']}_{seed}"] = {chr(65 + i): name for i, (name, _) in enumerate(shuffled)}
            blind = Image.new("RGB", (4 * 576, 2 * 795), "#17191b")
            blind_draw = ImageDraw.Draw(blind)
            for index, (_, path) in enumerate(shuffled):
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((576, 768), Image.Resampling.LANCZOS)
                x, y = (index % 4) * 576, (index // 4) * 795
                blind.paste(thumb, (x, y + 27))
                blind_draw.text((x + 12, y + 6), chr(65 + index), fill="white")
            blind.save(blind_dir / f"{case['label']}_{seed}.jpg", quality=94)
    if paired:
        with (out / "paired_differences.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    if len(blind_key) == sum(len(case["seeds"]) for case in cases):
        save_json(out / "blind_key.json", blind_key)
        ratings = out / "blind_ratings.csv"
        if not ratings.exists():
            with ratings.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["scene", "seed", "letter", "preference_rank_1_best", "prompt_errors", "artifacts", "notes"])
                writer.writeheader()
                for key in blind_key:
                    scene, seed = key.rsplit("_", 1)
                    for letter in blind_key[key]:
                        writer.writerow({"scene": scene, "seed": seed, "letter": letter})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--out", type=Path, default=Path("I:/HaDeS_MOSS/baked_turbo_v094_2026-11-11"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs; rerun to resume")
    parser.add_argument("--plan", action="store_true", help="Print the frozen test plan without queueing work")
    args = parser.parse_args()
    cases = [source_case(label, filename) for label, filename in SOURCES.items()]
    spec = {"model": MODEL, "clip": CLIP, "vae": VAE, "size": list(SIZE), "sampler": "euler",
            "scheduler": "simple", "shift": 5.0, "denoise": 1.0, "slg": "off", "turbo_input": None,
            "cases": cases, "configs": [list(config) for config in CONFIGS]}
    jobs = [(case, seed, config) for case in cases for seed in case["seeds"] for config in CONFIGS]
    print(f"{len(jobs)} jobs, {len(cases)} scenes, {SIZE[0]}x{SIZE[1]}, output {args.out}", flush=True)
    if args.plan:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        old_configs = previous.pop("configs")
        current = dict(spec)
        current_configs = current.pop("configs")
        if previous != current or old_configs != current_configs[:len(old_configs)]:
            raise RuntimeError("Existing manifest differs; use another output directory")
    save_json(manifest, spec)
    try:
        for case, seed, config in jobs[:args.limit]:
            run_one(args.server, args.out, case, seed, config)
    finally:
        rows = summaries(args.out, cases)
        print(f"{len(rows)}/{len(jobs)} images recorded", flush=True)


if __name__ == "__main__":
    main()
