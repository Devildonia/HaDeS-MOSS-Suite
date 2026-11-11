"""Compare baked v0.9.4 with SFT/Base plus the v0.9 Turbo LoRA at 8 steps / CFG 1.

Reuses the eight existing baked-model images; queues only the 16 missing renders.
All evidence is written to --out. The source archive is read-only.
"""

import argparse
import csv
import io
import json
import random
import struct
import time
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from moss_baked_turbo_eval import SIZE, graph_for, metrics, request, save_json, sha


REFERENCE = Path("I:/HaDeS_MOSS/baked_turbo_v094_2026-11-11")
LORA = "MOSS\\HaDeS-MOSS-BeTa-Turbo-8steps-v0.9.safetensors"
LORA_FILE = Path("H:/Comfy_UI_V61/ComfyUI/models/loras") / LORA
MODELS = {
    "sft_v09": "mossImage_sft_bf16.safetensors",
    "base_v09": "mossImage_base_bf16.safetensors",
}
CONFIG = ("s08_cfg1", 8, 1.0)


def comparison_graph(case, seed, model):
    graph = graph_for(case, seed, CONFIG[1], CONFIG[2])
    graph["model"]["inputs"]["unet_name"] = MODELS[model]
    graph["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["model", 0], "lora_name": LORA, "strength_model": 1.0}}
    graph["sampler"]["inputs"]["model"] = ["lora", 0]
    return graph


def reference_record(case, seed):
    stem = f"{case['label']}_{seed}_{CONFIG[0]}"
    record = json.loads((REFERENCE / f"{stem}.json").read_text(encoding="utf-8"))
    image = REFERENCE / f"{stem}.png"
    if record["state"] != "success" or record["image_sha256"] != sha(image.read_bytes()):
        raise RuntimeError(f"Reference changed or incomplete: {stem}")
    if record["source_sha256"] != case["source_sha256"]:
        raise RuntimeError(f"Source changed: {stem}")
    sampler = record["graph"]["sampler"]["inputs"]
    if (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"],
            sampler["flow_shift"], sampler["denoise"], sampler["slg"]) != (8, 1.0, "euler", "simple", 5.0, 1.0, "off"):
        raise RuntimeError(f"Unexpected reference settings: {stem}")
    return record, image


def run_one(server, out, case, seed, model):
    stem = f"{case['label']}_{seed}_{model}"
    image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
    graph = comparison_graph(case, seed, model)
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
            raise RuntimeError(f"Prior result changed: {stem}")
        print("reuse", stem, flush=True)
        return old
    queue = request(server, "/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI queue is occupied; no existing work was interrupted")
    prompt_id = request(server, "/prompt", {"prompt": graph, "client_id": "hades_moss_v094_vs_v09_eval"})["prompt_id"]
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
              "image_sha256": sha(raw), "output": output, "seconds": seconds, "metrics": values}
    image_path.write_bytes(raw)
    save_json(record_path, record)
    print("saved", stem, f"{seconds}s", flush=True)
    return record


def write_summaries(out, cases):
    rows = []
    key = {}
    blind_dir = out / "blind"
    for case in cases:
        for seed in case["seeds"]:
            reference, reference_image = reference_record(case, seed)
            records = {"baked_v094": (reference, reference_image)}
            for model in MODELS:
                stem = f"{case['label']}_{seed}_{model}"
                record_path, image_path = out / f"{stem}.json", out / f"{stem}.png"
                if record_path.exists() and image_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if record.get("state") == "success" and record["image_sha256"] == sha(image_path.read_bytes()):
                        records[model] = (record, image_path)
            for name, (record, image_path) in records.items():
                rows.append({"scene": case["label"], "seed": seed, "variant": name,
                             "image": str(image_path), "seconds": record["seconds"], **record["metrics"]})
            if len(records) != 3:
                continue
            blind_dir.mkdir(exist_ok=True)
            variants = list(records.items())
            random.Random(int(sha(f"v094_vs_v09:{case['label']}:{seed}".encode()), 16)).shuffle(variants)
            key[f"{case['label']}_{seed}"] = {chr(65 + i): name for i, (name, _) in enumerate(variants)}
            sheet = Image.new("RGB", (3 * 576, 795), "#17191b")
            draw = ImageDraw.Draw(sheet)
            for index, (_, (_, path)) in enumerate(variants):
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((576, 768), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (index * 576, 27))
                draw.text((index * 576 + 12, 6), chr(65 + index), fill="white")
            sheet.save(blind_dir / f"{case['label']}_{seed}.jpg", quality=94)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(key) == sum(len(case["seeds"]) for case in cases):
        save_json(out / "blind_key.json", key)
        ratings = out / "blind_ratings.csv"
        if not ratings.exists():
            with ratings.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["scene", "seed", "letter", "preferred", "prompt_fidelity", "artifacts", "notes"])
                for sheet, letters in key.items():
                    scene, seed = sheet.rsplit("_", 1)
                    for letter in letters:
                        writer.writerow([scene, seed, letter, "", "", "", ""])
    return len(rows), len(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--out", type=Path, default=Path("I:/HaDeS_MOSS/v094_vs_sft_base_v09_2026-11-11"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs; rerun to resume")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    source = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    if (source["model"], source["size"], source["clip"], source["vae"]) != (
            "HaDeS_MOSS_v0.9.4_Turbo.safetensors", list(SIZE),
            "qwen3.5_9b_int8_convrot.safetensors", "flux2-vae.safetensors"):
        raise RuntimeError("Unexpected source archive")
    cases = source["cases"]
    with LORA_FILE.open("rb") as file:
        header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
    for case in cases:
        for seed in case["seeds"]:
            reference_record(case, seed)
    manifest = {"reference_archive": str(REFERENCE), "reference_config": list(CONFIG),
                "models": MODELS, "lora": LORA, "lora_strength": 1.0,
                "lora_sha256": sha(LORA_FILE.read_bytes()), "lora_metadata": header.get("__metadata__", {}),
                "clip": source["clip"], "vae": source["vae"], "size": source["size"],
                "sampler": source["sampler"], "scheduler": source["scheduler"], "shift": source["shift"],
                "denoise": source["denoise"], "slg": source["slg"], "cases": cases}
    jobs = [(case, seed, model) for case in cases for seed in case["seeds"] for model in MODELS]
    print(f"{len(jobs)} new jobs; {len(cases) * 2} existing references; output {args.out}", flush=True)
    if args.plan:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Existing manifest differs; use another output directory")
    save_json(path, manifest)
    try:
        for case, seed, model in jobs[:args.limit]:
            run_one(args.server, args.out, case, seed, model)
    finally:
        count, sheets = write_summaries(args.out, cases)
        print(f"{count}/24 images, {sheets}/8 blind sheets", flush=True)


if __name__ == "__main__":
    main()
