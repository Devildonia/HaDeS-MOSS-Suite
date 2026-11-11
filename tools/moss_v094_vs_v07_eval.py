"""Paired practical comparison: baked v0.9.4 versus SFT with released Turbo v0.7.

Reuses eight baked images and fixed prompts from moss_v094_sft_heldout.py.
Queues only the eight missing SFT + v0.7 renders.
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

from PIL import Image, ImageDraw

from moss_baked_turbo_eval import MODEL, SIZE, graph_for, metrics, request, save_json, sha
from moss_v094_sft_heldout import CASES, CONFIG, NEGATIVE


REFERENCE = Path("I:/HaDeS_MOSS/v094_vs_sft_v09_heldout_2026-11-11")
SFT_MODEL = "mossImage_sft_bf16.safetensors"
LORA = "MOSS\\HaDeS-MOSS-Turbo-8steps-v0.7.safetensors"
LORA_FILE = Path("H:/Comfy_UI_V61/ComfyUI/models/loras") / LORA


def reference_record(case):
    stem = f"{case['label']}_{case['seed']}_baked_v094"
    record = json.loads((REFERENCE / f"{stem}.json").read_text(encoding="utf-8"))
    image_path = REFERENCE / f"{stem}.png"
    if record.get("state") != "success" or record["image_sha256"] != sha(image_path.read_bytes()):
        raise RuntimeError(f"Reference changed or incomplete: {stem}")
    return record, image_path


def comparison_graph(case):
    graph = graph_for({"positive": case["positive"], "negative": NEGATIVE}, case["seed"], *CONFIG)
    graph["model"]["inputs"]["unet_name"] = SFT_MODEL
    graph["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["model", 0], "lora_name": LORA, "strength_model": 1.0}}
    graph["sampler"]["inputs"]["model"] = ["lora", 0]
    return graph


def run_one(server, out, case):
    stem = f"{case['label']}_{case['seed']}_sft_v07"
    image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
    graph = comparison_graph(case)
    fingerprint = sha(json.dumps(graph, sort_keys=True).encode())
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
    prompt_id = request(server, "/prompt", {"prompt": graph, "client_id": "hades_moss_v094_vs_v07_eval"})["prompt_id"]
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
    record = {"fingerprint": fingerprint, "state": "success", "graph": graph, "prompt_id": prompt_id,
              "image_sha256": sha(raw), "output": output, "seconds": seconds, "metrics": values}
    image_path.write_bytes(raw)
    save_json(record_path, record)
    print("saved", stem, f"{seconds}s", flush=True)
    return record


def write_summaries(out):
    rows = []
    key = {}
    blind = out / "blind"
    for case in CASES:
        reference, reference_image = reference_record(case)
        records = {"baked_v094": (reference, reference_image)}
        stem = f"{case['label']}_{case['seed']}_sft_v07"
        image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
        if record_path.exists() and image_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("state") == "success" and record["image_sha256"] == sha(image_path.read_bytes()):
                records["sft_v07"] = (record, image_path)
        for variant, (record, path) in records.items():
            rows.append({"scene": case["label"], "group": case["group"], "seed": case["seed"],
                         "variant": variant, "image": str(path), "seconds": record["seconds"],
                         **record["metrics"]})
        if len(records) != 2:
            continue
        blind.mkdir(exist_ok=True)
        variants = list(records.items())
        random.Random(int(sha(f"v094_vs_v07:{case['label']}:{case['seed']}".encode()), 16)).shuffle(variants)
        key[case["label"]] = {chr(65 + i): name for i, (name, _) in enumerate(variants)}
        sheet = Image.new("RGB", (2 * 768, 1052), "#17191b")
        draw = ImageDraw.Draw(sheet)
        for index, (_, (_, path)) in enumerate(variants):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            letter = chr(65 + index)
            rgb.save(blind / f"{case['label']}_{letter}.png")
            sheet.paste(rgb.resize((768, 1024), Image.Resampling.LANCZOS), (index * 768, 28))
            draw.text((index * 768 + 12, 7), letter, fill="white")
        sheet.save(blind / f"{case['label']}.jpg", quality=94)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(key) == len(CASES):
        save_json(out / "blind_key.json", key)
        (blind / "prompts.md").write_bytes((REFERENCE / "blind" / "prompts.md").read_bytes())
        rating_path = blind / "ratings.csv"
        if not rating_path.exists():
            with rating_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["scene", "group", "seed", "letter", "preferred", "prompt_fidelity_1to5",
                                 "detail_1to5", "artifacts_1to5", "notes"])
                for case in CASES:
                    for letter in "AB":
                        writer.writerow([case["label"], case["group"], case["seed"], letter, "", "", "", "", ""])
    return len(rows), len(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--out", type=Path, default=Path("I:/HaDeS_MOSS/v094_vs_sft_v07_2026-11-11"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs; rerun to resume")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    with LORA_FILE.open("rb") as file:
        header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
    for case in CASES:
        reference_record(case)
    manifest = {"reference_archive": str(REFERENCE), "baked_model": MODEL, "sft_model": SFT_MODEL,
                "lora": LORA, "lora_strength": 1.0, "lora_sha256": sha(LORA_FILE.read_bytes()),
                "lora_metadata": header.get("__metadata__", {}), "size": list(SIZE),
                "clip": "qwen3.5_9b_int8_convrot.safetensors", "vae": "flux2-vae.safetensors",
                "steps": CONFIG[0], "cfg": CONFIG[1], "sampler": "euler", "scheduler": "simple",
                "flow_shift": 5.0, "denoise": 1.0, "slg": "off", "negative": NEGATIVE,
                "cases": list(CASES)}
    print(f"{len(CASES)} new jobs; {len(CASES)} existing baked references; output {args.out}", flush=True)
    if args.plan:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Existing manifest differs; use another output directory")
    save_json(path, manifest)
    try:
        for case in CASES[:args.limit]:
            run_one(args.server, args.out, case)
    finally:
        images, sheets = write_summaries(args.out)
        print(f"{images}/16 images, {sheets}/8 blind sheets", flush=True)


if __name__ == "__main__":
    main()
