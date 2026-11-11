"""Eight new paired scenes: baked v0.9.4 against SFT plus Turbo LoRA v0.9.

The prompts and seeds are frozen below. Generates full-resolution blind copies
without model metadata, paired contact sheets, exact graphs and image metrics.
"""

import argparse
import csv
import io
import json
import random
import time
import urllib.parse
from pathlib import Path

from PIL import Image, ImageDraw

from moss_baked_turbo_eval import MODEL, SIZE, graph_for, metrics, request, save_json, sha
from moss_v094_vs_v09_eval import LORA, LORA_FILE, comparison_graph


NEGATIVE = "worst quality, bad anatomy, extra fingers, distorted hands, illegible text, watermark, jpeg artifact"
CASES = (
    {"label": "watchmaker", "group": "person", "seed": 2401, "positive":
     "A documentary portrait of a 62-year-old woman watchmaker seated at a wooden workbench. "
     "Her face, silver hair and both hands are visible as she adjusts a brass pocket watch with fine tweezers. "
     "Natural skin pores, age lines, subtle freckles and individual hair strands; worn navy cotton apron, "
     "tiny gears and tools sharply resolved near her hands. Soft daylight from a window on the left, "
     "warm practical lamp behind her, realistic anatomy and photographic depth of field."},
    {"label": "trumpeter", "group": "person", "seed": 2402, "positive":
     "A close three-quarter portrait of a Black male jazz trumpeter backstage after a performance. "
     "He is in his late thirties, holding a slightly tarnished brass trumpet with one hand while the other "
     "rests on its valves. Realistic dark skin texture, small beads of sweat, natural fingers, short curly hair, "
     "creased burgundy velvet jacket and white shirt. A single warm stage light and dim blue backstage fill, "
     "natural expression, believable metal reflections, candid editorial photography."},
    {"label": "climber", "group": "person", "seed": 2403, "positive":
     "A medium portrait of a young East Asian mountaineer paused beside a snowy ridge at dawn. "
     "Wind-reddened cheeks, damp eyelashes, loose strands of dark hair escaping a knitted cap, "
     "gloved hands gripping the straps of a weathered climbing pack. The insulated teal jacket shows "
     "distinct stitched seams, frost crystals and wet fabric texture. Cold diffuse mountain light, "
     "distant layered peaks, realistic proportions and restrained natural colors, documentary photograph."},
    {"label": "flower_seller", "group": "person", "seed": 2404, "positive":
     "An elderly South Asian flower seller standing at an open-air market in morning light, shown from the waist up. "
     "Her expressive face and both hands are clearly visible as she ties a small bouquet of orange marigolds. "
     "Fine facial lines, natural brown skin texture, gray hair, simple cotton sari with visible weave, "
     "flower petals and stems in crisp detail. Market stalls softly blurred behind her, believable hand anatomy, "
     "warm but natural color, candid photographic composition."},
    {"label": "watch_macro", "group": "nonperson", "seed": 2405, "positive":
     "Macro photograph of an opened antique mechanical pocket watch resting on dark walnut. "
     "The exposed brass balance wheel, tiny steel screws, ruby bearings and interlocking gears are physically coherent. "
     "Fine scratches, tarnish, dust and precise machined edges are visible. A folded cream linen cloth and "
     "watchmaker's loupe sit nearby. Directional window light, controlled specular highlights, shallow depth of field, "
     "photorealistic material texture and no people."},
    {"label": "rainy_alley", "group": "nonperson", "seed": 2406, "positive":
     "A narrow empty city alley just after rain at blue hour. Wet brick, metal shutters, tangled utility cables "
     "and puddles reflect a small illuminated storefront sign that clearly reads CAFE 24. A red bicycle leans "
     "against the left wall, a green delivery crate sits near the doorway. Strong perspective, physically "
     "plausible reflections, fine surface wear, cool ambient light balanced by warm shop light. No people."},
    {"label": "citrus_still_life", "group": "nonperson", "seed": 2407, "positive":
     "A carefully composed still life of three oranges, one cut lemon and a glass jug of water on a blue glazed "
     "tile kitchen counter. Citrus peel pores, translucent pulp, condensation droplets, chipped tile edges "
     "and a loosely folded natural linen napkin are clearly visible. Soft north-window daylight from the right, "
     "subtle realistic shadows, restrained color, photographic material detail and no people."},
    {"label": "observatory", "group": "nonperson", "seed": 2408, "positive":
     "Interior of an empty lunar observatory control room with a massive circular viewing window showing the Moon's "
     "horizon and a distant blue Earth. Modular consoles, bundled cables, worn aluminum floor panels and a single "
     "orange safety rail have clear mechanical structure. A wall stencil reads OBSERVATORY 7 in large legible letters. "
     "Cool overhead task lights mix with pale sunlight from the window, deep perspective, realistic science-fiction "
     "production design, no people."},
)
VARIANTS = ("baked_v094", "sft_v09")
CONFIG = (8, 1.0)


def graph_for_variant(case, variant):
    source = {"positive": case["positive"], "negative": NEGATIVE}
    if variant == "baked_v094":
        return graph_for(source, case["seed"], *CONFIG)
    return comparison_graph(source, case["seed"], "sft_v09")


def run_one(server, out, case, variant):
    stem = f"{case['label']}_{case['seed']}_{variant}"
    image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
    graph = graph_for_variant(case, variant)
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
    prompt_id = request(server, "/prompt", {"prompt": graph, "client_id": "hades_moss_v094_sft_heldout"})["prompt_id"]
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
        records = {}
        for variant in VARIANTS:
            stem = f"{case['label']}_{case['seed']}_{variant}"
            image_path, record_path = out / f"{stem}.png", out / f"{stem}.json"
            if not image_path.exists() or not record_path.exists():
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("state") != "success" or record["image_sha256"] != sha(image_path.read_bytes()):
                raise RuntimeError(f"Result changed or failed: {stem}")
            records[variant] = image_path
            rows.append({"scene": case["label"], "group": case["group"], "seed": case["seed"],
                         "variant": variant, "image": str(image_path), "seconds": record["seconds"],
                         **record["metrics"]})
        if len(records) != 2:
            continue
        blind.mkdir(exist_ok=True)
        variants = list(records.items())
        random.Random(int(sha(f"v094_sft_heldout:{case['label']}:{case['seed']}".encode()), 16)).shuffle(variants)
        key[case["label"]] = {chr(65 + i): name for i, (name, _) in enumerate(variants)}
        sheet = Image.new("RGB", (2 * 768, 1052), "#17191b")
        draw = ImageDraw.Draw(sheet)
        for index, (_, path) in enumerate(variants):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            letter = chr(65 + index)
            rgb.save(blind / f"{case['label']}_{letter}.png")  # Re-encode without the graph metadata.
            sheet.paste(rgb.resize((768, 1024), Image.Resampling.LANCZOS), (index * 768, 28))
            draw.text((index * 768 + 12, 7), letter, fill="white")
        sheet.save(blind / f"{case['label']}.jpg", quality=94)
    if rows:
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if len(key) == len(CASES):
        save_json(out / "blind_key.json", key)
        ratings = blind / "ratings.csv"
        if not ratings.exists():
            with ratings.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["scene", "group", "seed", "letter", "preferred", "prompt_fidelity_1to5",
                                 "detail_1to5", "artifacts_1to5", "text_accuracy", "notes"])
                for case in CASES:
                    for letter in "AB":
                        writer.writerow([case["label"], case["group"], case["seed"], letter, "", "", "", "", "", ""])
        prompts = blind / "prompts.md"
        prompts.write_text("# Blind comparison prompts\n\nMark one preferred image per scene in `ratings.csv`. "
                           "For detail, inspect the full-resolution A/B PNGs. The numerical ratings are optional: "
                           "1 = poor, 5 = excellent; for artifacts, 5 = fewest visible defects.\n\n" +
                           "\n\n".join(f"## {case['label']} ({case['group']}, seed {case['seed']})\n\n"
                                       f"{case['positive']}\n\nNegative: {NEGATIVE}"
                                       for case in CASES) + "\n", encoding="utf-8")
    return len(rows), len(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--out", type=Path, default=Path("I:/HaDeS_MOSS/v094_vs_sft_v09_heldout_2026-11-11"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs; rerun to resume")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    manifest = {"baked_model": MODEL, "sft_model": "mossImage_sft_bf16.safetensors", "lora": LORA,
                "lora_strength": 1.0, "lora_sha256": sha(LORA_FILE.read_bytes()),
                "clip": "qwen3.5_9b_int8_convrot.safetensors", "vae": "flux2-vae.safetensors",
                "size": list(SIZE), "steps": CONFIG[0], "cfg": CONFIG[1], "sampler": "euler",
                "scheduler": "simple", "flow_shift": 5.0, "denoise": 1.0, "slg": "off",
                "negative": NEGATIVE, "cases": list(CASES)}
    jobs = [(case, variant) for case in CASES for variant in VARIANTS]
    print(f"{len(jobs)} jobs; 8 new scenes; output {args.out}", flush=True)
    if args.plan:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Existing manifest differs; use another output directory")
    save_json(path, manifest)
    try:
        for case, variant in jobs[:args.limit]:
            run_one(args.server, args.out, case, variant)
    finally:
        images, sheets = write_summaries(args.out)
        print(f"{images}/16 images, {sheets}/8 blind sheets", flush=True)


if __name__ == "__main__":
    main()
