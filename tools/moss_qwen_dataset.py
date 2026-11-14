"""Daily Qwen-Image dataset batches: random prompts, renders, automatic checks, review sheets.

    generate  the MOSS Prompt Enhancer (detailed mode, empty idea) writes random prompts, the seed drawing a medium, a subject
              area and a place; Qwen-Image-2512 (GGUF) with its Lightning 8-step LoRA renders them. One record per image with
              its provenance (prompt, draw, model, LoRA, settings, date) and status "pending".
    check     flags near-duplicates before the human review: a 64-bit difference hash within --dup-bits of an image of the
              same or an earlier batch. Whether the image shows the drawn medium is left to the reviewer: asked by Qwen3.5-9B,
              it called most Qwen-Image photographs 3D renders and comic covers paintings (7 of 8 test images flagged wrongly).
    sheet     review sheets with id and flags.

All through a running ComfyUI (the HaDeS MOSS Suite and ComfyUI-GGUF installed).

    python tools/moss_qwen_dataset.py generate --root I:/HaDeS_MOSS/qwen_dataset --batch 2026-11-14 --n 40 --seed-start 1000
    python tools/moss_qwen_dataset.py check    --root I:/HaDeS_MOSS/qwen_dataset --batch 2026-11-14
    python tools/moss_qwen_dataset.py sheet    --root I:/HaDeS_MOSS/qwen_dataset --batch 2026-11-14
"""
import argparse, io, json, os, re, sys, time, urllib.parse
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from moss_edit_pairs import req  # noqa: E402  (ComfyUI API helper with retry)

QWEN = "qwen-image-2512-Q8_0.gguf"
LIGHTNING = "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"
def run(graph, cid):
    pid = req("/prompt", {"prompt": graph, "client_id": cid})["prompt_id"]
    while pid not in (h := req(f"/history/{pid}")):
        time.sleep(0.5)
    if h[pid]["status"]["status_str"] != "success":
        raise RuntimeError(h[pid]["status"])
    return h[pid]["outputs"]


def records_path(a):
    return Path(a.root) / a.batch / "records.json"


def load(a):
    p = records_path(a)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save(a, recs):
    records_path(a).write_text(json.dumps(recs, indent=1, ensure_ascii=False), encoding="utf-8")


def size_of(a, seed):
    w, h = a.sizes.split(",")[seed % len(a.sizes.split(","))].lower().split("x")
    return int(w), int(h)


def generate(a):
    out = Path(a.root) / a.batch
    out.mkdir(parents=True, exist_ok=True)
    recs = load(a)
    have = {r["enhancer_seed"] for r in recs}
    todo = [s for s in range(a.seed_start, a.seed_start + a.n) if s not in have]
    for c in range(0, len(todo), a.chunk):  # prompts of a chunk first, then its renders: fewer model swaps, little lost on a stop
        recs = generate_chunk(a, out, recs, todo[c:c + a.chunk])
    print(len(recs), "records in", records_path(a))


def generate_chunk(a, out, recs, seeds):
    new = []
    for s in seeds:
        w, h = size_of(a, s)
        o = run({"clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
                 "enh": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {"clip": ["clip", 0], "idea": "", "mode": "detailed", "width": w,
                         "height": h, "temperature": 0.7, "seed": s, "max_tokens": 1024, "negative": ""}},
                 "info": {"class_type": "PreviewAny", "inputs": {"source": ["enh", 6]}}}, "hades_qwen_dataset")
        draw = re.search(r"random: (.*?)(?: \||\n)", o["info"]["text"][0])
        new.append({"id": f"q{s:05d}", "enhancer_seed": s, "draw": draw.group(1) if draw else "", "prompt": o["enh"]["text"][0]})
        print("prompt", s, new[-1]["draw"], flush=True)
    for r in new:
        t0 = time.time()
        w, h = size_of(a, r["enhancer_seed"])
        o = run({"unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN}},
                 "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": LIGHTNING, "strength_model": 1.0}},
                 "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": 3.0}},
                 "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
                 "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                 "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": r["prompt"]}},
                 "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}},
                 "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                 "ks": {"class_type": "KSampler", "inputs": {"model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
                        "seed": r["enhancer_seed"], "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
                 "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
                 "out": {"class_type": "PreviewImage", "inputs": {"images": ["dec", 0]}}}, "hades_qwen_dataset")
        Image.open(io.BytesIO(req("/view?" + urllib.parse.urlencode(o["out"]["images"][0]), raw=True))).save(out / f"{r['id']}.png")
        r.update({"file": f"{r['id']}.png", "model": QWEN, "lora": LIGHTNING, "steps": 8, "cfg": 1.0, "sampler": "euler/simple", "shift": 3.0,
                  "size": f"{w}x{h}", "image_seed": r["enhancer_seed"], "batch": a.batch, "status": "pending", "flags": []})
        recs.append(r)
        save(a, recs)
        print("render", r["id"], f"{time.time() - t0:.1f}s", flush=True)
    return recs


def dhash(path):
    g = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def check(a):
    recs = load(a)
    seen = []  # hashes of earlier batches, then of this batch as it goes
    for d in sorted(Path(a.root).iterdir()):
        if d.is_dir() and d.name < a.batch and (d / "records.json").exists():
            seen += [(r["id"], r["dhash"]) for r in json.loads((d / "records.json").read_text(encoding="utf-8")) if "dhash" in r]
    for r in recs:
        p = Path(a.root) / a.batch / r["file"]
        flags = [f for f in r.get("flags", []) if not f.startswith(("medium", "duplicate"))]
        r.pop("medium_seen", None)
        r["dhash"] = h = dhash(p)
        close = [i for i, hh in seen if bin(h ^ hh).count("1") <= a.dup_bits]
        if close:
            flags.append("duplicate of " + ", ".join(close))
        seen.append((r["id"], h))
        r["flags"] = flags
        print(r["id"], "; ".join(flags) or "ok", flush=True)
    save(a, recs)
    print(sum(bool(r["flags"]) for r in recs), "of", len(recs), "flagged")


def sheet(a):
    from PIL import ImageFont
    recs = load(a)
    out = Path(a.root) / a.batch
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16) if os.path.exists("C:/Windows/Fonts/arial.ttf") else None
    tw, th, cols = 360, 480, 5
    for s in range(0, len(recs), cols * 4):
        chunk = recs[s:s + cols * 4]
        img = Image.new("RGB", (cols * tw, ((len(chunk) + cols - 1) // cols) * th), (18, 18, 20))
        d = ImageDraw.Draw(img)
        for i, r in enumerate(chunk):
            im = Image.open(out / r["file"]).convert("RGB")
            im.thumbnail((tw, th))
            x, y = (i % cols) * tw, (i // cols) * th
            img.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
            colour = (255, 80, 80) if r.get("flags") else (255, 255, 0)
            d.rectangle((x, y, x + tw - 1, y + 22), fill=(0, 0, 0))
            d.text((x + 4, y + 2), f"{r['id']} {'FLAG' if r.get('flags') else ''}", fill=colour, font=font)
        img.save(out / f"sheet_{s // (cols * 4):02d}.jpg", quality=88)
    print("sheets in", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["generate", "check", "sheet"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch", required=True, help="batch folder name, e.g. the date")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--sizes", default="1104x1472,1472x1104,1328x1328", help="Qwen-Image sizes, one per image in turn by seed")
    ap.add_argument("--chunk", type=int, default=50, help="prompts written and rendered this many at a time")
    ap.add_argument("--dup-bits", type=int, default=6, help="difference-hash distance at or below which two images count as duplicates")
    a = ap.parse_args()
    {"generate": generate, "check": check, "sheet": sheet}[a.mode](a)


if __name__ == "__main__":
    main()
