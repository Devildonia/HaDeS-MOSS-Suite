"""Text-to-image LoRA dataset from one's own generated images, picked from a moss_dataset_inventory.py CSV.

    select   candidates from the inventory: own generations (no grids, no FLUX.1-dev output, whose licence forbids
             training other models on it), >= 0.9 MP, no explicit or suggestive prompt words, no booru character
             tags, one image per prompt, a cap per source model, split between photo and anime by the inventory guess
    prepare  copies resized to ~1 MP (aspect kept, sides a multiple of 16) into <out>/pairs
    caption  the MOSS text encoder (Qwen3.5, through ComfyUI) answers nudity / watermark questions and writes a
             natural-language caption that names the style
    final    keeps the images that pass, up to --n, as a moss_edit_train.py --t2i folder (pairs.json)
    sheet    contact sheets of the final images for a look by eye

    python tools/moss_forge_dataset.py select  --inventory inv.csv --out I:/forge_lora --per-look 750
    python tools/moss_forge_dataset.py prepare --out I:/forge_lora
    python tools/moss_forge_dataset.py caption --out I:/forge_lora
    python tools/moss_forge_dataset.py final   --out I:/forge_lora --n 1000
"""
import argparse, csv, io, json, os, random, re, sys, time
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from moss_edit_pairs import req, upload  # noqa: E402  (ComfyUI API helpers with retry)

csv.field_size_limit(10 ** 7)
SUGGESTIVE = re.compile(r"(nsfw|nude|naked|nipple|pussy|penis|vagina|\bsex|cum\b|anus|topless|explicit|genital|hentai|erotic|breast|cleavage|"
                        r"lingerie|underwear|panties|bikini|swimsuit|bra\b|\bass\b|butt|thigh|see-through|bdsm|bondage|seductive|sexy|lewd|"
                        r"spread|cameltoe|areola|naughty|undress|bath|shower|loli|shota|child|kid\b|teen)", re.I)
CHARACTER = re.compile(r"[a-z0-9]_\([a-z0-9_ .:-]+\)|cosplay", re.I)  # booru character tags such as name_(series)
ASK_NUDE = "Does this picture show nudity, underwear or sexual content? Answer only yes or no."
ASK_MARK = "Is there a watermark, a signature, a logo or a text overlay in this picture? Answer only yes or no."
ASK_CAPTION = ("Describe this picture for an image generator in 2 to 4 sentences of natural language. Begin with the medium and style, "
               "for example \"Photograph of\", \"Anime illustration of\", \"3D render of\" or \"Digital painting of\". Then the main subject, "
               "its appearance and action, the setting, the lighting, the colours and the framing. Describe only what is visible. "
               "Do not name characters, artists or franchises.")


def norm_prompt(p):
    p = re.sub(r"<[^>]*>", " ", p.lower())
    p = re.sub(r":[0-9.]+", " ", p)
    p = re.sub(r"[()\[\]{}]", " ", p)
    return " ".join(p.replace("_", " ").split())


def select(a):
    rows = list(csv.DictReader(open(a.inventory, encoding="utf-8")))
    ok = [r for r in rows if r["source"] == "txt2img-images" and r["meta"] and "flux" not in r["model"].lower()
          and float(r["mp"]) >= 0.9 and r["explicit"] == "no" and r["look"] in ("photo", "anime") and r["prompt"]
          and not SUGGESTIVE.search(r["prompt"]) and not CHARACTER.search(r["prompt"])]
    print(f"{len(ok)} of {len(rows)} rows pass the metadata filters", flush=True)
    by_prompt = {}
    for r in ok:  # one image per prompt: the original PNG beats its upscaled JPEG, then the smaller file
        k = norm_prompt(r["prompt"])
        best = by_prompt.get(k)
        rank = (r["ext"] != ".png", float(r["mp"]))
        if best is None or rank < (best["ext"] != ".png", float(best["mp"])):
            by_prompt[k] = r
    uniq = list(by_prompt.values())
    print(f"{len(uniq)} distinct prompts", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(uniq)
    picked, per_model = [], {}
    for look in ("photo", "anime"):
        n = 0
        for r in uniq:
            key = (look, r["model"])
            if r["look"] != look or per_model.get(key, 0) >= a.per_model:
                continue
            per_model[key] = per_model.get(key, 0) + 1
            picked.append(r)
            n += 1
            if n == a.per_look:
                break
        print(f"{look}: {n} from {len({m for (l, m) in per_model if l == look})} models", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = [{"id": f"f{i:05d}", "src": os.path.join(r["root"], r["path"]), "model": r["model"], "look": r["look"],
              "prompt_style": r["prompt_style"], "orig_prompt": r["prompt"]} for i, r in enumerate(picked)]
    (out / "candidates.json").write_text(json.dumps(cands, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cands)} candidates -> {out / 'candidates.json'}")


def prepare(a):
    out = Path(a.out)
    (out / "pairs").mkdir(exist_ok=True)
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    t0 = time.time()
    for i, c in enumerate(cands):
        dst = out / "pairs" / f"{c['id']}.png"
        if dst.exists():
            continue
        im = Image.open(c["src"]).convert("RGB")
        k = (a.size * a.size / (im.width * im.height)) ** 0.5
        im.resize((round(im.width * k / 16) * 16, round(im.height * k / 16) * 16), Image.LANCZOS).save(dst)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(cands)} {time.time() - t0:.0f}s", flush=True)
    print("prepared", len(cands))


def textgen(img, question, max_length):
    buf = io.BytesIO()
    small = img.copy()
    small.thumbnail((768, 768))
    small.save(buf, "PNG")
    g = {"clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
         "img": {"class_type": "LoadImage", "inputs": {"image": upload("hades_forge_vlm.png", buf.getvalue())}},
         "gen": {"class_type": "TextGenerate", "inputs": {"clip": ["clip", 0], "prompt": question, "max_length": max_length,
                 "sampling_mode": "off", "image": ["img", 0], "thinking": False}},
         "show": {"class_type": "PreviewAny", "inputs": {"source": ["gen", 0]}}}
    pid = req("/prompt", {"prompt": g, "client_id": "hades_forge_dataset"})["prompt_id"]
    while pid not in (h := req(f"/history/{pid}")):
        time.sleep(0.3)
    assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
    return h[pid]["outputs"]["show"]["text"][0].strip()


def whole_sentences(t):
    """Drops a sentence cut by max_length."""
    return t if t.endswith((".", "!", "?")) or "." not in t else t[:t.rfind(".") + 1]


def caption(a):
    out = Path(a.out)
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    path = out / "captions.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t0, n = time.time(), 0
    for c in cands:
        if c["id"] in done:
            continue
        img = Image.open(out / "pairs" / f"{c['id']}.png").convert("RGB")
        nude = textgen(img, ASK_NUDE, 8).strip(".").lower()
        rec = {"nude": nude}
        if nude.startswith("no"):
            rec["mark"] = textgen(img, ASK_MARK, 8).strip(".").lower()
            if rec["mark"].startswith("no"):
                rec["caption"] = whole_sentences(" ".join(textgen(img, ASK_CAPTION, a.max_length).split()))
        done[c["id"]] = rec
        n += 1
        if n % 20 == 0:
            path.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
            print(f"{len(done)}/{len(cands)} {time.time() - t0:.0f}s ({(time.time() - t0) / n:.1f} s/image)", flush=True)
    path.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
    print("captioned", len(done))


def final(a):
    out = Path(a.out)
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    caps = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    good = [c for c in cands if caps.get(c["id"], {}).get("caption") and len(caps[c["id"]]["caption"].split()) >= 12]
    looks = {}
    for c in good:
        looks.setdefault(c["look"], []).append(c)
    per = a.n // max(1, len(looks))
    keep = [c for l in sorted(looks) for c in looks[l][:per]]
    pairs = [{"pair_id": c["id"], "scene_id": c["id"], "a": f"{c['id']}.png", "b": f"{c['id']}.png", "instruction": whole_sentences(caps[c["id"]]["caption"]),
              "object": "", "vlm": ["yes", "no"], "count": 1, "mask_area": 0.0, "model": c["model"], "look": c["look"], "src": c["src"]} for c in keep]
    (out / "pairs.json").write_text(json.dumps(pairs, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"{len(good)} of {len(cands)} pass; kept {len(pairs)}: " + ", ".join(f"{l} {min(per, len(v))}" for l, v in sorted(looks.items())))


def sheet(a):
    out = Path(a.out)
    pairs = json.loads((out / "pairs.json").read_text(encoding="utf-8"))
    t, cols = 200, 10
    for s in range(0, len(pairs), cols * 5):
        chunk = pairs[s:s + cols * 5]
        img = Image.new("RGB", (cols * t, ((len(chunk) + cols - 1) // cols) * t), (18, 18, 20))
        for i, p in enumerate(chunk):
            im = Image.open(out / "pairs" / p["a"]).convert("RGB")
            im.thumbnail((t, t))
            img.paste(im, ((i % cols) * t + (t - im.width) // 2, (i // cols) * t + (t - im.height) // 2))
            ImageDraw.Draw(img).text(((i % cols) * t + 3, (i // cols) * t + 3), p["pair_id"], fill=(255, 255, 0))
        (out / "sheets").mkdir(exist_ok=True)
        img.save(out / "sheets" / f"sheet_{s // (cols * 5):03d}.jpg", quality=85)
    print("sheets in", out / "sheets")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "prepare", "caption", "final", "sheet"])
    ap.add_argument("--inventory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-look", type=int, default=750)
    ap.add_argument("--per-model", type=int, default=30, help="at most this many images of one source model per look")
    ap.add_argument("--size", type=int, default=1024, help="prepared images have about size x size pixels")
    ap.add_argument("--max-length", type=int, default=260)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"select": select, "prepare": prepare, "caption": caption, "final": final, "sheet": sheet}[a.mode](a)


if __name__ == "__main__":
    main()
