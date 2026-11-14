"""Read-only inventory of image folders for a finetune dataset: one CSV row per image, no image is decoded.

Per image: source folder (first level under the root), format, size, megapixels, where the generation
parameters were found (A1111 / Forge PNG text or JPEG EXIF, ComfyUI prompt), model name and hash, hires
upscale, prompt style (tags / natural / mixed) and a style and explicit-content guess from the prompt
words. The guesses only sort the inventory; the dataset choice is made on the images later.

    python tools/moss_dataset_inventory.py --roots C:/images J:/dataset --out I:/inventory.csv
"""
import argparse, csv, json, os, re, time
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".webp")
NSFW = re.compile(r"\b(nsfw|nude|naked|nudity|nipples?|pussy|penis|vagina|sex|cum|anus|topless|explicit|genitals?|hentai|erotic)\b", re.I)
ANIME_W = re.compile(r"\b(1girl|2girls|1boy|solo|anime|manga|illustration|chibi|cel[ _]shad\w*|danbooru|masterpiece)\b", re.I)
PHOTO_W = re.compile(r"\b(photo\w*|realistic|photorealistic|raw photo|dslr|35mm|film grain|cinematic|bokeh|skin texture)\b", re.I)
ANIME_M = re.compile(r"(illustrious|noob|pony|animagine|anime|hassaku|wai|kohaku|novelai|nai)", re.I)
PHOTO_M = re.compile(r"(juggernaut|realvis|realistic|epicreal|photon|cyberreal|lustify|flux|real)", re.I)


def a1111(text):
    """(prompt, settings dict) from an A1111 / Forge parameters block."""
    lines = text.strip().split("\n")
    last = lines[-1] if "Steps:" in lines[-1] else ""
    body = "\n".join(lines[:-1] if last else lines)
    prompt = body.split("\nNegative prompt:")[0].strip()
    sett = dict(m.groups() for m in re.finditer(r"([A-Za-z][\w /-]*?): (\"[^\"]*\"|[^,]*)", last))
    return prompt, sett


def exif_comment(im):
    try:
        uc = im.getexif().get_ifd(0x8769).get(0x9286)
    except Exception:
        return ""
    if not uc:
        return ""
    if isinstance(uc, bytes):
        head, body = uc[:8], uc[8:]
        enc = "utf-16-be" if head.startswith(b"UNICODE") and body[:1] == b"\x00" else ("utf-16-le" if head.startswith(b"UNICODE") else "utf-8")
        return body.decode(enc, "ignore").strip("\x00")
    return str(uc)


def style_of(prompt):
    parts = [p.strip() for p in re.split(r"[,\n]", prompt) if p.strip()]
    if not parts:
        return ""
    short = sum(len(p.split()) <= 3 for p in parts) / len(parts)
    under = "_" in prompt
    return "tags" if short > 0.75 or (under and short > 0.5) else ("natural" if short < 0.4 else "mixed")


def row(path, root):
    rel = os.path.relpath(path, root)
    r = {"root": root, "path": rel, "source": rel.split(os.sep)[0] if os.sep in rel else "", "ext": os.path.splitext(path)[1].lower(),
         "bytes": 0, "w": 0, "h": 0, "mp": 0.0, "meta": "", "model": "", "model_hash": "", "hires": "", "prompt_style": "",
         "look": "", "explicit": "", "prompt": "", "error": ""}
    try:
        r["bytes"] = os.path.getsize(path)
        with Image.open(path) as im:
            r["w"], r["h"] = im.size
            prompt, sett = "", {}
            if "parameters" in im.info:
                r["meta"] = "a1111"
                prompt, sett = a1111(str(im.info["parameters"]))
            elif "prompt" in im.info:
                r["meta"] = "comfyui"
                try:
                    g = json.loads(im.info["prompt"])
                    texts = [t for v in g.values() if isinstance(v, dict) for k, t in v.get("inputs", {}).items()
                             if k in ("text", "text2", "t5xxl", "clip_l", "prompt", "text_g", "positive") and isinstance(t, str)]
                    prompt = max(texts, key=len) if texts else ""
                    ck = [v["inputs"].get("ckpt_name") or v["inputs"].get("unet_name") for v in g.values() if isinstance(v, dict)
                          and (v.get("inputs", {}).get("ckpt_name") or v.get("inputs", {}).get("unet_name"))]
                    sett = {"Model": ck[0]} if ck else {}
                except Exception:
                    pass
            elif r["ext"] in (".jpg", ".jpeg", ".webp"):
                c = exif_comment(im)
                if "Steps:" in c:
                    r["meta"] = "a1111-exif"
                    prompt, sett = a1111(c)
        r["mp"] = round(r["w"] * r["h"] / 1e6, 2)
        r["model"] = sett.get("Model", "").strip('"')
        r["model_hash"] = sett.get("Model hash", "")
        r["hires"] = sett.get("Hires upscale", "") or sett.get("Hires upscaler", "")
        if prompt:
            r["prompt_style"] = style_of(prompt)
            a, p = len(ANIME_W.findall(prompt)), len(PHOTO_W.findall(prompt))
            if ANIME_M.search(r["model"]):
                a += 2
            if PHOTO_M.search(r["model"]):
                p += 2
            r["look"] = "anime" if a > p else ("photo" if p > a else "unknown")
            r["explicit"] = "yes" if NSFW.search(prompt) else "no"
            r["prompt"] = prompt[:500].replace("\n", " ")
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"[:200]
    return r


def walk(root):
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(EXTS):
                yield os.path.join(d, f)


def job(args):
    return row(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images per root (timing test)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = []
    for root in a.roots:
        files = list(walk(root))
        tasks += [(f, root) for f in (files[:a.limit] if a.limit else files)]
    print(f"{len(tasks)} images", flush=True)
    t0 = time.time()
    with open(a.out, "w", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = None
        for i, r in enumerate(ex.map(job, tasks, chunksize=64)):
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(r))
                w.writeheader()
            w.writerow(r)
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(tasks)} in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
