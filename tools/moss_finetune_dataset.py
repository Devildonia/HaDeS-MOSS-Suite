"""The finetune dataset from one's own generations, picked by quotas from moss_dataset_inventory.py CSVs.

    select   one row per distinct prompt from the user's own models (name list below), usable (prompt in the metadata, >= 0.9 MP),
             no explicit + young-character prompt and, by default, no young-character words at all; then quotas per block
             (Velvet Paradox, anime / illustration, realism, a reserve for scarce traits) with the scarce traits (monochrome,
             skin / ethnicity, two or more characters, camera angle) picked first, an explicit share and a cap per model,
             oversampled so the reviews can drop some
    prepare  copies resized to ~1 MP (aspect kept, sides a multiple of 16) into <out>/pairs (moss_forge_dataset.prepare)
    dedupe   drops near-duplicate images (64-bit dHash, Hamming distance <= --dhash-max)
    vlm      MOSS-Image's own text encoder (Qwen3.5, through ComfyUI) answers: watermark / signature / text overlay?, and for
             explicit prompts whether anyone looks under 18; then writes a natural-language caption kept next to the
             original prompt (two captions per image). Resumable.
    sheets   contact sheets per block, numbered, for the review by eye
    final    the images that pass, up to --n, with both captions -> <out>/dataset.json

    python tools/moss_finetune_dataset.py select --inventories inv1.csv inv2.csv --out I:/finetune6000 --n 6000
"""
import argparse, csv, json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import moss_forge_dataset as forge  # noqa: E402  (prepare, textgen, whole_sentences, norm_prompt)

csv.field_size_limit(10 ** 8)
# the user's own models (2026-11-19): names with HaDeS / Velvet, and these finetunes without it in the name
OWN = re.compile(r"hades|hadees|velvet|renaissance|shirea|\bocd|toxic.?muse|genuine.?illustrious|hyper illustrious|scyrax", re.I)
YOUNG = re.compile(r"\b(loli|shota|child|children|kid|kids|toddler|teen|teenager|young girl|young boy|little girl|little boy|schoolgirl|"
                   r"schoolboy|school uniform|serafuku|randoseru|elementary|middle school|aged down|petite girl|underage|baby)\b", re.I)
MONO = re.compile(r"\b(monochrome|greyscale|grayscale|black and white|b&w|lineart|line art|ink drawing|sketch|manga panel|screentone)\b", re.I)
CAMERA = re.compile(r"\b(from above|from below|from behind|from side|low angle|high angle|dutch angle|fisheye|bird'?s[- ]eye|worm'?s[- ]eye|"
                    r"overhead|aerial view|close-up|closeup|wide shot|full shot|cowboy shot|profile view|pov)\b", re.I)
MULTI = re.compile(r"\b(2girls|2boys|3girls|multiple girls|multiple boys|1boy, 1girl|1girl, 1boy|couple|hetero|yuri|yaoi|two women|two men|"
                   r"two girls|two people|group of|duo|crowd)\b", re.I)
SKIN = re.compile(r"\b(dark skin|dark-skinned|dark skinned|black woman|black man|black girl|african|ebony|brown skin|tan skin|tanned|"
                  r"asian|japanese|chinese|korean|indian|south asian|latina|latino|hispanic|middle eastern|arab|inuit|native american|"
                  r"indigenous|mixed race)\b", re.I)
REALISM_XL = re.compile(r"realism.*xl|xl.*realism", re.I)
REALISM = re.compile(r"realism|photon|real\b", re.I)
TRAITS = {"mono": MONO, "skin": SKIN, "multi": MULTI, "camera": CAMERA}
ASK_MARK = "Is there a watermark, a signature, a logo or a text overlay added on top of this picture? Answer only yes or no."
ASK_AGE = "Does any person in this picture look younger than 18 years old? Answer only yes or no."
ASK_CAPTION = forge.ASK_CAPTION


def block_of(r):
    m = r["model"]
    if re.search("velvet", m, re.I):
        return "velvet"
    if REALISM_XL.search(m) or (REALISM.search(m) and r["look"] == "photo"):
        return "realism"
    return "anime"


def load_rows(paths, allow_young):
    rows = []
    for path in paths:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["root"].replace("\\", "/").startswith("C:/Hades/ForgeNeo") and r["source"] != "txt2img-images":
                continue  # grids: not single generations
            p = r["prompt"]
            if not (p and not r["error"] and float(r["mp"] or 0) >= 0.9 and OWN.search(r["model"])):
                continue
            young = bool(YOUNG.search(p))
            if young and (r["explicit"] == "yes" or not allow_young):
                continue
            rows.append(r)
    # one row per prompt: the original PNG beats its re-encoded copies, then the size nearest 1.5 MP
    best = {}
    for r in rows:
        k = forge.norm_prompt(r["prompt"])
        rank = (r["ext"] != ".png", abs(float(r["mp"]) - 1.5))
        if k not in best or rank < best[k][0]:
            best[k] = (rank, r)
    return [r for _, r in best.values()]


def select(a):
    rows = load_rows(a.inventories, a.allow_young)
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    for r in rows:
        r["block"] = block_of(r)
        r["traits"] = [t for t, rx in TRAITS.items() if rx.search(r["prompt"])]
        r["exp"] = r["explicit"] == "yes"
    print(f"{len(rows)} distinct usable prompts from own models; blocks: {dict(Counter(r['block'] for r in rows))}", flush=True)
    over = a.oversample
    quotas = {"velvet": round(a.n * 0.15 * over), "anime": round(a.n * 0.55 * over), "realism": round(a.n * 0.20 * over)}
    reserve = round(a.n * over) - sum(quotas.values())
    picked, used, per_model = [], set(), Counter()

    def take(pool, k, explicit_share):
        """k rows from pool: scarce traits first (rarest first), explicit kept near explicit_share, cap per model."""
        weight = {"mono": 4, "skin": 4, "multi": 2, "camera": 1}
        pool = sorted(pool, key=lambda r: -sum(weight[t] for t in r["traits"]))
        got, n_exp = [], 0
        for r in pool:
            if len(got) >= k:
                break
            key = id(r)
            if key in used or per_model[r["model"]] >= a.per_model:
                continue
            if r["exp"] and n_exp >= explicit_share * k:
                continue
            used.add(key)
            per_model[r["model"]] += 1
            n_exp += r["exp"]
            got.append(r)
        return got

    for block, k in quotas.items():
        got = take([r for r in rows if r["block"] == block], k, a.explicit_share)
        picked += got
        print(f"{block}: {len(got)} of {k}", flush=True)
    # reserve: whatever is left that carries monochrome or skin / ethnicity, any block; when those are all taken already,
    # the rest of the pool by the same trait order (two or more characters, camera angles first)
    got = take([r for r in rows if {"mono", "skin"} & set(r["traits"])], reserve, a.explicit_share)
    got += take([r for r in rows if r["block"] != "velvet"], reserve - len(got), a.explicit_share)
    picked += got
    print(f"reserve (monochrome, skin / ethnicity, then the rest): {len(got)} of {reserve}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = [{"id": f"d{i:05d}", "src": os.path.join(r["root"], r["path"]), "model": r["model"], "look": r["look"], "block": r["block"],
              "traits": r["traits"], "explicit": r["exp"], "prompt_style": r["prompt_style"], "orig_prompt": r["prompt"], "mp": float(r["mp"])}
             for i, r in enumerate(picked)]
    (out / "candidates.json").write_text(json.dumps(cands, indent=0, ensure_ascii=False), encoding="utf-8")
    tally = Counter(t for c in cands for t in c["traits"])
    print(f"{len(cands)} candidates -> {out / 'candidates.json'} | explicit {sum(c['explicit'] for c in cands)} | "
          + ", ".join(f"{t} {tally[t]}" for t in TRAITS) + f" | models {len(per_model)}")


def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(im.getdata())
    return sum(1 << i for i in range(size * size) if px[(i // size) * (size + 1) + i % size] > px[(i // size) * (size + 1) + i % size + 1])


def dedupe(a):
    out = Path(a.out)
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    hashes, keep, dropped = [], [], []
    for c in cands:
        h = dhash(out / "pairs" / f"{c['id']}.png")
        dup = next((k for k, hk in hashes if bin(h ^ hk).count("1") <= a.dhash_max), None)
        if dup is None:
            hashes.append((c["id"], h))
            keep.append(c)
        else:
            dropped.append({"id": c["id"], "duplicate_of": dup})
    (out / "candidates.json").write_text(json.dumps(keep, indent=0, ensure_ascii=False), encoding="utf-8")
    (out / "duplicates.json").write_text(json.dumps(dropped, indent=0), encoding="utf-8")
    print(f"{len(dropped)} near-duplicates dropped, {len(keep)} candidates left")


def vlm(a):
    out = Path(a.out)
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    path = out / "vlm.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t0, n = time.time(), 0
    for c in cands:
        if c["id"] in done:
            continue
        img = Image.open(out / "pairs" / f"{c['id']}.png").convert("RGB")
        rec = {"mark": forge.textgen(img, ASK_MARK, 8).strip(".").lower()}
        if c["explicit"]:
            rec["minor"] = forge.textgen(img, ASK_AGE, 8).strip(".").lower()
        if rec["mark"].startswith("no") and not rec.get("minor", "no").startswith("yes"):
            rec["caption"] = forge.whole_sentences(" ".join(forge.textgen(img, ASK_CAPTION, a.max_length).split()))
        done[c["id"]] = rec
        n += 1
        if n % 20 == 0:
            path.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t0
            print(f"{len(done)}/{len(cands)} {el:.0f}s ({el / n:.1f} s/image, about {(len(cands) - len(done)) * el / n / 3600:.1f} h left)", flush=True)
    path.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
    print("vlm done", len(done))


ASK_STAMP = ("Is there an artist signature, a watermark, a website address or a logo stamped over this picture, not part of the "
             "scene itself? Answer only yes or no.")
ASK_GIBBERISH = "Does this picture contain written text that is not real readable words, such as scribbles or fake letters? Answer only yes or no."
ASK_NUDE = forge.ASK_NUDE


def safety(a):
    """Second pass. (1) The images the first question dropped are asked apart about a stamp (signature, watermark, site, logo)
    and about fake text, so a speech balloon or a printed shirt with real words is not lost with a signature. (2) Every
    image, whatever its prompt says: nudity / underwear / sexual content, and if yes, whether anyone looks under 18 (the
    first pass asked the age only for prompts with explicit words). Resumable; results in vlm2.json."""
    out = Path(a.out)
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    v = json.loads((out / "vlm.json").read_text(encoding="utf-8"))
    path = out / "vlm2.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t0, n = time.time(), 0
    for c in cands:
        if c["id"] in done:
            continue
        img = Image.open(out / "pairs" / f"{c['id']}.png").convert("RGB")
        rec = {}
        if v[c["id"]]["mark"].startswith("yes"):
            rec["stamp"] = forge.textgen(img, ASK_STAMP, 8).strip(".").lower()
            rec["gibberish"] = forge.textgen(img, ASK_GIBBERISH, 8).strip(".").lower()
        rec["nude"] = forge.textgen(img, ASK_NUDE, 8).strip(".").lower()
        if rec["nude"].startswith("yes") and "minor" not in v[c["id"]]:
            rec["minor"] = forge.textgen(img, ASK_AGE, 8).strip(".").lower()
        if (v[c["id"]]["mark"].startswith("yes") and rec["stamp"].startswith("no") and rec["gibberish"].startswith("no")
                and not v[c["id"]].get("caption") and not v[c["id"]].get("minor", "no").startswith("yes")
                and not rec.get("minor", "no").startswith("yes")):
            rec["caption"] = forge.whole_sentences(" ".join(forge.textgen(img, ASK_CAPTION, a.max_length).split()))
        done[c["id"]] = rec
        n += 1
        if n % 50 == 0:
            path.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t0
            print(f"{len(done)}/{len(cands)} {el:.0f}s ({el / n:.1f} s/image, about {(len(cands) - len(done)) * el / n / 3600:.1f} h left)", flush=True)
    path.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
    print("safety done", len(done))


def passing(out):
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    v = json.loads((out / "vlm.json").read_text(encoding="utf-8")) if (out / "vlm.json").exists() else {}
    rejected = set(json.loads((out / "rejected.json").read_text(encoding="utf-8"))) if (out / "rejected.json").exists() else set()
    # the second pass: an image anyone in which looks under 18 with nudity is out; a caption it wrote recovers an image the first
    # question had dropped for real text in the scene
    v2 = json.loads((out / "vlm2.json").read_text(encoding="utf-8")) if (out / "vlm2.json").exists() else {}
    for k, r in v2.items():
        if r.get("minor", "no").startswith("yes"):
            rejected.add(k)
        if r.get("caption") and k in v:
            v[k] = {**v[k], "caption": r["caption"]}
    return [c for c in cands if c["id"] not in rejected and v.get(c["id"], {}).get("caption")], v


def sheets(a):
    out = Path(a.out)
    good, _ = passing(out)
    font = ImageFont.truetype("arial.ttf", 15)
    t, cols, rows_per = 200, 10, 6
    by_block = defaultdict(list)
    for c in good:
        by_block[c["block"]].append(c)
    (out / "sheets").mkdir(exist_ok=True)
    for block, items in by_block.items():
        for s in range(0, len(items), cols * rows_per):
            chunk = items[s:s + cols * rows_per]
            img = Image.new("RGB", (cols * t, ((len(chunk) + cols - 1) // cols) * t), (18, 18, 20))
            d = ImageDraw.Draw(img)
            for i, c in enumerate(chunk):
                im = Image.open(out / "pairs" / f"{c['id']}.png").convert("RGB")
                im.thumbnail((t, t))
                x, y = (i % cols) * t, (i // cols) * t
                img.paste(im, (x + (t - im.width) // 2, y + (t - im.height) // 2))
                d.rectangle((x, y, x + 62, y + 18), fill=(0, 0, 0))
                d.text((x + 3, y + 1), c["id"], fill=(255, 220, 0), font=font)
            img.save(out / "sheets" / f"{block}_{s // (cols * rows_per):03d}.jpg", quality=85)
    print("sheets in", out / "sheets", {b: len(v) for b, v in by_block.items()})


def final(a):
    out = Path(a.out)
    good, v = passing(out)
    quota = {"velvet": round(a.n * 0.15), "anime": round(a.n * 0.55), "realism": round(a.n * 0.20)}
    keep, by_block = [], defaultdict(list)
    for c in good:
        by_block[c["block"]].append(c)
    for block, k in quota.items():
        keep += by_block[block][:k]
    rest = [c for c in good if c not in keep]
    keep += rest[:a.n - len(keep)]  # the reserve fills what is left
    data = [{"id": c["id"], "image": f"pairs/{c['id']}.png", "caption": v[c["id"]]["caption"], "tags": c["orig_prompt"], "model": c["model"],
             "block": c["block"], "traits": c["traits"], "explicit": c["explicit"], "src": c["src"]} for c in keep]
    (out / "dataset.json").write_text(json.dumps(data, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"{len(good)} pass; kept {len(data)}: " + ", ".join(f"{b} {sum(d['block'] == b for d in data)}" for b in ("velvet", "anime", "realism"))
          + f" | explicit {sum(d['explicit'] for d in data)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "prepare", "dedupe", "vlm", "safety", "sheets", "final"])
    ap.add_argument("--inventories", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--oversample", type=float, default=1.25)
    ap.add_argument("--explicit-share", type=float, default=0.15)
    ap.add_argument("--per-model", type=int, default=60, help="at most this many images of one source model")
    ap.add_argument("--allow-young", action="store_true", help="keep non-explicit prompts with young-character words (off by default)")
    ap.add_argument("--dhash-max", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "prepare":
        return forge.prepare(a)
    {"select": select, "dedupe": dedupe, "vlm": vlm, "safety": safety, "sheets": sheets, "final": final}[a.mode](a)


if __name__ == "__main__":
    main()
