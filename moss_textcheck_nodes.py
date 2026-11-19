"""MOSS Text Check: reads the text in a generated image with MOSS-Image's own text encoder (Qwen3.5-9B, whose file holds the
vision tower) and compares it with the text that was asked for: per layout text item inside its box, or a list of expected
texts anywhere in the image. No other model is loaded. Measured (tools/RESULTS.md, MOSS Text Check): 75 of 75 right verdicts on
text drawn in common fonts, 19 of 20 on text MOSS painted itself (the miss: a word cut by the image edge read as whole)."""
import json
import re
import unicodedata

import torch
import comfy.utils

from .moss_common import format_info
from .moss_enhancer_nodes import chat, decode_fallback, fit_image

READ_BOX = ("Transcribe exactly the text written in this picture, with its spelling and capitals; join separate lines with "
            "a space. Answer with the text only. If there is no legible text, answer NONE.")
READ_ALL = ("List every piece of text written in this picture, one per line, exactly as spelled. Answer with the list only. "
            "If there is no legible text, answer NONE.")
SYSTEM = "You read text in images and transcribe it exactly. You never correct spelling and never add text that is not there."
CLOSE = 0.6  # display cutoff between "close" and "missing"; not a measured accuracy


def norm(text):
    """Comparison form: accents kept, case folded, punctuation dropped, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def literal(text):
    """The text as written, only whitespace collapsed: what the `literal` flag of an item compares."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def similarity(expected, read):
    """1 - edit distance / longer length, on the comparison form; 1.0 is an exact match. A text made only of punctuation or
    symbols ("!!!", "?") has an empty comparison form, so it is compared as written; an empty read never matches visible text."""
    a, b = norm(expected), norm(read)
    if not a:
        a, b = literal(expected).casefold(), literal(read).casefold()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def clean_answer(answer):
    answer = re.sub(r"<think>.*?(</think>|$)", "", answer, flags=re.DOTALL).strip().strip("`").strip()
    answer = re.sub(r"^(text|answer)\s*:\s*", "", answer, flags=re.I).strip().strip('"').strip()
    return "" if answer.upper().rstrip(".") == "NONE" else answer


def best_line(expected, lines):
    """(similarity, line) of the read line closest to `expected`; a text split over up to four consecutive read lines (a
    title on three lines, a year set apart) is tried joined too."""
    cands = [" ".join(lines[i:i + n]) for n in range(1, 5) for i in range(len(lines) - n + 1)]
    return max(((similarity(expected, c), c) for c in cands), default=(0.0, ""))


def verdict(sim):
    return "exact" if sim >= 0.9999 else "close" if sim >= CLOSE else "missing"


def layout_items(layout):
    """(canvas width, height, [(text, (x1, y1, x2, y2))]) of a layout draft's text items; None for a caption draft."""
    draft = json.loads(layout)
    if not isinstance(draft, dict) or "items" not in draft:
        return None
    w, h = draft.get("width"), draft.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError("MOSS Text Check: the layout has no width / height")
    out = []
    for item in draft["items"]:
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
            box = item.get("bbox")
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
                raise ValueError(f"MOSS Text Check: text item {item['text']!r} has no xyxy bbox")
            out.append((item["text"].strip(), tuple(float(v) for v in box)))
    return w, h, out


QUOTED = re.compile(r'"([^"\n]{1,80})"|“([^”\n]{1,80})”|«([^»\n]{1,80})»')


def quoted_texts(draft):
    """The texts in quotes of a caption draft ({"caption": ...}, as the Prompt Enhancer writes in caption / detailed mode): how a
    MOSS-Image prompt asks for visible text (`a sign that reads "OPEN"`). [] for anything else."""
    try:
        data = json.loads(draft)
    except (json.JSONDecodeError, TypeError):
        return []
    caption = data.get("caption") if isinstance(data, dict) else None
    if not isinstance(caption, str):
        return []
    found = [next(g for g in m.groups() if g is not None).strip() for m in QUOTED.finditer(caption)]
    return list(dict.fromkeys(f for f in found if any(c.isalnum() for c in f)))


def crop_box(image, box, canvas, margin=0.12, min_side=384, max_side=2048):
    """The box (canvas pixels) on the image, grown by `margin` of its size, upscaled so its short side is at least
    `min_side` (its long side at most `max_side`); returns (crop, (x1, y1, x2, y2) in image pixels)."""
    ih, iw = int(image.shape[1]), int(image.shape[2])
    sx, sy = iw / canvas[0], ih / canvas[1]
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(iw, int(round(x2 + mx))), min(ih, int(round(y2 + my)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"MOSS Text Check: box {box} is outside the image")
    crop = image[:1, y1:y2, x1:x2, :3]
    short, long = min(x2 - x1, y2 - y1), max(x2 - x1, y2 - y1)
    if short < min_side:
        # a thin box is upscaled at most until its long side is max_side: the reader resizes to about 1 MP anyway, and an unbounded
        # factor made a 1024 x 4 box a 98304 x 384 tensor (audit 0.8.1)
        k = max(1.0, min(min_side / short, max_side / long))
        crop = comfy.utils.common_upscale(crop.movedim(-1, 1), round((x2 - x1) * k), round((y2 - y1) * k), "bicubic",
                                          "disabled").movedim(1, -1).clamp(0.0, 1.0)
    return crop, (x1, y1, x2, y2)


def read(clip, image, question, max_tokens):
    tokens = clip.tokenize(chat(SYSTEM, question, 1), images=[fit_image(image)])
    with decode_fallback():
        ids = clip.generate(tokens, do_sample=False, max_length=max_tokens, temperature=0.01, top_k=1, top_p=1.0, min_p=0.0,
                            repetition_penalty=1.0, seed=0)
    return clean_answer(clip.decode(ids))


def check(clip, image, layout="", expected_text="", max_tokens=96, reader=read):
    """Report rows and a mask of the boxes that failed. `reader(clip, image, question, max_tokens)` -> answer text."""
    image = image[:1]
    ih, iw = int(image.shape[1]), int(image.shape[2])
    mask = torch.zeros(1, ih, iw)
    rows, boxes = [], []
    parsed = layout_items(layout) if layout and layout.strip() else None
    if parsed:
        cw, chh, items = parsed
        for text, box in items:
            crop, pix = crop_box(image, box, (cw, chh))
            got = reader(clip, crop, READ_BOX, max_tokens)
            # the box may hold other text too: the best read line (or the whole answer) counts
            sim, got = max(best_line(text, [ln.strip() for ln in got.splitlines() if ln.strip()]), (similarity(text, got), got))
            rows.append({"text": text, "read": got, "similarity": round(sim, 3), "verdict": verdict(sim), "box": list(box),
                         "literal": literal(text) == literal(got)})
            boxes.append(pix)
    expected = [t.strip() for t in expected_text.splitlines() if t.strip()]
    if not parsed and layout and layout.strip():  # a caption draft: its quoted texts are the texts asked for
        expected += [q for q in quoted_texts(layout) if q not in expected]
    if not rows and not expected:  # no text asked for: nothing to read, and the workflow goes on
        return [], mask, []
    whole = reader(clip, image, READ_ALL, max_tokens * 2) if (expected or any(r["verdict"] != "exact" for r in rows)) else ""
    lines = [ln.strip(" -*•\t") for ln in whole.splitlines() if ln.strip(" -*•\t")]
    for r, pix in zip(rows, boxes):
        if r["verdict"] != "exact":
            sim, line = best_line(r["text"], lines)
            if r["verdict"] == "missing" and verdict(sim) != "missing":  # not in its box, but somewhere else
                r.update(verdict="outside its box", read_elsewhere=line, similarity_elsewhere=round(sim, 3))
            x1, y1, x2, y2 = pix
            mask[0, y1:y2, x1:x2] = 1.0
    for text in expected:
        sim, line = best_line(text, lines)
        rows.append({"text": text, "read": line, "similarity": round(sim, 3), "verdict": verdict(sim), "box": None,
                     "literal": literal(text) == literal(line)})
    return rows, mask, lines


class HaDeSMossTextCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The generated image (the first of a batch is checked)."}),
                "clip": ("CLIP", {"tooltip": "MOSS-Image's text encoder (CLIPLoader, type moss_image): its file holds the vision tower "
                                             "that reads the image. The same CLIP that encodes the prompt; no other model is loaded."}),
            },
            "optional": {
                "layout": ("STRING", {"forceInput": True, "tooltip": "Layout draft JSON (Prompt Enhancer, Layout Editor or Layout "
                                                                      "Compiler): each text item is read inside its box, scaled from the "
                                                                      "layout canvas to the image. A caption draft (Prompt Enhancer in "
                                                                      "caption / detailed mode): the texts in quotes are looked for anywhere."}),
                "expected_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Texts to look for anywhere in the image, one "
                                                                                            "per line (for prompts without a layout)."}),
                "max_tokens": ("INT", {"default": 96, "min": 16, "max": 512, "tooltip": "Longest answer per reading, in tokens."}),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("report", "failed_mask", "passed", "info")
    OUTPUT_TOOLTIPS = ("One line per text: what was asked, what was read, the verdict.",
                       "The layout boxes whose text is not exact (white), for a retry of those areas only.",
                       "True when every text is read exactly.",
                       "The same results for Suite Telemetry.")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Reads the text in the image with MOSS-Image's own text encoder and compares it with the text asked for: each "
                   "layout text item inside its box, or expected texts anywhere. Verdicts: exact (letters and digits, case and punctuation "
                   "folded; the report says when the literal text differs), close (some letters differ), missing, or outside its box. Measured: 75 of 75 right verdicts on text drawn in common fonts (28-110 px), 19 of 20 on "
                   "text MOSS painted itself; a word cut by the image edge can be read as whole (tools/RESULTS.md).")

    def run(self, image, clip, layout="", expected_text="", max_tokens=96):
        if not hasattr(clip, "generate"):
            raise ValueError("MOSS Text Check: this CLIP cannot generate text; use MOSS-Image's text encoder (type moss_image)")
        rows, mask, lines = check(clip, image, layout or "", expected_text or "", max_tokens)
        passed = all(r["verdict"] == "exact" for r in rows)
        if not rows:
            text = ("HaDeS MOSS Text Check | nothing to check: no text item in the layout, no quoted text in the caption draft, "
                    "no expected_text")
            info = format_info(text, "text_check", passed=True, items=[], read_whole=[], nothing_to_check=True)
            return {"ui": {"text": [text]}, "result": (text, mask, True, info)}
        # exact compares letters and digits with case and punctuation folded; the report says when the literal text differs
        report = "\n".join(f"[{r['verdict']}] {r['text']!r} -> read {r['read']!r} ({r['similarity']:.2f})"
                           + ("; case or punctuation differ" if r["verdict"] == "exact" and not r.get("literal", True) else "")
                           + (f"; found elsewhere: {r['read_elsewhere']!r}" if "read_elsewhere" in r else "") for r in rows)
        n_ok = sum(r["verdict"] == "exact" for r in rows)
        text = f"HaDeS MOSS Text Check | {n_ok}/{len(rows)} exact | " + ("passed" if passed else "not passed") + "\n" + report
        info = format_info(text, "text_check", passed=passed, items=rows, read_whole=lines)
        return {"ui": {"text": [text]}, "result": (report, mask, passed, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossTextCheck": HaDeSMossTextCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossTextCheck": "MOSS Text Check"}
