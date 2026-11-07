"""MOSS Prompt Enhancer: MOSS-Image's own text encoder (Qwen3.5-9B, loaded as the moss_image CLIP) writes the prompt.

The instructions are adapted from OpenMOSS' skills/moss-image-prompt-enhancer (SKILL.md and references/layout.md,
Apache-2.0, https://github.com/OpenMOSS/MOSS-Image): a caption, or a pixel Layout draft that the Layout Compiler of this
suite compiles. The same CLIP that encodes the prompt for MOSS-Image writes it, so no other model is loaded; with an image
input it also sees the image (the text encoder file holds the vision tower).
"""
import contextlib
import json
import re

import torch
import comfy.utils

from .moss_common import format_info
from .moss_pipeline_nodes import compile_layout, snap16

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
MODES = ["caption", "layout"]

_COMMON = """You turn the user's idea into a concrete description for the MOSS-Image text-to-image model.

Preserve the user's subjects, counts, relationships, medium, style, exclusions and intended visible wording. Resolve
ordinary creative gaps yourself. Do not invent factual specifications as if the user supplied them.

Write descriptive prose in English unless the user asks for another language. Keep visible text verbatim in its original
language, with its punctuation, capitalization and line breaks. A quoted sentence in an instruction is not necessarily
text to render: tell visible wording apart from constraints.

Describe visible choices: subject and action, relevant appearance, setting, camera and framing, medium, lighting,
materials, palette and mood. Add detail where it makes the image more specific. Avoid quality tags (masterpiece, 8k,
best quality), unrequested style changes and notes about the process. Keep a minimalist brief minimalist.

For designed images (posters, covers, signs) write the actual copy, never placeholders such as "headline here", and keep
the text readable at the chosen size."""

_CAPTION = """

Answer with one JSON object and nothing else: {"caption": "..."}. The caption is cohesive prose, spatial relations in
words, any visible wording quoted exactly. No coordinates or layout fields."""

_LAYOUT = """

Answer with one JSON object and nothing else, a pixel Layout draft:
{"width": W, "height": H, "bbox_coord_space": "abs", "bbox_format": "xyxy",
 "summary": {"caption": "...", "meta": ["..."], "canvas": "single"},
 "items": [{"label": "object", "caption": "...", "bbox": [x_min, y_min, x_max, y_max]},
           {"label": "text", "caption": "...", "text": "...", "bbox": [x_min, y_min, x_max, y_max]}]}

- width and height are given below; use them unchanged.
- summary.caption: the whole image (medium, setting, lighting, palette, global treatment). canvas is single,
  multi_panel or collage. meta is optional: a few medium or genre tags.
- items: a flat array, the most important first. label is object, character or text. caption describes the item's
  visible appearance; for text, its typography and surface. text items carry the exact visible characters in "text",
  with \\n for intended line breaks.
- bbox: actual pixels from the top-left corner, x in 0..W and y in 0..H. Boxes may overlap. A box is a placement, not a
  border to draw.
- Make every box big enough for its text at that size: a long line needs a wide box, two lines a taller one.
- No other fields."""

_IMAGE = """

An image is attached. With no written idea it is the whole brief: describe it so the model can recreate it. With an
idea, the image is the starting point and the idea wins wherever they disagree."""


def instructions(mode, has_image):
    return _COMMON + (_LAYOUT if mode == "layout" else _CAPTION) + (_IMAGE if has_image else "")


def user_message(idea, mode, width, height, negative):
    parts = [f"Idea:\n{idea.strip()}" if idea.strip() else "No written idea: use the image."]
    if negative.strip():
        parts.append(f"Keep out of the image:\n{negative.strip()}")
    if mode == "layout":
        parts.append(f"Canvas: width {width}, height {height}.")
    return "\n\n".join(parts)


def chat(system, user, n_images):
    # A string that starts with <|im_start|> skips the MOSS tokenizer's own conditioning template; the empty think block
    # turns the model's reasoning off, which would only spend tokens here
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{VISION_BLOCK * n_images}{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def first_json(text):
    """The first JSON object in the model's answer (code fences and stray prose around it ignored), or None."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_caption(text):
    obj = first_json(text)
    if obj and isinstance(obj.get("caption"), str) and obj["caption"].strip():
        return obj["caption"].strip()
    # no JSON: the answer itself, without a leading "Caption:" label or quotes
    plain = re.sub(r"^\W*(caption|prompt)\W*:\s*", "", text.strip(), flags=re.IGNORECASE)
    return plain.strip().strip('"').strip()


def fit_draft(obj, width, height):
    """The model's layout draft on the requested canvas: width / height set, boxes rounded to integers and clamped into the
    canvas (a box outside it would fail the compiler's bounds check), keys the compiler does not know dropped, items with
    no usable box left out. Returns (draft, notes)."""
    notes = []
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    summary = {k: v for k, v in summary.items() if k in ("caption", "meta", "canvas", "palette")}
    if not str(summary.get("caption", "")).strip():
        summary["caption"] = str(obj.get("caption", "")).strip() or "An image."
        notes.append("no summary caption in the answer")
    items = []
    for i, it in enumerate(obj.get("items") or []):
        if not isinstance(it, dict):
            continue
        box = it.get("bbox") or it.get("bbox_2d")
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            notes.append(f"item {i} has no box, left out")
            continue
        x0, y0, x1, y1 = box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        clamped = [max(0, min(width, round(x0))), max(0, min(height, round(y0))), max(0, min(width, round(x1))), max(0, min(height, round(y1)))]
        if clamped != [round(x0), round(y0), round(x1), round(y1)]:
            notes.append(f"item {i} box clamped into the canvas")
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            notes.append(f"item {i} box empty after clamping, left out")
            continue
        label = it.get("label") if it.get("label") in ("object", "character", "text") else "object"
        item = {"label": label, "caption": str(it.get("caption", "")).strip() or str(it.get("text", "")).strip() or label, "bbox": clamped}
        if label == "text":
            if not str(it.get("text", "")).strip():
                notes.append(f"text item {i} has no text, kept as object")
                item["label"] = "object"
            else:
                item["text"] = str(it["text"])
        for k in ("tags", "palette"):
            if isinstance(it.get(k), list):
                item[k] = [str(v) for v in it[k] if str(v).strip()]
        items.append(item)
    if not items:
        raise ValueError("MOSS Prompt Enhancer: the layout answer has no usable items")
    return {"width": width, "height": height, "bbox_coord_space": "abs", "bbox_format": "xyxy", "summary": summary, "items": items}, notes


@contextlib.contextmanager
def decode_fallback():
    """With ComfyUI 0.43 / comfy-kitchen 0.2.40 the native GQA decode kernel refuses this text encoder's tensors
    ("Unsupported Flash Attention tensor strides") and text generation stops. While this node generates, a refused call
    runs comfy-kitchen's own torch implementation of the same attention instead. Nothing changes where the kernel works."""
    try:
        import comfy_kitchen
        import comfy_kitchen.flash_attention as fa
        native = comfy_kitchen.flash_attention_decode_gqa
        torch_impl = fa._decode_gqa_torch
    except (ImportError, AttributeError):
        yield
        return
    state = {"fallback": False}

    def decode(q, k, v, kv_lengths, return_lse=False, causal=True):
        if not state["fallback"]:
            try:
                return native(q, k, v, kv_lengths, return_lse=return_lse, causal=causal)
            except RuntimeError as e:
                if "strides" not in str(e):
                    raise
                state["fallback"] = True
        out, lse = torch_impl(q, k, v, kv_lengths, causal)
        return (out, lse) if return_lse else out
    comfy_kitchen.flash_attention_decode_gqa = decode
    try:
        yield state
    finally:
        comfy_kitchen.flash_attention_decode_gqa = native


def fit_image(image, max_pixels=1024 * 1024):
    """The first image of the batch at most ~1 MP: the vision tower would otherwise turn a large image into many tokens."""
    image = image[:1]
    h, w = int(image.shape[1]), int(image.shape[2])
    if h * w <= max_pixels:
        return image
    k = (max_pixels / float(h * w)) ** 0.5
    return comfy.utils.common_upscale(image.movedim(-1, 1), max(28, int(w * k)), max(28, int(h * k)), "area", "disabled").movedim(1, -1)


class HaDeSMossPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MOSS-Image's text encoder (CLIPLoader, type moss_image): the same Qwen3.5-9B that encodes the prompt "
                                             "writes it here, so no other model is loaded."}),
                "idea": ("STRING", {"multiline": True, "default": "", "tooltip": "What you want, in your own words. May be empty when an image is connected."}),
                "mode": (MODES, {"default": "caption", "tooltip": "caption: a detailed paragraph for CLIPTextEncode. layout: a pixel Layout draft "
                                                                   "(boxes for objects and text), compiled here; its draft output also feeds the "
                                                                   "Layout Compiler for the matching latent."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Layout mode: the canvas the boxes are drawn on. "
                                                                                                    "Use the same size for the image."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05,
                                          "tooltip": "0 always writes the same answer for the same input. Higher values invent more of what the "
                                                     "idea leaves open."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096,
                                       "tooltip": "Length limit of the answer. A layout with many items needs more; a cut answer cannot be parsed."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional reference the model sees: the whole brief when the idea is empty, otherwise the starting "
                                               "point (the idea wins). Resized to about 1 MP."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Things to keep out; told to the model, not encoded."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "draft", "width", "height", "info")
    FUNCTION = "enhance"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Writes a detailed MOSS-Image prompt from a short idea and / or an image, with MOSS-Image's own text encoder and instructions "
                   "adapted from OpenMOSS' official prompt enhancer.")

    def enhance(self, clip, idea, mode, width, height, temperature, seed, max_tokens, image=None, negative=""):
        if not idea.strip() and image is None:
            raise ValueError("MOSS Prompt Enhancer: write an idea, connect an image, or both")
        if not hasattr(clip, "generate"):
            raise ValueError("MOSS Prompt Enhancer: this CLIP cannot generate text; use MOSS-Image's text encoder (type moss_image)")
        width, height = snap16(width), snap16(height)
        images = [fit_image(image)] if image is not None else []
        text = chat(instructions(mode, bool(images)), user_message(idea, mode, width, height, negative or ""), len(images))
        tokens = clip.tokenize(text, images=images) if images else clip.tokenize(text)
        with decode_fallback() as state:
            ids = clip.generate(tokens, do_sample=temperature > 0, max_length=max_tokens, temperature=max(temperature, 0.01),
                                top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=seed)
        answer = clip.decode(ids)
        fallback = bool(state and state["fallback"])

        notes = []
        if mode == "caption":
            prompt = parse_caption(answer)
            draft = json.dumps({"caption": prompt}, ensure_ascii=False)
        else:
            obj = first_json(answer)
            if obj is None:
                raise ValueError("MOSS Prompt Enhancer: no JSON layout in the answer (raise max_tokens if it was cut). Answer starts: "
                                 + answer[:400])
            draft_obj, notes = fit_draft(obj, width, height)
            draft = json.dumps(draft_obj, ensure_ascii=False, indent=2)
            prompt = compile_layout(draft_obj)[0]
        info = format_info(f"HaDeS MOSS Prompt Enhancer | {mode}" + (f" | {width}x{height}" if mode == "layout" else "")
                           + (" | image" if images else "") + f" | {len(prompt)} chars"
                           + (" | torch decode fallback" if fallback else "") + "".join(f"\nnote: {n}" for n in notes),
                           "prompt_enhancer", mode=mode, width=width, height=height, image=bool(images), decode_fallback=fallback, notes=notes)
        return {"ui": {"text": [prompt]}, "result": (prompt, draft, width, height, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossPromptEnhancer": HaDeSMossPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossPromptEnhancer": "MOSS Prompt Enhancer"}
