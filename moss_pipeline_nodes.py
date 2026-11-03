import json
import math

import torch
import comfy.model_management

from .moss_common import format_info

# width : height; the size follows from the megapixels, snapped to the 16 px latent grid
ASPECTS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9), "9:21": (9, 21)}
LATENT_CHANNELS = 128  # FLUX.2 latent, 16x downscale (latent_formats.Flux2)


def snap16(v):
    return max(16, int(round(v / 16.0)) * 16)


class HaDeSMossLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(ASPECTS) + ["custom"], {"default": "1:1"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                         "tooltip": "Image area, 1.0 = 1024x1024. Measured on square images: 768 (0.56) to 2048 (4.0) all render clean at "
                                                    "the default shift 5.0, larger ones show more scene around the subject (tools/RESULTS.md). "
                                                    "Other aspect ratios are not measured."}),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Used with aspect_ratio custom; snapped to 16 px."}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Used with aspect_ratio custom; snapped to 16 px."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Optional: checks that it is a 128-channel FLUX.2 VAE, the one MOSS-Image uses."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "info")
    FUNCTION = "generate"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = "Empty MOSS-Image latent (128 channels, 16 px grid) from an aspect ratio and megapixels; width / height out for layout prompts."

    def generate(self, aspect_ratio, megapixels, custom_width, custom_height, batch_size, vae=None):
        if vae is not None and getattr(vae, "latent_channels", LATENT_CHANNELS) != LATENT_CHANNELS:
            raise ValueError(f"HaDeS MOSS Latent: the VAE has {vae.latent_channels} latent channels; MOSS-Image needs the 128-channel "
                             f"FLUX.2 VAE (flux2-vae / flux2-vae_small).")
        if aspect_ratio == "custom":
            w, h = snap16(custom_width), snap16(custom_height)
        else:
            aw, ah = ASPECTS[aspect_ratio]
            unit = math.sqrt(megapixels * 1024 * 1024 / (aw * ah))
            w, h = snap16(aw * unit), snap16(ah * unit)
        latent = torch.zeros([batch_size, LATENT_CHANNELS, h // 16, w // 16], device=comfy.model_management.intermediate_device())
        mp = w * h / (1024 * 1024)
        text = f"HaDeS MOSS Latent | {w}x{h} ({mp:.2f} MP, {aspect_ratio}) | batch {batch_size} | {(w // 16) * (h // 16)} image tokens"
        if not 0.55 <= mp <= 4.05:
            text += "\nOutside the measured range (768x768 to 2048x2048)."
        return ({"samples": latent}, w, h, format_info(text, "latent", width=w, height=h, batch=batch_size))


# ==============================================================================
# LAYOUT COMPILER
# ==============================================================================
# The draft schema and its compilation follow OpenMOSS' skills/moss-image-prompt-enhancer (scripts/compile_prompt.py and
# references/layout.md, Apache-2.0): absolute top-left xyxy pixel boxes in, centered yxyx boxes out, width / height dropped.
_SUMMARY_FIELDS = {"caption", "meta", "canvas", "palette"}
_ITEM_FIELDS = {"label", "caption", "bbox", "text", "tags", "palette"}
_DRAFT_FIELDS = {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}

EXAMPLE_DRAFT = json.dumps({
    "width": 1024, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
    "summary": {"canvas": "single", "caption": "Minimalist concert poster on off-white paper, a red sun behind a black mountain silhouette, bold sans-serif type."},
    "items": [
        {"label": "object", "caption": "Large red sun behind a black mountain silhouette.", "bbox": [160, 64, 864, 544]},
        {"label": "text", "caption": "Huge bold black condensed title on two lines.", "text": "NORTHERN\nLIGHTS", "bbox": [192, 576, 832, 960]},
    ],
}, ensure_ascii=False, indent=2)


def _check(condition, message):
    if not condition:
        raise ValueError(f"HaDeS MOSS Layout: {message}")


def _fields(value, allowed, required, where):
    _check(isinstance(value, dict), f"{where} must be an object")
    _check(not value.keys() - allowed, f"{where}: unknown fields {sorted(value.keys() - allowed)}")
    _check(not required - value.keys(), f"{where}: missing fields {sorted(required - value.keys())}")


def _text(value, where):
    _check(isinstance(value, str) and value.strip(), f"{where} must be nonempty text")


def _strings(value, where):
    _check(isinstance(value, list), f"{where} must be an array")
    for i, v in enumerate(value):
        _text(v, f"{where}[{i}]")


def centered_box(box, width, height, where):
    """[x1, y1, x2, y2] top-left pixels -> [y1, x1, y2, x2] around the canvas center; edges floored / ceiled outward."""
    _check(isinstance(box, list) and len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box),
           f"{where} must be four numbers [x1, y1, x2, y2]")
    x1, y1, x2, y2 = box
    _check(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"{where} {box} must lie inside the {width}x{height} canvas with x1 < x2 and y1 < y2")
    return [math.floor(round(y1, 9)) - height // 2, math.floor(round(x1, 9)) - width // 2,
            math.ceil(round(y2, 9)) - height // 2, math.ceil(round(x2, 9)) - width // 2]


def compile_layout(draft):
    """Draft dict -> (model prompt, width, height, text items as (text, box width, box height)); a caption draft has no canvas."""
    if "caption" in draft:
        _fields(draft, {"caption"}, {"caption"}, "caption draft")
        _text(draft["caption"], "caption")
        return draft["caption"], None, None, []
    _fields(draft, _DRAFT_FIELDS, _DRAFT_FIELDS, "layout draft")
    width, height = draft["width"], draft["height"]
    for name, v in (("width", width), ("height", height)):
        _check(type(v) is int and v > 0 and v % 16 == 0, f"{name} must be a positive multiple of 16")
    _check(draft["bbox_coord_space"] == "abs", "bbox_coord_space must be \"abs\" (top-left pixels)")
    _check(draft["bbox_format"] == "xyxy", "bbox_format must be \"xyxy\"")
    summary = draft["summary"]
    _fields(summary, _SUMMARY_FIELDS, {"caption"}, "summary")
    _text(summary["caption"], "summary.caption")
    if "canvas" in summary:
        _check(summary["canvas"] in ("single", "multi_panel", "collage"), "summary.canvas must be single, multi_panel or collage")
    for key in ("meta", "palette"):
        if key in summary:
            _strings(summary[key], f"summary.{key}")
    _check(isinstance(draft["items"], list) and draft["items"], "items must be a nonempty array")
    items, texts = [], []
    for i, item in enumerate(draft["items"]):
        where = f"items[{i}]"
        _fields(item, _ITEM_FIELDS, {"label", "caption", "bbox"}, where)
        _check(item["label"] in ("object", "character", "text"), f"{where}.label must be object, character or text")
        _text(item["caption"], f"{where}.caption")
        _check(item["label"] != "text" or "text" in item, f"{where}: text items need the literal text")
        if "text" in item:
            _text(item["text"], f"{where}.text")
        for key in ("tags", "palette"):
            if key in item:
                _strings(item[key], f"{where}.{key}")
        items.append({k: centered_box(v, width, height, f"{where}.bbox") if k == "bbox" else v for k, v in item.items()})
        if "text" in item:
            x1, y1, x2, y2 = item["bbox"]
            texts.append((item["text"], x2 - x1, y2 - y1))
    prompt = json.dumps({"summary": summary, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"},
                        ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return prompt, width, height, texts


class HaDeSMossLayoutCompiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "draft": ("STRING", {"multiline": True, "default": EXAMPLE_DRAFT,
                                     "tooltip": "OpenMOSS layout draft: width / height (multiples of 16), bbox_coord_space \"abs\", bbox_format \"xyxy\" "
                                                "(top-left pixels), summary {caption, meta, canvas, palette} and flat items {label, caption, bbox, "
                                                "text, tags, palette}. Line breaks in text as \\n. A {\"caption\": ...} draft passes through as plain text."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "latent", "width", "height", "info")
    FUNCTION = "compile"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Compiles an OpenMOSS layout draft into the model-facing MOSS-Image prompt (centered yxyx boxes) and the empty latent of the "
                   "draft's own canvas, so prompt and image size cannot drift apart.")

    def compile(self, draft, batch_size):
        def unique(pairs):
            keys = [k for k, _ in pairs]
            duplicates = sorted({k for k in keys if keys.count(k) > 1})
            _check(not duplicates, f"duplicate JSON key {duplicates[0]!r}" if duplicates else "")
            return dict(pairs)
        try:
            data = json.loads(draft, object_pairs_hook=unique)
        except json.JSONDecodeError as e:
            raise ValueError(f"HaDeS MOSS Layout: the draft is not valid JSON ({e}).") from e
        _check(isinstance(data, dict), "the draft must be a JSON object")
        prompt, width, height, texts = compile_layout(data)
        if width is None:
            width = height = 1024
            text = "HaDeS MOSS Layout | caption draft, passed through as plain text | latent 1024x1024"
        else:
            text = f"HaDeS MOSS Layout | {width}x{height} | {len(data['items'])} items, {len(texts)} with text | prompt {len(prompt)} chars"
            multi = [(t, w, h) for t, w, h in texts if "\n" in t]
            if multi:
                text += ("\nMulti-line text (check that the box fits the lines: a two-line title in an 832x192 box came out on one line in "
                         "3 of 6 seeds, in a 640x384 box on two lines in 6 of 6, tools/RESULTS.md):")
                text += "".join(f"\n  {t.count(chr(10)) + 1} lines in a {w:g}x{h:g} box: {t[:40]!r}" for t, w, h in multi)
        latent = torch.zeros([batch_size, LATENT_CHANNELS, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return (prompt, {"samples": latent}, width, height, format_info(text, "layout", width=width, height=height, items=len(texts)))


NODE_CLASS_MAPPINGS = {"HaDeSMossLatent": HaDeSMossLatent, "HaDeSMossLayoutCompiler": HaDeSMossLayoutCompiler}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLatent": "MOSS Latent", "HaDeSMossLayoutCompiler": "MOSS Layout Compiler"}
