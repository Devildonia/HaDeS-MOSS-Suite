"""MOSS Prompt Enhancer: MOSS-Image's own text encoder (Qwen3.5-9B, loaded as the moss_image CLIP) writes the prompt.

The instructions are adapted from OpenMOSS' skills/moss-image-prompt-enhancer (SKILL.md and references/layout.md,
Apache-2.0, https://github.com/OpenMOSS/MOSS-Image): a caption, or a pixel Layout draft that the Layout Compiler of this
suite compiles. The same CLIP that encodes the prompt for MOSS-Image writes it, so no other model is loaded; with an image
input it also sees the image (the text encoder file holds the vision tower).
"""
import contextlib
import json
import random
import re

import torch
import comfy.utils
import node_helpers

from .moss_common import format_info
from .moss_pipeline_nodes import compile_layout, snap16

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
MODES = ["caption", "detailed", "layout", "as written"]
# The negative prompt of the official ComfyUI MOSS-Image templates, used in every measurement in tools/RESULTS.md
DEFAULT_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

_INTRO = "You turn the user's idea into a concrete description for the MOSS-Image text-to-image model.\n\n"

_PRESERVE = """Preserve the user's subjects, counts, relationships, medium, style, exclusions and intended visible wording. Resolve
ordinary creative gaps yourself. Do not invent factual specifications as if the user supplied them.

"""

_SHARED = """The idea may be a list of Danbooru-style tags: turn them into prose (underscores are spaces). A count tag such as 1girl,
1boy, 2girls or multiple_boys gives only the number and sex of the characters, not their age: write each as an adult,
"a young woman" or "a young man", never "girl", "boy" or "child", unless the idea itself states an age.

Write descriptive prose in English unless the user asks for another language. Keep visible text verbatim in its original
language, with its punctuation, capitalization and line breaks. A quoted sentence in an instruction is not necessarily
text to render: tell visible wording apart from constraints.

"""

_DESCRIBE = """Describe visible choices: subject and action, relevant appearance, setting, camera and framing, medium, lighting,
materials, palette and mood. Add detail where it makes the image more specific. Avoid quality tags (masterpiece, 8k,
best quality), unrequested style changes and notes about the process. Keep a minimalist brief minimalist.

"""

# detailed mode: free to expand the idea, unlike caption, which keeps to it
_EXPAND = """Expand the idea into a long, richly detailed prompt of about 120 to 200 words. Keep its main subject, its medium
and style, any counts and exclusions it gives, and its visible wording. Beyond that you may add: setting, secondary
figures and objects, props, weather, time of day, a small story moment, textures and materials that fit the idea.

The medium comes first and never changes: when the idea names one (anime, manga, illustration, painting, poster, comic,
3D render, pixel art...), begin the prompt with it and keep every detail in that medium. Only when the idea names no
medium, choose the one that fits it best.

Write it like a photographer or an illustrator briefing a shoot. Say where the light comes from and what it does. For a
photograph give the shot type, lens and aperture and the genre (documentary, editorial, product, street...). For any
other medium name the technique and the materials (brush, paper, ink, clay, screen print, cel shading, line art...) and
use no camera or lens words. Prefer concrete,
specific and slightly imperfect details (wear, dust, grain, stains, uneven surfaces) over generic praise. Avoid quality
tags (masterpiece, 8k, best quality, ultra detailed), empty words (stunning, breathtaking, epic) and notes about the
process.

"""

_DESIGN = """For designed images (posters, covers, signs) write the actual copy, never placeholders such as "headline here", and keep
the text readable at the chosen size."""

_COMMON = _INTRO + _PRESERVE + _SHARED + _DESCRIBE + _DESIGN
_DETAILED = _INTRO + _SHARED + _EXPAND + _DESIGN

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


# detailed mode with no idea and no image: the seed draws a medium, a subject area and a place / time, so prompts vary by
# construction instead of drifting to the model's favourite scenes
SURPRISE_MEDIA = [
    "documentary photograph", "studio portrait photograph", "street photograph", "product photograph", "food photograph",
    "wildlife photograph", "aerial photograph", "cinematic film still", "black and white film photograph", "oil painting",
    "watercolor painting", "Chinese ink wash painting", "ukiyo-e woodblock print", "pen and ink illustration",
    "children's book illustration", "anime key visual", "black and white manga page", "Franco-Belgian comic panel",
    "American comic book cover", "3D animated film still", "claymation scene", "pixel art", "low-poly 3D illustration",
    "vintage travel poster", "scientific illustration plate", "flat infographic poster",
]
SURPRISE_SUBJECTS = [
    "a craft or trade at work", "a market or a small shop", "a meal being cooked or shared", "an animal in its habitat",
    "a sport or a game", "a festival or celebration", "a quiet moment at home", "a vehicle or a machine",
    "architecture or an interior", "a landscape in dramatic weather", "a city at night", "life by the sea or a river",
    "a musician or a performance", "a science or space scene", "a myth or a fairy tale", "an everyday object up close",
    "a group of friends or a family", "an elderly person's daily life", "a workplace", "a journey",
]
SURPRISE_PLACES = [
    "a mountain village in the Andes", "Paris in the 1950s", "present-day Lagos", "a Nordic island in winter", "medieval Japan",
    "a desert town in Morocco", "New York in the 1970s", "a rainforest in Borneo", "present-day Seoul", "the Scottish coast",
    "an imagined far future", "a Mediterranean fishing town", "the Australian outback", "Shanghai in the 1920s",
    "a Mexican city during a festival", "Victorian London", "the Mongolian steppe", "an Indian city in the monsoon",
    "a Canadian forest in autumn", "an imagined fantasy kingdom",
]

_SURPRISE = """

No idea is given: invent one original image inside the medium, subject area and place given by the user message. Pick a
specific, unexpected scene there, not the most obvious one. Vary the people: different ages (adults, or a family), sexes,
ethnicities and builds; no lone young woman posing for the camera unless the subject area is a portrait."""


def surprise_brief(seed):
    rng = random.Random(seed)
    return rng.choice(SURPRISE_MEDIA), rng.choice(SURPRISE_SUBJECTS), rng.choice(SURPRISE_PLACES)


def instructions(mode, has_image, surprise=False):
    return ((_DETAILED if mode == "detailed" else _COMMON) + (_LAYOUT if mode == "layout" else _CAPTION) + (_IMAGE if has_image else "")
            + (_SURPRISE if surprise else ""))


def user_message(idea, mode, width, height, negative, brief=None):
    if brief:
        parts = ["No idea: invent one.\nMedium: {}\nSubject area: {}\nPlace and time: {}".format(*brief)]
    else:
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
    if obj and isinstance(obj.get("caption"), str):
        return obj["caption"].strip()  # may be empty: the caller rejects it
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
    """On PyTorch 2.8.0+cu129 the native GQA decode kernel of comfy-kitchen 0.2.40 refuses every input, even plain contiguous
    tensors ("Unsupported Flash Attention tensor strides"; ComfyUI warns that its CUDA operations need cu130+), and text
    generation stops. While this node generates, a refused call
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


def encode(clip, text):
    """CLIPTextEncode's conditioning for `text`."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


# edit LoRAs: the core's own table (reference method, and whether the text encoder also sees the image), so both stay in step
try:
    from comfy_extras.nodes_moss_image import EDIT_FORMATS
except ImportError:  # an older core without MOSS-Image edit support
    EDIT_FORMATS = {}
EDIT_OFF = "off"


def edit_reference(image, megapixels=1.0):
    """The reference at about `megapixels`, sides snapped to 16 px with its aspect kept: the size the edited image is generated at."""
    h, w = int(image.shape[1]), int(image.shape[2])
    k = (megapixels * 1024 * 1024 / float(h * w)) ** 0.5
    nw, nh = max(256, snap16(w * k)), max(256, snap16(h * k))
    return comfy.utils.common_upscale(image[:1, :, :, :3].movedim(-1, 1), nw, nh, "lanczos", "center").movedim(1, -1).clamp(0.0, 1.0), nw, nh


def encode_edit(clip, text, ref, latent, edit_format):
    """TextEncodeMOSSImageEdit's conditioning: ai-toolkit formats also show the reference (at most 384x384) to the text encoder after a
    "Picture 1:" marker; every format carries the reference latent at t = 0 with the format's method."""
    method, vision = EDIT_FORMATS[edit_format]
    if vision:
        scale = min(1.0, (384 * 384 / float(ref.shape[1] * ref.shape[2])) ** 0.5)
        small = comfy.utils.common_upscale(ref.movedim(-1, 1), round(ref.shape[2] * scale), round(ref.shape[1] * scale), "area", "disabled").movedim(1, -1)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize("Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + text, images=[small]))
    else:
        cond = encode(clip, text)
    return node_helpers.conditioning_set_values(cond, {"reference_latents": [latent], "reference_latents_method": method})


class HaDeSMossPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MOSS-Image's text encoder (CLIPLoader, type moss_image): the same Qwen3.5-9B that encodes the prompt "
                                             "writes it here, so no other model is loaded."}),
                "idea": ("STRING", {"multiline": True, "default": "", "tooltip": "What you want, in your own words. May be empty when an image is connected. "
                                                                                  "Empty with no image in detailed mode: a random prompt, the seed "
                                                                                  "drawing its medium, subject area and place."}),
                "mode": (MODES, {"default": "caption", "tooltip": "caption: the model writes a detailed paragraph that keeps to the idea. "
                                                                   "detailed: a longer prompt (about 120-200 words) that may add setting, "
                                                                   "props and story details, with light, lens or technique spelled out; "
                                                                   "another option, not a better one. layout: it writes a pixel "
                                                                   "Layout draft (boxes for objects and text), compiled here; the draft output also "
                                                                   "feeds the Layout Compiler for the matching latent. as written: no generation, the "
                                                                   "idea is encoded as typed (a plain prompt box)."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Layout mode: the canvas the boxes are drawn on. "
                                                                                                    "Use the same size for the image."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05,
                                          "tooltip": "0 always writes the same answer for the same input. Higher values invent more of what the "
                                                     "idea leaves open."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "Starts on fixed: the same idea writes the same prompt until you change the seed (or connect the MOSS "
                                            "Latent's seed to share one value with the sampler)."}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096,
                                       "tooltip": "Length limit of the answer. A layout with many items needs more; a cut answer cannot be parsed."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional reference the model sees: the whole brief when the idea is empty, otherwise the starting "
                                               "point (the idea wins). Resized to about 1 MP."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "The negative prompt, encoded to the negative output; empty uses the "
                                                                                     "official MOSS-Image template's. What you write here is also told "
                                                                                     "to the model as things to keep out."}),
                "edit_format": ([EDIT_OFF] + list(EDIT_FORMATS), {"default": EDIT_OFF, "tooltip":
                    "Edit mode, for an image-edit LoRA loaded in the Multi-LoRA: the image becomes the reference latent of both outputs (needs "
                    "the vae input). Moss-Image-Edit: MOSS-edit and Moss-Image-Edit LoRAs (use mode 'as written' with a direct edit instruction, "
                    "or 'caption' to have the model write the full target description). ai-toolkit (kv_cache on / off): LoRAs trained with "
                    "ai-toolkit (e.g. the pose LoRA); the text encoder also sees the image. The same formats as the core Text Encode MOSS-Image Edit."}),
                "vae": ("VAE", {"tooltip": "Edit mode: encodes the image as the reference latent."}),
                "edit_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "tooltip":
                    "Edit mode: the reference is resized to this area with its aspect kept (16 px grid); width / height out are that size, "
                    "so link them to the MOSS Latent's custom_width / custom_height to generate at the reference's size."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt", "draft", "width", "height", "info")
    OUTPUT_TOOLTIPS = ("The prompt encoded with this CLIP, for the sampler's positive input.",
                       "The negative prompt encoded with this CLIP, for the sampler's negative input.",
                       "The prompt as text.", "The editable JSON draft (a caption, or the layout for the Layout Compiler).",
                       "Canvas width.", "Canvas height.", "Summary, with any changes made to a layout answer.")
    FUNCTION = "enhance"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("A positive / negative prompt box that can also write the prompt: from a short idea and / or an image, with MOSS-Image's own "
                   "text encoder and instructions adapted from OpenMOSS' official prompt enhancer. Outputs the encoded conditionings for the "
                   "sampler and the text.")

    def enhance(self, clip, idea, mode, width, height, temperature, seed, max_tokens, image=None, negative="", edit_format=EDIT_OFF, vae=None,
                edit_megapixels=1.0):
        width, height = snap16(width), snap16(height)
        negative = negative or ""
        neg_text = negative.strip() or DEFAULT_NEGATIVE
        editing = edit_format != EDIT_OFF
        ref = ref_latent = None
        if editing:
            if edit_format not in EDIT_FORMATS:
                raise ValueError(f"MOSS Prompt Enhancer: edit format '{edit_format}' is not in this ComfyUI core; update ComfyUI.")
            if image is None or vae is None:
                raise ValueError("MOSS Prompt Enhancer: edit mode needs the image (the picture to edit) and the vae inputs.")
            if getattr(vae, "latent_channels", 128) != 128:
                raise ValueError(f"MOSS Prompt Enhancer: the VAE has {vae.latent_channels} latent channels; MOSS-Image needs the 128-channel FLUX.2 VAE.")
            if mode == "layout":
                raise ValueError("MOSS Prompt Enhancer: edit mode works with caption, detailed or as written, not layout.")
            ref, width, height = edit_reference(image, edit_megapixels)
            ref_latent = vae.encode(ref)

        def outputs(prompt):
            if editing:
                return encode_edit(clip, prompt, ref, ref_latent, edit_format), encode_edit(clip, neg_text, ref, ref_latent, edit_format)
            return encode(clip, prompt), encode(clip, neg_text)

        if mode == "as written":
            if not idea.strip():
                raise ValueError("MOSS Prompt Enhancer: as written needs an idea to encode")
            prompt = idea
            info = format_info("HaDeS MOSS Prompt Enhancer | as written (no generation)"
                               + (f" | edit {edit_format}, reference {width}x{height}" if editing else (" | image ignored" if image is not None else "")),
                               "prompt_enhancer", mode=mode, width=width, height=height, image=editing, decode_fallback=False, notes=[],
                               prompt=prompt, negative=neg_text, edit_format=edit_format)
            return {"ui": {"text": [prompt]}, "result": (*outputs(prompt), prompt,
                                                         json.dumps({"caption": prompt}, ensure_ascii=False), width, height, info)}
        brief = surprise_brief(seed) if mode == "detailed" and not idea.strip() and image is None else None
        if not idea.strip() and image is None and brief is None:
            raise ValueError("MOSS Prompt Enhancer: write an idea, connect an image, or both (or use detailed mode with both empty for a random prompt)")
        if not hasattr(clip, "generate"):
            raise ValueError("MOSS Prompt Enhancer: this CLIP cannot generate text; use MOSS-Image's text encoder (type moss_image)")
        images = [fit_image(image)] if image is not None else []
        text = chat(instructions(mode, bool(images), surprise=bool(brief)), user_message(idea, mode, width, height, negative, brief), len(images))
        tokens = clip.tokenize(text, images=images) if images else clip.tokenize(text)
        with decode_fallback() as state:
            ids = clip.generate(tokens, do_sample=temperature > 0, max_length=max_tokens, temperature=max(temperature, 0.01),
                                top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=seed)
        answer = clip.decode(ids)
        fallback = bool(state and state["fallback"])

        notes = []
        if mode in ("caption", "detailed"):
            prompt = parse_caption(answer)
            if not prompt:
                raise ValueError("MOSS Prompt Enhancer: the answer has no caption (raise max_tokens if it was cut, or try another seed). "
                                 "Answer starts: " + answer[:400])
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
                           + (" | image" if images else "") + (f" | edit {edit_format}, reference {width}x{height}" if editing else "")
                           + (" | random: {} / {} / {}".format(*brief) if brief else "")
                           + f" | {len(prompt)} chars"
                           + (" | torch decode fallback" if fallback else "") + "".join(f"\nnote: {n}" for n in notes),
                           "prompt_enhancer", mode=mode, width=width, height=height, image=bool(images), decode_fallback=fallback, notes=notes,
                           prompt=prompt, negative=neg_text, edit_format=edit_format,
                           random_brief=list(brief) if brief else None)
        return {"ui": {"text": [prompt]}, "result": (*outputs(prompt), prompt, draft, width, height, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossPromptEnhancer": HaDeSMossPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossPromptEnhancer": "MOSS Prompt Enhancer"}
