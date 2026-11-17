"""MOSS Compare Sheet: up to four images side by side, each labelled with the LoRA (or model) its branch used, saved as one
PNG. The labels come from each branch's HaDeS MOSS info (Multi-LoRA, LoRA Block Weight, KSampler turbo, Loader, or a Suite
Telemetry json), so a checkpoint comparison needs no script and no stitch nodes."""
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

from .moss_common import format_info, parse_info_meta

SLOTS = "ABCD"
GAP, PAD = 16, 12


def branch_metas(text):
    """HADES_META entries of a Suite Telemetry json or of one info output; [] for anything else."""
    if not text:
        return []
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return [m for m in data["nodes"] if isinstance(m, dict) and "node" in m]
    meta = parse_info_meta(text)
    return [meta] if meta is not None else []


def stem(name):
    return os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]


def strength_text(s):
    return "" if abs(float(s) - 1.0) < 1e-6 else f" x{float(s):g}"


def auto_label(metas):
    """The LoRAs of the branch (Multi-LoRA rows, LoRA Block Weight, KSampler turbo), else its model file, else ''."""
    parts = []
    for m in metas:
        if m.get("node") == "multi_lora":
            parts += [stem(r["name"]) + strength_text(r.get("strength", 1.0)) for r in m.get("loras") or [] if isinstance(r, dict) and r.get("name")]
        elif m.get("node") == "lora_block_weight" and m.get("lora"):
            parts.append(stem(m["lora"]) + strength_text(m.get("strength_model", 1.0)))
        elif m.get("node") == "ksampler" and m.get("turbo"):
            parts.append(stem(m["turbo"]) + strength_text(m.get("turbo_strength", 1.0)))
    if parts:
        return " + ".join(parts)
    loader = next((m for m in metas if m.get("node") == "loader"), None)
    return stem(loader.get("model_file") or loader.get("model", "")) if loader else ""


def model_of(metas):
    loader = next((m for m in metas if m.get("node") == "loader"), None)
    return stem(loader.get("model_file") or loader.get("model", "")) if loader else ""


def settings_of(metas):
    """The sampling settings a branch ran with (its KSampler info), as ordered display fields."""
    ks = next((m for m in metas if m.get("node") == "ksampler"), None)
    if ks is None:
        return {}
    out = {}
    if "seed" in ks:
        out["seed"] = f"seed {ks['seed']}"
    if "steps" in ks:
        out["steps"] = f"{ks['steps']} steps"
    if "cfg" in ks:
        out["cfg"] = f"CFG {float(ks['cfg']):g}"
    if ks.get("sampler") or ks.get("scheduler"):
        out["sampler"] = f"{ks.get('sampler', '?')} / {ks.get('scheduler', '?')}"
    if ks.get("width") and ks.get("height"):
        out["size"] = f"{ks['width']}x{ks['height']}"
    if float(ks.get("denoise", 1.0)) < 0.9999:
        out["denoise"] = f"denoise {float(ks['denoise']):g}"
    return out


def prompt_of(metas):
    return next((m.get("prompt") for m in metas if m.get("node") == "prompt_enhancer" and m.get("prompt")), "")


def split_common(per_branch):
    """(common fields shown once, {slot: fields that differ}) over the branches that carry the field."""
    keys = list(dict.fromkeys(k for f in per_branch.values() for k in f))
    common, differ = [], {s: [] for s in per_branch}
    for k in keys:
        vals = {s: f[k] for s, f in per_branch.items() if k in f}
        if len(set(vals.values())) == 1 and len(vals) == len(per_branch):
            common.append(next(iter(vals.values())))
        else:
            for s, v in vals.items():
                differ[s].append(v)
    return common, differ


def auto_footer(metas):
    """'seed N | prompt' from one branch's infos (kept for single-branch use)."""
    fields = settings_of(metas)
    prompt = prompt_of(metas)
    return " | ".join(([fields["seed"]] if "seed" in fields else []) + ([prompt] if prompt else []))


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def fit(draw, text, fnt, width):
    """Shortens text with '...' until it fits one line."""
    if draw.textlength(text, font=fnt) <= width:
        return text
    while text and draw.textlength(text + "...", font=fnt) > width:
        text = text[:-1]
    return text + "..."


def build_sheet(panels, labels, footer):
    """panels: PIL images; all scaled to the first one's height, joined left to right with labels above and footer below."""
    h = panels[0].height
    panels = [p if p.height == h else p.resize((round(p.width * h / p.height), h), Image.LANCZOS) for p in panels]
    width = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    f_label = font(max(18, h // 45))
    f_foot = font(max(14, h // 65))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    head = f_label.size + 2 * PAD
    foot_lines = wrap(probe, footer, f_foot, width - 2 * PAD)[:4] if footer else []
    foot = (f_foot.size + 6) * len(foot_lines) + (2 * PAD if foot_lines else 0)
    sheet = Image.new("RGB", (width, head + h + foot), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for p, label in zip(panels, labels):
        sheet.paste(p, (x, head))
        draw.text((x + PAD, PAD), fit(draw, label, f_label, p.width - 2 * PAD), fill="black", font=f_label)
        x += p.width + GAP
    for k, line in enumerate(foot_lines):
        draw.text((PAD, head + h + PAD + k * (f_foot.size + 6)), line, fill=(60, 60, 60), font=f_foot)
    return sheet


def to_pil(t):
    return Image.fromarray(np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8))


class HaDeSMossCompareSheet:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for s in SLOTS[1:]:
            optional[f"image_{s}"] = ("IMAGE", {"tooltip": f"Image {s}; leave empty to compare fewer branches."})
        for s in SLOTS:
            optional[f"info_{s}"] = ("STRING", {"forceInput": True, "tooltip": f"Branch {s}'s Multi-LoRA, LoRA Block Weight, KSampler or Loader "
                                                                                 "info, or a Suite Telemetry json of that branch: the label is read from it."})
        for s in SLOTS:
            optional[f"label_{s}"] = ("STRING", {"default": "", "tooltip": f"Label over image {s}. Empty: the LoRA names (and strength when not 1) "
                                                                            f"from info_{s}, else its model file, else '{s}'."})
        optional["blind"] = ("BOOLEAN", {"default": False, "tooltip": "Blind comparison: the panels are shuffled and numbered 1 to 4 "
                                                                    "without names; the key (number -> branch) is in the info output and "
                                                                    "in the PNG. Pick your favourite before looking at it."})
        optional["blind_seed"] = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Order of the blind shuffle; change it "
                                                                                              "for a new order of the same images."})
        optional["footer"] = ("STRING", {"default": "", "multiline": True, "tooltip": "Text under the sheet. Empty: seed and prompt read from the "
                                                                                       "branch infos (KSampler and Prompt Enhancer) when connected."})
        return {
            "required": {
                "image_A": ("IMAGE", {"tooltip": "Image A, the leftmost. Its height sets the sheet's height; other images are scaled to it."}),
                "filename_prefix": ("STRING", {"default": "MOSS/compare/sheet", "tooltip": "Subfolder and name in ComfyUI/output, as in the core Save Image."}),
                "save": ("BOOLEAN", {"default": True, "tooltip": "Off: only builds the sheet for the image output and the preview."}),
                "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Write ComfyUI's prompt and workflow into the PNG, so it loads back into ComfyUI."}),
            },
            "optional": optional,
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "info")
    OUTPUT_TOOLTIPS = ("The labelled sheet, one per batch index.", "Files saved and labels used, for Suite Telemetry.")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Puts up to four images side by side, A to D, each labelled with the LoRA or model its branch used (read from the "
                   "branch's info), with seed and prompt underneath, and saves the sheet as one PNG.")

    def compare(self, image_A, filename_prefix, save, embed_workflow, prompt=None, extra_pnginfo=None, footer="", blind=False,
                blind_seed=0, **kw):
        images = [(s, image_A if s == "A" else kw.get(f"image_{s}")) for s in SLOTS]
        images = [(s, im) for s, im in images if im is not None]
        metas = {s: branch_metas(kw.get(f"info_{s}")) for s, _ in images}
        # what is the same in every branch goes in the footer once; what differs (model, seed, steps...) goes on that panel's label
        slots = [s for s, _ in images]
        fields = {s: settings_of(metas[s]) for s in slots}
        models = {s: model_of(metas[s]) for s in slots}
        known_models = {m for m in models.values() if m}
        if len(known_models) > 1:
            for s in slots:
                if models[s]:
                    fields[s] = {"model": models[s], **fields[s]}
        common, differ = split_common({s: f for s, f in fields.items() if f})
        labels, sources = [], []
        for s in slots:
            given = (kw.get(f"label_{s}") or "").strip()
            auto = auto_label(metas[s])
            extra = [v for v in differ.get(s, []) if v != auto]
            text = given or " · ".join([auto] * bool(auto) + extra)
            labels.append(f"{s}: {text}" if text else s)
            sources.append("label" if given else ("info" if (auto or extra) else "slot"))
        if not footer.strip():
            prompts = {p for p in (prompt_of(metas[s]) for s in slots) if p}
            prompt_text = next(iter(prompts)) if len(prompts) == 1 else ("prompts differ between branches" if prompts else "")
            if len(known_models) == 1:
                common.insert(0, next(iter(known_models)))
            footer = " | ".join(common + ([prompt_text] if prompt_text else []))
        batch = min(im.shape[0] for _, im in images)
        order, key = list(range(len(images))), {}
        shown = labels
        if blind:  # shuffled panels numbered 1..n; the key goes to info and the PNG, never onto the sheet
            random.Random(blind_seed).shuffle(order)
            shown = [str(k + 1) for k in range(len(order))]
            key = {str(k + 1): labels[i] for k, i in enumerate(order)}
        sheets = [build_sheet([to_pil(images[i][1][b]) for i in order], shown, footer.strip()) for b in range(batch)]

        results, files = [], []
        if save:
            out_dir = folder_paths.get_output_directory()
            full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, sheets[0].width, sheets[0].height)
            for b, sheet in enumerate(sheets):
                info = PngInfo()
                if not args.disable_metadata:
                    info.add_text("hades_moss_compare", json.dumps({"labels": labels, "footer": footer.strip(), "blind_key": key,
                                                                    "branches": {s: metas[s] for s, _ in images}}, ensure_ascii=False))
                    if embed_workflow:
                        if prompt is not None:
                            info.add_text("prompt", json.dumps(prompt))
                        for k, v in (extra_pnginfo or {}).items():
                            info.add_text(k, json.dumps(v))
                file = f"{filename.replace('%batch_num%', str(b))}_{counter:05}_.png"
                sheet.save(os.path.join(full_folder, file), pnginfo=info, compress_level=4)
                results.append({"filename": file, "subfolder": subfolder, "type": "output"})
                files.append(f"{subfolder}/{file}" if subfolder else file)
                counter += 1
        else:
            for b, sheet in enumerate(sheets):
                tmp_dir = folder_paths.get_temp_directory()
                full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path("hades_compare", tmp_dir, sheet.width, sheet.height)
                file = f"{filename}_{counter:05}_.png"
                sheet.save(os.path.join(full_folder, file), compress_level=1)
                results.append({"filename": file, "subfolder": subfolder, "type": "temp"})

        notes = [f"{s}: no label in info_{s}, named '{s}'" for (s, _), src in zip(images, sources) if src == "slot"]
        sizes = {tuple(im.shape[1:3]) for _, im in images}
        if len(sizes) > 1:
            notes.append(f"images of different sizes {sorted(sizes)}: scaled to image A's height")
        if any(im.shape[0] != batch for _, im in images):
            notes.append(f"batch sizes differ: {batch} sheet(s), the extra images were left out")
        head = ("blind, key: " + " | ".join(f"{n} = {lab}" for n, lab in key.items())) if blind else " | ".join(labels)
        text = (f"HaDeS MOSS Compare Sheet | {len(images)} images | " + head
                + (f"\nsaved: {files[0]}" + (f" (+{len(files) - 1})" if len(files) > 1 else "") if files else "\nnot saved (preview only)")
                + "".join(f"\nnote: {n}" for n in notes))
        info = format_info(text, "compare_sheet", labels=labels, label_sources=sources, files=files, footer=footer.strip(), notes=notes,
                           **({"blind_key": key, "blind_seed": blind_seed} if blind else {}))
        out = torch.from_numpy(np.stack([np.asarray(s, dtype=np.float32) / 255.0 for s in sheets]))
        return {"ui": {"images": results}, "result": (out, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossCompareSheet": HaDeSMossCompareSheet}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossCompareSheet": "MOSS Compare Sheet"}
