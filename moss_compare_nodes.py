"""MOSS Compare Sheet: up to four images side by side, each labelled with the LoRA (or model) its branch used, saved as one
PNG. The labels come from each branch's HaDeS MOSS info (Multi-LoRA, LoRA Block Weight, KSampler turbo, Loader, or a Suite
Telemetry json), so a checkpoint comparison needs no script and no stitch nodes."""
import json
import os

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
            parts += [stem(r["name"]) + strength_text(r.get("strength", 1.0)) for r in m.get("loras", [])]
        elif m.get("node") == "lora_block_weight":
            parts.append(stem(m["lora"]) + strength_text(m.get("strength_model", 1.0)))
        elif m.get("node") == "ksampler" and m.get("turbo"):
            parts.append(stem(m["turbo"]) + strength_text(m.get("turbo_strength", 1.0)))
    if parts:
        return " + ".join(parts)
    loader = next((m for m in metas if m.get("node") == "loader"), None)
    return stem(loader.get("model_file") or loader.get("model", "")) if loader else ""


def auto_footer(metas):
    """'seed N | prompt' from the first branch that carries them."""
    seed = next((m["seed"] for m in metas if m.get("node") == "ksampler" and "seed" in m), None)
    prompt = next((m.get("prompt") for m in metas if m.get("node") == "prompt_enhancer" and m.get("prompt")), "")
    bits = ([f"seed {seed}"] if seed is not None else []) + ([prompt] if prompt else [])
    return " | ".join(bits)


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

    def compare(self, image_A, filename_prefix, save, embed_workflow, prompt=None, extra_pnginfo=None, footer="", **kw):
        images = [(s, image_A if s == "A" else kw.get(f"image_{s}")) for s in SLOTS]
        images = [(s, im) for s, im in images if im is not None]
        metas = {s: branch_metas(kw.get(f"info_{s}")) for s, _ in images}
        labels, sources = [], []
        for s, _ in images:
            given = (kw.get(f"label_{s}") or "").strip()
            auto = auto_label(metas[s])
            labels.append(f"{s}: {given or auto or s}" if (given or auto) else s)
            sources.append("label" if given else ("info" if auto else "slot"))
        if not footer.strip():
            footer = next((t for t in (auto_footer(metas[s]) for s, _ in images) if t), "")
        batch = min(im.shape[0] for _, im in images)
        sheets = [build_sheet([to_pil(im[b]) for _, im in images], labels, footer.strip()) for b in range(batch)]

        results, files = [], []
        if save:
            out_dir = folder_paths.get_output_directory()
            full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, sheets[0].width, sheets[0].height)
            for b, sheet in enumerate(sheets):
                info = PngInfo()
                if not args.disable_metadata:
                    info.add_text("hades_moss_compare", json.dumps({"labels": labels, "footer": footer.strip(),
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
        text = (f"HaDeS MOSS Compare Sheet | {len(images)} images | " + " | ".join(labels)
                + (f"\nsaved: {files[0]}" + (f" (+{len(files) - 1})" if len(files) > 1 else "") if files else "\nnot saved (preview only)")
                + "".join(f"\nnote: {n}" for n in notes))
        info = format_info(text, "compare_sheet", labels=labels, label_sources=sources, files=files, footer=footer.strip(), notes=notes)
        out = torch.from_numpy(np.stack([np.asarray(s, dtype=np.float32) / 255.0 for s in sheets]))
        return {"ui": {"images": results}, "result": (out, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossCompareSheet": HaDeSMossCompareSheet}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossCompareSheet": "MOSS Compare Sheet"}
