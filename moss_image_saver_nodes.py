"""MOSS Image Saver: a PNG with the generation settings written as A1111 "parameters" (the text Civitai and most image
browsers read), taken from Suite Telemetry's json output, plus ComfyUI's own prompt / workflow chunks."""
import hashlib
import json
import os
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

from .moss_common import format_info, parse_info_meta

# sha256 per file, kept while the file is unchanged (same size and modification time) for this ComfyUI session
_HASHES = {}


def file_sha256(path):
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def read_metas(text):
    """The HADES_META entries of Suite Telemetry's json output, or of one info output connected directly."""
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return [m for m in data["nodes"] if isinstance(m, dict) and "node" in m]
    meta = parse_info_meta(text)
    if meta is not None:
        return [meta]
    raise ValueError("HaDeS MOSS Image Saver: connect Suite Telemetry's json output (or the info output of a HaDeS MOSS node) to metadata.")


def first(metas, node):
    return next((m for m in metas if m.get("node") == node), None)


def stem(name):
    return os.path.splitext(os.path.basename(str(name)))[0]


def resources(metas):
    """[(kind, folder, file name, strength)] the image was made with: the diffusion model (MOSS Loader) and every LoRA
    (Multi-LoRA rows, LoRA Block Weight, the KSampler's turbo input)."""
    out = []
    loader = first(metas, "loader")
    if loader and loader.get("model_file"):
        out.append(("model", "diffusion_models", loader["model_file"], None))
    for m in metas:
        if m.get("node") == "multi_lora":
            out += [("lora", "loras", r["name"], r.get("strength", 1.0)) for r in m.get("loras", [])]
        elif m.get("node") == "lora_block_weight":
            out.append(("lora", "loras", m["lora"], m.get("strength_model", 1.0)))
        elif m.get("node") == "ksampler" and m.get("turbo"):
            out.append(("lora", "loras", m["turbo"], m.get("turbo_strength", 1.0)))
    return out


def a1111_parameters(metas, positive=None, negative=None, with_hashes=True):
    """(parameters text, notes): prompt, 'Negative prompt:' and one settings line, as A1111 writes them."""
    notes = []
    enh = first(metas, "prompt_enhancer")
    positive = positive if positive is not None else (enh or {}).get("prompt", "")
    negative = negative if negative is not None else (enh or {}).get("negative", "")
    if not positive:
        notes.append("no prompt text: connect a Prompt Enhancer's info to Telemetry, or a string to positive")
    ks, latent, loader = first(metas, "ksampler"), first(metas, "latent"), first(metas, "loader")
    if ks is None:
        notes.append("no MOSS KSampler info: the sampling settings are missing")
    fields = []
    if ks:
        fields += [("Steps", ks["steps"]), ("Sampler", ks["sampler"]), ("Schedule type", ks["scheduler"]), ("CFG scale", f"{ks['cfg']:g}"),
                   ("Seed", ks["seed"]), ("Size", f"{ks['width']}x{ks['height']}")]
    elif latent:
        fields += [("Seed", latent.get("seed", 0)), ("Size", f"{latent['width']}x{latent['height']}")]

    hashes, lora_hashes = {}, []
    for kind, folder, name, strength in resources(metas):
        path = folder_paths.get_full_path(folder, name)
        short = file_sha256(path)[:10] if (with_hashes and path) else None
        if path is None:
            notes.append(f"{name} not found under models/{folder}: no hash")
        if kind == "model":
            if short:
                fields.append(("Model hash", short))
                hashes["model"] = short
            fields.append(("Model", stem(name)))
        else:
            if short:
                lora_hashes.append(f"{stem(name)}: {short}")
                hashes[f"lora:{stem(name)}"] = short
    if ks and ks.get("denoise", 1.0) < 1.0:
        fields.append(("Denoising strength", f"{ks['denoise']:g}"))
    if ks:
        fields.append(("Shift", f"{ks['flow_shift']:g}"))
    if loader:
        fields += [("VAE", stem(loader["vae"])), ("Text encoder", stem(loader["text_encoder"]))]
    if lora_hashes:
        fields.append(("Lora hashes", '"' + ", ".join(lora_hashes) + '"'))
    if hashes:
        fields.append(("Hashes", json.dumps(hashes, separators=(", ", ": "))))
    fields.append(("Version", "ComfyUI, HaDeS MOSS Suite"))

    lines = [positive.strip()]
    if negative and negative.strip():
        lines.append(f"Negative prompt: {negative.strip()}")
    lines.append(", ".join(f"{k}: {v}" for k, v in fields))
    return "\n".join(lines), notes


def expand_tokens(prefix, metas):
    """%seed%, %model%, %steps%, %cfg%, %sampler%, %width%, %height% from the metadata; ComfyUI expands %date:...% itself."""
    ks, loader = first(metas, "ksampler") or {}, first(metas, "loader") or {}
    values = {"seed": ks.get("seed", ""), "model": stem(loader.get("model", "")) if loader else "", "steps": ks.get("steps", ""),
              "cfg": f"{ks['cfg']:g}" if "cfg" in ks else "", "sampler": ks.get("sampler", ""), "width": ks.get("width", ""), "height": ks.get("height", "")}
    out = prefix
    for k, v in values.items():
        out = out.replace(f"%{k}%", re.sub(r'[\\/:*?"<>|]+', "_", str(v)))
    return out


class HaDeSMossImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "MOSS/%model%_%seed%",
                                               "tooltip": "Subfolder and name in ComfyUI/output. %seed%, %model%, %steps%, %cfg%, %sampler%, %width%, "
                                                          "%height% come from the metadata; %date:yyyy-MM-dd% as in the core Save Image."}),
                "hashes": ("BOOLEAN", {"default": True, "tooltip": "SHA256 of the model and LoRA files (the first 10 characters, AutoV2) in Model hash, "
                                                                   "Lora hashes and Hashes, which Civitai uses to link the resources. The first save with a "
                                                                   "model reads the whole file once (seconds for 10 GB); later saves reuse it while the "
                                                                   "file is unchanged. Nothing is looked up online."}),
                "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Also write ComfyUI's prompt and workflow, as the core Save Image does, "
                                                                           "so the image loads back into ComfyUI."}),
            },
            "optional": {
                "metadata": ("STRING", {"forceInput": True, "tooltip": "Suite Telemetry's json output (it gathers the Loader, Latent, KSampler, "
                                                                        "Multi-LoRA and Prompt Enhancer infos), or one info output connected directly. "
                                                                        "Without it the images are still saved, with the workflow and only the prompt "
                                                                        "texts given here as parameters."}),
                "primary_stage_metadata": ("STRING", {"forceInput": True, "tooltip": "Optional: json from a second Suite Telemetry connected only to the "
                                                                                      "stage this image represents. Its one KSampler and relevant Loader, "
                                                                                      "LoRA and prompt infos supply A1111 parameters and filename tokens; "
                                                                                      "metadata still records the complete pipeline."}),
                "positive": ("STRING", {"forceInput": True, "tooltip": "Prompt text when no Prompt Enhancer is in the pipeline (e.g. CLIPTextEncode's text)."}),
                "negative": ("STRING", {"forceInput": True, "tooltip": "Negative prompt text when no Prompt Enhancer is in the pipeline."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("parameters", "info")
    OUTPUT_TOOLTIPS = ("The A1111 parameters text written into the images.", "The files saved and what was written into them, for Suite Telemetry.")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Saves PNG images with the generation settings as A1111 parameters (prompt, negative prompt, steps, sampler, CFG, seed, "
                   "size, model and LoRAs with their hashes), read from Suite Telemetry, and ComfyUI's workflow.")

    def save(self, images, filename_prefix, hashes, embed_workflow, metadata=None, positive=None, negative=None, prompt=None,
             extra_pnginfo=None, primary_stage_metadata=None):
        metas = read_metas(metadata) if metadata else []
        primary = read_metas(primary_stage_metadata) if primary_stage_metadata else None
        if primary is not None:
            for node in ("ksampler", "loader", "prompt_enhancer"):
                count = sum(m.get("node") == node for m in primary)
                if (node == "ksampler" and count != 1) or (node != "ksampler" and count > 1):
                    raise ValueError(f"HaDeS MOSS Image Saver: primary_stage_metadata needs exactly one KSampler and at most one "
                                     f"Loader and Prompt Enhancer (found {count} {node} entries).")
        selected = primary if primary is not None else metas
        parameters, notes = a1111_parameters(selected, positive, negative, hashes)
        if primary is not None and not first(primary, "loader"):
            notes.append("primary stage has no MOSS Loader info: model and model hash are missing")
        if not metadata and primary is None:
            notes.insert(0, "metadata not connected: saved without sampling settings (connect Suite Telemetry's json for them)")
        prefix = expand_tokens(filename_prefix, selected)
        out_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, out_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for batch_number, image in enumerate(images):
            img = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
            info = PngInfo()
            if not args.disable_metadata:
                info.add_text("parameters", parameters)
                info.add_text("hades_moss", json.dumps(metas if metadata else selected, ensure_ascii=False))
                if primary is not None:
                    info.add_text("hades_moss_primary_stage", json.dumps(primary, ensure_ascii=False))
            if embed_workflow and not args.disable_metadata:
                if prompt is not None:
                    info.add_text("prompt", json.dumps(prompt))
                for k, v in (extra_pnginfo or {}).items():
                    info.add_text(k, json.dumps(v))
            file = f"{filename.replace('%batch_num%', str(batch_number))}_{counter:05}_.png"
            img.save(os.path.join(full_folder, file), pnginfo=info, compress_level=4)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        files = [f"{r['subfolder']}/{r['filename']}" if r["subfolder"] else r["filename"] for r in results]
        info = format_info(f"HaDeS MOSS Image Saver | {len(files)} image(s) | {files[0] if files else ''}"
                           + "".join(f"\nnote: {n}" for n in notes), "image_saver", files=files,
                           metadata=bool(metadata), primary_stage=primary is not None, hashes=bool(hashes),
                           embed_workflow=bool(embed_workflow) and not args.disable_metadata, metadata_written=not args.disable_metadata, notes=notes)
        if notes:
            parameters_note = "\n".join(f"note: {n}" for n in notes)
            return {"ui": {"images": results, "text": [parameters_note]}, "result": (parameters, info)}
        return {"ui": {"images": results}, "result": (parameters, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossImageSaver": HaDeSMossImageSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossImageSaver": "MOSS Image Saver"}
