import os
import re

import torch
import folder_paths
import safetensors.torch
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils

from .moss_common import (check_compatible, format_info, is_quantized, key_patches, key_patches_weight, key_slot, model_weights,
                          rescale_for, residual_scale)
from .moss_merger_nodes import GROUPS, layer_group

_SAFE_FILENAME_RE = re.compile(r"[^\w\-. ]+")


def lora_output_path(export_filename, default_name, overwrite_existing=False):
    """`models/loras/<name>.safetensors`, kept inside the folder and auto-incremented unless overwriting."""
    base_dir = os.path.abspath(folder_paths.get_folder_paths("loras")[0])
    os.makedirs(base_dir, exist_ok=True)
    name = os.path.basename(export_filename.strip()).removesuffix(".safetensors")
    name = _SAFE_FILENAME_RE.sub("_", name).strip(" .") or default_name
    candidate = os.path.join(base_dir, f"{name}.safetensors")
    counter = 1
    while os.path.exists(candidate) and not overwrite_existing:
        candidate = os.path.join(base_dir, f"{name}_{counter:03d}.safetensors")
        counter += 1
    return candidate


class HaDeSMossLoraExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_tuned": ("MODEL", {"tooltip": "Finetune or merge to extract (its merge / LoRA patches are included)."}),
                "model_base": ("MODEL", {"tooltip": "Model the LoRA will be applied to: the one the finetune came from (SFT or Base)."}),
                "rank": ("INT", {"default": 32, "min": 4, "max": 256, "step": 4,
                                 "tooltip": "LoRA rank. Higher keeps more of the difference (see 'energy kept' in info) and makes a larger file."}),
                "export_filename": ("STRING", {"default": "HaDeS_MOSS_Extracted_LoRA", "tooltip": "Saved under models/loras."}),
            },
            "optional": {
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_lora_path", "info")
    FUNCTION = "extract"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Saves model_tuned - model_base as a LoRA: truncated SVD per linear layer, full diffs for norms and biases."

    def extract(self, model_tuned, model_base, rank, export_filename, overwrite_existing=False):
        wts, wbs = model_weights(model_tuned), model_weights(model_base)
        n_layers = check_compatible([("model_base", wbs), ("model_tuned", wts)])
        # an fp16-fixed finetune is brought to the base's residual scale, or the residual writes would extract as -15/16 of the weights
        scale = residual_scale(wbs, wts)
        quantized = [name for name, ws in (("model_tuned", wts), ("model_base", wbs)) if is_quantized(ws)]
        device = comfy.model_management.get_torch_device()

        lora_sd = {}
        stats = {"lora": 0, "diff": 0, "identical": 0}
        energy_kept = []
        for k in wts:
            if k not in wbs:
                continue
            pk = f"diffusion_model.{k}"
            t = key_patches_weight(key_patches(model_tuned, pk), device, pk)
            if scale is not None:
                r = rescale_for(k, wts[k], scale)
                if r is not None:
                    t = t * r.to(device)
            diff = t - key_patches_weight(key_patches(model_base, pk), device, pk)
            if not torch.any(diff):
                stats["identical"] += 1
                continue
            prefix = pk.rsplit(".", 1)[0]
            if k.endswith(".bias"):
                lora_sd[f"{prefix}.diff_b"] = diff.cpu().to(torch.bfloat16)
                stats["diff"] += 1
            elif diff.ndim == 2:
                u, s, vh = torch.linalg.svd(diff, full_matrices=False)
                r = min(rank, s.shape[0])
                sqrt_s = s[:r].sqrt()
                lora_sd[f"{prefix}.lora_up.weight"] = (u[:, :r] * sqrt_s).cpu().to(torch.bfloat16).contiguous()
                lora_sd[f"{prefix}.lora_down.weight"] = (sqrt_s.unsqueeze(1) * vh[:r]).cpu().to(torch.bfloat16).contiguous()
                lora_sd[f"{prefix}.alpha"] = torch.tensor(float(r))
                energy = s.square()
                energy_kept.append(float(energy[:r].sum() / energy.sum()))
                stats["lora"] += 1
            else:
                lora_sd[f"{prefix}.diff"] = diff.cpu().to(torch.bfloat16)
                stats["diff"] += 1
        if not lora_sd:
            raise ValueError("HaDeS MOSS LoRA Extractor: the two models are identical in every layer.")

        path = lora_output_path(export_filename, "HaDeS_MOSS_Extracted_LoRA", overwrite_existing)
        mean_energy = sum(energy_kept) / len(energy_kept) if energy_kept else 1.0
        safetensors.torch.save_file(lora_sd, path, metadata={"format": "pt", "ss_base_model_version": "moss_image",
                                                              "hades/rank": str(rank), "hades/energy_kept": f"{mean_energy:.4f}"})

        lines = [
            f"HaDeS MOSS LoRA Extractor | {os.path.basename(path)} ({os.path.getsize(path) / 1024 ** 2:.1f} MB) | rank {rank} | {n_layers} layers",
            f"Energy kept: {mean_energy:.1%} (mean over {stats['lora']} low-rank layers) | full diffs (norms, biases): {stats['diff']} | "
            f"identical skipped: {stats['identical']}",
        ]
        if scale is not None:
            lines.append(f"model_tuned's residual stream was brought to model_base's scale (x{scale:.4g}, fp16-fixed file) before extracting.")
        if quantized:
            lines.append(f"WARNING: {', '.join(quantized)} quantized (int8 / fp8); the difference includes its rounding.")
        if mean_energy < 0.5:
            lines.append("Note: less than half of the difference fits this rank; raise rank for a closer match.")
        return {"ui": {"text": [path]}, "result": (path, format_info("\n".join(lines), "lora_extractor", saved_path=path, rank=rank,
                                                                      layers=stats["lora"], energy_kept=round(mean_energy, 4), quantized=quantized))}


def load_lora_patches(lora_name, model):
    """A LoRA file as ComfyUI patches keyed by model weight key, or (key, offset) for split q/k/v on a fused file.
    The core maps every community MOSS format (kohya, PEFT split or fused), tools/RESULTS.md."""
    sd = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True))
    return comfy.lora.load_lora(sd, comfy.lora.model_lora_keys_unet(model.model, {}))


class HaDeSMossLoraBlockWeight:
    @classmethod
    def INPUT_TYPES(cls):
        weight = {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05}
        required = {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        }
        for g in GROUPS:
            required[g] = ("FLOAT", {**weight, "tooltip": f"Multiplier for the LoRA on {g.replace('_', ' ', 1).replace('_', '-')}."})
        required["other_layers"] = ("FLOAT", {**weight, "tooltip": "Multiplier for LoRA weights outside the layers (text_embedder, modulation, embedders, head)."})
        return {"required": required}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "apply"
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Applies one LoRA with a separate strength per group of 6 layers. At 1.0 everywhere it is the stock LoRA loader."

    def apply(self, model, lora_name, strength_model, other_layers, **groups):
        n_layers = len(model.model.diffusion_model.layers)
        weights = [groups[g] for g in GROUPS]
        m = model.clone()
        by_strength = {}
        counts = dict.fromkeys(GROUPS + ["other_layers"], 0)
        for k, patch in load_lora_patches(lora_name, model).items():
            target = k[0] if isinstance(k, tuple) else k
            slot = key_slot(target.removeprefix("diffusion_model."))
            group = GROUPS[layer_group(slot, n_layers)] if isinstance(slot, int) else "other_layers"
            counts[group] += 1
            s = round(strength_model * (groups[group] if group in groups else other_layers), 4)
            if s != 0.0:
                by_strength.setdefault(s, {})[k] = patch
        for s, patches in by_strength.items():
            m.add_patches(patches, strength_patch=s)
        text = (f"HaDeS MOSS LoRA Block Weight | {lora_name} | model {strength_model:+.2f} | "
                f"{' / '.join(f'{w:.2f}' for w in weights)} | other {other_layers:.2f}\n"
                f"LoRA weights per group: {', '.join(f'{g} {c}' for g, c in counts.items())}")
        return (m, format_info(text, "lora_block_weight", lora=lora_name, strength_model=strength_model, weights=weights,
                               other_layers=other_layers, counts=counts))


NODE_CLASS_MAPPINGS = {"HaDeSMossLoraExtractor": HaDeSMossLoraExtractor, "HaDeSMossLoraBlockWeight": HaDeSMossLoraBlockWeight}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLoraExtractor": "MOSS LoRA Extractor", "HaDeSMossLoraBlockWeight": "MOSS LoRA Block Weight"}
