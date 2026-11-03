import os
import re

import torch
import folder_paths
import safetensors.torch
import comfy.model_management

from .moss_common import (check_compatible, format_info, is_quantized, key_patches, key_patches_weight, model_weights, rescale_for,
                          residual_scale)

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


NODE_CLASS_MAPPINGS = {"HaDeSMossLoraExtractor": HaDeSMossLoraExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLoraExtractor": "MOSS LoRA Extractor"}
