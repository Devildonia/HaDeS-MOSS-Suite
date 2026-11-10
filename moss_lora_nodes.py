import json
import math
import os
import re

import torch
import folder_paths
import safetensors.torch
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import comfy.weight_adapter

from .moss_common import (check_compatible, format_info, format_model_info, is_quantized, key_patches, key_patches_weight, key_slot, model_weights,
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
                "tuned_name": ("STRING", {"default": "", "tooltip": "Optional. Name of model_tuned (e.g. its file), written into the LoRA as hades/model_tuned; "
                                                                    "the MOSS Metadata Reader shows it."}),
                "base_name": ("STRING", {"default": "", "tooltip": "Optional. Name of model_base, written as hades/model_base."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_lora_path", "info")
    FUNCTION = "extract"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Saves model_tuned - model_base as a LoRA: truncated SVD per linear layer, full diffs for norms and biases."

    def extract(self, model_tuned, model_base, rank, export_filename, overwrite_existing=False, tuned_name="", base_name=""):
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
                                                              "hades/rank": str(rank), "hades/energy_kept": f"{mean_energy:.4f}",
                                                              **({"hades/model_tuned": tuned_name.strip()} if tuned_name.strip() else {}),
                                                              **({"hades/model_base": base_name.strip()} if base_name.strip() else {})})

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
    The core maps every community MOSS format (kohya, PEFT split or fused), tools/RESULTS.md. A LoRA none of whose keys
    maps to this model (one made for another model) is an error: the core would load it as nothing, without a word."""
    sd = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True))
    patches = comfy.lora.load_lora(sd, comfy.lora.model_lora_keys_unet(model.model, {}))
    if not patches:
        raise ValueError(f"HaDeS MOSS LoRA: '{lora_name}' changes no weight of this MOSS-Image model (none of its keys match); "
                         "it is probably a LoRA for another model.")
    return patches


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
        weights = [groups[g] for g in GROUPS]
        m = model.clone()
        counts = add_lora_by_group(m, model, lora_name, strength_model, groups, other_layers)
        text = (f"HaDeS MOSS LoRA Block Weight | {lora_name} | model {strength_model:+.2f} | "
                f"{' / '.join(f'{w:.2f}' for w in weights)} | other {other_layers:.2f}\n"
                f"LoRA weights per group: {', '.join(f'{g} {c}' for g, c in counts.items())}")
        return (m, format_model_info(text, "lora_block_weight", m, lora=lora_name, strength_model=strength_model, weights=weights,
                                     other_layers=other_layers, counts=counts, sources={"model": model.get_attachment("hades_moss_recipe")}))


def add_lora_by_group(m, model, lora_name, strength, groups, other_layers=1.0):
    """Adds `lora_name` to the clone `m` at `strength` times the weight of each group of 6 layers (missing groups 1.0) and
    other_layers outside them. Returns the number of LoRA weights per group."""
    n_layers = len(model.model.diffusion_model.layers)
    by_strength = {}
    counts = dict.fromkeys(GROUPS + ["other_layers"], 0)
    for k, patch in load_lora_patches(lora_name, model).items():
        target = k[0] if isinstance(k, tuple) else k
        slot = key_slot(target.removeprefix("diffusion_model."))
        group = GROUPS[layer_group(slot, n_layers)] if isinstance(slot, int) else "other_layers"
        counts[group] += 1
        s = round(strength * (other_layers if group == "other_layers" else groups.get(group, 1.0)), 4)
        if s != 0.0:
            by_strength.setdefault(s, {})[k] = patch
    for s, patches in by_strength.items():
        m.add_patches(patches, strength_patch=s)
    return counts


def lora_metadata(lora_name):
    """The safetensors header metadata of a LoRA file ({} when it has none or is not a safetensors file)."""
    try:
        with safetensors.safe_open(folder_paths.get_full_path_or_raise("loras", lora_name), "pt") as f:
            return f.metadata() or {}
    except Exception:
        return {}


_STEPS_RE = re.compile(r"^\s*(\d+)\s+steps?\b", re.IGNORECASE)
_CFG1_RE = re.compile(r"\bCFG 1(\.0+)?(?![.\d])")  # "CFG 1" or "CFG 1.0", not "CFG 1.5"


def few_step_settings(meta):
    """(steps, settings text) when the file's own metadata says it is a few-step LoRA sampled at CFG 1, as the HaDeS MOSS Turbo
    files do ("4 steps, CFG 1, euler / simple, strength 1.0, shift 5"); None otherwise. Nothing is guessed from the file name."""
    settings = str(meta.get("settings", ""))
    match = _STEPS_RE.match(settings)
    if match and _CFG1_RE.search(settings):
        return int(match.group(1)), settings.strip()
    return None


def parse_lora_rows(text):
    """The Multi-LoRA list: JSON [{name, on, strength, groups: {layers_00_05: w, ..., other_layers: w}}]; on, strength and groups optional."""
    try:
        rows = json.loads(text or "[]")
    except json.JSONDecodeError as e:
        raise ValueError(f"HaDeS MOSS Multi-LoRA: the LoRA list is not valid JSON ({e}).") from e
    if not isinstance(rows, list):
        raise ValueError("HaDeS MOSS Multi-LoRA: the LoRA list must be a JSON list of rows.")
    out = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("name", ""), str):
            raise ValueError(f"HaDeS MOSS Multi-LoRA: row {i} needs a 'name' (fields: name, on, strength, groups).")
        if not row.get("name"):
            continue  # a row added in the UI with no LoRA picked yet
        groups = row.get("groups") or {}
        if not isinstance(groups, dict):
            raise ValueError(f"HaDeS MOSS Multi-LoRA: row {i} 'groups' must be an object of group: weight.")
        strength = float(row.get("strength", 1.0))
        groups = {g: float(groups[g]) for g in GROUPS + ["other_layers"] if g in groups}
        if not all(math.isfinite(v) for v in [strength, *groups.values()]):
            raise ValueError(f"HaDeS MOSS Multi-LoRA: row {i} strengths must be finite numbers.")
        out.append({"name": row["name"], "on": bool(row.get("on", True)), "strength": strength, "groups": groups})
    return out


class HaDeSMossMultiLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off passes the model through untouched and keeps the list."}),
                "loras": ("STRING", {"default": "[]", "tooltip": "The list the node's rows edit, as JSON: "
                                                                    "[{name, on, strength, groups: {layers_00_05: 1.0, ..., other_layers: 1.0}}]."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "apply"
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = ("Stacks any number of LoRAs, each with its own on / off, strength and optional strength per group of 6 layers. "
                   "A row at strength 1.0 with every group at 1.0 is the stock LoRA loader. Few-step LoRAs are named in info with the "
                   "settings their file's metadata gives.")

    def apply(self, model, enabled, loras):
        rows = parse_lora_rows(loras)
        if not enabled:
            return (model, format_info("HaDeS MOSS Multi-LoRA | off (model passed through)", "multi_lora", enabled=False, loras=[], turbo=[]))
        active = [r for r in rows if r["on"] and r["strength"] != 0.0]
        m = model.clone()
        lines, turbo, applied = [f"HaDeS MOSS Multi-LoRA | {len(active)} of {len(rows)} on"], [], []
        for r in active:
            groups = r["groups"]
            counts = add_lora_by_group(m, model, r["name"], r["strength"], groups, groups.get("other_layers", 1.0))
            custom = {g: w for g, w in groups.items() if w != 1.0}
            lines.append(f"  {r['strength']:+.2f}  {r['name']}  ({sum(counts.values())} weights)"
                         + (f"  (groups: {', '.join(f'{g} {w:.2f}' for g, w in custom.items())})" if custom else ""))
            applied.append({"name": r["name"], "strength": r["strength"], "weights": sum(counts.values()), **({"groups": custom} if custom else {})})
            few = few_step_settings(lora_metadata(r["name"]))
            if few is not None:
                turbo.append({"name": r["name"], "steps": few[0], "strength": r["strength"]})
                lines.append(f"      few-step LoRA, file metadata: {few[1]}")
        if len(turbo) > 1:
            lines.append(f"WARNING: {len(turbo)} few-step LoRAs stacked; each was trained alone and they were not measured together.")
        for t in turbo:
            if t["strength"] != 1.0:
                lines.append(f"Note: {t['name']} at {t['strength']:+.2f}; its metadata gives the strength it was trained at.")
        return (m, format_model_info("\n".join(lines), "multi_lora", m, enabled=True, loras=applied, turbo=turbo,
                                     sources={"model": model.get_attachment("hades_moss_recipe")}))


# ==============================================================================
# LoRA FACTORS: any MOSS LoRA format as up / down per model weight
# ==============================================================================
def lora_factors(lora_name, model, strength, device):
    """The LoRA as {model weight key: (up, down)} with alpha and `strength` folded into up, so that delta = up @ down, plus
    {model weight key: full diff}. Split q/k/v LoRAs on a fused file land in their rows of proj.qkv, so every format comes
    out in the model's own layout."""
    factors, diffs = {}, {}
    for k, patch in load_lora_patches(lora_name, model).items():
        key, offset = k if isinstance(k, tuple) else (k, None)
        rows = comfy.utils.get_attr(model.model, key).shape[0]
        if isinstance(patch, comfy.weight_adapter.LoRAAdapter):
            up, down, alpha, mid, dora_scale, reshape = patch.weights
            if mid is not None or dora_scale is not None or reshape is not None:
                raise ValueError(f"HaDeS MOSS LoRA: '{lora_name}' uses LoCon mid / DoRA / reshape on '{key}'; only plain LoRA layers are supported.")
            up, down = up.to(device, torch.float32).flatten(1), down.to(device, torch.float32).flatten(1)
            up = up * (strength * (float(alpha) / down.shape[0] if alpha is not None else 1.0))
            if offset is not None:
                full = up.new_zeros(rows, up.shape[1])
                full[offset[1]:offset[1] + offset[2]] = up
                up = full
            factors.setdefault(key, []).append((up, down))
        elif isinstance(patch, tuple) and patch[0] == "diff":
            d = patch[1][0].to(device, torch.float32) * strength
            if offset is not None:
                full = d.new_zeros((rows,) + d.shape[1:])
                full[offset[1]:offset[1] + offset[2]] = d
                d = full
            diffs[key] = diffs.get(key, 0) + d
        else:
            raise ValueError(f"HaDeS MOSS LoRA: '{lora_name}' has a {type(patch).__name__} layer on '{key}'; only plain LoRA and full diffs are supported.")
    return {k: (torch.cat([u for u, _ in fs], 1), torch.cat([d for _, d in fs], 0)) for k, fs in factors.items()}, diffs


def compress(up, down, max_rank, energy_target):
    """Exact SVD of up @ down through QR of the factors (no full matrix), truncated to max_rank or to the smallest rank that keeps
    energy_target of the energy. Returns (up, down, energy kept)."""
    qu, ru = torch.linalg.qr(up)
    qd, rd = torch.linalg.qr(down.T)
    u, s, vh = torch.linalg.svd(ru @ rd.T)
    energy = s.square()
    cum = torch.cumsum(energy, 0) / energy.sum().clamp(min=1e-12)
    r = min(max_rank, s.shape[0])
    if energy_target > 0.0:
        r = min(r, int(torch.searchsorted(cum, torch.tensor(energy_target, device=cum.device)).item()) + 1)
    sqrt_s = s[:r].sqrt()
    return (qu @ u[:, :r]) * sqrt_s, sqrt_s.unsqueeze(1) * (vh[:r] @ qd.T), float(cum[r - 1])


def save_lora(path, factors, diffs, max_rank, energy_target, metadata):
    """Compresses every factor pair and writes diffusion_model.* keys. Returns (energies, ranks)."""
    out, energies, ranks = {}, [], []
    for key, (up, down) in factors.items():
        if key in diffs:
            continue  # the core replaces factors with .diff on the same weight; export their exact sum below
        new_up, new_down, kept = compress(up, down, max_rank, energy_target)
        prefix = key.removesuffix(".weight")
        out[f"{prefix}.lora_up.weight"] = new_up.cpu().to(torch.bfloat16).contiguous()
        out[f"{prefix}.lora_down.weight"] = new_down.cpu().to(torch.bfloat16).contiguous()
        out[f"{prefix}.alpha"] = torch.tensor(float(new_up.shape[1]))
        energies.append(kept)
        ranks.append(new_up.shape[1])
    for key, d in diffs.items():
        if key in factors:
            up, down = factors[key]
            d = d + up @ down
        out[f"{key.removesuffix('.weight').removesuffix('.bias')}.{'diff_b' if key.endswith('.bias') else 'diff'}"] = d.cpu().to(torch.bfloat16).contiguous()
    safetensors.torch.save_file(out, path, metadata={"format": "pt", "ss_base_model_version": "moss_image", **metadata})
    return energies, ranks


def resize_summary(path, energies, ranks, energy_target, target_rank, n_diffs):
    if not ranks:
        return [f"Saved {os.path.basename(path)} ({os.path.getsize(path) / 1024 ** 2:.1f} MB) | {n_diffs} full diffs | no rank compression"], None
    mean = sum(energies) / len(energies)
    lines = [f"Saved {os.path.basename(path)} ({os.path.getsize(path) / 1024 ** 2:.1f} MB) | rank max {max(ranks)}, mean {sum(ranks) / len(ranks):.1f} "
             f"({'dynamic ' + format(energy_target, '.3f') if energy_target > 0.0 else 'fixed'}, cap {target_rank}) | {len(ranks)} LoRA layers, {n_diffs} full diffs",
             f"Energy kept: mean {mean:.1%}, worst layer {min(energies):.1%}"]
    return lines, mean


class HaDeSMossLoraResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model the LoRA is for; its layout (fused or split q/k/v) is the output's."}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "target_rank": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1, "tooltip": "Maximum rank per layer."}),
                "export_filename": ("STRING", {"default": "HaDeS_MOSS_Resized_LoRA", "tooltip": "Saved under models/loras; the source is never overwritten."}),
            },
            "optional": {
                "dynamic_energy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                                             "tooltip": "0 = fixed rank. Otherwise each layer gets the smallest rank that keeps this share of its energy, capped at target_rank."}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_lora_path", "info")
    FUNCTION = "resize"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Lowers a LoRA's rank with an exact SVD of its factors. Any MOSS LoRA format in, the model's layout out (split q/k/v becomes fused proj.qkv on fused files)."

    def resize(self, model, lora_name, target_rank, export_filename, dynamic_energy=0.0, overwrite_existing=False):
        device = comfy.model_management.get_torch_device()
        factors, diffs = lora_factors(lora_name, model, 1.0, device)
        if not factors:
            raise ValueError(f"HaDeS MOSS LoRA Resize: '{lora_name}' has no LoRA layer that applies to this model.")
        src = folder_paths.get_full_path_or_raise("loras", lora_name)
        path = lora_output_path(export_filename, "HaDeS_MOSS_Resized_LoRA", overwrite_existing)
        if os.path.exists(path) and os.path.samefile(path, src):
            raise ValueError("HaDeS MOSS LoRA Resize: the output would overwrite the source LoRA; choose another export_filename.")
        energies, ranks = save_lora(path, factors, diffs, target_rank, dynamic_energy,
                                    {"hades/source_lora": lora_name, "hades/target_rank": str(target_rank), "hades/dynamic_energy": str(dynamic_energy)})
        lines, mean = resize_summary(path, energies, ranks, dynamic_energy, target_rank, len(diffs))
        lines.insert(0, f"HaDeS MOSS LoRA Resize | {lora_name} ({os.path.getsize(src) / 1024 ** 2:.1f} MB)")
        return {"ui": {"text": [path]}, "result": (path, format_info("\n".join(lines), "lora_resize", saved_path=path, rank=target_rank,
                                                                      layers=len(ranks), energy_kept=round(mean, 4) if mean is not None else None))}


class HaDeSMossLoraMerger:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model the LoRAs are for; its layout (fused or split q/k/v) is the output's."}),
                "lora_a": (loras,),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_b": (loras,),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1,
                                 "tooltip": "Maximum rank per layer of the merged LoRA. The sum of both ranks keeps the merge exact."}),
                "export_filename": ("STRING", {"default": "HaDeS_MOSS_Merged_LoRA", "tooltip": "Saved under models/loras."}),
            },
            "optional": {
                "dynamic_energy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                                             "tooltip": "0 = fixed rank. Otherwise each layer gets the smallest rank that keeps this share of its energy, capped at rank."}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_lora_path", "info")
    FUNCTION = "merge"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = ("Merges two LoRAs into one file: weight_a * A + weight_b * B, compressed per layer with an exact SVD of the stacked factors. "
                   "At 1.0 / 1.0 and a rank of at least rank A + rank B it is the same as stacking both LoRAs.")

    def merge(self, model, lora_a, weight_a, lora_b, weight_b, rank, export_filename, dynamic_energy=0.0, overwrite_existing=False):
        device = comfy.model_management.get_torch_device()
        fa, da = lora_factors(lora_a, model, weight_a, device)
        fb, db = lora_factors(lora_b, model, weight_b, device)
        factors = {}
        for k in fa.keys() | fb.keys():
            parts = [f[k] for f in (fa, fb) if k in f]
            factors[k] = (torch.cat([u for u, _ in parts], 1), torch.cat([d for _, d in parts], 0))
        diffs = {k: da.get(k, 0) + db.get(k, 0) for k in da.keys() | db.keys()}
        if not factors and not diffs:
            raise ValueError("HaDeS MOSS LoRA Merger: neither LoRA applies to this model.")
        sources = [folder_paths.get_full_path_or_raise("loras", n) for n in (lora_a, lora_b)]
        path = lora_output_path(export_filename, "HaDeS_MOSS_Merged_LoRA", overwrite_existing)
        if os.path.exists(path) and any(os.path.samefile(path, src) for src in sources):
            raise ValueError("HaDeS MOSS LoRA Merger: the output would overwrite a source LoRA; choose another export_filename.")
        energies, ranks = save_lora(path, factors, diffs, rank, dynamic_energy,
                                    {"hades/lora_a": lora_a, "hades/lora_b": lora_b, "hades/weights": f"{weight_a},{weight_b}", "hades/rank": str(rank)})
        lines, mean = resize_summary(path, energies, ranks, dynamic_energy, rank, len(diffs))
        lines.insert(0, f"HaDeS MOSS LoRA Merger | {weight_a:+.2f} x {lora_a} + {weight_b:+.2f} x {lora_b} | layers only in A: {len(fa.keys() - fb.keys())}, "
                        f"only in B: {len(fb.keys() - fa.keys())}, shared: {len(fa.keys() & fb.keys())}")
        return {"ui": {"text": [path]}, "result": (path, format_info("\n".join(lines), "lora_merger", saved_path=path, rank=rank,
                                                                      layers=len(ranks), energy_kept=round(mean, 4) if mean is not None else None))}


NODE_CLASS_MAPPINGS = {
    "HaDeSMossLoraExtractor": HaDeSMossLoraExtractor,
    "HaDeSMossLoraBlockWeight": HaDeSMossLoraBlockWeight,
    "HaDeSMossMultiLora": HaDeSMossMultiLora,
    "HaDeSMossLoraMerger": HaDeSMossLoraMerger,
    "HaDeSMossLoraResize": HaDeSMossLoraResize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HaDeSMossLoraExtractor": "MOSS LoRA Extractor",
    "HaDeSMossLoraBlockWeight": "MOSS LoRA Block Weight",
    "HaDeSMossMultiLora": "MOSS Multi-LoRA",
    "HaDeSMossLoraMerger": "MOSS LoRA Merger",
    "HaDeSMossLoraResize": "MOSS LoRA Resize",
}
