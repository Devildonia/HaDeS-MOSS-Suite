import math
import zlib

import torch
import comfy.weight_adapter
from comfy.quant_ops import QuantizedTensor

from .moss_common import (PARTS, apply_block_merge, check_compatible, format_model_info, is_quantized, key_patches, key_patches_weight, logger,
                          model_weights, rescale_for, residual_scale, same_weight)

N_GROUPS = 7  # groups of 6 layers on the 42-layer model; other depths are split proportionally
RATIO = {"min": -1.0, "max": 2.0, "step": 0.01}


def group_name(g):
    lo, hi = g * 6, g * 6 + 5
    return f"layers_{lo:02d}_{hi:02d}"


GROUPS = [group_name(g) for g in range(N_GROUPS)]

# Share of the SFT -> Base change each group carries, min-max over layout / edges / texture / color and over swap-in (only that group
# from Base) and leave-out (Base without that group), tools/RESULTS.md
MEASURED = {
    "layers_00_05": "0.17-0.40", "layers_06_11": "0.10-0.28", "layers_12_17": "0.11-0.31", "layers_18_23": "0.20-0.48",
    "layers_24_29": "0.44-0.76 (strongest)", "layers_30_35": "0.37-0.65", "layers_36_41": "0.17-0.39",
    "text_embedder": "0.12-0.32", "time_modulation": "0.17-0.31", "io": "0.07-0.20 (weakest)",
}


def tooltip(name, what):
    return (f"{what} 0.0 = Model 1, 1.0 = Model 2. Measured on SFT -> Base, this group carries {MEASURED[name]} of the change "
            f"(taken alone or left out, per image metric; tools/RESULTS.md).")


def layer_group(i, n_layers):
    return i * N_GROUPS // n_layers


def parse_recipe(text, n_layers):
    """CSV of one ratio per layer followed by text_embedder, time_modulation, io."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != n_layers + len(PARTS):
        raise ValueError(f"HaDeS MOSS Merger: recipe_override needs {n_layers + len(PARTS)} comma-separated values "
                         f"({n_layers} layers, then {', '.join(PARTS)}), got {len(parts)}.")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"HaDeS MOSS Merger: recipe_override must be numbers only, got {[p for p in parts if not p.replace('.', '', 1).lstrip('-').isdigit()][:3]}.") from None
    if not all(math.isfinite(v) for v in values):
        raise ValueError("HaDeS MOSS Merger: recipe_override values must be finite numbers.")
    return values[:n_layers], dict(zip(PARTS, values[n_layers:]))


class HaDeSMossModelBlockMerger:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"model1": ("MODEL",), "model2": ("MODEL",)}
        for g in GROUPS:
            required[g] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip(g, f"Ratio for {g.replace('_', ' ', 1).replace('_', '-')}.")})
        required["text_embedder"] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip("text_embedder", "Projection of the Qwen3.5 hidden states and its 2 text blocks.")})
        required["time_modulation"] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip("time_modulation", "time_embedding and shared_block_modulation (the one modulation shared by every layer).")})
        required["io"] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip("io", "video_embedder (latent in) and the output head (final norm, linear and modulation).")})
        return {
            "required": required,
            "optional": {
                "recipe_override": ("STRING", {"default": "", "tooltip": "Optional per-layer recipe: one ratio per layer (42), then text_embedder, time_modulation, io. Replaces the sliders when set."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "merge"
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Merges two MOSS-Image models per layer group, lazily: no merged copy in RAM, the merge is computed when the model loads."

    def merge(self, model1, model2, text_embedder, time_modulation, io, recipe_override="", **groups):
        n_layers = len(model1.model.diffusion_model.layers)
        if recipe_override.strip():
            layer_ratios, part_ratios = parse_recipe(recipe_override, n_layers)
            mode = "recipe"
        else:
            layer_ratios = [groups[GROUPS[layer_group(i, n_layers)]] for i in range(n_layers)]
            part_ratios = {"text_embedder": text_embedder, "time_modulation": time_modulation, "io": io}
            mode = "groups"
        m, merged, n_layers, quantized, scale = apply_block_merge(model1, model2, layer_ratios, part_ratios)
        text = f"HaDeS MOSS Model Merged ({mode}) | {n_layers} layers | modified tensors: {merged}"
        if scale is not None:
            text += (f"\nmodel2 stores its residual stream scaled by {1 / scale:.4g} relative to model1 (an fp16-fixed file); "
                     f"it was brought to model1's scale before merging, which gives the same model2.")
        if quantized:
            text += (f"\nWARNING: {' and '.join(quantized)} quantized (int8 / fp8). Their dequantized weights were merged: "
                     f"the rounding of the quantized file is kept, the original weights are not recovered.")
            if "model1" in quantized:
                text += " The result is quantized again into model1's format; connect the bf16 model as model1 to keep it in bf16."
        return (m, format_model_info(text, "model_block_merger", m, mode=mode, n_layers=n_layers, modified_tensors=merged, quantized=quantized,
                                     model2_residual_rescale=scale, layer_ratios=layer_ratios, part_ratios=part_ratios,
                                     sources={"model1": model1.get_attachment("hades_moss_recipe"),
                                              "model2": model2.get_attachment("hades_moss_recipe")}))


# ==============================================================================
# MULTI-MODEL MERGER (TIES / DARE / TASK ARITHMETIC / MODEL STOCK / SLERP)
# ==============================================================================
MULTI_MERGE_METHODS = ("TIES", "DARE-TIES", "DARE", "Task Arithmetic", "Model Stock", "SLERP")


def ties_trim(tau, density):
    """Keep the `density` fraction of entries with the largest magnitude."""
    k = int(tau.numel() * density)
    if k >= tau.numel():
        return tau
    if k <= 0:
        return torch.zeros_like(tau)
    # The (n - k)-th smallest magnitude; a sort is ~40x faster than kthvalue on CUDA for these sizes
    threshold = tau.abs().flatten().sort().values[tau.numel() - k - 1]
    return tau * (tau.abs() > threshold)


def dare_drop(tau, density, generator):
    """Drop entries with probability 1 - density and rescale the rest, so the expected task vector is unchanged."""
    keep = torch.rand(tau.shape, generator=generator, device=tau.device) < density
    return tau * keep / density


def sign_consensus(taus):
    """TIES merge: elect each entry's sign from the summed task vectors, then average only the agreeing ones."""
    stacked = torch.stack(taus)
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    return (stacked * agree).sum(0) / agree.sum(0).clamp(min=1)


def slerp(a, b, t):
    """Spherical interpolation; linear for zero-length or nearly collinear inputs."""
    if t == 0.0:
        return a
    if t == 1.0:
        return b
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).clamp(-1.0, 1.0)
    c = float(cos)
    angle = math.acos(c)
    if angle < 1e-4 or math.pi - angle < 1e-4 or (c == 0.0 and (a.norm() == 0 or b.norm() == 0)):
        return a + (b - a) * t
    omega = torch.arccos(cos)
    so = torch.sin(omega)
    return a * (torch.sin((1.0 - t) * omega) / so) + b * (torch.sin(t * omega) / so)


def multi_merge_tensor(method, base, thetas, weights, density, scale, generators=None):
    """One tensor of a multi-model merge, in float32. `base` is None for SLERP; every other method works on task
    vectors (model - base). Weighted methods are normalized by the weight sum, so scale 1.0 has the strength of an
    average and higher values extrapolate."""
    if method == "SLERP":
        return slerp(thetas[0], thetas[1], weights[1] / (weights[0] + weights[1]))
    taus = [theta - base for theta in thetas]
    if method == "Model Stock":
        # Jang et al. 2024: interpolate between the average and the base by the angle between the task vectors
        n = len(taus)
        flat = [tau.flatten() for tau in taus]
        cos = [float(torch.nn.functional.cosine_similarity(flat[i], flat[j], dim=0)) for i in range(n) for j in range(i + 1, n)]
        c = max(0.0, sum(cos) / len(cos))
        t = n * c / (1.0 + (n - 1) * c)
        return base + t * torch.stack(taus).mean(0)
    total = sum(weights)
    n = sum(w != 0 for w in weights) if method in ("TIES", "DARE-TIES") else len(weights)
    taus = [tau * (w * n / total) for tau, w in zip(taus, weights)]
    if method in ("DARE", "DARE-TIES"):
        taus = [dare_drop(tau, density, g) for tau, g in zip(taus, generators)]
    if method == "TIES":
        taus = [ties_trim(tau, density) for tau in taus]
    if method in ("TIES", "DARE-TIES"):
        return base + scale * sign_consensus(taus)
    return base + scale * torch.stack(taus).mean(0)


class MossMultiMergeAdapter(comfy.weight_adapter.WeightAdapterBase):
    """Lazy multi-model merge patch, evaluated at load time like MossMergeAdapter (no merged copies in RAM)."""
    name = "hades_moss_multi_merge"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        # (base key patches or None, model key patches, model rescales, model weights, method, density, scale, seed, storage dtype)
        self.weights = weights

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        base_kp, model_kps, rescales, model_weights_, method, density, scale, seed, storage_dtype = self.weights
        device = weight.device
        base = key_patches_weight(base_kp, device, key) if base_kp is not None else None
        thetas = [key_patches_weight(kp, device, key) for kp in model_kps]
        thetas = [t if r is None else t * r.to(device) for t, r in zip(thetas, rescales)]
        generators = None
        if method in ("DARE", "DARE-TIES"):
            # Same seed -> same drop mask on every load (per key and per model)
            key_seed = zlib.crc32(key.encode("utf-8"))
            generators = [torch.Generator(device=device).manual_seed((seed + key_seed + 1000003 * i) % (1 << 63)) for i in range(len(thetas))]
        return multi_merge_tensor(method, base, thetas, model_weights_, density, scale, generators).to(storage_dtype).to(weight.dtype)


class HaDeSMossMultiModelMerger:
    @classmethod
    def INPUT_TYPES(cls):
        weight = lambda n: ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": f"Relative weight of model_{n} (normalized by the sum of the connected models' weights)."})
        return {
            "required": {
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
                "method": (list(MULTI_MERGE_METHODS), {
                    "default": "TIES",
                    "tooltip": "TIES: trim each task vector to its largest entries, elect a sign per entry, average the agreeing ones. "
                               "DARE: randomly drop entries and rescale the rest (DARE-TIES then elects signs like TIES). "
                               "Task Arithmetic: weighted average of the task vectors. Model Stock: average pulled toward base_model "
                               "by the angle between the task vectors (ignores weights, density and scale). "
                               "SLERP: spherical blend of model_1 and model_2 only, t = weight_2 / (weight_1 + weight_2); needs no base_model.",
                }),
                "density": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                      "tooltip": "TIES: fraction of each task vector kept (largest entries). DARE: probability of keeping each entry."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Strength of the merged task vector on top of base_model. 1.0 has the strength of an average; above 1.0 extrapolates."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "DARE drop mask. The same seed gives the same merge on the same kind of device."}),
                "weight_1": weight(1),
                "weight_2": weight(2),
            },
            "optional": {
                "base_model": ("MODEL", {"tooltip": "The checkpoint the models were finetuned from (SFT or Base); when they come from both, "
                                                    "Base, which SFT was trained from. Required by every method except SLERP."}),
                "model_3": ("MODEL",),
                "model_4": ("MODEL",),
                "weight_3": weight(3),
                "weight_4": weight(4),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "merge"
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Merges 2-4 MOSS-Image models with TIES, DARE, Task Arithmetic, Model Stock or SLERP, lazily (no merged copy in RAM)."

    def merge(self, model_1, model_2, method, density, scale, seed, weight_1, weight_2,
              base_model=None, model_3=None, model_4=None, weight_3=1.0, weight_4=1.0):
        models = [(f"model_{i}", m, w) for i, (m, w) in enumerate(((model_1, weight_1), (model_2, weight_2), (model_3, weight_3), (model_4, weight_4)), 1)
                  if m is not None]
        if method == "SLERP":
            if len(models) != 2:
                raise ValueError("HaDeS MOSS Multi-Model Merger: SLERP blends exactly model_1 and model_2; disconnect model_3 / model_4.")
            if weight_1 + weight_2 <= 0:
                raise ValueError("HaDeS MOSS Multi-Model Merger: SLERP needs weight_1 + weight_2 > 0.")
            anchor, anchor_name = model_1, "model_1"
        else:
            if base_model is None:
                raise ValueError(f"HaDeS MOSS Multi-Model Merger: {method} needs base_model, the checkpoint the models were finetuned from.")
            if sum(w for _, _, w in models) <= 0:
                raise ValueError("HaDeS MOSS Multi-Model Merger: the model weights add up to 0.")
            anchor, anchor_name = base_model, "base_model"

        m = anchor.clone()
        w0s = model_weights(m)
        wss = [model_weights(mm) for _, mm, _ in models]
        n_layers = check_compatible([(anchor_name, w0s)] + [(name, ws) for (name, _, _), ws in zip(models, wss)])
        # every model is brought to the anchor's residual scale (fp16-fixed files)
        scales = [residual_scale(w0s, ws) for ws in wss]
        quantized = [name for (name, _, _), ws in zip(models, wss) if is_quantized(ws)]
        if is_quantized(w0s):
            quantized.insert(0, anchor_name)
        model_weights_ = tuple(float(w) for _, _, w in models)

        merged = 0
        for k, w0 in w0s.items():
            if any(k not in ws for ws in wss):
                continue
            patch_key = f"diffusion_model.{k}"
            exact = not any(isinstance(w, QuantizedTensor) for w in [w0] + [ws[k] for ws in wss])
            patched = patch_key in m.patches or any(patch_key in mm.patches for _, mm, _ in models)
            if exact and not patched and all(s is None for s in scales) and all(same_weight(w0, ws[k]) for ws in wss):
                continue
            m.add_patches({patch_key: MossMultiMergeAdapter(set(), (
                key_patches(m, patch_key) if method != "SLERP" else None,
                tuple(key_patches(mm, patch_key) for _, mm, _ in models),
                tuple(rescale_for(k, ws[k], s) if s is not None else None for ws, s in zip(wss, scales)),
                model_weights_, method, float(density), float(scale), int(seed),
                torch.bfloat16 if isinstance(w0, QuantizedTensor) else w0.dtype,
            ))}, 1.0)
            merged += 1
        if merged == 0:
            raise ValueError("HaDeS MOSS Multi-Model Merger: the models are identical to the base in every merged tensor.")

        params = "weights, density and scale unused" if method == "Model Stock" else (
            f"t = {model_weights_[1] / sum(model_weights_):.2f}" if method == "SLERP" else f"density {density:.2f} | scale {scale:.2f}")
        text = (f"HaDeS MOSS Multi-Model Merge | {method} | {len(models)} models | {params} | "
                f"weights {', '.join(f'{w:.2f}' for w in model_weights_)} | {n_layers} layers | modified tensors: {merged}")
        rescaled = [f"{name} x{s:.4g}" for (name, _, _), s in zip(models, scales) if s is not None]
        if rescaled:
            text += f"\nResidual stream brought to {anchor_name}'s scale (fp16-fixed files): {', '.join(rescaled)}."
        if quantized:
            text += (f"\nWARNING: {', '.join(quantized)} quantized (int8 / fp8); their dequantized weights were merged, rounding included.")
            if method in ("TIES", "DARE-TIES", "DARE") and any(q != anchor_name for q in quantized):
                text += (f" {method} trims task vectors entry by entry, and a quantized model's task vector carries its rounding: in layers "
                         f"its finetune did not change it is rounding alone (tools/RESULTS.md).")
            if anchor_name in quantized:
                text += f" The result is quantized again into {anchor_name}'s format; use a bf16 {anchor_name} to keep it in bf16."
            logger.warning(f"HaDeS MOSS Multi-Model Merger: {', '.join(quantized)} quantized; merging their dequantized weights.")
        sources = {name: mm.get_attachment("hades_moss_recipe") for name, mm, _ in models}
        if method != "SLERP":
            sources["base_model"] = base_model.get_attachment("hades_moss_recipe")
        return (m, format_model_info(text, "multi_model_merger", m, method=method, models=len(models), density=float(density), scale=float(scale),
                                     modified_tensors=merged, quantized=quantized, rescaled=rescaled,
                                     weights=model_weights_, seed=int(seed), n_layers=n_layers, sources=sources))


NODE_CLASS_MAPPINGS = {
    "HaDeSMossModelBlockMerger": HaDeSMossModelBlockMerger,
    "HaDeSMossMultiModelMerger": HaDeSMossMultiModelMerger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HaDeSMossModelBlockMerger": "MOSS Model Block Merger",
    "HaDeSMossMultiModelMerger": "MOSS Multi-Model Merger",
}
