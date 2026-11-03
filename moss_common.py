import json
import logging
import re

import torch
import comfy.lora
import comfy.model_management
import comfy.weight_adapter
from comfy.quant_ops import QuantizedTensor

logger = logging.getLogger("HaDeSMossSuite")

# ==============================================================================
# INFO CONTRACT - machine-readable trailer on every node's `info` output
# ==============================================================================
HADES_META_PREFIX = "HADES_META "


def format_info(text, node, **meta):
    return f"{text}\n{HADES_META_PREFIX}{json.dumps({'node': node, **meta}, separators=(',', ':'))}"


def parse_info_meta(text):
    """Return the HADES_META trailer of an info string, or None for foreign strings."""
    for line in reversed(str(text).splitlines()):
        if line.startswith(HADES_META_PREFIX):
            try:
                return json.loads(line[len(HADES_META_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


# ==============================================================================
# MOSS-IMAGE TOPOLOGY
# ==============================================================================
# Everything outside `layers.N` falls in one of three parts. They are named after what they are, not after what they
# are supposed to do: what each part changes in the image is measured with tools/moss_eval.py.
PARTS = {
    "text_embedder": ("text_embedder.",),
    "time_modulation": ("time_embedding.", "shared_block_modulation."),
    "io": ("video_embedder.", "final_norm_video.", "final_linear_video.", "video_head_modulation."),
}
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def count_layers(keys):
    return 1 + max((int(m.group(1)) for k in keys if (m := _LAYER_RE.match(k))), default=-1)


def key_slot(key):
    """'layers.N.*' -> N, a PARTS name for the rest, None for keys of no part (e.g. rope buffers)."""
    m = _LAYER_RE.match(key)
    if m:
        return int(m.group(1))
    for part, prefixes in PARTS.items():
        if key.startswith(prefixes):
            return part
    return None


def require_unquantized(key, *weights):
    """Stored int8 / ConvRot / fp8 weights are not values that can be blended."""
    for w in weights:
        if isinstance(w, QuantizedTensor) or not w.is_floating_point() or w.element_size() < 2:
            raise ValueError(
                f"HaDeS MOSS Merger: '{key}' is quantized ({w.dtype}); load both models unquantized "
                f"(the bf16 checkpoints) to merge them."
            )


# ==============================================================================
# LAZY MERGE PATCH
# ==============================================================================
def same_weight(a, b):
    """Exact equality of two stored weights that may sit on different devices (one model already loaded on the GPU)."""
    return torch.equal(a, b.to(a.device))


def key_patches_weight(key_patches, device, key):
    """One get_key_patches() entry baked like a saved model: patches applied in float32, rounded to the stored dtype."""
    (w, convert), patches = key_patches[0], key_patches[1:]
    baked = comfy.lora.calculate_weight(patches, convert(comfy.model_management.cast_to_device(w, device, torch.float32, copy=True), inplace=True), key)
    return baked.to(w.dtype).float()


class MossMergeAdapter(comfy.weight_adapter.WeightAdapterBase):
    """Lazy merge patch: (1 - ratio) * model1 + ratio * model2, each with its own patches, computed in float32 at load time
    and rounded to model1's storage dtype, so the loaded weights match a merge baked into that dtype. Both sides start from
    the stored tensors: the incoming weight may already be cast to a lossy compute dtype."""
    name = "hades_moss_merge"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights  # (model1 key patches, model2 key patches, ratio, model1 storage dtype)

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        key_patches1, key_patches2, ratio, storage_dtype = self.weights
        w1 = key_patches_weight(key_patches1, weight.device, key)
        w2 = key_patches_weight(key_patches2, weight.device, key)
        return (w1 + (w2 - w1) * ratio).to(storage_dtype).to(weight.dtype)


def apply_block_merge(model1, model2, layer_ratios, part_ratios):
    """Merge model2 into model1 with one ratio per `layers.N` and per PARTS entry. Returns (model, merged tensors, layers)."""
    m = model1.clone()
    sd1 = m.model.diffusion_model.state_dict()
    sd2 = model2.model.diffusion_model.state_dict()
    n_layers = count_layers(sd1.keys())
    if n_layers != count_layers(sd2.keys()):
        raise ValueError(f"HaDeS MOSS Merger: model1 has {n_layers} layers, model2 has {count_layers(sd2.keys())}.")
    if n_layers != len(layer_ratios):
        raise ValueError(f"HaDeS MOSS Merger: got {len(layer_ratios)} layer ratios for a {n_layers}-layer model.")
    fused1, fused2 = ("layers.0.proj.qkv.weight" in sd1), ("layers.0.proj.qkv.weight" in sd2)
    if fused1 != fused2:
        raise ValueError("HaDeS MOSS Merger: one model stores q/k/v fused (proj.qkv) and the other split (proj.q/k/v); "
                         "load both from files of the same layout.")

    key_patches1 = m.get_key_patches("diffusion_model.")
    key_patches2 = model2.get_key_patches("diffusion_model.")
    merged = 0
    for k, w1 in sd1.items():
        slot = key_slot(k)
        if slot is None or k not in sd2:
            continue
        w2 = sd2[k]
        if w1.shape != w2.shape:
            raise ValueError(f"HaDeS MOSS Merger: shape mismatch on '{k}': {tuple(w1.shape)} vs {tuple(w2.shape)}.")
        require_unquantized(k, w1, w2)
        ratio = layer_ratios[slot] if isinstance(slot, int) else part_ratios[slot]
        if ratio == 0.0:
            continue
        patch_key = f"diffusion_model.{k}"
        if patch_key not in m.patches and patch_key not in model2.patches and same_weight(w1, w2):
            continue
        m.add_patches({patch_key: MossMergeAdapter(set(), (key_patches1[patch_key], key_patches2[patch_key], ratio, w1.dtype))}, 1.0)
        merged += 1
    return m, merged, n_layers
