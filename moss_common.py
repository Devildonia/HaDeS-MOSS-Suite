import json
import logging
import re

import torch
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.utils
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


def model_weights(model):
    """Merge-able parameters of a MOSS MODEL by key (no 'diffusion_model.' prefix). Quantized layers (int8 ConvRot, fp8)
    come back as their QuantizedTensor, not as the raw int8 / weight_scale / comfy_quant pieces of the state_dict."""
    dm = model.model.diffusion_model
    return {k: comfy.utils.get_attr(dm, k) for k in dm.state_dict() if k.endswith((".weight", ".bias")) and key_slot(k) is not None}


def require_float(key, w):
    if not isinstance(w, QuantizedTensor) and not w.is_floating_point():
        raise ValueError(f"HaDeS MOSS Merger: '{key}' is stored as {w.dtype} without a quantization layout ComfyUI can dequantize.")


# ==============================================================================
# LAZY MERGE PATCH
# ==============================================================================
def same_weight(a, b):
    """Exact equality of two stored weights that may sit on different devices (one model already loaded on the GPU)."""
    return torch.equal(a, b.to(a.device))


def key_patches(model, key):
    """get_key_patches() for one key. The ModelPatcher method itself fails on quantized models: it reads every state_dict
    entry as a module attribute, comfy_quant included."""
    weight, _, convert = comfy.model_patcher.get_key_weight(model.model, key)
    if key in model.backup:
        weight = model.backup[key].weight
    if key in model.hook_backup:
        weight = model.hook_backup[key][0]
    return [(weight, convert or (lambda a, **kwargs: a))] + model.patches.get(key, [])


def key_patches_weight(key_patches, device, key):
    """One key_patches() entry baked like a saved model: patches applied in float32, rounded to the stored dtype.
    Quantized weights are dequantized first, so the result is the rounded (lossy) weight, not the pre-quantization one."""
    (w, convert), patches = key_patches[0], key_patches[1:]
    w32 = comfy.model_management.cast_to_device(w, device, torch.float32, copy=True)
    if isinstance(w32, QuantizedTensor):
        w32 = w32.dequantize().float()
    baked = comfy.lora.calculate_weight(patches, convert(w32, inplace=True), key)
    return baked.to(torch.bfloat16 if isinstance(w, QuantizedTensor) else w.dtype).float()


class MossMergeAdapter(comfy.weight_adapter.WeightAdapterBase):
    """Lazy merge patch: (1 - ratio) * model1 + ratio * model2, each with its own patches, computed in float32 at load time
    and rounded to model1's storage dtype, so the loaded weights match a merge baked into that dtype. Both sides start from
    the stored tensors: the incoming weight may already be cast to a lossy compute dtype."""
    name = "hades_moss_merge"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights  # (model1 key patches, model2 key patches, ratio, model1 storage dtype, model2 rescale or None)

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        key_patches1, key_patches2, ratio, storage_dtype, rescale2 = self.weights
        w1 = key_patches_weight(key_patches1, weight.device, key)
        w2 = key_patches_weight(key_patches2, weight.device, key)
        if rescale2 is not None:
            w2 = w2 * rescale2.to(w2.device)
        return (w1 + (w2 - w1) * ratio).to(storage_dtype).to(weight.dtype)


# ==============================================================================
# RESIDUAL STREAM SCALE ("fp16-fixed" files)
# ==============================================================================
# Every MOSS norm is an RMSNorm, so scaling all writes into the residual stream by one factor gives the same model.
# fp16 conversions use that to stay in fp16 range (miaomiaoHaremMOSS_baseV11_fp16: x 1/16, tools/RESULTS.md).
# The writes: the latent embedder, the text embedder's projection and block outputs, and the two gates of the shared modulation.
_RESIDUAL_WRITER_RE = re.compile(r"^(video_embedder|text_embedder\.proj|text_embedder\.blocks\.\d+\.(proj|ffn_down))\.(weight|bias)$")
_MODULATION_GATES = (2, 5)  # shared_block_modulation chunks: shift, scale, gate (attention), shift, scale, gate (ffn)


def residual_scale(w1s, w2s):
    """Factor that brings model2's residual stream to model1's (least squares on the latent embedder), or None when they match."""
    a = w1s["video_embedder.weight"]
    b = w2s["video_embedder.weight"]
    a = (a.dequantize() if isinstance(a, QuantizedTensor) else a).float().cpu()
    b = (b.dequantize() if isinstance(b, QuantizedTensor) else b).float().cpu()
    s = ((a * b).sum() / (b * b).sum()).item()
    # finetunes move this weight by well under 1%; a residual rescale moves it by a large factor
    return s if abs(s - 1.0) > 0.05 else None


def rescale_for(key, w, s):
    """model2's rescale factor for one key: a scalar, per-row factors for the modulation, None for keys off the residual stream."""
    if _RESIDUAL_WRITER_RE.match(key):
        return torch.tensor(s)
    if key.startswith("shared_block_modulation."):
        rows = torch.ones(w.shape[0])
        for c in _MODULATION_GATES:
            rows.view(6, -1)[c] = s
        return rows.view(-1, *([1] * (w.ndim - 1)))
    return None


def apply_block_merge(model1, model2, layer_ratios, part_ratios):
    """Merge model2 into model1 with one ratio per `layers.N` and per PARTS entry.
    model2's residual stream is brought to model1's scale first (fp16-fixed files).
    Returns (model, merged tensors, layers, quantized side names, model2 residual rescale or None)."""
    m = model1.clone()
    w1s, w2s = model_weights(m), model_weights(model2)
    n_layers = count_layers(w1s.keys())
    if n_layers != count_layers(w2s.keys()):
        raise ValueError(f"HaDeS MOSS Merger: model1 has {n_layers} layers, model2 has {count_layers(w2s.keys())}.")
    if n_layers != len(layer_ratios):
        raise ValueError(f"HaDeS MOSS Merger: got {len(layer_ratios)} layer ratios for a {n_layers}-layer model.")
    if ("layers.0.proj.qkv.weight" in w1s) != ("layers.0.proj.qkv.weight" in w2s):
        raise ValueError("HaDeS MOSS Merger: one model stores q/k/v fused (proj.qkv) and the other split (proj.q/k/v); "
                         "load both from files of the same layout.")

    quantized = [name for name, ws in (("model1", w1s), ("model2", w2s)) if any(isinstance(w, QuantizedTensor) for w in ws.values())]
    scale = residual_scale(w1s, w2s)
    merged = 0
    for k, w1 in w1s.items():
        if k not in w2s:
            continue
        w2 = w2s[k]
        if w1.shape != w2.shape:
            raise ValueError(f"HaDeS MOSS Merger: shape mismatch on '{k}': {tuple(w1.shape)} vs {tuple(w2.shape)}.")
        require_float(k, w1)
        require_float(k, w2)
        slot = key_slot(k)
        ratio = layer_ratios[slot] if isinstance(slot, int) else part_ratios[slot]
        if ratio == 0.0:
            continue
        patch_key = f"diffusion_model.{k}"
        exact = not isinstance(w1, QuantizedTensor) and not isinstance(w2, QuantizedTensor)
        if exact and patch_key not in m.patches and patch_key not in model2.patches and same_weight(w1, w2):
            continue
        storage = torch.bfloat16 if isinstance(w1, QuantizedTensor) else w1.dtype
        rescale = rescale_for(k, w2, scale) if scale is not None else None
        m.add_patches({patch_key: MossMergeAdapter(set(), (key_patches(m, patch_key), key_patches(model2, patch_key), ratio, storage, rescale))}, 1.0)
        merged += 1
    if quantized:
        logger.warning(f"HaDeS MOSS Merger: {' and '.join(quantized)} quantized; merging their dequantized (rounded) weights.")
    if scale is not None:
        logger.info(f"HaDeS MOSS Merger: model2's residual stream rescaled by {scale:.4g} to match model1.")
    return m, merged, n_layers, quantized, scale
