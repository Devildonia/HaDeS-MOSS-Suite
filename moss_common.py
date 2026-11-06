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


def key_patches_float(key_patches, device, key):
    """One key_patches() entry with its patches applied in float32, not rounded. Quantized weights are dequantized first."""
    (w, convert), patches = key_patches[0], key_patches[1:]
    w32 = comfy.model_management.cast_to_device(w, device, torch.float32, copy=True)
    if isinstance(w32, QuantizedTensor):
        w32 = w32.dequantize().float()
    return comfy.lora.calculate_weight(patches, convert(w32, inplace=True), key)


def key_patches_weight(key_patches, device, key):
    """One key_patches() entry baked like a saved model: patches applied in float32, rounded to the stored dtype.
    Quantized weights are dequantized first, so the result is the rounded (lossy) weight, not the pre-quantization one."""
    w = key_patches[0][0]
    return key_patches_float(key_patches, device, key).to(torch.bfloat16 if isinstance(w, QuantizedTensor) else w.dtype).float()


def stochastic_round_bf16(x, seed):
    """float32 -> bfloat16, each element rounded up or down to a neighbouring bfloat16 value with probability proportional to its
    distance (unbiased: the expected value is x). ComfyUI's stochastic_rounding() rounds bfloat16 to nearest, which drops most
    of a change smaller than half a bfloat16 step. Seeded, so the same inputs give the same file."""
    bits = x.float().contiguous().view(torch.int32)
    gen = torch.Generator(device=bits.device).manual_seed(seed)
    noise = torch.randint(0, 1 << 16, bits.shape, generator=gen, device=bits.device, dtype=torch.int32)
    out = ((bits + noise) & -65536).view(torch.float32)
    # inf / nan, and the few finite values the carry would push to inf, keep round-to-nearest
    keep = ~torch.isfinite(out) | ~torch.isfinite(x)
    if keep.any():
        out = torch.where(keep, x.float().to(torch.bfloat16).float(), out)
    return out.to(torch.bfloat16)


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
    """Factor that brings the second model's residual stream to the first's (least squares on the latent embedder), or None
    when they match."""
    a = w1s["video_embedder.weight"]
    b = w2s["video_embedder.weight"]
    a = (a.dequantize() if isinstance(a, QuantizedTensor) else a).float().cpu()
    b = (b.dequantize() if isinstance(b, QuantizedTensor) else b).float().cpu()
    s = ((a * b).sum() / (b * b).sum()).item()
    # finetunes move this weight by well under 1%; a residual rescale moves it by a large factor and leaves it otherwise equal
    # (miaomiao: residual 0.0000 after x16). Unrelated weights also fit some small s, but leave a residual near 1.
    residual = ((a - s * b).norm() / a.norm().clamp(min=1e-12)).item()
    return s if abs(s - 1.0) > 0.05 and residual < 0.1 else None


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


def is_quantized(ws):
    return any(isinstance(w, QuantizedTensor) for w in ws.values())


def check_compatible(names_ws):
    """Same depth, same q/k/v layout and same shapes for every (name, weights) pair. Returns the number of layers."""
    (name0, ws0), rest = names_ws[0], names_ws[1:]
    n_layers = count_layers(ws0.keys())
    for name, ws in rest:
        if count_layers(ws.keys()) != n_layers:
            raise ValueError(f"HaDeS MOSS Merger: {name0} has {n_layers} layers, {name} has {count_layers(ws.keys())}.")
        if ("layers.0.proj.qkv.weight" in ws0) != ("layers.0.proj.qkv.weight" in ws):
            raise ValueError(f"HaDeS MOSS Merger: {name0} and {name} store q/k/v differently (fused proj.qkv vs split proj.q/k/v); "
                             "load them from files of the same layout.")
        for k, w in ws.items():
            if k in ws0 and w.shape != ws0[k].shape:
                raise ValueError(f"HaDeS MOSS Merger: shape mismatch on '{k}': {tuple(ws0[k].shape)} ({name0}) vs {tuple(w.shape)} ({name}).")
    for name, ws in names_ws:
        for k, w in ws.items():
            require_float(k, w)
    return n_layers


def apply_block_merge(model1, model2, layer_ratios, part_ratios):
    """Merge model2 into model1 with one ratio per `layers.N` and per PARTS entry.
    model2's residual stream is brought to model1's scale first (fp16-fixed files).
    Returns (model, merged tensors, layers, quantized side names, model2 residual rescale or None)."""
    m = model1.clone()
    w1s, w2s = model_weights(m), model_weights(model2)
    n_layers = check_compatible([("model1", w1s), ("model2", w2s)])
    if n_layers != len(layer_ratios):
        raise ValueError(f"HaDeS MOSS Merger: got {len(layer_ratios)} layer ratios for a {n_layers}-layer model.")

    quantized = [name for name, ws in (("model1", w1s), ("model2", w2s)) if is_quantized(ws)]
    scale = residual_scale(w1s, w2s)
    merged = 0
    for k, w1 in w1s.items():
        if k not in w2s:
            continue
        w2 = w2s[k]
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
