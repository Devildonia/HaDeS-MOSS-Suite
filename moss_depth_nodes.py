import importlib
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import folder_paths

from .moss_common import format_info

CKPTS = ["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"]
AUX_HINT = ("MOSS Depth Map needs the comfyui_controlnet_aux node pack (ComfyUI Manager: \"ComfyUI's ControlNet Auxiliary Preprocessors\") "
            "and its Depth Anything V2 weights; run its Depth Anything V2 node once to download them.")
_MODELS = {}  # ckpt -> (file identity, detector); one detector per checkpoint, kept on the CPU between runs


def aux_modules():
    """comfyui_controlnet_aux's Depth Anything V2 module and its DINOv2 attention module, or a clear error. The pack adds its src
    folder to sys.path when ComfyUI loads it; if that has not happened, the folder is looked up among the custom node paths."""
    try:
        da = importlib.import_module("custom_controlnet_aux.depth_anything_v2")
    except ImportError:
        for base in folder_paths.get_folder_paths("custom_nodes"):
            src = os.path.join(base, "comfyui_controlnet_aux", "src")
            if os.path.isdir(src) and src not in sys.path:
                sys.path.insert(0, src)
        try:
            da = importlib.import_module("custom_controlnet_aux.depth_anything_v2")
        except ImportError as e:
            raise RuntimeError(AUX_HINT) from e
    att = importlib.import_module("custom_controlnet_aux.depth_anything_v2.dinov2_layers.attention")
    return da, att


def weights_path(ckpt):
    """Where the pack keeps the weights (its annotator ckpts folder / the Hugging Face repo name / the file). Never downloaded here."""
    util = importlib.import_module("custom_controlnet_aux.util")
    root = os.environ.get("AUX_ANNOTATOR_CKPTS_PATH") or getattr(util, "annotator_ckpts_path", "")
    return os.path.join(root, util.DEPTH_ANYTHING_V2_MODEL_NAME_DICT[ckpt], ckpt)


def file_identity(path):
    st = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns)


def single_scale_forward(self, x, attn_bias=None):
    """DINOv2 attention with the 1 / sqrt(head_dim) scale applied once, by scaled_dot_product_attention (memory-efficient kernels when
    the backend has them). The pack's MemEffAttention scales q and then calls an attention that scales again: the encoder sees a
    blur and the map loses the outline (tools/RESULTS.md)."""
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias).transpose(1, 2).reshape(b, n, c)
    return self.proj_drop(self.proj(x))


def fix_attention(model, att):
    """Gives this model's own attention blocks the single-scale forward, as instance methods: the pack's classes, and every other
    model built from them, stay as they are (audit 0.9.9 M3). Returns the number of blocks changed."""
    n = 0
    for module in model.modules():
        if isinstance(module, att.Attention):
            module.forward = types.MethodType(single_scale_forward, module)
            n += 1
    return n


def detector(da, att, ckpt, path):
    """The detector for this file: loaded from the resolved local path (never downloaded) and reloaded when the file under that name
    changes, the old one dropped (audit 0.9.9 M4)."""
    ident = file_identity(path)
    cached = _MODELS.get(ckpt)
    if cached is not None and cached[0] == ident:
        return cached[1]
    _MODELS.pop(ckpt, None)
    # the pack joins its ckpts folder with this argument; an absolute folder makes it read exactly the file checked here
    det = da.DepthAnythingV2Detector.from_pretrained(os.path.dirname(path), filename=ckpt)
    if fix_attention(det.model, att) == 0:
        raise RuntimeError("MOSS Depth Map: no DINOv2 attention block found in the Depth Anything V2 model; this version of "
                           "comfyui_controlnet_aux is not the one the node was written for.")
    _MODELS[ckpt] = (ident, det)
    return det


class HaDeSMossDepthMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photo or render whose pose you want to keep."}),
                "ckpt_name": (CKPTS, {"default": CKPTS[0], "tooltip": "Depth Anything V2 model, from comfyui_controlnet_aux's ckpts folder. "
                                      "vitl (Large) is the one measured."}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64,
                                       "tooltip": "Size of the map's shorter side. The model itself reads the image at 518 px on its shorter "
                                                  "side whatever this value; 1024 is the measured setting."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_map", "info")
    FUNCTION = "make"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("A depth map for MOSS Control (near is white) with Depth Anything V2, made with the attention scaled once. "
                   "comfyui_controlnet_aux 1.1.6's own node scales it twice and gives a blur with no outline on this install; "
                   "this node uses that pack's model and weights with its own attention, leaves the pack unchanged, and downloads nothing.")

    def make(self, image, ckpt_name, resolution):
        da, att = aux_modules()
        path = weights_path(ckpt_name)
        if not os.path.isfile(path):
            raise RuntimeError(f"MOSS Depth Map: {ckpt_name} is not in {os.path.dirname(path)}. " + AUX_HINT)
        device = comfy.model_management.get_torch_device()
        start = time.perf_counter()
        det = detector(da, att, ckpt_name, path)
        det.to(device)
        maps = []
        try:
            with torch.no_grad():
                for img in image:
                    arr = np.clip(255.0 * img[:, :, :3].cpu().numpy(), 0, 255).astype(np.uint8)
                    out = det(arr, detect_resolution=int(resolution), max_depth=1)
                    maps.append(torch.from_numpy(np.asarray(out).astype(np.float32) / 255.0))
        finally:
            det.to("cpu")
            comfy.model_management.soft_empty_cache()
        seconds = time.perf_counter() - start
        out = torch.stack(maps)
        h, w = int(out.shape[1]), int(out.shape[2])
        text = (f"HaDeS MOSS Depth Map | {ckpt_name} | {len(maps)} image(s) | {w} x {h} | {seconds:.1f} s\n"
                "Attention scaled once (the pack's own node scales it twice on this install). Near is white.")
        info = format_info(text, "depth_map", ckpt=ckpt_name, resolution=int(resolution), images=len(maps), width=w, height=h,
                           attention="single scale")
        return (out, info)


NODE_CLASS_MAPPINGS = {"HaDeSMossDepthMap": HaDeSMossDepthMap}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossDepthMap": "MOSS Depth Map"}
