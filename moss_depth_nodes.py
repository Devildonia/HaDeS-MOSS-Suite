import contextlib
import importlib
import os
import sys
import time

import numpy as np
import torch

import comfy.model_management
import folder_paths

from .moss_common import format_info

CKPTS = ["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"]
AUX_HINT = ("MOSS Depth Map needs the comfyui_controlnet_aux node pack (ComfyUI Manager: \"ComfyUI's ControlNet Auxiliary Preprocessors\") "
            "and its Depth Anything V2 weights; run its Depth Anything V2 node once to download them.")
_MODELS = {}  # ckpt -> detector, kept on the CPU between runs


def aux_modules():
    """comfyui_controlnet_aux's Depth Anything V2 detector and its DINOv2 attention module, or a clear error. The pack adds its src
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


def weights_path(da, ckpt):
    """Where the pack keeps the weights (its annotator ckpts folder / the Hugging Face repo name / the file). Never downloaded here."""
    util = importlib.import_module("custom_controlnet_aux.util")
    root = os.environ.get("AUX_ANNOTATOR_CKPTS_PATH") or getattr(util, "annotator_ckpts_path", "")
    return os.path.join(root, util.DEPTH_ANYTHING_V2_MODEL_NAME_DICT[ckpt], ckpt)


@contextlib.contextmanager
def single_scaled_attention(att):
    """The pack's MemEffAttention multiplies q by the scale and then calls xFormers / scaled_dot_product_attention, which scale
    again: the encoder sees a blur and the map loses the outline (tools/RESULTS.md). For this call only, the plain DINOv2
    Attention.forward (one scale) is used; the pack is restored afterwards."""
    original = att.MemEffAttention.forward
    att.MemEffAttention.forward = lambda self, x, attn_bias=None: att.Attention.forward(self, x)
    try:
        yield
    finally:
        att.MemEffAttention.forward = original


class HaDeSMossDepthMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photo or render whose pose you want to keep."}),
                "ckpt_name": (CKPTS, {"default": CKPTS[0], "tooltip": "Depth Anything V2 model, from comfyui_controlnet_aux's ckpts folder. "
                                      "vitl (Large) is the one measured."}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64,
                                       "tooltip": "Size of the map's shorter side. 1024 is the measured setting."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_map", "info")
    FUNCTION = "make"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("A depth map for MOSS Control (near is white) with Depth Anything V2, made with the attention scaled once. "
                   "comfyui_controlnet_aux 1.1.6's own node scales it twice and gives a blur with no outline on this install; "
                   "this node uses that pack's model and weights, fixes the attention for its own call only, and downloads nothing.")

    def make(self, image, ckpt_name, resolution):
        da, att = aux_modules()
        path = weights_path(da, ckpt_name)
        if not os.path.isfile(path):
            raise RuntimeError(f"MOSS Depth Map: {ckpt_name} is not in {os.path.dirname(path)}. " + AUX_HINT)
        device = comfy.model_management.get_torch_device()
        start = time.perf_counter()
        det = _MODELS.get(ckpt_name)
        if det is None:
            # the file exists (checked above), so the pack's loader reads it locally and never downloads
            det = da.DepthAnythingV2Detector.from_pretrained(filename=ckpt_name)
            _MODELS[ckpt_name] = det
        det.to(device)
        maps = []
        try:
            with torch.no_grad(), single_scaled_attention(att):
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
