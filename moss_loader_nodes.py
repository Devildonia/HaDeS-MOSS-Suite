import os

import torch
import comfy.sd
import folder_paths
import nodes
from comfy.ldm.moss_image.model import MOSSImage
from comfy.quant_ops import QuantizedTensor
from comfy.text_encoders.moss_image import MOSSImageTEModel

from .moss_common import format_info, model_weights

LATENT_CHANNELS = 128  # FLUX.2 VAE, the one MOSS-Image uses


def model_precision(model):
    """The precision the weights have in memory, 'bf16' / 'fp16' / 'fp32' (ComfyUI may upcast on the CPU), or the quantized layout with how many weights use it (e.g. 'int8 (TensorWiseINT8Layout, 168 weights)')."""
    ws = model_weights(model)
    quant = [w for w in ws.values() if isinstance(w, QuantizedTensor)]
    if quant:
        layouts = sorted({lay if isinstance(lay := getattr(w, "layout_cls", None), str) else getattr(lay, "__name__", type(w).__name__) for w in quant})
        kind = "fp8" if any("fp8" in l.lower() or "float8" in l.lower() for l in layouts) else "int8"
        return f"{kind} ({', '.join(layouts)}, {len(quant)} weights)"
    dtypes = {w.dtype for w in ws.values()}
    names = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32", torch.float8_e4m3fn: "fp8_e4m3fn", torch.float8_e5m2: "fp8_e5m2"}
    return " + ".join(sorted(names.get(d, str(d).removeprefix("torch.")) for d in dtypes))


def first_match(names, *keys):
    """The first name containing one of `keys` (case-insensitive, in key order), as a new node's default; None when nothing matches."""
    for key in keys:
        for n in names:
            if key in n.lower():
                return n
    return None


class HaDeSMossLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unets, clips, vaes = folder_paths.get_filename_list("diffusion_models"), folder_paths.get_filename_list("text_encoders"), nodes.VAELoader.vae_list(nodes.VAELoader)
        default = lambda names, *keys: {"default": d} if (d := first_match(names, *keys)) else {}
        return {
            "required": {
                "unet_name": (unets, {**default(unets, "moss_image_sft", "mossimage_sft", "moss"), "tooltip": "MOSS-Image diffusion model (SFT, Base, a merge or an int8 ConvRot file), models/diffusion_models."}),
                "clip_name": (clips, {**default(clips, "qwen3.5_9b", "mossimage", "moss"), "tooltip": "MOSS-Image's text encoder (Qwen3.5-9B: qwen3.5_9b_int8_convrot or qwen3.5_9b_w4a8_gptq), "
                                                                                         "loaded as type moss_image, models/text_encoders."}),
                "vae_name": (vaes, {**default(vaes, "flux2-vae.", "flux2-vae", "flux2"), "tooltip": "The FLUX.2 VAE (flux2-vae or flux2-vae_small), 128 latent channels, models/vae."}),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"advanced": True, "tooltip": "As in the core Load Diffusion Model; default keeps the file's precision."}),
                "clip_device": (["default", "cpu"], {"advanced": True, "tooltip": "As in the core Load CLIP; cpu keeps the text encoder off the GPU."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "info")
    FUNCTION = "load"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Loads the three MOSS-Image files in one node: diffusion model, text encoder (always as type moss_image) and VAE, "
                   "the same as the core Load Diffusion Model, Load CLIP and Load VAE. Checks that each one is the MOSS-Image kind and "
                   "writes their names and precision to info for Suite Telemetry.")

    def load(self, unet_name, clip_name, vae_name, weight_dtype="default", clip_device="default"):
        model = nodes.UNETLoader().load_unet(unet_name, weight_dtype)[0]
        dm = model.model.diffusion_model
        if not isinstance(dm, MOSSImage):
            raise ValueError(f"HaDeS MOSS Loader: '{unet_name}' loads as {type(dm).__name__}, not a MOSS-Image model.")
        try:
            clip = nodes.CLIPLoader().load_clip(clip_name, "moss_image", clip_device)[0]
        except Exception as e:
            raise ValueError(f"HaDeS MOSS Loader: '{clip_name}' did not load as MOSS-Image's text encoder (Qwen3.5-9B): {e}") from e
        # the core loads any text encoder under type moss_image without complaint; only Qwen3.5-9B becomes MOSS-Image's
        if not isinstance(clip.cond_stage_model, MOSSImageTEModel):
            raise ValueError(f"HaDeS MOSS Loader: '{clip_name}' is not MOSS-Image's text encoder (it loaded as "
                             f"{type(clip.cond_stage_model).__name__}); use Qwen3.5-9B: qwen3.5_9b_int8_convrot or qwen3.5_9b_w4a8_gptq.")
        vae = nodes.VAELoader().load_vae(vae_name)[0]
        channels = getattr(vae, "latent_channels", None)
        if channels != LATENT_CHANNELS:
            raise ValueError(f"HaDeS MOSS Loader: '{vae_name}' has {channels} latent channels; MOSS-Image needs the 128-channel FLUX.2 VAE "
                             "(flux2-vae / flux2-vae_small).")

        precision = model_precision(model)
        attention = getattr(getattr(dm.layers[0], "comfy_attention", None), "method", None) or "the server's"
        lines = [f"HaDeS MOSS Loader | {len(dm.layers)} layers | {precision} | attention: {attention}",
                 f"  model: {unet_name}" + (f" (loaded as {weight_dtype})" if weight_dtype != "default" else ""),
                 f"  text encoder: {clip_name}" + (" (cpu)" if clip_device == "cpu" else ""),
                 f"  vae: {vae_name}"]
        # model_file / text_encoder_file: the names as listed under models/, which the MOSS Image Saver hashes
        meta = {"model": os.path.basename(unet_name), "model_file": unet_name, "text_encoder_file": clip_name, "precision": precision, "weight_dtype": weight_dtype, "attention": attention,
                "layers": len(dm.layers), "text_encoder": os.path.basename(clip_name), "vae": vae_name}
        return (model, clip, vae, format_info("\n".join(lines), "loader", **meta))


NODE_CLASS_MAPPINGS = {"HaDeSMossLoader": HaDeSMossLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLoader": "MOSS Loader"}
