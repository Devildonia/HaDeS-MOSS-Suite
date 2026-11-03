import math

import torch
import comfy.model_management

from .moss_common import format_info

# width : height; the size follows from the megapixels, snapped to the 16 px latent grid
ASPECTS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9), "9:21": (9, 21)}
LATENT_CHANNELS = 128  # FLUX.2 latent, 16x downscale (latent_formats.Flux2)


def snap16(v):
    return max(16, int(round(v / 16.0)) * 16)


class HaDeSMossLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(ASPECTS) + ["custom"], {"default": "1:1"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                         "tooltip": "Image area, 1.0 = 1024x1024. Measured on square images: 768 (0.56) to 2048 (4.0) all render clean at "
                                                    "the default shift 5.0, larger ones show more scene around the subject (tools/RESULTS.md). "
                                                    "Other aspect ratios are not measured."}),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Used with aspect_ratio custom; snapped to 16 px."}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Used with aspect_ratio custom; snapped to 16 px."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Optional: checks that it is a 128-channel FLUX.2 VAE, the one MOSS-Image uses."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "info")
    FUNCTION = "generate"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = "Empty MOSS-Image latent (128 channels, 16 px grid) from an aspect ratio and megapixels; width / height out for layout prompts."

    def generate(self, aspect_ratio, megapixels, custom_width, custom_height, batch_size, vae=None):
        if vae is not None and getattr(vae, "latent_channels", LATENT_CHANNELS) != LATENT_CHANNELS:
            raise ValueError(f"HaDeS MOSS Latent: the VAE has {vae.latent_channels} latent channels; MOSS-Image needs the 128-channel "
                             f"FLUX.2 VAE (flux2-vae / flux2-vae_small).")
        if aspect_ratio == "custom":
            w, h = snap16(custom_width), snap16(custom_height)
        else:
            aw, ah = ASPECTS[aspect_ratio]
            unit = math.sqrt(megapixels * 1024 * 1024 / (aw * ah))
            w, h = snap16(aw * unit), snap16(ah * unit)
        latent = torch.zeros([batch_size, LATENT_CHANNELS, h // 16, w // 16], device=comfy.model_management.intermediate_device())
        mp = w * h / (1024 * 1024)
        text = f"HaDeS MOSS Latent | {w}x{h} ({mp:.2f} MP, {aspect_ratio}) | batch {batch_size} | {(w // 16) * (h // 16)} image tokens"
        if not 0.55 <= mp <= 4.05:
            text += "\nOutside the measured range (768x768 to 2048x2048)."
        return ({"samples": latent}, w, h, format_info(text, "latent", width=w, height=h, batch=batch_size))


NODE_CLASS_MAPPINGS = {"HaDeSMossLatent": HaDeSMossLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLatent": "MOSS Latent"}
