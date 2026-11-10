import numpy as np
import torch
from PIL import Image, ImageFilter

import comfy.utils
import node_helpers

from .moss_common import format_info

LATENT_CHANNELS = 128
COLOR_MODES = ["off (depth only)", "flat regions", "as given"]


def fit_maps(image, width, height):
    """An IMAGE batch resized in pixels to width x height (center crop to the aspect first), as the model would crop the latent."""
    samples = image[:, :, :, :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, "lanczos", "center").movedim(1, -1).clamp(0.0, 1.0)


def flat_regions(image, colors, blur):
    """A photo turned into flat color regions (the adapter's color input is one flat color per region): blurred, then reduced to
    `colors` colors by median cut, so it guides the palette and the large areas without the photo's texture."""
    out = []
    for img in image:
        pil = Image.fromarray(np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8))
        if blur > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(blur))
        pil = pil.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
        out.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
    return torch.stack(out).to(image.device)


class HaDeSMossControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT", {"tooltip": "The latent to generate (MOSS Latent's): the maps are resized to its size in pixels before the VAE."}),
                "color_mode": (COLOR_MODES, {"default": "off (depth only)",
                                             "tooltip": "off: depth only, the clothes, colors and textures stay with the prompt (the adapter was trained "
                                                        "like that 35% of the time, its file metadata). flat regions: the color_map image (a photo too) "
                                                        "is blurred and reduced to a few flat colors, which guide the palette and large areas. as given: "
                                                        "the color_map as it comes, for a real segmentation map (one flat color per region); a photo "
                                                        "here makes the adapter copy its texture."}),
                "colors": ("INT", {"default": 8, "min": 2, "max": 64, "tooltip": "flat regions: number of colors."}),
                "blur": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 32.0, "step": 0.5, "tooltip": "flat regions: blur radius in pixels before the colors are reduced."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                       "tooltip": "Scales the maps' contribution. 1.0 is the trained strength; 0 is no control. This is the strength "
                                                  "that works: the adapter file's own strength in a LoRA loader changes only its LoRA layers, never "
                                                  "the widened video_embedder, which is replaced at full strength (ComfyUI's 'set' patch)."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                          "tooltip": "Ending early keeps the layout and depth while the last steps loosen the shapes."}),
            },
            "optional": {
                "depth_map": ("IMAGE", {"tooltip": "Depth map, near is white (e.g. Depth Anything)."}),
                "color_map": ("IMAGE", {"tooltip": "Color regions, or a photo with color_mode flat regions."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive", "negative", "maps", "info")
    FUNCTION = "apply"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("Color / depth control for MOSS-Image with a control adapter loaded as a LoRA (e.g. moss-color-depth-control, which widens "
                   "video_embedder to 384 channels). Works on the Prompt Enhancer's conditioning; resizes the maps in pixels to the latent "
                   "before the VAE; turns a photo into flat color regions; info for Telemetry and the Image Saver.")

    def apply(self, positive, negative, vae, latent, color_mode, colors, blur, strength, start_percent, end_percent, depth_map=None, color_map=None):
        if getattr(vae, "latent_channels", LATENT_CHANNELS) != LATENT_CHANNELS:
            raise ValueError(f"HaDeS MOSS Control: the VAE has {vae.latent_channels} latent channels; MOSS-Image needs the 128-channel FLUX.2 VAE.")
        h, w = latent["samples"].shape[-2] * 16, latent["samples"].shape[-1] * 16
        use_color = color_map is not None and not color_mode.startswith("off")
        if depth_map is None and not use_color:
            raise ValueError("HaDeS MOSS Control: connect a depth_map, or a color_map with color_mode flat regions / as given.")
        depth = fit_maps(depth_map, w, h) if depth_map is not None else None
        color = fit_maps(color_map, w, h) if use_color else None
        if color is not None and color_mode == "flat regions":
            color = flat_regions(color, colors, blur)
        encoded = [vae.encode(m) if m is not None else None for m in (color, depth)]
        ref = next(e for e in encoded if e is not None)
        batch = max(e.shape[0] for e in encoded if e is not None)
        # [color | depth]; a missing map is a zero latent, which is no control (the embedder is linear)
        control = torch.cat([comfy.utils.resize_to_batch_size(torch.zeros_like(ref) if e is None else e, batch) for e in encoded], dim=1) * strength
        values = {"concat_latent_image": control}
        pos = node_helpers.conditioning_set_values_with_timestep_range(positive, values, start_percent, end_percent)
        neg = node_helpers.conditioning_set_values_with_timestep_range(negative, values, start_percent, end_percent)

        preview = [m for m in (color, depth) if m is not None]
        maps = torch.cat([comfy.utils.resize_to_batch_size(m, batch) for m in preview], dim=0)
        used = "+".join(n for n, m in (("color", color), ("depth", depth)) if m is not None)
        mode = color_mode if use_color else "off (depth only)"
        text = (f"HaDeS MOSS Control | {used} | color {mode}" + (f" ({colors} colors, blur {blur:g})" if use_color and color_mode == "flat regions" else "")
                + f" | strength {strength:g} | {start_percent:g}-{end_percent:g} | maps at {w}x{h}")
        info = format_info(text, "control", maps=used, color_mode=mode, strength=float(strength), start_percent=float(start_percent),
                           end_percent=float(end_percent), width=w, height=h)
        return (pos, neg, maps, info)


NODE_CLASS_MAPPINGS = {"HaDeSMossControl": HaDeSMossControl}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossControl": "MOSS Control (color / depth)"}
