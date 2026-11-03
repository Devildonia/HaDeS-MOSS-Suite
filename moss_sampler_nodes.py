import copy

import comfy.samplers
import nodes
from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

from .moss_common import format_info, parse_info_meta

# single block, start / end of the negative-pass skip; "Base template" is SkipLayerGuidanceDiTSimple as the official
# image_moss_image_base_t2i workflow sets it
SLG_MODES = {
    "off": None,
    "block 22, first half (Base template)": ("22", 0.0, 0.5),
    "block 22, all steps": ("22", 0.0, 1.0),
}


def with_shift(model, shift):
    """The model with its own sampling object at another shift (multiplier and type kept); unchanged when shift is 0 or equal."""
    ms = model.get_model_object("model_sampling")
    if shift <= 0.0 or abs(float(ms.shift) - shift) < 1e-4:
        return model
    m = model.clone()
    new_ms = copy.deepcopy(ms)
    new_ms.set_parameters(shift=shift, multiplier=float(ms.multiplier))
    m.add_object_patch("model_sampling", new_ms)
    return m


class HaDeSMossKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                                  "tooltip": "25 as in the official templates. Measured with euler, 25 steps render nearly the 60-step image with any scheduler, "
                                             "and euler / beta does so at 15 (tools/RESULTS.md)."}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01, "tooltip": "2.5 as in the official templates."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde",
                                                                    "tooltip": "er_sde as in the official templates. It is stochastic: the same seed gives another image than euler, "
                                                                               "equally clean in tests; which looks better is not measured."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1,
                                         "tooltip": "0 keeps the model's own shift (5.0). Measured from 768 to 2048 px, 5.0 stays clean at every size and raising it "
                                                    "changes the image less than a new seed (tools/RESULTS.md)."}),
                "slg": (list(SLG_MODES), {"default": "off",
                                          "tooltip": "Skip-layer guidance on the negative pass. Measured on SFT and Base, block 22 did not change text accuracy "
                                                     "(both models spelled every test string right without it); on photos it changed details, "
                                                     "and on Base one face entirely (tools/RESULTS.md)."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Optional: also decode the result. Needed when the image output is connected."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "info")
    FUNCTION = "sample"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = "KSampler with the official MOSS-Image settings as defaults, optional shift and SLG, optional decode, and the settings used in info."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, flow_shift, slg, vae=None,
               prompt=None, unique_id=None):
        # the image output connected without a VAE would reach the next node as None; say so here instead
        if vae is None and any(v == [str(unique_id), 1] for node in (prompt or {}).values() for v in node.get("inputs", {}).values()):
            raise ValueError("HaDeS MOSS KSampler: the image output is connected but no VAE is; connect the VAE input to decode.")
        m = with_shift(model, flow_shift)
        if SLG_MODES[slg] is not None:
            layers, start, end = SLG_MODES[slg]
            m = SkipLayerGuidanceDiTSimple.execute(m, start, end, double_layers="", single_layers=layers).args[0]
        out = nodes.common_ksampler(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
        image = vae.decode(out["samples"]) if vae is not None else None
        if image is not None and image.ndim == 5:  # video-shaped VAE output
            image = image.reshape(-1, *image.shape[-3:])
        shift = float(m.get_model_object("model_sampling").shift)
        h, w = out["samples"].shape[-2] * 16, out["samples"].shape[-1] * 16
        text = (f"HaDeS MOSS KSampler | {w}x{h} | {sampler_name} / {scheduler} | {steps} steps | CFG {cfg:.2f} | shift {shift:g} | "
                f"denoise {denoise:.2f} | SLG {slg} | seed {seed}")
        return (out, image, format_info(text, "ksampler", width=w, height=h, steps=int(steps), cfg=float(cfg), flow_shift=shift,
                                        sampler=sampler_name, scheduler=scheduler, seed=int(seed), denoise=float(denoise), slg=slg))


class HaDeSMossSamplerMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"info": ("STRING", {"forceInput": True, "tooltip": "info output of a MOSS KSampler."})}}

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "flow_shift")
    FUNCTION = "read"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = "The settings a MOSS KSampler actually sampled with, read from its info output, for image-saver metadata."

    def read(self, info):
        meta = parse_info_meta(info)
        if meta is None or meta.get("node") != "ksampler":
            raise ValueError("HaDeS MOSS Sampler Metadata needs the info output of a MOSS KSampler.")
        values = (meta["seed"], meta["steps"], meta["cfg"], meta["sampler"], meta["scheduler"], meta["denoise"], meta["flow_shift"])
        return {"ui": {"settings": [list(values)]}, "result": values}


NODE_CLASS_MAPPINGS = {"HaDeSMossKSampler": HaDeSMossKSampler, "HaDeSMossSamplerMetadata": HaDeSMossSamplerMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossKSampler": "MOSS KSampler", "HaDeSMossSamplerMetadata": "MOSS Sampler Metadata"}
