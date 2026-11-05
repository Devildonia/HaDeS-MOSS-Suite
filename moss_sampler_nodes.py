import copy

import comfy.samplers
import folder_paths
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


# the settings the turbo LoRAs were trained and measured at (tools/RESULTS.md, "Our own few-step LoRA")
TURBO_SETTINGS = {"steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}


class HaDeSMossTurbo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "A few-step MOSS-Image LoRA, e.g. HaDeS MOSS Turbo or Turbo Diverse (Hugging Face). Both were measured on SFT at "
                               "4 steps, CFG 1, euler / simple, strength 1.0: Turbo held text best (layout JSON 6 / 6, 16:9 poster 3 / 3), "
                               "Turbo Diverse kept SFT's spread of faces on an open prompt (tools/RESULTS.md)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "1.0 is the only strength measured."}),
                "steps": ("INT", {"default": TURBO_SETTINGS["steps"], "min": 1, "max": 50,
                                  "tooltip": "4 is the step count the LoRAs were trained on and the only one measured; other counts run, "
                                             "their quality is untested."}),
            },
        }

    RETURN_TYPES = ("MOSS_TURBO",)
    RETURN_NAMES = ("turbo",)
    FUNCTION = "pick"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = ("Few-step LoRA for the MOSS KSampler's turbo input: the KSampler applies it to its model and samples at the settings it "
                   "was trained for (CFG 1, euler / simple, and this node's steps, 4 by default), whatever its own widgets say.")

    def pick(self, lora_name, strength, steps=TURBO_SETTINGS["steps"]):
        return ({"lora_name": lora_name, "strength": float(strength), **TURBO_SETTINGS, "steps": int(steps)},)


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
                "turbo": ("MOSS_TURBO", {"tooltip": "Optional: a MOSS Turbo LoRA. When connected, the LoRA is applied to the model and the KSampler "
                                                    "samples at CFG 1, euler / simple with SLG off and the turbo node's steps, ignoring those widgets; "
                                                    "seed, denoise and flow_shift still apply. Measured at 4 steps, denoise 1.0 and the model's "
                                                    "shift 5.0 (tools/RESULTS.md)."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "info")
    FUNCTION = "sample"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = "KSampler with the official MOSS-Image settings as defaults, optional shift and SLG, optional decode, and the settings used in info."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, flow_shift, slg, vae=None,
               turbo=None, prompt=None, unique_id=None):
        # the image output connected without a VAE would reach the next node as None; say so here instead
        if vae is None and any(v == [str(unique_id), 1] for node in (prompt or {}).values() for v in node.get("inputs", {}).values()):
            raise ValueError("HaDeS MOSS KSampler: the image output is connected but no VAE is; connect the VAE input to decode.")
        if turbo is not None:  # the stock LoraLoaderModelOnly path, then the settings the LoRA was trained at
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, turbo["lora_name"], turbo["strength"])[0]
            steps, cfg, sampler_name, scheduler, slg = turbo["steps"], turbo["cfg"], turbo["sampler_name"], turbo["scheduler"], "off"
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
        extra = {}
        if turbo is not None:
            text += f" | turbo {turbo['lora_name']} x {turbo['strength']:g}"
            extra = {"turbo": turbo["lora_name"], "turbo_strength": turbo["strength"]}
        return (out, image, format_info(text, "ksampler", width=w, height=h, steps=int(steps), cfg=float(cfg), flow_shift=shift,
                                        sampler=sampler_name, scheduler=scheduler, seed=int(seed), denoise=float(denoise), slg=slg, **extra))


class HaDeSMossSamplerMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"info": ("STRING", {"forceInput": True, "tooltip": "info output of a MOSS KSampler."})}}

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "flow_shift", "turbo")
    OUTPUT_TOOLTIPS = ("", "", "", "", "", "", "", "The turbo LoRA file and strength when the KSampler had one (e.g. "
                                                                 "\"HaDeS-MOSS-Turbo-v0.6.safetensors:1\"), empty otherwise.")
    FUNCTION = "read"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = "The settings a MOSS KSampler actually sampled with, read from its info output, for image-saver metadata (turbo LoRA included)."

    def read(self, info):
        meta = parse_info_meta(info)
        if meta is None or meta.get("node") != "ksampler":
            raise ValueError("HaDeS MOSS Sampler Metadata needs the info output of a MOSS KSampler.")
        turbo = f"{meta['turbo']}:{meta.get('turbo_strength', 1.0):g}" if meta.get("turbo") else ""
        values = (meta["seed"], meta["steps"], meta["cfg"], meta["sampler"], meta["scheduler"], meta["denoise"], meta["flow_shift"], turbo)
        return {"ui": {"settings": [list(values)]}, "result": values}


NODE_CLASS_MAPPINGS = {"HaDeSMossKSampler": HaDeSMossKSampler, "HaDeSMossTurbo": HaDeSMossTurbo, "HaDeSMossSamplerMetadata": HaDeSMossSamplerMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossKSampler": "MOSS KSampler", "HaDeSMossTurbo": "MOSS Turbo LoRA", "HaDeSMossSamplerMetadata": "MOSS Sampler Metadata"}
