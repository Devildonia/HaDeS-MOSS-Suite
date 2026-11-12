import copy

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
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


DENOISE_CORE, DENOISE_EXACT = "steps / denoise (core)", "exact"


def core_schedule_steps(steps, denoise):
    """The schedule ComfyUI builds for denoise < 1 (comfy/samplers.py, KSampler.set_steps): int(steps / denoise) steps, of which the
    last `steps` run. At few steps many denoise values give the same schedule, and one that rounds to `steps` starts from pure noise."""
    return steps if denoise > 0.9999 else int(steps / denoise)


def full_sigmas(model_sampling, sampler_name, scheduler, steps):
    """KSampler.calculate_sigmas without a model patcher (some samplers drop the penultimate sigma)."""
    discard = sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps + 1 if discard else steps).cpu()
    return torch.cat([sigmas[:-2], sigmas[-1:]]) if discard else sigmas


def percent_of_sigma(model_sampling, sigma):
    """The inverse of model_sampling.percent_to_sigma (decreasing in percent), by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if float(model_sampling.percent_to_sigma(mid)) > sigma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_sigmas(model_sampling, sampler_name, scheduler, steps, denoise):
    """`steps` steps from the point `denoise` of the model's own noise-time axis (percent_to_sigma(1 - denoise)) down to 0, with the
    scheduler's spacing: its full schedule with every sigma moved from time t to denoise * t. Equals the full schedule at denoise 1 and
    is what the core's int(steps / denoise) tends to as steps grow, but continuous at few steps."""
    sigmas = full_sigmas(model_sampling, sampler_name, scheduler, steps)
    out = [float(model_sampling.percent_to_sigma(1.0 - denoise * (1.0 - percent_of_sigma(model_sampling, float(s))))) if s > 0 else 0.0
           for s in sigmas]
    return torch.tensor(out, dtype=sigmas.dtype)


def start_sigma(model_sampling, sampler_name, scheduler, steps, denoise, mode):
    """The first sigma the sampler runs from, in either mode."""
    if mode == DENOISE_EXACT and denoise < 0.9999:
        return float(model_sampling.percent_to_sigma(1.0 - denoise))
    return float(full_sigmas(model_sampling, sampler_name, scheduler, core_schedule_steps(steps, denoise))[-(steps + 1)])


def ksample_sigmas(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, sigmas):
    """nodes.common_ksampler with an explicit sigma schedule."""
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial", None),
                                                          latent.get("downscale_ratio_temporal", None))
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, noise_mask=latent.get("noise_mask"),
                                  sigmas=sigmas, callback=latent_preview.prepare_callback(model, steps),
                                  disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


# the settings the turbo LoRAs were trained and measured at (tools/RESULTS.md, "Our own few-step LoRA"); the node's defaults
TURBO_SETTINGS = {"steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}


class HaDeSMossTurbo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "A few-step MOSS-Image LoRA, e.g. HaDeS MOSS Turbo, Turbo Diverse or Turbo 8-steps (Hugging Face), measured on SFT "
                               "at CFG 1, euler / simple, strength 1.0. At 4 steps Turbo held text best (layout JSON 6 / 6, 16:9 poster 3 / 3) "
                               "and Turbo Diverse kept SFT's spread of faces on an open prompt; Turbo 8-steps at 8 steps beat akhaliq's 4-step "
                               "LoRA 18 to 5 in blind pairs and matched lodestones' Base / Kromoss 8-step LoRA run on SFT up to 1024 (tools/RESULTS.md)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "1.0 is the only strength measured."}),
                "steps": ("INT", {"default": TURBO_SETTINGS["steps"], "min": 1, "max": 50,
                                  "tooltip": "4 for Turbo and Turbo Diverse, 8 for Turbo 8-steps: the counts measured. At 1024 Turbo and "
                                             "Turbo Diverse give 6-10% less colour than SFT at 4 steps (6-14% at 2048) and are back near it at 8 (only colour measured "
                                             "at 8 for them). Other counts run, their quality is untested (tools/RESULTS.md)."}),
            },
            "optional": {
                "cfg": ("FLOAT", {"default": TURBO_SETTINGS["cfg"], "min": 1.0, "max": 4.0, "step": 0.05, "round": 0.01,
                                  "tooltip": "1.0 is what the turbo LoRAs were trained at. Above 1 the negative prompt takes part and each step runs "
                                             "the model twice (about 1.9x the time). Measured with one seed per prompt: Turbo 8-steps at 1.5 rendered "
                                             "10 hard prompts on SFT and Base without burning; on a merge with a turbo baked in, 1.25-1.5 brought back "
                                             "prompt details lost at 1.0, saturation rose with every step and 2.5 looked overdone; texture-heavy "
                                             "prompts looked better at 1.0-1.2 (tools/RESULTS.md, \"Turbo LoRAs above CFG 1\")."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": TURBO_SETTINGS["sampler_name"],
                                                                    "tooltip": "euler: what the turbo LoRAs were trained and measured with; er_sde was used in the CFG runs above."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": TURBO_SETTINGS["scheduler"],
                                                                   "tooltip": "simple: what the turbo LoRAs were trained and measured with."}),
            },
        }

    RETURN_TYPES = ("MOSS_TURBO", "STRING")
    RETURN_NAMES = ("turbo", "info")
    FUNCTION = "pick"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = ("Few-step LoRA for the MOSS KSampler's turbo input: the KSampler applies it to its model and samples at this node's "
                   "steps, CFG, sampler and scheduler (by default the settings the LoRAs were trained at: 4 steps, CFG 1, euler / simple), "
                   "whatever its own widgets say.")

    def pick(self, lora_name, strength, steps=TURBO_SETTINGS["steps"], cfg=TURBO_SETTINGS["cfg"], sampler_name=TURBO_SETTINGS["sampler_name"],
             scheduler=TURBO_SETTINGS["scheduler"]):
        info = format_info(f"HaDeS MOSS Turbo LoRA | {lora_name} | strength {float(strength):g} | {int(steps)} steps, CFG {float(cfg):g}, "
                           f"{sampler_name} / {scheduler}", "turbo", lora=lora_name, strength=float(strength), steps=int(steps), cfg=float(cfg),
                           sampler=sampler_name, scheduler=scheduler)
        return ({"lora_name": lora_name, "strength": float(strength), "steps": int(steps), "cfg": float(cfg), "sampler_name": sampler_name,
                 "scheduler": scheduler}, info)


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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Below 1, how much of the input latent is redrawn. With denoise_mode steps / denoise (core) ComfyUI "
                                                 "builds int(steps / denoise) steps and runs the last ones, so at few steps it moves in jumps: at 4 "
                                                 "steps 0.7 and 0.8 give the same image and 0.9 starts from pure noise (the input is ignored). info "
                                                 "and Suite Telemetry show the start sigma actually used."}),
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
                "denoise_mode": ([DENOISE_CORE, DENOISE_EXACT], {"default": DENOISE_CORE, "tooltip":
                    "steps / denoise (core): the stock KSampler's schedule. exact: denoise is a point on the model's noise-time axis and every "
                    "step runs from there (start sigma = percent_to_sigma(1 - denoise), the scheduler's spacing kept), so it is continuous at few "
                    "steps; at denoise 1 both are the same. Measured on img2img with Turbo v1.0 at 4 and 8 steps (four inputs, tools/RESULTS.md): "
                    "up to 0.5 both modes give the same image; above it exact rose smoothly with every 0.1, where core repeated 0.7 at 0.8 "
                    "(4 steps) and ignored the input at 0.9 (4 and 8 steps); exact 0.9 (sigma 0.978) kept a poster's layout and text, "
                    "photographs became new scenes."}),
                "turbo": ("MOSS_TURBO", {"tooltip": "Optional: a MOSS Turbo LoRA. When connected, the LoRA is applied to the model and the KSampler "
                                                    "samples with SLG off and the turbo node's steps, CFG, sampler and scheduler (4, 1, euler / simple "
                                                    "by default), ignoring those widgets; "
                                                    "seed, denoise and flow_shift still apply. Measured at 4 or 8 steps, denoise 1.0 (Turbo 8-steps also at "
                                                    "0.25 / 0.40 on a 2x-upscaled image) and the model's shift 5.0 (tools/RESULTS.md)."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "info")
    FUNCTION = "sample"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = "KSampler with the official MOSS-Image settings as defaults, optional shift and SLG, optional decode, and the settings used in info."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, flow_shift, slg, vae=None,
               turbo=None, prompt=None, unique_id=None, denoise_mode=DENOISE_CORE):
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
        ms = m.get_model_object("model_sampling")
        exact = denoise_mode == DENOISE_EXACT and 0.0 < denoise < 0.9999
        if exact:
            out = ksample_sigmas(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                 exact_sigmas(ms, sampler_name, scheduler, steps, denoise))
        else:
            out = nodes.common_ksampler(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
        sigma0 = start_sigma(ms, sampler_name, scheduler, steps, denoise, denoise_mode) if denoise > 0.0 else 0.0
        image = vae.decode(out["samples"]) if vae is not None else None
        if image is not None and image.ndim == 5:  # video-shaped VAE output
            image = image.reshape(-1, *image.shape[-3:])
        shift = float(ms.shift)
        h, w = out["samples"].shape[-2] * 16, out["samples"].shape[-1] * 16
        text = (f"HaDeS MOSS KSampler | {w}x{h} | {sampler_name} / {scheduler} | {steps} steps | CFG {cfg:.2f} | shift {shift:g} | "
                f"denoise {denoise:.2f}{' exact' if exact else ''} (start sigma {sigma0:.3f}) | SLG {slg} | seed {seed}")
        extra = {"denoise_mode": "exact" if exact else "core", "start_sigma": round(sigma0, 4)}
        if not exact and denoise < 0.9999:
            extra["schedule_steps"] = core_schedule_steps(steps, denoise)
        if turbo is not None:
            text += f" | turbo {turbo['lora_name']} x {turbo['strength']:g}"
            extra.update(turbo=turbo["lora_name"], turbo_strength=turbo["strength"])
        return (out, image, format_info(text, "ksampler", width=w, height=h, steps=int(steps), cfg=float(cfg), flow_shift=shift,
                                        sampler=sampler_name, scheduler=scheduler, seed=int(seed), denoise=float(denoise), slg=slg, **extra))


class HaDeSMossSamplerMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"info": ("STRING", {"forceInput": True, "tooltip": "info output of a MOSS KSampler."})}}

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "flow_shift", "turbo", "info")
    OUTPUT_TOOLTIPS = ("", "", "", "", "", "", "", "The turbo LoRA file and strength when the KSampler had one (e.g. "
                                                                 "\"HaDeS-MOSS-Turbo-v0.6.safetensors:1\"), empty otherwise.",
                       "The values read, for Suite Telemetry.")
    FUNCTION = "read"
    CATEGORY = "HaDeS/MOSS Samplers"
    DESCRIPTION = "The settings a MOSS KSampler actually sampled with, read from its info output, for image-saver metadata (turbo LoRA included)."

    def read(self, info):
        meta = parse_info_meta(info)
        if meta is None or meta.get("node") != "ksampler":
            raise ValueError("HaDeS MOSS Sampler Metadata needs the info output of a MOSS KSampler.")
        turbo = f"{meta['turbo']}:{meta.get('turbo_strength', 1.0):g}" if meta.get("turbo") else ""
        values = (meta["seed"], meta["steps"], meta["cfg"], meta["sampler"], meta["scheduler"], meta["denoise"], meta["flow_shift"], turbo)
        info = format_info(f"HaDeS MOSS Sampler Metadata | seed {values[0]} | {values[1]} steps, CFG {values[2]:g}, {values[3]} / {values[4]}"
                           + (f" | turbo {turbo}" if turbo else ""), "sampler_metadata",
                           **dict(zip(("seed", "steps", "cfg", "sampler", "scheduler", "denoise", "flow_shift", "turbo"), values)))
        return {"ui": {"settings": [list(values)]}, "result": values + (info,)}


NODE_CLASS_MAPPINGS = {"HaDeSMossKSampler": HaDeSMossKSampler, "HaDeSMossTurbo": HaDeSMossTurbo, "HaDeSMossSamplerMetadata": HaDeSMossSamplerMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossKSampler": "MOSS KSampler", "HaDeSMossTurbo": "MOSS Turbo LoRA", "HaDeSMossSamplerMetadata": "MOSS Sampler Metadata"}
