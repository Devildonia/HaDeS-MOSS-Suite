"""Can fixed per-step biases stand in for a few-step LoRA's late-layer change? (dev tool, not a node)

Calibration: --unet with a rank-2 LoRA cut to layers 24-41 (tools/RESULTS.md, "Few-step LoRAs") samples CALIB_PROMPTS x --calib-seeds
at the LoRA's settings; for every LoRA weight and step it records the mean of the change the LoRA adds to that weight's output,
up @ mean(down @ x), separately over image tokens and text tokens.

Test, on held-out seeds (--test-seeds) and TEST_PROMPTS, decoded and saved as PNG:
  lora       the rank-2 LoRA (reference)
  no_qkv     the LoRA without its qkv part (does that part matter?)
  qkv_bias   the LoRA without its qkv part, plus the calibrated qkv biases
  all_bias   no LoRA, the calibrated biases on every weight
  none       the model alone at the same settings
Reports distances to `lora` per config (tools/moss_eval.py image metrics) and writes contact sheets.

Run from the ComfyUI folder with the server's models freed (POST /free):
  python custom_nodes/HaDeS-MOSS-Suite/tools/moss_bias_turbo.py --lora HaDeS_test_akh4_L24-41_r2.safetensors --steps 4 --out H:/moss_eval/bias_turbo
"""
import argparse
import importlib.util
import json
import os
import sys

sys.argv, ARGV = [sys.argv[0], "--use-sage-attention"], sys.argv[1:]
sys.path.insert(0, os.getcwd())
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.sample  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

_spec = importlib.util.spec_from_file_location("moss_eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "moss_eval.py"))
moss_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moss_eval)

CALIB_PROMPTS = ("neutral_woman", "photo_portrait", "illustration")
TEST_PROMPTS = ("neutral_woman", "photo_portrait", "illustration", "text_sign")
PROMPTS = {**moss_eval.GROUP_PROMPTS, **moss_eval.COMPARE}
CONFIGS = ("lora", "no_qkv", "qkv_bias", "all_bias", "none")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--clip", default="mossImage_base_txt_int4.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--lora", required=True, help="models/loras file: layers 24-41 at rank 2, fused proj.qkv")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--calib-seeds", type=int, nargs="+", default=[1000, 1001])
    p.add_argument("--test-seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args(ARGV)
    os.makedirs(args.out, exist_ok=True)

    base = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.unet))
    clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", args.clip)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", args.vae)))
    lora_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", args.lora), safe_load=True)
    factors = {}
    for k in lora_sd:
        if k.endswith(".lora_down.weight"):
            name = k[len("diffusion_model."):-len(".lora_down.weight")]
            pre = k[:-len(".lora_down.weight")]
            scale = lora_sd[pre + ".alpha"].float().item() / lora_sd[k].shape[0]
            factors[name] = (lora_sd[pre + ".lora_up.weight"].float() * scale, lora_sd[k].float())
    dm = base.model.diffusion_model
    for name in factors:
        if name.endswith("ffn_down"):
            comfy.utils.get_attr(dm, name).comfy_force_forward = True  # the fused act + linear path would skip forward hooks
    n_img = (args.size // 16) ** 2
    state = {"step": 0}

    def with_lora(drop_qkv):
        sd = {k: v for k, v in lora_sd.items() if not (drop_qkv and ".proj.qkv." in k)}
        return comfy.sd.load_lora_for_models(base, None, sd, 1.0, 0.0)[0] if sd else base

    def sample(model, prompt, seed, hooks):
        handles = [comfy.utils.get_attr(dm, name).register_forward_pre_hook(fn) if kind == "pre" else
                   comfy.utils.get_attr(dm, name).register_forward_hook(fn) for name, kind, fn in hooks]
        try:
            pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            latent = torch.zeros([1, 128, args.size // 16, args.size // 16])
            state["step"] = 0

            def callback(step, x0, x, total):
                state["step"] = step + 1

            return comfy.sample.sample(model, comfy.sample.prepare_noise(latent, seed), args.steps, 1.0, "euler", "simple", pos, neg, latent,
                                       seed=seed, callback=callback, disable_pbar=True)
        finally:
            for h in handles:
                h.remove()

    # calibration: mean z per weight, step and token kind, over all calibration images
    sums = {}

    def recorder(name):
        def fn(module, inputs):
            x = inputs[0][0].float()
            z = x @ factors[name][1].to(x.device).T
            for kind, part in (("img", z[:n_img]), ("txt", z[n_img:])):
                s = sums.setdefault((name, state["step"], kind), [0, 0])
                s[0] = s[0] + part.sum(0).cpu()
                s[1] += part.shape[0]
        return fn

    lora_model = with_lora(False)
    for pname in CALIB_PROMPTS:
        for seed in args.calib_seeds:
            sample(lora_model, PROMPTS[pname], seed, [(n, "pre", recorder(n)) for n in factors])
            print(f"calibrated {pname} {seed}", flush=True)
    biases = {key: factors[key[0]][0] @ (s / n) for key, (s, n) in sums.items()}  # out-dim vector

    def adder(name):
        def fn(module, inputs, output):
            b_img = biases.get((name, state["step"], "img"))
            if b_img is None:
                return output
            out = output.clone()
            out[:, :n_img] += b_img.to(out.device, out.dtype)
            out[:, n_img:] += biases[(name, state["step"], "txt")].to(out.device, out.dtype)
            return out
        return fn

    qkv = [n for n in factors if n.endswith(".qkv")]
    plans = {
        "lora": (lora_model, []),
        "no_qkv": (with_lora(True), []),
        "qkv_bias": (with_lora(True), [(n, "post", adder(n)) for n in qkv]),
        "all_bias": (base, [(n, "post", adder(n)) for n in factors]),
        "none": (base, []),
    }
    feats = {}
    for c, (model, hooks) in plans.items():
        feats[c] = {}
        for pname in TEST_PROMPTS:
            os.makedirs(os.path.join(args.out, c, pname), exist_ok=True)
            for seed in args.test_seeds:
                path = os.path.join(args.out, c, pname, f"{seed}.png")
                if not os.path.exists(path):
                    img = vae.decode(sample(model, PROMPTS[pname], seed, hooks))
                    img = img.reshape(-1, *img.shape[-3:])[0]
                    Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)
                feats[c][f"{pname}_{seed}"] = moss_eval.features(Image.open(path).convert("RGB"))
                print(f"{c} {pname} {seed}", flush=True)
    rows = {c: {m: round(float(np.mean([moss_eval.distance(feats[c][s], feats["lora"][s])[m] for s in feats["lora"]])), 3) for m in moss_eval.METRICS}
            for c in CONFIGS if c != "lora"}
    report = {"lora": args.lora, "steps": args.steps, "calib": {"prompts": CALIB_PROMPTS, "seeds": args.calib_seeds},
              "test": {"prompts": TEST_PROMPTS, "seeds": args.test_seeds}, "distance_to_lora": rows}
    with open(os.path.join(args.out, "report_bias_turbo.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for pname in TEST_PROMPTS:
        print(moss_eval.contact_sheet(args.out, pname, PROMPTS[pname], args.test_seeds, list(CONFIGS), cols=len(args.test_seeds), tile=256))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
