"""What a few-step LoRA's late-layer change does to MOSS-Image activations (dev tool, not a node).

Loads --unet with a rank-2 LoRA cut to layers 24-41 (tools/RESULTS.md, "Few-step LoRAs") through the ComfyUI core in this process,
samples at the LoRA's settings and records, for every LoRA weight, the two numbers z = down @ x that the rank-2 change reads from each
image token at each step (the change adds up @ z). Reports per weight and step how constant z is over the tokens of one image
(share = mean^2 / (mean^2 + variance), 1 = the same z on every token) and how much the per-image mean of z varies between
images (spread = std of the per-image means / their mean magnitude). A change that reads a constant z acts as a fixed per-step
bias up @ z.

Run from the ComfyUI folder with the server's models freed (POST /free), for example:
  python custom_nodes/HaDeS-MOSS-Suite/tools/moss_activations.py --lora HaDeS_test_akh4_L24-41_r2.safetensors --steps 4 --out H:/moss_eval/activations
"""
import argparse
import json
import os
import sys

sys.argv, ARGV = [sys.argv[0], "--use-sage-attention"], sys.argv[1:]
sys.path.insert(0, os.getcwd())
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import torch  # noqa: E402

import comfy.sample  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

PROMPTS = {
    "neutral_woman": "photograph of a woman sitting in a café",
    "photo_portrait": "Studio photograph of an elderly fisherman with a weathered face and a grey knitted cap, soft window light from the left, shallow depth of field, 85mm lens, muted colors.",
    "illustration": "Watercolor illustration of a fox reading a book under a mushroom in a rainy forest, soft pastel palette, visible paper texture.",
}
WEIGHTS = ("qkv", "o", "ffn_up_gate", "ffn_down")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--clip", default="mossImage_base_txt_int4.safetensors")
    p.add_argument("--lora", required=True, help="models/loras file: layers 24-41 at rank 2, fused proj.qkv")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args(ARGV)
    os.makedirs(args.out, exist_ok=True)

    model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.unet))
    clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", args.clip)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    lora_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", args.lora), safe_load=True)
    model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, 1.0, 0.0)

    downs = {}
    for k, v in lora_sd.items():
        if k.endswith(".lora_down.weight"):
            name = k[len("diffusion_model."):-len(".lora_down.weight")]
            downs[name] = v.float()
    n_img = (args.size // 16) ** 2
    state = {"step": 0, "run": None}
    records = {}  # (weight, step) -> list of (mean z [2], var z [2]) per run

    dm = model.model.diffusion_model

    def hook(name):
        def fn(module, inputs):
            x = inputs[0]
            z = x[0, :n_img].float() @ downs[name].to(x.device).T  # image tokens x 2
            records.setdefault((name, state["step"]), []).append((z.mean(0).cpu(), z.var(0).cpu()))
        return fn

    handles = []
    for name in downs:
        module = comfy.utils.get_attr(dm, name)
        if name.endswith("ffn_down"):
            module.comfy_force_forward = True  # the fused act + linear path would skip forward hooks
        handles.append(module.register_forward_pre_hook(hook(name)))

    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    for pname, prompt in PROMPTS.items():
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        for seed in args.seeds:
            latent = torch.zeros([1, 128, args.size // 16, args.size // 16])
            noise = comfy.sample.prepare_noise(latent, seed)
            state["step"] = 0

            def callback(step, x0, x, total):
                state["step"] = step + 1

            comfy.sample.sample(model, noise, args.steps, 1.0, "euler", "simple", pos, neg, latent, seed=seed, callback=callback, disable_pbar=True)
            print(f"sampled {pname} {seed}", flush=True)
    for h in handles:
        h.remove()

    report = {}
    for (name, step), runs in sorted(records.items()):
        means = torch.stack([m for m, _ in runs])  # runs x 2
        vars_ = torch.stack([v for _, v in runs])
        share = (means.square() / (means.square() + vars_).clamp(min=1e-12)).mean(0)
        spread = means.std(0) / means.abs().mean(0).clamp(min=1e-12)
        layer, weight = int(name.split(".")[1]), name.split(".")[-1]
        report.setdefault(weight, {}).setdefault(str(step), []).append(
            {"layer": layer, "share": [round(v, 3) for v in share.tolist()], "spread": [round(v, 3) for v in spread.tolist()]})
    with open(os.path.join(args.out, f"activations_{os.path.splitext(args.lora)[0]}.json"), "w", encoding="utf-8") as f:
        json.dump({"lora": args.lora, "steps": args.steps, "runs": len(PROMPTS) * len(args.seeds), "report": report}, f, indent=1)
    for weight, steps in report.items():
        for step, rows in steps.items():
            share = torch.tensor([r["share"] for r in rows]).mean().item()
            spread = torch.tensor([r["spread"] for r in rows]).mean().item()
            print(f"{weight:12s} step {step}: share {share:.3f}  spread between images {spread:.3f}  ({len(rows)} layers)")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
