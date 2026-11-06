"""Train an image-free "slider" LoRA for MOSS-Image: a direction between two prompt suffixes, learned on any prompt (dev tool, not a node).

The idea of Concept Sliders (Gandikota et al., 2023) on MOSS-Image's flow matching, with no images:
  - a noisy latent x_t comes from the model itself: a clean image rendered by a turbo LoRA in a few steps (default HaDeS MOSS
    Turbo Diverse v0.6, 4 steps, CFG 1), noised to a random level t;
  - target = frozen model's clean-image prediction for the prompt + eta x (prediction with --positive appended - prediction with
    --negative appended), all three at the same x_t;
  - the slider LoRA, on the prompt alone, is trained toward that target (v-space mean squared error).
So the LoRA learns to apply the --positive / --negative difference without the words in the prompt; its strength in ComfyUI
scales it.

With --pairs N it learns from image pairs instead (the paper's visual sliders), with no outside images: the model renders N
images (25 steps, CFG 2.5, euler / simple, at --size), each gets a Gaussian-blurred copy (radius from --blur-mild, or
--blur-strong for a --strong-frac share), and both go through the VAE. Blurring keeps the mean colour, brightness and the
composition, so a pair differs only in fine detail. Each update noises both latents with the same noise and level; the model
with the LoRA at +1 is fitted to the sharp latent and at -1 to the blurred one (v-space). What the two terms share (the model
not reproducing its own image exactly) cancels between them, so the LoRA keeps the sharp - blurred difference. With
--pair-kind colour the poorer copy is desaturated instead (Lab chroma scaled, optionally cooled; lightness and neutral pixels
unchanged), so the LoRA learns richer colour that leaves whites and greys alone. The model it trains on can be SFT, or SFT with layers taken from Base (--merge-base, the HaDeS MOSS Model Block Merger
recipe computed once in memory), which is the model it is meant to load on.

Prompts, LoRA format, checkpoints (ckpt.pt, atomic, resume with the same --out), lora_ema.safetensors and previews are as in
moss_dmd_train.py, whose prompt generator and LoRA class it reuses. Previews: the preview prompts at 25 steps, CFG 2.5, euler /
simple, LoRA off and on (top / bottom row).

Run from the ComfyUI folder with the server's models freed (POST /free):
  python custom_nodes/HaDeS-MOSS-Suite/tools/moss_slider_train.py --merge-base mossImage_base_bf16.safetensors --out I:/slider_colour
"""
import argparse
import json
import os
import random
import sys
import time

import moss_dmd_train as dmd  # noqa: E402  (sets up ComfyUI's argument parsing and sys.path, as when run on its own)
from moss_dmd_train import ARGV, LoRASet, layer_names, make_prompts  # noqa: E402

import numpy as np  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

PREVIEW_PROMPTS = [
    "A busy street market in Marrakech with spice stalls, colorful lanterns and fabrics, photograph.",
    "Portrait photograph of a young man in his twenties sitting on a park bench.",
    "A wooden bakery sign that reads \"FRESH BREAD\", hanging above a shop door.",
]


def _srgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    c = xyz @ np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])).T
    c = np.clip(c, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return (np.clip(c, 0, 1) * 255).round().astype(np.uint8)


def dull_colour(rgb, factor, cool):
    """Scale Lab chroma by `factor` and move coloured pixels toward blue by up to `cool` b* units (full shift at chroma 30);
    lightness, and pixels with no chroma (white, grey, black), are unchanged."""
    lab = _srgb_to_lab(rgb.astype(np.float64))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    lab[..., 1:] *= factor
    lab[..., 2] -= cool * np.clip(chroma / 30, 0, 1)
    return _lab_to_srgb(lab)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--merge-base", default=None, help="diffusion_models file merged into --unet on --merge-layers at --merge-ratio before training")
    p.add_argument("--merge-layers", default="24-35")
    p.add_argument("--merge-ratio", type=float, default=0.5, help="0 = --unet, 1 = --merge-base (the Block Merger's convention)")
    p.add_argument("--clip", default="qwen3.5_9b_int8_convrot.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--turbo", default="MOSS/HaDeS-MOSS-Turbo-Diverse-v0.6.safetensors", help="LoRA that renders the clean images x_t is made from")
    p.add_argument("--turbo-steps", type=int, default=4)
    p.add_argument("--pairs", type=int, default=0, help="image-pair mode: number of rendered images (see above); 0 = prompt mode")
    p.add_argument("--blur-mild", default="0.8-1.5", help="Gaussian blur radius range, pixels at --size, for most pairs")
    p.add_argument("--blur-strong", default="2.5-4.0", help="radius range for the --strong-frac share of pairs")
    p.add_argument("--strong-frac", type=float, default=0.2)
    p.add_argument("--pair-kind", choices=("blur", "colour"), default="blur",
                   help="blur: the poorer image is Gaussian-blurred (detail). colour: its Lab chroma is scaled down (--chroma-*) and coloured "
                        "pixels shifted toward blue by --cool b* units at chroma 30 and up; lightness and neutral pixels stay as they are")
    p.add_argument("--chroma-mild", default="0.75-0.9", help="colour pairs: chroma factor range for most pairs")
    p.add_argument("--chroma-strong", default="0.4-0.6", help="colour pairs: chroma factor range for the --strong-frac share")
    p.add_argument("--cool", type=float, default=0.0, help="colour pairs: b* shift toward blue on coloured pixels (0 = desaturate only)")
    p.add_argument("--pairs-from", default=None, help="pairs.pt of an earlier pair run: decode its clean images instead of rendering new ones")
    p.add_argument("--positive", default=None, help="prompt mode: suffix the slider moves toward, e.g. \"warm golden tones, rich natural colors\"")
    p.add_argument("--negative", default=None, help="prompt mode: suffix it moves away from, e.g. \"cool tones, muted desaturated colors\"")
    p.add_argument("--eta", type=float, default=1.0, help="scale of the positive - negative difference in the target")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--layers", default="0-41")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--num-prompts", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(ARGV)
    if not args.pairs and not (args.positive and args.negative):
        raise SystemExit("prompt mode needs --positive and --negative (or --pairs N for image-pair mode)")
    os.makedirs(args.out, exist_ok=True)
    lock = os.path.join(args.out, "TRAIN_RUNNING")
    if os.path.exists(lock):
        raise SystemExit(f"{lock} exists: another run may be writing to this folder (delete the file if it is not)")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    try:
        train(args)
    finally:
        os.remove(lock)


def train(args):
    patcher = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.unet))
    bm = patcher.model
    dm = bm.diffusion_model
    if args.merge_base:
        other = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.merge_base)).model.diffusion_model
        lo, hi = (int(v) for v in args.merge_layers.split("-"))
        theirs = dict(other.named_parameters())
        n = 0
        with torch.no_grad():
            for name, q in dm.named_parameters():
                parts = name.split(".")
                if parts[0] == "layers" and lo <= int(parts[1]) <= hi:
                    q.copy_((q.float() * (1 - args.merge_ratio) + theirs[name].float() * args.merge_ratio).to(q.dtype))
                    n += 1
        del other, theirs
        print(f"merged {n} tensors of layers {args.merge_layers} from {args.merge_base} at {args.merge_ratio}", flush=True)

    clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", args.clip)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", args.vae)))
    prompts = make_prompts(args.num_prompts, args.seed)
    cache_path = os.path.join(args.out, "prompts.pt")
    cache = torch.load(cache_path) if os.path.exists(cache_path) else {}
    needed = [""] + PREVIEW_PROMPTS + (prompts if args.pairs else [q for pr in prompts for q in (pr, f"{pr} {args.positive}", f"{pr} {args.negative}")])
    t0 = time.time()
    for i, prompt in enumerate(needed):
        if prompt not in cache:
            with torch.no_grad():
                cache[prompt] = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].to(torch.bfloat16).cpu()
            if i % 300 == 0:
                print(f"encoded {i + 1} / {len(needed)} prompts ({time.time() - t0:.0f}s)", flush=True)
    torch.save(cache, cache_path)
    with open(os.path.join(args.out, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=0, ensure_ascii=False)
    del clip

    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    device = comfy.model_management.get_torch_device()
    dm.requires_grad_(False)
    init_gen = torch.Generator(device=device).manual_seed(args.seed)
    slider = LoRASet(dm, layer_names(args.layers), args.rank, device, init_gen)
    tsd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", args.turbo))
    turbo = {k[len("diffusion_model."):-len(".lora_up.weight")]: (tsd[k].float().to(device), tsd[k.replace("lora_up", "lora_down")].float().to(device))
             for k in tsd if k.endswith(".lora_up.weight")}
    ema = {k: v.detach().clone() for k, v in slider.named_parameters()}
    hooked = sorted(set(slider.names) | set(turbo))
    for n in hooked:
        if n.endswith("ffn_down"):
            comfy.utils.get_attr(dm, n).comfy_force_forward = True  # the fused act + linear path would skip forward hooks
    state = {"mode": None, "scale": 1.0}  # mode None = frozen model, "slider", "turbo"; scale = slider strength

    def hook(name):
        def fn(module, inputs, output):
            scale = 1.0
            if state["mode"] == "slider" and name in slider.names:
                up, down = slider.factors(name)
                scale = state["scale"]
            elif state["mode"] == "turbo" and name in turbo:
                up, down = turbo[name]
            else:
                return output
            return output + (scale * ((inputs[0].float() @ down.T) @ up.T)).to(output.dtype)
        return fn

    handles = [comfy.utils.get_attr(dm, n).register_forward_hook(hook(n)) for n in hooked]
    opt = torch.optim.Adam(slider.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    log, start = [], 1
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        slider.load_state_dict(ck["slider"])
        opt.load_state_dict(ck["opt"])
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        rng.setstate(ck["rng"])
        noise_gen.set_state(ck["noise_gen"].cpu())
        log, start = ck["log"], ck["update"] + 1
        print(f"resumed from update {ck['update']}", flush=True)

    def ctx(prompt):
        return cache[prompt].to(device)

    def denoise(x, sigma, context, mode):
        state["mode"] = mode
        try:
            return bm.apply_model(x, sigma.view(1), c_crossattn=context).float()
        finally:
            state["mode"] = None

    def euler(x, den, s_from, s_to):
        return den + (x - den) * (s_to / s_from)

    def swap_ema():
        backup = {k: v.detach().clone() for k, v in slider.named_parameters()}
        with torch.no_grad():
            for k, v in slider.named_parameters():
                v.copy_(ema[k])
        return backup

    def restore(backup):
        with torch.no_grad():
            for k, v in slider.named_parameters():
                v.copy_(backup[k])

    def save_lora(path):
        backup = swap_ema()
        sd = slider.comfy_state_dict()
        restore(backup)
        sd = {k: v.to(torch.bfloat16) if v.ndim else v for k, v in sd.items()}
        meta = {"trainer": "HaDeS moss_slider_train", "eta": str(args.eta),
                **({"pairs": str(args.pairs), "pair_kind": args.pair_kind, "strong_frac": str(args.strong_frac),
                    **({"blur_mild": args.blur_mild, "blur_strong": args.blur_strong} if args.pair_kind == "blur"
                       else {"chroma_mild": args.chroma_mild, "chroma_strong": args.chroma_strong, "cool": str(args.cool)})}
                   if args.pairs else {"positive": args.positive, "negative": args.negative}),
                "layers": args.layers, "rank": str(args.rank), "trained_on": args.unet + (f" + {args.merge_base} {args.merge_layers} @ {args.merge_ratio}" if args.merge_base else "")}
        safetensors.torch.save_file(sd, path + ".tmp", metadata=meta)
        os.replace(path + ".tmp", path)

    def save_ckpt(update):
        torch.save({"update": update, "slider": slider.state_dict(), "opt": opt.state_dict(), "ema": ema, "rng": rng.getstate(),
                    "noise_gen": noise_gen.get_state(), "log": log, "args": vars(args)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        save_lora(os.path.join(args.out, "lora_ema.safetensors"))
        with open(os.path.join(args.out, "report_slider.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "update": update, "log": log}, f, indent=0)

    hw = args.size // 16
    t_sig = comfy.samplers.calculate_sigmas(bm.model_sampling, "simple", args.turbo_steps).to(device)
    p_sig = comfy.samplers.calculate_sigmas(bm.model_sampling, "simple", 25).to(device)

    def preview(update):
        backup = swap_ema()
        tiles = []
        try:
            with torch.no_grad():
                for mode in (None, "slider"):
                    for i, prompt in enumerate(PREVIEW_PROMPTS):
                        x = comfy.sample.prepare_noise(torch.zeros([1, 128, hw, hw]), 99 + i).to(device)
                        for s in range(25):
                            cond = denoise(x, p_sig[s], ctx(prompt), mode)
                            uncond = denoise(x, p_sig[s], ctx(""), mode)
                            x = euler(x, uncond + 2.5 * (cond - uncond), p_sig[s], p_sig[s + 1])
                        img = vae.decode(bm.process_latent_out(x).cpu())
                        img = img.reshape(-1, *img.shape[-3:])[0]
                        tiles.append(Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)))
        finally:
            restore(backup)
        n = len(PREVIEW_PROMPTS)
        sheet = Image.new("RGB", (args.size * n, args.size * 2))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % n) * args.size, (i // n) * args.size))
        sheet.save(os.path.join(args.out, f"preview_u{update:05d}.png"))
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    pairs = []
    if args.pairs:
        from PIL import ImageFilter
        pair_path = os.path.join(args.out, "pairs.pt")
        done = torch.load(pair_path) if os.path.exists(pair_path) else {}
        source = torch.load(args.pairs_from) if args.pairs_from else {}
        prng = random.Random(args.seed + 3)
        t1 = time.time()
        for i in range(args.pairs):
            prompt = prompts[i % len(prompts)]
            mild, strong = (args.blur_mild, args.blur_strong) if args.pair_kind == "blur" else (args.chroma_mild, args.chroma_strong)
            lo, hi = (float(v) for v in (strong if prng.random() < args.strong_frac else mild).split("-"))
            amount = prng.uniform(lo, hi)
            if i not in done:
                with torch.no_grad():
                    if i in source:  # the clean image of another pair run (same model, prompts and seeds), decoded
                        prompt = source[i]["prompt"]
                        img = vae.decode(bm.process_latent_out(source[i]["sharp"].float()))
                    else:
                        x = comfy.sample.prepare_noise(torch.zeros([1, 128, hw, hw]), 5000 + i).to(device)
                        for s in range(25):
                            cond = denoise(x, p_sig[s], ctx(prompt), None)
                            uncond = denoise(x, p_sig[s], ctx(""), None)
                            x = euler(x, uncond + 2.5 * (cond - uncond), p_sig[s], p_sig[s + 1])
                        img = vae.decode(bm.process_latent_out(x).cpu())
                    img = img.reshape(-1, *img.shape[-3:])[0].clamp(0, 1).cpu()
                    pil = Image.fromarray((img.numpy() * 255).round().astype(np.uint8))
                    if args.pair_kind == "blur":
                        soft = pil.filter(ImageFilter.GaussianBlur(amount))
                    else:
                        soft = Image.fromarray(dull_colour(np.asarray(pil), amount, args.cool))
                    lat = [bm.process_latent_in(vae.encode(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None])).to(torch.bfloat16).cpu()
                           for im in (pil, soft)]
                comfy.model_management.load_models_gpu([patcher], force_full_load=True)
                done[i] = {"prompt": prompt, "amount": amount, "sharp": lat[0], "blur": lat[1]}
                if i < 6:
                    side = Image.new("RGB", (args.size * 2, args.size))
                    side.paste(pil, (0, 0))
                    side.paste(soft, (args.size, 0))
                    side.save(os.path.join(args.out, f"pair_{i:02d}_{args.pair_kind}{amount:.2f}.png"))
                if (i + 1) % 16 == 0 or i + 1 == args.pairs:
                    torch.save(done, pair_path + ".tmp")
                    os.replace(pair_path + ".tmp", pair_path)
                    print(f"pairs: {i + 1} / {args.pairs} ({time.time() - t1:.0f}s)", flush=True)
            pairs.append(done[i])

    def prompt_step():
        """Prompt mode: x_t from a turbo image; target = frozen prediction + eta x (positive - negative)."""
        prompt = prompts[rng.randrange(len(prompts))]
        c, c_pos, c_neg = ctx(prompt), ctx(f"{prompt} {args.positive}"), ctx(f"{prompt} {args.negative}")
        with torch.no_grad():
            x = torch.randn(1, 128, hw, hw, device=device, generator=noise_gen)
            for s in range(args.turbo_steps):
                x = euler(x, denoise(x, t_sig[s], c, "turbo"), t_sig[s], t_sig[s + 1])
            tau = torch.tensor(rng.random() * 0.96 + 0.02, device=device)
            eps = torch.randn(x.shape, device=device, generator=noise_gen)
            x_t = (1 - tau) * x + tau * eps
            target = denoise(x_t, tau, c, None) + args.eta * (denoise(x_t, tau, c_pos, None) - denoise(x_t, tau, c_neg, None))
        pred = denoise(x_t, tau, c, "slider")
        loss = ((pred - target) / tau).square().mean()
        loss.backward()
        return loss.detach(), tau

    def pair_step():
        """Pair mode: same noise and level on both latents; LoRA at +1 fitted to the sharp one, at -1 to the blurred one."""
        pr = pairs[rng.randrange(len(pairs))]
        c = ctx(pr["prompt"])
        sharp, blurred = pr["sharp"].to(device).float(), pr["blur"].to(device).float()
        tau = torch.tensor(rng.random() * 0.96 + 0.02, device=device)
        eps = torch.randn(sharp.shape, device=device, generator=noise_gen)
        loss = 0.0
        try:
            for scale, x0 in ((1.0, sharp), (-1.0, blurred)):
                state["scale"] = scale
                pred = denoise((1 - tau) * x0 + tau * eps, tau, c, "slider")
                part = ((pred - x0) / tau).square().mean()
                part.backward()
                loss = loss + part.detach()
        finally:
            state["scale"] = 1.0
        return loss, tau

    if start == 1:
        preview(0)
    t0 = time.time()
    for update in range(start, args.updates + 1):
        opt.zero_grad(set_to_none=True)
        comfy.model_management.in_training = True
        try:
            with torch.enable_grad():
                loss, tau = pair_step() if pairs else prompt_step()
        finally:
            comfy.model_management.in_training = False
        if not torch.isfinite(loss) or any(not torch.isfinite(q.grad).all() for q in slider.parameters() if q.grad is not None):
            # a finite loss can still carry an inf gradient; clipping it would turn every weight into NaN
            print(f"u{update}: non-finite loss or gradient, step skipped", flush=True)
            continue
        gn = torch.nn.utils.clip_grad_norm_(list(slider.parameters()), 1.0)
        opt.step()
        decay = min(args.ema_decay, (1 + update) / (10 + update))
        with torch.no_grad():
            for k, v in slider.named_parameters():
                ema[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
        log.append({"u": update, "tau": round(tau.item(), 3), "loss": round(loss.item(), 5), "grad_norm": round(float(gn), 4),
                    "up_norm": round(sum(q.detach().square().sum().item() for q in slider.up.values()) ** 0.5, 5)})
        if update % 10 == 0 or update == start:
            el = time.time() - t0
            print(f"u{update}: loss {np.mean([e['loss'] for e in log[-10:]]):.4f} grad_norm {log[-1]['grad_norm']:.3g} up_norm {log[-1]['up_norm']:.4f} "
                  f"({el / (update - start + 1):.1f} s/update, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
        if update % args.save_every == 0 or update == args.updates:
            save_ckpt(update)
        if update % args.sample_every == 0 or update == args.updates:
            save_lora(os.path.join(args.out, f"lora_u{update:05d}.safetensors"))
            preview(update)
    for h in handles:
        h.remove()
    print(f"done: {args.updates} updates", flush=True)


if __name__ == "__main__":
    main()
