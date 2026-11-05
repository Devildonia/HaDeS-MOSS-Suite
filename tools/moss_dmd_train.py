"""Train a few-step MOSS-Image LoRA with DMD (distribution matching distillation), on one GPU (dev tool, not a node).

Follows the recipe of akhaliq/MOSS-Image-SFT-4-steps (train_dmd.py and train_args.json there, Apache 2.0): teacher = frozen SFT at
CFG 3 with the negative pass skipping layer 22; fake score = SFT + a fresh LoRA trained by flow matching on the generator's images,
one update per generator update; generator gradient = v_real - v_fake at a noised copy of its image (noise level ~ Beta(2.5, 1));
Adam betas (0, 0.999), gradient norm clip 1, EMA of the generator. What differs:
  - one copy of the base model: the generator's and the fake score's LoRAs are forward hooks switched on per pass (fp32, beside the
    bf16 weights), and the teacher is the base with both off;
  - the generator LoRA lives only where tools/RESULTS.md found the few-step effect (default layers 24-41, rank 2) and starts from
    zero, with no warm start from another LoRA;
  - the generator samples the stock euler / simple schedule at --steps, CFG 1 (what the user runs in ComfyUI); each update runs it
    without gradient from noise down to a random step and takes the gradient on that step's clean-image prediction (DMD2's
    backward simulation), instead of backpropagating through every step;
  - one prompt per update (batch 1); prompts from a subject x style x text generator below, none of them a moss_eval test prompt.

Every --save-every updates it writes ckpt.pt (both LoRAs, optimizers, EMA, random states; written to a temporary file and renamed,
so a crash never leaves half a checkpoint) and lora_ema.safetensors (the EMA generator, ComfyUI LoRA format: loads in the stock
LoRA loader). Started again with the same --out, it resumes from ckpt.pt. Every --sample-every updates it also keeps
lora_uNNNNN.safetensors and renders the preview prompts (preview_uNNNNN.png).

Uses PyTorch attention (SageAttention has no backward). Run from the ComfyUI folder with the server's models freed (POST /free):
  python custom_nodes/HaDeS-MOSS-Suite/tools/moss_dmd_train.py --size 512 --updates 1000 --out H:/moss_eval/dmd512
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.argv, ARGV = [sys.argv[0], "--use-pytorch-cross-attention"], sys.argv[1:]
sys.path.insert(0, os.getcwd())
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import numpy as np  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

WEIGHTS = ("qkv", "o", "ffn_up_gate", "ffn_down")

SUBJECTS = [
    "a lighthouse keeper holding a lantern on a stormy night", "a mountain village covered in snow at dusk",
    "two kids flying a kite on a windy beach", "a jazz trio playing on a small stage", "a vintage car parked on an empty desert road",
    "a girl feeding pigeons in a stone plaza", "a bowl of ramen on a wooden counter, steam rising", "a fishing boat in a foggy harbor",
    "a cat sleeping on a windowsill between potted plants", "a crowded train station at rush hour", "an astronaut tending plants in a greenhouse on Mars",
    "a grandfather knitting by a fireplace", "a waterfall in a tropical rainforest", "a skateboarder jumping down a flight of stairs",
    "a library with tall wooden shelves and a rolling ladder", "a dragon curled around a castle tower", "a young man reading a newspaper on a park bench",
    "a busy night market with food stalls and lanterns", "a glass perfume bottle on white marble", "a winding river through green farmland at sunrise",
    "a golden retriever running through shallow water", "a chef plating a dish in a professional kitchen", "a cyclist passing a graffiti wall",
    "a sailing ship on rough seas", "a cozy reading nook with a cat and plants", "a girl with a red umbrella waiting at a bus stop in the rain",
    "an isometric island with a windmill and a river", "a medieval market square full of people", "a forest temple at dusk with fireflies",
    "a bear and a rabbit having a picnic on a hill", "a robot watering flowers in a city garden", "a hot air balloon over a canyon",
    "a ballet dancer mid-leap in an empty studio", "a farmer with a basket of tomatoes in a field", "a close-up of a hummingbird at a flower",
    "a rainy city street at night with reflections", "a teenage boy playing guitar on a rooftop", "an old man playing chess in a park",
    "a snowy pine forest at dawn", "a bowl of fruit on a kitchen table by a window",
]
STYLES = [
    "Documentary photograph, natural light", "Studio photograph, soft key light", "Cinematic photograph, shallow depth of field",
    "35mm film photograph, grain", "Watercolor illustration, loose washes", "Flat vector illustration, bold shapes",
    "Oil painting, impasto brush strokes", "Anime illustration, clean line art, cel shading", "3D render, soft global illumination",
    "Ink sketch, cross-hatching", "Isometric illustration, pastel colors", "Pixel art, 16-bit style",
]
TEXT_DESIGNS = [
    ("a bookshop window with gold lettering", "RARE BOOKS"), ("a concert flyer with big red letters above a guitar", "LIVE TONIGHT"),
    ("a wooden crate stamped with black letters", "FRAGILE"), ("a retro diner sign in glowing red neon", "EAT"),
    ("a minimal coffee bag label with a small mountain logo", "DARK ROAST"), ("a movie poster over a rainy platform", "NIGHT TRAIN"),
    ("a hand-lettered wedding sign, white script on green", "WELCOME"), ("a tram destination board in orange dot-matrix letters", "CENTRAL STATION"),
    ("a classroom chalkboard in white chalk", "MATH QUIZ FRIDAY"), ("a price tag on a fruit stand", "APPLES 1.99"),
    ("a running race banner over a road with runners", "FINISH"), ("a postcard with large letters over a pagoda", "GREETINGS FROM KYOTO"),
    ("a coffee shop chalkboard menu", "ESPRESSO 2.50"), ("a neon sign on a brick wall in pink letters", "OPEN LATE"),
    ("a book cover with elegant serif letters over a botanical pattern", "THE SILENT GARDEN"), ("a street sign at an intersection", "MAPLE AVE"),
    ("an infographic poster with arrows and clouds", "WATER CYCLE"), ("a vintage travel poster with yellow trams", "VISIT LISBON"),
    ("a birthday cake with blue icing", "HAPPY 30"), ("a museum banner hanging from columns", "ANCIENT EGYPT"),
    ("a gym schedule board", "YOGA 18:30"), ("a subway sign with a white arrow", "EXIT C"), ("a tea tin label with a crane", "GREEN TEA"),
    ("a protest placard held up in a crowd", "SAVE THE BEES"), ("a cinema marquee with black letters", "NOW SHOWING"),
]
PREVIEW_PROMPTS = [
    "Photograph of a small record store front with a red awning and a sign that reads \"VINYL DAYS\" in white letters, afternoon light.",
    "Photograph of a young woman laughing on a city rooftop at sunset, wind in her hair, warm light.",
]


LIGHTS = ["soft morning light", "golden hour", "overcast daylight", "night, artificial light", "dramatic side light", "bright midday sun"]


def make_prompts(n, seed):
    """n distinct prompts, about 40% with text; fails if n is more than the generator can make."""
    styles = [s.split(",")[0] for s in STYLES]
    text = [f"{st} of {scene} that reads \"{words}\"." for st in styles for scene, words in TEXT_DESIGNS]
    plain = [f"{st}: {subject}, {light}." for st in STYLES for subject in SUBJECTS for light in LIGHTS]
    n_text = min(len(text), round(n * 0.4))
    if n - n_text > len(plain):
        raise SystemExit(f"--num-prompts {n}: the generator makes at most {len(plain) + n_text}")
    rng = random.Random(seed)
    return sorted(rng.sample(text, n_text) + rng.sample(plain, n - n_text))


class LoRASet(torch.nn.Module):
    """up / down factors per linear, fp32; delta = (x @ down.T) @ up.T (alpha = rank)."""

    def __init__(self, dm, names, rank, device, gen):
        super().__init__()
        self.names = list(names)
        self.up = torch.nn.ParameterDict()
        self.down = torch.nn.ParameterDict()
        for n in self.names:
            w = comfy.utils.get_attr(dm, n).weight
            key = n.replace(".", "_")
            down = torch.empty(rank, w.shape[1], device=device)
            torch.nn.init.kaiming_uniform_(down, a=math.sqrt(5), generator=gen)  # nn.Linear's init, as the recipe
            self.down[key] = torch.nn.Parameter(down)
            self.up[key] = torch.nn.Parameter(torch.zeros(w.shape[0], rank, device=device))

    def factors(self, name):
        key = name.replace(".", "_")
        return self.up[key], self.down[key]

    def comfy_state_dict(self):
        sd = {}
        for n in self.names:
            up, down = self.factors(n)
            pre = f"diffusion_model.{n}"
            sd[f"{pre}.lora_up.weight"] = up.detach().float().cpu().contiguous()
            sd[f"{pre}.lora_down.weight"] = down.detach().float().cpu().contiguous()
            sd[f"{pre}.alpha"] = torch.tensor(float(down.shape[0]))
        return sd


def layer_names(spec):
    lo, hi = (int(v) for v in spec.split("-"))
    return [f"layers.{i}.proj.{w}" for i in range(lo, hi + 1) for w in WEIGHTS]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--clip", default="mossImage_base_txt_int4.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4, help="generator steps (euler / simple, CFG 1)")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--layers", default="24-41", help="generator LoRA layers")
    p.add_argument("--rank", type=int, default=2, help="generator LoRA rank")
    p.add_argument("--fake-layers", default="0-41")
    p.add_argument("--fake-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5, help="generator learning rate (first try, not measured; the recipe's 5e-6 continued a warm start)")
    p.add_argument("--fake-lr", type=float, default=1e-4, help="fake score learning rate (2 x --lr, the recipe's ratio)")
    p.add_argument("--guidance", type=float, default=3.0, help="teacher CFG")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--num-prompts", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-lora", default=None, help="start the generator from this LoRA file (written by this trainer, same --layers / --rank) instead of zero")
    p.add_argument("--fake-warmup", type=int, default=0, help="first N updates train only the fake score (for a generator that does not start from zero)")
    p.add_argument("--checkpoint", action="store_true", help="gradient checkpointing per block (recomputes each block in backward: less memory, needed at 1024)")
    p.add_argument("--out", required=True)
    args = p.parse_args(ARGV)
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
    clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", args.clip)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", args.vae)))
    bm = patcher.model
    dm = bm.diffusion_model

    # prompt embeddings, cached in --out (CPU, bf16)
    prompts = make_prompts(args.num_prompts, args.seed)
    cache_path = os.path.join(args.out, "prompts.pt")
    cache = torch.load(cache_path) if os.path.exists(cache_path) else {}
    t0 = time.time()
    for i, prompt in enumerate([""] + PREVIEW_PROMPTS + prompts):
        if prompt not in cache:
            with torch.no_grad():
                cache[prompt] = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].to(torch.bfloat16).cpu()
            if i % 100 == 0:
                print(f"encoded {i + 1} prompts ({time.time() - t0:.0f}s)", flush=True)
    torch.save(cache, cache_path)
    with open(os.path.join(args.out, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=0, ensure_ascii=False)
    del clip

    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    device = comfy.model_management.get_torch_device()
    sigmas = comfy.samplers.calculate_sigmas(bm.model_sampling, "simple", args.steps).to(device)
    print("generator sigmas: " + ", ".join(f"{s:.4f}" for s in sigmas.tolist()), flush=True)

    dm.requires_grad_(False)
    init_gen = torch.Generator(device=device).manual_seed(args.seed)
    gen = LoRASet(dm, layer_names(args.layers), args.rank, device, init_gen)
    fake = LoRASet(dm, layer_names(args.fake_layers), args.fake_rank, device, init_gen)
    if args.init_lora:
        sd = comfy.utils.load_torch_file(args.init_lora)
        with torch.no_grad():
            for n in gen.names:
                up, down = gen.factors(n)
                up.copy_(sd[f"diffusion_model.{n}.lora_up.weight"].to(device))
                down.copy_(sd[f"diffusion_model.{n}.lora_down.weight"].to(device))
        print(f"generator starts from {args.init_lora}", flush=True)
    ema = {k: v.detach().clone() for k, v in gen.named_parameters()}
    hooked = sorted(set(gen.names) | set(fake.names))
    for n in hooked:
        if n.endswith("ffn_down"):
            comfy.utils.get_attr(dm, n).comfy_force_forward = True  # the fused act + linear path would skip forward hooks
    state = {"mode": None}  # None = teacher (base), "gen", "fake"

    def hook(name):
        def fn(module, inputs, output):
            lora = gen if state["mode"] == "gen" else fake if state["mode"] == "fake" else None
            if lora is None or name not in lora.names:
                return output
            up, down = lora.factors(name)
            return output + ((inputs[0].float() @ down.T) @ up.T).to(output.dtype)
        return fn

    handles = [comfy.utils.get_attr(dm, n).register_forward_hook(hook(n)) for n in hooked]
    skip22 = {"patches_replace": {"dit": {("single_block", 22): lambda a, extra: {"img": a["img"]}}}}

    opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.999))
    fake_opt = torch.optim.Adam(fake.parameters(), lr=args.fake_lr, betas=(0.0, 0.999))
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    log, skipped, start = [], 0, 1
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        gen.load_state_dict(ck["gen"])
        fake.load_state_dict(ck["fake"])
        opt.load_state_dict(ck["opt"])
        fake_opt.load_state_dict(ck["fake_opt"])
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        rng.setstate(ck["rng"])
        np_rng.set_state(ck["np_rng"])
        noise_gen.set_state(ck["noise_gen"].cpu())
        log, skipped, start = ck["log"], ck["skipped"], ck["update"] + 1
        print(f"resumed from update {ck['update']}", flush=True)

    def ctx(prompt):
        return cache[prompt].to(device)

    def ckpt_block(i):
        # calls dm.layers[i] itself: the core's original_block closure reads its loop variable late, which a recompute in
        # backward would see as the last block. No references or attention bias here (ref, ref_kv, mask = None).
        def fn(a, extra):
            mode = state["mode"]  # the recompute runs in backward, after denoise() has reset the mode

            def run(img):
                old, state["mode"] = state["mode"], mode
                try:
                    return dm.layers[i](img, a["mod"], a["pe"], a["transformer_options"], None, None, mask=None)
                finally:
                    state["mode"] = old
            return {"img": torch.utils.checkpoint.checkpoint(run, a["img"], use_reentrant=False)}
        return fn

    ckpt_options = {"patches_replace": {"dit": {("single_block", i): ckpt_block(i) for i in range(len(dm.layers))}}}

    def denoise(x, sigma, context, mode, options=None):
        state["mode"] = mode
        if args.checkpoint and torch.is_grad_enabled():
            assert options is None
            options = ckpt_options
        try:
            return bm.apply_model(x, sigma.view(1), c_crossattn=context, transformer_options=options or {}).float()
        finally:
            state["mode"] = None

    def euler(x, den, s_from, s_to):
        return den + (x - den) * (s_to / s_from)

    def with_grad(fn):
        comfy.model_management.in_training = True
        try:
            with torch.enable_grad():
                return fn()
        finally:
            comfy.model_management.in_training = False

    def save_lora(path, use_ema=True):
        backup = swap_ema() if use_ema else None
        sd = gen.comfy_state_dict()
        if backup is not None:
            restore(backup)
        tmp = path + ".tmp"
        safetensors.torch.save_file(sd, tmp, metadata={"trainer": "HaDeS moss_dmd_train", "steps": str(args.steps),
                                                       "layers": args.layers, "rank": str(args.rank)})
        os.replace(tmp, path)

    def swap_ema():
        backup = {k: v.detach().clone() for k, v in gen.named_parameters()}
        with torch.no_grad():
            for k, v in gen.named_parameters():
                v.copy_(ema[k])
        return backup

    def restore(backup):
        with torch.no_grad():
            for k, v in gen.named_parameters():
                v.copy_(backup[k])

    def save_ckpt(update):
        tmp = ckpt_path + ".tmp"
        torch.save({"update": update, "gen": gen.state_dict(), "fake": fake.state_dict(), "opt": opt.state_dict(),
                    "fake_opt": fake_opt.state_dict(), "ema": ema, "rng": rng.getstate(), "np_rng": np_rng.get_state(),
                    "noise_gen": noise_gen.get_state(), "log": log, "skipped": skipped, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)
        save_lora(os.path.join(args.out, "lora_ema.safetensors"))
        with open(os.path.join(args.out, "report_dmd.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "update": update, "skipped": skipped, "log": log}, f, indent=0)

    def preview(update):
        backup = swap_ema()
        tiles = []
        try:
            with torch.no_grad():
                for i, prompt in enumerate(PREVIEW_PROMPTS):
                    latent = torch.zeros([1, 128, args.size // 16, args.size // 16])
                    x = comfy.sample.prepare_noise(latent, 99 + i).to(device)
                    for s in range(args.steps):
                        x = euler(x, denoise(x, sigmas[s], ctx(prompt), "gen"), sigmas[s], sigmas[s + 1])
                    img = vae.decode(bm.process_latent_out(x).cpu())
                    img = img.reshape(-1, *img.shape[-3:])[0]
                    tiles.append(Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)))
        finally:
            restore(backup)
        strip = Image.new("RGB", (args.size * len(tiles), args.size))
        for i, t in enumerate(tiles):
            strip.paste(t, (i * args.size, 0))
        strip.save(os.path.join(args.out, f"preview_u{update:05d}.png"))
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    if start == 1:
        preview(0)
    neg = ctx("")
    hw = args.size // 16
    t0 = time.time()
    for update in range(start, args.updates + 1):
        context = ctx(prompts[rng.randrange(len(prompts))])
        x = torch.randn(1, 128, hw, hw, device=device, generator=noise_gen)
        k = rng.randrange(args.steps)

        # generator: no gradient down to step k, then its clean-image prediction at step k with gradient
        with torch.no_grad():
            for s in range(k):
                x = euler(x, denoise(x, sigmas[s], context, "gen"), sigmas[s], sigmas[s + 1])
        warm = update <= args.fake_warmup
        if warm:
            with torch.no_grad():
                x0 = denoise(x, sigmas[k], context, "gen")
        else:
            x0 = with_grad(lambda: denoise(x, sigmas[k], context, "gen"))
        x0_det = x0.detach()

        # fake score: flow matching on the generator's image (v-space loss, as the recipe)
        tau = torch.tensor(rng.random() * 0.96 + 0.02, device=device)
        eps = torch.randn(x0_det.shape, device=device, generator=noise_gen)
        x_t = (1 - tau) * x0_det + tau * eps

        def fake_step():
            den = denoise(x_t, tau, context, "fake")
            loss = ((den - x0_det) / tau).square().mean()
            fake_opt.zero_grad(set_to_none=True)
            loss.backward()
            return loss
        fake_loss = with_grad(fake_step)
        fake_ok = torch.isfinite(fake_loss) and all(torch.isfinite(q.grad).all() for q in fake.parameters() if q.grad is not None)
        if fake_ok:
            fake_opt.step()
        if warm:
            log.append({"u": update, "k": k, "fake_loss": round(fake_loss.item(), 5), "warmup": True})
            if update % 10 == 0 or update == start:
                print(f"u{update} (fake warm-up): fake_loss {np.mean([e['fake_loss'] for e in log[-10:]]):.4f} "
                      f"({torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
            if update % args.save_every == 0:
                save_ckpt(update)
            continue

        # generator DMD gradient: v_real - v_fake = (den_fake - den_real) / tau_g at a noised copy of its image
        tau_g = torch.tensor(float(np_rng.beta(2.5, 1.0)) * 0.96 + 0.02, device=device)
        eps_g = torch.randn(x0_det.shape, device=device, generator=noise_gen)
        x_g = (1 - tau_g) * x0_det + tau_g * eps_g
        with torch.no_grad():
            cond = denoise(x_g, tau_g, context, None)
            uncond = denoise(x_g, tau_g, neg, None, skip22)
            real = uncond + args.guidance * (cond - uncond)
            fake_den = denoise(x_g, tau_g, context, "fake")
            grad = (fake_den - real) / tau_g
        opt.zero_grad(set_to_none=True)
        with_grad(lambda: x0.backward(grad))
        grads = [q.grad for q in gen.parameters() if q.grad is not None]
        if not fake_ok or not torch.isfinite(grad).all() or any(not torch.isfinite(g).all() for g in grads):
            skipped += 1
            print(f"u{update}: non-finite loss or gradient, step skipped ({skipped} so far)", flush=True)
            continue
        gn = torch.nn.utils.clip_grad_norm_(list(gen.parameters()), 1.0)
        opt.step()
        u = update - args.fake_warmup
        decay = min(args.ema_decay, (1 + u) / (10 + u))  # warm-up of the EMA from the starting generator
        with torch.no_grad():
            for kk, v in gen.named_parameters():
                ema[kk].mul_(decay).add_(v.detach(), alpha=1 - decay)

        log.append({"u": update, "k": k, "fake_loss": round(fake_loss.item(), 5), "gap": round((fake_den - real).abs().mean().item(), 5),
                    "grad_norm": round(float(gn), 4), "up_norm": round(sum(q.detach().square().sum().item() for q in gen.up.values()) ** 0.5, 5)})
        if update % 10 == 0 or update == start:
            el = time.time() - t0
            print(f"u{update}: fake_loss {np.mean([e['fake_loss'] for e in log[-10:]]):.4f} gap {np.mean([e['gap'] for e in log[-10:] if 'gap' in e]):.4f} "
                  f"grad_norm {log[-1]['grad_norm']:.3g} up_norm {log[-1]['up_norm']:.4f} "
                  f"({el / (update - start + 1):.1f} s/update, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
        if update % args.save_every == 0 or update == args.updates:
            save_ckpt(update)
        if update % args.sample_every == 0 or update == args.updates:
            save_lora(os.path.join(args.out, f"lora_u{update:05d}.safetensors"))
            preview(update)
    for h in handles:
        h.remove()
    print(f"done: {args.updates} updates, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
