"""Fit few-step biases to MOSS-Image with no LoRA (dev tool, not a node).

tools/RESULTS.md ("Few-step LoRAs") found that a few-step LoRA's effect is close to one bias vector per linear of layers 24-41 with a
per-step scale. This fits such biases directly: the student is --unet at --steps euler / simple / CFG 1 steps with one learnable
bias per linear of layers 24-41 and per step (zeros at the start, nothing from a LoRA); the teacher is the same model at
--teacher-steps euler / simple / CFG --teacher-cfg, deterministic, so every noise has one teacher image. At each student step the
loss is the mean squared distance between the student's clean-image prediction and the teacher's final latent for that noise; the
student's own trajectory feeds the next step (detached). Only the biases train.

Before fitting it reports the loss with zero biases and with --compare-biases (e.g. biases calibrated from a LoRA by
tools/moss_bias_turbo.py --calib-only), to check that the objective rewards what the LoRA does. Saves biases.safetensors in --out
in the moss_bias_turbo.py format (test them there with --biases).

Uses PyTorch attention (SageAttention has no backward). Run from the ComfyUI folder with the server's models freed (POST /free):
  python custom_nodes/HaDeS-MOSS-Suite/tools/moss_bias_fit.py --out H:/moss_eval/bias_fit --compare-biases H:/moss_eval/biases_3p/biases.safetensors
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.argv, ARGV = [sys.argv[0], "--use-pytorch-cross-attention"], sys.argv[1:]
sys.path.insert(0, os.getcwd())
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import safetensors.torch  # noqa: E402
import torch  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

# training prompts, none of them a moss_eval test prompt
TRAIN_PROMPTS = [
    "Photograph of a young man reading a newspaper on a park bench in autumn, soft afternoon light.",
    "Close-up portrait of an old woman with silver hair and deep wrinkles, smiling, dark background, studio light.",
    "Photograph of a busy night market in Taipei with neon signs, steam rising from food stalls, crowds.",
    "Product photo of a glass perfume bottle on white marble, soft shadows, minimal.",
    "Aerial photograph of a winding river through green farmland at sunrise, mist.",
    "Photograph of a golden retriever puppy running through shallow water at the beach.",
    "Photograph of a chef plating a dish in a professional kitchen, motion, warm light.",
    "Street photograph of a cyclist passing a red brick wall covered in graffiti, overcast.",
    "Oil painting of a stormy sea with a lighthouse on a cliff, dramatic clouds, thick brush strokes.",
    "Flat vector illustration of a cozy reading nook with plants and a cat, pastel colors.",
    "Anime style illustration of a girl with a red umbrella standing in the rain at a bus stop.",
    "Pencil sketch of an old wooden sailing ship, detailed cross-hatching, white paper.",
    "3D render of a small isometric island with a windmill, trees and a river, soft lighting.",
    "Ink and watercolor illustration of a crowded medieval market square, warm palette.",
    "Pixel art of a forest temple at dusk with fireflies, 16-bit style.",
    "Children's book illustration of a bear and a rabbit having a picnic on a hill.",
    "Photograph of a coffee shop chalkboard menu that reads \"ESPRESSO 2.50\" and \"LATTE 3.80\" in white chalk.",
    "Minimal poster with the words \"SUMMER JAZZ NIGHTS\" in large orange letters on a navy background, a saxophone line drawing.",
    "Photograph of a neon sign on a brick wall that says \"OPEN LATE\" in pink letters.",
    "Book cover design with the title \"THE SILENT GARDEN\" in elegant serif letters over a dark green botanical pattern.",
    "Photograph of a street sign at an intersection reading \"MAPLE AVE\" and \"5TH ST\".",
    "Infographic poster titled \"WATER CYCLE\" with arrows, clouds, rain and a sun, clean flat style, labels.",
    "Vintage travel poster that reads \"VISIT LISBON\" with yellow trams and terracotta roofs.",
    "Photograph of a birthday cake with \"HAPPY 30\" written in blue icing, candles lit.",
]
# --prompt-set large: TRAIN_PROMPTS plus subject x style pairs and more text designs, none of them a moss_eval test prompt
SUBJECTS = ["a lighthouse keeper with a lantern", "a mountain village in winter", "two kids flying a kite on a beach", "a jazz trio on a small stage",
            "a vintage car parked on a desert road", "a girl feeding pigeons in a plaza", "a bowl of ramen on a wooden counter",
            "a fisherman's boat in a foggy harbor", "a cat sleeping on a windowsill with plants", "a busy train station at rush hour",
            "an astronaut tending a greenhouse on Mars", "a grandmother knitting by a fireplace", "a waterfall in a tropical forest",
            "a skateboarder jumping over stairs", "a library with tall shelves and a ladder", "a dragon curled around a castle tower"]
STYLES = ["documentary photograph, natural light", "studio portrait photograph, soft key light", "watercolor illustration, loose washes",
          "flat vector illustration, bold shapes", "oil painting, impasto brush strokes", "anime illustration, clean line art, cel shading"]
TEXT_DESIGNS = ["Bookshop window with gold lettering that reads \"RARE BOOKS\".",
                "Concert flyer that says \"LIVE TONIGHT\" in big red letters above a guitar silhouette.",
                "Wooden crate stamped with the words \"FRAGILE\" and \"THIS SIDE UP\".",
                "Retro diner sign that reads \"EAT\" in glowing red neon.",
                "Minimal coffee bag label that reads \"DARK ROAST\" with a small mountain logo.",
                "Movie poster with the title \"NIGHT TRAIN\" over a rainy platform.",
                "Hand-lettered wedding sign that says \"WELCOME\" in white script on green.",
                "Tram destination board reading \"CENTRAL STATION\" in orange dot-matrix letters.",
                "Classroom chalkboard with \"MATH QUIZ FRIDAY\" written in white chalk.",
                "Grocery price tag reading \"APPLES 1.99\" on a fruit stand.",
                "Running race banner that says \"FINISH\" over a road with runners.",
                "Postcard design with \"GREETINGS FROM KYOTO\" in large letters over a pagoda."]


def prompt_set(name):
    if name == "base":
        return list(TRAIN_PROMPTS)
    pairs = [f"{s[0].upper()}{s[1:]}, {style}." for i, s in enumerate(SUBJECTS) for j, style in enumerate(STYLES) if (i + j) % 2 == 0]
    return list(TRAIN_PROMPTS) + pairs + TEXT_DESIGNS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--clip", default="mossImage_base_txt_int4.safetensors")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--teacher-steps", type=int, default=25)
    p.add_argument("--teacher-cfg", type=float, default=2.5)
    p.add_argument("--seeds", type=int, default=1, help="teacher seeds per training prompt (from 5000)")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0, help="gradient norm clip")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--layers", default="24-41")
    p.add_argument("--compare-biases", help="biases.safetensors to score with the same loss before fitting")
    p.add_argument("--prompt-set", choices=("base", "large"), default="base")
    p.add_argument("--max-prompts", type=int, default=0, help="use the first N training prompts (smoke tests; 0 = all)")
    p.add_argument("--param", choices=("per_step", "shared"), default="per_step",
                   help="per_step: a free bias vector per linear and step; shared: one vector per linear times a learned scale per step")
    p.add_argument("--objective", choices=("final", "teacher_step"), default="final",
                   help="final: the clean-image prediction against the teacher's final latent; teacher_step: from the student's state the "
                        "teacher runs its own --teacher-steps schedule (CFG) down to the next student sigma, and the student's one euler step "
                        "must land there")
    p.add_argument("--out", required=True)
    args = p.parse_args(ARGV)
    os.makedirs(args.out, exist_ok=True)
    lock = os.path.join(args.out, "FIT_RUNNING")
    if os.path.exists(lock):
        raise SystemExit(f"{lock} exists: another fit may be writing to this folder (delete the file if it is not)")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    try:
        fit(args)
    finally:
        os.remove(lock)


def fit(args):
    lo, hi = (int(v) for v in args.layers.split("-"))
    prompts = prompt_set(args.prompt_set)
    prompts = prompts[:args.max_prompts] if args.max_prompts else prompts

    patcher = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.unet))
    clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", args.clip)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    bm = patcher.model
    dm = bm.diffusion_model
    names = [f"layers.{i}.proj.{w}" for i in range(lo, hi + 1) for w in ("qkv", "o", "ffn_up_gate", "ffn_down")]
    for name in names:
        if name.endswith("ffn_down"):
            comfy.utils.get_attr(dm, name).comfy_force_forward = True
    out_dim = {n: comfy.utils.get_attr(dm, n).weight.shape[0] for n in names}
    lat_hw = args.size // 16

    # teacher: deterministic euler, one final latent per (prompt, seed)
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    data, cache = [], os.path.join(args.out, "teacher.pt")
    saved = torch.load(cache) if os.path.exists(cache) else {}
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        for seed in range(5000, 5000 + args.seeds):
            key = f"{hashlib.md5(prompt.encode()).hexdigest()[:12]}_{seed}_{args.teacher_steps}_{args.teacher_cfg:g}"
            latent = torch.zeros([1, 128, lat_hw, lat_hw])
            noise = comfy.sample.prepare_noise(latent, seed)
            if key not in saved:
                with torch.no_grad():  # not inference_mode: its tensors could not be reused under autograd
                    saved[key] = comfy.sample.sample(patcher, noise, args.teacher_steps, args.teacher_cfg, "euler", "simple", pos, neg, latent,
                                                     seed=seed, disable_pbar=True).cpu()
                torch.save(saved, cache)
                print(f"teacher {key} ({time.time() - t0:.0f}s)", flush=True)
            data.append((pos[0][0], noise, saved[key]))

    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    device = comfy.model_management.get_torch_device()
    sigmas = comfy.samplers.calculate_sigmas(bm.model_sampling, "simple", args.steps).to(device)
    if args.param == "per_step":
        biases = torch.nn.ParameterDict({f"{n.replace('.', '_')}__{s}": torch.nn.Parameter(torch.zeros(out_dim[n], device=device))
                                         for n in names for s in range(args.steps)})
    else:  # one vector per linear, one scale per linear and step
        biases = torch.nn.ParameterDict({f"{n.replace('.', '_')}__v": torch.nn.Parameter(torch.zeros(out_dim[n], device=device)) for n in names})
        biases.update({f"{n.replace('.', '_')}__s{s}": torch.nn.Parameter(torch.ones((), device=device)) for n in names for s in range(args.steps)})
    state = {"step": 0, "on": True}

    def bias_of(name, step):
        key = name.replace(".", "_")
        if args.param == "per_step":
            return biases[f"{key}__{step}"]
        return biases[f"{key}__v"] * biases[f"{key}__s{step}"]

    def adder(name):
        def fn(module, inputs, output):
            if not state["on"]:
                return output
            return output + bias_of(name, state["step"]).to(output.dtype)
        return fn

    handles = [comfy.utils.get_attr(dm, n).register_forward_hook(adder(n)) for n in names]

    teacher_sigmas = comfy.samplers.calculate_sigmas(bm.model_sampling, "simple", args.teacher_steps).to(device)
    neg_context = neg[0][0]
    timing = {"student_fwd": 0.0, "student_bwd": 0.0, "teacher": 0.0}

    def denoise(x, sigma, context):
        return bm.apply_model(x, sigma.view(1), c_crossattn=context.to(device))

    def euler(x, den, s_from, s_to):
        return den + (x - den) * (s_to / s_from)

    def teacher_to(x, s_from, s_to, context):
        """The teacher (biases off, CFG) from x at sigma s_from down to s_to along its own schedule."""
        t = time.time()
        state["on"] = False
        try:
            path = [s_from] + [s for s in teacher_sigmas.tolist() if s_to < s < s_from] + [s_to]
            with torch.no_grad():
                for a, b in zip(path[:-1], path[1:]):
                    a_t = torch.tensor(a, device=device)
                    cond = denoise(x, a_t, context)
                    uncond = denoise(x, a_t, neg_context)
                    x = euler(x, uncond + args.teacher_cfg * (cond - uncond), a, b)
        finally:
            state["on"] = True
        timing["teacher"] += time.time() - t
        return x

    def episode(context, noise, target, train):
        """Student trajectory from noise; returns the mean per-step loss (and backpropagates it when training)."""
        x = noise.to(device)  # flow model at full denoise: x_T = noise
        target = bm.process_latent_in(target.to(device))
        total = 0.0
        for s in range(args.steps):
            state["step"] = s
            goal = target if args.objective == "final" else teacher_to(x, sigmas[s].item(), sigmas[s + 1].item(), context)
            if train:
                comfy.model_management.in_training = True
                try:
                    with torch.enable_grad():
                        t = time.time()
                        den = denoise(x, sigmas[s], context)
                        pred = den if args.objective == "final" else euler(x, den, sigmas[s], sigmas[s + 1])
                        loss = (pred.float() - goal.float()).square().mean()
                        torch.cuda.synchronize(); timing["student_fwd"] += time.time() - t; t = time.time()
                        (loss / args.steps).backward()
                        torch.cuda.synchronize(); timing["student_bwd"] += time.time() - t
                finally:
                    comfy.model_management.in_training = False
                den = den.detach()
            else:
                with torch.no_grad():
                    den = denoise(x, sigmas[s], context)
                    pred = den if args.objective == "final" else euler(x, den, sigmas[s], sigmas[s + 1])
                    loss = (pred.float() - goal.float()).square().mean()
            total += loss.item() / args.steps
            x = euler(x, den, sigmas[s], sigmas[s + 1]).detach()
        return total

    def score(label):
        val = sum(episode(c, n, t, False) for c, n, t in data) / len(data)
        print(f"loss {label}: {val:.5f}", flush=True)
        return val

    report = {"loss_zero": score("zero biases")}
    if args.compare_biases and args.param != "per_step":
        print("--compare-biases needs --param per_step; skipped", flush=True)
    elif args.compare_biases:
        cmp = safetensors.torch.load_file(args.compare_biases)
        with torch.no_grad():
            for n in names:
                for s in range(args.steps):
                    k = f"{n}.step{s}.img"
                    if k in cmp:
                        biases[f"{n.replace('.', '_')}__{s}"].copy_(cmp[k].to(device))
        report["loss_compare"] = score(f"compare ({os.path.basename(os.path.dirname(args.compare_biases))})")
        with torch.no_grad():
            for b in biases.values():
                b.zero_()

    # shared: Adam on the scales at the vectors' rate would move them as fast as a whole vector element; keep them slower
    groups = [{"params": [b for k, b in biases.items() if not k.split("__")[1].startswith("s")], "lr": args.lr}]
    if args.param == "shared":
        groups.append({"params": [b for k, b in biases.items() if k.split("__")[1].startswith("s")], "lr": args.lr * 0.1})

    opt = torch.optim.Adam(groups)
    rng = random.Random(0)
    log, skipped, t0 = [], 0, time.time()
    for it in range(1, args.iters + 1):
        c, n, t = data[rng.randrange(len(data))]
        opt.zero_grad(set_to_none=True)
        loss = episode(c, n, t, True)
        grads = [b.grad for b in biases.values() if b.grad is not None]
        if not (loss == loss) or any(not torch.isfinite(g).all() for g in grads):  # NaN / inf: drop the step, keep the biases
            skipped += 1
            print(f"iter {it}: non-finite loss or gradient, step skipped ({skipped} so far)", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(list(biases.values()), args.grad_clip)
        opt.step()
        log.append(loss)
        if it % 10 == 0 or it <= 3:
            k = min(10, len(log))
            print(f"iter {it}: loss {sum(log[-k:]) / k:.5f} ({time.time() - t0:.0f}s; "
                  + ", ".join(f"{a} {b:.0f}s" for a, b in timing.items()) + ")", flush=True)
        if it % 100 == 0 or it == args.iters:
            tensors = {f"{n}.step{s}.{kind}": bias_of(n, s).detach().float().cpu().contiguous()
                       for n in names for s in range(args.steps) for kind in ("img", "txt")}
            safetensors.torch.save_file(tensors, os.path.join(args.out, "biases.safetensors"),
                                        metadata={"fit": "moss_bias_fit", "iters": str(it), "steps": str(args.steps)})
    report["loss_fitted"] = score("fitted")
    for h in handles:
        h.remove()
    report.update(iters=args.iters, lr=args.lr, grad_clip=args.grad_clip, skipped=skipped, objective=args.objective, size=args.size,
                  param=args.param, prompt_set=args.prompt_set, prompts=len(prompts), seeds=args.seeds, teacher_steps=args.teacher_steps, log=log)
    with open(os.path.join(args.out, "report_bias_fit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
