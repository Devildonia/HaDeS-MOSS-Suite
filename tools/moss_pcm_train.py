"""Train a few-step MOSS-Image LoRA with phased consistency distillation (PCM), on one GPU (dev tool, not a node).

Follows Phased Consistency Models (Wang et al. 2024, arXiv 2405.18407), variant PCD* (their Alg. 2): the teacher's ODE without a
baked-in CFG and a condition-drop ratio, so the student keeps a conditional and an unconditional branch and real CFG and negative
prompts still work at inference. Adapted here to MOSS-Image's flow matching (x_t = (1 - t) x0 + t noise, the model predicts x0):
  - phases: the stock euler / simple schedule at --steps (what the user runs in ComfyUI) cuts t = 1 .. 0 into --steps phases;
    the student's step is one euler step from t down to its phase's lower edge s, f(x_t, t) = D + (x_t - D) s / t, so sampling the
    student at --steps euler / simple steps chains the phases exactly;
  - consistency inside a phase: each phase is split into --sub uniform sub-intervals; x at the upper point t' is made from a data
    latent and noise, one teacher euler step takes it to the lower point t, and f_student(x_t', t') is pulled toward
    f_target(x_t, t) (the same LoRA without gradient; --ema-decay > 0 for an EMA target instead); at the phase edge the target is x_t;
  - teacher: the frozen model, conditional on the prompt, or on the empty prompt with probability --cond-drop (the same drop for
    the student), no CFG, no skipped layer;
  - data: latents of the teacher's own images (default: the 2,048 Base images at 512 rendered for Turbo 8-steps v2.0, euler / simple
    25 steps CFG 3, with their prompts' cached embeddings), so the student learns the teacher's trajectories, not a dataset's;
  - the LoRA lives in --layers at --rank (default 24-41, rank 16), starts from zero; no adversarial loss.

Every --save-every updates: ckpt.pt (resumable) and lora_last.safetensors; every --sample-every: lora_uNNNNN.safetensors and a preview
strip of the preview prompts at --steps, CFG 1 and CFG --preview-cfg. Run from the ComfyUI folder with the server's models freed:
  python custom_nodes/Comfyui-HaDeS-Moss-Suite/tools/moss_pcm_train.py --unet mossImage_base_bf16.safetensors --out I:/HaDeS_MOSS/pcm/pilot
"""
import argparse
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
from moss_dmd_train import PREVIEW_PROMPTS, LoRASet, layer_names  # noqa: E402
from moss_train_guard import prompt_tag, run_lock, seal_run  # noqa: E402


MAX_RUN_SKIPS = 20  # non-finite gradients in a row before the run stops (with its checkpoint saved) instead of spinning


def untagged_draw_ok(run, seed, pairs, steps, size):
    """An untagged regression cache (written before the prompt tag, e.g. Turbo v2.0's, 2026-11-25) does not say which prompt each
    pair was made for. It is accepted only when the data run's own report_dmd.json shows it was drawn with this seed, this number
    of pairs, these teacher steps and this size, so the draw rebuilt here is the one that made it (audit 0.9.9 A4). Returns the
    reason it is not, or None."""
    path = os.path.join(run, "report_dmd.json")
    try:
        a = json.load(open(path, encoding="utf-8")).get("args", {})
    except (OSError, ValueError):
        return f"{path} is missing or unreadable"
    want = {"seed": seed, "reg_pairs": pairs, "reg_teacher_steps": steps, "size": size}
    bad = [f"{k} {a.get(k)!r} (asked {v!r})" for k, v in want.items() if a.get(k) != v]
    return f"{path} records " + ", ".join(bad) if bad else None


def load_data(run, n_prompts, seed, pairs, steps, size):
    """(prompt, latent) pairs of a moss_dmd_train run's regression cache, with that run's prompt embeddings; the prompt order is
    rebuilt the way the trainer drew it and checked against the tag in each key. Returns (data, cache, manifest): the manifest
    records where every pair came from (index, prompt tag, key, tagged or verified untagged)."""
    prompts = json.load(open(os.path.join(run, "prompts.json"), encoding="utf-8"))
    if len(prompts) != n_prompts:
        raise SystemExit(f"{run}: {len(prompts)} prompts, expected {n_prompts}")
    reg_prompts = random.Random(seed + 7).sample(prompts, pairs)
    done = torch.load(os.path.join(run, "reg_pairs.pt"))
    cache = torch.load(os.path.join(run, "prompts.pt"))
    data, rows, untagged = [], [], []
    for i, prompt in enumerate(reg_prompts):
        tag = prompt_tag(prompt)
        key = f"{i}_{steps}_{size}_{tag}"
        tagged = key in done
        if not tagged:
            key = f"{i}_{steps}_{size}"
            untagged.append(i)
        if key not in done:
            raise SystemExit(f"{run}: pair {i} missing ({i}_{steps}_{size}_{tag})")
        if prompt not in cache:
            raise SystemExit(f"{run}/prompts.pt has no embedding for pair {i}'s prompt {prompt[:80]!r}")
        data.append((prompt, done[key]))
        rows.append([i, tag, key, tagged])
    if untagged:
        why = untagged_draw_ok(run, seed, pairs, steps, size)
        if why:
            raise SystemExit(f"{run}: {len(untagged)} regression pairs have no prompt tag and their draw cannot be verified ({why}); "
                             "their targets may belong to other prompts. Rebuild the cache with moss_dmd_train or pass the run's own "
                             "--data-seed / --data-pairs / --data-steps / --size")
    manifest = {"data_run": os.path.abspath(run), "pairs": len(data), "tagged": len(data) - len(untagged),
                "untagged_verified": len(untagged), "draw": {"seed": seed, "pairs": pairs, "steps": steps, "size": size},
                "rows": rows}
    return data, cache, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_base_bf16.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--data-run", default=r"I:/HaDeS_MOSS/moss_turbo_runs/dmd512_base_v20", help="a moss_dmd_train run with reg_pairs.pt, prompts.json, prompts.pt")
    p.add_argument("--data-prompts", type=int, default=2048)
    p.add_argument("--data-pairs", type=int, default=2048)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--data-steps", type=int, default=25)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=8, help="phases = inference steps (euler / simple)")
    p.add_argument("--sub", type=int, default=8, help="sub-intervals per phase (teacher euler steps across a phase)")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch", type=int, default=4, help="samples per update (accumulated)")
    p.add_argument("--layers", default="24-41")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5, help="pilot choice, not measured")
    p.add_argument("--cond-drop", type=float, default=0.1, help="probability of the empty prompt for teacher and student (PCD*)")
    p.add_argument("--ema-decay", type=float, default=0.0, help="0: the target is the current LoRA without gradient (the paper's LoRA setting)")
    p.add_argument("--preview-cfg", type=float, default=3.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(ARGV)
    os.makedirs(args.out, exist_ok=True)
    with run_lock(args.out):
        seal_run(args.out, args, {"unet": folder_paths.get_full_path_or_raise("diffusion_models", args.unet),
                                  "vae": folder_paths.get_full_path_or_raise("vae", args.vae), "data_run": args.data_run}, structural=("steps",))
        train(args)


def train(args):
    data, cache, manifest = load_data(args.data_run, args.data_prompts, args.data_seed, args.data_pairs, args.data_steps, args.size)
    man_path = os.path.join(args.out, "data_manifest.json")
    if os.path.exists(man_path):  # a resume must read the same pairs it was trained on
        with open(man_path, encoding="utf-8") as f:
            if json.load(f).get("rows") != manifest["rows"]:
                raise SystemExit(f"{man_path}: the data run now gives other pairs than this run was trained on; use a new --out")
    else:
        tmp = f"{man_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=0, ensure_ascii=False)
        os.replace(tmp, man_path)
    for pr in [""] + PREVIEW_PROMPTS:
        if pr not in cache:
            raise SystemExit(f"{args.data_run}/prompts.pt has no embedding for {pr!r}")
    print(f"{len(data)} data latents from {args.data_run} ({manifest['tagged']} tagged, {manifest['untagged_verified']} untagged with "
          "a verified draw)", flush=True)
    patcher = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.unet))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", args.vae)))
    bm = patcher.model
    dm = bm.diffusion_model
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    device = comfy.model_management.get_torch_device()

    edges = comfy.samplers.calculate_sigmas(bm.model_sampling, "simple", args.steps).float().to(device)  # s_0 = 1 > ... > s_steps = 0
    grid = [torch.linspace(float(edges[m]), float(edges[m + 1]), args.sub + 1, device=device) for m in range(args.steps)]
    print("phase edges: " + ", ".join(f"{s:.4f}" for s in edges.tolist()), flush=True)

    dm.requires_grad_(False)
    gen = LoRASet(dm, layer_names(args.layers), args.rank, device, torch.Generator(device=device).manual_seed(args.seed))
    ema = {k: v.detach().clone() for k, v in gen.named_parameters()} if args.ema_decay > 0 else None
    for n in gen.names:
        if n.endswith("ffn_down"):
            comfy.utils.get_attr(dm, n).comfy_force_forward = True
    state = {"on": False}

    def hook(name):
        def fn(module, inputs, output):
            if not state["on"]:
                return output
            up, down = gen.factors(name)
            return output + ((inputs[0].float() @ down.T) @ up.T).to(output.dtype)
        return fn
    handles = [comfy.utils.get_attr(dm, n).register_forward_hook(hook(n)) for n in gen.names]

    def denoise(x, t, context, lora):
        state["on"] = lora
        try:
            return bm.apply_model(x, t.view(1), c_crossattn=context).float()
        finally:
            state["on"] = False

    def step_to(x, den, t, s):  # euler from t to s with the clean-image prediction den
        return den + (x - den) * (s / t)

    def with_grad(fn):
        comfy.model_management.in_training = True
        try:
            with torch.enable_grad():
                return fn()
        finally:
            comfy.model_management.in_training = False

    def swap(params):
        backup = {k: v.detach().clone() for k, v in gen.named_parameters()}
        with torch.no_grad():
            for k, v in gen.named_parameters():
                v.copy_(params[k])
        return backup

    def save_lora(path, params=None):
        backup = swap(params) if params is not None else None
        sd = gen.comfy_state_dict()
        if backup is not None:
            swap(backup)
        tmp = path + ".tmp"
        safetensors.torch.save_file(sd, tmp, metadata={"trainer": "HaDeS moss_pcm_train", "steps": str(args.steps), "layers": args.layers,
                                                       "rank": str(args.rank), "method": "PCM (PCD*, flow)"})
        os.replace(tmp, path)

    opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    log, start, skipped, run_skips = [], 1, 0, 0
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        gen.load_state_dict(ck["gen"])
        opt.load_state_dict(ck["opt"])
        if ema is not None:
            ema = {k: v.to(device) for k, v in ck["ema"].items()}
        rng.setstate(ck["rng"])
        noise_gen.set_state(ck["noise_gen"].cpu())
        log, start, skipped = ck["log"], ck["update"] + 1, ck.get("skipped", 0)
        print(f"resumed from update {ck['update']}", flush=True)

    ctx = lambda prompt: cache[prompt].to(device)
    empty = ctx("")

    def preview(update):
        tiles = []
        with torch.no_grad():
            for cfg in (1.0, args.preview_cfg):
                for i, prompt in enumerate(PREVIEW_PROMPTS):
                    x = comfy.sample.prepare_noise(torch.zeros([1, 128, args.size // 16, args.size // 16]), 99 + i).to(device)
                    for s in range(args.steps):
                        cond = denoise(x, edges[s], ctx(prompt), True)
                        if cfg == 1.0:
                            den = cond
                        else:
                            unc = denoise(x, edges[s], empty, True)
                            den = unc + cfg * (cond - unc)
                        x = step_to(x, den, edges[s], edges[s + 1])
                    img = vae.decode(bm.process_latent_out(x).cpu())
                    img = img.reshape(-1, *img.shape[-3:])[0]
                    tiles.append(Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)))
        strip = Image.new("RGB", (args.size * len(PREVIEW_PROMPTS), args.size * 2))
        for j, t in enumerate(tiles):
            strip.paste(t, ((j % len(PREVIEW_PROMPTS)) * args.size, (j // len(PREVIEW_PROMPTS)) * args.size))
        strip.save(os.path.join(args.out, f"preview_u{update:05d}.png"))
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    if start == 1:
        preview(0)
    t0 = time.time()
    for update in range(start, args.updates + 1):
        opt.zero_grad(set_to_none=True)
        losses, edge_hits = [], 0
        for _ in range(args.batch):
            prompt, x0 = data[rng.randrange(len(data))]
            c = empty if rng.random() < args.cond_drop else ctx(prompt)
            m, n = rng.randrange(args.steps), rng.randrange(args.sub)
            t_hi, t_lo, s_edge = grid[m][n], grid[m][n + 1], edges[m + 1]
            x0 = x0.to(device).float()
            eps = torch.randn(x0.shape, device=device, generator=noise_gen)
            x_hi = (1 - t_hi) * x0 + t_hi * eps
            with torch.no_grad():
                x_lo = step_to(x_hi, denoise(x_hi, t_hi, c, False), t_hi, t_lo)  # one teacher euler step, no CFG
                if n == args.sub - 1:  # t_lo is the phase edge: f(x, s) = x
                    target = x_lo
                    edge_hits += 1
                else:
                    backup = swap(ema) if ema is not None else None
                    target = step_to(x_lo, denoise(x_lo, t_lo, c, True), t_lo, s_edge)
                    if backup is not None:
                        swap(backup)

            def student():
                pred = step_to(x_hi, denoise(x_hi, t_hi, c, True), t_hi, s_edge)
                loss = (pred - target).square().mean()
                (loss / args.batch).backward()
                return loss.detach()
            losses.append(float(with_grad(student)))
        grads = [q.grad for q in gen.parameters() if q.grad is not None]
        finite = all(torch.isfinite(g).all() for g in grads)
        # a non-finite gradient skips the optimizer step only: the update still counts as an attempt and still reaches the
        # checkpoint / LoRA / preview saves below, so the last update always leaves a coherent final file (audit 0.9.9 M10)
        if finite:
            run_skips = 0
            gn = torch.nn.utils.clip_grad_norm_(list(gen.parameters()), 1.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in gen.named_parameters():
                        ema[k].mul_(args.ema_decay).add_(v.detach(), alpha=1 - args.ema_decay)
        else:
            skipped += 1
            run_skips += 1
            opt.zero_grad(set_to_none=True)
            print(f"u{update}: non-finite gradient, optimizer step skipped ({skipped} so far)", flush=True)
        log.append({"u": update, "loss": round(float(np.mean(losses)), 6) if np.isfinite(np.mean(losses)) else None,
                    "grad_norm": round(float(gn), 4) if finite else None, "edge": edge_hits, "skipped": not finite,
                    "up_norm": round(sum(q.detach().square().sum().item() for q in gen.up.values()) ** 0.5, 5)})
        stop = run_skips >= MAX_RUN_SKIPS
        if update % 10 == 0 or update == start:
            el = time.time() - t0
            fin = [e["loss"] for e in log[-10:] if e["loss"] is not None]
            gtxt = f"{log[-1]['grad_norm']:.3g}" if finite else "skipped"
            print(f"u{update}: loss {np.mean(fin) if fin else float('nan'):.5f} grad_norm {gtxt} up_norm {log[-1]['up_norm']:.4f} "
                  f"({el / (update - start + 1):.2f} s/update, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
        if update % args.save_every == 0 or update == args.updates or stop:
            tmp = ckpt_path + ".tmp"
            torch.save({"update": update, "gen": gen.state_dict(), "opt": opt.state_dict(), "ema": ema, "rng": rng.getstate(),
                        "noise_gen": noise_gen.get_state(), "log": log, "skipped": skipped, "args": vars(args)}, tmp)
            os.replace(tmp, ckpt_path)
            save_lora(os.path.join(args.out, "lora_last.safetensors"))
            with open(os.path.join(args.out, "report_pcm.json"), "w", encoding="utf-8") as f:
                json.dump({"args": vars(args), "update": update, "optimizer_steps": update - skipped, "skipped": skipped,
                           "log": log}, f, indent=0)
        if stop:
            for h in handles:
                h.remove()
            raise SystemExit(f"u{update}: {run_skips} non-finite gradients in a row; stopped with the checkpoint and lora_last saved at "
                             f"u{update} ({update - skipped} optimizer steps, {skipped} skipped). Lower --lr or check the data.")
        if update % args.sample_every == 0 or update == args.updates:
            save_lora(os.path.join(args.out, f"lora_u{update:05d}.safetensors"), ema)
            preview(update)
    for h in handles:
        h.remove()
    print(f"done: {args.updates} updates ({args.updates - skipped} optimizer steps, {skipped} skipped for a non-finite gradient)", flush=True)


if __name__ == "__main__":
    main()
