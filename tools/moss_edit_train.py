"""Train a MOSS-Image edit LoRA from zero on reference / target pairs (dev tool, not a node).

The pairs come from tools/moss_edit_pairs.py: reference A (B with an object added), the instruction "Remove the <object>." and
target B. Training uses the layout of the `Moss-Image-Edit` edit format, so the LoRA works in the core's Text Encode MOSS-Image
Edit and in the MOSS Prompt Enhancer's edit mode without changes:
  - the reference latent goes to the model as `ref_latents` with `index_timestep_zero_text_start`: the reference tokens sit at
    the text start, are run at t = 0 and attend only to each other; the target attends to them;
  - the instruction is encoded as plain text (this format gives the text encoder no image);
  - the loss is flow matching on the target only (the model returns only the target's tokens), in v-space as
    tools/moss_dmd_train.py; noise level u ~ U(0, 1) through the model's shift (5), as at inference;
  - the LoRA is a forward hook on every block's linears, so it also acts on the reference pass, as the published edit LoRAs do.
A share of the steps (--identity) trains reference A + an empty instruction -> A itself, so "no instruction" means "copy the
reference" and CFG has a direction to push away from. None of the values below are measured yet; the defaults are first tries.

Latents and text embeddings are cached in --out (cache_<size>.pt), so the text encoder and VAE are not kept loaded. Every
--save-every steps it writes ckpt.pt (LoRA, optimizer, random states; resumable) and lora_last.safetensors; every --sample-every
steps lora_sNNNNN.safetensors and a preview sheet: for --preview-pairs pairs, reference, target and the LoRA's edit (euler /
normal, --preview-steps steps, CFG --preview-cfg, the edit benchmark's negative with the same reference).

--grad-check runs one step and reports whether the gradient reaches the reference pass (its keys / values), then stops.

Run from the ComfyUI folder with the server's models freed (POST /free):
  python custom_nodes/Comfyui-HaDeS-Moss-Suite/tools/moss_edit_train.py --pairs I:/HaDeS_MOSS/moss_edit_pairs/pilot_qwen2509 --out I:/HaDeS_MOSS/moss_edit_runs/overfit512
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
from PIL import Image, ImageDraw  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

WEIGHTS = ("qkv", "o", "ffn_up_gate", "ffn_down")
METHOD = "index_timestep_zero_text_start"  # the core's EDIT_FORMATS["Moss-Image-Edit"]
NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
# --vary-instruction: a removal pair's instruction is drawn from these each time, so the LoRA ties the edit to the object, not to one sentence
VARIANTS = ("Remove the {}.", "Erase the {}.", "Get rid of the {}.", "Delete the {} from the picture.", "Take out the {}.", "Remove the {} from the image.")


def instructions(p, vary):
    if vary and p["instruction"].startswith("Remove the ") and p.get("object"):
        return [v.format(p["object"]) for v in VARIANTS]
    return [p["instruction"]]


class LoRASet(torch.nn.Module):
    """up / down factors per linear, fp32; delta = (x @ down.T) @ up.T * alpha / rank. Same layout as moss_dmd_train's."""

    def __init__(self, dm, names, rank, alpha, device, gen):
        super().__init__()
        self.names, self.scale, self.alpha = list(names), alpha / rank, alpha
        self.up, self.down = torch.nn.ParameterDict(), torch.nn.ParameterDict()
        for n in self.names:
            w = comfy.utils.get_attr(dm, n).weight
            down = torch.empty(rank, w.shape[1], device=device)
            torch.nn.init.kaiming_uniform_(down, a=math.sqrt(5), generator=gen)
            self.down[n.replace(".", "_")] = torch.nn.Parameter(down)
            self.up[n.replace(".", "_")] = torch.nn.Parameter(torch.zeros(w.shape[0], rank, device=device))

    def factors(self, name):
        k = name.replace(".", "_")
        return self.up[k], self.down[k]

    def comfy_state_dict(self):
        sd = {}
        for n in self.names:
            up, down = self.factors(n)
            sd[f"diffusion_model.{n}.lora_up.weight"] = up.detach().float().cpu().contiguous()
            sd[f"diffusion_model.{n}.lora_down.weight"] = down.detach().float().cpu().contiguous()
            sd[f"diffusion_model.{n}.alpha"] = torch.tensor(float(self.alpha))
        return sd


def layer_names(spec):
    lo, hi = (int(v) for v in spec.split("-"))
    return [f"layers.{i}.proj.{w}" for i in range(lo, hi + 1) for w in WEIGHTS]


def load_pairs(folder, max_area):
    """Usable pairs of a moss_edit_pairs folder: the text encoder saw the object in A and not in B, and the pasted area is small."""
    m = json.loads(open(os.path.join(folder, "pairs.json"), encoding="utf-8").read())
    keep = [p for p in m if p.get("vlm") == ["yes", "no"] and p.get("count", 1) == 1 and p.get("mask_area", 0) <= max_area]
    print(f"{len(keep)} of {len(m)} pairs usable (vlm yes/no, count 1 where checked, mask area <= {max_area})", flush=True)
    return sorted(keep, key=lambda p: p["pair_id"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--clip", default="qwen3.5_9b_int8_convrot.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--pairs", required=True, nargs="+", help="moss_edit_pairs folders")
    p.add_argument("--max-area", type=float, default=0.3, help="drop pairs whose pasted area covers more of the image (the editor ignored the scene)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1000, help="optimizer steps")
    p.add_argument("--accum", type=int, default=4, help="pairs per optimizer step (gradient accumulation, batch 1)")
    p.add_argument("--layers", default="0-41")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--identity", type=float, default=0.1, help="share of samples trained as A + empty instruction -> A")
    p.add_argument("--hold-out", type=int, default=4, help="at least this many pairs kept out of training, whole scenes at a time, shown in the previews")
    p.add_argument("--preview-pairs", type=int, default=4, help="training pairs shown in the previews, beside the held-out ones")
    p.add_argument("--preview-steps", type=int, default=20)
    p.add_argument("--preview-cfg", type=float, default=4.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-checkpoint", action="store_true", help="keep every block's activations (faster, but at 512 it already fills 32 GB)")
    p.add_argument("--grad-check", action="store_true")
    p.add_argument("--hold-out-objects", type=int, default=0, help="keep this many whole objects out of training (no pair removes or keeps them); the previews show up to --hold-out of them")
    p.add_argument("--vary-instruction", action="store_true", help="draw each removal instruction from several phrasings (Remove / Erase / Get rid of ...)")
    p.add_argument("--t2i", action="store_true", help="plain text-to-image LoRA: target b with its instruction as caption, no reference, no identity samples")
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


def build_cache(args, pairs):
    path = os.path.join(args.out, f"cache_{args.size}.pt")
    cache = torch.load(path) if os.path.exists(path) else {"lat": {}, "txt": {}}
    texts = sorted({t for p in pairs for t in instructions(p, args.vary_instruction)} | {"", NEG})
    imgs = sorted({(p["_dir"], p["a"]) for p in pairs} | {(p["_dir"], p["b"]) for p in pairs})
    if all(t in cache["txt"] for t in texts) and all(f"{d}/{n}" in cache["lat"] for d, n in imgs):
        return cache
    clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", args.clip)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    with torch.no_grad():
        for t in texts:
            if t not in cache["txt"]:
                cache["txt"][t] = clip.encode_from_tokens_scheduled(clip.tokenize(t))[0][0].to(torch.bfloat16).cpu()
    del clip
    comfy.model_management.unload_all_models()
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", args.vae)))
    t0 = time.time()
    with torch.no_grad():
        for i, (d, n) in enumerate(imgs):
            key = f"{d}/{n}"
            if key in cache["lat"]:
                continue
            im = Image.open(os.path.join(d, "pairs", n)).convert("RGB")
            # same area as size x size, aspect kept, sides a multiple of 16 (the latent's x16 reduction); square images are unchanged
            k = args.size / (im.width * im.height) ** 0.5
            im = im.resize((max(16, round(im.width * k / 16) * 16), max(16, round(im.height * k / 16) * 16)), Image.LANCZOS)
            cache["lat"][key] = vae.encode(torch.from_numpy(np.asarray(im, np.float32) / 255)[None]).to(torch.bfloat16).cpu()
            if (i + 1) % 50 == 0:
                print(f"encoded {i + 1} / {len(imgs)} images ({time.time() - t0:.0f}s)", flush=True)
    del vae
    comfy.model_management.unload_all_models()
    torch.save(cache, path + ".tmp")
    os.replace(path + ".tmp", path)
    return cache


def train(args):
    pairs = []
    for d in args.pairs:
        pairs += [{**p, "_dir": d} for p in load_pairs(d, args.max_area)]
    rng = random.Random(args.seed)
    # held out by scene: a pair whose B scene is also a training target (with another object) would test memory, not editing
    scenes = sorted({p["scene_id"] for p in pairs})
    rng.shuffle(scenes)
    held_scenes, held = set(), []
    if args.hold_out_objects:
        # held out by object: no pair removes or keeps these objects in training, so the previews show objects the LoRA never saw
        objs = sorted({p["object"] for p in pairs})
        rng.shuffle(objs)
        held_objs = set(objs[:args.hold_out_objects])
        involved = lambda p: p["object"] in held_objs or bool(held_objs & set(p.get("kept", [])))  # noqa: E731
        held_all = [p for p in pairs if p["object"] in held_objs and not p.get("kept")]
        held = [next(p for p in held_all if p["object"] == o) for o in sorted(held_objs) if any(p["object"] == o for p in held_all)][:args.hold_out]
        held_scenes = {p["scene_id"] for p in held}
        pairs_train = [p for p in pairs if not involved(p) and p["scene_id"] not in held_scenes]
        print(f"held-out objects: {sorted(held_objs)}", flush=True)
    for sc in [] if args.hold_out_objects else scenes:
        if len(held) >= args.hold_out:
            break
        held_scenes.add(sc)
        held += [p for p in pairs if p["scene_id"] == sc]
    train_pairs = pairs_train if args.hold_out_objects else [p for p in pairs if p["scene_id"] not in held_scenes]
    with open(os.path.join(args.out, "split.json"), "w", encoding="utf-8") as f:
        json.dump({"held_out": [p["pair_id"] for p in held], "train": [p["pair_id"] for p in train_pairs]}, f, indent=0)
    print(f"{len(train_pairs)} training pairs, {len(held)} held out", flush=True)
    cache = build_cache(args, pairs)

    patcher = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", args.unet))
    bm, dm = patcher.model, patcher.model.diffusion_model
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    device = comfy.model_management.get_torch_device()
    shift = float(getattr(bm.model_sampling, "shift", 5.0))
    print(f"model shift {shift}", flush=True)

    dm.requires_grad_(False)
    lora = LoRASet(dm, layer_names(args.layers), args.rank, args.alpha, device, torch.Generator(device=device).manual_seed(args.seed))
    for n in lora.names:
        if n.endswith("ffn_down"):
            comfy.utils.get_attr(dm, n).comfy_force_forward = True  # the fused act + linear path would skip forward hooks
    state = {"on": True}

    def hook(name):
        def fn(module, inputs, output):
            if not state["on"]:
                return output
            up, down = lora.factors(name)
            return output + (((inputs[0].float() @ down.T) @ up.T) * lora.scale).to(output.dtype)
        return fn

    handles = [comfy.utils.get_attr(dm, n).register_forward_hook(hook(n)) for n in lora.names]
    opt = torch.optim.AdamW(lora.parameters(), lr=args.lr, weight_decay=0.0)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    log, start = [], 1
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path) and not args.grad_check:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        lora.load_state_dict(ck["lora"])
        opt.load_state_dict(ck["opt"])
        rng.setstate(ck["rng"])
        noise_gen.set_state(ck["noise_gen"].cpu())
        log, start = ck["log"], ck["step"] + 1
        print(f"resumed from step {ck['step']}", flush=True)

    def lat(p, side):
        return bm.process_latent_in(cache["lat"][f"{p['_dir']}/{p[side]}"].to(device).float())

    def txt(t):
        return cache["txt"][t].to(device)

    # gradient checkpointing: the reference pass as a whole (only its keys / values are kept), the target pass per block. The
    # block replacement calls dm.layers[i] itself with this pass's reference keys / values: the core's original_block closure
    # reads its loop variables late, which a recompute in backward would see as the last block.
    orig_refs = dm.preprocess_refs
    kv_now = {}

    def refs_ckpt(refs, transformer_options={}):
        kv = torch.utils.checkpoint.checkpoint(lambda *r: orig_refs(list(r), transformer_options), *refs, use_reentrant=False)             if torch.is_grad_enabled() else orig_refs(refs, transformer_options)
        kv_now["kv"] = kv
        return kv

    def ckpt_block(i):
        def fn(a, extra):
            kv = kv_now["kv"]
            on = state["on"]

            def run(img):
                old, state["on"] = state["on"], on
                try:
                    return dm.layers[i](img, a["mod"], a["pe"], a["transformer_options"], None, kv[:, i] if kv is not None else None, mask=None)
                finally:
                    state["on"] = old
            return {"img": torch.utils.checkpoint.checkpoint(run, a["img"], use_reentrant=False)}
        return fn

    ckpt_options = {"patches_replace": {"dit": {("single_block", i): ckpt_block(i) for i in range(len(dm.layers))}}}

    def denoise(x, sigma, context, ref):
        options = {}
        if ref is None:  # --t2i: no reference pass
            kv_now["kv"] = None
            if not args.no_checkpoint and torch.is_grad_enabled():
                options = ckpt_options
            return bm.apply_model(x, sigma.view(1), c_crossattn=context, transformer_options=options).float()
        if not args.no_checkpoint and torch.is_grad_enabled():
            dm.preprocess_refs = refs_ckpt
            options = ckpt_options
        try:
            return bm.apply_model(x, sigma.view(1), c_crossattn=context, ref_latents=[ref], ref_latents_method=METHOD,
                                  transformer_options=options).float()
        finally:
            dm.preprocess_refs = orig_refs

    def with_grad(fn):
        comfy.model_management.in_training = True
        try:
            with torch.enable_grad():
                return fn()
        finally:
            comfy.model_management.in_training = False

    def sample_loss(p):
        ref, ident = lat(p, "a"), (not args.t2i) and rng.random() < args.identity
        x0 = ref if ident else lat(p, "b")
        context = txt("" if ident else rng.choice(instructions(p, args.vary_instruction)))
        if args.t2i:
            ref = None
        u = rng.random()
        sigma = torch.tensor(min(max(shift * u / (1 + (shift - 1) * u), 0.02), 0.999), device=device)
        eps = torch.randn(x0.shape, device=device, generator=noise_gen)
        x_t = (1 - sigma) * x0 + sigma * eps
        den = denoise(x_t, sigma, context, ref)
        return ((den - x0) / sigma).square().mean(), float(sigma), ident

    if args.grad_check:
        seen = {}
        inner = orig_refs

        def spy(refs, transformer_options={}):
            kv = inner(refs, transformer_options)
            seen["requires_grad"] = kv.requires_grad
            if kv.requires_grad:
                kv.register_hook(lambda g: seen.__setitem__("kv_grad_norm", float(g.float().norm())))
            return kv
        orig_refs = spy  # noqa: F841 (read by refs_ckpt / denoise through the closure)
        with_grad(lambda: sample_loss(train_pairs[0])[0].backward())
        up_grad = sum(float(q.grad.norm()) ** 2 for q in lora.up.values() if q.grad is not None) ** 0.5
        print(json.dumps({"ref_kv_requires_grad": seen.get("requires_grad"), "ref_kv_grad_norm": seen.get("kv_grad_norm"),
                          "lora_up_grad_norm": up_grad, "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
        for h in handles:
            h.remove()
        return

    def save_lora(path):
        tmp = path + ".tmp"
        safetensors.torch.save_file(lora.comfy_state_dict(), tmp, metadata={
            "trainer": "HaDeS moss_edit_train", **({} if args.t2i else {"edit_format": "Moss-Image-Edit"}), "base_model": args.unet,
            "layers": args.layers, "rank": str(args.rank), "alpha": str(args.alpha), "size": str(args.size)})
        os.replace(tmp, path)

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"step": step, "lora": lora.state_dict(), "opt": opt.state_dict(), "rng": rng.getstate(),
                    "noise_gen": noise_gen.get_state(), "log": log, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)
        save_lora(os.path.join(args.out, "lora_last.safetensors"))
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "step": step, "log": log}, f, indent=0)

    vae_holder = {}

    def preview(step):
        shown = held + train_pairs[:args.preview_pairs]
        sig = comfy.samplers.calculate_sigmas(bm.model_sampling, "normal", args.preview_steps).to(device)
        edits = []
        with torch.no_grad():
            for i, p in enumerate(shown):
                ref = lat(p, "a")
                shape = ref.shape
                if args.t2i:
                    ref = None
                x = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(1000 + i)) * sig[0]
                for s in range(args.preview_steps):
                    cond = denoise(x, sig[s], txt(p["instruction"]), ref)
                    uncond = denoise(x, sig[s], txt(NEG), ref)
                    den = uncond + args.preview_cfg * (cond - uncond)
                    x = den + (x - den) * (sig[s + 1] / sig[s])
                edits.append(bm.process_latent_out(x).cpu())
        if "vae" not in vae_holder:
            vae_holder["vae"] = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", args.vae)))
        t = 256
        sheet = Image.new("RGB", (3 * t, len(shown) * (t + 18)), (18, 18, 20))
        d = ImageDraw.Draw(sheet)
        for i, (p, e) in enumerate(zip(shown, edits)):
            with torch.no_grad():
                img = vae_holder["vae"].decode(e)
            img = img.reshape(-1, *img.shape[-3:])[0]
            tiles = [Image.open(os.path.join(p["_dir"], "pairs", p["a"])).convert("RGB"), Image.open(os.path.join(p["_dir"], "pairs", p["b"])).convert("RGB"),
                     Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))]
            for j, im in enumerate(tiles):
                sheet.paste(im.resize((t, t)), (j * t, i * (t + 18) + 18))
            d.text((2, i * (t + 18) + 2), f"{'HELD OUT ' if p in held else ''}{p['pair_id']}: {p['instruction']}  (reference | target | step {step})", fill=(230, 230, 230))
        sheet.save(os.path.join(args.out, f"preview_s{step:05d}.jpg"), quality=88)
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    if start == 1:
        state["on"] = False
        preview(0)  # SFT without the LoRA
        state["on"] = True
    t0 = time.time()
    for step in range(start, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses, sigmas, idents = [], [], 0
        for _ in range(args.accum):
            p = train_pairs[rng.randrange(len(train_pairs))]

            def run():
                loss, s, ident = sample_loss(p)
                (loss / args.accum).backward()
                return loss, s, ident
            loss, s, ident = with_grad(run)
            losses.append(float(loss))
            sigmas.append(s)
            idents += ident
        grads = [q.grad for q in lora.parameters() if q.grad is not None]
        if not all(math.isfinite(v) for v in losses) or any(not torch.isfinite(g).all() for g in grads):
            print(f"s{step}: non-finite loss or gradient, step skipped", flush=True)
            continue
        gn = torch.nn.utils.clip_grad_norm_(list(lora.parameters()), 1.0)
        opt.step()
        log.append({"s": step, "loss": round(float(np.mean(losses)), 5), "sigma": round(float(np.mean(sigmas)), 3), "identity": idents, "grad_norm": round(float(gn), 4)})
        if step % 10 == 0 or step == start:
            el = time.time() - t0
            print(f"s{step}: loss {np.mean([e['loss'] for e in log[-10:]]):.4f} grad_norm {log[-1]['grad_norm']:.3g} "
                  f"({el / (step - start + 1):.1f} s/step, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(step)
        if step % args.sample_every == 0 or step == args.steps:
            save_lora(os.path.join(args.out, f"lora_s{step:05d}.safetensors"))
            preview(step)
    for h in handles:
        h.remove()
    print(f"done: {args.steps} steps", flush=True)


if __name__ == "__main__":
    main()
