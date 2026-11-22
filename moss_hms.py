"""HMS v1: the HaDeS MOSS Suite quantization recipe, for the MOSS Model Saver.

A MOSS-Image DiT saved in ComfyUI's own native formats (comfy-kitchen W4A8 / W6A8, loaded by the stock UNETLoader):
attention (qkv, o) and the FFN of layers 30-41 in W6A8, the other FFN in W4A8, then calibrated on the model's own
activations: activation-weighted scales and 4-bit codebook (C1), then GPTQ in the ConvRot basis (C2). The calibration
runs the model on seven fixed prompts (four scenes, three with quoted text), so every HMS v1 file is made the same way.
Ported unchanged from the quant lab tools (tools/moss_codec_capture.py, moss_hms_calibrate.py, moss_hms_gptq.py,
moss_native_mix_convert.py); measurements in tools/RESULTS.md ("HMS v1: decision", "HMS v1 on SFT and Bulle d'Encre").
"""
import json
import logging
import time

import torch

log = logging.getLogger(__name__)

HMS_VERSION = 1
PROTECTED_FFN = tuple(range(30, 42))
ATTENTION = ("qkv", "o")
FAMILIES = ("qkv", "o", "ffn_up_gate", "ffn_down")
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
SEED, SIZE = 1001, 1024
# The seven calibration prompts (prompts_40_variados.txt indices 3, 16, 23, 29, 37, 1, 21), in the order the lab's files
# sorted (cal_03, cal_16, cal_23, cal_29, cal_37, cal_street, cal_watercolor): the order of the GPTQ statistics.
PROMPTS = (
    (3, "Studio portrait of a young Senegalese man with a close-cropped beard wearing a charcoal turtleneck against a seamless deep "
        "olive backdrop. Single large octabox from camera left with a black flag on the right for strong falloff, crisp catchlights, "
        "visible pores and subtle skin sheen, 85mm lens at f/5.6, editorial menswear photography."),
    (16, "Photograph of an old violin maker's workbench crowded with chisels, wood shavings, clamps and an unfinished violin body without "
         "strings, a handwritten note pinned to the wall that reads \"varnish Thursday\". Warm tungsten desk lamp, shallow depth of field, "
         "rich textures of maple and spruce."),
    (23, "Detailed pen and ink illustration of a sprawling steampunk airship docked at a cliffside city, rope ladders, cargo nets, brass "
         "pipes and tiny workers on the deck. Fine cross-hatching for the shadows, a limited wash of sepia ink, the look of a Victorian "
         "engraving in an old adventure novel."),
    (29, "American superhero comic cover of an original heroine in a teal and copper suit landing in a crouch on a cracked street, debris "
         "flying, a city skyline burning orange behind her. Bold inks, dramatic foreshortening, halftone dots in the shadows, a masthead "
         "at the top reading \"THE COPPER TIDE #1\"."),
    (37, "Photograph of a handwritten letter on yellowed paper lying on a wooden desk beside a fountain pen and a pressed flower, the first "
         "line clearly readable: \"My dear Elena, the sea was calm today.\" Soft side light from a window, shallow depth of field, ink "
         "slightly feathered into the paper fibres."),
    (1, "Candid street photograph in Havana at late afternoon: a boy in a faded red t-shirt leaps over a puddle while two old men play "
        "dominoes on a folding table under a peeling turquoise doorway. Warm side light grazing the cracked plaster, deep shadows, slight "
        "motion blur on the boy, 28mm lens, Kodak Portra colour palette."),
    (21, "Loose watercolor painting of a narrow Venetian canal with laundry strung between balconies and a single gondola tied to a "
         "striped pole, reflections broken by ripples. Wet-on-wet sky, granulating pigments, white paper left untouched for the "
         "highlights, a few crisp pencil lines visible underneath."),
)


def _kitchen():
    """The comfy-kitchen eager helpers HMS uses (private functions; checked with comfy-kitchen 0.2.40)."""
    try:
        from comfy_kitchen.backends.eager import w4a8_int8 as core
        from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight as rotate
        from comfy_kitchen.tensor.w4a8_packed import _pack_codes
        for name in ("_grid_levels", "_assign_grid", "dequantize_w4a8_int8_weight"):
            getattr(core, name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError("HMS v1 needs comfy-kitchen's eager W4A8 / W6A8 functions (checked with comfy-kitchen 0.2.40); this "
                           f"ComfyUI's comfy-kitchen does not provide them: {e}") from e
    return core, rotate, _pack_codes


def target_format(block, family):
    return "w6a8_int8" if family in ATTENTION or block in PROTECTED_FFN else "asym_w4a8_int8"


def capture(model, clip, steps, cfg, sampler_name, scheduler, n_layers, progress=None):
    """Two runs of the seven prompts (seed 1001, 1024 x 1024): the first counts every projection's calls and collects ConvRot
    energies, the second keeps 48 image-token rows of the calls at 0, 16, 32 and 48 / 49 of each projection's run.
    Returns per scene {name: (energy, rows)}, the row counts, and {name: rows tensor}."""
    import nodes
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    _, rotate, _ = _kitchen()
    negative, = nodes.CLIPTextEncode().encode(clip, NEGATIVE)
    latent = EmptyFlux2LatentImage.execute(width=SIZE, height=SIZE, batch_size=1).result[0]
    layers = model.model.diffusion_model.layers
    names = [f"layers.{i}.proj.{f}" for i in range(n_layers) for f in FAMILIES]
    state = {"mode": None, "counts": {}, "store": {}, "indices": {}}

    def record(name, x):
        """x: the projection's input [batch, tokens, features] (after its input activation)."""
        i = state["counts"][name]
        state["counts"][name] += 1
        state["calls"] = state.get("calls", 0) + 1
        if state["calls"] % 500 == 0:
            log.debug("HMS capture: %d projection calls (%s pass), %.1f s", state["calls"], state["mode"], time.perf_counter() - state.get("t0", 0))
        positions = torch.linspace(0, SIZE * SIZE // 256 - 1, 48, device=x.device).long()
        if state["mode"] == "diag":
            rows = x[:, positions].reshape(-1, x.shape[-1]).float()
            state["device"] = x.device
            # to RAM at once: thousands of small tensors kept in VRAM slowed every later allocation under the core's dynamic
            # VRAM (the calls got slower and slower: minutes per sampling step after a few hundred calls)
            state["store"][name].append((rotate(rows.contiguous(), 256).square().sum(0).cpu(), len(rows)))
        elif i in state["indices"][name]:
            state["store"][name].append(x[:, positions].reshape(-1, x.shape[-1]).cpu().contiguous())

    # as the lab's capture: every projection forced through its module forward, so a pre-hook sees its input (ffn_down
    # included, which otherwise goes through the core's fused linear_input_act)
    handles, forced = [], []
    for i in range(n_layers):
        for f in FAMILIES:
            layer = layers[i].proj[f]
            forced.append((layer, getattr(layer, "comfy_force_forward", None)))
            layer.comfy_force_forward = True
            handles.append(layer.register_forward_pre_hook(lambda m, v, n=f"layers.{i}.proj.{f}": record(n, v[0])))
    diagonals, row_counts, rows = [], [], {n: [] for n in names}
    try:
        for n_prompt, (_, prompt) in enumerate(PROMPTS):
            t0 = time.perf_counter()
            positive, = nodes.CLIPTextEncode().encode(clip, prompt)
            sample = lambda: nodes.KSampler().sample(model, SEED, steps, cfg, sampler_name, scheduler, positive, negative, latent,  # noqa: E731
                                                     denoise=1.0)
            state.update(mode="diag", counts=dict.fromkeys(names, 0), store={n: [] for n in names}, t0=time.perf_counter())
            sample()
            log.info("HMS calibration %d/7: first pass done (%.1f s)", n_prompt + 1, time.perf_counter() - t0)
            energies, counts, totals = {}, {}, {}
            for n, calls in state["store"].items():
                total = len(calls)
                if total < 4:
                    raise RuntimeError(f"HMS calibration: {n} ran only {total} times; use at least 4 sampling steps")
                picked = [round(t * (total - 1) / 49) for t in (0, 16, 32, 48)]
                counts[n] = sum(calls[j][1] for j in picked)
                # summed on the GPU as the lab did (float sums depend on the device), back to RAM afterwards
                energy = (sum(calls[j][0].to(state["device"]) for j in picked) / counts[n]).cpu()
                if not energy.isfinite().all() or energy.max() <= 0:
                    raise RuntimeError(f"HMS calibration: invalid activation energy for {n}")
                energies[n], totals[n] = energy, total
                state["indices"][n] = set(picked)
            if progress:
                progress.update(1)
            state.update(mode="rows", counts=dict.fromkeys(names, 0), store={n: [] for n in names})
            sample()
            for n in names:
                if state["counts"][n] != totals[n] or len(state["store"][n]) != 4:
                    raise RuntimeError(f"HMS calibration: {n} ran differently in the second pass")
                rows[n].append(torch.cat(state["store"][n]))
            log.info("HMS calibration %d/7: second pass done (%.1f s)", n_prompt + 1, time.perf_counter() - t0)
            diagonals.append(energies)
            row_counts.append(counts)
            if progress:
                progress.update(1)
    finally:
        for h in handles:
            h.remove()
        for layer, old in forced:
            if old is None:
                del layer.comfy_force_forward
            else:
                layer.comfy_force_forward = old
    return diagonals, row_counts, {n: torch.cat(v) for n, v in rows.items()}


def _fit_table(weight, diagonal, device):
    """Seeded sampling over individual weights, normalized with their 16-column groups (C1 codebook)."""
    rows, cols = weight.shape
    group = weight.view(rows, -1, 16)
    rng = torch.Generator(device=device).manual_seed(0)
    idx = torch.randint(rows * cols, (300000,), generator=rng, device=device)
    x = weight.flatten()[idx] / group.abs().amax(-1).flatten()[idx // 16].clamp_min(1e-8)
    d = diagonal[idx % cols]
    table = torch.quantile(x, torch.linspace(0, 1, 16, device=device))
    for _ in range(25):
        code = torch.bucketize(x, (table[:-1] + table[1:]) / 2)
        for j in range(16):
            mass = torch.where(code == j, d, 0.)
            total = mass.sum()
            table[j] = torch.where(total > 0, (x * mass).sum() / total, table[j])
    return table


def calibrate_c1(weight, baseline_scale, channel, diagonal, bits, device):
    """Activation-weighted scale search and (4-bit) codebook; returns the C1 scales and codebook."""
    core, rotate, _pack_codes = _kitchen()
    rows, cols = weight.shape
    rotated = rotate(weight.float().to(device), 256)
    d = diagonal.to(device).view(1, -1, 16)
    table = _fit_table(rotated, d.flatten(), device) if bits == 4 else torch.arange(-31, 32, device=device).float()
    scale_parts = []
    for start in range(0, rows, 128):
        group = rotated[start:start + 128].view(-1, cols // 16, 16)
        ch = channel[start:start + 128].to(device)
        base = baseline_scale[start:start + 128].to(device)
        raw = base.view(torch.uint8).int()
        target = group / ch.view(-1, 1, 1)
        maximum = target.abs().amax(-1) / (31 if bits == 6 else 1)
        candidates = [base, (raw - 1).clamp(1, 126).byte().view(base.dtype), (raw + 1).clamp(1, 126).byte().view(base.dtype)]
        candidates.extend((maximum * factor).clamp(0, 448).to(base.dtype) for factor in (0.8, 0.85, 0.9, 0.95, 1.0))
        best_err = torch.full_like(maximum, float("inf"))
        best_scale = torch.zeros_like(maximum)
        best_idx = torch.zeros_like(group, dtype=torch.int32)
        for scale in candidates:
            levels = core._grid_levels(table, scale)
            idx = core._assign_grid(group, levels, ch, target=target)
            err = ((levels.gather(2, idx.long()) - target).square() * d).sum(-1)
            better = err < best_err
            best_idx = torch.where(better.unsqueeze(-1), idx, best_idx)
            best_scale = torch.where(better, scale.float(), best_scale)
            best_err = torch.minimum(best_err, err)
        scale = best_scale.to(base.dtype)
        packed = _pack_codes((best_idx + int(bits == 6)).view(-1, cols), bits)
        expected = rotate((core._grid_levels(table, scale).gather(2, best_idx.long()) * ch.view(-1, 1, 1)).reshape(-1, cols).bfloat16(), 256)
        if not torch.equal(core.dequantize_w4a8_int8_weight(packed, scale, ch, table if bits == 4 else None), expected):
            raise RuntimeError("HMS C1: the core's dequant differs from the reconstruction")
        scale_parts.append(scale.cpu())
    return torch.cat(scale_parts), (table.cpu() if bits == 4 else None)


def gptq(weight, rows, scale, ch, lut, device):
    """GPTQ in the ConvRot basis with the C1 grids fixed (group 16, GPTQ block 128); returns the packed codes, verified
    against the core's dequant."""
    core, rotate, _pack_codes = _kitchen()
    scale, ch = scale.to(device), ch.to(device)
    lut = lut.to(device) if lut is not None else None
    w = rotate(weight.float().to(device), 256)
    x = rotate(rows.float().to(device), 256)
    h = x.T @ x / len(rows)
    h.diagonal().add_(h.diagonal().mean() * 0.01)
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h, x
    n, k = w.shape
    bits = 6 if lut is None else 4
    table = torch.arange(-31, 32, device=device).float() if lut is None else lut
    w.div_(ch[:, None])  # channel-scale units: scaling a row commutes with the GPTQ updates
    codes = torch.empty((n, k), dtype=torch.uint8, device=device)
    recon = torch.empty_like(w)
    for start in range(0, k, 128):
        stop = min(k, start + 128)
        local = w[:, start:stop].contiguous()
        errors = torch.empty_like(local)
        for i in range(stop - start):
            j = start + i
            if j % 16 == 0:
                levels = core._grid_levels(table, scale[:, j // 16:j // 16 + 1]).squeeze(1).contiguous()
            col = local[:, i:i + 1].contiguous()
            pos = torch.searchsorted(levels, col)
            lo = (pos - 1).clamp_(0, len(table) - 1)
            hi = pos.clamp_(0, len(table) - 1)
            a, b = levels.gather(1, lo), levels.gather(1, hi)
            pick = (col - b).abs() < (col - a).abs()
            q = torch.where(pick, b, a)
            codes[:, j:j + 1] = torch.where(pick, hi, lo) + int(bits == 6)
            recon[:, j:j + 1] = q
            error = (col - q) / upper[j, j]
            errors[:, i:i + 1] = error
            local[:, i:].sub_(error * upper[j, j:stop])
        if stop < k:
            w[:, stop:].addmm_(errors, upper[start:stop, stop:], alpha=-1)
    packed = _pack_codes(codes.int(), bits)
    for i in range(0, n, 128):
        s = slice(i, i + 128)
        expected = rotate((recon[s] * ch[s, None]).bfloat16(), 256)
        if not torch.equal(expected, core.dequantize_w4a8_int8_weight(packed[s], scale[s], ch[s], lut)):
            raise RuntimeError("HMS GPTQ: reconstruction mismatch")
    return packed.cpu()


def quantize_projection(name, weight, diagonals, row_counts, rows, device):
    """One projection: the native template (stock quantizer), C1 on the weighted energies, then GPTQ. Returns the tensors of
    its stock layout (weight, weight_s_rel, weight_s_channel, [weight_codebook]) and comfy_quant marker."""
    import comfy.quant_ops
    block, family = int(name.split(".")[1]), name.split(".")[3]
    fmt = target_format(block, family)
    bits = 6 if fmt == "w6a8_int8" else 4
    w = weight.to(device, torch.bfloat16)
    opts = {"bits": 6, "codebook": False} if bits == 6 else {}
    template = comfy.quant_ops.QuantizedTensor.from_float(w, "AsymW4A8Int8Layout", **opts).state_dict(name + ".weight")
    channel = template[name + ".weight_s_channel"]
    diagonal = sum(d[name] * c[name] for d, c in zip(diagonals, row_counts)) / sum(c[name] for c in row_counts)
    scale, table = calibrate_c1(w, template[name + ".weight_s_rel"], channel, diagonal, bits, device)
    out = {name + ".weight": gptq(w, rows, scale, channel, table, device), name + ".weight_s_rel": scale,
           name + ".weight_s_channel": channel.cpu()}
    if table is not None:
        out[name + ".weight_codebook"] = table
    for k, v in out.items():
        ref = template[k]
        if tuple(v.shape) != tuple(ref.shape) or v.dtype != ref.dtype:
            raise RuntimeError(f"HMS: {k} does not match the native layout")
    out[name + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": fmt}, separators=(",", ":")).encode()), dtype=torch.uint8)
    return out


def recipe_metadata(steps, cfg, sampler_name, scheduler):
    return json.dumps(dict(method="HMS", version=HMS_VERSION, attention="w6a8_int8", protected_ffn_blocks=list(PROTECTED_FFN),
                           other_ffn="asym_w4a8_int8", calibration="C1 activation-weighted scale + codebook, GPTQ in the ConvRot basis",
                           prompt_file="prompts_40_variados.txt", prompt_indices_zero_based=[i for i, _ in PROMPTS],
                           calibration_sampling=dict(seed=SEED, size=SIZE, steps=steps, cfg=cfg, sampler=sampler_name, scheduler=scheduler)),
                      separators=(",", ":"))
