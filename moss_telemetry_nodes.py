import json

from .moss_common import parse_info_meta

N_INFO = 8
NODE_NAMES = {
    "model_block_merger": "Model Block Merger", "multi_model_merger": "Multi-Model Merger", "model_saver": "Model Saver",
    "lora_extractor": "LoRA Extractor", "lora_block_weight": "LoRA Block Weight", "lora_merger": "LoRA Merger", "lora_resize": "LoRA Resize",
    "latent": "Latent", "layout": "Layout Compiler", "ksampler": "KSampler", "prompt_enhancer": "Prompt Enhancer",
    "multi_lora": "Multi-LoRA", "layout_editor": "Layout Editor", "loader": "Loader", "metadata_reader": "Metadata Reader", "control": "Control",
}


def diagnose(metas):
    """[(level, message)] from the HADES_META trailers; every rule cites a measurement in tools/RESULTS.md."""
    out = []
    by_node = {}
    for m in metas:
        by_node.setdefault(m["node"], []).append(m)

    # a layout prompt places its boxes on its own canvas: with a layout in the pipeline, every canvas must match it. Without one,
    # a latent and a KSampler of different sizes are normal (img2img, an upscale pass) and are not compared.
    layouts = [(NODE_NAMES[k], m) for k in ("layout_editor", "layout") for m in by_node.get(k, [])[:1]]
    layouts += [("Prompt Enhancer (layout)", m) for m in by_node.get("prompt_enhancer", []) if m.get("mode") == "layout"][:1]
    if layouts:
        canvases = layouts + [(NODE_NAMES[k], m) for k in ("latent", "ksampler") for m in by_node.get(k, [])[:1]]
        sizes = {n: (m["width"], m["height"]) for n, m in canvases}
        if len(set(sizes.values())) > 1:
            out.append(("error", "Canvas mismatch: " + ", ".join(f"{n} {w}x{h}" for n, (w, h) in sizes.items())
                        + ". A layout prompt places its boxes on its own canvas; generate at that size (use the Layout Compiler's latent)."))

    for m in by_node.get("ksampler", []):
        w, h = m["width"], m["height"]
        if not (768 * 768 <= w * h <= 2048 * 2048):
            out.append(("note", f"KSampler at {w}x{h}: outside the measured 768x768 to 2048x2048 range."))
        if abs(m["flow_shift"] - 5.0) > 1e-3:
            out.append(("note", f"KSampler shift {m['flow_shift']:g}: 5.0 measured clean from 768 to 2048 px; other shifts changed images less than a new seed."))
        if m.get("turbo") and abs(m.get("turbo_strength", 1.0) - 1.0) > 1e-6:
            out.append(("note", f"Turbo LoRA at strength {m['turbo_strength']:g}: the turbo LoRAs were measured at 1.0 only."))
        if m.get("turbo") and m.get("steps", 4) not in (4, 8):
            out.append(("note", f"Turbo LoRA at {m['steps']} steps: the turbo LoRAs were measured at 4 steps (Turbo, Turbo Diverse) and 8 "
                                "(Turbo 8-steps) only."))
        if m.get("turbo") and m.get("cfg", 1.0) > 1.0 + 1e-6:
            out.append(("note", f"Turbo LoRA at CFG {m['cfg']:g}: trained at CFG 1; above it the negative prompt takes part and each step runs "
                                "twice (about 1.9x the time). Measured with one seed per prompt: 1.25-1.5 brought back prompt details, saturation "
                                "rose with CFG, 2.5 looked overdone, texture-heavy prompts looked better at 1.0-1.2 (tools/RESULTS.md)."))
        if m.get("turbo") and (m.get("sampler"), m.get("scheduler")) not in (("euler", "simple"), ("er_sde", "simple")):
            out.append(("note", f"Turbo LoRA with {m.get('sampler')} / {m.get('scheduler')}: measured with euler / simple (training) and "
                                "er_sde / simple (the CFG runs) only."))
        if m.get("turbo") and m.get("denoise", 1.0) < 1.0:
            out.append(("note", f"Turbo LoRA at denoise {m['denoise']:.2f}: measured at denoise 1.0, and Turbo 8-steps at 0.25 / 0.40 "
                                "on a 2x-upscaled image only."))
        if m.get("slg", "off") != "off":
            out.append(("note", f"KSampler SLG '{m['slg']}': measured, it did not change text accuracy on SFT or Base."))

    for m in by_node.get("model_block_merger", []) + by_node.get("multi_model_merger", []):
        if m.get("quantized"):
            out.append(("warning", f"{NODE_NAMES[m['node']]}: {', '.join(m['quantized'])} quantized; their int8 / fp8 rounding is part of the merge."))
            if m["node"] == "multi_model_merger" and m.get("method") in ("TIES", "DARE-TIES", "DARE"):
                out.append(("warning", f"Multi-Model Merger {m['method']} with a quantized model: its task vector carries int8 rounding, "
                                       "about as large as the change of the int8 finetune measured."))
        rescale = m.get("model2_residual_rescale") or m.get("rescaled")
        if rescale:
            out.append(("note", f"{NODE_NAMES[m['node']]}: fp16-fixed residual scale normalized ({rescale})."))

    for kind in ("lora_extractor", "lora_resize", "lora_merger"):
        for m in by_node.get(kind, []):
            if m["energy_kept"] < 0.979:  # the lowest energy measured within the noise floor
                out.append(("warning", f"{NODE_NAMES[kind]}: {m['energy_kept']:.1%} energy kept. Measured: 97.9-100% stayed within the "
                                       "noise floor of the source, 90.4-91.1% drifted visibly. Raise the rank to stay close."))

    for m in by_node.get("loader", []):
        if m["precision"].startswith("int8"):
            out.append(("note", f"Loader: {m['model']} is int8. Measured on an RTX 5090 with the whole model in VRAM: 57% slower than bf16 at "
                                "1024 x 1024 and 49% at 2048 (SageAttention, PyTorch 2.8.0+cu129), for about half the VRAM (5.2 GB against 9.7 GB)."))
        if m.get("weight_dtype", "default") != "default":
            out.append(("note", f"Loader: the model was loaded as {m['weight_dtype']}; that conversion was not measured."))

    # few-step LoRAs in a Multi-LoRA against the KSampler: the settings come from each file's own metadata
    for m in by_node.get("multi_lora", []):
        for t in m.get("turbo", []):
            for k in by_node.get("ksampler", []):
                if k.get("turbo"):
                    out.append(("warning", f"Multi-LoRA holds the few-step LoRA {t['name']} and the KSampler's turbo input adds {k['turbo']}: "
                                           "two few-step LoRAs, never measured together."))
                else:
                    if k["steps"] != t["steps"]:
                        out.append(("warning", f"Multi-LoRA: {t['name']} was trained for {t['steps']} steps (file metadata); "
                                               f"the KSampler runs {k['steps']}."))
                    if k["cfg"] > 1.0 + 1e-6:
                        out.append(("note", f"Multi-LoRA: {t['name']} was trained at CFG 1; the KSampler runs CFG {k['cfg']:g}. Measured with one "
                                            "seed per prompt: 1.25-1.5 brought back prompt details, 2.5 looked overdone (tools/RESULTS.md)."))
                    elif k["cfg"] < 1.0 - 1e-6:
                        out.append(("warning", f"Multi-LoRA: {t['name']} was trained at CFG 1; the KSampler runs CFG {k['cfg']:g}."))

    for m in by_node.get("control", []):
        if abs(m.get("strength", 1.0) - 1.0) > 1e-6:
            out.append(("note", f"Control at strength {m['strength']:g}: 1.0 is the adapter's trained strength; other values are not measured."))
        if m.get("color_mode") == "as given":
            out.append(("note", "Control color_mode 'as given': the adapter reads the color map as flat region colors; a photo there "
                                "makes it copy the photo's texture (use flat regions for a photo)."))

    for m in by_node.get("layout", []):
        for lines, w, h in m.get("multiline", []):
            out.append(("note", f"Layout text on {lines} lines in a {w:g}x{h:g} box ({w / h:.1f}:1). Measured with two lines: a 4.3:1 box "
                                "kept them in 3 of 6 seeds, a 1.7:1 box in 6 of 6."))
    return out


class HaDeSMossSuiteTelemetry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"report_title": ("STRING", {"default": "HaDeS MOSS Suite Telemetry"})},
            "optional": {f"info_{i}": ("STRING", {"forceInput": True}) for i in range(1, N_INFO + 1)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("report", "json", "summary")
    FUNCTION = "audit"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = "Reads the info outputs of HaDeS MOSS nodes and checks the pipeline against what was measured (canvas sizes, shift, SLG, quantized merges, LoRA energy, few-step LoRA settings, int8 models, layout line breaks)."

    def audit(self, report_title, **infos):
        metas, foreign = [], 0
        for i in range(1, N_INFO + 1):
            text = infos.get(f"info_{i}")
            if not text:
                continue
            meta = parse_info_meta(text)
            if meta is not None and meta.get("node") in NODE_NAMES:
                metas.append(meta)
            else:
                foreign += 1
        findings = diagnose(metas)
        counts = {lvl: sum(1 for l, _ in findings if l == lvl) for lvl in ("error", "warning", "note")}

        lines = [f"# {report_title}", "", f"Nodes read: {', '.join(NODE_NAMES[m['node']] for m in metas) or 'none'}"
                 + (f" ({foreign} input(s) without a HaDeS MOSS trailer ignored)" if foreign else ""), ""]
        if findings:
            lines += ["## Findings", ""] + [f"- **{lvl.upper()}** {msg}" for lvl, msg in findings]
        else:
            lines.append("No findings: the connected nodes are consistent with what was measured.")
        # the prompt texts (Prompt Enhancer) and file paths (Loader) ride in the json output for the MOSS Image Saver; the report stays short
        hidden = {"node", "prompt", "negative", "model_file", "text_encoder_file"}
        lines += ["", "## Settings", ""] + [f"- {NODE_NAMES[m['node']]}: " + ", ".join(f"{k}={v}" for k, v in m.items() if k not in hidden) for m in metas]
        summary = (f"{len(metas)} nodes | {counts['error']} errors, {counts['warning']} warnings, {counts['note']} notes"
                   + (f" | {findings[0][1][:80]}" if findings else ""))
        data = json.dumps({"nodes": metas, "findings": [{"level": l, "message": m} for l, m in findings]}, ensure_ascii=False, indent=1)
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report, data, summary)}


NODE_CLASS_MAPPINGS = {"HaDeSMossSuiteTelemetry": HaDeSMossSuiteTelemetry}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossSuiteTelemetry": "MOSS Suite Telemetry"}
