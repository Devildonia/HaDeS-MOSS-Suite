import json

from .moss_common import parse_info_meta

N_INFO = 8
NODE_NAMES = {
    "model_block_merger": "Model Block Merger", "multi_model_merger": "Multi-Model Merger", "model_saver": "Model Saver",
    "lora_extractor": "LoRA Extractor", "lora_block_weight": "LoRA Block Weight", "lora_merger": "LoRA Merger", "lora_resize": "LoRA Resize",
    "latent": "Latent", "layout": "Layout Compiler", "ksampler": "KSampler",
}


def diagnose(metas):
    """[(level, message)] from the HADES_META trailers; every rule cites a measurement in tools/RESULTS.md."""
    out = []
    by_node = {}
    for m in metas:
        by_node.setdefault(m["node"], []).append(m)

    sizes = {NODE_NAMES[k]: (m["width"], m["height"]) for k in ("layout", "latent", "ksampler") for m in by_node.get(k, [])[:1]}
    if len(set(sizes.values())) > 1:
        out.append(("error", "Canvas mismatch: " + ", ".join(f"{n} {w}x{h}" for n, (w, h) in sizes.items())
                    + ". A layout prompt places its boxes on its own canvas; generate at that size (use the Layout Compiler's latent)."))

    for m in by_node.get("ksampler", []):
        w, h = m["width"], m["height"]
        if not (768 * 768 <= w * h <= 2048 * 2048):
            out.append(("note", f"KSampler at {w}x{h}: outside the measured 768x768 to 2048x2048 range."))
        if abs(m["flow_shift"] - 5.0) > 1e-3:
            out.append(("note", f"KSampler shift {m['flow_shift']:g}: 5.0 measured clean from 768 to 2048 px; other shifts changed images less than a new seed."))
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
            if m["energy_kept"] < 0.9:
                out.append(("warning", f"{NODE_NAMES[kind]}: {m['energy_kept']:.1%} energy kept; below about 90% the output drifted visibly "
                                       "from the source in tests. Raise the rank."))

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
    DESCRIPTION = "Reads the info outputs of HaDeS MOSS nodes and checks the pipeline against what was measured (canvas sizes, shift, SLG, quantized merges, LoRA energy, layout line breaks)."

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
        lines += ["", "## Settings", ""] + [f"- {NODE_NAMES[m['node']]}: " + ", ".join(f"{k}={v}" for k, v in m.items() if k != "node") for m in metas]
        summary = (f"{len(metas)} nodes | {counts['error']} errors, {counts['warning']} warnings, {counts['note']} notes"
                   + (f" | {findings[0][1][:80]}" if findings else ""))
        data = json.dumps({"nodes": metas, "findings": [{"level": l, "message": m} for l, m in findings]}, ensure_ascii=False, indent=1)
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report, data, summary)}


NODE_CLASS_MAPPINGS = {"HaDeSMossSuiteTelemetry": HaDeSMossSuiteTelemetry}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossSuiteTelemetry": "MOSS Suite Telemetry"}
