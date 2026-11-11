"""Development checks and image measurements; no model execution or file writes."""
import json
import math
import re

import torch
from comfy_api.latest import io, ui

from .moss_common import format_info
from .moss_common import parse_info_meta
from .moss_image_saver_nodes import read_metas
from .moss_pipeline_nodes import EXAMPLE_DRAFT, compile_layout


def select_regions(selection, count):
    indices = set()
    for token in selection.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", token)
        if match is None:
            raise ValueError("Layout Regions: selection must contain box numbers or ranges, e.g. 1,3-5 (numbering starts at 1).")
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= count:
            raise ValueError(f"Layout Regions: selection {token.strip()!r} is outside boxes 1–{count}, or is reversed.")
        indices.update(range(first, last + 1))
    return sorted(indices)


class HaDeSMossLayoutRegions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HaDeSMossLayoutRegions", display_name="MOSS Layout Regions", category="HaDeS/MOSS Dev Tools", is_output_node=True,
            description="Makes a binary mask from selected Layout Editor boxes. Boxes are regions, not object segmentation. No sampling or file writes.",
            inputs=[
                io.String.Input("draft", default=EXAMPLE_DRAFT, multiline=True, tooltip="Layout Editor / Prompt Enhancer draft, in top-left pixel coordinates."),
                io.String.Input("selection", default="1", tooltip="1-based box numbers in draft order: 1,3-5. Duplicates are counted once. Check selection after reordering boxes."),
                io.Int.Input("padding", default=0, min=0, max=16384, tooltip="Pixels added on every side of each selected box, clipped to the canvas."),
            ],
            outputs=[io.Mask.Output(display_name="mask"), io.Image.Output(display_name="preview"),
                     io.String.Output(display_name="regions_json"), io.String.Output(display_name="info")],
        )

    @classmethod
    def execute(cls, draft, selection="1", padding=0):
        data = json.loads(draft)
        if not isinstance(data, dict) or "items" not in data:
            raise ValueError("Layout Regions: use a layout draft with boxes, not a caption draft.")
        _, width, height, _ = compile_layout(data)
        if type(padding) is not int or padding < 0:
            raise ValueError("Layout Regions: padding must be a non-negative integer.")
        indices = select_regions(selection, len(data["items"]))
        mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
        regions = []
        for index in indices:
            item = data["items"][index - 1]
            x0, y0, x1, y1 = [round(v, 9) for v in item["bbox"]]
            box = [max(0, math.floor(x0) - padding), max(0, math.floor(y0) - padding),
                   min(width, math.ceil(x1) + padding), min(height, math.ceil(y1) + padding)]
            left, top, right, bottom = box
            mask[:, top:bottom, left:right] = 1
            regions.append({"index": index, "label": item["label"], "caption": item["caption"],
                            "source_bbox_xyxy": item["bbox"], "mask_bbox_xyxy": box})
        pixels = int(mask.sum().item())
        record = {"schema_version": 1, "width": width, "height": height, "padding": padding,
                  "coordinate_space": "top_left_pixels", "edge_rule": "floor_start_ceil_end_exclusive",
                  "regions": regions, "union_pixels": pixels, "union_percent": 100 * pixels / (width * height)}
        text = f"MOSS Layout Regions | {width}×{height} | boxes {', '.join(map(str, indices))} | padding {padding}px"
        text += f"\nUnion: {pixels} pixels ({record['union_percent']:.3f}%). White = inside; boxes are not segmentation."
        text += "".join(f"\n{r['index']}: {r['label']} — {r['caption']} | {r['mask_bbox_xyxy']}" for r in regions)
        info = format_info(text, "layout_regions", **record)
        preview = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        return io.NodeOutput(mask, preview, json.dumps(record, ensure_ascii=False, indent=2), info, ui=ui.PreviewText(text))


def image_stats(reference, candidate, delta, changed, region):
    pixels = int(region.sum().item())
    if pixels == 0:
        return None
    a, b, difference = reference[region], candidate[region], delta[region]
    # Luma on encoded RGB, not linear-light luminance. HSV saturation is defined per pixel.
    luma = (b - a) @ torch.tensor([0.2126, 0.7152, 0.0722])
    def saturation(rgb):
        maximum, minimum = rgb.amax(dim=-1), rgb.amin(dim=-1)
        return (maximum - minimum) / maximum.clamp_min(torch.finfo(rgb.dtype).tiny)
    return {"pixels": pixels, "mae_255": float(difference.double().mean() * 255),
            "max_abs_255": float(difference.max() * 255),
            "changed_percent": 100 * int(changed[region].sum().item()) / pixels,
            "luma_delta_255": float(luma.double().mean() * 255),
            "hsv_saturation_delta": float((saturation(b) - saturation(a)).double().mean())}


class HaDeSMossImageAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HaDeSMossImageAudit", display_name="MOSS Image Audit", category="HaDeS/MOSS Dev Tools", is_output_node=True,
            description="Measures paired RGB images globally and inside/outside a mask. No resizing, alignment, quality score or automatic edit verdict. CPU analysis; no model loading.",
            inputs=[
                io.Image.Input("reference"), io.Image.Input("candidate"),
                io.Float.Input("change_threshold", default=2.0, min=0.0, max=255.0, step=0.1,
                               tooltip="A pixel changed if max absolute RGB difference × 255 is strictly greater than this value."),
                io.Float.Input("difference_gain", default=4.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Display only: clamp(abs(candidate-reference) × gain, 0, 1). Metrics are unaffected."),
                io.String.Input("reference_label", default="reference"), io.String.Input("candidate_label", default="candidate"),
                io.Mask.Input("region_mask", optional=True, tooltip="Values >= 0.5 are inside. One mask may be shared across all pairs; otherwise one mask per pair. Exact dimensions required."),
            ],
            outputs=[io.Image.Output(display_name="difference_map"), io.String.Output(display_name="report"),
                     io.String.Output(display_name="metrics_json")],
        )

    @classmethod
    def execute(cls, reference, candidate, change_threshold=2.0, difference_gain=4.0,
                reference_label="reference", candidate_label="candidate", region_mask=None):
        if reference.ndim != 4 or reference.shape[-1] != 3 or any(n == 0 for n in reference.shape):
            raise ValueError("Image Audit: expected non-empty RGB images [batch, height, width, 3]. Handle alpha explicitly before comparing.")
        if candidate.shape != reference.shape:
            raise ValueError("Image Audit: reference and candidate must have identical RGB dimensions and batch counts. Resize / pair them explicitly first.")
        if not math.isfinite(change_threshold) or not 0 <= change_threshold <= 255:
            raise ValueError("Image Audit: change_threshold must be finite and between 0 and 255.")
        if not math.isfinite(difference_gain) or difference_gain < 0:
            raise ValueError("Image Audit: difference_gain must be finite and non-negative.")
        batch, height, width, _ = reference.shape
        if region_mask is not None:
            if region_mask.ndim != 3 or tuple(region_mask.shape[1:]) != (height, width) or region_mask.shape[0] not in (1, batch):
                raise ValueError("Image Audit: mask must be [1 or batch, height, width], at the exact image size.")
        maps = torch.empty((batch, height, width, 3), dtype=torch.float32, device="cpu")
        rows = []
        for index in range(batch):
            a, b = reference[index].detach().to(device="cpu", dtype=torch.float32), candidate[index].detach().to(device="cpu", dtype=torch.float32)
            for name, image in (("reference", a), ("candidate", b)):
                if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(f"Image Audit: {name} pair {index + 1} must contain finite RGB values in [0,1].")
            delta = (b - a).abs()
            changed = delta.amax(dim=-1) * 255 > change_threshold
            maps[index] = (delta * difference_gain).clamp(0, 1)
            all_pixels = torch.ones((height, width), dtype=torch.bool)
            row = {"pair": index + 1, "global": image_stats(a, b, delta, changed, all_pixels)}
            if region_mask is not None:
                mask = region_mask[0 if region_mask.shape[0] == 1 else index].detach().to(device="cpu", dtype=torch.float32)
                if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
                    raise ValueError("Image Audit: mask values must be finite and in [0,1].")
                inside = mask >= 0.5
                row.update(inside=image_stats(a, b, delta, changed, inside), outside=image_stats(a, b, delta, changed, ~inside))
            rows.append(row)
        record = {"schema_version": 1, "reference_label": reference_label, "candidate_label": candidate_label,
                  "width": width, "height": height, "batch_pairs": batch, "change_threshold_255": change_threshold,
                  "change_rule": "max_abs_rgb_255 > threshold", "mask_rule": ">= 0.5 inside; single mask shared or paired masks",
                  "mask_present": region_mask is not None, "difference_gain": difference_gain,
                  "luma_definition": "encoded RGB dot [0.2126,0.7152,0.0722], not linear-light luminance",
                  "saturation_definition": "HSV (max-min)/max; black=0; candidate minus reference, scale 0..1",
                  "results": rows, "note": "Pixel differences and color shifts are not edit success, identity or quality scores. No resizing or alignment."}
        lines = [f"MOSS Image Audit | {reference_label} → {candidate_label}", f"{width}×{height} | {batch} pair(s) | changed threshold > {change_threshold:g}/255",
                 "Differences are not quality or edit-success scores. Positive color deltas mean candidate increased."]
        for row in rows:
            for region in ("global", "inside", "outside"):
                if region not in row:
                    continue
                s = row[region]
                prefix = f"Pair {row['pair']} / {region}: "
                lines.append(prefix + ("no pixels (null)" if s is None else
                    f"{s['pixels']} px | MAE {s['mae_255']:.6f} | max {s['max_abs_255']:.6f} | changed {s['changed_percent']:.3f}% | "
                    f"luma Δ {s['luma_delta_255']:+.6f}/255 | HSV saturation Δ {s['hsv_saturation_delta']:+.6f}"))
        text = "\n".join(lines)
        report = format_info(text, "image_audit", width=width, height=height, batch_pairs=batch, mask_present=region_mask is not None,
                             reference_label=reference_label, candidate_label=candidate_label)
        return io.NodeOutput(maps, report, json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), ui=ui.PreviewText(text))


_PAIR_FIELDS = ("seed", "steps", "cfg", "sampler", "scheduler", "flow_shift", "denoise", "width", "height", "slg")
_PROMPT_FIELDS = ("prompt", "negative")


def _single_stage(text, side):
    try:
        metas = read_metas(text)
    except ValueError as exc:
        raise ValueError(f"Pair Guard: {side} metadata must be Suite Telemetry JSON or one MOSS info string.") from exc
    if not all(isinstance(meta, dict) and isinstance(meta.get("node"), str) for meta in metas):
        raise ValueError(f"Pair Guard: {side} metadata has an invalid Suite trailer.")
    ks = [m for m in metas if m.get("node") == "ksampler"]
    if len(ks) != 1:
        raise ValueError(f"Pair Guard: {side} metadata needs exactly one MOSS KSampler; found {len(ks)}. Connect a stage-specific Telemetry.")
    enh = [m for m in metas if m.get("node") == "prompt_enhancer"]
    if len(enh) > 1:
        raise ValueError(f"Pair Guard: {side} metadata has multiple Prompt Enhancers; use a stage-specific Telemetry.")
    return metas, ks[0], enh[0] if enh else None


class HaDeSMossPairGuard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HaDeSMossPairGuard", display_name="MOSS Pair Guard", category="HaDeS/MOSS Dev Tools", is_output_node=True,
            description="Checks that two images are a valid matched pair. Optional stage metadata checks sampling and prompt controls; model/LoRA differences are reported but allowed. No pixels are altered.",
            inputs=[
                io.Image.Input("reference"), io.Image.Input("candidate"),
                io.String.Input("reference_metadata", optional=True, tooltip="Suite Telemetry json for the reference stage, or one KSampler info string."),
                io.String.Input("candidate_metadata", optional=True, tooltip="Suite Telemetry json for the candidate stage, or one KSampler info string."),
                io.Boolean.Input("require_metadata", default=False, tooltip="Fail if both stage metadata inputs are absent. A single connected side always fails."),
                io.Boolean.Input("compare_prompt", default=True, tooltip="Compare prompt and negative text when both stage Telemetry inputs include a Prompt Enhancer. Missing prompt data is reported as unverified."),
            ],
            outputs=[io.Image.Output(display_name="reference"), io.Image.Output(display_name="candidate"),
                     io.String.Output(display_name="report"), io.String.Output(display_name="checks_json")],
        )

    @classmethod
    def execute(cls, reference, candidate, reference_metadata=None, candidate_metadata=None, require_metadata=False, compare_prompt=True):
        if reference.ndim != 4 or candidate.ndim != 4 or reference.shape[-1] != 3 or candidate.shape[-1] != 3:
            raise ValueError("Pair Guard: both inputs must be RGB IMAGE batches [batch, height, width, 3].")
        if any(n <= 0 for n in reference.shape) or any(n <= 0 for n in candidate.shape):
            raise ValueError("Pair Guard: empty image dimensions or batches cannot be paired.")
        if reference.shape != candidate.shape:
            raise ValueError(f"Pair Guard: image shape mismatch: reference {tuple(reference.shape)} vs candidate {tuple(candidate.shape)}. Pair or resize explicitly.")
        if bool(reference_metadata) != bool(candidate_metadata):
            raise ValueError("Pair Guard: connect metadata for both stages or neither; one-sided controls cannot be verified.")
        if require_metadata and not reference_metadata:
            raise ValueError("Pair Guard: require_metadata is on but stage metadata is absent.")
        batch, height, width, _ = reference.shape
        mismatches, unverified, differences = [], [], []
        checked = {}
        if reference_metadata:
            a_metas, a_ks, a_enh = _single_stage(reference_metadata, "reference")
            b_metas, b_ks, b_enh = _single_stage(candidate_metadata, "candidate")
            for side, ks in (("reference", a_ks), ("candidate", b_ks)):
                for field, image_value in (("width", width), ("height", height)):
                    if field in ks and ks[field] != image_value:
                        mismatches.append(f"{side} metadata {field}={ks[field]!r}, image {field}={image_value}")
            for field in _PAIR_FIELDS:
                a, b = a_ks.get(field), b_ks.get(field)
                if a is None and b is None:
                    unverified.append(f"sampling {field}")
                elif a is None or b is None or a != b:
                    mismatches.append(f"{field}: reference={a!r}, candidate={b!r}")
                else:
                    checked[field] = a
            if compare_prompt:
                if a_enh is None or b_enh is None:
                    unverified.append("prompt (both stages need Prompt Enhancer info)")
                else:
                    for field in _PROMPT_FIELDS:
                        a, b = a_enh.get(field), b_enh.get(field)
                        if a is None and b is None:
                            unverified.append(field)
                        elif a is None or b is None or a != b:
                            mismatches.append(f"{field}: reference and candidate differ or one is missing")
                        else:
                            checked[field] = a
            for kind, field in (("loader", "model_file"), ("multi_lora", "loras")):
                a = [m.get(field) for m in a_metas if m.get("node") == kind]
                b = [m.get(field) for m in b_metas if m.get("node") == kind]
                if a != b:
                    differences.append(f"{kind}: reference={a!r}, candidate={b!r}")
        else:
            unverified.append("sampling settings and prompt (metadata not connected)")
        if mismatches:
            raise ValueError("Pair Guard: controls do not match: " + "; ".join(mismatches))
        record = {"schema_version": 1, "status": "pass", "width": width, "height": height, "batch_pairs": batch,
                  "metadata_checked": bool(reference_metadata), "compare_prompt": compare_prompt,
                  "matched_controls": checked, "unverified": unverified, "allowed_differences": differences,
                  "note": "This guard checks pairing and declared controls, not image quality, hidden graph state, or whether metadata describes the connected image."}
        lines = [f"MOSS Pair Guard | PASS | {width}×{height} | {batch} pair(s)",
                 f"Stage metadata: {'checked' if reference_metadata else 'absent; geometry only'}",
                 f"Matched controls: {', '.join(checked) if checked else 'none'}"]
        lines += ["Unverified: " + ", ".join(unverified)] if unverified else []
        lines += ["Allowed difference: " + item for item in differences]
        lines.append("No pixel alignment or quality judgment. Images pass through unchanged.")
        report = format_info("\n".join(lines), "pair_guard", status="pass", width=width, height=height, batch_pairs=batch,
                             metadata_checked=bool(reference_metadata), matched_controls=sorted(checked), unverified=unverified,
                             allowed_differences=len(differences))
        return io.NodeOutput(reference, candidate, report, json.dumps(record, ensure_ascii=False, indent=2), ui=ui.PreviewText(report))


class HaDeSMossRecipePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HaDeSMossRecipePreview", display_name="MOSS Recipe Preview", category="HaDeS/MOSS Dev Tools", is_output_node=True,
            description="Previews the recipe text that Model Saver will embed as hades/recipe. Connect its recipe output to Model Saver's recipe input; no file or model weight is read.",
            inputs=[
                io.String.Input("recipe_info", default="", multiline=True, tooltip="Info from MOSS Loader, Multi-LoRA or a merger; can also be typed notes. Its full text is passed to Model Saver."),
                io.Boolean.Input("include_metadata", default=True, tooltip="Mirror Model Saver's include_metadata setting. With this off, Model Saver writes no hades/recipe."),
                io.Model.Input("model", optional=True, tooltip="Optional consistency check against the model patcher's attached Suite recipe; no weights are read."),
            ],
            outputs=[io.String.Output(display_name="recipe"), io.String.Output(display_name="report"),
                     io.String.Output(display_name="preview_json")],
        )

    @classmethod
    def execute(cls, recipe_info="", include_metadata=True, model=None):
        recipe = recipe_info.strip()
        if not recipe:
            raise ValueError("Recipe Preview: recipe_info is empty; Model Saver would write no hades/recipe.")
        metas = []
        for line in recipe.splitlines():
            if line.startswith("HADES_META "):
                meta = parse_info_meta(line)
                if not isinstance(meta, dict) or not isinstance(meta.get("node"), str):
                    raise ValueError("Recipe Preview: malformed HADES_META trailer; fix or remove it before saving.")
                metas.append(meta)
        attached = model.get_attachment("hades_moss_recipe") if model is not None else None
        match = None if model is None or not metas or attached is None else attached in metas
        notes = []
        if not include_metadata:
            notes.append("Model Saver include_metadata is off: hades/recipe will not be written. Set the Saver to the same value.")
        if not metas:
            notes.append("No structured Suite trailer: Metadata Reader will display this text but cannot follow its source files.")
        if model is not None and attached is None:
            notes.append("Connected model has no Suite recipe attachment; provenance cannot be cross-checked.")
        elif match is False:
            notes.append("Connected model's Suite recipe does not match any trailer in recipe_info. Check for a stale or wrong branch.")
        if model is None:
            notes.append("Model not connected: this previews text only and cannot verify which model will be saved.")
        record = {"schema_version": 1, "will_embed_if_saver_matches": bool(include_metadata), "characters": len(recipe),
                  "structured_nodes": [m.get("node") for m in metas], "model_attachment_match": match, "notes": notes,
                  "note": "Preview is advisory. It does not inspect applied patches, source files, or the Saver's actual settings."}
        lines = ["MOSS Recipe Preview", f"Target: hades/recipe | {len(recipe)} characters | {len(metas)} Suite trailer(s)",
                 f"Connected model attachment: {'not checked' if match is None else 'matches' if match else 'DIFFERS'}",
                 f"Will be embedded if Model Saver uses these settings: {'yes' if include_metadata else 'no'}"]
        lines += ["NOTE: " + note for note in notes]
        lines += ["", "Recipe text sent to Model Saver:", recipe]
        # the trailer goes last, after the recipe's own trailers, so parse_info_meta reads this node's
        report = format_info("\n".join(lines), "recipe_preview", characters=len(recipe), structured_nodes=[m.get("node") for m in metas],
                             model_attachment_match=match, will_embed_if_saver_matches=bool(include_metadata), notes=len(notes))
        return io.NodeOutput(recipe, report, json.dumps(record, ensure_ascii=False, indent=2), ui=ui.PreviewText(report))


NODE_CLASS_MAPPINGS = {"HaDeSMossLayoutRegions": HaDeSMossLayoutRegions, "HaDeSMossImageAudit": HaDeSMossImageAudit,
                       "HaDeSMossPairGuard": HaDeSMossPairGuard, "HaDeSMossRecipePreview": HaDeSMossRecipePreview}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLayoutRegions": "MOSS Layout Regions", "HaDeSMossImageAudit": "MOSS Image Audit",
                              "HaDeSMossPairGuard": "MOSS Pair Guard", "HaDeSMossRecipePreview": "MOSS Recipe Preview"}
