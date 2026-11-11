"""Image-space measurements and explicit regions; no model execution or file writes."""
import json
import math
import re

import torch
from comfy_api.latest import io, ui

from .moss_common import format_info
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


NODE_CLASS_MAPPINGS = {"HaDeSMossLayoutRegions": HaDeSMossLayoutRegions, "HaDeSMossImageAudit": HaDeSMossImageAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossLayoutRegions": "MOSS Layout Regions", "HaDeSMossImageAudit": "MOSS Image Audit"}
