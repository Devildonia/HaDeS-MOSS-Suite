"""Builds the example workflows in examples/ (ComfyUI editor format) from compact graph definitions (dev tool, not a node).

Widget order, optional inputs and seed controls are read from a running ComfyUI's /object_info, so the files match the
installed nodes. --run also queues every example (muted nodes left out) with the model names mapped to local files.

Example:
  python tools/make_examples.py --run --map moss_image_sft_bf16.safetensors=mossImage_sft_bf16.safetensors
"""
import argparse
import json
import os
import time
import urllib.request
import uuid

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
SCALAR_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
MUTED = 2  # LiteGraph node mode: never executed until the user unmutes it

# Official file names (Comfy-Org/MOSS-Image and the ComfyUI templates)
SFT, BASE = "moss_image_sft_bf16.safetensors", "moss_image_base_bf16.safetensors"
TE, VAE = "qwen3.5_9b_w4a8_gptq.safetensors", "flux2-vae.safetensors"
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
# Download links the editor offers for missing models (fetched only when the user asks), as in the official templates
HF = "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main"
TURBO, TURBO_DIVERSE = "HaDeS-MOSS-Turbo-v0.6.safetensors", "HaDeS-MOSS-Turbo-Diverse-v0.6.safetensors"
TURBO_8STEPS = "HaDeS-MOSS-Turbo-8steps-v0.7.safetensors"
UPSCALER = "RealESRGAN_x2plus.pth"
HF_TURBO = "https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo/resolve/main"
# file -> (models folder, download URL)
MODEL_FILES = {SFT: ("diffusion_models", f"{HF}/diffusion_models/{SFT}"), BASE: ("diffusion_models", f"{HF}/diffusion_models/{BASE}"),
               TE: ("text_encoders", f"{HF}/text_encoders/{TE}"), VAE: ("vae", f"{HF}/vae/{VAE}"),
               TURBO: ("loras", f"{HF_TURBO}/{TURBO}"), TURBO_DIVERSE: ("loras", f"{HF_TURBO}/{TURBO_DIVERSE}"),
               TURBO_8STEPS: ("loras", f"{HF_TURBO}/{TURBO_8STEPS}"),
               UPSCALER: ("upscale_models", f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{UPSCALER}")}
LORA = "moss_lora.safetensors"


def loaders():
    return {
        "clip": ("CLIPLoader", {"clip_name": TE, "type": "moss_image", "device": "default"}),
        "vae": ("VAELoader", {"vae_name": VAE}),
        "neg": ("CLIPTextEncode", {"clip": ("clip", 0), "text": NEGATIVE}),
    }


def ksampler(model, latent, positive="pos"):
    return ("HaDeSMossKSampler", {"model": model, "positive": (positive, 0), "negative": ("neg", 0), "latent_image": latent, "seed": 1000,
                                  "steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0, "flow_shift": 0.0,
                                  "slg": "off", "vae": ("vae", 0)})


EXAMPLES = {
    "HaDeS MOSS Suite - Merge, Save and Telemetry": {
        "sft": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        "base": ("UNETLoader", {"unet_name": BASE, "weight_dtype": "default"}),
        # SFT with layers 24-35 from Base: the groups that carry most of the SFT -> Base change (tools/RESULTS.md)
        "merge": ("HaDeSMossModelBlockMerger", {"model1": ("sft", 0), "model2": ("base", 0), "layers_00_05": 0.0, "layers_06_11": 0.0, "layers_12_17": 0.0,
                                                "layers_18_23": 0.0, "layers_24_29": 1.0, "layers_30_35": 1.0, "layers_36_41": 0.0, "text_embedder": 0.0,
                                                "time_modulation": 0.0, "io": 0.0, "recipe_override": ""}),
        **loaders(),
        "pos": ("CLIPTextEncode", {"clip": ("clip", 0), "text": "Studio photograph of an elderly fisherman with a weathered face and a grey knitted cap, "
                                                                 "soft window light from the left, shallow depth of field, 85mm lens, muted colors."}),
        "latent": ("HaDeSMossLatent", {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1}),
        "ks": ksampler(("merge", 0), ("latent", 0)),
        "save": ("SaveImage", {"images": ("ks", 1), "filename_prefix": "HaDeS_MOSS_merge"}),
        "saver": ("HaDeSMossModelSaver", {"model": ("merge", 0), "filename_prefix": "diffusion_models/MOSS_HaDeS_Merged", "precision": "bfloat16",
                                          "include_metadata": True, "custom_author": "HaDeS", "custom_model_title": ""}, MUTED),
        "multi": ("HaDeSMossMultiModelMerger", {"model_1": ("sft", 0), "model_2": ("base", 0), "method": "SLERP", "density": 0.5, "scale": 1.0, "seed": 0,
                                                "weight_1": 1.0, "weight_2": 1.0}, MUTED),
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("merge", 1), "info_2": ("latent", 3), "info_3": ("ks", 2)}),
        "report": ("PreviewAny", {"source": ("tel", 0)}),
    },
    "HaDeS MOSS Suite - Layout Poster": {
        "unet": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        **loaders(),
        "layout": ("HaDeSMossLayoutCompiler", {"draft": None, "batch_size": 1}),  # None = the node's own default draft
        "pos": ("CLIPTextEncode", {"clip": ("clip", 0), "text": ("layout", 0)}),
        "ks": ksampler(("unet", 0), ("layout", 1)),
        "save": ("SaveImage", {"images": ("ks", 1), "filename_prefix": "HaDeS_MOSS_layout"}),
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("layout", 4), "info_2": ("ks", 2)}),
        "report": ("PreviewAny", {"source": ("tel", 0)}),
    },
    "HaDeS MOSS Suite - Layout Editor": {
        "unet": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        **loaders(),
        # 1. optional: the Prompt Enhancer writes a layout draft from an idea (and / or an image); the editor offers it with IMPORT
        "enh": ("HaDeSMossPromptEnhancer", {"clip": ("clip", 0), "idea": "A travel poster for a mountain lake at sunrise, with the title "
                                                                         "\"LAKE ALMA\" at the top and a small line \"summer 2027\" at the bottom.",
                                            "mode": "layout", "width": 1024, "height": 1536, "temperature": 0.7, "seed": 1, "max_tokens": 1024,
                                            "negative": ""}, MUTED),
        "img": ("LoadImage", {"image": "example.png", "upload": "image"}),
        # 2. the editor: boxes on the canvas; its draft goes to the compiler, which outputs the prompt and the latent of the same size
        "editor": ("HaDeSMossLayoutEditor", {"draft": None, "draft_in": ("enh", 4), "image": ("img", 0)}),
        "layout": ("HaDeSMossLayoutCompiler", {"draft": ("editor", 0), "batch_size": 1}),
        "pos": ("CLIPTextEncode", {"clip": ("clip", 0), "text": ("layout", 0)}),
        "ks": ksampler(("unet", 0), ("layout", 1)),
        "save": ("SaveImage", {"images": ("ks", 1), "filename_prefix": "HaDeS_MOSS_layout_editor"}),
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("editor", 3), "info_2": ("layout", 4),
                                            "info_3": ("ks", 2)}),
        "report": ("PreviewAny", {"source": ("tel", 0)}),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Layout Editor\n\n"
                                  "Layout Editor: the boxes of the layout draft on a canvas of its size.\n"
                                  "  - pick OBJECT / CHARACTER / TEXT in the toolbar, then drag on an empty part of the canvas to draw a box\n"
                                  "  - drag a box to move it, drag its corners to resize it (16 px grid)\n"
                                  "  - with a box selected, the rows below edit its label, caption, text and pixels;\n"
                                  "    with none selected, the caption of the whole image\n"
                                  "  - click the size (1024 x 1024) to change the canvas; the boxes scale with it\n"
                                  "  - boxes outlined in red still need a caption (text boxes also their text)\n"
                                  "Its draft goes to the Layout Compiler, which writes the prompt and the latent of the same size.\n\n"
                                  "Background: the Load Image wired to the editor's image input shows behind the boxes at once.\n"
                                  "Pick your picture in the Load Image node (the editor has no upload button of its own);\n"
                                  "BG ON / OFF hides it, MATCH IMAGE sets the canvas to its proportions. Only a guide: the model never sees it.\n\n"
                                  "Optional (muted, right-click > Mode > Always to use):\n"
                                  "  - MOSS Prompt Enhancer in layout mode writes a draft from the idea, and from the picture through its\n"
                                  "    image input. Run once, then press IMPORT in the editor's toolbar to load it, and correct the boxes.\n"
                                  "    Your edits are never replaced by themselves."}),
        "_layout": {"note": (0, -620), "unet": (0, 0), "clip": (0, 230), "vae": (0, 450), "img": (0, 650),
                    "enh": (450, 0), "editor": (950, 0), "layout": (1550, 0), "pos": (1550, 600), "neg": (450, 900),
                    "ks": (2000, 0), "save": (2450, 0), "tel": (2450, 480), "report": (2900, 480)},
    },
    "HaDeS MOSS Suite - Turbo": {
        "unet": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        **loaders(),
        # the turbo LoRA and the settings it was trained at (4 steps, CFG 1, euler / simple), applied by the KSampler
        "turbo": ("HaDeSMossTurbo", {"lora_name": TURBO, "strength": 1.0, "steps": 4}),
        "pos": ("CLIPTextEncode", {"clip": ("clip", 0), "text": "Photograph of a hand-painted wooden sign outside a bakery that reads \"FRESH BREAD DAILY\" "
                                                                 "in large white letters and \"open 7am - 3pm\" in smaller letters below, morning light."}),
        "latent": ("HaDeSMossLatent", {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1}),
        "ks": (lambda k: (k[0], {**k[1], "turbo": ("turbo", 0)}))(ksampler(("unet", 0), ("latent", 0))),
        "save": ("SaveImage", {"images": ("ks", 1), "filename_prefix": "HaDeS_MOSS_turbo"}),
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("latent", 3), "info_2": ("ks", 2)}),
        "report": ("PreviewAny", {"source": ("tel", 0)}),
    },
    "HaDeS MOSS Suite - Turbo 8 steps + Upscale x2": {
        # 1. generate: Turbo 8-steps v0.7 at 8 steps (blind pairs and colour in tools/RESULTS.md, "At 8 steps")
        "unet": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        **loaders(),
        "pos": ("CLIPTextEncode", {"clip": ("clip", 0), "text": "An old watchmaker repairing a pocket watch at his workbench, brass gears and "
                                                                 "tweezers, warm lamp light, close-up photograph."}),
        "latent": ("HaDeSMossLatent", {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1}),
        "turbo": ("HaDeSMossTurbo", {"lora_name": TURBO_8STEPS, "strength": 1.0, "steps": 8}),
        "ks": (lambda k: (k[0], {**k[1], "turbo": ("turbo", 0)}))(ksampler(("unet", 0), ("latent", 0))),
        "save": ("SaveImage", {"images": ("ks", 1), "filename_prefix": "HaDeS_MOSS_1024"}),
        # 2. upscale x2 with a model: pixels only, nothing redrawn (about 1.5 s at 2048)
        "upmodel": ("UpscaleModelLoader", {"model_name": UPSCALER}),
        "up": ("ImageUpscaleWithModel", {"upscale_model": ("upmodel", 0), "image": ("ks", 1)}),
        "save_up": ("SaveImage", {"images": ("up", 0), "filename_prefix": "HaDeS_MOSS_x2"}),
        # 3. optional MOSS pass on the upscaled image (muted): Turbo 8-steps at denoise 0.25 adds hair, pores and stitching the
        #    upscaler smooths, about 16 s at 2048; SFT at 25 steps and the same denoise took 75 s and was no better by eye
        "enc": ("VAEEncode", {"pixels": ("up", 0), "vae": ("vae", 0)}, MUTED),
        "ks2": (lambda k: (k[0], {**k[1], "turbo": ("turbo", 0), "denoise": 0.25}, MUTED))(ksampler(("unet", 0), ("enc", 0))),
        "save_ref": ("SaveImage", {"images": ("ks2", 1), "filename_prefix": "HaDeS_MOSS_x2_refined"}, MUTED),
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("latent", 3), "info_2": ("ks", 2)}),
        "report": ("PreviewAny", {"source": ("tel", 0)}),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Turbo 8 steps + Upscale x2\n\n"
                                  "1. Generate (always on): MOSS-Image SFT with HaDeS Turbo 8-steps v0.7, 8 steps, CFG 1, up to 1024 px.\n"
                                  "   Without the turbo: unplug the turbo input of the MOSS KSampler (25 steps, CFG 2.5).\n\n"
                                  "2. Upscale x2 (on): RealESRGAN x2 doubles the image in pixels, in about a second. Nothing is redrawn.\n"
                                  "   To turn it off: right-click the group > Set Group Nodes to Never.\n\n"
                                  "3. Upscale x2 + MOSS pass (off): re-renders the upscaled image at denoise 0.25 with the same turbo,\n"
                                  "   adding hair, pores and fine texture the upscaler smooths (about 16 s at 2048). Needs group 2.\n"
                                  "   To turn it on: right-click the group > Set Group Nodes to Always.\n"
                                  "   Denoise 0.40 (also measured) adds a little more detail and changes the image a little more.\n\n"
                                  "Models: RealESRGAN_x2plus.pth in models/upscale_models, the turbo LoRA in models/loras.\n"
                                  "Measurements: tools/RESULTS.md in the HaDeS MOSS Suite."}),
        "_layout": {"note": (0, -620), "unet": (0, 0), "clip": (0, 230), "vae": (0, 450), "turbo": (0, 640),
                    "pos": (450, 0), "neg": (450, 250), "latent": (450, 500),
                    "ks": (900, 0), "save": (1350, 0), "tel": (1350, 480), "report": (1800, 480),
                    "upmodel": (2300, 0), "up": (2300, 180), "save_up": (2750, 0),
                    "enc": (2300, 700), "ks2": (2750, 700), "save_ref": (3200, 700)},
        "_groups": [("1. Generate (Turbo 8 steps)", ["unet", "clip", "vae", "turbo", "pos", "neg", "latent", "ks", "save", "tel", "report"], "#335533"),
                    ("2. Upscale x2 (model)", ["upmodel", "up", "save_up"], "#334466"),
                    ("3. Upscale x2 + MOSS pass (turn on to use)", ["enc", "ks2", "save_ref"], "#553355")],
    },
    "HaDeS MOSS Suite - LoRA Toolkit": {
        "unet": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        "tuned": ("UNETLoader", {"unet_name": "your_moss_finetune.safetensors", "weight_dtype": "default"}, MUTED),
        "blockw": ("HaDeSMossLoraBlockWeight", {"model": ("unet", 0), "lora_name": LORA, "strength_model": 1.0, "layers_00_05": 1.0, "layers_06_11": 1.0,
                                                "layers_12_17": 1.0, "layers_18_23": 1.0, "layers_24_29": 1.0, "layers_30_35": 1.0, "layers_36_41": 1.0,
                                                "other_layers": 1.0}),
        **loaders(),
        "pos": ("CLIPTextEncode", {"clip": ("clip", 0), "text": "Watercolor illustration of a fox reading a book under a mushroom in a rainy forest, "
                                                                 "soft pastel palette, visible paper texture."}),
        "latent": ("HaDeSMossLatent", {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1}),
        "ks": ksampler(("blockw", 0), ("latent", 0)),
        "save": ("SaveImage", {"images": ("ks", 1), "filename_prefix": "HaDeS_MOSS_lora"}),
        "lmerge": ("HaDeSMossLoraMerger", {"model": ("unet", 0), "lora_a": LORA, "weight_a": 1.0, "lora_b": LORA, "weight_b": 1.0, "rank": 64,
                                           "export_filename": "HaDeS_MOSS_Merged_LoRA", "dynamic_energy": 0.0, "overwrite_existing": False}, MUTED),
        "resize": ("HaDeSMossLoraResize", {"model": ("unet", 0), "lora_name": LORA, "target_rank": 32, "export_filename": "HaDeS_MOSS_Resized_LoRA",
                                           "dynamic_energy": 0.0, "overwrite_existing": False}, MUTED),
        "extract": ("HaDeSMossLoraExtractor", {"model_tuned": ("tuned", 0), "model_base": ("unet", 0), "rank": 32,
                                               "export_filename": "HaDeS_MOSS_Extracted_LoRA", "overwrite_existing": False}, MUTED),
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("blockw", 1), "info_2": ("ks", 2)}),
        "report": ("PreviewAny", {"source": ("tel", 0)}),
    },
}


def fetch(server, path, data=None):
    req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def input_specs(info):
    """[(name, type, options, optional)] in the node's declared order; a combo's choices go into options["choices"]."""
    order = info.get("input_order", {})
    out = []
    for section in ("required", "optional"):
        specs = info["input"].get(section, {})
        for name in order.get(section, list(specs)):
            spec = specs[name]
            options = dict(spec[1]) if len(spec) > 1 else {}
            if isinstance(spec[0], list):  # old-style combo: [choices, options]
                kind, options["choices"] = "COMBO", spec[0]
            else:
                kind = spec[0]
                if kind == "COMBO":
                    options["choices"] = options.get("options", [])
            out.append((name, kind, options, section == "optional"))
    return out


def widget_default(kind, options):
    if "default" in options:
        return options["default"]
    if kind == "COMBO":
        return options["choices"][0] if options["choices"] else ""
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(kind, "")


def is_widget(kind, options):
    return kind in SCALAR_TYPES and not options.get("forceInput")


def to_editor(example, object_info):
    """Compact graph -> ComfyUI editor workflow (LiteGraph 0.4). Optional keys: _layout {name: (x, y)} fixes positions, _groups
    [(title, [names], color)] draws groups around nodes; a "Note" node is a frontend note with the given text."""
    graph = {k: v for k, v in example.items() if not k.startswith("_")}
    layout = example.get("_layout", {})
    ids = {name: i + 1 for i, name in enumerate(graph)}
    depth = {}

    def node_depth(name):
        if name not in depth:
            srcs = [v[0] for v in graph[name][1].values() if isinstance(v, tuple)]
            depth[name] = 1 + max((node_depth(s) for s in srcs), default=-1)
        return depth[name]

    nodes, links, rows, heights = [], [], {}, {}
    out_links = {name: {} for name in graph}
    for name, (cls, inputs, *mode) in graph.items():
        if cls == "Note":
            heights[name] = 520
            nodes.append({"id": ids[name], "type": "Note", "pos": list(layout.get(name, (0, 0))), "size": [800, 520], "flags": {}, "order": 0,
                          "mode": 0, "inputs": [], "outputs": [], "widgets_values": [inputs["text"]], "properties": {}, "_outputs": None})
            continue
        info = object_info[cls]
        node_inputs, widgets = [], []
        for inp, kind, options, optional in input_specs(info):
            value = inputs.get(inp)
            linked = isinstance(value, tuple)
            if is_widget(kind, options):
                widgets.append(widget_default(kind, options) if (value is None or linked) else value)
                if options.get("control_after_generate"):
                    widgets.append("fixed")
            if linked or not is_widget(kind, options):
                entry = {"name": inp, "type": kind, "link": None}
                if is_widget(kind, options):
                    entry["widget"] = {"name": inp}
                if optional:
                    entry["shape"] = 7
                if linked:
                    link_id = len(links) + 1
                    src, slot = value
                    links.append([link_id, ids[src], slot, ids[name], len(node_inputs), object_info[graph[src][0]]["output"][slot]])
                    out_links[src].setdefault(slot, []).append(link_id)
                    entry["link"] = link_id
                node_inputs.append(entry)
        col = node_depth(name)
        y = rows.get(col, 0)
        # HaDeS MOSS nodes add a banner and draw one 25 px row per widget; the layout compiler opens at 520 px, the editor's canvas ~700 px
        heights[name] = {"HaDeSMossLayoutCompiler": 560, "HaDeSMossLayoutEditor": 700}.get(cls, 190 + 30 * (len(widgets) + len(node_inputs)))
        rows[col] = y + heights[name]
        properties = {"Node name for S&R": cls}
        models = [{"name": w, "url": MODEL_FILES[w][1], "directory": MODEL_FILES[w][0]} for w in widgets if w in MODEL_FILES]
        if models:
            properties["models"] = models
        nodes.append({"id": ids[name], "type": cls, "pos": list(layout.get(name, (col * 500, y))), "size": [520 if cls == "HaDeSMossLayoutEditor" else 380, 100], "flags": {}, "order": 0, "mode": mode[0] if mode else 0,
                      "inputs": node_inputs, "widgets_values": widgets, "properties": properties, "_outputs": info})
    for node, name in zip(nodes, graph):
        info = node.pop("_outputs")
        if info is None:
            continue
        node["outputs"] = [{"name": n, "type": t, "links": out_links[name].get(i, [])} for i, (n, t) in enumerate(zip(info["output_name"], info["output"]))]
    for order, name in enumerate(sorted((n for n in graph if graph[n][0] != "Note"), key=node_depth)):
        nodes[ids[name] - 1]["order"] = order
    groups = []
    for title, members, color in example.get("_groups", []):
        xs = [nodes[ids[m] - 1]["pos"][0] for m in members]
        ys = [nodes[ids[m] - 1]["pos"][1] for m in members]
        x0, y0 = min(xs) - 30, min(ys) - 80
        x1 = max(xs) + 380 + 30
        y1 = max(nodes[ids[m] - 1]["pos"][1] + heights[m] for m in members) + 30
        groups.append({"id": len(groups) + 1, "title": title, "bounding": [x0, y0, x1 - x0, y1 - y0], "color": color, "font_size": 24, "flags": {}})
    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(sorted(graph)))), "revision": 0, "last_node_id": len(nodes), "last_link_id": len(links),
            "nodes": nodes, "links": links, "groups": groups, "config": {}, "extra": {}, "version": 0.4}


def to_api(example, object_info, mapping):
    """The runnable API prompt: muted nodes and the nodes fed only by them left out, local file names substituted."""
    graph = {k: v for k, v in example.items() if not k.startswith("_") and v[0] != "Note"}
    live = {n for n, (_, _, *mode) in graph.items() if not mode}
    changed = True
    while changed:
        changed = False
        for n in list(live):
            if any(isinstance(v, tuple) and v[0] not in live for v in graph[n][1].values()):
                live.discard(n)
                changed = True
    prompt = {}
    for n in live:
        cls, inputs = graph[n][0], graph[n][1]
        resolved = {}
        for k, v in inputs.items():
            if isinstance(v, tuple):
                resolved[k] = [v[0], v[1]]
            elif v is None:
                resolved[k] = next(widget_default(kind, o) for name, kind, o, _ in input_specs(object_info[cls]) if name == k)
            else:
                resolved[k] = mapping.get(v, v) if isinstance(v, str) else v
        prompt[n] = {"class_type": cls, "inputs": resolved}
    return prompt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--run", action="store_true", help="also queue every example and wait for it")
    p.add_argument("--map", nargs="*", default=[], help="official=local file name pairs for --run")
    args = p.parse_args()
    mapping = dict(m.split("=", 1) for m in args.map)
    object_info = fetch(args.server, "/object_info")
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for title, graph in EXAMPLES.items():
        path = os.path.join(EXAMPLES_DIR, f"{title}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_editor(graph, object_info), f, indent=1, ensure_ascii=False)
        print("wrote", path)
        if args.run:
            pid = fetch(args.server, "/prompt", json.dumps({"prompt": to_api(graph, object_info, mapping)}).encode())["prompt_id"]
            while True:
                hist = fetch(args.server, f"/history/{pid}")
                if pid in hist:
                    break
                time.sleep(1)
            status = hist[pid]["status"]
            report = hist[pid]["outputs"].get("report", {}).get("text", [""])[0]
            print("  run:", status["status_str"], "|", report.split("## Settings")[0].strip().replace("\n", " / ")[:400])


if __name__ == "__main__":
    main()
