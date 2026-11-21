"""Builds the example workflows in examples/ and examples/dev/ (ComfyUI editor format) from compact graph definitions.

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
TE, VAE = "qwen3.5_9b_int8_convrot.safetensors", "flux2-vae.safetensors"
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
# Download links the editor offers for missing models (fetched only when the user asks), as in the official templates
HF = "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main"
TURBO, TURBO_DIVERSE = "HaDeS-MOSS-Turbo-v0.6.safetensors", "HaDeS-MOSS-Turbo-Diverse-v0.6.safetensors"
TURBO_8STEPS = "HaDeS-MOSS-Turbo-8steps-v0.7.safetensors"
TURBO_V10 = "HaDeS-MOSS-Turbo-8steps-v1.0.safetensors"  # for Base
UPSCALER = "RealESRGAN_x2plus.pth"
BULLE = "HaDeS_MOSS_preview_beta.safetensors"  # Bulle d'Encre, the fourth video tutorial's model
HF_TURBO = "https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo/resolve/main"
# file -> (models folder, download URL)
MODEL_FILES = {SFT: ("diffusion_models", f"{HF}/diffusion_models/{SFT}"), BASE: ("diffusion_models", f"{HF}/diffusion_models/{BASE}"),
               TE: ("text_encoders", f"{HF}/text_encoders/{TE}"), VAE: ("vae", f"{HF}/vae/{VAE}"),
               TURBO: ("loras", f"{HF_TURBO}/{TURBO}"), TURBO_DIVERSE: ("loras", f"{HF_TURBO}/{TURBO_DIVERSE}"),
               TURBO_8STEPS: ("loras", f"{HF_TURBO}/{TURBO_8STEPS}"), TURBO_V10: ("loras", f"{HF_TURBO}/{TURBO_V10}"),
               UPSCALER: ("upscale_models", f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{UPSCALER}"),
               BULLE: ("diffusion_models", f"https://huggingface.co/DeVilDoNia/Bulle-d-Encre/resolve/main/{BULLE}")}
LORA = "moss_lora.safetensors"
CONTROL_ADAPTER = "moss-color-depth-control.safetensors"  # darask0/MOSS-Image-color-depth-control (Hugging Face), trained on Base


def loader(unet=SFT):
    """MOSS Loader: MODEL 0, CLIP 1, VAE 2, info 3."""
    return {"load": ("HaDeSMossLoader", {"unet_name": unet, "clip_name": TE, "vae_name": VAE, "weight_dtype": "default", "clip_device": "default"})}


def prompt(text, mode="as written", seed=1, **extra):
    """MOSS Prompt Enhancer: positive 0, negative 1, prompt 2, draft 3, width 4, height 5, info 6. "as written" encodes `text` as typed
    (CLIPTextEncode for both prompts); an empty negative is the official template's. `text` may be a link, e.g. a compiler's prompt."""
    return ("HaDeSMossPromptEnhancer", {"clip": ("load", 1), "idea": text, "mode": mode, "width": 1024, "height": 1024, "temperature": 0.7,
                                        "seed": seed, "max_tokens": 1024, "negative": "", **extra})


def latent(seed=1000):
    """MOSS Latent: latent 0, width 1, height 2, info 3, seed 4 (the seed shared with the samplers)."""
    return ("HaDeSMossLatent", {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1, "seed": seed})


def ksampler(model, latent_image, seed=1000, enh="enh"):
    """seed: a number, or a link such as ("latent", 4), the MOSS Latent's shared seed."""
    return ("HaDeSMossKSampler", {"model": model, "positive": (enh, 0), "negative": (enh, 1), "latent_image": latent_image, "seed": seed,
                                  "steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0, "flow_shift": 0.0,
                                  "slg": "off", "vae": ("load", 2)})


def with_turbo(ks, **extra):
    cls, inputs, *mode = ks
    return (cls, {**inputs, "turbo": ("turbo", 0), **extra}, *mode)


def telemetry(*infos):
    return {"tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", **{f"info_{i + 1}": src for i, src in enumerate(infos)}}),
            "report": ("PreviewAny", {"source": ("tel", 0)})}


def synthetic_info(node, **fields):
    """A model-free info trailer for exercising provenance wiring in a Dev example."""
    return "UI stage validation\nHADES_META " + json.dumps({"node": node, **fields}, separators=(",", ":"))


EXAMPLES = {
    "HaDeS MOSS Suite - Dev Tools": {
        "reference": ("EmptyImage", {"width": 256, "height": 256, "batch_size": 1, "color": 0x304050}),
        "patch": ("EmptyImage", {"width": 256, "height": 256, "batch_size": 1, "color": 0xB04050}),
        "editor": ("HaDeSMossLayoutEditor", {"draft": json.dumps({"width": 256, "height": 256, "bbox_coord_space": "abs", "bbox_format": "xyxy",
            "summary": {"caption": "Known red-channel change in one rectangle"},
            "items": [{"label": "object", "caption": "test region", "bbox": [32, 64, 128, 192]}]}), "image": ("reference", 0)}),
        "regions": ("HaDeSMossLayoutRegions", {"draft": ("editor", 0), "selection": "1", "padding": 0}),
        "candidate": ("ImageCompositeMasked", {"destination": ("reference", 0), "source": ("patch", 0), "x": 0, "y": 0,
                                                 "resize_source": False, "mask": ("regions", 0)}),
        "audit": ("HaDeSMossImageAudit", {"reference": ("reference", 0), "candidate": ("candidate", 0), "region_mask": ("regions", 0),
            "change_threshold": 2.0, "difference_gain": 1.0, "reference_label": "RGB 48,64,80", "candidate_label": "red +128 inside region"}),
        "mask_preview": ("PreviewImage", {"images": ("regions", 1)}),
        "candidate_preview": ("PreviewImage", {"images": ("candidate", 0)}),
        "difference_preview": ("PreviewImage", {"images": ("audit", 0)}),
        "note": ("Note", {"text": "DEV TOOLS — no models required. Queue this workflow to verify the nodes.\n\n"
            "Layout Editor -> Layout Regions produces the selected box mask. Native composition changes only its red channel by 128/255.\n\n"
            "Expected (float32 tolerance): region 12,288 pixels (18.75%); inside MAE about 42.6667/255, outside MAE 0; global MAE 8/255 and changed 18.75%.\n\n"
            "Move the box or change padding and rerun: measurements follow the actual union. No automatic resizing or alignment.\n\n"
            "For real experiments replace the two Image Audit inputs with matched-size reference/candidate images. Put the reference in the Layout Editor background and match its canvas. Select boxes by 1-based draft order.\n\n"
            "White = inside, not semantic segmentation. Differences do not establish edit success or quality. Per-pair metrics are in metrics_json; report can feed Suite Telemetry. Preview images are temporary."}),
        "_layout": {"reference": (0, 0), "patch": (0, 230), "editor": (420, 0), "regions": (1020, 0), "candidate": (1460, 0),
                    "audit": (1900, 0), "mask_preview": (1020, 600), "candidate_preview": (1460, 350), "difference_preview": (1900, 650), "note": (0, 850)},
    },
    "HaDeS MOSS Suite - Stage Metadata (Dev)": {
        "image": ("EmptyImage", {"width": 64, "height": 64, "batch_size": 1, "color": 0x3399ff}),
        "base_loader": ("PrimitiveStringMultiline", {"value": synthetic_info("loader", model="base_ui", model_file="base_ui.safetensors",
            precision="bfloat16", vae="flux2-vae.safetensors", text_encoder="qwen.safetensors")}),
        "base_prompt": ("PrimitiveStringMultiline", {"value": synthetic_info("prompt_enhancer", prompt="Base scene.", negative="blurry")}),
        "base_ks": ("PrimitiveStringMultiline", {"value": synthetic_info("ksampler", width=64, height=64, steps=8, cfg=1.0,
            flow_shift=5.0, sampler="euler", scheduler="simple", seed=11, denoise=1.0, slg="off")}),
        "refine_loader": ("PrimitiveStringMultiline", {"value": synthetic_info("loader", model="refine_ui", model_file="refine_ui.safetensors",
            precision="bfloat16", vae="flux2-vae.safetensors", text_encoder="qwen.safetensors")}),
        "refine_prompt": ("PrimitiveStringMultiline", {"value": synthetic_info("prompt_enhancer", prompt="Refine scene.", negative="blurry")}),
        "refine_ks": ("PrimitiveStringMultiline", {"value": synthetic_info("ksampler", width=64, height=64, steps=12, cfg=1.0,
            flow_shift=5.0, sampler="euler", scheduler="simple", seed=22, denoise=1.0, slg="off")}),
        "full": ("HaDeSMossSuiteTelemetry", {"report_title": "Complete pipeline", "info_1": ("base_loader", 0),
            "info_2": ("base_prompt", 0), "info_3": ("base_ks", 0), "info_4": ("refine_loader", 0),
            "info_5": ("refine_prompt", 0), "info_6": ("refine_ks", 0)}),
        "stage": ("HaDeSMossSuiteTelemetry", {"report_title": "Selected refine stage", "info_1": ("refine_loader", 0),
            "info_2": ("refine_prompt", 0), "info_3": ("refine_ks", 0)}),
        "saver": ("HaDeSMossImageSaver", {"images": ("image", 0), "filename_prefix": "stage_ui_%model%_%seed%_%steps%",
            "hashes": False, "embed_workflow": True, "metadata": ("full", 1), "primary_stage_metadata": ("stage", 1)}),
        "_layout": {"image": (0, 0), "base_loader": (0, 350), "base_prompt": (0, 750), "base_ks": (0, 1150),
                    "refine_loader": (450, 350), "refine_prompt": (450, 750), "refine_ks": (450, 1150),
                    "full": (900, 350), "stage": (900, 850), "saver": (1400, 350)},
    },
    "HaDeS MOSS Suite - Merge, Save and Telemetry": {
        **loader(),  # SFT, its text encoder and VAE
        "base": ("UNETLoader", {"unet_name": BASE, "weight_dtype": "default"}),
        # SFT with layers 24-35 from Base: the groups that carry most of the SFT -> Base change (tools/RESULTS.md)
        "merge": ("HaDeSMossModelBlockMerger", {"model1": ("load", 0), "model2": ("base", 0), "layers_00_05": 0.0, "layers_06_11": 0.0, "layers_12_17": 0.0,
                                                "layers_18_23": 0.0, "layers_24_29": 1.0, "layers_30_35": 1.0, "layers_36_41": 0.0, "text_embedder": 0.0,
                                                "time_modulation": 0.0, "io": 0.0, "recipe_override": ""}),
        "enh": prompt("Studio photograph of an elderly fisherman with a weathered face and a grey knitted cap, "
                      "soft window light from the left, shallow depth of field, 85mm lens, muted colors."),
        "latent": latent(),
        "ks": ksampler(("merge", 0), ("latent", 0), seed=("latent", 4)),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_merge_%seed%", "hashes": True, "embed_workflow": True}),
        "saver": ("HaDeSMossModelSaver", {"model": ("merge", 0), "filename_prefix": "diffusion_models/MOSS_HaDeS_Merged", "precision": "bfloat16",
                                          "include_metadata": True, "custom_author": "HaDeS", "custom_model_title": "", "rounding": "stochastic"}, MUTED),
        "multi": ("HaDeSMossMultiModelMerger", {"model_1": ("load", 0), "model_2": ("base", 0), "method": "SLERP", "density": 0.5, "scale": 1.0, "seed": 0,
                                                "weight_1": 1.0, "weight_2": 1.0}, MUTED),
        **telemetry(("load", 3), ("enh", 6), ("merge", 1), ("latent", 3), ("ks", 2)),
    },
    "HaDeS MOSS Suite - Prompt Enhancer": {
        # the third video tutorial (tools/moss_video.py prompt_enhancer): every mode on the first video's fisherman, Base + Turbo
        # 8-steps v1.0; texts, word counts and times in tools/RESULTS.md, "Every mode on one short idea"
        **loader(BASE),
        "enh": prompt("a photograph of an old fisherman on a dock", mode="detailed", seed=1, image=("img", 0)),
        "img": ("LoadImage", {"image": "example.png", "upload": "image"}, MUTED),
        "latent": latent(seed=1013),
        "turbo": ("HaDeSMossTurbo", {"lora_name": TURBO_V10, "strength": 1.0, "steps": 8}),
        "ks": with_turbo(ksampler(("load", 0), ("latent", 0), seed=("latent", 4)), sampler_name="euler"),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_enhancer_%seed%", "hashes": True, "embed_workflow": True}),
        "text": ("PreviewAny", {"source": ("enh", 2)}),
        **telemetry(("load", 3), ("enh", 6), ("latent", 3), ("ks", 2), ("turbo", 1)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Prompt Enhancer (the third video tutorial)\n\n"
                                  "MOSS Prompt Enhancer: a prompt box that can also write the prompt, with MOSS-Image's own text\n"
                                  "encoder (Qwen3.5); no other model is loaded. The text preview shows what it wrote.\n"
                                  "  - as written: the idea is encoded as typed (what the first two videos used)\n"
                                  "  - caption: a paragraph that keeps to the idea (56-118 words in the video's tests)\n"
                                  "  - detailed: a photographer's or illustrator's brief (168-238 words). With no medium in the idea it\n"
                                  "    picks one: 8 of 9 tries chose an oil painting for this fisherman; write 'a photograph of' for a photo\n"
                                  "  - corrections: paste the detailed prompt back as the idea and add 'Corrections: ...' at the end (any\n"
                                  "    language). A hat and a cat: 6 of 6, the rest kept. Name what contradicts the change too: 'a young man'\n"
                                  "    kept the white beard; 'a young man with a short dark beard' worked 6 of 6\n"
                                  "  - detailed with an empty idea: a random prompt (the seed draws a medium, a subject and a place / time)\n"
                                  "  - layout: boxes for every element (see the Layout examples)\n"
                                  "  - tags (1boy, white beard, ...) are rewritten as prose; count tags are written as adults\n"
                                  "The enhancer's own seed and temperature decide the text: same seed, same text; temperature 0 always\n"
                                  "gives the same answer. MOSS Latent's seed 1013 is the image seed of the videos.\n\n"
                                  "Negative: empty uses the official MOSS-Image negative. What you write is also told to the enhancer:\n"
                                  "with 'hat', the six test captions said 'no hat' and no image had one (5 of 6 had one without it).\n"
                                  "At CFG 1 (the turbo) the sampler does not use the negative itself.\n\n"
                                  "Optional (muted): Load Image. Unmute it (right-click > Mode > Always) to have the prompt written\n"
                                  "from the picture: with an empty idea the picture is the whole brief, otherwise the starting point.\n\n"
                                  "About 2 s per answer on an RTX 5090. Measurements: tools/RESULTS.md in the HaDeS MOSS Suite."}),
        "_layout": {"note": (0, -640), "load": (0, 0), "img": (0, 420), "enh": (440, 0), "latent": (940, 0), "ks": (1380, 0),
                    "save": (1820, 0), "turbo": (940, 450), "text": (440, 720), "tel": (1380, 640), "report": (1820, 640)},
    },
    "HaDeS MOSS Suite - Layout Poster": {
        **loader(),
        "layout": ("HaDeSMossLayoutCompiler", {"draft": None, "batch_size": 1}),  # None = the node's own default draft
        "enh": prompt(("layout", 0)),
        "ks": ksampler(("load", 0), ("layout", 1)),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_layout_%seed%", "hashes": True, "embed_workflow": True}),
        **telemetry(("load", 3), ("enh", 6), ("layout", 4), ("ks", 2)),
    },
    "HaDeS MOSS Suite - Layout Editor": {
        **loader(),
        # 1. optional: the Prompt Enhancer writes a layout draft from an idea (and / or an image); the editor offers it with IMPORT
        "draft": ("HaDeSMossPromptEnhancer", {"clip": ("load", 1), "idea": "A travel poster for a mountain lake at sunrise, with the title "
                                                                           "\"LAKE ALMA\" at the top and a small line \"summer 2027\" at the bottom.",
                                              "mode": "layout", "width": 1024, "height": 1536, "temperature": 0.7, "seed": 1, "max_tokens": 1024,
                                              "negative": ""}, MUTED),
        "img": ("LoadImage", {"image": "example.png", "upload": "image"}),
        # 2. the editor: boxes on the canvas; its draft goes to the compiler, which outputs the prompt and the latent of the same size
        "editor": ("HaDeSMossLayoutEditor", {"draft": None, "draft_in": ("draft", 3), "image": ("img", 0)}),
        "layout": ("HaDeSMossLayoutCompiler", {"draft": ("editor", 0), "batch_size": 1}),
        "enh": prompt(("layout", 0)),
        "ks": ksampler(("load", 0), ("layout", 1)),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_layout_editor_%seed%", "hashes": True, "embed_workflow": True}),
        **telemetry(("load", 3), ("enh", 6), ("editor", 3), ("layout", 4), ("ks", 2)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Layout Editor\n\n"
                                  "Layout Editor: the boxes of the layout draft on a canvas of its size.\n"
                                  "  - pick OBJECT / CHARACTER / TEXT in the toolbar, then drag on an empty part of the canvas to draw a box\n"
                                  "  - drag a box to move it, drag its corners to resize it (16 px grid)\n"
                                  "  - with a box selected, the rows below edit its label, caption, text and pixels;\n"
                                  "    with none selected, the caption of the whole image\n"
                                  "  - click the size (1024 x 1024) to change the canvas; the boxes scale with it\n"
                                  "  - boxes outlined in red still need a caption (text boxes also their text)\n"
                                  "Its draft goes to the Layout Compiler, which writes the prompt and the latent of the same size;\n"
                                  "the Prompt Enhancer on the right encodes that prompt as written.\n\n"
                                  "Background: the Load Image wired to the editor's image input shows behind the boxes at once.\n"
                                  "Pick your picture in the Load Image node (the editor has no upload button of its own);\n"
                                  "BG ON / OFF hides it, MATCH IMAGE sets the canvas to its proportions. Only a guide: the model never sees it.\n\n"
                                  "Optional (muted, right-click > Mode > Always to use):\n"
                                  "  - the Prompt Enhancer on the left, in layout mode, writes a draft from the idea, and from the picture through\n"
                                  "    its image input. Press DRAFT LAYOUT under the canvas: only the nodes up to the editor run\n"
                                  "    (no sampling) and its boxes replace the canvas. Move, resize or delete them, then press GENERATE\n"
                                  "    (the same as Run). A normal Run never replaces your boxes: it only offers the new draft with IMPORT."}),
        "_layout": {"note": (0, -620), "load": (0, 0), "img": (0, 380), "draft": (450, 0), "editor": (950, 0), "layout": (1550, 0),
                    "enh": (1550, 600), "ks": (2050, 0), "save": (2500, 0), "tel": (2500, 480), "report": (2950, 480)},
    },
    "HaDeS MOSS Suite - Layout from a Sentence": {
        **loader(),
        # 1. the idea in plain words, saying where each thing goes; the Prompt Enhancer in layout mode turns it into boxes
        "draft": ("HaDeSMossPromptEnhancer", {"clip": ("load", 1), "idea": "A man seen from behind on the left side of a narrow street at night, "
                                                                           "a full moon in the upper right, a small dog at the far end of the street, "
                                                                           "part of an old brick building on the right, and the signature \"HaDeS\" "
                                                                           "in the lower right corner.",
                                              "mode": "layout", "width": 1024, "height": 1536, "temperature": 0.7, "seed": 1, "max_tokens": 1024,
                                              "negative": ""}),
        # 2. DRAFT LAYOUT brings the boxes onto the canvas; move them, then GENERATE
        "editor": ("HaDeSMossLayoutEditor", {"draft": None, "draft_in": ("draft", 3)}),
        "layout": ("HaDeSMossLayoutCompiler", {"draft": ("editor", 0), "batch_size": 1}),
        "enh": prompt(("layout", 0)),
        "ks": ksampler(("load", 0), ("layout", 1)),
        # 3. the text boxes read back from the image
        "check": ("HaDeSMossTextCheck", {"image": ("ks", 1), "clip": ("load", 1), "layout": ("editor", 0), "expected_text": "", "max_tokens": 96}),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_layout_sentence_%seed%", "hashes": True, "embed_workflow": True}),
        # not the draft writer: the editor holds the canvas actually used
        **telemetry(("load", 3), ("editor", 3), ("layout", 4), ("enh", 6), ("ks", 2), ("check", 3)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Layout from a Sentence\n\n"
                                  "1. Write the idea in the Prompt Enhancer on the left (layout mode), saying where each thing goes:\n"
                                  "   left / right, upper / lower, in the centre, at the far end. Any language the text encoder reads.\n"
                                  "   Give the exact words of any text (a title, a signature), or it may invent them.\n"
                                  "2. Press DRAFT LAYOUT under the Layout Editor's canvas: only the enhancer and the editor run (no sampling)\n"
                                  "   and the boxes it wrote appear on the canvas, one per thing, with its name and caption.\n"
                                  "3. Move, resize, draw or delete boxes; select one to edit its caption or text.\n"
                                  "4. Press GENERATE (the same as Run): the image is made with your boxes. The enhancer runs again only if\n"
                                  "   its inputs changed; your boxes are never replaced by a normal run (it only offers the new draft with IMPORT).\n"
                                  "5. MOSS Text Check reads each text box back from the image and says exact / close / missing.\n\n"
                                  "Tried on 8 ideas, one seed each (tools/RESULTS.md, Layout mode): the enhancer put all 28 stated elements\n"
                                  "in the stated region; MOSS drew 26 inside their boxes. Check the boxes before you press GENERATE."}),
        "_layout": {"note": (0, -620), "load": (0, 0), "draft": (450, 0), "editor": (950, 0), "layout": (1550, 0),
                    "enh": (1550, 600), "ks": (2050, 0), "check": (2050, 560), "save": (2500, 0), "tel": (2500, 480), "report": (2950, 480)},
    },
    "HaDeS MOSS Suite - Compare Models": {
        **loader(),  # SFT, its text encoder and VAE
        "base": ("UNETLoader", {"unet_name": BASE, "weight_dtype": "default"}),
        "enh": prompt("Photograph of a street musician playing a battered accordion in a rainy square at dusk, warm shop lights, puddles."),
        "latent": latent(),
        # 1. MOSS Compare Sampler: every model sampled with the same prompt, latent, seed and settings, saved as one labelled sheet
        "cmp": ("HaDeSMossCompareSampler", {"model_A": ("load", 0), "model_B": ("base", 0), "positive": ("enh", 0), "negative": ("enh", 1),
                                            "latent_image": ("latent", 0), "vae": ("load", 2), "seed": ("latent", 4), "steps": 25, "cfg": 2.5,
                                            "sampler_name": "er_sde", "scheduler": "simple", "filename_prefix": "MOSS/compare/HaDeS_MOSS_compare",
                                            "save": True, "embed_workflow": True, "prompt_info": ("enh", 6), "blind": False, "blind_seed": 0}),
        # 2. MOSS Compare Sheet (muted): any images side by side, here two seeds of one model, labelled from each sampler's info
        "ks_a": ksampler(("load", 0), ("latent", 0), seed=1000) + (MUTED,),
        "ks_b": ksampler(("load", 0), ("latent", 0), seed=1001) + (MUTED,),
        "sheet": ("HaDeSMossCompareSheet", {"image_A": ("ks_a", 1), "image_B": ("ks_b", 1), "info_A": ("ks_a", 2), "info_B": ("ks_b", 2),
                                            "label_A": "seed 1000", "label_B": "seed 1001", "filename_prefix": "MOSS/compare/HaDeS_MOSS_sheet",
                                            "save": True, "embed_workflow": True, "blind": False, "blind_seed": 0}, MUTED),
        **telemetry(("load", 3), ("enh", 6), ("latent", 3), ("cmp", 2)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Compare Models\n\n"
                                  "1. MOSS Compare Sampler (on): SFT (A) against Base (B) with the same prompt, latent, seed and settings,\n"
                                  "   saved as one sheet. Plug up to four models into model_A..model_D. Each column is labelled from what\n"
                                  "   the model carries (its file, merge or LoRAs), not from the wiring: a Multi-LoRA switched off shows the bare model.\n"
                                  "   A MOSS Turbo LoRA on the turbo input applies to every column. blind shuffles and numbers the columns;\n"
                                  "   the key goes to info and the PNG, never onto the sheet.\n\n"
                                  "2. MOSS Compare Sheet (muted, right-click > Mode > Always to use): puts any images side by side,\n"
                                  "   here two seeds of one model, each labelled from its sampler's info or from label_A..label_D."}),
        "_layout": {"note": (0, -560), "load": (0, 0), "base": (0, 380), "enh": (450, 0), "latent": (950, 0), "cmp": (1400, 0),
                    "tel": (1850, 0), "report": (2300, 0), "ks_a": (950, 900), "ks_b": (1400, 900), "sheet": (1850, 900)},
        "_groups": [("1. Compare Sampler", ["load", "base", "enh", "latent", "cmp", "tel", "report"], "#335533"),
                    ("2. Compare Sheet (turn on to use)", ["ks_a", "ks_b", "sheet"], "#553355")],
    },
    "HaDeS MOSS Suite - Turbo": {
        **loader(),
        # the turbo LoRA and the settings it was trained at (4 steps, CFG 1, euler / simple), applied by the KSampler
        "turbo": ("HaDeSMossTurbo", {"lora_name": TURBO, "strength": 1.0, "steps": 4}),
        "enh": prompt("Photograph of a hand-painted wooden sign outside a bakery that reads \"FRESH BREAD DAILY\" "
                      "in large white letters and \"open 7am - 3pm\" in smaller letters below, morning light."),
        "latent": latent(),
        "ks": with_turbo(ksampler(("load", 0), ("latent", 0), seed=("latent", 4))),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_turbo_%seed%", "hashes": True, "embed_workflow": True}),
        # the settings the KSampler actually sampled with (the turbo's, not its own widgets), for other image savers
        "used": ("HaDeSMossSamplerMetadata", {"info": ("ks", 2)}),
        "used_text": ("PreviewAny", {"source": ("used", 8)}),
        **telemetry(("load", 3), ("enh", 6), ("latent", 3), ("ks", 2)),
    },
    "HaDeS MOSS Suite - Turbo 8 steps + Upscale x2": {
        # the second video tutorial (tools/moss_video.py upscale_telemetry): the first video's fisherman, then two ways to 2048.
        # Base + Turbo 8-steps v1.0; times and 100 % crops in tools/RESULTS.md, "The same pass on Base with Turbo 8-steps v1.0"
        **loader(BASE),
        "enh": prompt("Photograph of an old fisherman with a grey beard mending a net on a wooden dock at sunrise, his face and weathered "
                      "hands in view, thick wool sweater, seagulls in the background."),
        "latent": latent(seed=1013),
        "turbo": ("HaDeSMossTurbo", {"lora_name": TURBO_V10, "strength": 1.0, "steps": 8}),
        "ks": with_turbo(ksampler(("load", 0), ("latent", 0), seed=("latent", 4)), sampler_name="euler"),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_1024_%seed%", "hashes": True, "embed_workflow": True}),
        # 2. upscale x2 with a model: pixels only, nothing redrawn (about 1 s at 2048)
        "upmodel": ("UpscaleModelLoader", {"model_name": UPSCALER}),
        "up": ("ImageUpscaleWithModel", {"upscale_model": ("upmodel", 0), "image": ("ks", 1)}),
        "save_up": ("HaDeSMossImageSaver", {"images": ("up", 0), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_x2_%seed%", "hashes": True, "embed_workflow": True}),
        # 3. the MOSS pass on the upscaled image: the same turbo at denoise 0.25 redraws skin pores, beard hairs and the net's threads
        #    the upscaler smooths, about 11 s at 2048 (SFT + Turbo v0.7 measured earlier: the same by eye as SFT at 25 steps, 75 s)
        "enc": ("VAEEncode", {"pixels": ("up", 0), "vae": ("load", 2)}),
        "ks2": with_turbo(ksampler(("load", 0), ("enc", 0), seed=("latent", 4)), sampler_name="euler", denoise=0.25),
        "save_ref": ("HaDeSMossImageSaver", {"images": ("ks2", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_x2_refined_%seed%", "hashes": True, "embed_workflow": True}),
        **telemetry(("load", 3), ("enh", 6), ("latent", 3), ("ks", 2), ("turbo", 1), ("ks2", 2)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Turbo 8 steps + Upscale x2 (the second video tutorial)\n\n"
                                  "1. Generate: MOSS-Image Base with HaDeS Turbo 8-steps v1.0, 8 steps, CFG 1, euler / simple, 1024 px.\n"
                                  "   The first video's fisherman (seed 1013). The MOSS Latent's seed feeds both samplers\n"
                                  "   (fixed: set its after run to randomize for new images).\n\n"
                                  "2. Upscale x2: RealESRGAN x2 doubles the image to 2048 in about a second. Nothing is redrawn.\n\n"
                                  "3. Upscale x2 + MOSS pass: VAE Encode and a second MOSS KSampler with the same turbo at denoise 0.25\n"
                                  "   redraw the fine detail the upscaler smooths (skin pores, beard hairs), about 11 s at 2048 on an RTX 5090.\n"
                                  "   Denoise 0.40 adds a little more and changes the image a little more.\n"
                                  "   To skip it: right-click the group > Set Group Nodes to Never.\n\n"
                                  "Suite Telemetry reads every info output, checks the settings against the measurements and feeds the\n"
                                  "   Image Savers' metadata. With two samplers, the PNGs' A1111 parameters are the first sampler's\n"
                                  "   (the 1024 image); connect a second Telemetry to primary_stage_metadata to pick another stage.\n\n"
                                  "Models: RealESRGAN_x2plus.pth in models/upscale_models, the turbo LoRA in models/loras.\n"
                                  "Measurements: tools/RESULTS.md in the HaDeS MOSS Suite."}),
        "_layout": {"note": (0, -620), "load": (0, 0), "enh": (440, 0), "latent": (940, 0), "ks": (1380, 0), "save": (1820, 0),
                    "turbo": (940, 450), "tel": (1380, 640), "report": (1820, 640),
                    "upmodel": (2300, 0), "up": (2300, 150), "save_up": (2620, 0),
                    "enc": (2300, 640), "ks2": (2500, 640), "save_ref": (2940, 640)},
        "_groups": [("1. Generate (Turbo 8 steps)", ["load", "turbo", "enh", "latent", "ks", "save", "tel", "report"], "#335533"),
                    ("2. Upscale x2 (model)", ["upmodel", "up", "save_up"], "#334466"),
                    ("3. Upscale x2 + MOSS pass", ["enc", "ks2", "save_ref"], "#553355")],
    },
    "HaDeS MOSS Suite - Basic Generation": {
        # the first video tutorial, built node by node: Base at 25 steps, then Turbo 8-steps v1.0 at 8 (tools/moss_video.py basic_generation)
        **loader(BASE),
        "enh": prompt("Photograph of an old fisherman with a grey beard mending a net on a wooden dock at sunrise, his face and weathered "
                      "hands in view, thick wool sweater, seagulls in the background."),
        "latent": latent(seed=1013),
        "turbo": ("HaDeSMossTurbo", {"lora_name": TURBO_V10, "strength": 1.0, "steps": 8}),
        "ks": with_turbo(ksampler(("load", 0), ("latent", 0), seed=("latent", 4)), sampler_name="euler"),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_%seed%", "hashes": True, "embed_workflow": True}),
        # Telemetry gathers the settings for the saver's A1111 parameters (prompt, steps, CFG, seed, model and LoRA hashes)
        "tel": ("HaDeSMossSuiteTelemetry", {"report_title": "HaDeS MOSS Suite Telemetry", "info_1": ("load", 3), "info_2": ("enh", 6),
                                            "info_3": ("latent", 3), "info_4": ("ks", 2)}),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Basic Generation (the first video tutorial)\n\n"
                                  "MOSS Loader: MOSS-Image Base, the Qwen3.5 9B int8 text encoder and the FLUX.2 VAE.\n"
                                  "MOSS Prompt Enhancer, 'as written': your sentence is the prompt.\n"
                                  "MOSS KSampler: 25 steps, CFG 2.5, euler / simple when the turbo input is empty.\n"
                                  "MOSS Turbo LoRA (Turbo 8-steps v1.0, trained for Base): 8 steps, CFG 1, euler / simple.\n"
                                  "   For 25 steps again: switch the turbo node off or unplug its wire.\n"
                                  "   With HaDeS MOSS v1.0 Turbo as the model, leave the turbo input empty: it is baked in.\n\n"
                                  "Suite Telemetry feeds the Image Saver's metadata: the PNG gets the prompt, the settings and\n"
                                  "   the model / LoRA hashes as A1111 parameters (the video's graph left it out).\n\n"
                                  "Models: the turbo LoRA in models/loras (huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo).\n"
                                  "Seed 1013 in MOSS Latent is the video's; set its after run to randomize for new images."}),
        "_layout": {"load": (0, 0), "enh": (440, 0), "latent": (940, 0), "ks": (1380, 0), "save": (1820, 0), "turbo": (940, 450),
                    "note": (0, 360), "tel": (1820, 450)},
    },
    "HaDeS MOSS Suite - Control (color, depth)": {
        **loader(),
        # the control adapter loads as a LoRA; ComfyUI replaces its widened video_embedder at full strength whatever this value,
        # so keep it at 1.0 and set the control's strength on the MOSS Control node
        "loras": ("HaDeSMossMultiLora", {"model": ("load", 0), "enabled": True,
                                         "loras": json.dumps([{"name": CONTROL_ADAPTER, "on": True, "strength": 1.0}])}),
        "enh": prompt("Photograph of a woman in a red raincoat walking through a narrow cobbled street at dusk, warm shop lights, light rain."),
        "latent": latent(),
        "depth": ("LoadImage", {"image": "depth_map.png"}),
        "ctrl": ("HaDeSMossControl", {"positive": ("enh", 0), "negative": ("enh", 1), "vae": ("load", 2), "latent": ("latent", 0),
                                      "depth_map": ("depth", 0), "color_mode": "off (depth only)", "colors": 8, "blur": 4.0,
                                      "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}),
        "maps": ("PreviewImage", {"images": ("ctrl", 2)}),
        "ks": ksampler(("loras", 0), ("latent", 0), seed=("latent", 4), enh="ctrl"),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_control_%seed%",
                                         "hashes": True, "embed_workflow": True}),
        **telemetry(("load", 3), ("enh", 6), ("loras", 1), ("latent", 3), ("ctrl", 3), ("ks", 2)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - Control (color / depth)\n\n"
                                  "1. Load a control adapter in the Multi-LoRA (e.g. moss-color-depth-control from\n"
                                  "   darask0/MOSS-Image-color-depth-control, trained on MOSS-Image Base). Keep it at 1.0:\n"
                                  "   ComfyUI replaces its widened video_embedder at full strength anyway.\n"
                                  "2. Load a depth map (near is white), e.g. from Depth Anything (comfyui_controlnet_aux).\n"
                                  "   MOSS Control resizes it in pixels to the latent before the VAE.\n"
                                  "3. color_mode: off = depth only (clothes and colors from the prompt);\n"
                                  "   flat regions = connect a photo to color_map, reduced to a few flat colors for the palette;\n"
                                  "   as given = a real segmentation map (a photo there makes the adapter copy its texture).\n"
                                  "4. strength 1.0 is the trained value; end_percent below 1.0 loosens the shapes at the end.\n\n"
                                  "Not measured yet: the adapter on SFT (this example) against Base, and with the turbo LoRAs."}),
        "_layout": {"note": (0, -560), "load": (0, 0), "loras": (0, 380), "enh": (450, 0), "latent": (950, 0), "depth": (950, 420),
                    "ctrl": (1400, 0), "maps": (1400, 620), "ks": (1850, 0), "save": (2300, 0), "tel": (2300, 480), "report": (2750, 480)},
    },
    "HaDeS MOSS Suite - LoRA Toolkit": {
        # the fourth video tutorial (tools/moss_video.py lora_toolkit): Bulle d'Encre's two LoRAs extracted, stacked, split by layers,
        # merged, resized and read; measurements in tools/RESULTS.md, "The LoRA toolkit on Bulle d'Encre"
        **loader(BASE),
        # what a file says about itself (header only): Bulle d'Encre's recipe; later, the recipe of the LoRAs made here
        "meta": ("HaDeSMossMetadataReader", {"lora_name": "(none)", "model_name": BULLE, "sha256": False}),
        "meta_text": ("PreviewAny", {"source": ("meta", 0)}),
        # Bulle d'Encre's base, rebuilt: SFT with 30 % of Base in layers 24-29 only
        "sft": ("UNETLoader", {"unet_name": SFT, "weight_dtype": "default"}),
        "base": ("UNETLoader", {"unet_name": BASE, "weight_dtype": "default"}),
        "bm": ("HaDeSMossModelBlockMerger", {"model1": ("sft", 0), "model2": ("base", 0), "layers_00_05": 0.0, "layers_06_11": 0.0,
                                             "layers_12_17": 0.0, "layers_18_23": 0.0, "layers_24_29": 0.3, "layers_30_35": 0.0,
                                             "layers_36_41": 0.0, "text_embedder": 0.0, "time_modulation": 0.0, "io": 0.0}),
        # 1. (muted) Bulle d'Encre minus its base, saved as models/loras/bulle_extracted.safetensors (252 MB, about 100 s)
        "bulle": ("UNETLoader", {"unet_name": BULLE, "weight_dtype": "default"}, MUTED),
        "extract": ("HaDeSMossLoraExtractor", {"model_tuned": ("bulle", 0), "model_base": ("bm", 0), "rank": 64,
                                               "export_filename": "bulle_extracted", "overwrite_existing": False, "tuned_name": "Bulle d'Encre",
                                               "base_name": "SFT + 30% Base in layers 24-29 (Block Merger)"}, MUTED),
        "extract_text": ("PreviewAny", {"source": ("extract", 1)}, MUTED),
        # 2. the base + the extracted LoRA (switch the Multi-LoRA on once the file exists); the Turbo row at 0.5 for 8 steps / CFG 1
        "loras": ("HaDeSMossMultiLora", {"model": ("bm", 0), "enabled": False, "loras": json.dumps(
            [{"name": "bulle_extracted.safetensors", "on": True, "strength": 1.0}, {"name": TURBO_V10, "on": False, "strength": 0.5}])}),
        "enh": prompt("Photograph of an old fisherman with a grey beard mending a net on a wooden dock at sunrise, his face and weathered "
                      "hands in view, thick wool sweater, seagulls in the background."),
        "latent": latent(seed=1013),
        "ks": (lambda k: (k[0], {**k[1], "steps": 15, "sampler_name": "euler"}))(ksampler(("loras", 0), ("latent", 0), seed=("latent", 4))),
        "save": ("HaDeSMossImageSaver", {"images": ("ks", 1), "metadata": ("tel", 1), "filename_prefix": "MOSS/HaDeS_MOSS_lora_%seed%", "hashes": True, "embed_workflow": True}),
        # 3. (muted) the two rows in one file, then a smaller copy
        "lmerge": ("HaDeSMossLoraMerger", {"model": ("bm", 0), "lora_a": "bulle_extracted.safetensors", "weight_a": 1.0, "lora_b": TURBO_V10,
                                           "weight_b": 0.5, "rank": 72, "export_filename": "bulle_turbo_merged", "dynamic_energy": 0.0,
                                           "overwrite_existing": False}, MUTED),
        "resize": ("HaDeSMossLoraResize", {"model": ("bm", 0), "lora_name": "bulle_turbo_merged.safetensors", "target_rank": 32,
                                           "export_filename": "bulle_turbo_r32", "dynamic_energy": 0.0, "overwrite_existing": False}, MUTED),
        # one LoRA with a strength per group of 6 layers (muted: the Multi-LoRA rows do the same, folded under each row)
        "lbw": ("HaDeSMossLoraBlockWeight", {"model": ("bm", 0), "lora_name": TURBO_V10, "strength_model": 0.5}, MUTED),
        **telemetry(("load", 3), ("enh", 6), ("loras", 1), ("latent", 3), ("ks", 2), ("bm", 1)),
        "note": ("Note", {"text": "HaDeS MOSS Suite - LoRA Toolkit (the fourth video tutorial)\n\n"
                                  "Bulle d'Encre (huggingface.co/DeVilDoNia/Bulle-d-Encre) is SFT with 30% of Base in layers 24-29, plus\n"
                                  "Turbo 8-steps v1.0 at 0.5 and the author's style LoRA at 0.8. Its recipe shows in the Metadata Reader.\n\n"
                                  "As it opens, a run gives the rebuilt base alone (the Block Merger) at 15 steps, CFG 2.5, euler / simple.\n"
                                  "1. Unmute the Load Diffusion Model (Bulle d'Encre), the LoRA Extractor and its text preview (Ctrl+M)\n"
                                  "   and run: models/loras/bulle_extracted.safetensors, rank 64, 252 MB, 97% of the difference, about 100 s.\n"
                                  "   Then mute them again and press R so the lists see the new file.\n"
                                  "2. Switch the Multi-LoRA on: the base + the extracted LoRA gives Bulle d'Encre's look back.\n"
                                  "   Turn the Turbo row on (0.5) and set the KSampler to 8 steps, CFG 1: the same scene and look.\n"
                                  "   Each row folds out a weight per group of 6 layers: here layers 24-35 carry most of the style.\n"
                                  "3. Unmute the LoRA Merger and run: one file, 255 MB, the same maths as the two rows within 0.24%.\n"
                                  "   Then the LoRA Resize at rank 32: 126 MB, within the noise of the merged file's images.\n"
                                  "4. Pick bulle_turbo_r32 in the Metadata Reader: the whole chain back to Bulle d'Encre.\n\n"
                                  "Files are saved at the root of models/loras. Measurements: tools/RESULTS.md in the HaDeS MOSS Suite."}),
        "_layout": {"note": (0, -700), "meta": (0, 0), "meta_text": (420, 0), "sft": (0, 520), "base": (0, 660), "bm": (320, 520),
                    "extract": (760, 520), "bulle": (760, 820), "extract_text": (1180, 520), "load": (0, 1160), "enh": (440, 1160),
                    "latent": (940, 1160), "loras": (1380, 1160), "ks": (1820, 1160), "save": (2260, 1160), "tel": (1820, 1700),
                    "report": (2260, 1700), "lmerge": (0, 1900), "resize": (440, 1900), "lbw": (880, 1900)},
        "_groups": [("1. Extract (muted)", ["bulle", "extract", "extract_text"], "#553355"),
                    ("2. Generate with the LoRAs", ["load", "enh", "latent", "loras", "ks", "save"], "#335533"),
                    ("3. Merge and resize (muted)", ["lmerge", "resize"], "#334466")],
    },
}

# the fifth video tutorial (tools/moss_video.py metadata): the second video's workflow, the refined image's Saver told which stage
# made it through a second Telemetry on primary_stage_metadata; measurements in tools/RESULTS.md, "What the MOSS Image Saver writes"
_upscale = EXAMPLES["HaDeS MOSS Suite - Turbo 8 steps + Upscale x2"]
EXAMPLES["HaDeS MOSS Suite - Metadata"] = {
    **{k: v for k, v in _upscale.items() if k not in ("note", "_layout", "_groups")},
    # the refine stage alone: its Loader, Prompt Enhancer, KSampler and Turbo
    "tel2": ("HaDeSMossSuiteTelemetry", {"report_title": "Refine stage", "info_1": ("load", 3), "info_2": ("enh", 6), "info_3": ("ks2", 2),
                                         "info_4": ("turbo", 1)}),
    "save_ref": (_upscale["save_ref"][0], {**_upscale["save_ref"][1], "primary_stage_metadata": ("tel2", 1)}),
    # the generation stage alone, for the 1024 image and its plain x2 upscale
    "tel1": ("HaDeSMossSuiteTelemetry", {"report_title": "Generate stage", "info_1": ("load", 3), "info_2": ("enh", 6), "info_3": ("latent", 3),
                                         "info_4": ("ks", 2), "info_5": ("turbo", 1)}),
    "save": (_upscale["save"][0], {**_upscale["save"][1], "primary_stage_metadata": ("tel1", 1)}),
    "save_up": (_upscale["save_up"][0], {**_upscale["save_up"][1], "primary_stage_metadata": ("tel1", 1)}),
    "note": ("Note", {"text": "HaDeS MOSS Suite - Metadata (the fifth video tutorial)\n\n"
                              "Every MOSS Image Saver gets Suite Telemetry's json on metadata: the PNG carries the prompt, the settings,\n"
                              "the model and every LoRA with their hashes as A1111 parameters (what Civitai and image browsers read), the\n"
                              "Suite's own copy of every node's settings (hades_moss) and ComfyUI's workflow (drop the PNG into ComfyUI).\n\n"
                              "Two samplers: with Telemetry on both stages the A1111 parameters are the first sampler's (1024, no denoise)\n"
                              "and the Saver says so. The refined image's Saver also gets a second Telemetry (Refine stage) on\n"
                              "primary_stage_metadata, fed only by that stage: its PNG says 2048x2048, Denoising strength 0.25, and still\n"
                              "keeps the whole pipeline. The 1024 and x2 Savers get a Generate stage Telemetry the same way.\n\n"
                              "Model: the file the KSampler's model comes from, with its hash; a merge made in the workflow is written\n"
                              "'merge: a + b' without a model hash (Suite 0.9.3).\n"
                              "Privacy: hashes off removes the hashes, embed_workflow off removes the workflow; ComfyUI's\n"
                              "--disable-metadata removes every text from the PNG.\n"
                              "File names: %model%, %seed%, %steps%, %cfg%, %sampler%, %width%, %height%, %date:...%.\n\n"
                              "The first save with hashes reads the model once (about 6 s for 10 GB here); later saves reuse it.\n"
                              "Measurements: tools/RESULTS.md in the HaDeS MOSS Suite."}),
    "_layout": {**_upscale["_layout"], "tel2": (2500, 1200), "tel1": (1820, 1200)},
    "_groups": [g if not g[0].startswith("3.") else (g[0], g[1] + ["tel2"], g[2]) for g in _upscale["_groups"]],
}

DEV_EXAMPLE_TITLES = {"HaDeS MOSS Suite - Dev Tools", "HaDeS MOSS Suite - Stage Metadata (Dev)"}


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
        heights[name] = {"HaDeSMossLayoutCompiler": 560, "HaDeSMossLayoutEditor": 700, "HaDeSMossPromptEnhancer": 680,
                         "HaDeSMossMultiLora": 300}.get(cls, 190 + 30 * (len(widgets) + len(node_inputs)))
        rows[col] = y + heights[name]
        properties = {"Node name for S&R": cls}
        models = [{"name": w, "url": MODEL_FILES[w][1], "directory": MODEL_FILES[w][0]} for w in widgets if w in MODEL_FILES]
        if models:
            properties["models"] = models
        nodes.append({"id": ids[name], "type": cls, "pos": list(layout.get(name, (col * 500, y))), "size": [{"HaDeSMossLayoutEditor": 520, "HaDeSMossPromptEnhancer": 440}.get(cls, 380), 100], "flags": {}, "order": 0, "mode": mode[0] if mode else 0,
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
    """The runnable API prompt: muted nodes left out, with them the nodes that need one of their outputs (a link from a muted node
    into an optional input is dropped instead, as ComfyUI does), local file names substituted."""
    graph = {k: v for k, v in example.items() if not k.startswith("_") and v[0] != "Note"}
    optional = {n: {name for name, _, _, opt in input_specs(object_info[cls]) if opt} for n, (cls, *_) in graph.items()}
    live = {n for n, (_, _, *mode) in graph.items() if not mode}
    changed = True
    while changed:
        changed = False
        for n in list(live):
            if any(isinstance(v, tuple) and v[0] not in live and k not in optional[n] for k, v in graph[n][1].items()):
                live.discard(n)
                changed = True
    prompt = {}
    for n in live:
        cls, inputs = graph[n][0], graph[n][1]
        resolved = {}
        for k, v in inputs.items():
            if isinstance(v, tuple):
                if v[0] in live:
                    resolved[k] = [v[0], v[1]]
            elif v is None:
                resolved[k] = next(widget_default(kind, o) for name, kind, o, _ in input_specs(object_info[cls]) if name == k)
            else:
                if isinstance(v, str):  # file names, also inside a Multi-LoRA's JSON list
                    v = mapping.get(v, v)
                    for official, local in mapping.items():
                        v = v.replace(json.dumps(official)[1:-1], json.dumps(local)[1:-1]) if k == "loras" else v
                resolved[k] = v
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
        directory = os.path.join(EXAMPLES_DIR, "dev") if title in DEV_EXAMPLE_TITLES else EXAMPLES_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{title}.json")
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
