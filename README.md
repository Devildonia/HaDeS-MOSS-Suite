# HaDeS MOSS Suite

ComfyUI nodes for the **MOSS-Image** model family (OpenMOSS / Comfy Org: 42-layer single-stream DiT, `Qwen3.5-9B` text encoder, FLUX.2 VAE), in the style of the [HaDeS Anima Suite](https://github.com/Devildonia/HaDeS-Anima-Suite).

Claims about what layer groups do are measured with [`tools/moss_eval.py`](https://github.com/Devildonia/HaDeS-MOSS-Suite/blob/main/tools/moss_eval.py), not assumed.

**Turbo (0.2.0, Turbo 8-steps in 0.3.0):** few-step LoRAs we trained for MOSS-Image SFT, [HaDeS MOSS Turbo and Turbo Diverse](https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo) (4 steps at CFG 1, 3.4 MB each), joined by Turbo 8-steps v0.7 (8 steps), and a `turbo` input on `MOSS KSampler` that applies them with their settings. How they were trained and measured: [`tools/RESULTS.md`](tools/RESULTS.md), [`tools/moss_dmd_train.py`](https://github.com/Devildonia/HaDeS-MOSS-Suite/blob/main/tools/moss_dmd_train.py).

**New in 0.4.0:** `MOSS Loader` (model, text encoder and VAE in one node), `MOSS Multi-LoRA` (any number of LoRAs, searchable picker, strength per layer group), `MOSS Prompt Enhancer` (writes a caption or a layout with MOSS-Image's own text encoder), `MOSS Layout Editor` (boxes edited on a canvas), a shared seed on `MOSS Latent` and stochastic rounding in `MOSS Model Saver`. **0.4.1:** `MOSS Image Saver` (PNG with A1111 metadata and hashes for Civitai). **0.5.0:** `MOSS Metadata Reader` and model / LoRA recipes. **0.6.0:** `MOSS Control (color / depth)`. **0.6.1:** edit mode in `MOSS Prompt Enhancer` for MOSS-Image edit LoRAs. **0.7.0:** Dev Tools (`MOSS Layout Regions`, `MOSS Image Audit`) and `tools/moss_edit_eval.py`. **0.7.1:** `MOSS Image Saver` multi-stage metadata (`primary_stage_metadata`) and native preview widget fix. **0.7.2:** Dev Tools additions (`MOSS Pair Guard`, `MOSS Recipe Preview`) and model-free `examples/dev/` organization.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Devildonia/HaDeS-MOSS-Suite
```

No extra dependencies. Needs a ComfyUI with MOSS-Image support (tested on 0.43.0) and the MOSS-Image weights from `Comfy-Org/MOSS-Image`.

## Nodes

### `MOSS Loader` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossLoader`
- **Purpose:** the three MOSS-Image files in one node, loaded as the core Load Diffusion Model, Load CLIP and Load VAE do, with the text encoder always as type `moss_image`. Stops with a clear error when the diffusion model is not MOSS-Image, the text encoder does not load as MOSS-Image's Qwen3.5-9B or the VAE is not the 128-channel FLUX.2 one.
- **Inputs:** `unet_name`, `clip_name`, `vae_name` (a new node starts on the MOSS files it finds: an SFT model, Qwen3.5-9B, flux2-vae); advanced `weight_dtype` and `clip_device` (as in the core loaders).
- **Outputs:** `MODEL`, `CLIP`, `VAE`, `info` (file names, number of layers, the precision the weights have in memory, e.g. bf16 or int8 with its layout, and the attention the file sets, e.g. comfy-kitchen int8 on Comfy-Org's int8 files, or the server's).
- **Telemetry:** with `info` connected, Suite Telemetry names the model, text encoder and VAE, notes an int8 model with its measured cost on an RTX 5090 (57% slower than bf16 at 1024, 49% at 2048, for about half the VRAM) and a `weight_dtype` conversion, which was not measured.

### `MOSS Model Block Merger` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossModelBlockMerger`
- **Purpose:** merges two MOSS-Image models (e.g. SFT and Base) with one ratio per group of 6 layers and for the parts outside the layers. `0.0` = Model 1, `1.0` = Model 2; values outside `0..1` extrapolate.
- **Inputs:**
  - `model1`, `model2` (`MODEL`) with the same q/k/v layout. Quantized models (int8 ConvRot, fp8) are dequantized and merged with a warning in `info`: the quantization rounding stays in the result, and a quantized `model1` makes the result quantized again, so connect the bf16 model as `model1`. The one int8 finetune measured differs from SFT by about as much as the int8 rounding itself ([`tools/RESULTS.md`](tools/RESULTS.md)).
  - `layers_00_05` … `layers_36_41` (`FLOAT`, default `0.5`).
  - `text_embedder` (projection of the Qwen3.5 hidden states and its 2 text blocks), `time_modulation` (`time_embedding` + `shared_block_modulation`), `io` (`video_embedder` + output head).
  - `recipe_override` (`STRING`, optional): one ratio per layer (42), then `text_embedder`, `time_modulation`, `io`, comma-separated.
- **fp16-fixed files:** some fp16 releases store the residual stream scaled down (measured: x 1/16 on `video_embedder`, the text embedder outputs and the modulation gates), which is the same model in another scale. The merger detects the scale and brings `model2` to `model1`'s before merging, and says so in `info`.
- **Outputs:** `MODEL`, `info`.
- **Memory:** the merge is a lazy patch computed when the model loads; no merged copy is kept in RAM.
- **Recipe:** `info` keeps every layer / part ratio and the source recipes known to the Suite. Multi-Model Merger also records its weights and seed. Sources loaded outside the Suite are unidentified; file names are not guessed.

### `MOSS Multi-Model Merger` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossMultiModelMerger`
- **Purpose:** merges 2–4 MOSS-Image models with the community's standard methods. Every method except SLERP works on task vectors (`model − base_model`):
  - `TIES`: keeps each task vector's largest entries (`density`), elects a sign per entry from their sum and averages the agreeing ones.
  - `DARE` / `DARE-TIES`: drops entries at random with keep probability `density` and rescales the rest, then averages (`DARE`) or elects signs like TIES (`DARE-TIES`). `seed` fixes the drop mask.
  - `Task Arithmetic`: weighted average of the task vectors.
  - `Model Stock` (Jang et al. 2024): the average of the models pulled toward `base_model`, per tensor, by the angle between their task vectors; it ignores weights, `density` and `scale`.
  - `SLERP`: spherical blend of `model_1` and `model_2` only, `t = weight_2 / (weight_1 + weight_2)`; needs no `base_model`.
- **Inputs:** `model_1`, `model_2` (`MODEL`), `model_3`, `model_4`, `base_model` (optional; required by every method except SLERP), `weight_1`…`weight_4` (`0..2`, normalized by their sum), `density` (`0.05..1`, default `0.5`), `scale` (`0..2`, default `1.0`; `1.0` has the strength of an average), `seed`.
- **base_model:** the checkpoint the models were finetuned from. Community MOSS finetunes come from SFT or from Base; when they mix, use Base, which SFT was trained from.
- **Inputs it accepts:** like the Block Merger, quantized models are dequantized (with a warning; with TIES / DARE a quantized model's task vector carries its rounding), and fp16-fixed files are brought to the anchor's residual scale (`base_model`, or `model_1` for SLERP).
- **Not measured:** which method gives better images.

### `MOSS LoRA Extractor` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossLoraExtractor`
- **Purpose:** saves `model_tuned − model_base` as a LoRA in `models/loras/`: truncated SVD per linear layer (`lora_up` / `lora_down` / `alpha`), full `.diff` / `.diff_b` for norms and biases, identical tensors skipped. Keys are `diffusion_model.*` with fused `proj.qkv`, which the stock LoRA loaders apply to the Comfy-Org files.
- **Inputs:** `model_tuned`, `model_base` (`MODEL`; merge / LoRA patches on either are included), `rank` (`4..256`, default `32`), `export_filename`, `overwrite_existing`, optional `tuned_name` and `base_name` (written into the LoRA as `hades/model_tuned` / `hades/model_base`, so the Metadata Reader can say what it was extracted from).
- **Inputs it accepts:** quantized models are dequantized (with a warning: the difference includes their rounding), and an fp16-fixed `model_tuned` is brought to `model_base`'s residual scale first.
- **Outputs:** `saved_lora_path`, `info` (size, energy kept: the share of the difference the rank holds).
- **Measured:** `galenaCAT − SFT` at rank 64 (252 MB) keeps 95.9% of the energy and renders 4–12 from the finetune on SFT; a rank-32 LoRA of an int8 finetune reproduced it within the int8 noise on three of four images. See [`tools/RESULTS.md`](tools/RESULTS.md).

### `MOSS LoRA Block Weight` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossLoraBlockWeight`
- **Purpose:** applies one LoRA with a separate strength per group of 6 layers (`layers_00_05` … `layers_36_41`) and `other_layers` for LoRA weights outside the layers. Loads every community MOSS format through ComfyUI's own key map (kohya, PEFT split or fused q/k/v); at 1.0 everywhere it renders the same image as the stock `LoraLoaderModelOnly`.
- **Inputs:** `model`, `lora_name`, `strength_model`, the seven group multipliers and `other_layers` (`-2..3`, default `1.0`).
- **Outputs:** `MODEL`, `info` (how many LoRA weights fall in each group, e.g. 20 instead of 24 in `layers_18_23` for LoRAs trained without layer 22).
- **Measured:** layers 30–35 carry a large share of all four community LoRAs tested, but beyond that each acts in different layers (24–35, 30–41 or 00–05 and 30–35), so the node has no style / identity presets; `tools/moss_eval.py --mode lora` measures a given LoRA. See [`tools/RESULTS.md`](tools/RESULTS.md).

### `MOSS Multi-LoRA` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossMultiLora`
- **Purpose:** stacks any number of LoRAs in one node. Each row has its own on / off, strength and, folded under the row, a weight per group of 6 layers and `other_layers` that multiplies its strength, as in LoRA Block Weight (same code). A row at 1.0 with every group at 1.0 is the stock LoRA loader.
- **Inputs:** `model`, `enabled` (off passes the model through and keeps the list), `loras` (the list as JSON, edited by the rows: `[{name, on, strength, groups}]`).
- **Outputs:** `MODEL`, `info` (the LoRAs applied with their strengths and changed groups).
- **Few-step LoRAs:** a LoRA whose own file metadata gives its settings at CFG 1 (the HaDeS MOSS Turbo files: `4 steps, CFG 1, euler / simple, strength 1.0, shift 5`, `8 steps` for Turbo 8-steps) gets a STEPS tag on its row and is named in `info`; nothing is guessed from file names, so other few-step LoRAs carry no tag. Two of them stacked give a warning: each was trained alone. Suite Telemetry compares those settings with the KSampler's.
- **Interface:** **+ ADD LORA** and a click on a row's name open a LoRA picker with a search box (words in any order, arrow keys and Enter); strength and group weights drag, take a typed value on click and go back to 1.0 on right-click; right-click on the group button resets the row's groups.
- **No presets:** which layers a LoRA acts in differs from LoRA to LoRA (see LoRA Block Weight), so the node ships no group presets.

### `MOSS LoRA Merger` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossLoraMerger`
- **Purpose:** merges two LoRAs into one file, `weight_a × A + weight_b × B`, compressed per layer with an exact SVD of the stacked factors (QR first, never the full matrix). Any community MOSS format in (kohya, PEFT split or fused q/k/v), the model's layout out.
- **Inputs:** `model` (the model the LoRAs are for; it sets the q/k/v layout), `lora_a`, `weight_a`, `lora_b`, `weight_b`, `rank` (max per layer; rank A + rank B is exact), `export_filename`, optional `dynamic_energy` and `overwrite_existing`. Never overwrites a source LoRA.
- **Outputs:** `saved_lora_path`, `info` (layers in A only / B only / both, size, rank, energy kept).
- **Measured:** retro (rank 32) + lenovo (rank 16) merged at rank 48 or 32 renders within the noise of loading the two LoRAs in the opposite order ([`tools/RESULTS.md`](tools/RESULTS.md)).

### `MOSS LoRA Resize` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossLoraResize`
- **Purpose:** lowers a LoRA's rank with the same exact SVD, fixed (`target_rank`) or per layer by an energy target (`dynamic_energy`), and saves a new file in `models/loras/`; the source is never overwritten. Split q/k/v LoRAs come out fused on fused models.
- **Inputs:** `model`, `lora_name`, `target_rank`, `export_filename`, optional `dynamic_energy`, `overwrite_existing`.
- **Outputs:** `saved_lora_path`, `info` (sizes, rank, mean and worst-layer energy kept).
- **Measured:** AnubisCitronOC (142 MB) at rank 32 (126 MB, 97.9% energy) stays within the noise floor; rank 16 (63 MB, 91.1%) and dynamic 0.90 (56 MB) drift visibly from the source.

### `MOSS Metadata Reader` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossMetadataReader`
- **Purpose:** shows what a LoRA or diffusion model file says about itself and what its tensors show, reading only the safetensors header (instant on a 10 GB model; the weights are not loaded).
  - **Metadata:** every key of the file (the Suite's `hades/*`, kohya / ai-toolkit `ss_*`, `modelspec.*`, a saved ComfyUI workflow...); long values are cut in `info` and whole in `metadata_json`.
  - **Measured from the tensors:** size, tensor count and dtypes; for a LoRA its kind (lora_up / lora_down, PEFT, full diffs, LoHa, LoKr), rank (per layer when it varies), alpha, the MOSS-Image layers and modules it touches, or a note when no key name is MOSS-Image's; for a model whether it is MOSS-Image, its layers, parameters, fused or split q/k/v, int8 ConvRot.
  - **Recipe:** for LoRAs made with the Suite's Merger, Resize or Extractor, the chain of sources from their `hades/*` keys, followed into the source files still in `models/loras` (a source moved to another subfolder is found by its file name when only one LoRA has it), with the weight of each original file in the result. For models saved by the MOSS Model Saver with a `recipe`, that text, with the model of a MOSS Loader info and the LoRAs of a Multi-LoRA or LoRA Block Weight info in it followed into their own recipes. A source replaced since under the same name is read as it is now.
- **Inputs:** `lora_name` and `model_name` (each can be `(none)`), optional `sha256` (hashes the files, cached while unchanged and shared with the MOSS Image Saver; seconds on a 10 GB model).
- **Outputs:** `info` (text plus HADES_META with the recipe; into a Save Text node it keeps the recipe on disk), `metadata_json`.
- Nested merger recipes are followed too. The reader refreshes when referenced source files change or become available.

### `MOSS Model Saver` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossModelSaver`
- **Purpose:** saves a `MODEL` with its merge / LoRA patches baked in, loadable by the stock `Load Diffusion Model` node, and passes the model through.
- **Inputs:**
  - `filename_prefix`: path under `ComfyUI/output` (default `diffusion_models/MOSS_HaDeS_Merged`). Move the file to `models/diffusion_models` to load it.
  - `precision`: `bfloat16` (10.2 GB), `int8 ConvRot (Comfy-Org layout)` (5.4 GB: the four linears of every layer in int8 with Hadamard rotation and int8 attention, everything else in bfloat16, the same keys and dtypes as Comfy-Org's `*_int8_convrot` files) or `float32`.
  - `include_metadata`, `custom_author`, `custom_model_title`: ModelSpec metadata.
  - `recipe` (optional): how the model was made, written into the file as `hades/recipe`. Link the `info` of the MOSS Loader, Multi-LoRA or a merger (several joined with the core Concatenate node) or type notes; the MOSS Metadata Reader shows it and follows the models and LoRAs it names. Empty: no recipe, as before.
  - `rounding` (`bfloat16` output, weights with merge / LoRA patches): `stochastic` (default) rounds each one up or down with probability by distance, seeded per weight so the file is reproducible; `nearest` is ComfyUI's own saving. Unpatched weights are copied unchanged either way.
- **Outputs:** `model` (pass-through), `saved_path`, `info`.
- **Measured** ([`tools/RESULTS.md`](tools/RESULTS.md), "Baking subtle LoRAs into bfloat16"): two LoRAs whose changes are smaller than a bfloat16 step, baked into a merge. Rounded to nearest the file kept about 80–85% of their effect (as the LoRAs loaded live, which round the same way); stochastic rounding kept all of it, and its renders sat as close to the float32 result as two stochastic files with different seeds sit to each other.
- **Measured:** a live merge renders slightly differently from the same merge saved and reloaded (same subject and composition, small details change), in ComfyUI's own merger too; see [`tools/RESULTS.md`](tools/RESULTS.md).

### `MOSS Control (color / depth)` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossControl`
- **Purpose:** color / depth control with a MOSS-Image control adapter loaded as a LoRA (e.g. `moss-color-depth-control`, trained on Base, which widens `video_embedder` to 384 input channels). Takes the Prompt Enhancer's positive and negative, resizes the maps in pixels to the latent before the VAE (the core node encodes them at their own size and the model then resizes the latents), and adds them as `concat_latent_image` in a start / end range.
- **Inputs:** `positive`, `negative`, `vae`, `latent`; `color_mode`: `off (depth only)` (the adapter was trained depth-only 35% of the time, its file metadata: clothes and colors stay with the prompt), `flat regions` (a photo blurred and reduced to `colors` flat colors, so it guides palette and large areas without its texture), `as given` (a real segmentation map); `colors`, `blur`, `strength` (1.0 trained), `start_percent`, `end_percent`; optional `depth_map`, `color_map`.
- **Outputs:** `positive`, `negative`, `maps` (what was encoded, to check), `info` (for Telemetry and the Image Saver).
- **Note:** the adapter's strength in a LoRA loader scales only its LoRA layers: ComfyUI replaces the widened `video_embedder` at full strength whatever that value (a `set` patch), and at 0 the adapter is not applied at all, so the control conditioning then meets a 128-channel embedder. Use this node's `strength`. Not measured yet: quality on SFT against Base, and with the turbo LoRAs.

### `MOSS Latent` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossLatent`
- **Purpose:** empty MOSS-Image latent (128 channels, 16 px grid) from an aspect ratio (`1:1`, `4:3`, `3:2`, `16:9`, `21:9` and their portrait forms) and a size in megapixels (`1.0` = 1024×1024), or a custom width / height snapped to 16 px. Outputs `width` / `height` for layout prompts.
- **Inputs:** `aspect_ratio`, `megapixels` (`0.25..4.0`, default `1.0`), `custom_width`, `custom_height`, `batch_size`, `seed` (with control after generate, starting on `fixed` so the image changes only when you change it); optional `vae` (rejects a VAE that is not the 128-channel FLUX.2 one).
- **Outputs:** `latent`, `width`, `height`, `info`, `seed`.
- **Shared seed:** connect `seed` to the `seed` of the MOSS KSampler (and of a second sampler, e.g. a refine pass) and of the Prompt Enhancer, so one value set here reproduces the whole image, prompt included. The seed does not change the latent, which is empty; it is shown in the node's status line and written to `info`.
- **Measured:** square images from 768 to 2048 render clean at the model's default shift 5.0, with no repeated subjects; larger canvases show more of the scene. Raising the shift is not needed at any of these sizes. Non-square ratios are not measured. See [`tools/RESULTS.md`](tools/RESULTS.md).
- **Upscaling a MOSS latent (hi-res fix):** use the core `Upscale Latent By` (`LatentUpscaleBy`), not `Upscale Latent` (`LatentUpscale`). The latter takes a size in pixels and divides it by 8, as for SD VAEs, while the MOSS latent is 16 px per cell: asked for 1024 x 1024, it gives a 2048 x 2048 image.

### `MOSS Layout Compiler` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossLayoutCompiler`
- **Purpose:** compiles an OpenMOSS layout draft (the schema of the official `moss-image-prompt-enhancer`: `width` / `height`, `bbox_coord_space: "abs"`, `bbox_format: "xyxy"` in top-left pixels, `summary` and a flat `items` array) into the model-facing prompt (boxes centered on the canvas, `yxyx`), with the same validation as the official `compile_prompt.py` and byte-identical output. Also outputs the empty latent and `width` / `height` of the draft's own canvas, so prompt and image size cannot drift apart. A `{"caption": ...}` draft passes through as plain text.
- **Inputs:** `draft` (`STRING`, JSON), `batch_size`.
- **Outputs:** `prompt` (to `CLIPTextEncode`), `latent`, `width`, `height`, `info` (lists multi-line text items with their box size).
- **Line breaks:** write them as `\n` in `text`. Measured, the box shape decides whether they are kept: a two-line title in an 832×192 box came out on one line in 3 of 6 seeds, in a 640×384 box on two lines in 6 of 6; a literal newline instead of `\n` changed nothing ([`tools/RESULTS.md`](tools/RESULTS.md)).
- Unlike the core `MOSS-Image Layout Prompt` (boxes from `Create Bounding Boxes`), this node takes hand-written or LLM-written drafts, including `meta`, `canvas`, `palette` and `tags`.

### `MOSS Layout Editor` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossLayoutEditor`
- **Purpose:** edits a layout draft on a canvas of the draft's size: boxes colored by label (object, character, text), moved and resized with the mouse on the 16 px latent grid, drawn by dragging on an empty part of the canvas with the label picked in the toolbar, deleted or brought to the front. Under the canvas, the selected box's label, caption, text (new lines are line breaks in the image) and exact pixels; with no box selected, the caption of the whole image. Clicking the canvas size changes it and scales the boxes. Boxes missing a caption, or text items missing their text, are outlined in red, and the run stops with the Layout Compiler's error until they are filled in.
- **Inputs:** `draft` (the JSON the canvas edits, shown as text with the standard widgets); optional `draft_in` (e.g. the Prompt Enhancer's `draft`: after a run the toolbar shows **IMPORT**, which replaces the canvas with it; it never overwrites your edits by itself) and `image` (a background for placing the boxes: pick it in a `Load Image` node wired here, shown at once even when that node is muted, otherwise after a run; **MATCH IMAGE** sets the canvas to its aspect ratio at about the same area; not sent to the model).
- **Outputs:** `draft` (to the Layout Compiler's `draft`), `width`, `height`, `info`.
- **Image → layout:** the Prompt Enhancer in `layout` mode with an `image` writes the draft; import it here to correct the boxes.

### `MOSS Prompt Enhancer` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossPromptEnhancer`
- **Purpose:** a positive / negative prompt box for the sampler that can also write the prompt: a detailed MOSS-Image prompt from a short idea and / or an image, with MOSS-Image's own text encoder: the Qwen3.5-9B file loaded as the `moss_image` CLIP holds the language head and the vision tower, so the same CLIP that encodes the prompt writes it and no other model is loaded. The instructions are adapted from OpenMOSS' official [`moss-image-prompt-enhancer`](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer) skill (Apache-2.0).
- **Inputs:** `clip` (MOSS-Image's text encoder), `idea`, `mode` (`caption`: the model writes a paragraph; `layout`: it writes a pixel Layout draft with boxes for objects and text, compiled here; `as written`: no generation, the idea is encoded as typed), `width` / `height` (layout canvas), `temperature` (0 always gives the same answer), `seed`, `max_tokens`; optional `image` (the whole brief with no idea, otherwise the starting point; resized to about 1 MP) and `negative` (the negative prompt; empty uses the official MOSS-Image template's `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`; what you write is also told to the model as things to keep out).
- **Outputs:** `positive` / `negative` (conditionings for the sampler, encoded with the same CLIP), `prompt` (the text), `draft` (the editable JSON: a caption draft, or the layout draft for the Layout Compiler, which also outputs the matching latent), `width`, `height`, `info`.
- **Layout answers** are fitted before compiling: the canvas set to `width` / `height`, boxes rounded and clamped into it, fields the compiler does not know dropped, items without a box left out; `info` lists what was changed.
- **Slow text generation on older PyTorch builds:** on PyTorch 2.8.0+cu129 (RTX 5090, comfy-kitchen 0.2.40) comfy-kitchen's GQA decode kernel refuses every input ("Unsupported Flash Attention tensor strides", checked with plain contiguous tensors of several shapes), so text generation stops, in the core `TextGenerate` node too; ComfyUI also warns at start that its optimized CUDA operations need PyTorch cu130 or newer. While this node generates, a refused call runs comfy-kitchen's own torch version of the same attention: about 13 tokens per second, 30–40 s per answer; `info` says when that happened. On PyTorch 2.9.1+cu130 (same GPU, comfy-kitchen 0.2.40) the native kernel runs and an answer takes 2–3 s (three captions and a poster layout).
- **Tried, not measured** ([`tools/RESULTS.md`](tools/RESULTS.md)): one seed each, a caption and a poster layout rendered against the idea as written.

- **Edit mode** (`edit_format`, `vae`, `edit_megapixels`): for an image-edit LoRA loaded in the Multi-LoRA, the image becomes the reference latent of both outputs, replacing two core `Text Encode MOSS-Image Edit` nodes with the same formats: `Moss-Image-Edit` (`MOSS-edit` / `Moss-Image-Edit`: use `as written` with a direct edit instruction, which also runs cleanly with `HaDeS MOSS Turbo 8-steps` on the KSampler's `turbo` input, or `caption` to have the model write the full target description from the image and the idea), `ai-toolkit` and `ai-toolkit (kv_cache off)` (LoRAs trained with ai-toolkit, e.g. the pose LoRA; the text encoder also sees the image). The reference is resized to `edit_megapixels` with its aspect kept (16 px grid), and `width` / `height` out are that size: link them to the MOSS Latent's `custom_width` / `custom_height` to generate at the reference's size.

### `MOSS KSampler` (`HaDeS/MOSS Samplers`)
- **Class:** `HaDeSMossKSampler`
- **Purpose:** KSampler with the official ComfyUI MOSS-Image template settings as defaults (25 steps, CFG 2.5, `er_sde` / `simple`, the model's shift 5.0), an optional flow shift, optional skip-layer guidance and optional decode. With the same settings it renders the same image as the stock graph (`KSampler`, `ModelSamplingAuraFlow`, `SkipLayerGuidanceDiTSimple`, `VAEDecode`).
- **Inputs:** `model`, `positive`, `negative`, `latent_image`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`, `flow_shift` (`0` keeps the model's 5.0), `slg` (`off`, `block 22, first half (Base template)`, `block 22, all steps`); optional `vae`, optional `turbo` (from `MOSS Turbo LoRA`: applies the LoRA and samples with SLG off and the turbo node's steps, CFG, sampler and scheduler, whatever those widgets say).
- **Outputs:** `latent`, `image` (when a VAE is connected), `info` (the settings actually used).
- **Measured** ([`tools/RESULTS.md`](tools/RESULTS.md)): shift 5.0 is clean from 768 to 2048 px; SLG on block 22 did not change text accuracy on SFT or Base (blocks 24 / 26 made the only errors, on Base), so it is off by default; with euler, 25 steps render nearly the 60-step image with any scheduler and euler / beta does so at 15.

### `MOSS Turbo LoRA` (`HaDeS/MOSS Samplers`)
- **Class:** `HaDeSMossTurbo`
- **Purpose:** picks a few-step LoRA for the `turbo` input of `MOSS KSampler`, which loads it the way the stock `LoraLoaderModelOnly` does and samples at its own `steps`, `cfg`, `sampler_name` and `scheduler` widgets: 4 steps, CFG 1, euler / simple by default, the settings the LoRAs were trained at (seed, denoise and flow shift still come from the KSampler). Above CFG 1 the negative prompt takes part and each step runs the model twice (about 1.9x the time); measured with one seed per prompt, Turbo 8-steps at CFG 1.5 rendered 10 hard prompts on SFT and Base without burning, and on a merge with a turbo baked in 1.25–1.5 brought back prompt details lost at CFG 1 while saturation rose with every step and 2.5 looked overdone; texture-heavy prompts looked better at 1.0–1.2 ([`tools/RESULTS.md`](tools/RESULTS.md), "Turbo LoRAs above CFG 1"). Measured counts: 4 for Turbo and Turbo Diverse, 8 for Turbo 8-steps; Turbo and Turbo Diverse give 6–10% less colour than SFT at 4 steps at 1024 (6–14% at 2048) and are back near it at 8 (only colour was measured for them at 8). Other counts run, their quality is untested. Checked in ComfyUI: it renders the stock `LoraLoaderModelOnly` + `KSampler` image pixel for pixel (both LoRAs, 8 images, with the KSampler's own widgets left at 25 steps, CFG 2.5, er_sde / beta and SLG on).
- **Inputs:** `lora_name` (from `models/loras`), `strength` (default `1.0`, the only strength measured), `steps` (default `4`).
- **Outputs:** `turbo` (for the MOSS KSampler), `info` (the LoRA and settings, for Suite Telemetry).
- **The LoRAs:** HaDeS MOSS Turbo and Turbo Diverse v0.6 (for 4 steps) and Turbo 8-steps v0.7 (for 8 steps), 3.4 MB each, on [Hugging Face](https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo); put them in `models/loras`. Trained in this repository (`tools/moss_dmd_train.py`) from MOSS-Image SFT, which they are for.
- **Measured** ([`tools/RESULTS.md`](tools/RESULTS.md), 1024 x 1024 against akhaliq's 4-step v2.0 LoRA at the same settings): Turbo spelled every test string right on signs and posters at 1024 x 1024, 6 / 6 on layout JSON and 3 / 3 on a 16:9 poster, while at 2048 x 2048 the sign's small line was right in 1 / 3 (akhaliq's 2 / 3); Turbo Diverse keeps SFT's spread of faces on an open prompt (about 10 East Asian women in 16 seeds, against about 15 for Turbo and akhaliq's LoRA). Turbo 8-steps at 8 steps, in blind pairs on 20 varied prompts: beat akhaliq's 4-step LoRA (full file) 18 to 5, and against lodestones' 8-step DMD2 LoRA for MOSS-Image Base / Kromoss run on SFT (full file) 10 to 7 at 1024 and 0 to 5 at 2048, where it has less colour; it is for up to 1024. lodestones' SFT 8-step LoRA was not compared. Turbo 8-steps at denoise 0.25 / 0.40 on a RealESRGAN x2-upscaled image: by eye it adds real texture (hair, pores) the upscaler smooths, as well as SFT at 25 steps in a fifth of the time. Not measured: other base models, other strengths or step counts, other img2img uses below denoise 1.0; they run (an int8 finetune as the base, img2img at denoise 0.6 were checked), their quality is untested, and Telemetry knows which model the KSampler got only from a MOSS Loader's `info`.

### `MOSS Sampler Metadata` (`HaDeS/MOSS Samplers`)
- **Class:** `HaDeSMossSamplerMetadata`
- **Purpose:** reads the settings a MOSS KSampler actually sampled with from its `info` (seed, steps, cfg, sampler, scheduler, denoise, flow shift, and `turbo`: the turbo LoRA file and strength, empty without one), for image-saver metadata, and `info` with those values for Suite Telemetry.

### `MOSS Suite Telemetry` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossSuiteTelemetry`
- **Purpose:** reads the `info` outputs of up to 8 HaDeS MOSS nodes (their `HADES_META` trailer) and checks the pipeline against what was measured:
  - **error:** with a layout in the pipeline (Layout Editor, Layout Compiler or the Prompt Enhancer in `layout` mode), a latent or KSampler on another canvas (a layout prompt places its boxes on its own canvas); without a layout, different latent and KSampler sizes (img2img, an upscale pass) are not compared;
  - **warning:** quantized models in a merge, TIES / DARE with a quantized model, LoRAs extracted / resized / merged below 97.9% energy (the lowest measured that stayed within the noise floor), a few-step LoRA in Multi-LoRA sampled at other steps than its file's metadata gives or below CFG 1, or stacked with the KSampler's turbo input;
  - **note:** an int8 model from the MOSS Loader (with its measured speed and VRAM) or one loaded with a `weight_dtype` conversion, KSampler size outside 768²–2048², shift other than 5.0, SLG on, turbo LoRA at a strength other than 1.0, at a step count other than 4 or 8, at denoise below 1.0, above CFG 1 (with what was measured there) or with a sampler other than euler / er_sde (simple), a few-step LoRA in Multi-LoRA above CFG 1, fp16-fixed residual scale normalized, multi-line layout text with its box proportions.
- **Outputs:** `report` (Markdown), `json` (every node's settings, for the MOSS Image Saver), `summary` (one line), `info` (the counts of nodes read and findings).

### `MOSS Image Saver` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossImageSaver`
- **Purpose:** saves PNG images with the generation settings written as A1111 `parameters`, the text Civitai and most image browsers read: prompt, `Negative prompt:` and one line with steps, sampler, schedule type, CFG, seed, size, model and model hash, LoRA hashes, denoise (below 1.0), shift, VAE and text encoder. Also ComfyUI's own `prompt` / `workflow` chunks, so the image loads back into ComfyUI, and the Suite's settings as JSON (`hades_moss`).
- **Inputs:** `images`, `metadata` (Suite Telemetry's `json` output, which gathers the Loader, Latent, KSampler, Multi-LoRA and Prompt Enhancer infos; one info output connected directly also works), `filename_prefix` (`%seed%`, `%model%`, `%steps%`, `%cfg%`, `%sampler%`, `%width%`, `%height%` from the metadata, `%date:...%` as in the core Save Image), `hashes`, `embed_workflow`; optional `positive` / `negative` text when the prompt does not come from a Prompt Enhancer.
- **Multiple sampling stages:** leave `metadata` connected to Telemetry for the whole pipeline. For the image's principal stage, connect a second Telemetry only to that stage's KSampler and its relevant Loader, Prompt Enhancer and LoRAs, then connect its `json` to the optional `primary_stage_metadata` input. The stage must have exactly one KSampler and no more than one Loader or Prompt Enhancer. Its own settings and resources supply the A1111 `parameters` and filename tokens; missing resources are not borrowed from another stage. The original `hades_moss` PNG chunk still holds the full pipeline, and `hades_moss_primary_stage` holds the selected stage. With the new input disconnected, existing workflows keep the first-KSampler behavior and unchanged PNG metadata.
- **Hashes:** SHA256 of the model (from the MOSS Loader) and of every LoRA (Multi-LoRA rows, LoRA Block Weight, the KSampler's turbo input), first 10 characters (AutoV2), in `Model hash`, `Lora hashes` and `Hashes`. The first save with a 10 GB model reads the whole file once; later saves in the same session reuse the hash while the file is unchanged. Nothing is looked up online. How Civitai reads these fields was not checked from here.
- ComfyUI's `--disable-metadata` suppresses all embedded metadata. The `parameters` output remains available as text.
- **Outputs:** `parameters` (the text written into the images), `info` (the files saved and what was written into them). The node shows the saved images like the core Save Image, and a note when the prompt or the sampling settings are missing from the metadata.

### Dev Tools (git repository only)

The four `HaDeS/MOSS Dev Tools` nodes below, their UI (`web/moss_dev_tools.js`) and the `examples/dev/` workflows are for measurements: they are in the GitHub repository and not in the Comfy Registry package. Clone the repository to get them; the Suite works the same without them.

### `MOSS Layout Regions` (`HaDeS/MOSS Dev Tools`)

- **Class:** `HaDeSMossLayoutRegions` (V3 schema).
- **Purpose:** rasterizes selected boxes from a Layout Editor / Prompt Enhancer layout draft into a binary mask. It does not segment objects or run a model.
- **Inputs:** `draft`, `selection` (1-based draft order: `1`, `1,3` or `2-4`), `padding` (non-negative pixels around each box, clipped to the canvas). Duplicate selections count once; multiple boxes form a union. Check the selection after reordering boxes.
- **Outputs:** `mask` (white inside, black outside), `preview` (RGB mask), `regions_json` (selected descriptions, original / rasterized coordinates, dimensions, union area), `info` (also accepted by Suite Telemetry). The report appears on the node, including after cached runs.
- **Geometry:** exactly the draft's canvas, no automatic resizing. Fractional starts round down and ends up, matching the Layout Compiler's nine-decimal normalization; end edges are exclusive. A box includes background around an object. Use native mask tools on a copy if you need feathering for composition.

### `MOSS Image Audit` (`HaDeS/MOSS Dev Tools`)

- **Class:** `HaDeSMossImageAudit` (V3 schema).
- **Inputs:** `reference`, `candidate`, optional `region_mask`, `change_threshold` (default 2 on a 0–255 scale), `difference_gain` (default 4, display only), `reference_label`, `candidate_label`.
- **Outputs:** `difference_map` = `clamp(abs(candidate-reference) × gain, 0, 1)`, `report` (readable plus HADES_META for Telemetry), `metrics_json` (versioned, per-pair results and definitions). The node displays a read-only report with wrapping / scrolling; resizing and zoom do not alter measurements. It writes no image or report files itself.
- **Metrics:** RGB absolute mean / maximum error on a 0–255 scale, percentage of pixels whose maximum channel difference × 255 is **strictly greater** than the threshold, candidate-minus-reference encoded-RGB luma (`0.2126 R + 0.7152 G + 0.0722 B`, not physical linear-light luminance), and mean HSV saturation change on a 0–1 scale (black has saturation zero). Color deltas are signed. Inputs are interpreted as encoded sRGB; no color-profile conversion is performed.
- **Regions:** mask values `>= 0.5` are inside; lower values are outside. No fractional weighting. Metrics are computed globally and for both regions; an empty region is JSON `null`, not zero. Without a mask only global metrics are reported. These are differences, not edit success, identity or quality scores.
- **Pairing:** RGB float images in `[0,1]`, identical `[batch,height,width,3]` shapes. No resizing, alignment, image broadcasting or alpha handling. A single mask may be shared across all pairs, or supply one mask per pair; spatial dimensions must match. Every pair retains its own results. Analysis runs on CPU one pair at a time; the difference-image batch remains in CPU memory.
- **Use:** connect reference and candidate outputs to Image Audit; connect Layout Editor `draft` → Layout Regions `draft`, then Regions `mask` → Audit `region_mask`. Match the layout canvas to the images. Connect `difference_map` to Preview Image / Compare Images and save `metrics_json` with your normal text-saving tool if needed. Models, LoRAs and generation settings must be controlled separately; these nodes do not infer their provenance.

The **Dev Tools** example runs without models: a native composition adds 128/255 to the red channel inside one box. Expected: 12,288 pixels (18.75%); inside MAE approximately 42.6667, outside MAE 0, global MAE 8. Float32 arithmetic can affect the last decimal places. Validation and limits are recorded in [`tools/RESULTS.md`](tools/RESULTS.md#dev-tools-validation-2026-11-11).

### `MOSS Pair Guard` (`HaDeS/MOSS Dev Tools`)

- **Purpose:** check that two RGB image batches have identical dimensions and pair counts before sending them to Image Audit. The node passes both images through without modifying pixels. It does not score quality or verify that metadata actually belongs to the connected images.
- **Optional provenance check:** connect stage-specific Suite Telemetry `json` outputs to `reference_metadata` and `candidate_metadata`, or one MOSS KSampler `info` string per side. Each side must identify exactly one KSampler. Pair Guard compares the declared seed, steps, CFG, sampler, scheduler, flow shift, denoise, size and SLG; different model and LoRA names are allowed and reported. If both sides include Prompt Enhancer metadata, `compare_prompt` also checks the generated prompt and negative text. Missing fields appear as unverified; a field present on only one side fails. `require_metadata` makes absent metadata an error.
- **Outputs:** unchanged `reference` and `candidate`, a read-only `report` (with HADES_META for Telemetry), and versioned `checks_json`. Mismatched controls stop execution with a specific error. Use separate Telemetry branches for multi-stage workflows; do not feed the whole pipeline's two samplers to one side.

### `MOSS Recipe Preview` (`HaDeS/MOSS Dev Tools`)

- **Purpose:** inspect the exact nonempty text that will be passed to `MOSS Model Saver`'s `recipe` input and stored as `hades/recipe` when metadata is enabled. Connect a Loader, Multi-LoRA or merger `info` output to `recipe_info`, then connect Recipe Preview's `recipe` output to Model Saver. It makes no file and reads no model weights.
- **Checks:** counts and validates Suite `HADES_META` trailers. An optional `model` socket compares the model patcher's attached Suite recipe with those trailers and warns about a stale branch; this is advisory and cannot prove which patches were applied. Set `include_metadata` to the Saver's value: if false, the report warns that `hades/recipe` will not be written. The node cannot inspect the Saver's actual setting unless you keep these switches in sync.
- **Outputs:** trimmed `recipe` text, a read-only `report` showing that text and warnings (its own HADES_META last, for Telemetry), and versioned `preview_json`. Hand-written notes without a structured trailer are accepted, but the Metadata Reader cannot follow their source files.

The **Pair Guard and Recipe Preview** example in `examples/dev/` is model-free. Synthetic KSampler metadata and a dummy recipe demonstrate successful checks, a deliberate seed mismatch, and a guarded Image Audit connection; the dummy file names are not provenance evidence.

## Interface

The nodes draw their own moss-green rows on the canvas (the HaDeS Anima Suite widget library with its own palette), in the classic canvas and in Nodes 2.0:

- every widget becomes a styled row by type (combo, slider, number, toggle, text); the layout draft keeps ComfyUI's text editor;
- tooltips are the node's own, with the measurements behind each value;
- **Block Merger:** a bar chart of the 42 layers and 3 parts colored by depth group, each group labeled with the share of the SFT → Base change it carries, and buttons **M1**, **50/50**, **24–35 M2** (the groups that carry most of that change) and **M2**;
- **LoRA Block Weight:** a bar per layer group; **Multi-LoRA:** one row per LoRA with a searchable picker and a status line (rows on, few-step LoRAs); **Layout Editor:** the draft's boxes on a canvas, edited with the mouse; **KSampler** and **Latent:** a status line with the settings or the final size. With the turbo input connected the KSampler's status reads TURBO with the turbo node's sampler, steps and CFG, and the rows it overrides (steps, cfg, sampler, scheduler, SLG) are drawn faded, still editable for when it is unplugged.

*Settings → HaDeS MOSS Suite → Use standard ComfyUI widgets* (or *HaDeS MOSS Suite* in a node's right-click menu) switches to the plain widgets. It does not affect HaDeS Anima Suite nodes when both suites are installed.

## Example workflows

In [`examples/`](examples) (drag a file into ComfyUI); model-free development examples are in [`examples/dev/`](examples/dev). Generation examples use the official Comfy-Org file names and carry their download links, so ComfyUI offers to download missing models.

Every generation example loads the three files with the **MOSS Loader**, writes and encodes the prompts with the **Prompt Enhancer** (`as written`, the plain text encoder, unless said otherwise), shares the **MOSS Latent**'s seed with its samplers, ends in a **Suite Telemetry** report that reads the loader and the enhancer too, and saves through the **MOSS Image Saver** with A1111 metadata.

- **Merge, Save and Telemetry:** SFT with layers 24–35 from Base through the Block Merger, MOSS Latent and MOSS KSampler, telemetry report. Model Saver and a SLERP Multi-Model Merger come muted, ready to switch on.
- **Prompt Enhancer:** a short idea written into a detailed prompt (`caption` mode) and encoded, with a text preview of what was written; the latent's seed drives both the writing and the sampler, so one seed reproduces the whole image. A Load Image comes muted, to have the prompt written from a picture. A note explains the modes.
- **Layout Poster:** an OpenMOSS layout draft through the Layout Compiler, whose prompt and latent feed the sampler, with telemetry.
- **Layout Editor:** the Layout Editor's draft through the Layout Compiler into the sampler, with a note on how to edit the boxes; a Load Image is the background (pick your picture there); a second Prompt Enhancer in `layout` mode (writes a draft to IMPORT) comes muted.
- **Turbo:** MOSS Turbo LoRA into the MOSS KSampler's `turbo` input, 4 steps at CFG 1 on SFT; ComfyUI offers to download the LoRA from Hugging Face.
- **Turbo 8 steps + Upscale x2:** Turbo 8-steps v0.7 at 8 steps, then two groups to switch on or off from the group's right-click menu: a 2x upscale with RealESRGAN x2 (on; pixels only, about a second) and, on top of it, a MOSS pass at 2048 with the same turbo and seed at denoise 0.25 (off; adds hair, pores and fine texture the upscaler smooths, about 16 s). A note in the workflow explains both.
- **Control (color, depth):** a depth map into MOSS Control with a color / depth control adapter in the Multi-LoRA (kept at 1.0), on the Prompt Enhancer's conditioning, with a preview of the encoded maps and the control in Telemetry. Load your own depth map and adapter; a note explains the color modes.
- **LoRA Toolkit:** a LoRA through the Multi-LoRA into the sampler (add rows for more LoRAs; each row's group button holds its strength per group of layers); LoRA Merger, LoRA Resize and LoRA Extractor (with a finetune loader) come muted, since they write files. Pick your own LoRA in the Multi-LoRA's row.
- **Dev Tools** (`examples/dev/`): Layout Editor → Layout Regions → native masked composition → Image Audit, with mask / candidate / difference previews and known numerical results. No model files are needed.
- **Pair Guard and Recipe Preview (Dev)** (`examples/dev/`): synthetic paired images and stage metadata → Pair Guard → Image Audit; a separate dummy recipe → Recipe Preview. No model is loaded or saved.
- **Stage Metadata (Dev)** (`examples/dev/`): two synthetic sampling stages feed a full-pipeline Telemetry and a refine-only Telemetry. Image Saver uses the refine stage for A1111 parameters and filename tokens while preserving both stages in PNG metadata. It saves a 64×64 solid-color test PNG; no diffusion model is loaded. The model names in its synthetic info are placeholders.

Every example was run end to end in ComfyUI 0.43.0 (muted nodes left out). `tools/make_examples.py` regenerates them from the node definitions of a running ComfyUI.

## Tests

`tests/test_moss_suite.py` checks the nodes against the real ComfyUI core on a tiny synthetic MOSS-Image model (CPU, a few seconds, no model download). From the ComfyUI folder:

```bash
python custom_nodes/HaDeS-MOSS-Suite/tests/test_moss_suite.py
```

Frontend setting tests (Node.js, no extra packages):

```bash
node --test custom_nodes/HaDeS-MOSS-Suite/tests/test_moss_widgets.mjs
```

Dev Tools geometry / numerical / V3 schema checks (CPU, no model loading): `python custom_nodes/HaDeS-MOSS-Suite/tests/test_moss_dev_nodes.py`.

Every new node also requires real frontend acceptance in classic HaDeS, classic standard widgets, and Nodes 2.0 / Vue. The Dev Tools were checked in all three on frontend 1.57.4; mode-switch fixes and execution evidence are in [`tools/RESULTS.md`](tools/RESULTS.md#three-view-frontend-acceptance-2026-11-11).

Set `HADES_MOSS_TEST_CUDA=1` to include the CUDA quantized-export checks; the default run uses CPU. CI tests ComfyUI v0.43.0 on Linux and Windows, and upstream master on Linux. These checks use small synthetic weights; they do not measure image quality or full-model VRAM. Audit-fix evidence: [`docs/AUDIT_0.6.1.md`](docs/AUDIT_0.6.1.md).

## Tools

- `tools/moss_edit_eval.py --mode spectra | prepare | run | report`: controlled-reference MOSS-edit layer ablations against a running ComfyUI. Persistent images and exact run records go to `--out`; protocol and measured results: [`tools/RESULTS.md`](tools/RESULTS.md#moss-edit-layer-ablation-2026-11-10).
- `tools/moss_eval.py --mode bias | compare`: ethnic bias on neutral and explicit prompts, SFT vs Base on photo, illustration, text and layout prompts, with stock nodes only.
- `tools/moss_eval.py --mode groups`: which layer groups carry the SFT → Base change (needs this suite installed).
- `tools/moss_eval.py --mode lora --lora <name> --lora-model <file>`: which layer groups carry a LoRA's effect.
- `tools/moss_eval.py --mode bias_merge`: ethnic bias of SFT / Base merges.
- `tools/moss_eval.py --mode shift`: square sizes x flow shifts (composition scale, native detail).
- `tools/moss_eval.py --mode slg`: skip-layer guidance blocks on text and portrait prompts, SFT and Base.
- `tools/moss_eval.py --mode schedulers`: sampler x scheduler x steps against a 60-step reference.
- `tools/moss_eval.py --mode attention | progressive`: attention backends and int8 speed; two-resolution sampling.
- `tools/moss_eval.py --mode turbo | turbo_groups | turbo_files`: few-step LoRAs against each other, which layer groups carry them, LoRA files at any prompt set, size or aspect (`--height`).
- `tools/moss_dmd_train.py`: trains a few-step LoRA by DMD on one GPU (shared base, hooked LoRAs, regression term, gradient checkpointing, atomic checkpoints and resume); the turbo LoRAs come from it.
- `tools/moss_activations.py`, `tools/moss_bias_turbo.py`, `tools/moss_bias_fit.py`: per-step biases in place of (or on top of) a few-step LoRA.
