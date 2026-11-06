# HaDeS MOSS Suite

ComfyUI nodes for the **MOSS-Image** model family (OpenMOSS / Comfy Org: 42-layer single-stream DiT, `Qwen3.5-9B` text encoder, FLUX.2 VAE), in the style of the [HaDeS Anima Suite](https://github.com/Devildonia/HaDeS-Anima-Suite).

Claims about what layer groups do are measured with [`tools/moss_eval.py`](https://github.com/Devildonia/HaDeS-MOSS-Suite/blob/main/tools/moss_eval.py), not assumed.

**Turbo (0.2.0, Turbo 8-steps in 0.3.0):** few-step LoRAs we trained for MOSS-Image SFT, [HaDeS MOSS Turbo and Turbo Diverse](https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo) (4 steps at CFG 1, 3.4 MB each), joined by Turbo 8-steps v0.7 (8 steps), and a `turbo` input on `MOSS KSampler` that applies them with their settings. How they were trained and measured: [`tools/RESULTS.md`](tools/RESULTS.md), [`tools/moss_dmd_train.py`](https://github.com/Devildonia/HaDeS-MOSS-Suite/blob/main/tools/moss_dmd_train.py).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Devildonia/HaDeS-MOSS-Suite
```

No extra dependencies. Needs a ComfyUI with MOSS-Image support (tested on 0.43.0) and the MOSS-Image weights from `Comfy-Org/MOSS-Image`.

## Nodes

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
- **Inputs:** `model_tuned`, `model_base` (`MODEL`; merge / LoRA patches on either are included), `rank` (`4..256`, default `32`), `export_filename`, `overwrite_existing`.
- **Inputs it accepts:** quantized models are dequantized (with a warning: the difference includes their rounding), and an fp16-fixed `model_tuned` is brought to `model_base`'s residual scale first.
- **Outputs:** `saved_lora_path`, `info` (size, energy kept: the share of the difference the rank holds).
- **Measured:** `galenaCAT − SFT` at rank 64 (252 MB) keeps 95.9% of the energy and renders 4–12 from the finetune on SFT; a rank-32 LoRA of an int8 finetune reproduced it within the int8 noise on three of four images. See [`tools/RESULTS.md`](tools/RESULTS.md).

### `MOSS LoRA Block Weight` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossLoraBlockWeight`
- **Purpose:** applies one LoRA with a separate strength per group of 6 layers (`layers_00_05` … `layers_36_41`) and `other_layers` for LoRA weights outside the layers. Loads every community MOSS format through ComfyUI's own key map (kohya, PEFT split or fused q/k/v); at 1.0 everywhere it renders the same image as the stock `LoraLoaderModelOnly`.
- **Inputs:** `model`, `lora_name`, `strength_model`, the seven group multipliers and `other_layers` (`-2..3`, default `1.0`).
- **Outputs:** `MODEL`, `info` (how many LoRA weights fall in each group, e.g. 20 instead of 24 in `layers_18_23` for LoRAs trained without layer 22).
- **Measured:** layers 30–35 carry a large share of all four community LoRAs tested, but beyond that each acts in different layers (24–35, 30–41 or 00–05 and 30–35), so the node has no style / identity presets; `tools/moss_eval.py --mode lora` measures a given LoRA. See [`tools/RESULTS.md`](tools/RESULTS.md).

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

### `MOSS Model Saver` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossModelSaver`
- **Purpose:** saves a `MODEL` with its merge / LoRA patches baked in, loadable by the stock `Load Diffusion Model` node, and passes the model through.
- **Inputs:**
  - `filename_prefix`: path under `ComfyUI/output` (default `diffusion_models/MOSS_HaDeS_Merged`). Move the file to `models/diffusion_models` to load it.
  - `precision`: `bfloat16` (10.2 GB), `int8 ConvRot (Comfy-Org layout)` (5.4 GB: the four linears of every layer in int8 with Hadamard rotation and int8 attention, everything else in bfloat16, the same keys and dtypes as Comfy-Org's `*_int8_convrot` files) or `float32`.
  - `include_metadata`, `custom_author`, `custom_model_title`: ModelSpec metadata.
  - `rounding` (`bfloat16` output, weights with merge / LoRA patches): `stochastic` (default) rounds each one up or down with probability by distance, seeded per weight so the file is reproducible; `nearest` is ComfyUI's own saving. Unpatched weights are copied unchanged either way.
- **Outputs:** `model` (pass-through), `saved_path`, `info`.
- **Measured** ([`tools/RESULTS.md`](tools/RESULTS.md), "Baking subtle LoRAs into bfloat16"): two LoRAs whose changes are smaller than a bfloat16 step, baked into a merge. Rounded to nearest the file kept about 80–85% of their effect (as the LoRAs loaded live, which round the same way); stochastic rounding kept all of it, and its renders sat as close to the float32 result as two stochastic files with different seeds sit to each other.
- **Measured:** a live merge renders slightly differently from the same merge saved and reloaded (same subject and composition, small details change), in ComfyUI's own merger too; see [`tools/RESULTS.md`](tools/RESULTS.md).

### `MOSS Latent` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossLatent`
- **Purpose:** empty MOSS-Image latent (128 channels, 16 px grid) from an aspect ratio (`1:1`, `4:3`, `3:2`, `16:9`, `21:9` and their portrait forms) and a size in megapixels (`1.0` = 1024×1024), or a custom width / height snapped to 16 px. Outputs `width` / `height` for layout prompts.
- **Inputs:** `aspect_ratio`, `megapixels` (`0.25..4.0`, default `1.0`), `custom_width`, `custom_height`, `batch_size`; optional `vae` (rejects a VAE that is not the 128-channel FLUX.2 one).
- **Outputs:** `latent`, `width`, `height`, `info`.
- **Measured:** square images from 768 to 2048 render clean at the model's default shift 5.0, with no repeated subjects; larger canvases show more of the scene. Raising the shift is not needed at any of these sizes. Non-square ratios are not measured. See [`tools/RESULTS.md`](tools/RESULTS.md).
- **Upscaling a MOSS latent (hi-res fix):** use the core `Upscale Latent By` (`LatentUpscaleBy`), not `Upscale Latent` (`LatentUpscale`). The latter takes a size in pixels and divides it by 8, as for SD VAEs, while the MOSS latent is 16 px per cell: asked for 1024 x 1024, it gives a 2048 x 2048 image.

### `MOSS Layout Compiler` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossLayoutCompiler`
- **Purpose:** compiles an OpenMOSS layout draft (the schema of the official `moss-image-prompt-enhancer`: `width` / `height`, `bbox_coord_space: "abs"`, `bbox_format: "xyxy"` in top-left pixels, `summary` and a flat `items` array) into the model-facing prompt (boxes centered on the canvas, `yxyx`), with the same validation as the official `compile_prompt.py` and byte-identical output. Also outputs the empty latent and `width` / `height` of the draft's own canvas, so prompt and image size cannot drift apart. A `{"caption": ...}` draft passes through as plain text.
- **Inputs:** `draft` (`STRING`, JSON), `batch_size`.
- **Outputs:** `prompt` (to `CLIPTextEncode`), `latent`, `width`, `height`, `info` (lists multi-line text items with their box size).
- **Line breaks:** write them as `\n` in `text`. Measured, the box shape decides whether they are kept: a two-line title in an 832×192 box came out on one line in 3 of 6 seeds, in a 640×384 box on two lines in 6 of 6; a literal newline instead of `\n` changed nothing ([`tools/RESULTS.md`](tools/RESULTS.md)).
- Unlike the core `MOSS-Image Layout Prompt` (boxes from `Create Bounding Boxes`), this node takes hand-written or LLM-written drafts, including `meta`, `canvas`, `palette` and `tags`.

### `MOSS KSampler` (`HaDeS/MOSS Samplers`)
- **Class:** `HaDeSMossKSampler`
- **Purpose:** KSampler with the official ComfyUI MOSS-Image template settings as defaults (25 steps, CFG 2.5, `er_sde` / `simple`, the model's shift 5.0), an optional flow shift, optional skip-layer guidance and optional decode. With the same settings it renders the same image as the stock graph (`KSampler`, `ModelSamplingAuraFlow`, `SkipLayerGuidanceDiTSimple`, `VAEDecode`).
- **Inputs:** `model`, `positive`, `negative`, `latent_image`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`, `flow_shift` (`0` keeps the model's 5.0), `slg` (`off`, `block 22, first half (Base template)`, `block 22, all steps`); optional `vae`, optional `turbo` (from `MOSS Turbo LoRA`: applies the LoRA and samples at CFG 1, euler / simple, SLG off and the turbo node's steps, whatever those widgets say).
- **Outputs:** `latent`, `image` (when a VAE is connected), `info` (the settings actually used).
- **Measured** ([`tools/RESULTS.md`](tools/RESULTS.md)): shift 5.0 is clean from 768 to 2048 px; SLG on block 22 did not change text accuracy on SFT or Base (blocks 24 / 26 made the only errors, on Base), so it is off by default; with euler, 25 steps render nearly the 60-step image with any scheduler and euler / beta does so at 15.

### `MOSS Turbo LoRA` (`HaDeS/MOSS Samplers`)
- **Class:** `HaDeSMossTurbo`
- **Purpose:** picks a few-step LoRA for the `turbo` input of `MOSS KSampler`, which loads it the way the stock `LoraLoaderModelOnly` does and samples at CFG 1, euler / simple and its own `steps` widget, 4 by default (seed, denoise and flow shift still come from the KSampler). Measured counts: 4 for Turbo and Turbo Diverse, 8 for Turbo 8-steps; Turbo and Turbo Diverse give 6–10% less colour than SFT at 4 steps at 1024 (6–14% at 2048) and are back near it at 8 (only colour was measured for them at 8). Other counts run, their quality is untested. Checked in ComfyUI: it renders the stock `LoraLoaderModelOnly` + `KSampler` image pixel for pixel (both LoRAs, 8 images, with the KSampler's own widgets left at 25 steps, CFG 2.5, er_sde / beta and SLG on).
- **Inputs:** `lora_name` (from `models/loras`), `strength` (default `1.0`, the only strength measured), `steps` (default `4`).
- **The LoRAs:** HaDeS MOSS Turbo and Turbo Diverse v0.6 (for 4 steps) and Turbo 8-steps v0.7 (for 8 steps), 3.4 MB each, on [Hugging Face](https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo); put them in `models/loras`. Trained in this repository (`tools/moss_dmd_train.py`) from MOSS-Image SFT, which they are for.
- **Measured** ([`tools/RESULTS.md`](tools/RESULTS.md), 1024 x 1024 against akhaliq's 4-step v2.0 LoRA at the same settings): Turbo spelled every test string right on signs and posters at 1024 x 1024, 6 / 6 on layout JSON and 3 / 3 on a 16:9 poster, while at 2048 x 2048 the sign's small line was right in 1 / 3 (akhaliq's 2 / 3); Turbo Diverse keeps SFT's spread of faces on an open prompt (about 10 East Asian women in 16 seeds, against about 15 for Turbo and akhaliq's LoRA). Turbo 8-steps at 8 steps, in blind pairs on 20 varied prompts: beat akhaliq's 4-step LoRA (full file) 18 to 5, and against lodestones' 8-step DMD2 LoRA for MOSS-Image Base / Kromoss run on SFT (full file) 10 to 7 at 1024 and 0 to 5 at 2048, where it has less colour; it is for up to 1024. lodestones' SFT 8-step LoRA was not compared. Turbo 8-steps at denoise 0.25 / 0.40 on a RealESRGAN x2-upscaled image: by eye it adds real texture (hair, pores) the upscaler smooths, as well as SFT at 25 steps in a fifth of the time. Not measured: other base models, other strengths or step counts, other img2img uses below denoise 1.0; they run (an int8 finetune as the base, img2img at denoise 0.6 were checked), their quality is untested, and Telemetry cannot tell which model the KSampler got.

### `MOSS Sampler Metadata` (`HaDeS/MOSS Samplers`)
- **Class:** `HaDeSMossSamplerMetadata`
- **Purpose:** reads the settings a MOSS KSampler actually sampled with from its `info` (seed, steps, cfg, sampler, scheduler, denoise, flow shift, and `turbo`: the turbo LoRA file and strength, empty without one), for image-saver metadata.

### `MOSS Suite Telemetry` (`HaDeS/MOSS Pipeline`)
- **Class:** `HaDeSMossSuiteTelemetry`
- **Purpose:** reads the `info` outputs of up to 8 HaDeS MOSS nodes (their `HADES_META` trailer) and checks the pipeline against what was measured:
  - **error:** layout, latent and KSampler on different canvas sizes (a layout prompt places its boxes on its own canvas);
  - **warning:** quantized models in a merge, TIES / DARE with a quantized model, LoRAs extracted / resized / merged below 90% energy;
  - **note:** KSampler size outside 768²–2048², shift other than 5.0, SLG on, turbo LoRA at a strength other than 1.0, at a step count other than 4 or 8 or at denoise below 1.0, fp16-fixed residual scale normalized, multi-line layout text with its box proportions.
- **Outputs:** `report` (Markdown), `json`, `summary` (one line).

## Interface

The nodes draw their own moss-green rows on the canvas (the HaDeS Anima Suite widget library with its own palette), in the classic canvas and in Nodes 2.0:

- every widget becomes a styled row by type (combo, slider, number, toggle, text); the layout draft keeps ComfyUI's text editor;
- tooltips are the node's own, with the measurements behind each value;
- **Block Merger:** a bar chart of the 42 layers and 3 parts colored by depth group, each group labeled with the share of the SFT → Base change it carries, and buttons **M1**, **50/50**, **24–35 M2** (the groups that carry most of that change) and **M2**;
- **LoRA Block Weight:** a bar per layer group; **KSampler** and **Latent:** a status line with the settings or the final size. With the turbo input connected the KSampler's status reads TURBO and the rows it overrides (steps, cfg, sampler, scheduler, SLG) are drawn faded, still editable for when it is unplugged.

*Settings → HaDeS MOSS Suite → Use standard ComfyUI widgets* (or *HaDeS MOSS Suite* in a node's right-click menu) switches to the plain widgets. It does not affect HaDeS Anima Suite nodes when both suites are installed.

## Example workflows

In [`examples/`](examples) (drag a file into ComfyUI). They use the official Comfy-Org file names and carry their download links, so ComfyUI offers to download missing models.

- **Merge, Save and Telemetry:** SFT with layers 24–35 from Base through the Block Merger, MOSS Latent and MOSS KSampler, telemetry report. Model Saver and a SLERP Multi-Model Merger come muted, ready to switch on.
- **Layout Poster:** an OpenMOSS layout draft through the Layout Compiler, whose prompt and latent feed the sampler, with telemetry.
- **Turbo:** MOSS Turbo LoRA into the MOSS KSampler's `turbo` input, 4 steps at CFG 1 on SFT; ComfyUI offers to download the LoRA from Hugging Face.
- **Turbo 8 steps + Upscale x2:** Turbo 8-steps v0.7 at 8 steps, then two groups to switch on or off from the group's right-click menu: a 2x upscale with RealESRGAN x2 (on; pixels only, about a second) and, on top of it, a MOSS pass at 2048 with the same turbo at denoise 0.25 (off; adds hair, pores and fine texture the upscaler smooths, about 16 s). A note in the workflow explains both.
- **LoRA Toolkit:** a LoRA through LoRA Block Weight into the sampler; LoRA Merger, LoRA Resize and LoRA Extractor (with a finetune loader) come muted, since they write files. Pick your own LoRA in `lora_name`.

Every example was run end to end in ComfyUI 0.43.0 (muted nodes left out). `tools/make_examples.py` regenerates them from the node definitions of a running ComfyUI.

## Tests

`tests/test_moss_suite.py` checks the nodes against the real ComfyUI core on a tiny synthetic MOSS-Image model (CPU, a few seconds, no model download). From the ComfyUI folder:

```bash
python custom_nodes/HaDeS-MOSS-Suite/tests/test_moss_suite.py
```

## Tools

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
