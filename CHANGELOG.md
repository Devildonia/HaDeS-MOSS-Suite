# Changelog

All notable changes to the **HaDeS MOSS Suite** are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Documentation
- **README / `tools/RESULTS.md`:** upscaling a MOSS latent needs the core `Upscale Latent By`; `Upscale Latent` assumes a x8 VAE and doubles the image size on MOSS.

### Added
- **`tools/moss_eval.py`:** an `attention` mode that times images under the attention backend the server runs and compares runs pixel by pixel. `tools/RESULTS.md` records that every measurement used SageAttention, and compares it with PyTorch attention and int8 files in speed and images.
- **`tools/moss_eval.py`:** `progressive` mode (two-resolution sampling through stock `KSamplerAdvanced`, with re-noise and backtracking), `turbo` mode (few-step LoRAs with the usual and the bypass loader) and `turbo_groups` mode (which layer groups of a few-step LoRA carry its effect). Results in `tools/RESULTS.md`: two-resolution sampling saves about 15% cleanly at 1024 x 1024; the DMD2 8-step and the 4-step LoRAs render 5.6x / 9.8x faster, lose about half their change to bf16 rounding in the early layers with the usual loader, and do their work in layers 24–41.
- **`tools/moss_eval.py`:** `turbo_files` mode (renders LoRA files derived from a few-step LoRA, any prompt set or size, against the original). Results in `tools/RESULTS.md`: cut to layers 24–41 at rank 2 (3.6 MB), both few-step LoRAs render their originals' images, text and layout prompts (54 of 54 spelled right) and 2048 x 2048 included; their directions are not readable from the SFT weights' own spectrum.
- **`tools/moss_activations.py`** and **`tools/moss_bias_turbo.py`:** record what a reduced few-step LoRA reads from the late-layer activations, and replace it by fixed per-step biases calibrated from those activations. Results in `tools/RESULTS.md`: on the qkv weights the change is nearly constant over tokens and images; with no LoRA, the calibrated biases on layers 24–41 turn SFT at 4 steps / CFG 1 into clean images and correct text on held-out seeds.
- **`tools/moss_bias_fit.py`:** fits those per-step biases with no LoRA, against SFT at 25 steps as the teacher; `moss_bias_turbo.py --biases` tests them, `--calib-only` saves calibrated ones. Results in `tools/RESULTS.md`: after 19 minutes of fitting at 512 x 512, SFT at 4 steps goes from broken to coherent images with legible text, below the LoRA's quality. `--teacher-lora` / `--blend` / `--init-biases` fit them against the few-step LoRA, or a blend of it and SFT, as the teacher: from zero in 5 minutes the biases reach the LoRA's quality with no LoRA loaded, and they keep it down to a half-LoRA teacher, losing it below.

## [0.1.1] - 2026-11-04

Fixes from the pre-publication audit; 0.1.0 never reached the Comfy Registry.

### Fixed
- **MOSS Model Saver:** an int8 model (e.g. an int8 ConvRot finetune) saved as `bfloat16` or `float32` kept its int8 layers and their quantization keys; they are now dequantized with their patches applied, so the file holds no int8 weights.
- **MOSS KSampler:** with the `image` output connected and no VAE, the output was `None` and the next node failed on it; the KSampler now stops before sampling and asks for the VAE.
- **MOSS Model Block Merger:** a `recipe_override` with non-numeric entries now names them in the error.
- **MOSS Suite Telemetry:** the LoRA energy warning now fires below 97.9%, the lowest energy measured within the source's noise floor, and cites the measured ranges.
- **README:** the title no longer says *work in progress*.
- **Tooltips:** the layer-group shares and the SLG note now say exactly what was measured (group taken alone or left out; on Base, SLG changed one face entirely).

### Added
- **Tests:** int8 model saved as `bfloat16` / `float32` holds no int8 tensors; int8 output matches the Comfy-Org layout; KSampler image output without a VAE is reported.

## [0.1.0] - 2026-11-04

First release: 12 nodes for MOSS-Image, every recommended value and tooltip claim measured (`tools/RESULTS.md`).

### Added
- **MOSS Model Block Merger:** merges two MOSS-Image models with one ratio per group of 6 layers (`layers_00_05` … `layers_36_41`) and for `text_embedder`, `time_modulation` (`time_embedding` + `shared_block_modulation`) and `io` (`video_embedder` + output head), or per layer through `recipe_override`. The merge is a lazy patch computed when the model loads, so no merged copy is kept in RAM. Quantized models (int8 ConvRot, fp8) are dequantized and merged with a warning in `info`: their rounding is kept, and with a quantized `model1` the result is quantized again into its format. fp16-fixed files, which store the residual stream scaled down (e.g. x 1/16) to fit fp16, are detected (a scale that leaves the latent embedder otherwise equal) and brought to model1's scale before merging. Mixed fused / split q/k/v files are rejected with a clear error.
- **MOSS Multi-Model Merger:** merges 2–4 MOSS-Image models with TIES, DARE, DARE-TIES, Task Arithmetic, Model Stock or SLERP as a lazy patch, ported from the HaDeS Anima Suite. Accepts the same quantized and fp16-fixed inputs as the Block Merger, bringing every model to the anchor's residual scale.
- **MOSS Model Saver:** saves a MOSS-Image model with its merge / LoRA patches baked in, as `bfloat16`, `float32` or `int8 ConvRot` with the same layout as Comfy-Org's `*_int8_convrot` files (layer linears in int8, int8 attention, the rest in bfloat16). The files load in the stock `Load Diffusion Model` node.
- **MOSS LoRA Extractor:** saves the difference between two MOSS-Image models as a LoRA (truncated SVD per linear layer, full diffs for norms and biases) loadable by the stock LoRA loaders. Accepts quantized and fp16-fixed inputs like the mergers.
- **MOSS LoRA Block Weight:** applies one LoRA with a strength per group of 6 layers, loading every community MOSS LoRA format through ComfyUI's key map; identical to the stock loader at 1.0. No group presets: four community LoRAs act in different layers (`tools/RESULTS.md`).
- **MOSS LoRA Merger** and **MOSS LoRA Resize:** merge two LoRAs into one file or lower a LoRA's rank, with an exact SVD of the factors per layer (fixed rank or an energy target). Any community MOSS LoRA format is read through ComfyUI's key map and written in the model's layout (split q/k/v becomes fused `proj.qkv` on fused files). Source files are never overwritten.
- **MOSS Latent:** empty 128-channel MOSS-Image latent from an aspect ratio and megapixels (1.0 = 1024x1024) or a custom size snapped to 16 px, with width / height outputs and an optional check that the VAE is the 128-channel FLUX.2 one. Square sizes from 768 to 2048 were measured clean at the default shift 5.0, with no need to raise the shift.
- **MOSS Layout Compiler:** compiles an OpenMOSS layout draft into the MOSS-Image prompt with the same validation and byte-identical output as the official `compile_prompt.py`, plus the empty latent and width / height of the draft's canvas. Its info lists multi-line text items with their box size.
- **MOSS KSampler** and **MOSS Sampler Metadata:** a KSampler with the official template settings as defaults, an optional flow shift, optional skip-layer guidance on block 22 (off by default: it did not change text accuracy in tests) and optional decode, identical to the stock graph at the same settings; the metadata node reads the settings actually used from its info.
- **MOSS Suite Telemetry:** reads the info outputs of the other nodes and reports canvas mismatches between layout, latent and sampler, quantized inputs in merges, low-energy LoRAs, settings outside the measured range and multi-line layout text, each with the measurement behind it.
- **Canvas interface:** moss-green styled rows on every node, from the HaDeS Anima Suite widget library with its own palette and icon; works in the classic canvas and Nodes 2.0. Block Merger and LoRA Block Weight get a per-layer bar chart, KSampler and Latent a status line. A *Use standard ComfyUI widgets* setting (and a per-node menu entry) restores the plain widgets, independent of the HaDeS Anima Suite one.
- **Example workflows:** Merge, Save and Telemetry; Layout Poster; LoRA Toolkit. Official Comfy-Org model names with download links; file-writing nodes come muted. Built by `tools/make_examples.py` from the live node definitions and run end to end in ComfyUI.
- **Tests and CI:** `tests/test_moss_suite.py` runs against the real ComfyUI core on a tiny synthetic MOSS-Image model (CPU, seconds): layout compilation and rejections, latent sizes, block / multi merges (ratios, layer groups, fp16-fixed rescale, int8 input), LoRA factors against the core's patch math for split and fused formats, LoRA merger / resize / extractor files loaded back by the core, LoRA block weight groups and telemetry findings. GitHub Actions runs it on the latest ComfyUI.
- **`tools/moss_eval.py`:** phase 0 checks (ethnic bias, SFT vs Base, text and layout) with stock ComfyUI nodes, a `groups` mode that measures which layer groups carry the difference between two models, a `lora` mode that does the same for one LoRA, a `shift` mode for sizes x flow shifts, `slg` and `schedulers` modes, and a `bias_merge` mode that counts the ethnic bias of SFT / Base merges. Results in `tools/RESULTS.md`.
