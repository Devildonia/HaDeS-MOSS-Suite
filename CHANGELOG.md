# Changelog

All notable changes to the **HaDeS MOSS Suite** are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **MOSS Model Block Merger:** merges two MOSS-Image models with one ratio per group of 6 layers (`layers_00_05` … `layers_36_41`) and for `text_embedder`, `time_modulation` (`time_embedding` + `shared_block_modulation`) and `io` (`video_embedder` + output head), or per layer through `recipe_override`. The merge is a lazy patch computed when the model loads, so no merged copy is kept in RAM. Quantized models (int8 ConvRot, fp8) are dequantized and merged with a warning in `info`: their rounding is kept, and with a quantized `model1` the result is quantized again into its format. fp16-fixed files, which store the residual stream scaled down (e.g. x 1/16) to fit fp16, are detected and brought to model1's scale before merging. Mixed fused / split q/k/v files are rejected with a clear error.
- **MOSS Multi-Model Merger:** merges 2–4 MOSS-Image models with TIES, DARE, DARE-TIES, Task Arithmetic, Model Stock or SLERP as a lazy patch, ported from the HaDeS Anima Suite. Accepts the same quantized and fp16-fixed inputs as the Block Merger, bringing every model to the anchor's residual scale.
- **MOSS Latent:** empty 128-channel MOSS-Image latent from an aspect ratio and megapixels (1.0 = 1024x1024) or a custom size snapped to 16 px, with width / height outputs and an optional check that the VAE is the 128-channel FLUX.2 one. Square sizes from 768 to 2048 were measured clean at the default shift 5.0, with no need to raise the shift.
- **MOSS LoRA Merger** and **MOSS LoRA Resize:** merge two LoRAs into one file or lower a LoRA's rank, with an exact SVD of the factors per layer (fixed rank or an energy target). Any community MOSS LoRA format is read through ComfyUI's key map and written in the model's layout (split q/k/v becomes fused `proj.qkv` on fused files). Source files are never overwritten.
- **MOSS LoRA Block Weight:** applies one LoRA with a strength per group of 6 layers, loading every community MOSS LoRA format through ComfyUI's key map; identical to the stock loader at 1.0. No group presets: four community LoRAs act in different layers (`tools/RESULTS.md`).
- **MOSS LoRA Extractor:** saves the difference between two MOSS-Image models as a LoRA (truncated SVD per linear layer, full diffs for norms and biases) loadable by the stock LoRA loaders. Accepts quantized and fp16-fixed inputs like the mergers.
- **MOSS Model Saver:** saves a MOSS-Image model with its merge / LoRA patches baked in, as `bfloat16`, `float32` or `int8 ConvRot` with the same layout as Comfy-Org's `*_int8_convrot` files (layer linears in int8, int8 attention, the rest in bfloat16). The files load in the stock `Load Diffusion Model` node.
- **`tools/moss_eval.py`:** phase 0 checks (ethnic bias, SFT vs Base, text and layout) with stock ComfyUI nodes, a `groups` mode that measures which layer groups carry the difference between two models, a `lora` mode that does the same for one LoRA, a `shift` mode for sizes x flow shifts, and a `bias_merge` mode that counts the ethnic bias of SFT / Base merges. Results in `tools/RESULTS.md`.
