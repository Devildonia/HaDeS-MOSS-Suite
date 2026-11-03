# HaDeS MOSS Suite (work in progress)

ComfyUI nodes for the **MOSS-Image** model family (OpenMOSS / Comfy Org: 42-layer single-stream DiT, `Qwen3.5-9B` text encoder, FLUX.2 VAE), in the style of the [HaDeS Anima Suite](https://github.com/Devildonia/HaDeS-Anima-Suite).

Claims about what layer groups do are measured with [`tools/moss_eval.py`](tools/moss_eval.py), not assumed.

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

### `MOSS Model Saver` (`HaDeS/MOSS Mergers`)
- **Class:** `HaDeSMossModelSaver`
- **Purpose:** saves a `MODEL` with its merge / LoRA patches baked in, loadable by the stock `Load Diffusion Model` node, and passes the model through.
- **Inputs:**
  - `filename_prefix`: path under `ComfyUI/output` (default `diffusion_models/MOSS_HaDeS_Merged`). Move the file to `models/diffusion_models` to load it.
  - `precision`: `bfloat16` (10.2 GB), `int8 ConvRot (Comfy-Org layout)` (5.4 GB: the four linears of every layer in int8 with Hadamard rotation and int8 attention, everything else in bfloat16, the same keys and dtypes as Comfy-Org's `*_int8_convrot` files) or `float32`.
  - `include_metadata`, `custom_author`, `custom_model_title`: ModelSpec metadata.
- **Outputs:** `model` (pass-through), `saved_path`, `info`.
- **Measured:** a live merge renders slightly differently from the same merge saved and reloaded (same subject and composition, small details change), in ComfyUI's own merger too; see [`tools/RESULTS.md`](tools/RESULTS.md).

## Tools

- `tools/moss_eval.py --mode bias | compare`: ethnic bias on neutral and explicit prompts, SFT vs Base on photo, illustration, text and layout prompts, with stock nodes only.
- `tools/moss_eval.py --mode groups`: which layer groups carry the SFT → Base change (needs this suite installed).
