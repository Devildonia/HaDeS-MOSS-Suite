# HaDeS MOSS Suite 0.4.0 — pre-release audit (2026-11-07)

Scope: every node module (common, mergers, saver, LoRA tools and Multi-LoRA, loader, latent / layout compiler / layout editor,
prompt enhancer, samplers, telemetry), the canvas scripts, the example workflows, the Registry package and README / CHANGELOG
against the code. ComfyUI 0.43.0, PyTorch 2.9.1+cu130, RTX 5090.

## Verified

| Check | Result |
|---|---|
| Unit tests (`tests/test_moss_suite.py`, CPU) | 40 / 40 pass (32 before the 0.4.0 work) |
| Canvas scripts (`node --check`) | all four valid |
| Lazy merges (block, TIES / DARE / Task Arithmetic / Model Stock / SLERP) | code read: patches captured at merge time, both sides from stored tensors, DARE masks seeded per key |
| Stochastic rounding | unbiased for negative values too (sign-magnitude bits), inf / nan and overflow keep round-to-nearest |
| Layout compiler / editor | same validation in both; editor output compiles to the same prompt as its input |
| New nodes in Nodes 2.0 (Vue) | Multi-LoRA, Layout Editor, Prompt Enhancer: drawing, clicks, drags, pickers, collapse; checked live |
| MOSS Loader live | bf16 SFT, an int8 community file, wrong VAE and wrong text encoder (clear errors) |
| Example workflows | all 7 load in the editor and run end to end (`make_examples.py --run`), Telemetry clean except the measured two-line layout note |
| Registry package | last zip: tracked files minus `.comfyignore`; no `Claude.md`, `Moss.md`, tests, tools or weights |

## Findings

| # | Severity | Where | Finding | Fix (status) |
|---|---|---|---|---|
| 1 | High (silent) | MOSS Model Saver, `float32` | Patched weights (merge / LoRA) were baked by the core's lazy saving in the model's own dtype, so on a GPU (bfloat16 in memory) they were rounded to bfloat16 before reaching the float32 file; the metadata said `rounding: none`. Not visible on the CPU, where ComfyUI holds weights in float32. | Fixed: patched weights computed in float32. Test with the model held in bfloat16; it fails on the old code (98.4% of the weights wrong). |
| 2 | High (silent) | LoRA Block Weight, LoRA tools, Multi-LoRA | A LoRA none of whose keys match MOSS-Image (one for another model) loaded as nothing and was reported as applied. Run live: an Anima LoRA on MOSS gave `info` "+1.00 Anima Image Enhancer" and changed nothing. | Fixed: an error names the file; Multi-LoRA's `info` counts the weights each LoRA changes. Test. |
| 3 | Medium (security) | Multi-LoRA tooltip | The LoRA name, read from the workflow's JSON, went into the tooltip as HTML: a shared workflow could run script on hover. | Fixed: escaped. |
| 4 | Medium | Suite Telemetry | Canvas-mismatch **error** (with a layout message) for a latent and a KSampler of different sizes without any layout, e.g. an upscale pass; the Prompt Enhancer's layout canvas was not compared. | Fixed: canvases compared only with a layout in the pipeline, the enhancer's layout canvas included. Test. |
| 5 | Medium | README | Telemetry's energy warning documented as "below 90%"; the code (and the measurement) say 97.9%. | Fixed. |
| 6 | Low | Few-step LoRA detection | The `CFG 1` match in the file's metadata also took `CFG 1.5`. | Fixed in Python and JS. Test. |
| 7 | Low | Prompt Enhancer, caption | An answer `{"caption": ""}` was encoded as the leftover `}`. | Fixed: an empty caption is an error. Test. |
| 8 | Design | Prompt Enhancer, MOSS Latent | Seeds started on ComfyUI's default `randomize`, so a caption was rewritten on every run. A node can declare its own default (the Multi-Model Merger already starts on `fixed`). | Changed: both start on `fixed`. |
| 9 | Release | `pyproject.toml`, `__init__.py` | Description without the new nodes; no Registry `Icon`; version 0.3.0. | Fixed: description, `Icon`, 0.4.0. |

Items 3, 6 and 7 concern nodes new in 0.4.0, so the CHANGELOG lists them under Added, not Fixed.

## Measured on the way (not a finding)

KJNodes' `PathchSageAttentionKJ` in `auto` on top of `--use-sage-attention`, SFT bf16 at 1024, PyTorch 2.9.1+cu130: identical
images pixel for pixel and the same time as the server's SageAttention (13.16 s against 13.33 s, within the run-to-run spread),
so it adds nothing on MOSS-Image. The update to cu130 made the same render 12% faster (15.1 s to 13.3 s). The
`sageattn_qk_int8_pv_fp8_cuda++` mode was 4% faster on 2 images and changed them more than SageAttention vs SDPA; the machine
powered off during its third image (cooling set to silent), so it was not measured further.
