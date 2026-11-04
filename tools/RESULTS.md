# Measurement results (2026-11-03)

Measured with `tools/moss_eval.py` on an RTX 5090, ComfyUI 0.43.0 local API. 1024x1024, er_sde / simple / 25 steps / CFG 2.5,
model shift 5.0, negative `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` (the official ComfyUI templates),
text encoder `mossImage_base_txt_int4` (w4a8 with codebook), VAE `flux2-vae`. Models: `mossImage_sft_bf16` and `mossImage_base_bf16`
(fused `proj.qkv`, 42 layers). Ethnicity counts are by eye on the contact sheets. ComfyUI runs with `--use-sage-attention` (the
launcher's flag, read from the server on 2026-11-04), so every image here was sampled with SageAttention.

## Attention backends and int8 speed

`--mode attention`, RTX 5090 (32 GB, the whole model in VRAM in every run), café woman, fisherman and watercolor fox x seeds
1000–1002, er_sde / simple / 25 steps / CFG 2.5. Seconds per image count only runs where the model and both prompts came from the
cache (sampling + decode), median of 6 per cell. The server ran with `--use-sage-attention` or `--use-pytorch-cross-attention`
(checked from its argv). int8 = `mossImage_sft_bf16` saved by HaDeS MOSS Model Saver as int8 ConvRot (Comfy-Org layout, which sets
comfy-kitchen int8 attention per layer); "int8, no attention setting" = the same file with the 42 `comfy_attention.config` keys
removed, so it uses the server's attention.

| Model / attention | 1024 x 1024 | 2048 x 2048 |
|---|---|---|
| bf16, PyTorch SDPA | 16.31 s | 91.76 s |
| bf16, SageAttention | 15.13 s | 71.98 s |
| int8, no attention setting, PyTorch SDPA | 25.32 s | 129.69 s |
| int8, no attention setting, SageAttention | 23.70 s | 107.55 s |
| int8, comfy-kitchen int8 attention (Comfy-Org layout) | 24.08 s | 109.12 s |

Mean absolute pixel difference (0–255) over the 9 images, mean / worst image:

| Comparison | 1024 x 1024 | 2048 x 2048 |
|---|---|---|
| SageAttention, two runs | 0.0 / 0.0 | 0.0 / 0.0 |
| bf16: SageAttention vs PyTorch SDPA | 6.1 / 13.1 | 3.8 / 8.5 |
| int8: SageAttention vs PyTorch SDPA | 6.6 / 18.0 | 5.2 / 13.2 |
| int8: comfy-kitchen vs SageAttention | 7.6 / 17.2 | 7.1 / 18.5 |
| PyTorch SDPA: int8 vs bf16 | 8.2 / 15.5 | 5.7 / 10.3 |

- SageAttention saves 7% of the time per image at 1024 x 1024 and 22% at 2048 x 2048 on bf16 (6% / 17% on int8): the larger the
  canvas, the more of the time is attention.
- It renders the same image for the same seed (two runs identical pixel for pixel), so it adds no run-to-run noise to the
  measurements in this file, all of which ran with SageAttention.
- Against PyTorch SDPA it changes details, not the image: in the worst pair (a café collage) the same layout, person and scene, with
  a different head turn, cup and chairs (layout distance 3.0 at 1024, against ~20 between two seeds). The change is about 3/4 of what
  int8 quantization does. A seed shared between machines with different attention backends will not reproduce exactly.
- int8 is slower than bf16 on this GPU: +57% at 1024 and +49% at 2048 with SageAttention. With the same int8 weights, comfy-kitchen
  int8 attention runs at SageAttention's speed (+2%), so the cost is in the int8 layers, not the attention. int8 saves VRAM (5162 MB
  staged against 9693 MB for bf16); on a GPU where bf16 does not fit, the trade-off was not measured.
- Community int8 files checked here (`pornmasterMoss_sftV1`, `realismByStableYogi_v15INT8_MOSS`) carry no attention setting and use the
  server's attention.

## Resolution and flow shift

`--mode shift`, `mossImage_sft_bf16`, square sizes x flow shifts through `ModelSamplingAuraFlow` (multiplier 1, as the model's own
sampling; 5.0 is the model default), café woman, fisherman and watercolor fox prompts x seeds 1000 / 1001. Slope: low-frequency
spectral slope (more negative = larger masses); detail: high-frequency power of a native 512 px crop, against shift 5.0 at the same size.

| Size | Shift | slope vs 1024 / 5.0 | detail vs shift 5.0 | layout vs shift 5.0 |
|---|---|---|---|---|
| 768 (0.6 MP) | 3.0 / 5.0 | -0.057 / -0.050 | -0.013 / 0 | 3.84 / 0 |
| 1024 (1.0 MP) | 3.0 / 5.0 / 7.0 | +0.011 / 0 / -0.031 | -0.020 / 0 / -0.010 | 4.88 / 0 / 5.16 |
| 1536 (2.4 MP) | 5.0 / 7.5 | -0.074 / -0.102 | 0 / +0.051 | 0 / 5.26 |
| 2048 (4.2 MP) | 5.0 / 7.5 / 10.0 | -0.152 / -0.143 / -0.154 | 0 / +0.025 / +0.107 | 0 / 5.08 / 6.07 |

- Every size from 768 to 2048 rendered clean images at the default shift 5.0, with no repeated subjects and no collapsed frames; at
  2048 the native-resolution crop holds fine detail (skin, hair, knit). Larger canvases show more of the scene around a smaller
  subject (steeper slope), not repeated subjects.
- Raising the shift changes the composition less than a new seed does: layout distance 3.8–6.1 against shift 5.0, while seeds
  1000 and 1001 at shift 5.0 are 19.5 / 20.2 / 19.5 / 22.4 apart (768 / 1024 / 1536 / 2048, mean of the three prompts). It adds a
  little fine detail at 2048 (+0.107 at shift 10).
  Unlike Anima, where 2048 at its default shift collapsed into dark frames, MOSS needs no shift change with resolution; 5.0 stays.

## Skip-layer guidance (SLG)

`--mode slg`, SFT and Base, `SkipLayerGuidanceDiTSimple` on one single block for the negative pass, 4 seeds each on the bakery sign,
the three-line poster, the layout JSON poster and the fisherman portrait. Text images with every string spelled and set right, out of
12 per config (3 text prompts x 4 seeds), counted by eye:

| SLG | SFT | Base |
|---|---|---|
| none | 12 / 12 | 12 / 12 |
| block 22, first half (the official Base template) | 12 / 12 | 12 / 12 |
| block 22, all steps | 12 / 12 | 12 / 12 |
| block 24, first half | 12 / 12 | 11 / 12 ("BREAAD") |
| block 26, first half | 12 / 12 | 11 / 12 (deformed glyphs, "QUI∃T") |

- Without SLG both models already spelled every string right, so SLG on block 22 showed no gain in text; blocks 24 and 26 made the
  only two errors, both on Base.
- Block 22 over the first half or over all steps gave nearly the same images.
- On the portrait SLG changes details only (a little more contrast and wrinkles with 24 / 26); on Base it changed one face entirely.

## Samplers and schedulers

`--mode schedulers`, SFT, CFG 2.5, café woman, fisherman and watercolor fox x seeds 1000 / 1001, mean distance to a 60-step
euler / simple render of the same seed (lower = closer to the converged image), layout / edges / texture / color:

| euler | 15 steps | 25 steps |
|---|---|---|
| simple | 6.76 / 0.219 / 0.711 / 4.05 | 4.93 / 0.185 / 0.324 / 1.89 |
| beta | 5.69 / 0.192 / 0.646 / 2.76 | 6.13 / 0.189 / 0.386 / 1.95 |
| normal | 7.19 / 0.240 / 0.749 / 4.44 | 4.81 / 0.152 / 0.363 / 1.92 |

- euler converges early: at 25 steps every scheduler renders nearly the 60-step image, and euler / beta does so at 15 steps already.
- er_sde (the sampler of the official templates) is stochastic: it renders a different image of the same seed (layout ~18, edges
  ~0.91 from the euler reference at any scheduler or step count), so this convergence measure cannot rank it. By eye its renders are
  as clean, with a little more contrast and edge definition (most visible on the watercolor). Which looks better is not measured.

## Model Saver

A 0.5 SFT/Base merge (HaDeS MOSS Model Block Merger) saved with HaDeS MOSS Model Saver and reloaded with the stock `UNETLoader`,
fisherman prompt, seed 1000, mean absolute pixel difference (0-255):

| Comparison | Difference |
|---|---|
| SFT saved without changes vs the original SFT file | 0.0 |
| Saved bf16 weights vs (SFT + Base) / 2 computed directly | max 3e-8 per weight (rounding of the smallest values) |
| Live merge (this suite) vs live merge (core `ModelMergeSimple`, same ratio) | 0.0 |
| Live merge vs the saved bf16 file | 7.9 |
| Live merge vs the saved int8 ConvRot file | 2.9 |
| Saved bf16 vs saved int8 ConvRot | 8.8 |

- The saved file holds the merged weights, and the saver changes nothing on an unpatched model.
- A merge applied live renders a little differently from the same weights loaded from a file, in the core merger as in this one:
  same subject and composition, different small details (pose of the arms, framing). Judge a merge on the saved file when the
  exact image matters.
- The int8 ConvRot file has the same keys, dtypes and shapes as Comfy-Org's `moss_image_sft_int8_convrot` (5.41 GB).

## Merging an int8 ConvRot finetune

`pornmasterMoss_sftV1` (int8 ConvRot, SFT-derived: identical to SFT outside the layer linears) merged with `mossImage_sft_bf16`
through HaDeS MOSS Model Block Merger, checked on the weights (no render):

- At ratio 1.0 every merged weight equals the dequantized int8 weight (`torch.allclose` on three layers).
- The finetune's difference to SFT is about as large as the int8 rounding itself (relative norm, SFT re-quantized with the same layout
  as the yardstick):

| Weight | int8 rounding of SFT | finetune vs SFT | top-32 singular energy: finetune vs SFT / int8 noise |
|---|---|---|---|
| layers.0.proj.qkv | 0.0094 | 0.0120 | 0.478 / 0.034 |
| layers.10.proj.o | — | — | 0.044 / 0.043 |
| layers.20.proj.ffn_down | 0.0100 | 0.0099 | 0.150 / 0.028 |
| layers.35.proj.o | 0.0096 | 0.0088 | 0.043 / 0.042 |

- Where the finetune changed a layer (0, 20), its change is low rank and stands out from the noise; where it did not (10, 35), the
  difference to SFT is int8 rounding alone. A block merge spreads that noise by the ratio; methods that trim task vectors by magnitude
  (TIES / DARE) would keep noise in the untouched layers, so an int8 model is a poor input for them.

With the merger in ComfyUI (fisherman prompt, seed 1000, mean absolute pixel difference): SFT x pornmaster at ratio 1.0 is 2.4 from
pornmaster loaded directly and 6.7 from SFT; at 0.5 it is 4.7 / 4.3; pornmaster itself is 5.8 from SFT.

## fp16-fixed files (residual stream rescale)

`miaomiaoHaremMOSS_baseV11_fp16` (fp16, Base-derived) fitted tensor by tensor against `mossImage_base_bf16` (least-squares scale):

- 12 tensors are Base's exactly x 1/16 (residual 0.0000 after scaling): `video_embedder` and, in `text_embedder`, `proj` and every
  block's `proj` and `ffn_down` (weights and biases).
- `shared_block_modulation`: the two gate chunks (attention, ffn) x 1/16; the shift and scale chunks unchanged.
- `final_linear_video`, `video_head_modulation`, all norms: unchanged.

These are all the writes into the residual stream, and every norm in MOSS is an RMSNorm, so the file is the same model with its
residual stream scaled by 1/16 to stay in fp16 range. The finetune itself changed 64 of the 336 layer tensors (mean 0.2% vs Base).
Merged as stored, its task vector in those 14 tensors would be -15/16 of the weights. The merger fits the scale on `video_embedder`,
finds 16.0, and multiplies model2's residual writes and gates by it before merging: Base x miaomiao at ratio 1.0 gives Base's values
in those tensors (miaomiao did not change them beyond the scale) and miaomiao's in the layers.

In ComfyUI (fisherman prompt, seed 1000, mean absolute pixel difference): Base x miaomiao at ratio 1.0 is 1.8 from miaomiao loaded
directly and 22.3 from Base; at 0.5 it is 16.3 / 8.5; miaomiao itself is 22.5 from Base. All renders are clean.

## Multi-Model Merger methods

HaDeS MOSS Multi-Model Merger, weights 1.0 / 1.0, density 0.5, scale 1.0, seed 0; fisherman and watercolor fox prompts x 2 seeds,
judged by eye next to the input models. Two families:

- SFT family: `galenaCATGalenaCitronAnime_moss` (bf16) + `pornmasterMoss_sftV1` (int8), base_model SFT.
- Mixed family: `miaomiaoHaremMOSS_baseV11_fp16` (fp16-fixed) + `mossImage_sft_bf16`, base_model Base.

Every method rendered clean images in both families, and `info` reported the int8 input (with the task-vector note for TIES / DARE)
and the x16 residual rescale (x0.0625 of SFT for SLERP, whose anchor is model_1).

| Method | SFT family | Mixed family |
|---|---|---|
| TIES, DARE-TIES | close to galenaCAT (warm light, its fox style) | close to SFT |
| DARE, Task Arithmetic, SLERP | between the two models | between the two models |
| Model Stock | close to SFT, the base | close to Base, the base |

- TIES and DARE-TIES follow the model whose task vector is larger: galenaCAT (two LoRAs baked into SFT) over pornmaster, and SFT
  (the whole SFT stage) over miaomiao.
- Model Stock stays near base_model in both families, as in the HaDeS Anima Suite: the task vectors point in unrelated directions.
- Which method gives better images is not measured; it depends on what the models changed.

## LoRA Extractor

HaDeS MOSS LoRA Extractor on `finetune - mossImage_sft_bf16`, the LoRA applied to SFT with the stock `LoraLoaderModelOnly` at 1.0
(the ComfyUI log reported no skipped key), fisherman and watercolor fox prompts x seeds 1000 / 1001, mean absolute pixel difference:

| Finetune | Rank | Size | Energy kept | finetune vs SFT | SFT + LoRA vs finetune |
|---|---|---|---|---|---|
| galenaCAT (SFT + 2 LoRAs, bf16) | 32 | 126 MB | 90.6% | 31–66 | 8–24 |
| galenaCAT | 64 | 252 MB | 95.9% | 31–66 | 4–12 |
| pornmaster (int8) | 32 | 126 MB | 13.2% | 6–26 | 5–18 |

- Only the 168 layer linears differ in both finetunes; the other 210 tensors are SFT's and are skipped.
- On pornmaster the energy kept is low because most of the difference is int8 rounding (see above), which low rank leaves out.
  SFT quantized to int8 with the Model Saver renders 4.0 / 5.1 / 14.1 / 6.6 from SFT bf16 on the same four images: that is the noise
  floor. pornmaster is 6.7 / 17.2 / 24.1 / 27.0 from SFT int8, and its rank-32 LoRA on SFT bf16 is 5.0 / 18.1 / 9.4 / 6.3 from
  pornmaster: within the noise floor on three images, while on the fisherman seed 1001 it did not reproduce the change.

## LoRA Resize and LoRA Merger

Both nodes read a LoRA through ComfyUI's key map into `up @ down` factors per model weight; checked offline against the core's own
patch math on all four community formats, the delta matches to 1e-7 (1e-6 after the QR-SVD). Split q/k/v LoRAs come out as one fused
`proj.qkv` factor of rank 3 x r (Anubis, asura: 96).

In ComfyUI (stock `LoraLoaderModelOnly` at 1.0, fisherman and watercolor fox prompts x seeds 1000 / 1001, mean absolute pixel difference).
The noise floor first: re-saving retro at its own rank (lossless, 100% energy) renders 9.3 / 5.4 / 3.6 / 5.8 from the original, and
loading retro + lenovo in the opposite order renders 5.0 / 12.4 / 7.8 / 12.3 from the usual order.

| Output | Size | Energy kept (mean / worst layer) | source LoRA effect | output vs source |
|---|---|---|---|---|
| Anubis (split, rank 32, 142 MB) resized to rank 32 (fused) | 126 MB | 97.9% / 80.3% | 36.4 / 60.3 / 29.9 / 37.5 | 4.0 / 13.2 / 3.4 / 3.5 |
| Anubis resized to rank 16 | 63 MB | 91.1% / 63.2% | same | 10.1 / 17.2 / 9.6 / 8.3 |
| Anubis, dynamic 0.90 (cap 32, mean rank 14.6) | 56 MB | 90.4% / 80.3% | same | 12.4 / 19.1 / 11.3 / 13.7 |
| retro (rank 32) + lenovo (rank 16) merged at rank 48 | 185 MB | 100% / 100% | 37.5 / 66.9 / 40.0 / 85.8 (both stacked) | 5.1 / 12.4 / 6.9 / 7.9 |
| retro + lenovo merged at rank 32 | 123 MB | 98.0% / 92.7% | same | 6.8 / 13.3 / 5.6 / 6.1 |

- The merges at rank 48 and 32 stay within the noise floor of loading the two LoRAs in a different order: one file instead of two
  with no measurable loss at rank 32.
- Resizing Anubis to rank 32 (fused, from 3 x 32 split) stays within the floor; rank 16 and dynamic 0.90 (about 90% energy) move the
  image visibly beyond it, as in the HaDeS Anima Suite below ~90%.

## Where LoRAs act (LoRA Block Weight)

HaDeS MOSS LoRA Block Weight at 1.0 everywhere renders the same image as the stock `LoraLoaderModelOnly` (difference 0.0).
`--mode lora`, each LoRA on the model it was trained from, 3 prompts (café woman, fisherman, watercolor fox) x 2 seeds, no SLG.
Swap-in share (only that group of the LoRA, reference: the model without it), layout / edges / texture / color:

| Group | retro-scifi-90s (SFT, ai-toolkit) | lenovo_moss (SFT, ai-toolkit) | AnubisCitronOC (Base, kohya) | asura MOSS Base (Base, ai-toolkit) |
|---|---|---|---|---|
| 00–05 | 0.44 / 0.54 / 0.32 / 0.34 | 0.38 / 0.43 / 0.58 / 0.20 | 0.40 / 0.40 / 0.47 / 0.24 | **0.52 / 0.68 / 0.40 / 0.35** |
| 06–11 | 0.26 / 0.26 / 0.12 / 0.12 | 0.34 / 0.35 / 0.49 / 0.17 | 0.27 / 0.23 / 0.27 / 0.14 | 0.35 / 0.40 / 0.18 / 0.14 |
| 12–17 | 0.24 / 0.26 / 0.08 / 0.12 | 0.37 / 0.33 / 0.59 / 0.21 | 0.22 / 0.16 / 0.20 / 0.15 | 0.34 / 0.34 / 0.22 / 0.25 |
| 18–23 | 0.34 / 0.42 / 0.13 / 0.13 | 0.40 / 0.40 / 0.66 / 0.25 | 0.30 / 0.30 / 0.20 / 0.20 | 0.27 / 0.35 / 0.12 / 0.15 |
| 24–29 | 0.65 / 0.79 / 0.34 / 0.34 | 0.63 / 0.77 / 0.90 / 0.44 | 0.47 / 0.55 / 0.44 / 0.18 | 0.39 / 0.51 / 0.16 / 0.16 |
| 30–35 | **0.70 / 0.81 / 0.62 / 0.55** | **0.72 / 0.65 / 1.09 / 0.57** | 0.59 / 0.65 / 0.83 / 0.52 | **0.53 / 0.54 / 0.42 / 0.52** |
| 36–41 | 0.58 / 0.49 / 0.42 / 0.62 | 0.42 / 0.39 / 0.59 / 0.40 | **0.54 / 0.47 / 1.00 / 0.63** | 0.34 / 0.30 / 0.34 / 0.31 |

Leave-out (the full LoRA without that group, reference: the full LoRA) agrees on the strong groups; asura's leave-out puts 00–05 first
(0.61 / 0.70 / 0.93 / 0.57).

- Layers 30–35 carry a large share in all four. Beyond that each LoRA differs: retro and lenovo act most in 24–35, Anubis in 30–41
  (texture and color in 36–41), asura in 00–05 and 30–35. lenovo moves texture in every group, retro barely in 06–23.
- No group isolates one attribute, and a value above 1 means that group alone moves the metric more than the whole LoRA (groups offset
  each other).
- Layer 22: group 18–23 is weak for all four, the kohya LoRA that trains layer 22 (Anubis) included, so leaving it out (ai-toolkit) shows
  no difference here.
- A fixed "style" / "identity" group preset would be wrong for some of these LoRAs, so LoRA Block Weight has none; measure a LoRA with
  `tools/moss_eval.py --mode lora --lora <name> --lora-model <model it was trained on>`.

## Community LoRA formats in the core loader

Four community LoRAs on `mossImage_sft_bf16` (fused `proj.qkv`) through the stock `LoraLoaderModelOnly`, strength 1.0: the ComfyUI log
reported no skipped key for any of them and every one changed the image.

| Key format | Example | q/k/v | Loaded |
|---|---|---|---|
| kohya `lora_unet_layers_N_proj_q.lora_down/up` + `alpha` (sd-scripts on the Diffusers files) | AnubisCitronOC_MOSS_v1 | split | all keys |
| PEFT `diffusion_model.layers.N.proj.q.lora_A/B`, no alpha | asura MOSS Base | split | all keys |
| PEFT `diffusion_model.layers.N.proj.qkv.lora_A/B` (ai-toolkit on the Comfy-Org files) | lenovo_moss, retro-scifi-90s | fused | all keys |

- The core already maps split q/k/v LoRAs onto fused files, so the suite needs no key remapper (unlike Anima).
- The three ai-toolkit / PEFT LoRAs leave out layer 22, the layer the official Base template skips for SLG; the kohya one trains all 42.
- None of them touches `text_embedder`, the modulations or `io`.

## Phase 0: bias, SFT vs Base, text and layout

As in the templates, SFT samples without SLG and Base with SkipLayerGuidanceDiTSimple on single block 22 for the first half.

**Ethnic bias** (`--mode bias`, seeds 1000+):

| Prompt | SFT | Base |
|---|---|---|
| `photograph of a woman sitting in a café`, 16 seeds | ~10/16 East Asian | ~2–3/16 East Asian (some figures too small to tell) |
| elderly fisherman portrait (no ethnicity given), 6 seeds | 6/6 East Asian | 2/6 East Asian |
| Scandinavian blonde / Black man with beard / older Latina, 6 seeds each | 18/18 as asked | 18/18 as asked |

- Asked for an ethnicity, neither model drifts. The bias shows only when the prompt leaves it open, and mostly in SFT.
- SFT gives polished, uniform stock-photo looks; Base varies far more (black and white, film, polaroid, collages, small figures).

**Text and layout** (`--mode compare`, 6 seeds, SFT and Base):
- Bakery sign with two lines of text: spelled right in 12/12.
- Three lines given with `\n` in prose: three lines in 12/12.
- Layout JSON (core `MOSSImageLayoutPrompt` format): boxes followed and text spelled right in 12/12, but the `\n` inside a text item
  (`NORTHERN\nLIGHTS`) was ignored in 11/12 (one line). The box was wide and low (832 x 192 px), which may favor one line.
- Some seeds return collages or a framed poster on white, in both models.

**Line breaks in a layout text item** (SFT, 6 seeds, `NORTHERN\nLIGHTS` as the title of the same poster layout):

| Title box | `\n` escaped by json.dumps (as the core node and OpenMOSS' compiler write it) | literal newline in the prompt |
|---|---|---|
| 832 x 192 px (one-line proportions) | two lines in 3 / 6 | two lines in 3 / 6, one more with a shrunken second line |
| 640 x 384 px | two lines in 6 / 6 | two lines in 6 / 6 |

The box shape decides, not the encoding: in a box with one-line proportions MOSS may set the text on one line; in a box that fits
the lines it keeps them. OpenMOSS' own layout reference asks to check that line breaks and text length fit the box.

## Layer groups (SFT -> Base)

`--mode groups`, HaDeS MOSS Model Block Merger, 3 prompts (café woman, fisherman portrait, watercolor fox) x 4 seeds, no SLG in any
config. `share` = distance(variant, reference) / distance(Base, SFT). Swap-in = SFT with only that group from Base (reference SFT).
Leave-out = Base with only that group from SFT (reference Base). Metrics: layout / edges / texture / color (see the script docstring).

| Group | Swap-in | Leave-out |
|---|---|---|
| layers 00–05 | 0.35 / 0.31 / 0.23 / 0.17 | 0.40 / 0.33 / 0.32 / 0.20 |
| layers 06–11 | 0.22 / 0.23 / 0.13 / 0.10 | 0.28 / 0.21 / 0.19 / 0.12 |
| layers 12–17 | 0.26 / 0.24 / 0.14 / 0.11 | 0.31 / 0.23 / 0.23 / 0.17 |
| layers 18–23 | 0.40 / 0.40 / 0.20 / 0.20 | 0.48 / 0.48 / 0.32 / 0.27 |
| **layers 24–29** | **0.66 / 0.73 / 0.50 / 0.44** | **0.76 / 0.73 / 0.57 / 0.56** |
| **layers 30–35** | **0.63 / 0.65 / 0.61 / 0.37** | **0.54 / 0.52 / 0.49 / 0.44** |
| layers 36–41 | 0.31 / 0.31 / 0.17 / 0.21 | 0.39 / 0.33 / 0.25 / 0.21 |
| text_embedder | 0.30 / 0.24 / 0.12 / 0.14 | 0.32 / 0.26 / 0.23 / 0.14 |
| time_modulation | 0.27 / 0.22 / 0.25 / 0.17 | 0.31 / 0.26 / 0.21 / 0.18 |
| io | 0.16 / 0.12 / 0.11 / 0.07 | 0.20 / 0.13 / 0.20 / 0.08 |

Findings:
- Layers 24–35 carry most of the SFT -> Base change, on every metric. Layers 06–17 and `io` carry the least.
- As in Anima, no group isolates one attribute: each moves layout, edges, texture and color together.
- **Bias: not localized by this run.** Only one of the 4 café seeds was East Asian in SFT, and Base without SLG was itself East
  Asian in 3/4 fisherman seeds (with SLG in phase 0: 2/4 on the same seeds), so the run has little contrast to work with. In the one
  fisherman seed where SFT is East Asian and Base is not (1003), no single group from Base removed it, while Base with layers 24–29
  from SFT turned East Asian. Layers 24–29 take part, but one group alone is not enough to undo it.

## Bias by merge (`--mode bias_merge`)

16 seeds (1000–1015) per prompt, East Asian faces counted by eye (ambiguous faces counted as half, so counts are approximate).
SLG = the Base template's SkipLayerGuidanceDiTSimple (single block 22, first half).

| Config | Café woman | Fisherman |
|---|---|---|
| SFT | 10/16 | 16/16 |
| SFT + SLG | ~10/16 | 16/16 |
| Base | ~2/16 | ~5–6/16 |
| Base + SLG | ~2–3/16 | ~3–4/16 |
| SFT with layers 24–35 from Base | ~3/16 | ~10/16 |
| SFT with layers 18–41 from Base | ~2–3/16 | ~9/16 |
| SFT with everything but `io` from Base | ~2/16 | ~4–5/16 |

Findings:
- SLG does not change SFT's bias; on Base it moves a seed or two.
- The bias is spread over the model, not held by one group. Layers 24–35 from Base remove most of it on the café prompt
  but only about a third on the fisherman, and taking 18–41 instead does no better.
- Removing it on both prompts needs nearly all of Base, and then the image is Base's: everything but `io` from Base renders
  almost the same images as Base, so the SFT look does not live in `io`.
- Layers 24–35 from Base land between the two models in look as well (more film / black-and-white frames than SFT, some SFT polish
  left). There is no merge here that keeps SFT's look and drops its bias, so the merger gets no "de-bias" preset.
- Naming the ethnicity in the prompt works on every model (phase 0), which stays the reliable fix.
