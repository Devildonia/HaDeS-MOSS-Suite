# Measurement results (2026-11-03)

Measured with `tools/moss_eval.py` on an RTX 5090, ComfyUI 0.43.0 local API. 1024x1024, er_sde / simple / 25 steps / CFG 2.5,
model shift 5.0, negative `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` (the official ComfyUI templates),
text encoder `mossImage_base_txt_int4` (w4a8 with codebook), VAE `flux2-vae`. Models: `mossImage_sft_bf16` and `mossImage_base_bf16`
(fused `proj.qkv`, 42 layers). Ethnicity counts are by eye on the contact sheets.

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
