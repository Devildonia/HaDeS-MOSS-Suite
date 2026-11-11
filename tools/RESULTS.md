# Measurement results (2026-11-03)

[MOSS-edit layer ablation (2026-11-10)](#moss-edit-layer-ablation-2026-11-10).

## Original benchmark environment

Measured with `tools/moss_eval.py` on an RTX 5090, ComfyUI 0.43.0 local API. 1024x1024, er_sde / simple / 25 steps / CFG 2.5,
model shift 5.0, negative `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` (the official ComfyUI templates),
text encoder `mossImage_base_txt_int4` (w4a8 with codebook), VAE `flux2-vae`. Models: `mossImage_sft_bf16` and `mossImage_base_bf16`
(fused `proj.qkv`, 42 layers). Ethnicity counts are by eye on the contact sheets. ComfyUI runs with `--use-sage-attention` (the
launcher's flag, read from the server on 2026-11-04), so every image here was sampled with SageAttention.

## Few-step LoRAs

Two community LoRAs, both applied to `mossImage_sft_bf16`: `lodestones-8-steps-dmd2-lora` (DMD2, rank 128, fp32) and
`moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16` (4 steps, resized from rank 128 to 99% Frobenius, ranks 2–121 per layer).
Both train every linear of all 42 layers with split q/k/v and nothing outside the layers.

**Correction (2026-11-05):** `lodestones-8-steps-dmd2-lora` came from a Civitai reupload described as an SFT LoRA. By SHA256
(`7683fe1f…`) it is `8-steps-dmd2-lora.safetensors` from `lodestones/Kromoss` (also `lora_dmd2.safetensors` in
`lodestones/MOSS-Image-Base-8-steps-dmd`): an 8-step DMD2 LoRA distilled from **MOSS-Image Base**, for Base and Kromoss. Every
"lodestones" / "DMD2" result in this file is that Base LoRA run on SFT, not lodestones' SFT LoRA
(`lodestones/MOSS-Image-SFT-8-steps-dmd`, `lora.safetensors`), which has not been tested here. akhaliq's 4-step v2.0 file matches
its own repo (`154c7f14…`).

**Size of the change** (‖ΔW‖ / ‖W‖ against SFT, mean per layer group, both LoRAs alike): 0.03% in layers 0–23, 0.06% in 24–29,
0.08–0.11% in 30–41; the largest are q / k in 32–41 and `ffn_up_gate` (up to 0.16%). In the 4-step LoRA, `v`, `ffn_up_gate` and
`ffn_down` of layers 24–38 kept rank 2–5 at 99% energy: there the change is nearly one direction per weight.

**bf16 rounding.** bf16 steps are about 0.4% of a weight, larger than these changes. The core LoRA loader adds the delta in fp32
and rounds to the weight's dtype to nearest (`comfy/float.py`, stochastic rounding only for fp8). The delta applied to the bf16
SFT weights, against the LoRA's own delta, checked offline on seven weights:

| Weights | Elements changed | Applied delta / intended (norm) | Cosine to intended |
|---|---|---|---|
| layers 0, 10, 22 (q, o, ffn_down) | 7–13% | 0.54–0.66 | 0.41–0.50 |
| layer 28 v | 13–14% | 0.79–0.80 | 0.61–0.62 |
| layers 32, 36, 40 (q, ffn_up_gate, k) | 21–30% | 0.93–1.00 | 0.76–0.87 |

Loaded the usual way on a bf16 model, these LoRAs therefore apply about half their change in the early and middle layers. The
core bypass loader (`LoraLoaderBypassModelOnly`, "for debugging") leaves the weights as they are and runs the LoRA beside them, but
it skips split q/k/v on a fused `proj.qkv` model (126 "not in model state_dict" warnings per LoRA); the bypass runs below load copies
with fused `proj.qkv` written by HaDeS MOSS LoRA Resize at rank 512 (lossless: the qkv rank is the sum of q, k and v), which load
with no warning.

**Images** (`--mode turbo`, euler / simple / CFG 1 as the cards give, 8 steps for DMD2 and 4 for the 4-step LoRA, against SFT at
er_sde / simple / 25 steps / CFG 2.5; café woman, fisherman, watercolor fox x seeds 1000–1002, SageAttention):

| Config | Seconds per image | Weights vs bypass, layout / edges / texture / color | By eye |
|---|---|---|---|
| SFT, 25 steps (reference) | 15.14 | | |
| SFT alone, 8 steps, CFG 1 | 2.72 | | broken (blur, smeared faces) |
| DMD2, 8 steps, usual loader | 2.72 (5.6x faster) | 10.1 / 0.42 / 0.57 / 3.8 | clean |
| DMD2, 8 steps, bypass | 3.07 | | clean |
| SFT alone, 4 steps, CFG 1 | 1.54 | | broken |
| 4-step LoRA, usual loader | 1.54 (9.8x faster) | 13.1 / 0.61 / 0.47 / 7.3 | clean, one café image with a ghosted double exposure |
| 4-step LoRA, bypass | 1.72 | | clean |

- Both LoRAs do the work: the same SFT at their step counts without them is broken.
- The rounding of the usual loader moves the images about half as much as a new seed (layout 10–13) and gave the one artifact of
  the 18 LoRA images; bypass costs 11–13% more time. Which of the two looks better on the clean images is not measured.
- Next to the reference, both LoRAs give polished, centred, softly lit images; seed-to-seed variety is not measured with enough
  seeds to say whether they narrow it.
- Merging these LoRAs into a bf16 file (as HaDeS MOSS Model Saver would) rounds the same way as the usual loader.

**Where the speed-up lives** (`--mode turbo_groups`: each LoRA at its step count with only some layer groups, through HaDeS MOSS
LoRA Block Weight, which at 1.0 everywhere renders the stock loader's image exactly; distance to the full LoRA's image of the same
seed, layout / edges / texture / color):

| Groups kept | DMD2, 8 steps | By eye | 4-step LoRA | By eye |
|---|---|---|---|---|
| 0–23 | 15.9 / 0.77 / 2.48 / 13.0 | broken, as SFT without the LoRA | 12.9 / 0.77 / 4.04 / 13.2 | broken |
| 24–41 | 4.4 / 0.12 / 0.10 / 0.9 | the full LoRA's image | 2.8 / 0.10 / 0.08 / 0.7 | the full LoRA's image |
| 30–41 | 6.9 / 0.27 / 0.18 / 2.1 | same scenes, small changes, clean | 6.2 / 0.27 / 0.18 / 2.2 | same scenes, clean |
| 36–41 | 15.0 / 0.70 / 0.49 / 7.5 | other images, still clean | 12.4 / 0.70 / 1.82 / 9.4 | broken (ghosting, blur) |

- Both LoRAs do their work in layers 24–41: that half alone renders practically the full LoRA's image (closer to it than the
  usual loader is to bypass), and layers 0–23 alone do nothing usable. 30–41 is still enough for both; 36–41 is enough only for DMD2.
- The early layers, where the usual loader loses about half of the change to bf16 rounding, are the ones that matter least, which
  is why the usual loader works anyway.

**Reduced LoRAs** (`--mode turbo_files`, usual loader): layers 24–41 only, from the fused copies, each weight cut to rank r by the
exact SVD of LoRA Resize, factors in bf16. Distance to the full LoRA's image of the same seed, layout / edges / texture / color:

| Layers 24–41 at rank | DMD2: size | DMD2 distance | 4-step: size | 4-step distance |
|---|---|---|---|---|
| (original file, all layers) | 1189 MB | 0 | 327 MB | 0 |
| full rank | 340 MB | 4.3 / 0.11 / 0.09 / 0.9 | 65 MB | 3.0 / 0.10 / 0.08 / 0.7 |
| 32 | 57 MB | 4.4 / 0.11 / 0.10 / 1.0 | 31 MB | 2.9 / 0.10 / 0.11 / 0.7 |
| 16 | 28 MB | 3.6 / 0.10 / 0.10 / 0.7 | 19 MB | 3.3 / 0.11 / 0.09 / 0.7 |
| 8 | 14 MB | 3.0 / 0.10 / 0.08 / 0.8 | 11 MB | 2.9 / 0.11 / 0.10 / 0.5 |
| 4 | 7 MB | 4.1 / 0.12 / 0.09 / 0.8 | 6 MB | 3.1 / 0.12 / 0.09 / 0.6 |
| 2 | 3.6 MB | 4.8 / 0.13 / 0.12 / 1.0 | 3.6 MB | 3.9 / 0.16 / 0.12 / 1.1 |

- Every reduced file renders the full LoRA's images by eye, rank 2 included, at the same speed (2.7 s / 1.5 s per image); the 4-step
  LoRA's ghosted café image comes back in every version, so they reproduce the LoRA, defects too. The distances stay at the level the
  full-rank 24–41 file has (its 5–6 mean pixel difference from the Block Weight route to the same layers is bf16 rounding noise).
- On these 9 images (three prompts, 1024 x 1024) a 3.6 MB rank-2 LoRA on layers 24–41 replaces the 1189 MB DMD2 LoRA.

Text, layout and 2048 x 2048, against each original LoRA file rendered in the same run (usual loader, 3 seeds per prompt):

| Version | Text and layout, 1024 x 1024: layout distance / strings right | 2048 x 2048: seconds / layout distance |
|---|---|---|
| DMD2 original | 0 / 9 of 9 | 12.81 / 0 |
| DMD2, 24–41 rank 8 | 2.4 / 9 of 9 | 12.61 / 3.3 |
| DMD2, 24–41 rank 2 | 2.8 / 9 of 9 | 12.65 / 4.5 |
| 4-step original | 0 / 9 of 9 | 7.02 / 0 |
| 4-step, 24–41 rank 8 | 2.2 / 9 of 9 | 6.95 / 2.2 |
| 4-step, 24–41 rank 2 | 2.7 / 9 of 9 | 6.97 / 2.9 |

- Text: the bakery sign ("FRESH BREAD DAILY" / "open 7am - 3pm"), the three-line poster and the layout JSON poster ("NORTHERN
  LIGHTS" / "14 MARCH 2027 · OSLO") are spelled right in all 54 images, the reduced versions setting the same text as the originals.
- 2048 x 2048: the reduced versions render the originals' images. DMD2 at 8 steps takes 12.8 s against 72.0 s for SFT at 25 steps
  (5.6x). The 4-step LoRA at 4 steps comes out darker and hazier at this size; its card asks for 6–8 steps above 1024, not measured.
**Can the directions be read from the model?** Offline, per weight of layers 24–41 (fused files): the rank-2 directions of each
LoRA's delta, compared between the two LoRAs and with the top 64 singular directions of the SFT weight (fraction of the LoRA's
direction inside that subspace; chance = 64 / dimension). Means over the 18 layers:

| Weight | DMD2 vs 4-step, output / input side (chance ~0.0005) | Inside SFT's top 64, output side (chance) | Inside SFT's top 64, input side (chance) | cos(ΔW, W) |
|---|---|---|---|---|
| qkv | 0.54 / 0.22 | 0.012 (0.007) | 0.166 (0.021) | 0.000 |
| o | 0.42 / 0.11 | 0.022 (0.021) | 0.021 (0.021) | 0.000 |
| ffn_up_gate | 0.33 / 0.21 | 0.007 (0.004) | 0.051 (0.021) | 0.000 |
| ffn_down | 0.40 / 0.21 | 0.049 (0.021) | 0.010 (0.008) | 0.000 |

- The two LoRAs share a large part of their directions, far above chance, but the 4-step LoRA's own card says it was distilled
  from lodestones' DMD (his SFT one, not the Base LoRA measured here), so the two are not independent evidence of a property of the
  model.
- The directions are not the weight's own: at most 17% of a direction lies in SFT's top 64 (qkv input side, 8x chance: the
  LoRA reads from residual directions qkv already reads strongly), mostly near chance, and ΔW is orthogonal to W (no rescaling).
  A turbo patch computed from the SFT weights alone, by their spectrum, is not supported by this; whether activations would give
  the directions is not measured.
**Official `akhaliq/MOSS-Image-SFT-4-steps` files** (`moss_image_sft_4step_v2.0`, `_3step_v1.0`, `_2step_v0.1`; each rank 128,
fp32, all 42 layers, split q/k/v). Their deltas against each other and against the files above, cosine on three weights
(layer 5 q, layer 30 ffn_down, layer 36 ffn_up_gate):

| Pair | 5 q | 30 ffn_down | 36 ffn_up_gate |
|---|---|---|---|
| the 4-step file measured above vs official 4-step v2.0 | 0.99 | 1.00 | 1.00 |
| 4-step v2.0 vs 3-step v1.0 | 1.00 | 1.00 | 1.00 |
| 4-step v2.0 vs 2-step v0.1 | 0.98 | 0.99 | 1.00 |
| 4-step v2.0 vs lodestones DMD2 | 0.00 | 0.57 | 0.53 |

- The 4-step file measured above is a resize of the official 4-step v2.0, so the results above hold for it. The three official
  files are nearly the same LoRA, used at 4, 3 or 2 steps.
- Against DMD2 they are unrelated in the early layers and half aligned in the late ones: the two trainings agree where the LoRAs do
  their work (layers 24–41) and not where they do not.

Cut to layers 24–41 at rank 8 and 2 (fused qkv, from the official files), each at its own step count, against the original file:

| LoRA, steps | Seconds per image | rank 8, photo / text layout distance | rank 2, photo / text layout distance | By eye |
|---|---|---|---|---|
| 4-step v2.0, 4 | 1.54 | 3.8 / 3.1 | 4.5 / 3.6 | cuts render the original's images |
| 3-step v1.0, 3 | 1.25 | 2.6 / 2.0 | 3.4 / 3.2 | cuts render the original's images |
| 2-step v0.1, 2 | 0.96 | 1.7 / 1.6 | 2.4 / 2.1 | original already hazy, glyph edges soft; the cuts the same |

- Text: every string is spelled right in all 81 text and layout images (3 LoRAs x 3 versions x 3 prompts x 3 seeds), cuts included.
- The 2-step LoRA's haze and soft glyphs are the LoRA's own (the original shows them); the rank-2 cut keeps the original's look at
  every step count tried.
- Base and other sizes with the official files are not measured.

**What the rank-2 change reads** (`tools/moss_activations.py`, the official 4-step cut to layers 24–41 at rank 2 on SFT, 4 steps,
euler / simple / CFG 1, café woman / fisherman / fox x seeds 1000–1001, 1024 x 1024). For each weight the change adds `up @ z`
with `z = down @ x` (2 numbers per token). share = mean² / (mean² + variance) of z over the 4096 image tokens of one image (1: the
same z on every token); spread = std of the per-image means over the 6 images / their mean magnitude. Means over the 18 layers and
both z components:

| Weight | share, steps 0 / 1 / 2 / 3 | spread between images, steps 0 / 1 / 2 / 3 |
|---|---|---|
| qkv | 0.91 / 0.88 / 0.87 / 0.87 | 0.10 / 0.08 / 0.13 / 0.14 |
| o | 0.59 / 0.56 / 0.55 / 0.55 | 0.39 / 0.36 / 0.49 / 0.43 |
| ffn_up_gate | 0.59 / 0.59 / 0.60 / 0.61 | 0.33 / 0.36 / 0.36 / 0.33 |
| ffn_down | 0.59 / 0.61 / 0.61 / 0.61 | 0.42 / 0.27 / 0.35 / 0.37 |

- On qkv the change reads nearly the same z on every token and in every image (share ~0.88, spread ~0.1): there it acts close to a
  fixed per-layer, per-step bias on q, k and v.
- On o and the FFN, about 60% of z is a per-image constant and 40% varies by token, and the constant part changes by 30–50%
  between images: there the change depends on the content.

**Fixed biases in place of the LoRA** (`tools/moss_bias_turbo.py`, same LoRA and settings). Calibration on café woman / fisherman /
fox x seeds 1000–1001 with the LoRA: for every LoRA weight, step and token kind (image, text), the mean of the change it adds,
`up @ mean(down @ x)`. Test on held-out seeds 2000–2002 and four prompts (the three above and the bakery sign), 12 images per config:

| Config | Distance to the LoRA's image, layout / edges / texture / color | By eye |
|---|---|---|
| the rank-2 LoRA (reference) | — | clean |
| LoRA without its qkv part | 9.7 / 0.48 / 0.61 / 4.6 | artifacts: ghosted strokes on one café image, "B&EAD" and a broken "DAILY" on the sign |
| LoRA without qkv + calibrated qkv biases | 10.4 / 0.42 / 0.40 / 4.3 | clean, sign spelled right in 3 of 3 |
| no LoRA, calibrated biases on all four weights | 14.3 / 0.55 / 0.52 / 7.9 | clean, sign spelled right in 3 of 3 |
| SFT alone, same 4 steps | 13.3 / 0.73 / 3.96 / 13.1 | broken (blur, smeared faces, unreadable sign) |

- With no LoRA at all, fixed vectors added to the outputs of the 72 late-layer weights (one per weight, step and token kind)
  turn SFT at 4 steps / CFG 1 from broken into clean images and correct text, on seeds the calibration never saw.
- The qkv part of the LoRA matters (without it the images get artifacts) and its fixed bias replaces it.
- The bias-only images differ from the LoRA's more (layout 14 against 10), and by eye they are more alike between seeds (same
  centred pose on the three café images); diversity is not measured.
- The biases are computed from the LoRA's own activations, so this replaces the LoRA's weights, not its training. Whether the
  biases can be found without a LoRA, and how they behave on Base, at other sizes or step counts, is not measured.

**One calibration prompt, and which steps** (`--calib-prompts illustration --bias-steps ...`): biases calibrated on the watercolor
fox alone (seeds 1000–1001), applied with no LoRA at some of the 4 steps only, tested on five prompts the calibration never saw
(café woman, fisherman, bakery sign, three-line poster, layout JSON poster) x seeds 2000–2002. Text = images with every string
right out of 9 (3 text prompts x 3 seeds):

| Biases at steps | Distance to the LoRA, layout / edges / texture / color | Text | By eye |
|---|---|---|---|
| the LoRA (reference) | — | 9 / 9 | clean |
| 0, 1, 2, 3 | 15.1 / 0.47 / 0.65 / 7.6 | 8 / 9 | clean; on layout seed 2000 the title letters overlap |
| 0 | 11.8 / 0.44 / 0.74 / 4.7 | 8 / 9 | composition and text in place, the whole image hazy and soft |
| 0, 1 | 13.3 / 0.45 / 0.43 / 5.2 | 8 / 9 | clean, like all four steps; layout seed 2000 garbled ("NORTHEAN") |
| 3 | 17.4 / 0.68 / 2.62 / 16.2 | 0 / 9 | broken, as SFT alone |
| 1, 2, 3 | 18.1 / 0.66 / 0.76 / 6.8 | 6 / 9 | sharp but off: invented captions on two café images, a fish in a fisherman's hand, doubled words |
| none (SFT alone) | 17.0 / 0.69 / 3.72 / 17.5 | 0 / 9 | broken |

- Biases calibrated on one watercolor prompt fix photos, the sign and the line-break poster at 4 steps; the layout poster fails on
  one of three seeds with every bias set (the LoRA gets it right), so they are general but not complete.
- The first step carries the structure: alone it gives the right composition and text, only soft; without it the image is sharp
  but the model invents content. Steps 0 and 1 together are as clean as all four, so steps 2 and 3 need no bias.

**What the biases are made of** (`--calib-only`, saved biases of three calibrations: fox alone, café woman alone, the three
prompts together). Each bias is `up @ mean(z)`, so its direction lies in the plane of the LoRA's two `up` columns by construction:
the cosines between calibrations (1.000 for qkv, o and ffn_up_gate, 0.98–1.00 for ffn_down, at every step, and step 0 against
steps 1–3 the same) only say that the two numbers of mean(z) keep their ratio. Those numbers, on one weight as an example
(layer 32 qkv, steps 0 / 1 / 2 / 3): fox (-19.5, -0.8) / (-22.3, -1.6) / (-23.1, -1.9) / (-22.1, -2.0); woman (-17.3, -0.5) /
(-21.5, -1.1) / (-22.0, -1.2) / (-21.1, -1.4). Bias norms, mean over the 18 layers:

| Weight | fox / woman norm ratio, steps 0 / 1 / 2 / 3 | norm, steps 0 / 1 / 2 / 3 (three prompts) |
|---|---|---|
| qkv | 1.15 / 1.00 / 1.00 / 1.00 | 6.4 / 7.6 / 7.9 / 7.6 |
| o | 1.39 / 1.19 / 1.16 / 1.15 | 5.8 / 6.4 / 6.6 / 6.1 |
| ffn_up_gate | 1.14 / 1.02 / 0.99 / 0.98 | 1.7 / 1.8 / 1.9 / 1.9 |
| ffn_down | 1.27 / 1.05 / 1.03 / 1.02 | 1.3 / 1.6 / 1.7 / 1.7 |

- One of the two numbers dominates (about 20 to 1 on the example), so each bias is close to one fixed vector per weight, the LoRA's
  first `up` column, times a scale.
- The scale barely depends on the prompt (within 0–20% from step 1 on, 14–39% at step 0) and changes by about 20% between steps.
- Across the 18 layers the vectors differ (9–12 of 18 components for 90% of the energy): one vector per layer and weight.
- So the measured turbo effect is close to adding one bias vector to each of the 72 late-layer linears (MOSS's linears have
  none), with a per-step scale: about 571 thousand numbers. Its directions still come from a LoRA; finding them without one, for
  example by fitting only these biases to SFT at many steps, is not measured.

**Biases fitted with no LoRA** (`tools/moss_bias_fit.py`, 512 x 512, PyTorch attention). Student: SFT at 4 euler / simple steps,
CFG 1, one bias per linear of layers 24–41 and per step (2.3 M numbers), zeros at the start. Teacher: the same SFT at 25 euler /
simple steps, CFG 2.5, biases off. Objective (`teacher_step`): from the student's state at each step the teacher runs its own
schedule down to the next student sigma, and the student's single euler step must land there (mean squared error). 24 training
prompts (photos, illustrations, signs, posters; none of them a test prompt) x 1 seed, Adam lr 5e-4, gradient clip 1.0, 200
iterations (about 19 minutes on the RTX 5090; one iteration with a non-finite gradient skipped).

- The objective does not score what the LoRA does: the LoRA-calibrated biases give a higher loss than zero biases with both
  objectives tried (`final`, the clean-image prediction against the teacher's last latent: 0.870 against 0.470 at 1024 x 1024;
  `teacher_step`: 0.029 against 0.017 at 512 x 512). The LoRA does not follow the teacher's path, so the loss cannot rank results,
  only the images can.
- Fitting lowered the loss from 0.0172 to 0.0029. A first run at lr 2e-3 with no clip went to NaN within 20 iterations.

Test of the fitted biases (`moss_bias_turbo.py --biases`, 512 x 512, held-out prompts and seeds 2000–2002):

| Config | Distance to the LoRA, layout / edges / texture / color | By eye |
|---|---|---|
| the 4-step LoRA (reference) | — | clean |
| fitted biases, all 4 steps | 12.1 / 0.59 / 0.73 / 7.2 | coherent scenes, sign spelled right in 3 of 3; faces over-textured, some clutter (fruit in a cup, a doubled fox), a little haze |
| fitted biases, steps 0–1 | 12.5 / 0.59 / 1.16 / 9.7 | softer and hazier, "DAiLY" on one sign |
| SFT alone, 4 steps | 14.6 / 0.75 / 1.90 / 13.7 | broken |

- With no LoRA and 19 minutes of fitting against SFT itself, the biases turn SFT at 4 steps from broken into coherent images with
  legible text, below the LoRA's quality. Unlike the LoRA-calibrated biases, the fitted ones need all four steps.
- The 512 x 512 fit transfers to 1024 x 1024: coherent scenes, the sign legible in 3 of 3 (rough letters on one), faces still
  over-textured.

Second round, 1000 iterations each on `--prompt-set large` (84 prompts: the 24 above, 48 subject x style pairs, 12 more text
designs) x 2 seeds, teacher at 12 steps instead of 25 to halve its cost, tested at 1024 x 1024 on the held-out prompts and seeds:

| Fit | Sign (3 seeds) | Café woman | Fisherman | Fox |
|---|---|---|---|---|
| first round: per-step biases, 24 prompts x 1, teacher 25 steps, 200 iterations | legible 3 / 3, one rough | coherent | coherent, over-textured | a little haze |
| A: shared (one vector per linear x per-step scale), large set, teacher 12 steps | "B&AD", smeared letters | ghosted, double-exposed | two of three hazy | hazier |
| B: per-step biases, large set, teacher 12 steps | legible 2 / 3 ("BFEAD" on one) | ghosted | two of three coherent | clearer than A, some haze |

- With the same data and teacher, the shared form (A) is worse than free per-step vectors (B): without a LoRA to give the
  direction, one vector per linear is too rigid.
- B with 7 times the data and 5 times the iterations is not better than the first round; the change it shares with A is the
  teacher at 12 steps, whose targets are less converged (euler at 15 steps still differs from 60, see "Samplers and schedulers").
  The teacher's step count is the likely cause; a run that changes only it is not measured.

C: B again with the teacher at 25 steps, the only change (14 of 1000 iterations skipped as non-finite, against 8 for B; 96 minutes):

| Fit | Sign (3 seeds) | Café woman | Fisherman | Fox |
|---|---|---|---|---|
| B (teacher 12 steps) | legible 2 / 3 | ghosted | two of three coherent | some haze |
| C (teacher 25 steps) | legible 3 / 3, one smudged "DAILY" | ghosted, blurred faces | three of three coherent, one over-textured | a little clearer than B, some haze |
| first round (24 prompts x 1, 200 iterations, teacher 25) | legible 3 / 3, one rough | faces clear, some ghosting | coherent, over-textured | a little haze |

- The better teacher helps (C over B on the sign and the fisherman), but C, with 7 times the data and 5 times the iterations, is
  still not better than the 200-iteration first round, and its café faces are blurrier. A longer fit on this objective seems to
  drift toward blur rather than away from it; checkpoints along one run were not kept, so where it peaks is not measured.

**The LoRA as the teacher** (`--teacher-lora HaDeS_test_akh4_L24-41_r2 --blend 1`): the teacher is SFT with the rank-2 4-step cut
added exactly (fp32, beside the weights) taking one euler step at CFG 1; from the student's state, the student's step must land
where the teacher's does. The student is SFT + biases, no LoRA. 24 prompts x 1 seed, 512 x 512, lr 2e-4, 300 iterations (5
minutes; 16 skipped as non-finite). Because of an ordering bug, fixed since, the run started from zero biases, not from the
calibrated ones it was given, and its "zero" and "compare" losses both measured the calibrated biases (0.00074); the fit ended at
0.00325.

Tested at 1024 x 1024 on the held-out prompts and seeds: clean images on all four prompts, the sign spelled right and sharp in 3 of
3, at the LoRA's quality by eye (layout distance to the LoRA's images 14.6: a different image, not a worse one). The café women
share one centred pose, as the LoRA-calibrated biases do.

- With a teacher that does what we want, fitting biases from zero works in 5 minutes, where the SFT teacher stalled at hazy
  images after 96. The objective, not the bias form or the fitting, was what held the SFT-teacher fits back.
- These biases come from a LoRA (Apache 2.0, akhaliq/MOSS-Image-SFT-4-steps) through its outputs, so they derive from it. The 4-step file measured here
  carries no source in its metadata (it says "Dynamic resize with sv_fro: 0.99 from 128").

**Retiring the LoRA teacher** (`--blend`): a chain of fits, each started from the previous one's biases (`--init-biases`), with the
teacher's step a blend of the LoRA teacher's and the SFT teacher's (blend x LoRA + (1 - blend) x SFT, both from the student's state).
Same settings as above, 300 iterations each (about 30 minutes; the first, blend 1, is the run above). Tested at 1024 x 1024, 4
steps, CFG 1, no LoRA, held-out prompts and seeds 2000–2002.

| Blend (LoRA share) | Fitted loss | Layout distance to the LoRA | Sign (3 seeds) | Faces and photos |
|---|---|---|---|---|
| 1.00 | 0.00325 | 14.6 | 3 / 3 sharp | clean; one café pose for every seed |
| 0.75 | 0.00126 | 12.8 | 3 / 3 sharp | clean |
| 0.50 | 0.00190 | 10.3 | 3 / 3 sharp, both lines | clean; café women vary by seed (seed 2002 a different woman from the LoRA's) |
| 0.25 | 0.00237 | 9.1 | 3 / 3 right, one "BREAD" rough | coherent; chalky, grainy texture on faces and the sign |
| 0.00 | 0.00272 | 11.1 | big letters right, small line garbled once and "DAIL," once | coherent; skin over-textured, translucent ghost objects (a cup) |

- Losses are not comparable across rows: each row has its own teacher.
- Quality holds down to a half-LoRA teacher and degrades below it the same way the SFT-teacher fits did (texture and ghosting,
  not blur), so starting from good biases does not keep the SFT objective from pulling them back. Fixed biases still need the few-step
  LoRA's step as part of their teacher.
- Layout distance to the LoRA falls as the LoRA share falls (to 0.25): the blended biases follow the 25-step SFT layout more; by eye
  that is where the extra variety at blend 0.5 comes from. Not measured against the 25-step SFT images.

**Our own few-step LoRA, trained by DMD** (`tools/moss_dmd_train.py`): akhaliq's recipe (teacher SFT at CFG 3 with the negative
pass skipping layer 22, a fake score trained 1:1 by flow matching, generator gradient v_real - v_fake at noise ~ Beta(2.5, 1), Adam
betas (0, 0.999), clip 1, EMA 0.999) with one bf16 base and hooked fp32 LoRAs. Generator LoRA at rank 2 on layers 24–41 starting
from zero (no warm start from any LoRA), trained on euler / simple, 4 steps, CFG 1; fake score rank 32 on all layers; lr 5e-5 /
1e-4; 1,024 generated prompts, batch 1; 512 x 512 only. 2,000 updates in 30 minutes (0.9 s per update, 26.0 GB peak, PyTorch
attention), 19 skipped as non-finite. The LoRA file is 7.1 MB in fp32.

The 512 x 512 preview prompts (not in the training set) went from SFT's 4-step blur at update 0 to clean images with the sign spelled
right at update 200, and stayed clean to 2,000.

Tested at 1024 x 1024 (never trained there) through the stock `LoraLoaderModelOnly`, euler / simple / 4 steps / CFG 1, SageAttention,
moss_eval `turbo_files` on the five COMPARE prompts below x seeds 1000–1002, against akhaliq's 4-step v2.0 file at the same settings:

| LoRA | Sign, both lines | Three-line poster | Portraits, street, watercolor | Detail vs akhaliq | Layout distance to akhaliq |
|---|---|---|---|---|---|
| akhaliq 4-step v2.0 (rank 128, all layers, 1.2 GB) | 3 / 3 | 3 / 3 | clean | 0 | 0 |
| ours, update 1,000 | 3 / 3 | 3 / 3 | clean | +0.374 | 11.0 |
| ours, update 2,000 | 3 / 3 | 3 / 3 | clean | +0.264 | 13.0 |

- A rank-2 LoRA on layers 24–41, trained from zero in 30 minutes at 512 x 512 on one GPU, renders these prompts at 1024 x 1024 at the
  quality of the published 4-step LoRA by eye, text included, with a little more measured detail. Same speed (1.5 s per image).
- Less variety across seeds than akhaliq's: the sign is a frontal board in all three seeds and the street is a centred
  vanishing-point view each time, where akhaliq's vary angle and decoration. Not measured beyond these three seeds.
- Not tested yet: 2048 x 2048, other aspect ratios, layout JSON prompts, the ethnic-bias prompts, a larger seed count.

**Continuing at 1024 x 1024 (v0.2):** the update-1,000 LoRA above as the starting generator (`--init-lora`), 600 updates at
1024 x 1024, the first 100 training only a fresh fake score (`--fake-warmup 100`), generator lr 2e-5, per-block gradient checkpointing
(`--checkpoint`; at 512 x 512 its first-update gradient matched the run without it, 6565 against 6542 before clipping, and the peak
fell from 25.2 to 11.1 GB). 37 minutes (about 3.7 s per full update), 13.3 GB peak, none skipped. Same test as above with seeds
1000–1005 and the café woman prompt added. Variety = mean pixel difference between every pair of seeds of one prompt at 64 x 64
(0–100, higher = seeds differ more):

| LoRA | Sign | Poster | Portrait | Street | Watercolor | Café woman | Mean variety |
|---|---|---|---|---|---|---|---|
| akhaliq 4-step v2.0 | 6 / 6 | 6 / 6 | clean | clean | clean | clean | 13.18 |
| v0.1, update 1,000 (512) | 6 / 6 | 6 / 6 | clean | clean | clean | clean | 12.65 |
| v0.2, update 300 (1024) | 6 / 6 | 6 / 6 | clean | clean | clean | clean | 12.24 |
| v0.2, update 600 (1024) | 6 / 6 | 6 / 6 | clean | clean | clean | clean | 11.88 |

- The 1024 continuation changed little: same compositions seed for seed, skin a little softer and less contrasted, wood grain on the
  sign smoother. By eye neither is better; no gain in quality was found.
- Variety went down step by step (12.65 -> 12.24 -> 11.88), further from akhaliq's 13.18: more of the same DMD objective narrows the
  seeds, as the mode-seeking of DMD would predict. Raising quality or variety needs a different change than more updates at 1024.

**Biases on top of our LoRA** (`moss_bias_fit.py --student-lora`): the student is SFT + the v0.1 update-1,000 LoRA (exact, fp32,
beside the weights) + per-step biases on layers 24–41 from zero; the teacher's step is half the same LoRA's step and half SFT at 25
steps / CFG 2.5 (`--teacher-lora` the same file, `--blend 0.5`, `teacher_step`), the blend that kept quality in the bias chain above.
512 x 512, the 24 base prompts x 1 seed, lr 2e-4, 300 iterations, 31 minutes, none skipped; loss 0.00736 with zero biases (the LoRA
alone), 0.00209 fitted. Tested at 1024 x 1024 with `moss_bias_turbo.py --biases-on-lora` (stock loader for the LoRA, biases added
by hooks, SageAttention), 6 held-out prompts x seeds 2000–2005:

| Prompt | LoRA alone | LoRA + biases |
|---|---|---|
| Three-line poster | 6 / 6 | 3 / 6: letters broken into fragments on three seeds |
| Sign | 6 / 6 | 6 / 6 large text; small line rough on one, split over two lines on another |
| Fisherman, street, watercolor | clean | hazier, lower contrast, darker; the street more film-like |
| Café woman | one centred pose, 6 / 6 East Asian | poses and framing vary, 2 / 6 not East Asian |
| Mean variety (as above) | 12.60 | 12.40 |

- The biases pull the LoRA toward SFT's 25-step images, with what that brought in the bias chain: more varied compositions (the
  café prompt), but haze, lower contrast and broken poster lettering. A loss three times lower did not give better images.
- The pixel variety measure does not see the café change, as the lower contrast shrinks every pairwise difference; by eye the biased
  seeds differ more. Ethnicity on 6 seeds is too few to count as a change in the bias.
- On a LoRA that already renders clean images, half-SFT biases cost quality; no gain in quality was found. Other blends were not
  tried.

**DMD plus a regression term (v0.4)** (`--reg-pairs 256 --reg-weight 0.25`, DMD v1's term without its LPIPS): before training, the
DMD teacher (CFG 3, layer 22 skipped in the negative pass) renders 256 training prompts at 25 euler / simple steps from fixed noises
(18.6 minutes at 512 x 512). Each update then also rolls the generator from one pair's noise down to a random step k (no gradient)
and adds 0.25 x (its step-k prediction - the teacher's latent) to the gradient of that prediction, beside the DMD gradient. Everything
else as v0.1 (from zero, 512 x 512, 2,000 updates, same seed): 1.2 s per update, 26.1 GB peak, 18 skipped. Regression loss (mean
squared, latent): 0.36 at update 1, 0.43–0.66 to update 1,000, 0.39 at 2,000. Same 1024 x 1024 test as v0.2:

| LoRA | Poster | Sign | Café woman | Fisherman | Street | Watercolor | Mean variety |
|---|---|---|---|---|---|---|---|
| akhaliq 4-step v2.0 | 6 / 6 | 6 / 6 | clean | clean | clean | clean | 13.18 |
| v0.1, update 1,000 | 6 / 6 | 6 / 6 | clean, one pose | clean | clean | clean | 12.65 |
| v0.4, update 1,000 | 1 / 6 | 3 / 6 (doubled, chalky letters) | ghosted, double exposures | gritty, a little hazy | clean | smeared | 14.52 |
| v0.4, update 2,000 | 4 / 6 | 4 / 6 (chalky, doubled on two) | varied poses, ghosting on two | varied, clean | clean | looser, some smears | 14.83 |

- The regression term brings the variety back, past akhaliq's (14.83 against 13.18, highest on the café, the sign and the street).
  Seed by seed, v0.4's images follow the same composition as akhaliq's (the scarf on fisherman seed 1002, the croissant on café seed
  1004, the ghost lettering on café seed 1001): the noise-to-image map of the 25-step teacher, which akhaliq's LoRA kept as well.
- It costs quality: ghosting and double exposures, doubled letters. That is what a mean-squared pull toward a fixed image does where
  the generator cannot predict it (it averages candidates); DMD v1 used LPIPS on decoded images instead. From update 1,000 to 2,000
  the ghosting receded and text improved, so the run had not settled.

**v0.4 continued by DMD alone (v0.5):** v0.4's update-2,000 LoRA as the starting generator, 700 updates at 512 x 512 with no
regression term, the first 100 training only a fresh fake score, lr 5e-5 (v0.1's). 0.8 s per update, 4 skipped. Same test:

| LoRA | Poster | Sign | Café woman | Fisherman | Watercolor | Mean variety |
|---|---|---|---|---|---|---|
| v0.4, update 2,000 | 4 / 6 | 4 / 6 | varied, ghosting on two | varied | looser, smears | 14.83 |
| v0.5, update 400 (300 DMD updates) | 6 / 6 | 6 / 6 | clean, one pose | clean, one pose | clean | 13.56 |
| v0.5, update 700 (600 DMD updates) | 6 / 6 | 6 / 6 | clean, one pose | clean, one pose | clean | 12.93 |

- 300 DMD updates remove the ghosting and the doubled letters, and with them most of the teacher's compositions: the café, the
  fisherman and the fox are back to one pose for every seed, as in v0.1. What stays varied is the sign (18.1) and the poster's skyline;
  the variety measure falls back toward v0.1's as DMD continues.
- DMD's pull to one image per prompt erases the regression's seed-to-image map within a few hundred updates. The two terms would have
  to act together through training (a smaller regression weight kept on, or a weight that falls slowly), not one after the other.

**A smaller regression weight, kept on (v0.6):** v0.4 again with `--reg-weight 0.05` instead of 0.25 (the same 256 teacher pairs,
reused from v0.4's cache: the first update's regression loss matched, 0.3599), from zero, 2,000 updates. 1.3 s per update, 23
skipped. Regression loss 0.6–0.75 from update 1,000 on, against 0.39 for v0.4: a looser pull. Same test:

| LoRA | Poster | Sign | Café woman | Fisherman | Watercolor | Mean variety |
|---|---|---|---|---|---|---|
| akhaliq 4-step v2.0 | 6 / 6 | 6 / 6 | clean, varied | clean | clean | 13.18 |
| v0.1, update 1,000 | 6 / 6 | 6 / 6 | clean, one pose | clean, one pose | clean, one pose | 12.65 |
| v0.6, update 1,000 | 6 / 6 | 6 / 6 | clean, varied (women, cups, food) | clean, varied (scarf, framing) | clean, varied | 13.51 |
| v0.6, update 2,000 | 6 / 6 | 6 / 6 | clean, a little less varied | clean, varied | clean, varied | 13.41 |

- No ghosting and every string right, and the variety of akhaliq's LoRA or a little more. Seed by seed the images keep features of the
  25-step teacher's (the scarf on fisherman seed 1002, a round orange object on café seed 1004 where the teacher and akhaliq have a
  croissant), so the regression term still sets the compositions while DMD keeps them sharp.
- The sign gains the decorations akhaliq's has (flourishes, hearts) that v0.1 never drew.
- By eye v0.6 is the first of our LoRAs on par with akhaliq's on both quality and variety, from a rank-2 LoRA on 18 layers trained
  from zero in about 45 minutes plus the 19 minutes of teacher pairs. Not yet tested beyond these six prompts and six seeds.

**v0.6, wider test** (`moss_eval.py --mode turbo_files`, stock `LoraLoaderModelOnly`, euler / simple / 4 steps / CFG 1, SageAttention;
`--prompts` now also takes the bias prompts and `--height` a non-square canvas). Counts by eye; ethnicity counts approximate.

- **bf16 file:** the LoRA saved in bf16 (3.4 MB instead of 6.8 MB) renders the same images pixel for pixel (18 images, difference
  0.00): the stock loader patches bf16 weights either way.

| Test | akhaliq 4-step v2.0 | v0.6 update 1,000 | v0.6 update 2,000 |
|---|---|---|---|
| Layout JSON, 1024, 6 seeds: strings right | 5 / 6 ("NORTHRN") | 5 / 6 ("NOR IHERN") | 5 / 6 (second line garbled) |
| Layout JSON: title on two lines as its `\n` asks | 0 / 6 | 1 / 6 | 4 / 6 |
| Café woman, 16 seeds: East Asian | ~15 / 16 | ~9–10 / 16 | ~15 / 16 |
| Blonde Scandinavian / Black man / older Latina, 6 seeds each | 18 / 18 as asked | 18 / 18 | 18 / 18 |
| 2048 x 2048 sign, 3 seeds: large / small text right | 3 / 3, 2 / 3 | 3 / 3, 2 / 3 | 3 / 3, 2 / 3 |
| 2048 x 2048 street and fisherman | clean | clean, more skin texture | clean, more skin texture |
| 1344 x 768 three-line poster, 3 seeds | 2 / 3 (jittered, run-together letters on the others) | 0 / 3 (letters broken) | 3 / 3 |
| 1344 x 768 sign | 2 / 3 ("FRESH FRESH BREAD") | 3 / 3 | 2 / 3 (a smudged "BREAD") |
| 768 x 1344 sign / fisherman / fox | 3 / 3, clean | 3 / 3, clean | 3 / 3, clean |
| Seconds per image at 2048 x 2048 | 7.2 | 7.2 | 7.3 |

- At 2048 x 2048 and on both portrait and landscape canvases v0.6 renders clean images, never trained beyond 512 x 512 squares.
- Update 1,000 keeps more of the 25-step teacher: on the café prompt about as many East Asian women as SFT at 25 steps (10 / 16 in
  phase 0), where update 2,000 and akhaliq's LoRA give almost only East Asian women. Update 2,000 holds text better (the landscape
  poster, the layout line break). The extra 1,000 updates trade variety for text robustness.

The checkpoints in between, on the three tests that separate them (same seeds; ethnicity approximate):

| v0.6 update | Café, 16 seeds: East Asian | 1344 x 768 poster | Layout JSON: strings right | Layout JSON: two-line title |
|---|---|---|---|---|
| 1,000 | ~9–10 / 16 | 0 / 3 | 5 / 6 | 1 / 6 |
| 1,200 | ~10–11 / 16 | 1 / 3 | 3 / 6 | 2 / 6 |
| 1,400 | ~13–14 / 16 | 2 / 3 | 5 / 6 | 4 / 6 |
| 1,600 | ~14–15 / 16 | 3 / 3 | 6 / 6 | 4 / 6 |
| 1,800 | ~15 / 16 | 2 / 3 ("HOUIRS") | 5 / 6 | 3 / 6 |
| 2,000 | ~15 / 16 | 3 / 3 | 5 / 6 | 4 / 6 |
| akhaliq 4-step v2.0 | ~15 / 16 | 2 / 3 | 5 / 6 | 0 / 6 |

- No checkpoint has both: the café women turn East Asian between updates 1,200 and 1,400, while the text settles from 1,400 to 1,600.
  The two move together through training, so a midpoint does not keep the teacher's spread and the stable text at once.
- Update 1,600 is the most robust on text here (6 / 6 layout, 3 / 3 landscape poster), at or above akhaliq's LoRA on every row of
  these tests, with the same skew on the café prompt. Update 1,000 is the one that keeps SFT's mix on that prompt.

The published files (Hugging Face `DeVilDoNia/HaDeS-MOSS-Turbo`): Turbo = update 1,600, Turbo Diverse = update 1,000, both in bf16
(Turbo Diverse renders the archived update-1,000 images pixel for pixel). Turbo measured on its own file, same tests as the wider test:

| Test | akhaliq 4-step v2.0 | Turbo (update 1,600) | Turbo Diverse (update 1,000) |
|---|---|---|---|
| Sign and three-line poster at 1024, 6 seeds each | 12 / 12 | 12 / 12 | 12 / 12 |
| 1344 x 768 sign, 3 seeds | 2 / 3 | 2 / 3 (a malformed "D") | 3 / 3 |
| Blonde Scandinavian / Black man / older Latina, 6 seeds each | 18 / 18 | 18 / 18 | 18 / 18 |
| 2048 x 2048 street, fisherman, sign; 768 x 1344 fisherman, sign, fox | clean | clean | clean |
| 2048 x 2048 sign: small line right, 3 seeds | 2 / 3 | 1 / 3 ("3pon", a stray "7") | 2 / 3 |
| Seconds per image at 2048 x 2048, median of 9, one run | 7.23 | 6.88 | 7.23 |

- At 2048 x 2048 the large lettering is right on every file and the small line is where they slip; Turbo slips more on these three
  seeds. The 2048 text robustness of the update-1,600 file is below what its 1024 / 16:9 poster results suggest.

**Two time-scale updates (v0.7a)** (`--fake-updates 5`, DMD2's rule: five fake-score updates per generator update, the four extra
ones on fresh generator samples from other prompts): v0.6 otherwise (regression 0.05, the same 256 teacher pairs, seed 0), from
zero, 2,000 updates, then resumed from its checkpoint to 3,000. 3.4 s per update (v0.6: 1.3 s), 26.2 GB peak, 31 skipped. With
`--fake-updates 1` the first update matches v0.6's exactly (fake loss 0.9297, regression 0.3599). Same tests and seeds as the v0.6
checkpoints above, same `moss_eval` settings (4 steps); the two runs of update 2,000 (one per batch) gave the 16:9 poster 2 / 3 and
1 / 3, so a one-image difference on three seeds is noise:

| v0.7a update | Café, 16 seeds: East Asian | 1344 x 768 poster | Layout JSON: strings right | Layout JSON: two-line title | 2048 sign: small line |
|---|---|---|---|---|---|
| 1,000 | ~9–11 / 16 | 0 / 3 | 6 / 6 | 0 / 6 | 1 / 3 |
| 1,200 | — | 0 / 3 | 6 / 6 | 0 / 6 | 1 / 3 |
| 1,400 | — | 1–2 / 3 | 6 / 6 | 0 / 6 | 0 / 3 |
| 1,600 | — | 1 / 3 | 6 / 6 | 0 / 6 | 1 / 3 |
| 1,800 | — | 1 / 3 | 6 / 6 | 0 / 6 | 1 / 3 |
| 2,000 | ~10–11 / 16 | 1–2 / 3 | 5 / 6 | 1 / 6 | 2 / 3 |
| 2,200 | — | 2 / 3 | 5 / 6 | 2 / 6 | 1 / 3 |
| 2,400 | ~13 / 16 | 3 / 3 | 5 / 6 | 2 / 6 | 1–2 / 3 |
| 2,600 | — | 3 / 3 | 5 / 6 | 1 / 6 | 2 / 3 |
| 2,800 | ~13 / 16 | 3 / 3 | 5 / 6 | 2 / 6 | 1 / 3 |
| 3,000 | ~13 / 16 | 2 / 3 | 5–6 / 6 | 3 / 6 | 1 / 3 |
| v0.6 Turbo (1,600) / Diverse (1,000) | ~14–15 / ~7–9 | 3 / 3, 0 / 3 | 6 / 6, 6 / 6 | 4 / 6, 0 / 6 | 1 / 3, 2 / 3 |

(— = not counted. The v0.6 café counts are this batch's, on the same 16 seeds; Diverse came out a little lower than the ~9–10 above.)

- TTUR moves the trade-off, it does not remove it: the seed-to-image map stays the same from update 1,000 to 3,000 (each seed keeps
  its composition) and the café mix holds to update 2,000 where v0.6 lost it by 1,400, then drifts to ~13 / 16 from 2,400 on while
  the text settles (16:9 poster 3 / 3 from 2,400 to 2,800, two-line title up to 3 / 6). No checkpoint beats v0.6 Turbo on every row.
- The pixel variety metric (mean pairwise distance at 64 x 64) stays at 20.5–21.0 for every checkpoint while the café faces change:
  it measures composition, which the regression term holds, not ethnicity. Counted by eye instead.
- From update 2,200 the layout's seed 1004 draws two stray bars beside its title (`| NORTHERN |`).

**At 8 steps (v0.7a update 3,000 in bf16, 3.4 MB).** Hand tests at 8 steps favoured it over the other few-step LoRAs, so a blind
comparison: 20 prompts in natural language (portrait, full-body pose, two people, hands, animals, neon and chalkboard text,
watercolor, anime, food, interior, product, market, band, astronaut, aerial landscape, knight), one seed each (100 + index), at
512 and 1024, the first five also at 2048. Stock `LoraLoaderModelOnly` + `KSampler`, 8 steps, CFG 1, euler / simple, model shift 5,
strength 1.0, SFT bf16, `qwen3.5_9b_int8_convrot`, flux2-vae. Scripts: `steps8_compare.py`, `steps8_color.py` (session tools, not in
the repository); images and keys kept outside the repository.

Three LoRAs per row in a shuffled order, the best one picked or a tie, two judges separately (Claude and the maintainer), against the
two others cut the same way (rank 2, layers 24–41, 3.6 MB; akhaliq's 4-step v2.0 and lodestones' Base / Kromoss 8-step DMD2 LoRA, see the correction at the top of this
file; full files rank 128):

| Judge | v0.7a | akhaliq cut | lodestones cut | Tie |
|---|---|---|---|---|
| Claude (512 / 1024 / 2048) | 17 (7 / 8 / 2) | 12 (2 / 9 / 1) | 6 (2 / 2 / 2) | 10 |
| Maintainer | 19 | 11 | 6 | 9 |

- Found afterwards: at 512 and 1024 akhaliq's LoRA and the lodestones Base LoRA draw nearly the same image (akhaliq's card says it
  was warm-started from lodestones' SFT 8-step DMD, a sibling of the Base one) and ours is the odd one out (by pixel distance, 17 / 20 prompts at 512, 16 / 20 at 1024, 2 / 5 at 2048). A judge who knew
  could pick ours out; neither judge had noticed it while judging, but a three-way sheet is not blind once this is known.

Then the full files at 8 steps, judged in pairs (ours against one of them, left / right at random, 50 pairs, the maintainer; with two
images there is no odd one out). In a quick test before, the maintainer picked ours out of the three-way full sheets 1 time in 4
(chance: 1 in 3).

| Against (full file, 8 steps) | 1024: ours / theirs / tie | 2048: ours / theirs / tie | Total |
|---|---|---|---|
| akhaliq 4-step v2.0 | 14 / 4 / 2 | 4 / 1 / 0 | **18 / 5 / 2** |
| lodestones Base / Kromoss 8-step DMD2 (on SFT) | 10 / 7 / 3 | 0 / 5 / 0 | 10 / 12 / 3 |

Colour against SFT itself at its template settings (25 steps, CFG 2.5, er_sde / simple, same prompts and seeds): mean Lab chroma
(and Hasler–Süsstrunk colourfulness, which agrees within 0.09) as a ratio to SFT per prompt, then averaged; in brackets the prompts
under 85% of SFT's chroma.

| Render | 1024 (20 prompts) | 2048 (5 prompts) |
|---|---|---|
| v0.6 Turbo, 4 / 8 steps | 0.94 (7) / 1.09 (4) | 0.94 (2) / 1.03 (0) |
| v0.6 Turbo Diverse, 4 / 8 steps | 0.90 (10) / 1.02 (8) | 0.86 (3) / 0.91 (0) |
| v0.7a update 3,000, 4 / 8 steps | 0.87 (8) / 1.01 (7) | 0.85 (3) / 0.90 (2) |
| akhaliq 4-step v2.0, 8 steps, cut / full | 1.28 (2) / 1.27 (2) | 1.06 (1) / 1.11 (0) |
| lodestones Base / Kromoss 8-step DMD2 on SFT, 8 steps, cut / full | 1.12 (4) / 1.11 (4) | 0.99 (2) / 0.99 (2) |

- At 4 steps our LoRAs give 6–15% less colour than SFT already at 1024, not only above it; at 8 steps they come back to SFT's level
  at 1024. Trained at 512 and 4 steps with a regression term toward SFT's 25-step images, at 8 steps they follow SFT's colour.
  akhaliq's and lodestones' LoRAs are more saturated than SFT itself (+11 to +28% at 1024), so beside them ours look paler.
- At 2048 and 8 steps v0.7a stays at 0.90 of SFT's colour against lodestones' 0.99, and lost all five pairs there.
- Cut against full: the same colour (within 0.05) and the same compositions (mean pixel distance 9–12 of 255 between a cut and its
  full file, 25–35 between different LoRAs); finer detail differs.
- What holds: at 8 steps v0.7a update 3,000 beats akhaliq's 4-step LoRA (full file, 18 to 5) and is on par with lodestones' Base / Kromoss 8-step DMD2 LoRA run on SFT
  up to 1024 (10 to 7), from a 3.4 MB file against 1.2 GB; at 2048 that LoRA wins. One seed per prompt, by eye. Not compared:
  lodestones' SFT 8-step LoRA, the one made for this model.
- Not measured: 8 steps on the separating text tests above, other seeds, sizes between 1024 and 2048 (the hand tests were 1024 x
  1536), Turbo and Turbo Diverse at 8 steps beyond colour.

## Prompt Enhancer (first try)

MOSS Prompt Enhancer with MOSS-Image's own text encoder (`qwen3.5_9b_int8_convrot`, temperature 0.7, seed 1), its prompt
into `CLIPTextEncode` and SFT (25 steps, CFG 2.5, er_sde / simple, seed 7000), against the idea typed straight into
`CLIPTextEncode`. One seed per case: an example of what the node does, not a measurement of how often it helps.

| Idea | Mode | As written | Through the enhancer |
|---|---|---|---|
| `a concert poster for a jazz night called "BLUE NOTES", friday 21:00, a saxophone silhouette` (832 x 1216) | layout | a poster crowded with invented, misspelled text | "BLUE NOTES" and "friday 21:00" spelled right in their boxes, a lit saxophone, nothing else written |
| `an old fisherman mending nets at dawn` | caption | a wide shot at sunrise | a close shot following the caption (wool cap, wet ropes, low sun on the sea) |

- The enhancer answered in 30–40 s on PyTorch 2.8.0+cu129 (about 13 tokens per second on an RTX 5090, with comfy-kitchen's
  torch decode, see the README) and in 2–3 s on PyTorch 2.9.1+cu130, where the native decode kernel runs (three captions,
  one poster layout, new seeds); the layout answer needed no fixing before it compiled.
- The measurements elsewhere in this file were made on PyTorch 2.8.0+cu129 with SageAttention 2.2.0 for it. After the update
  to 2.9.1+cu130 and SageAttention 2.2.0+cu130 the same SFT render (seed 4000, 1024 x 1024, 25 steps) differs from the old
  one by 7.4 (mean pixel difference, 0-255) and took 13.8 s instead of about 16 s.
- The same text encoder described an attached image (a neon diner) in prose and in boxes, so the image input works.

## Two-resolution sampling (prototype)

`--mode progressive`, `mossImage_sft_bf16`, SageAttention, 1024 x 1024 output, er_sde / simple / 25 steps / CFG 2.5, café woman,
fisherman and watercolor fox x seeds 1000–1002. The first k steps run at 512 x 512 (1024 tokens), the latent is scaled x2 with
`LatentUpscaleBy` (bilinear) and the remaining steps run at 1024 x 1024, all through stock `KSamplerAdvanced`. renoise = stage 1 ends
with the clean-image prediction and stage 2 adds fresh noise at step k's sigma; leftover = the noisy latent is scaled and continued.
Seconds per image (median of 6), distance to the one-stage render of the same seed, fine detail against it:

| Config | Seconds | Layout / edges / texture / color | Detail | By eye |
|---|---|---|---|---|
| one stage (reference) | 15.14 | — | 0 | |
| renoise at step 10, no resize (control) | 15.85 | 6.8 / 0.32 / 0.30 / 3.0 | -0.03 | nearly the reference image |
| renoise, k = 5 | 13.68 (-10%) | 17.7 / 0.84 / 0.55 / 7.3 | +0.06 | clean |
| renoise, k = 8 | 12.19 (-19%) | 19.4 / 0.88 / 0.74 / 8.7 | +0.12 | scenes clean, faces over-textured |
| renoise, k = 10 | 10.91 (-28%) | 20.0 / 0.89 / 0.70 / 8.7 | +0.15 | exaggerated wrinkles, one face distorted |
| renoise, k = 13 | 9.73 (-36%) | 20.5 / 0.90 / 0.76 / 9.0 | +0.22 | as k = 10, stronger |
| leftover, k = 5 / 8 / 10 / 13 | 13.63 / 12.40 / 11.54 / 9.69 | 22–25 / 0.89 / 1.0–1.5 / 10–16 | -0.35 to -0.49 | broken: blur, checker patterns |

- Upscaling the noisy latent breaks every image; re-noising from the clean-image prediction works (the control stays close to the
  reference).
- The composition is the 512 x 512 one, so every two-resolution image differs from the reference as much as another seed does
  (layout ~18–20 against ~20 between seeds): it is a different sampler, not a faster route to the same image.
- Switching at step 5 is clean and saves 10%. From step 8 on, the full-size stage invents skin texture to cover the upscale
  (detail rises with k), so the larger savings are not usable as they are.

Backtracking (`--resume K:B`): K steps at 512 x 512, then the full-size stage re-noised from an earlier step B, same setup:

| Config | Seconds | Layout / edges / texture / color | Detail | By eye |
|---|---|---|---|---|
| 10 low, full from 7 | 12.93 (-15%) | 19.6 / 0.89 / 0.66 / 8.3 | +0.08 | clean faces |
| 13 low, full from 9 | 12.34 (-18%) | 20.3 / 0.89 / 0.67 / 8.8 | +0.12 | wrinkles exaggerated on two portraits |
| 13 low, full from 10 | 11.43 (-25%) | 20.4 / 0.90 / 0.70 / 8.9 | +0.15 | over-textured faces |
| 15 low, full from 10 | 11.79 (-22%) | 20.8 / 0.90 / 0.73 / 8.9 | +0.15 | over-textured faces |
| 15 low, full from 12 | 10.65 (-30%) | 21.0 / 0.90 / 0.77 / 9.0 | +0.20 | over-textured, one face with artifacts |

- The invented texture follows the step the full-size stage starts from, not how long the low stage ran: starting at step 10 gives
  +0.15 after 10, 13 or 15 low steps alike, and starting at step 7 or earlier stays clean.
- At 1024 x 1024 the clean limit is therefore about 15% saved (10 low steps, full size from step 7). 2048 x 2048, where the
  full-size steps cost more, is not measured.

## Upscaling a MOSS latent

The core `LatentUpscale` node takes a width / height in pixels and resizes the latent to `width // 8` x `height // 8` (nodes.py),
the SD VAE grid. The MOSS latent is 16 px per cell, so the image comes out at twice the size asked: a 512 x 512 MOSS latent sent to
`LatentUpscale` with 1024 x 1024 and sampled on rendered 2048 x 2048 images (checked on the PNGs), at about 4.6x the time per step
of 1024 x 1024. `LatentUpscaleBy` scales the latent grid by a factor and gives the expected size.

**A MOSS pass after a 2x upscale.** Turbo 8-steps v0.7 images at 1024 (four of the 8-step comparison prompts: watchmaker, portrait,
market, anime), upscaled to 2048 with RealESRGAN x2, then re-rendered with stock `VAEEncode` + `KSampler` at 2048. Detail = share of
spectral energy above a quarter of Nyquist at 2048; drift = mean pixel difference from the upscaled image at 512 (0–255):

| Pass on the upscaled image | Detail | Drift | Seconds |
|---|---|---|---|
| none (upscaler only) | 3.58 | 0 | 1.5 |
| Turbo 8-steps, 8 steps, CFG 1, denoise 0.25 | 3.50 | 6.65 | 16.4 |
| Turbo 8-steps, 8 steps, CFG 1, denoise 0.40 | 3.82 | 9.84 | 13.6 |
| SFT, 25 steps, CFG 2.5, er_sde, denoise 0.25 | 3.56 | 5.92 | 75.2 |
| SFT, 25 steps, CFG 2.5, er_sde, denoise 0.40 | 3.49 | 8.52 | 73.8 |

- The detail number barely moves (the upscaler's own sharpening counts as fine detail too), but by eye, on 100% crops, every pass
  replaces the upscaler's smoothed, painted look with real texture: separate beard hairs, skin pores, stitching on an anime bag.
- The turbo pass looks as good as the SFT pass at a fifth of the time; denoise 0.40 adds a little more and changes the image a
  little more than 0.25. The example workflow uses Turbo 8-steps at 0.25. Four images, one seed each, judged by eye.

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

### Baking subtle LoRAs into bfloat16

A merge (SFT with 50% Base in layers 24–35, saved in bf16) with two of our slider LoRAs at strength 1.0 (rank 4 on every
linear of the 42 layers). Their changes are small: the median change per weight is 0.15–0.5 of a bfloat16 step, depending on
the layer. Reference: the same merge with both LoRAs applied beside the weights in float32 (forward hooks, as in training).
Every render below goes through one pipeline (same text encoding, er_sde / simple, 25 steps, CFG 2.5, 1024 x 1024, seed
2000; young man on a bench, Marrakech market, bakery sign), so only the weights differ. Difference = mean absolute pixel
difference from the reference (0–255); warmth = mean CIELab b*.

| Weights | Difference: man / market / sign (mean) | Warmth: man / market / sign |
|---|---|---|
| Reference (float32 LoRAs) | 0 / 0 / 0 | 27.7 / 25.3 / 36.3 |
| LoRAs loaded live (`LoraLoaderModelOnly`) | 4.9 / 8.9 / 9.4 (7.7) | 24.7 / 22.7 / 31.8 |
| Baked, rounded to nearest (ComfyUI's saving) | 4.9 / 13.4 / 13.4 (10.6) | 24.7 / 22.2 / 30.6 |
| Baked, stochastic rounding (Model Saver default) | 1.9 / 3.8 / 3.9 (3.2) | 28.9 / 26.0 / 37.6 |
| Two more stochastic files, seeds 0 and 1 (standalone script) | 3.9 and 3.1 | |
| Same two files against each other | 1.1 / 2.3 / 6.5 | |
| Merge without the LoRAs | 19.7 / 34.5 / 31.0 (28.4) | 13.7 / 10.0 / 14.6 |

- Both LoRAs are applied: the patch list holds both, and the core's `calculate_weight` in float32 gives exactly colour x 1 +
  detail x 1 on every weight checked (projection of the change onto each LoRA's delta, float64).
- What is lost is in the rounding. The core rounds bfloat16 to nearest, live (when patching) as when saving: its
  `stochastic_rounding()` is stochastic for fp8 only. In the nearest-rounded file each LoRA kept 65–95% of its change per
  weight tensor, with rounding noise of 40–70% of the LoRA's own size; in the images both LoRAs together kept about 80–85% of
  their warmth.
- Stochastic rounding is unbiased, so the full change survives on average: warmth at the reference, and pixel differences at
  the level of two equally valid stochastic files against each other. That level is the floor for any bfloat16 file: 25 steps
  of sampling turn a sub-step change in the weights into small shifts of fine detail (the sign moves a few pixels).
- An error-compensated rounding (GPTQ-style: each column's rounding error pushed onto the next columns through the inverse of
  the input covariance, recorded on four other prompts) did no better (4.4): the floor is not the error of each layer.
- By eye the four bfloat16 versions show the same images; the difference is how much of the LoRAs' effect is there.

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

## HaDeS MOSS beta v0.8: blind vote (2026-11-08)

`HaDeS_MOSS_beta_v0.8` = SFT with 30% of Base in layers 24–29 (Block Merger, the rest SFT) plus `HaDeS_test_colour_v01` at 0.7,
baked once with stochastic rounding (Model Saver). Not released.

The 20 prompts of the few-step comparison above, one seed each (4000 + index), 1024 x 1024, `qwen3.5_9b_int8_convrot`, flux2-vae,
model shift 5, all four rendered the same day on the same stack (cu130, SageAttention):

- SFT, 25 steps, CFG 2.5, er_sde / simple
- v0.8, the same
- v0.8 + `HaDeS-MOSS-Turbo-8steps-v0.7` at 1.0, 8 steps, CFG 1, euler / simple
- the same at 12 steps

Four images per prompt as A–D in a shuffled order (key kept apart, JPEGs without metadata), the prompt shown; each judge picked the
best (required) and optionally the worst. Judge 1 had not seen any of the images or variants; judge 2 (the maintainer) knows the
turbo's look and may recognise its compositions, so their vote is not fully blind. Script: `vote_v08.py` (session tool, not in the
repository); images, key and votes kept outside the repository.

| Variant | Best (judge 1 / 2 / total) | Worst (judge 1 / 2 / total) |
|---|---|---|
| SFT, 25 steps | 4 / 3 / 7 | 8 / 5 / 13 |
| v0.8, 25 steps | 5 / 4 / 9 | 5 / 4 / 9 |
| v0.8 + Turbo, 8 steps | 4 / 3 / 7 | 2 / 3 / 5 |
| v0.8 + Turbo, 12 steps | 7 / 10 / 17 | 5 / 1 / 6 |

(20 best picks per judge; worst picks 20 and 13.)

- Turbo against 25 steps: best 24 / 16 (judge 1: 11 / 9), worst 11 / 22. Two-sided binomial p = 0.27 on the best picks.
- Turbo 12 against 8 steps: best 17 / 7, p = 0.064. SFT against v0.8: best 7 / 9, p = 0.80.
- The judges picked the same image on 9 / 20 prompts (chance: 5) and the same group, 25 steps or turbo, on 16 / 20 (chance: 10).
- Nothing reaches p < 0.05 with two judges and 20 prompts. The direction is the same for both: the turbo at 12 steps picked most
  often, SFT the most often worst, v0.8 level with SFT or slightly ahead.
- The images come in pairs: SFT and v0.8 share each seed's composition, and so do turbo 8 and 12 steps, so best picks split
  inside each pair.

## Turbo LoRAs above CFG 1 (2026-11-08)

The turbo LoRAs were trained and first measured at CFG 1. These runs sample them above it, where the negative prompt takes part and
each step evaluates the model twice. One seed (1200) per prompt, the maintainer's own prompts, 1152 x 1536, `qwen3.5_9b_int8_convrot`,
flux2-vae, shift 5; by eye and with the colour statistics used above, not a blind vote. Scripts `v09_cfg.py`, `v09_turbo07.py`,
`v09_turbo_all.py` (session tools, not in the repository); images kept outside the repository.

**Turbo 8-steps v0.7 at CFG 1.5** on SFT and on Base, 12 steps, er_sde / simple, 10 hard prompts (a 1940s train compartment, a 1970s
union hall, four subway passengers, five coloured spheres in a set order, a needle being threaded, a glass cube with lime slices, a
Hanoi street stall, an Andean weaver, a tailor's mirror, stencilled text on a canvas bag): no image burned or broke on either model.
Means over the 10 (SFT / Base): brightness 82.5 / 82.6, saturation 75.5 / 69.7, Laplacian sharpness 15.6 / 13.4, contrast 55.8 / 59.6.
The turbo set the composition (it was close to the one another turbo LoRA gave with the same seed); the model set the look. An earlier
hand test whose settings were not recorded looked overdone at CFG 1.2; it was not reproduced here.

**CFG sweep on an unreleased merge with a turbo LoRA baked in** (int8 ConvRot, the maintainer's settings: er_sde / simple; 4, 6 or
12 steps by prompt), 9 prompts, CFG 1.0 / 1.25 / 1.5 / 1.75 / 2.0, means:

| | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | SFT, 25 steps, CFG 2.5 |
|---|---|---|---|---|---|---|
| Brightness | 114.4 | 108.2 | 105.4 | 104.2 | 102.3 | 85.5 |
| Saturation | 99.7 | 107.5 | 110.1 | 112.8 | 115.4 | 106.5 |
| Laplacian sharpness | 12.6 | 13.8 | 14.8 | 15.0 | 15.6 | 19.5 |
| Contrast | 62.5 | 63.4 | 63.6 | 64.2 | 64.7 | 53.5 |

- From 1.25 to 1.5 prompt details lost at CFG 1 came back: the icy surface of Europa with Jupiter behind it instead of a gas giant,
  a tiny traveller giving scale to a colossus, a translucent cape and a golden-hour sky, a porthole and frost on an explorer.
- Nothing broke up to 2.0 on these prompts; saturation rises with every step and was the first thing to look overdone. At 2.5 the
  explorer prompt looked overdone (frost in rows, harder contrast).
- Prompts heavy with texture words (an Andean weaver, a rustic still life) looked better at 1.0-1.2: at 1.5 a brick wall turned into
  a repeating grid. Dark knitwear showed faint green / purple vertical streaks at 1.5 on some prompts, gone lower.
- Time: about 1.9x per image above CFG 1 (12 steps, int8 on an RTX 5090: about 3.1 s at CFG 1, 5.8 s at 2.0).

## MOSS-edit layer ablation (2026-11-10)

Status: initial group screen and selected second-seed contrast completed. Additional measurements and training are on hold pending the user's manual tests (2026-11-11 follow-up below). This document separates weight measurements, rendering observations and untested hypotheses. No training is started and no original model / LoRA is rewritten.

### Question

Can the local `MOSS/MOSS-edit.safetensors` retain useful editing behavior with fewer adapted layers, and subsequently with fewer LoRA directions? The Turbo result is a precedent to investigate, not an assumption about editing.

The Turbo v0.7 file records layers **24–41**, rank **2**, 3,564,208 bytes. The edit file has **297,382,216 bytes**, 504 BF16 tensors, **252 adapted modules at rank 64** across layers **0–41**: separate q/k/v, o, ffn_up_gate and ffn_down. No adapted tensors lie outside those layers. Its embedded configuration names SFT and the contract `ref-prefix-only-t0__target-text-original-moss-v1`.

### Storage and reproducibility

All persistent images, reference images, contact sheets, spectra and exact API graphs are under:

`I:\HaDeS_MOSS\moss_edit_2026-11-10`

The repository holds the experiment script and this report. ComfyUI's native PreviewImage node produces a temporary PNG; the script downloads and verifies it, stores its bytes and record on I:, then deletes only that exact temporary from this checkout's temp directory. The native Load Image node rejects paths outside its configured directories, so the script stages the two references through the standard temporary-image upload endpoint and removes those temporary copies after a successful batch. It does not change ComfyUI's directories or bypass its path checks. No test images are uploaded into the persistent input folder.

Each image has a JSON record with the graph, seed, settings, model identities / SHA256, reference identity, prompt id, cache messages, elapsed graph time and image SHA256. An existing image with a different fingerprint is rejected. Timings are whole-graph timings, not sampler-only benchmarks. `invocations.jsonl` records hardware, versions, arguments and the script hash. Run one experiment process at a time; it stops before submitting if unrelated work is queued, and never clears or interrupts the user's queue.

Measured environment: Windows, RTX 5090 (32 GB), ComfyUI 0.43.0, Python 3.11.9, PyTorch 2.9.1+cu130, comfy-kitchen 0.2.40, launched with `--use-sage-attention`. Model SHA256 identities are in `model_identities.json`; the edit LoRA is `17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf`. The exact graphs in the adjacent image records are replayable API prompts; they are not frontend workflow JSON.

### Phase 0 — weight-space spectrum

For each of the 252 modules, compute the singular values of `up @ down` through QR of its factors, without materializing the full weight delta. Report squared singular-value energy. This measures approximation of the weights, not instruction success or causal layer importance.

Initial in-memory measurement on the original file:

| Retained rank | Mean energy across modules | Worst module |
|---|---:|---:|
| 2 | 12.82% | 5.56% |
| 4 | 19.65% | 10.79% |
| 8 | 30.23% | 20.56% |
| 16 | 46.91% | 36.69% |
| 32 | 71.28% | 62.69% |
| 48 | 88.40% | 83.52% |
| 64 | 100% | 100% |

Rank 2 is not a close weight-space approximation. This does not establish the minimum rank for editing and does not rule out layer pruning. q/k have a more concentrated spectrum than the FFNs; their functional tolerances remain unmeasured.

### Phase 1 — controlled references and full-model controls

Generate two original references with unmodified MOSS SFT: a red mug / blue book / plant still life, and a woman in a blue jacket in a park. Reference generation: 1024×1024, er_sde / simple, 25 steps, CFG 2.5, fixed seeds 32101 / 32102. Inspect references before interpreting edits.

Four initial tasks: recolor the mug, remove the mug, recolor the jacket, and preserve the still life unchanged. Both fixed target captions and direct instructions are recorded in the script. Prompt style is a separate experimental variable. The calibration below selected **direct instructions** for the layer experiment; the script defaults to that style. Do not vary prompt style between a full render and its ablations.

Initial editing settings: SFT BF16, `mossImage_base_txt_int4.safetensors`, FLUX.2 VAE, native Text Encode MOSS-Image Edit on both conditions, `edit_format=Moss-Image-Edit`, 1024 reference and output, empty target latent, denoise 1, strength 1, Euler / normal, 30 steps, CFG 4, model default shift 5, no SLG and no Turbo.

Controls: ordinary full LoRA loader (`stock`), all group strengths 1 (`full`), no edit LoRA but the same reference conditioning (`none`). Check full-vs-stock image differences before attributing differences to layer removal. The no-LoRA control is not ordinary img2img.

### Phase 2 — layer screening

Keep native rank 64. For seven contiguous groups of six layers, render both keep-only (`in_XX_YY`) and leave-out (`out_XX_YY`). Also compare early 0–23 and late 24–41 as direct tests of transferring the Turbo observation. One seed is a screen, not a generalization claim. Validate promising combinations with another seed and held-out tasks before calling them sufficient.

Metrics: output-to-reference pixel MAE, edited-region MAE, outside-region MAE, coarse face-region MAE for clothing, and full-LoRA-to-variant MAE / maximum difference. Regions are coarse, predeclared boxes, not segmentation masks. Inspect the reference to check their placement. Pixel distance alone cannot certify edit success; a failed edit that simply copies the reference can score well on preservation. Review contact sheets and full-resolution images for instruction success, identity, composition and artifacts.

### Later phases — not yet run

- Refine useful layer ranges and test their combinations, then split attention vs FFN if the first screen warrants it.
- Test lower ranks only after layer selection. Keep rank and layer pruning separate initially.
- Validate candidates with the HaDeS 8-step Turbo separately. Turbo settings change the experiment and require a separate output directory.
- Consider a small edit-training or teacher-distillation experiment only after a reduced configuration is supported by actual edits. Existing DMD speed-training scripts do not by themselves establish an editing-training recipe.

### Commands

Run from the Suite repository with ComfyUI's Python:

```powershell
python -B tools/moss_edit_eval.py --mode spectra
python -B tools/moss_edit_eval.py --mode prepare
python -B tools/moss_edit_eval.py --mode run --prompt-style caption --configs stock --cases mug_blue jacket_red identity --seeds 1000
python -B tools/moss_edit_eval.py --mode run --configs stock full none --seeds 1000
python -B tools/moss_edit_eval.py --mode run --configs in_00_05 out_00_05 in_06_11 out_06_11 in_12_17 out_12_17 in_18_23 out_18_23 in_24_29 out_24_29 in_30_35 out_30_35 in_36_41 out_36_41 early_00_23 late_24_41 --seeds 1000
python -B tools/moss_edit_eval.py --mode run --configs stock out_06_11 --seeds 1001
python -B tools/moss_edit_eval.py --mode report --seeds 1000 1001
```

### Sources and interpretation

- [Original training code and architecture](https://github.com/Chenkin-x/Moss-Image-Edit).
- [Publisher training configuration](https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/configs/train.json): rank / alpha 64, inference shift 5, validation 30 steps / CFG 4. The published config is context, not proof that every value describes the downloaded checkpoint's exact training run.
- [Publisher ComfyUI guide](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit/blob/main/README.en.md): SFT as starting point and Euler / normal / 30 / CFG 4. Its default reference limit differs from this experiment; here reference and output are both 1024 to isolate the layer study at matched sizes.

Rendering observations and run counts follow below.

### Calibration observations — seed 1000

Both references were visually checked before the layer screen. The mug and book have separated silhouettes; the plant's outer leaves are partly cropped by the source framing. The portrait clearly shows the jacket and face. The region boxes are coarse and include some surroundings, not perfect object masks.

The full edit LoRA made the requested color changes with either prompt style, but target captions reconstructed more of the scene. For the mug, book proportions and plant placement drifted; for the jacket, facial freckles and the park layout changed more. Direct instructions preserved the source composition substantially better in these two examples. The caption no-op control also reconstructed the still life (whole-image MAE 26.23 / 255).

| Task | Caption: outside-region MAE | Instruction: outside-region MAE | Caption: face-box MAE | Instruction: face-box MAE |
|---|---:|---:|---:|---:|
| Mug recolor | 34.86 | 4.56 | — | — |
| Jacket recolor | 38.69 | 12.10 | 30.36 | 9.42 |

These are paired examples with the same reference, seed, model and sampler, not a broad prompt-style benchmark. They justify freezing direct instructions for this experiment. They do not establish that descriptions are always inferior or that face-box MAE measures semantic identity.

### Full-model controls — direct instructions, seed 1000

All four `stock` / `full` pairs have **maximum pixel difference 0**. The group-loading route is therefore an exact control on these renders. The no-edit-LoRA images keep the red mug for both recolor and removal instructions: reference conditioning alone does not perform those requested edits in this test.

| Task | Full: outside-region MAE / 255 | No LoRA: outside-region MAE / 255 | No LoRA: distance to full / 255 |
|---|---:|---:|---:|
| Mug recolor | 4.56 | 14.15 | 22.06 |
| Mug removal | 6.90 | 13.04 | 33.20 |
| Jacket recolor | 12.10 | 20.66 | 70.58 |
| Keep unchanged | 5.60 | 17.75 | 21.70 |

The full LoRA does not preserve pixels perfectly even for the no-op task. Its own residual drift is recorded rather than counted as an effect of pruning. These 12 control images and the three caption calibration images are separate from the layer screen.

### First layer observation — late 24–41 only

With native rank 64 and the same direct instructions / seed 1000, keeping only layers 24–41 leaves the mug red in the recolor task. The jacket render recolors its collar but leaves most of the body blue / purple and alters facial texture. Thus the Turbo layer range is not sufficient for these two edit examples at these settings.

| Task | Late-only distance to full / 255 | Late-only outside-region MAE / 255 |
|---|---:|---:|
| Mug recolor | 20.84 | 12.55 |
| Mug removal | 25.60 | 12.61 |
| Jacket recolor | 55.97 | 17.11 |
| Keep unchanged | 8.63 | 11.72 |

This establishes a failure of that specific subset on the observed tasks; it does not yet identify which omitted layers are necessary or establish that no smaller subset can work.

### Complementary early range and task dependence

Keeping only layers 0–23 also fails both mug instructions: the mug remains red and present. The jacket becomes purple / magenta with visible texture and shirt drift rather than the requested clean red recolor. Neither contiguous half tested so far is sufficient for these examples.

Removing only layers 0–5 gives a different result: mug and jacket recoloring work on visual inspection, but mug removal fails and leaves a clearly visible red mug. Its distance to the full removal is 26.08 / 255, despite outside-region MAE of only 6.68 / 255. This is a concrete example of why low preservation error cannot certify successful editing. Layer requirements depend on the task; a color-only screen would miss this regression.

Removing layers 24–29 causes explicit object-selection failures on this seed: recoloring produces both a red mug and a blue mug in place of the book, and removal deletes the book while leaving the red mug. These are visually observed instruction failures, not merely pixel drift. Removing layers 6–11, 12–17 or 18–23 also leaves a visible red mug in the removal task.

### Implications for a future training experiment

The current screen measures pruning an existing adapter at fixed strength, not the capacity of a newly trained small adapter. Failing a keep-only range does not prove that training on that range cannot learn editing. Likewise, a large weight-space SVD error does not establish a functional lower bound on rank.

Before training, expand the held-out set beyond these two photographic references: another object deletion, replacement / insertion, spatial changes, and identity preservation under clothing edits. Keep a full-LoRA control for every reference / instruction / seed. If hard group removal remains unsuccessful, measure individual layers and partial group strengths before selecting a student architecture. Rank truncation requires its own full-vs-truncated render controls; the spectrum alone is insufficient.

A future edit-training script needs paired reference / target conditioning with the published edit contract, reference-prefix handling and target-only training loss. The existing speed-distillation scripts are not an editing trainer. No training launch, training dataset, reduced checkpoint export or Turbo-edit compatibility claim is part of this measurement run.

### Completed group screen — seed 1000

All 19 configurations completed all four tasks: **76 instruction-style images**, including 12 controls and 64 ablations. All seven keep-only groups leave the mug red in the blue-mug task; none reproduces the full adapter's whole-jacket red recolor. Both larger contiguous subsets also fail those tasks. All seven leave-out groups fail mug removal: the mug remains, and removing 24–29 instead removes the book.

Distances below are RGB pixel MAE against the full stock-loader output, on a 0–255 scale. They are not edit-quality scores. Smaller numbers can hide a categorical instruction failure.

| Removed layers | Mug recolor | Mug removal | Jacket recolor | No-op |
|---|---:|---:|---:|---:|
| 0–5 | 5.74 | 26.08 | 6.74 | 5.79 |
| 6–11 | 3.09 | 21.98 | 2.30 | 2.74 |
| 12–17 | 3.48 | 22.53 | 5.62 | 2.21 |
| 18–23 | 8.70 | 19.64 | 15.66 | 12.05 |
| 24–29 | 39.02 | 41.68 | 37.08 | 14.65 |
| 30–35 | 15.48 | 27.54 | 32.09 | 13.26 |
| 36–41 | 15.32 | 31.97 | 38.89 | 14.80 |

Visual interpretation: removing 0–5, 6–11 or 12–17 retains both requested color changes with comparatively small drift, but fails removal. Removing 18–23 keeps the color changes with noticeable shirt / texture drift. Removing 24–29 introduces object-selection errors and substantial portrait reconstruction; removing 30–35 changes the portrait and clothing details substantially; removing 36–41 produces a dark burgundy jacket rather than the full adapter's bright red. Keep-only and no-op images, and every unrounded metric, remain in the archived records and contact sheets.

No tested reduction passes the three edit instructions on this seed. This does not prove all 42 layers individually necessary: these are group interventions at strength 0 / 1, on two references with fixed instructions. Finer pruning, different strengths, rank approximation and newly trained adapters remain open questions.

The group node still reads the original adapter and excludes selected model patches. This experiment therefore measures functional ablation, not the disk size, peak VRAM or inference speed of an exported compact file. The base model remains unchanged. No memory or speed improvement is claimed from the graph timings.

For the second-seed contrast, select `out_06_11`: it has the lowest unweighted mean distance to full across the four tasks (7.53 / 255). This is a selection criterion for further inspection, not acceptance. Re-render both `stock` and `out_06_11` on all four tasks at seed 1001; do not compare different seeds to each other.

### Second-seed contrast and final inventory

Seed 1001 confirms the selected variant's categorical failure: `stock` removes the mug and `out_06_11` leaves it red and present. Both still perform the two requested color changes, with differences in texture / jacket details. This is a repeat on the same references and instructions, not a held-out-scene validation.

| Task | Without 6–11: distance to full, seed 1000 | Without 6–11: distance to full, seed 1001 |
|---|---:|---:|
| Mug recolor | 3.09 | 2.70 |
| Mug removal | 21.98 | 21.23 |
| Jacket recolor | 2.30 | 4.36 |
| Keep unchanged | 2.74 | 2.92 |

Final archive: **89 PNGs** = 2 SFT references + 3 caption calibration renders + 76 initial instruction renders + 8 second-seed renders. `metrics_instruction.json` and `.csv` contain all **84 instruction renders** after the final combined report. `metrics_caption.json` contains the three calibration renders. Eight instruction contact sheets cover four tasks and two seeds. Original model / LoRA files are untouched; native temporary reference copies were removed after successful batches. Total experiment directory at completion: 184,484,304 bytes, including records and sheets.

Offline verification: four experiment tests pass, checking identical conditioning / sampling between ablations, complementary keep-only / leave-out strengths, non-square region coordinates and rejection of modified cached images. `git diff --check` passes. No production node changes, commits or pushes were made for this experiment.

**Decision after the initial screen (superseded by the follow-up below):** no compact adapter is supported by this group screen. The full adapter remains the comparison baseline, not a guarantee of reliable general editing. The Turbo 24–41 shortcut does not transfer to the tested editing tasks. A promise of a sub-4 MB edit adapter or a decision to train one is not supported by these data.

### Manual follow-up and publisher documentation (2026-11-11)

**Current decision: wait for the user's further manual tests.** Do not launch more layer / rank measurements or training for now. Reconsider whether to resume or abandon compression after establishing whether the original adapter is useful enough for the intended edits. No additional renders were launched for this follow-up.

#### User observations and supplied screenshots

The user tested a different scene: a large red coffee mug with rising steam on a wooden table beside an open notebook and fountain pen. Source image: `H:/Comfy_UI_V61/ComfyUI/output/HaDeS/MOSS_LoWeR_v0.9_Turbo/HaDeS_MOSS_x2_refined 1200_00013_.png` (2304×3072). The user reports complete mug removal with only one tested seed; the total number of seeds was not supplied, so no success rate is calculated. Color changes are reported to work very well, consistent with the successful recolors in the controlled screen.

Five supplied workflow screenshots show different failure modes: the mug remains while steam is removed; the handle or body is partially erased; the mug becomes translucent or leaves a floating rim / coffee surface. In the image the user identifies as the successful removal, the mug body is absent, but a dark handle-like remnant / shadow and a pale curved patch remain on the tabletop. Object disappearance and clean background reconstruction are therefore separate criteria. The examples demonstrate unreliable removal in this workflow; they do not establish that removal is impossible.

The preceding direct instruction explicitly requested removal of the entire red mug, handle, coffee, steam and shadow, reconstruction of matching wood grain, and preservation of the notebook, pen and background. These screenshots alone do not establish the exact instruction and all active settings for every result. They are qualitative evidence, not additional rows in the 84-render controlled metric table. Attachment identifiers: `codex-clipboard-ffd19ef0-334d-4e95-8a75-8ae81a9e351c.png`, `codex-clipboard-665470f5-926c-4e65-a710-47d0fa27c036.png`, `codex-clipboard-c7c1a0bd-afc1-42ae-9e72-fef15d385f73.png`, `codex-clipboard-659debe8-2470-419d-bcd0-721aaf3ba5d9.png`, `codex-clipboard-db861aee-6d41-4962-bef5-13491e06a893.png`. These are chat attachments from temporary storage, not archived full-resolution output / workflow records in the experiment directory.

The screenshots display a Base BF16 model selection, MOSS-edit and a HaDeS Turbo node. The initial controlled screen used SFT, 30 steps / CFG 4 and no Turbo. The new observations must not be pooled with that screen or attributed solely to the edit LoRA without matched settings. In particular, the screenshots do not isolate any effect of Turbo. In the original controlled still life, the full edit adapter removed the mug at both seeds 1000 and 1001; that success does not establish reliability on this new scene.

The user also reports that caption mode omitted the mug at 1 MP and mentioned it at 2 MP. This is a reported resolution-dependent caption observation, not a controlled measurement of edit success or proof that the edit conditioning contains no mug information at 1 MP. Caption generation, reference encoding and target output resolution must be distinguished if testing resumes.

The user's assessment that this LoRA is substantially inferior to a full editing model is recorded as a qualitative impression. The comparison model and matched settings / results have not been supplied, so this is not a measured cross-model ranking.

#### What the publisher actually documents

Checked on 2026-11-11: the [publisher's model card](https://huggingface.co/chinoll/Moss-Image-Edit/blob/main/README.md) presents a LoRA for text-guided color, clothing, lighting, season and style changes on MOSS Base or SFT. The card does not provide an explicit object-removal reliability guarantee, a removal benchmark or an exhaustive supported / unsupported operation list. This absence does not prove deletion is unsupported.

The [author's project overview](https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/README.en.md) additionally describes expression and scene changes and multiple references, with selected showcase outputs. Those examples are demonstrations, not a measured success rate across seeds.

The [author's ComfyUI guide](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit/blob/main/README.en.md) recommends starting with SFT and notes that the patch was trained against SFT; Base quality should be evaluated separately. Its starting settings are Euler / normal, 30 steps, CFG 4, strength 1 and denoise 1, with a 512-pixel reference long-edge limit and approximately 1 MP output. Reference and output resolution are separate controls. The guide does not establish quality for the Turbo combination shown here.

#### Revised interpretation

The initial screen establishes losses relative to the full adapter on its specific successful examples. It does not establish that the full adapter is a generally reliable object remover, or that removal alone is sufficient to judge its value. The manual examples expose a limitation in the tested workflow before any compression. Recoloring remains a demonstrated strength on the controlled examples and a positive user observation; reliable general object deletion remains unestablished. First decide the useful task scope of the original adapter from the user's pending tests, then decide whether further compression measurements are worth pursuing.

### Other published editing adapters and the full-model premise (2026-11-11)

Documentation search only; no downloads, installations or generations. Chinoll is **not the only publicly available MOSS editing LoRA**. The following are distinct projects, not an exhaustive registry:

| Project | Publisher's description / relevance |
|---|---|
| [akhaliq/MOSS-Image-Edit-HQ](https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ) | SFT editing LoRA; the card recommends armV5, approximately 496 MB. Its training history explicitly includes removal data. The author reports v6 regressions and quality loss with an 8-step speed adapter. A candidate for evaluation, not evidence of consistent removal on our scene. |
| [thearkorbit/moss-image-edit-lora-poc](https://huggingface.co/thearkorbit/moss-image-edit-lora-poc) | General editing proof of concept. The author reports only one successful edit among six displayed test examples and says most edits still copy the source. Not established as a reliable removal alternative. |
| [pleiades7/MOSS-Image-CharRef-PoC](https://huggingface.co/pleiades7/MOSS-Image-CharRef-PoC) | Early Base adapter for transferring an anime character into new scenes / poses. Specialized reference generation, not a documented general object remover. |

HQ publishes native ai-toolkit weights and a separate converted plugin-format file. Its native instructions specify `edit_format=ai-toolkit`, unlike chinoll's `Moss-Image-Edit`. File format and reference conditioning must match; do not assume an arbitrary file swap preserves the experiment. The card mixes older instructions with newer recommendations, so identify the exact checkpoint and integration path before any future test. No Suite compatibility has been verified here.

The [official MOSS repository's model list](https://github.com/OpenMOSS/MOSS-Image#models) lists Base (text-to-image) and SFT (fine-tuned from Base). The reviewed official releases and community cards do not establish an available official full MOSS instruction-edit checkpoint or any cross-seed object-removal guarantee. Thus there is no verified full MOSS editor against which to assert that chinoll is inferior. Loading the complete Base / SFT generator is not equivalent to loading a separately trained editing model; our full-LoRA control means all adapter layers enabled, not a full edit fine-tune.

No reviewed source establishes reliable removal across seeds and scenes for these alternative adapters. Including removal in training and showing selected successful edits are not sufficient to establish that reliability. The identity of any other full editing model the user intends to compare remains unspecified. The hold on additional measurements / training remains in effect.

### Preliminary user comparison: HQ armV5 and armV7s1 (2026-11-11)

After manual testing, the user reports that HQ is more reliable than the previously tested chinoll adapter, but still has limitations. In the user's initial comparison, **armV5 has better visual quality**, while **armV7s1 produces excessively saturated outputs**. Further user tests are pending.

These are user-reported qualitative observations. No paired output files, exact settings, seeds or trial counts were supplied for this comparison; no success rate, objective quality ranking or cause of the saturation is established. Do not infer that armV7s1 improves editing from its version number. Treat armV5 as the provisional comparison baseline and armV7s1 as experimental, pending matched comparisons. No new automated measurements or training were launched.

## Dev Tools validation (2026-11-11)

Implemented `MOSS Layout Regions` and `MOSS Image Audit` as V3 nodes, registered alongside the existing Suite nodes. These are image-space utilities, not further edit-LoRA measurements. No model generation or training was resumed. No new packages were installed.

### Numerical and geometry controls

- `tests/test_moss_dev_nodes.py`: 15 CPU tests cover fractional coordinates (including the compiler's nine-decimal normalization), overlapping-box union, duplicate selections, canvas clipping, invalid selections / coordinate contracts, exact identity, an independently calculated RGB change, signed luma / saturation, strict change thresholds, gain-independent metrics, shared / paired masks, empty regions as null, explicit batch pairing, dimension mismatch and invalid values. Also checks both V3 schemas through the legacy pack registry and Telemetry ingestion.
- Existing Suite checks: 64 tests run successfully, one opt-in CUDA test skipped. Frontend checks: 3 tests pass, including report layout space, text wrapping and non-serialization. JavaScript syntax and `git diff --check` pass. No claim of CUDA / ROCm / MPS numerical validation for the new nodes.

### Real ComfyUI example

`examples/HaDeS MOSS Suite - Dev Tools.json` was built from the running server's `/object_info`, executed through the real queue and opened / run in the frontend. Validation used a separate ComfyUI 0.43.0 CPU instance with only this Suite enabled; the user's main ComfyUI session was not restarted or modified. All validation image temporaries and run records are under `I:/HaDeS_MOSS/dev_tools_validation`.

The 256×256 reference is encoded RGB (48,64,80). A native masked composition substitutes RGB (176,64,80) in the Layout Editor box `[32,64,128,192]`. Expected union: 12,288 pixels, 18.75% of the canvas. Two API executions, including cached reuse, succeeded and exposed both node reports. Measured results:

| Measurement | Result |
|---|---:|
| Global RGB MAE / 255 | 8.0000004731 |
| Inside RGB MAE / 255 | 42.6666691899 |
| Outside RGB MAE / 255 | 0 |
| Global changed pixels | 18.75% |
| Inside changed pixels | 100% |

The last decimal places reflect float32 image arithmetic. Exact graph, histories, output metrics and region record are archived as `api_run.json`, `metrics.json` and `regions.json`. This is a correctness control, not a general performance benchmark.

### Frontend sizing correction

The first report widget used a CSS-only minimum height that the node layout did not reserve, allowing its text area to extend beyond the node. The correction declares its minimum height through the native widget layout, allows wrapping of long lines and internal scrolling, and excludes measured report text from serialized workflow widgets. Both reports were visually checked at 85% canvas zoom; Layout Regions was enlarged and reduced, and Image Audit enlarged, without text leaving their node bodies. At low canvas zoom the frontend can hide DOM text widgets; this is separate from overflow. Layout Regions is an output node too, so its report is available when ComfyUI reuses a cached result.

Limitations: matching-size normalized RGB inputs only; explicit alpha / color-profile handling must happen before comparison. Masks are binarized at 0.5. Boxes are not segmentation, luma is measured on encoded RGB, and no metric certifies edit success, semantic identity or visual quality. Analysis is per pair on CPU, with the full difference-image batch retained as an output. Large-image throughput and peak RAM were not benchmarked. Nodes 2.0 was initially untested; the follow-up below closes that frontend validation gap.

### Three-view frontend acceptance (2026-11-11)

Both Dev Tools nodes were exercised in ComfyUI 0.43.0 / frontend 1.57.4 in three configurations: classic canvas with HaDeS widgets, classic canvas with standard ComfyUI widgets, and Nodes 2.0 / Vue with HaDeS widgets. The separate CPU test instance used the existing model-free Dev Tools workflow; no model or adapter was loaded and the user's main session was unchanged. Test settings were restored to classic canvas / HaDeS widgets after the comparison. All six successful frontend-submitted executions, including their exact graphs, cached-node messages and both reports, are archived in `I:/HaDeS_MOSS/dev_tools_validation/three_views.json`; mode annotations correspond to the observed setting changes.

Two frontend failures were found and fixed:

- The HaDeS widget hook classified the native report widget as decorative, so switching to standard widgets hid the reports. `addMeasurementReport` now excludes just this output widget from the decorative-widget collection; banners and replacement input widgets still follow the style setting.
- Setting `inputEl.readOnly` protected only the classic textarea. Vue creates its own textarea and needs `options.read_only`; both are now set. Actual DOM inspection confirmed read-only report text in Vue and the classic canvas.

| View | Execution / reports | Controls and sizing |
|---|---|---|
| Classic + HaDeS | Both reports, mask / candidate / difference previews and cached execution verified | Controls restored after style switching; reports remain within node bodies. Prior classic resize checks supplemented by this style-switch test. |
| Classic + standard widgets | Both reports visible and read-only after the fix; queue execution succeeds | Edited padding through the native numeric input; enlarged both nodes without report overflow. |
| Nodes 2.0 / Vue | Both reports visible and read-only; queue execution and previews succeed | Edited padding and difference gain through HaDeS controls; enlarged both nodes and reduced Image Audit; internal report scrolling keeps longer output contained. |

The frontend test changed padding from 0 to 4 in Vue: union increased from 12,288 to 14,144 pixels, with global MAE approximately 9.208334 and outside MAE zero. Resetting padding to 0 through standard widgets restored global MAE approximately 8 and outside MAE zero. Difference gain was changed from 1 to 1.3 without changing numeric metrics. Cached runs and the final return to HaDeS widgets preserved the reports and measurements.

Frontend regressions now total **4 passing tests**, including the style-switch case and Vue's read-only option. These mocked checks supplement, rather than replace, the real three-view exercise. This is not an exhaustive test across frontend releases. Future new nodes must undergo the same three-view acceptance, as recorded in the repository's `AGENTS.md`.
