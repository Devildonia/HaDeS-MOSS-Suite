# Measurement results (2026-11-03)

`<archive>` is the maintainer's local measurement archive (images and run records, not published); `<ComfyUI>` is the ComfyUI install the runs used.

[MOSS-edit layer ablation (2026-11-10)](#moss-edit-layer-ablation-2026-11-10); [Edit benchmark, scored blind (2026-11-12)](#edit-benchmark-chinoll-armv5-and-no-lora-scored-blind-2026-11-12).

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

**The same pass on Base with Turbo 8-steps v1.0** (2026-11-20, for the second tutorial). MOSS-Image Base bf16, Qwen3.5 9B int8,
FLUX.2 VAE, Turbo 8-steps v1.0 at 1.0 on the MOSS KSampler's `turbo` input (8 steps, CFG 1, euler / simple), the fisherman of the
first tutorial at 1024 x 1024, seed 1013; RealESRGAN x2 to 2048, then `VAEEncode` and a second MOSS KSampler with the same turbo,
`denoise_mode` core. Clean ComfyUI on port 8189 with `--use-sage-attention`, RTX 5090, models warm; seconds from `/history`
(`execution_start` to `execution_success`), each run re-executing only its own stage (ComfyUI's cache skips the rest), two runs per
denoise. Detail and drift are computed here with this run's own script (share of spectral energy outside a radius of a quarter of
the 2048 spectrum's half-width, in %; mean absolute difference from the RealESRGAN image, both resized to 512), so the detail values
are not on the same scale as the table above; compare them only within this table:

| Stage | Detail | Drift | Seconds |
|---|---|---|---|
| 1024, 8 steps (image only) | 0.37 (pixel-doubled to 2048) | 2.02 | 2.9 |
| RealESRGAN x2 | 0.52 | 0 | 1.1 |
| + MOSS pass, denoise 0.25 | 0.61 | 8.23 | 10.7–11.1 |
| + MOSS pass, denoise 0.40 | 0.76 | 12.66 | 10.7–10.8 |

- By eye on 100% crops (face and hands): the RealESRGAN image is sharp at the edges but smooth inside; the 0.25 pass adds skin
  pores and spots on the forehead and the hands, separates beard and hair strands and sharpens the net's threads, keeping the
  face. 0.40 adds a little more and redraws more (the forehead's wrinkles change). One image, one seed, judged by eye.
- Suite Telemetry on this pipeline: no findings for the 1024 stage; with the pass's KSampler connected, one note (turbo LoRA at
  denoise 0.25). The MOSS Image Saver with that Telemetry json notes that A1111 parameters take the first KSampler unless
  `primary_stage_metadata` picks a stage.

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

## Layout and Text Check on a poster (2026-11-21, for the eighth video tutorial)

HaDeS MOSS v1.0 Turbo (8 steps, CFG 1, euler / simple), 1024 x 1536, clean ComfyUI on port 8189, `--use-sage-attention`, RTX 5090. The idea,
in `MOSS Prompt Enhancer` `layout` mode (temperature 0.7): "A poster: an old fisherman with a grey beard mending a net on the left side,
the sun rising over the sea in the upper right with seagulls, the title "NORTH SEA" at the top, and the signature "HaDeS" in the lower
right corner." Scripts, drafts and sheets in `I:/HaDeS_MOSS/videos/layout_text/measure`.

- **Drafts, enhancer seeds 1-6** (box centres checked by script against the sentence): the title, the fisherman and the signature in their
  region in 6 / 6, with the exact words "NORTH SEA" and "HaDeS"; the sun as its own box in the upper right in 3 / 6 (seeds 2, 4, 6), inside
  a "sea" box covering the right half in 2 / 6, missing in 1 / 6; 3-6 items, none dropped. 2.4-3.5 s per draft (12.3 s for the first,
  loading the text encoder). Seed 2, the first with every element in its own box and region, is the one used below.
- **That draft at image seeds 1013-1020** (boxes drawn on each image, by eye): the title in its box at the top, the sun in its box in the
  upper right with seagulls beside it, and the signature in its box, 8 / 8; the fisherman on the left 8 / 8, his net and legs reaching
  past the narrow box (60-400 px wide) to the right. MOSS Text Check: `exact` for "NORTH SEA" and "HaDeS" in 8 / 8, which matches the
  images. 4.5 s per image with Text Check (13.3 s for the first after loading).
- **The sun box moved down to the horizon** (700-980 x 520-770, everything else unchanged), the same 8 seeds: the sun drawn in its new box
  8 / 8; both texts still `exact` 8 / 8. Dragged in the Layout Editor the box moves in 16 px steps and lands at 516-766: measured again
  there, the same result (8 / 8, both texts `exact` 8 / 8).
- **Times:** 4.2 s per image without Text Check; Text Check alone 0.7-1.0 s on this image. A one-letter typo in `expected_text` ("NORTH
  SEE") gives `close`, read 'NORTH SEA' (0.89).

## Compare models and LoRAs (2026-11-21, for the seventh video tutorial)

Clean ComfyUI on port 8189, `--use-sage-attention`, RTX 5090, the tutorials' fisherman (seed 1013). Scripts and sheets in
`I:/HaDeS_MOSS/videos/compare/measure`; images in `comfy_output/cm_measure`.

- **Models sheet** (MOSS Compare Sampler, SFT / Base / the sixth video's test recipe / Bulle d'Encre, 25 steps, CFG 2.5, er_sde / simple):
  each column is the MOSS KSampler's image of the same model and settings (max pixel difference 0, 0, 1, 0). 66.2 / 67.7 s for the four;
  the same four from four MOSS KSamplers 56.4 / 77.8 s (the models are swapped in and out of VRAM either way), so the sheet saves no time.
  Peak VRAM about 31.4-31.7 GB of 32 either way; peak RAM of the ComfyUI process about 14.5 GB (21 GB on the first run after loading).
  Before the 0.9.4 fix, Base and Bulle d'Encre (Load Diffusion Model) were labelled "unknown model"; now `mossImage_base_bf16` and
  `HaDeS_MOSS_preview_beta`.
- **LoRA sheet** (Bulle d'Encre's LoRA extracted again at rank 64 against its base rebuilt with the Block Merger, as in the fourth video:
  252 MB, 97.0 % energy, 104 s; on that base through four Multi-LoRAs, 15 steps, CFG 2.5, euler / simple): labels `merge: mossImage_sft_bf16
  + mossImage_base_bf16` (the Multi-LoRA switched off), `bulle_extracted x0.5`, `bulle_extracted x0.8`, `bulle_extracted`. Distance to
  Bulle d'Encre at the same settings, over the distance between its seeds 1013 and 1014 (46.4 / 255): off 0.86, x0.5 0.74, x0.8 0.59,
  x1.0 0.26. Strength is not linear: most of the look arrives between 0.8 and 1.0. 39.8 s for the four.
- **Blind:** panels numbered 1-4 with no names; blind_seed 0 and 1 give different orders; the key (number -> slot and label) is in the info
  output and in the PNG's `hades_moss_compare` chunk, not on the sheet. The footer keeps what every panel shares (the base model, seed,
  steps, prompt).
- **MOSS Compare Sheet** (two MOSS KSamplers on Base from Load Diffusion Model, seeds 1013 / 1014, labels empty): `mossImage_base_bf16 ·
  seed 1013` and `· seed 1014`, read from the KSamplers' infos (0.9.4; before, the panels read `A` and `B`).

## Merge, save and telemetry: a test recipe (2026-11-21, for the sixth video tutorial)

A recipe made only for the video, not a recommendation: SFT with 50 % of Base in layers 18-35 (Block Merger groups 18-23, 24-29 and
30-35 at 0.5, everything else 0). Clean ComfyUI on port 8189, `--use-sage-attention`, RTX 5090; the Merge example's KSampler settings
(25 steps, CFG 2.5, er_sde / simple, 1024 x 1024, SLG off). Distances are mean absolute pixel differences at 512 x 512 (0-255). Scripts and
contact sheets in `I:/HaDeS_MOSS/videos/merge_save/measure`.

- **Where it lands** (the tutorials' fisherman, seeds 1013 / 1014): the recipe is 34.3 / 30.2 from SFT and 41.2 / 48.3 from Base; SFT and
  Base are 48.2 / 47.6 apart, and two seeds of SFT 56.7. By eye it keeps SFT's polished look and moves toward Base's variety. 13-14 s per
  image for SFT, Base and the recipe alike (the first of a session longer).
- **Bias** (the two neutral prompts of "Bias by merge", seeds 1000-1015, East Asian faces counted by eye, approximate):

| Prompt | SFT | Test recipe | Base |
|---|---|---|---|
| `photograph of a woman sitting in a café` | 10/16 | ~3/16 | ~1-2/16 |
| studio fisherman portrait (`photo_portrait`) | 16/16 | ~13-14/16 | ~6/16 |

  SFT and Base match the earlier counts. The recipe removes most of the bias on the café prompt and little on the portrait: as found
  before, the bias is spread over the model. Naming the ethnicity in the prompt stays the reliable fix.
- **Model Saver:** bfloat16 (stochastic rounding) 10.16 GB (the node shows 9.47 GiB) in 29.6 s; int8 ConvRot 5.41 GB (5.04 GiB) in 15.7 s;
  both with the merger's info as `recipe`. The files land in ComfyUI's `output/diffusion_models`; moved to `models/diffusion_models` they
  load with Load Diffusion Model. The bf16 file renders 0.02 from the live merge on both seeds (the same image). The int8 file renders in
  6.3 s instead of 15 s, 5.2 and 19.5 from the live merge (0.09 and 0.33 of a seed-to-seed distance): the same scene, small changes, more
  on one seed.
- **Metadata Reader** on the saved file (1.0 s, header only): size, layers, parameters, precision, rounding, author, and the recipe with
  each group's ratio. Before 0.9.4 the recipe named only SFT (from the MOSS Loader), not Base (from Load Diffusion Model); fixed, the
  re-saved file names both.
- **Multi-Model Merger SLERP 0.5** against the Block Merger with every group at 0.5: 6.4 / 3.2 apart, about the same image; each 33.9-36.9
  from SFT and 30.8-40.7 from Base; the same time (16-23 s per render with the merge).
- **Telemetry** on the recipe: "No findings", the merger's ratios layer by layer, and the KSampler's `model=mossImage_sft_bf16.safetensors
  + mossImage_base_bf16.safetensors`. The MOSS Image Saver writes `Model: merge: mossImage_sft_bf16 + mossImage_base_bf16`, no model hash,
  with the note "the model is a merge made in the workflow".

## What the MOSS Image Saver writes (0.9.3, 2026-11-21, for the fifth video tutorial)

The Turbo 8 steps + Upscale x2 example (Base + Turbo 8-steps v1.0, the fisherman, seed 1013) on a freshly started clean ComfyUI on port
8189, run through the API (so the PNGs carry `prompt` but no `workflow` chunk; a run from the frontend adds `workflow`):

- **The 1024 PNG** (`metadata` from Suite Telemetry): chunks `parameters`, `hades_moss` (the infos of the Loader, Prompt Enhancer, Latent,
  KSampler and Turbo), `prompt`. `parameters`: the prompt, `Negative prompt:` (the official template's), then `Steps: 8, Sampler: euler,
  Schedule type: simple, CFG scale: 1, Seed: 1013, Size: 1024x1024, Model hash: cacd158728, Model: mossImage_base_bf16, Shift: 5, VAE:
  flux2-vae, Text encoder: qwen3.5_9b_int8_convrot, Lora hashes: "HaDeS-MOSS-Turbo-8steps-v1.0: fccd4ef0de", Hashes: {...}, Version:
  ComfyUI, HaDeS MOSS Suite`.
- **Hashing:** the first save with `hashes` on took 5.7 s for the Saver alone (the 10.2 GB model and the 3.4 MB LoRA, right after the
  model was loaded, so the file may have been in the OS cache; a cold disk will be slower); the next save 0.06 s.
- **Two stages:** the refined 2048 PNG with Telemetry on the whole pipeline wrote the first KSampler's settings (`Size: 1024x1024`, no
  denoise) and the note "2 KSamplers / 1 Loaders in metadata ..."; with a second Telemetry fed only by the Loader, Prompt Enhancer, the
  pass's KSampler and the Turbo, on `primary_stage_metadata`: `Size: 2048x2048, Denoising strength: 0.25`, no note, and the PNG keeps
  both `hades_moss` (all six infos) and `hades_moss_primary_stage` (four).
- **A merge made in the workflow** (the LoRA Toolkit example with the Multi-LoRA on): `Model: merge: mossImage_sft_bf16 +
  mossImage_base_bf16`, no `Model hash`, and the note "the model is a merge made in the workflow".
- **Privacy:** `hashes` off leaves no `Model hash`, `Lora hashes` or `Hashes`; `embed_workflow` off leaves `parameters` and `hades_moss`
  but no `prompt` / `workflow` chunk; ComfyUI's `--disable-metadata` leaves no text chunk at all (`tests/test_moss_suite.py`,
  `test_disable_metadata_removes_all_png_text`).

## The LoRA toolkit on Bulle d'Encre (2026-11-21, for the fourth video tutorial)

Bulle d'Encre (`HaDeS_MOSS_preview_beta`) is SFT with 30 % of Base in layers 24-29, plus Turbo 8-steps v1.0 at 0.5 and the mega style LoRA
at 0.8 (its `hades/recipe`). Clean ComfyUI on port 8189, `--use-sage-attention`, RTX 5090. Four prompts (the tutorials' fisherman, a Paris
café, a two-panel detective comic, a red fox), seed 1013; distances are mean absolute pixel differences at 512 x 512 (0-255) divided by
the distance between Bulle d'Encre's seeds 1013 and 1014 on the same prompt (1.0 = as different as another seed), averaged over the four
prompts. Files in `models/loras/HaDeS_video/`; scripts and sheets in `I:/HaDeS_MOSS/videos/lora_toolkit/measure`.

- **Extractor** (Bulle d'Encre minus its base, the base as the original merged file): rank 32 126 MB, 96.8 % energy, 0.33 from Bulle
  d'Encre at 15 steps / CFG 2.5 / euler simple (the base + the extracted LoRA at 1.0 in the Multi-LoRA); rank 64 252 MB, 97.0 %, 0.19;
  rank 128 504 MB, 97.2 %, 0.22. 92-111 s each. Rank 64 against the base rebuilt live with the Block Merger (layers 24-29 at 0.3, the
  rest 0): 97.0 %, 0.19, the same as against the original file. By eye, rank 64 gives Bulle d'Encre's compositions and look; the base
  alone (no LoRA) is 0.77 away and loses the look.
- **Turbo v1.0 at 0.5 as a second Multi-LoRA row** (CFG 1, euler simple): 8 steps 0.56, 6 steps 0.57, 4 steps 0.61 from Bulle d'Encre at 15
  steps; the same scene and look by eye, not the same image (the comic lost its second panel at all three step counts). Warm time 2.2 s
  at 8 steps against 8.0 s for Bulle d'Encre at 15 steps / CFG 2.5.
- **One group of layers off** (the rank-64 LoRA, the rest at 1.0, 15 steps / CFG 2.5): the share of the LoRA's change lost, as the distance
  to the full-LoRA image over the base alone's distance to it: layers 00-05 0.44, 06-11 0.29, 12-17 0.39, 18-23 0.56, 24-29 0.75, 30-35
  0.74, 36-41 0.48. By eye, the fisherman with 30-35 off is close to the base alone. LoRA Block Weight's info counts 24 weights in every
  group, so the counts alone do not show this; one LoRA, one seed.
- **Merger** (the rank-64 LoRA at 1.0 + Turbo v1.0 at 0.5, rank 72, exact for 64 + 2): 255 MB in 1 s. Rebuilt in float32, every layer is
  `A + 0.5 B` within 0.24 % (the bf16 rounding of the stored factors). Against the two rows live at 8 steps / CFG 1 it is 0.44, above the
  0.17 of the same two rows loaded in the opposite order: same scene and look, small details moved (most on the fisherman). At few steps
  a rounding difference is enough to move details.
- **Resize** of the merged file: rank 32 126 MB, 0.13 from the merged file's images (within the 0.17 floor above); `dynamic_energy` 0.90
  (ranks 1-26 per layer) 40 MB, 0.50 (the fisherman 0.83).
- **Metadata Reader** on the rank-32 file: the chain resize -> merge (rank 72) -> extracted from Bulle d'Encre minus the base (rank 64,
  97.0 %) + 0.5 x Turbo 8-steps v1.0, ending in "Original files in the result: +1 x the extracted LoRA, +0.5 x the turbo".
- The Extractor, Merger and Resize save under the root of `models/loras` whatever folder `export_filename` names (the name's folder part
  is dropped); the files here were moved into `HaDeS_video/` by hand, and the recipes still read them.

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

`<archive>/moss_edit_2026-11-10`

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

**Status:** layer / rank compression of the edit adapter is paused until it is established whether the original adapter is useful enough for the intended edits. No additional renders were launched for this follow-up.

#### User observations and supplied screenshots

The user tested a different scene: a large red coffee mug with rising steam on a wooden table beside an open notebook and fountain pen. Source image: `<ComfyUI>/output/HaDeS/MOSS_LoWeR_v0.9_Turbo/HaDeS_MOSS_x2_refined 1200_00013_.png` (2304×3072). The user reports complete mug removal with only one tested seed; the total number of seeds was not supplied, so no success rate is calculated. Color changes are reported to work very well, consistent with the successful recolors in the controlled screen.

Five supplied workflow screenshots show different failure modes: the mug remains while steam is removed; the handle or body is partially erased; the mug becomes translucent or leaves a floating rim / coffee surface. In the image the user identifies as the successful removal, the mug body is absent, but a dark handle-like remnant / shadow and a pale curved patch remain on the tabletop. Object disappearance and clean background reconstruction are therefore separate criteria. The examples demonstrate unreliable removal in this workflow; they do not establish that removal is impossible.

The preceding direct instruction explicitly requested removal of the entire red mug, handle, coffee, steam and shadow, reconstruction of matching wood grain, and preservation of the notebook, pen and background. These screenshots alone do not establish the exact instruction and all active settings for every result. They are qualitative evidence, not additional rows in the 84-render controlled metric table.

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

## Edit benchmark: chinoll, armV5 and no LoRA, scored blind (2026-11-12)

Do the published edit LoRAs remove objects cleanly enough, or is there room for our own edit LoRA? Criterion fixed before the
run: if the better LoRA removes cleanly in less than 70 % of the removal cases, the gap is real.

**Setup.** Eight references made with SFT (er_sde / simple, 25 steps, CFG 2.5, seeds 32200-32207, 1024²; checked by eye before
editing): mug, portrait, street, living room, perfume bottle, lake, anime classroom, "OPEN" sign. 24 instructions on them, written
before the run: 7 remove, 7 add, 3 recolor, 2 material, 3 style, 2 text. Three arms: chinoll's `MOSS-edit` (17a48b8c...), HQ
`armV5` and no edit LoRA (same reference conditioning). armV7s1 left out: its author recommends armV5 and the maintainer's tests
found v7 worse. SFT bf16, `mossImage_base_txt_int4`, flux2-vae, Euler / normal, 30 steps, CFG 4, shift 5, 1024², seeds 1000-1002,
LoRA strength 1, negative `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`. Conditioning by the **MOSS Prompt
Enhancer** (`as written`, `edit_format` Moss-Image-Edit, `edit_megapixels` 1.0). 216 renders. Images, records, metrics,
blind sheets, key and scores: `<archive>/moss_eval/edit_bench/`.

**Enhancer edit mode = core.** Before the run, the Enhancer and the core `TextEncodeMOSSImageEdit` were rendered with the same
graph otherwise (chinoll, seed 1000) on mug_remove, portrait_leather, lake_autumn and sign_closed: **maximum pixel difference 0**
on all four.

**Blind scoring.** One sheet per instruction, the original and three columns with the arms shuffled (seeded), the seeds in rows.
The maintainer scored every column and seed before the key was opened. Counts below are seeds out of 3.

Removal (clean = the object gone and nothing else lost or put in its place):

| Instruction | chinoll | armV5 | no LoRA |
|---|---|---|---|
| mug and its steam | 0 (steam gone 1 / 3, mug kept) | 0 (steam gone 3 / 3, mug kept) | 0 |
| red car | 1 (one also removes the other cars and adds one, one turns it white) | 1 (one removes every car, one cuts the car behind) | 0 |
| lamp post | 2 (third partial) | 3 | 0 |
| floor lamp | 3 | 3 | 0 |
| wooden dock | 1 | 3 | 0 |
| desks and chairs on the left | 0 (removes everything but the girl) | 1 (one partial and takes furniture on the right, one removes all furniture) | 0 |
| hanging sign | 0 (a blank wooden sign, a blank sign, "OPN") | 0 (a blank sign in all three) | 0 |
| **clean** | **7 / 21 (33 %)** | **11 / 21 (52 %)** | 0 / 21 |

**The better LoRA, armV5, removes cleanly in 52 % of the cases: below 70 %, so by the pre-declared criterion the gap is real.**
Both fail the same two cases: the mug (both take the steam and leave the mug) and the sign (the text goes, the board stays).

Other instructions (done; side effects in brackets):

| Instruction | chinoll | armV5 | no LoRA |
|---|---|---|---|
| blue mug | 3 | 3 | 0 |
| croissant on a plate | 3 (the pen disappears) | 3 (the pen disappears) | 0 |
| red leather jacket | 3 (third seed: woman further away, black trousers) | 3 | 0 (washed out, face changes) |
| sunglasses | 3 | 3 | 3 (washed out, degraded) |
| snowy winter | 1 (two seeds swap the woman for two girls) | 3 | 0 |
| cyclist | 3 | 3 | 1 |
| cat on the sofa | 3 | 2 | 3 |
| dark green sofa | 3 | 3 | 3 |
| frosted blue glass | 3 (the bottle changes in all) | 3 (the bottle changes in two) | 0 |
| rose petals | 3 | 3 | 0 |
| black background | 2 (one removes the bottle) | 3 (bottle kept) | 0 |
| autumn | 3 | 3 | partial (a few trees) |
| red canoe | 3 | 0 | 0 |
| navy blazer | 3 (one also recolors the bow tie) | 3 (bow tie recolored in all) | 0 (only the bow tie) |
| black cat at her feet | 3 | 3 | 1 |
| sign to "CLOSED" | partial (painted over; one pastes a new sign) | 2 (one blank sign) | 0 |
| sign to "BACK IN 5 MIN" | partial (painted over) | 2 (one blank sign) | 0 |
| **done** | **42 / 51** | **46 / 51** | **11 / 51** |

**Preservation outside the edit box** (median pixel MAE against the reference, 0-255, coarse predeclared boxes, not masks):

| Kind | chinoll | armV5 | no LoRA |
|---|---|---|---|
| remove | 5.5 | 17.3 | 20.4 |
| add | 6.8 | 15.9 | 20.5 |
| recolor | 5.7 | 11.9 | 11.4 |
| material | 8.9 | 19.1 | 10.9 |
| text | 11.7 | 14.9 | 14.3 |

Style instructions change the whole image and have no meaningful outside box. chinoll keeps the rest of the image closest to the
reference; armV5 changes it about as much as no LoRA does, but completes more instructions. A failed edit that copies the
reference scores well here, so this table is not an edit-quality ranking.

**Conclusions.**
- armV5 is the better of the two on what was asked (removal 11 / 21 against 7 / 21, other instructions 46 / 51 against 42 / 51);
  chinoll changes less of the rest of the image.
- Removal is the weak point of both: 52 % at best, and the mug and the sign fail with both. The gap the criterion asked about is
  real, so an own edit LoRA focused on removal is supported by this benchmark.
- Without an edit LoRA, SFT with the reference conditioning still completes a few additions and one recolor (sunglasses, cat,
  green sofa) but no removal.

**Limits.** One maintainer scored, on 24 instructions, 8 references and 3 seeds, at one setting (30 steps, CFG 4, strength 1,
SFT). Not measured: other strengths, step counts or CFG, Base, Turbo LoRAs, other resolutions, other edit LoRAs.

## Dev Tools validation (2026-11-11)

### Development example inventory (2026-11-11)

The development examples now live together in `examples/dev/`: `HaDeS MOSS Suite - Dev Tools.json` (moved from the root of `examples/`) and `HaDeS MOSS Suite - Stage Metadata (Dev).json` (copied from `<archive>/stage_metadata_validation/stage_metadata_ui_workflow.json`). The original `<archive>` workflow and run records remain there as test evidence. An inventory of JSON files under `<archive>` found no other ComfyUI editor workflow for these development examples. The API prompt in the stage validation folder is execution evidence, not a drag-and-drop editor example.

`tools/make_examples.py` now directs both definitions to `examples/dev/`, so regeneration preserves this layout. Both archived examples parse as ComfyUI editor JSON and have valid link endpoints (10 nodes / 11 links and 10 nodes / 12 links respectively). The stage example's image values match its generator definition. This organization check did not rerun the frontend; the actual three-view execution and save/reload checks are recorded in the validation sections below and in the stage metadata validation section. No claim of new frontend coverage is made here.

Implemented `MOSS Layout Regions` and `MOSS Image Audit` as V3 nodes, registered alongside the existing Suite nodes. These are image-space utilities, not further edit-LoRA measurements. No model generation or training was resumed. No new packages were installed.

### Numerical and geometry controls

- `tests/test_moss_dev_nodes.py`: 15 CPU tests cover fractional coordinates (including the compiler's nine-decimal normalization), overlapping-box union, duplicate selections, canvas clipping, invalid selections / coordinate contracts, exact identity, an independently calculated RGB change, signed luma / saturation, strict change thresholds, gain-independent metrics, shared / paired masks, empty regions as null, explicit batch pairing, dimension mismatch and invalid values. Also checks both V3 schemas through the legacy pack registry and Telemetry ingestion.
- Existing Suite checks: 64 tests run successfully, one opt-in CUDA test skipped. Frontend checks: 3 tests pass, including report layout space, text wrapping and non-serialization. JavaScript syntax and `git diff --check` pass. No claim of CUDA / ROCm / MPS numerical validation for the new nodes.

### Real ComfyUI example

`examples/dev/HaDeS MOSS Suite - Dev Tools.json` was built from the running server's `/object_info`, executed through the real queue and opened / run in the frontend. Validation used a separate ComfyUI 0.43.0 CPU instance with only this Suite enabled; the user's main ComfyUI session was not restarted or modified. All validation image temporaries and run records are under `<archive>/dev_tools_validation`.

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

Both Dev Tools nodes were exercised in ComfyUI 0.43.0 / frontend 1.57.4 in three configurations: classic canvas with HaDeS widgets, classic canvas with standard ComfyUI widgets, and Nodes 2.0 / Vue with HaDeS widgets. The separate CPU test instance used the existing model-free Dev Tools workflow; no model or adapter was loaded and the user's main session was unchanged. Test settings were restored to classic canvas / HaDeS widgets after the comparison. All six successful frontend-submitted executions, including their exact graphs, cached-node messages and both reports, are archived in `<archive>/dev_tools_validation/three_views.json`; mode annotations correspond to the observed setting changes.

Two frontend failures were found and fixed:

- The HaDeS widget hook classified the native report widget as decorative, so switching to standard widgets hid the reports. `addMeasurementReport` now excludes just this output widget from the decorative-widget collection; banners and replacement input widgets still follow the style setting.
- Setting `inputEl.readOnly` protected only the classic textarea. Vue creates its own textarea and needs `options.read_only`; both are now set. Actual DOM inspection confirmed read-only report text in Vue and the classic canvas.

| View | Execution / reports | Controls and sizing |
|---|---|---|
| Classic + HaDeS | Both reports, mask / candidate / difference previews and cached execution verified | Controls restored after style switching; reports remain within node bodies. Prior classic resize checks supplemented by this style-switch test. |
| Classic + standard widgets | Both reports visible and read-only after the fix; queue execution succeeds | Edited padding through the native numeric input; enlarged both nodes without report overflow. |
| Nodes 2.0 / Vue | Both reports visible and read-only; queue execution and previews succeed | Edited padding and difference gain through HaDeS controls; enlarged both nodes and reduced Image Audit; internal report scrolling keeps longer output contained. |

The frontend test changed padding from 0 to 4 in Vue: union increased from 12,288 to 14,144 pixels, with global MAE approximately 9.208334 and outside MAE zero. Resetting padding to 0 through standard widgets restored global MAE approximately 8 and outside MAE zero. Difference gain was changed from 1 to 1.3 without changing numeric metrics. Cached runs and the final return to HaDeS widgets preserved the reports and measurements.

Frontend regressions now total **4 passing tests**, including the style-switch case and Vue's read-only option. These mocked checks supplement, rather than replace, the real three-view exercise. This is not an exhaustive test across frontend releases. Future new nodes must undergo the same three-view acceptance.

## HaDeS MOSS v0.9.4 baked Turbo: CFG / step screen (2026-11-11)

### Question and controls

How does `HaDeS_MOSS_v0.9.4_Turbo.safetensors` respond to raising CFG and steps from its natural Turbo starting point, **8 steps / CFG 1**, at the resolution used in the maintainer's own images? This screen measures the baked BF16 model, not an SFT model with a live Turbo LoRA. The optional KSampler `turbo` input is absent; no LoRA or upscaler is applied.

The archive is `<archive>/baked_turbo_v094_2026-11-11`. The reproducible runner is `tools/moss_baked_turbo_eval.py`. Four original 1152×1536 PNGs in `<ComfyUI>/output/HaDeS/HaDeS_MOSS` supplied their **resolved positive and negative text** and one seed each: airlock (industrial geometry and warning labels), trawler portrait (face, raincoat and lettering), ceramic vessel (fine glaze texture) and comic cover (hand, illustration and multiple text elements). Seed 1200 supplied an independent second composition for each scene. The prompt enhancer was not regenerated during the screen. All eight settings per scene/seed used the same model, Qwen3.5 int8 ConvRot text encoder, FLUX.2 VAE, 1152×1536 empty latent, Euler / simple, shift 5, denoise 1 and SLG off. Only steps or CFG changed. The run used ComfyUI 0.43.0 on the RTX 5090; the user's open workflow was not changed.

The API graph uses native model / text encoder / VAE loaders and the Suite's MOSS Latent and MOSS KSampler. As a provenance check, each of the four source-seed images at **8 steps / CFG 1.2** was regenerated with this shorter fixed-text graph. The four new RGB PNGs are **pixel-identical** to their originals (MAE 0), confirming that removal of the prompt enhancer and disabled Multi-LoRA from the run graph did not change those images. Every generated PNG has a sidecar JSON with the exact graph, prompt ID, image hash, timing and metrics; `manifest.json` fixes the source hashes and full matrix. There are 64 successful PNGs, 64 successful sidecars, 8 labelled contact sheets, `metrics.csv`, and `paired_differences.csv` (each setting against 8/CFG 1 for the same scene and seed). The one failed loader attempt is retained separately as `airlock_902666576933789_s08_cfg1.failed.json`; it produced no image and is not in the measurements.

### Measured image and time changes

Each row is **8 images** (4 scenes × 2 seeds). Luma and HSV saturation are computed on encoded 8-bit RGB, not linear-light photometry; `near-clip` is the percentage of pixels with any channel ≤1 or ≥254. They measure appearance, not correctness, detail quality, or prompt adherence. Times are per-job execution-start to execution-success from ComfyUI history; the median reduces the influence of the initial cold run (20.36 s for the first image). Model loading, caching and other system work mean these are session-specific timings, not hardware-independent benchmarks.

| Steps / CFG | Median seconds | Mean luma / 255 | Mean HSV saturation / 255 | Mean near-clip pixels |
|---|---:|---:|---:|---:|
| 8 / 1.0 | 4.25 | 99.3 | 101.2 | 7.2% |
| 8 / 1.2 | 7.51 | 90.0 | 108.9 | 10.3% |
| 8 / 1.5 | 7.56 | 82.1 | 116.6 | 14.8% |
| 8 / 2.0 | 7.67 | 72.3 | 126.8 | 19.7% |
| 12 / 1.0 | 5.90 | 102.2 | 101.0 | 8.8% |
| 16 / 1.0 | 7.74 | 103.1 | 100.0 | 9.5% |
| 12 / 1.2 | 11.33 | 95.1 | 104.3 | 10.4% |
| 12 / 1.5 | 11.26 | 88.4 | 109.3 | 13.8% |

At 8 steps, each tested CFG above 1 **increased mean HSV saturation, reduced mean luma and increased near-clipped pixels against CFG 1 in all eight matched scene/seed pairs**. Mean saturation rises across the settings, though not every individual image rises at every increment (6/8 from 1.2→1.5; 7/8 from 1.5→2.0). Relative to CFG 1, the means at CFG 2 are +25.6 saturation units, −27.0 luma units and +12.5 percentage points near-clip. This is a reproducible directional appearance change, not a measure of improvement. CFG above 1 also invokes the negative branch: median 8-step time rose from 4.25 s at CFG 1 to 7.51–7.67 s (about 1.8×). Raising steps alone at CFG 1 did **not** produce the same saturation/darkening trend; 12 and 16 steps visibly changed composition as well as details. At 12 steps, CFG 1.2 and 1.5 cost about 11.3 s per image here (roughly 2.7× the 8/CFG 1 median).

Non-blind inspection of all eight labelled sheets found no universal visual winner. At 8 steps, moderate CFG can make some lettering and edges look more assertive, while higher CFG darkens the airlock, portrait and comic and can reduce apparent glaze texture in the ceramic scene. Extra steps change the placement or shape of objects and text, so a pixel MAE from the baseline cannot be interpreted as a quality score. Small lettering remains imperfect in some settings. These are observations to test with a blind preference and prompt-accuracy review, not success counts. The result supports keeping **8/CFG 1 as the safe reference** and treating 8/1.2 and 12/1.2 as comparison candidates; it does not support changing a general default yet.

Eight larger, randomly ordered sheets with only letters A–H are in `blind/`. The key is separate in `blind_key.json`; the archived `blind_ratings.csv` is the **empty scoring template**, preserved unchanged. The maintainer rated the sheets independently before the key was opened; the decoded results appear below. This first screen has four prompts and two seeds each, both selected partly from successful prior renders; no held-out prompt set or 1024×1024 check has been run. Do not infer population failure rates or a global optimum from it. Compare at most two candidates with 8/CFG 1 on new prompts/seeds and separately check 1024×1024.

AI-model reviews (GPT, Gemini and the assistant that ran the experiment) of these sheets are in [AI-model reviews](#ai-model-reviews-qualitative-not-measurements); they are not counted here.

### Maintainer preference decoded (2026-11-11)

The maintainer supplied `blind_ratings.csv`, preserved as `<archive>/baked_turbo_v094_2026-11-11/user_ratings.csv` (SHA-256 `54236468fb2181dc83925045b9016234a730be938f2b49b1d2921ec5d74ea11d`). Its 64 scene/seed/letter keys match the archived template exactly, with no missing or duplicate rows. Nine rows contain a letter in `preference_rank_1_best`: one choice on seven sheets and **two choices on trawler portrait / seed 1200**. The two marks on that sheet are treated as tied favorites, not an inferred first/second ranking. `prompt_errors`, `artifacts` and `notes` are blank throughout; blank fields do **not** establish that the images have no defects. The file contains no overall ranking among configurations.

| Scene / seed | Maintainer favorite(s) (decoded) |
|---|---|
| Airlock / original | D = 12 / 1.2 |
| Airlock / 1200 | A = 12 / 1.0 |
| Trawler portrait / original | E = 8 / 2.0 |
| Trawler portrait / 1200 | C = 8 / 1.5; G = 12 / 1.5 (tie) |
| Ceramic texture / original | F = 8 / 2.0 |
| Ceramic texture / 1200 | C = 12 / 1.2 |
| Comic text / original | C = 8 / 1.2 |
| Comic text / 1200 | F = 8 / 2.0 |

The maintainer selected **8 / CFG 2 on 3/8 sheets** and **12 / CFG 1.2 on 2/8**; the other four marks are one each for 12/1.0, 8/1.5, 12/1.5 and 8/1.2. No sheet selected 8/CFG 1, though it remains the technical reference setting, and no sheet selected 16/CFG 1. One reviewer and eight sheets: a direction for the next test, not a preferred default.

This preference result does not overturn the measured CFG tradeoff: at 8 steps, CFG 2 produced higher saturation, lower luma, more near-clipped pixels and roughly 1.8× the median execution time of CFG 1 in this session. Those changes may still be desirable for particular scenes. **Next controlled comparison:** retain 8/CFG 1 as baseline and test **8/CFG 2** (most maintainer picks) and **12/CFG 1.2** (second most) on held-out prompts/seeds, including a separate 1024×1024 check. Specify prompt fidelity, visible defects and preference per image before aggregating; this first rating file records preferences only. No held-out run or default change is claimed here.

### Loader issue found during validation

The first API attempt stopped before sampling: `MOSS Loader` passed its model patcher as the positional `model` argument to `format_model_info` and also passed the filename as `model=...` in metadata, raising `TypeError: format_model_info() got multiple values for argument 'model'`. This is an actual Suite bug, independent of the baked model. The helper's patcher parameter is now named `model_patcher`, preserving the `model` metadata key and attachment contract. A regression test was added; the full Suite test file passes **65 tests, one opt-in CUDA test skipped**. The already running ComfyUI process still has its old imported code until restart, so the screen used native ComfyUI loaders; the four pixel-exact reproductions validate that measurement path. No model file or source PNG was modified, and no commit or push was made.

## Baked v0.9.4 versus SFT/Base + Turbo LoRA v0.9 (2026-11-11)

### Frozen comparison and evidence

The eight existing baked-model images at **8 steps / CFG 1** from the preceding screen are the reference arm. For each of the same four resolved prompts and two seeds, the new arms load `mossImage_sft_bf16.safetensors` or `mossImage_base_bf16.safetensors`, then apply `MOSS/HaDeS-MOSS-BeTa-Turbo-8steps-v0.9.safetensors` through stock `LoraLoaderModelOnly` at strength 1.0. The maintainer explicitly confirmed using this file at eight inference steps. Its internal metadata says `steps: 4` despite `8steps` in the filename; this discrepancy is preserved in the experiment manifest and must not be silently interpreted as a recommendation for four-step inference.

All three arms use the **same resolved positive and negative text, seed, 1152×1536 latent, Qwen3.5 int8 ConvRot text encoder, FLUX.2 VAE, Euler/simple, flow shift 5, denoise 1, SLG off, 8 steps and CFG 1**. The baked arm has no live LoRA. The reproducible runner is `tools/moss_v094_vs_v09_eval.py`; its output is `<archive>/v094_vs_sft_base_v09_2026-11-11`. It queued the 16 missing images through the running local ComfyUI 0.43.0 instance, checking for an empty queue before each submission and leaving the user's open workflow untouched. All 16 runs succeeded. The eight baked reference PNGs were read and hash-checked, never regenerated or rewritten. A graph comparison verified that the 16 new requests match their corresponding reference in text-encoding inputs, latent, sampler parameters, VAE and preview path; model file and live LoRA are the intended differences. `metrics.csv` has all 24 image records, and each new PNG has a JSON sidecar with its exact graph, prompt ID, timing, hash and numerical metrics.

Eight three-image sheets with randomized A/B/C positions are in `blind/`. The mapping is separate in `blind_key.json`; the blank `blind_ratings.csv` remains available for additional reviewers. The maintainer submitted eight A/B/C choices without the mapping; future reviewers should avoid the decoded section, `blind_key.json`, the variant-named PNGs and `metrics.csv` until they have voted.

Numerical luma, saturation, near-clipping and run times are measured, but they do not establish which arm makes the better image. The 8/CFG 1 baked references came from an earlier run, so direct timing ratios also contain session and cache effects. Four source prompts were selected from successful baked renders and include the literal text `HaDeS MOSS v0.9.4`; this is a small, selected sample, not an unbiased quality benchmark. The Base plus v0.9 LoRA arm is a compatibility/quality test, not evidence that the LoRA was trained specifically on Base. Any conclusion from this screen is limited to these prompts, seeds, resolution and settings. No default or model artifact has been changed.

AI-model reviews (GPT, Gemini and the assistant that ran the experiment) of these sheets are in [AI-model reviews](#ai-model-reviews-qualitative-not-measurements); they are not counted here.

### Decoded maintainer preferences

The maintainer's eight choices were transcribed from chat into `user_blind_ratings.csv` (SHA-256 `9358445ac3bddc6a9bb5e92459b336be8b7e2685ef271ea24f826374e6de3ecd`), without altering the blank template. There was one unambiguous choice per sheet.

| Scene / seed | A / B / C mapping | Maintainer first |
|---|---|---|
| Airlock / original | Base / baked / SFT | C = SFT |
| Airlock / 1200 | Baked / Base / SFT | C = SFT |
| Trawler portrait / original | Base / baked / SFT | B = baked |
| Trawler portrait / 1200 | Base / SFT / baked | C = baked |
| Ceramic texture / original | SFT / baked / Base | B = baked |
| Ceramic texture / 1200 | SFT / Base / baked | A = SFT |
| Comic text / original | Base / baked / SFT | C = SFT |
| Comic text / 1200 | Base / SFT / baked | B = SFT |

**Maintainer:** SFT + v0.9 first on **5/8**, baked v0.9.4 first on **3/8**, Base + v0.9 first on **0/8**. By scene, both airlock and both comic sheets favor SFT; both trawler portraits favor baked; ceramic splits one each. Preferences, not image-quality measurements or prompt-accuracy rates.

| Variant | Images | Median seconds | Mean luma / 255 | Mean HSV saturation / 255 | Mean near-clip pixels |
|---|---:|---:|---:|---:|---:|
| Baked v0.9.4 | 8 | 4.25 | 99.3 | 101.2 | 7.2% |
| SFT + Turbo v0.9 | 8 | 4.89 | 93.5 | 98.5 | 2.7% |
| Base + Turbo v0.9 | 8 | 4.64 | 91.9 | 94.3 | 5.6% |

These image metrics describe tonal appearance, **not better quality**; the baked renders were collected earlier, so cache state differs. The vote does **not** support claiming that baked v0.9.4 generally makes a better image than SFT + v0.9 at these settings; it shows a possible portrait preference for baked v0.9.4, from two seeds of one prompt. A new prompt set with text and artifact scoring is needed before changing defaults or making a broad model-quality claim.

## Held-out v0.9.4 versus SFT + Turbo v0.9: eight new scenes (2026-11-11)

### Design and execution

The previous comparison suggested a possible v0.9.4 advantage in people, especially the trawler portrait, but its prompts were selected from successful baked renders. This second screen uses **eight new prompts fixed before rendering**, one seed each, with no source image, reroll or selection by outcome. Four center a person and expose face, hands and clothing: watchmaker (2401), trumpeter (2402), climber (2403) and flower seller (2404). Four contain no people and test mechanical structure, reflections, materials or text: open pocket watch (2405), rainy alley with `CAFE 24` (2406), citrus still life (2407) and lunar observatory with `OBSERVATORY 7` (2408). Their full exact text and negative prompt are frozen in `tools/moss_v094_sft_heldout.py` and the output manifest. None names the model or asks it to print a version number.

The two arms are `HaDeS_MOSS_v0.9.4_Turbo.safetensors` with no live LoRA, and `mossImage_sft_bf16.safetensors` with `MOSS/HaDeS-MOSS-BeTa-Turbo-8steps-v0.9.safetensors` at strength 1.0 through stock `LoraLoaderModelOnly`. Both use the same Qwen3.5 int8 ConvRot text encoder, FLUX.2 VAE, 1152×1536 latent, fixed positive/negative text, seed, Euler/simple, flow shift 5, denoise 1, SLG off, **8 steps and CFG 1**. This tests the complete rendering pipelines at the maintainer's requested natural Turbo setting; it cannot isolate which weight differences within the baked model cause any appearance change.

The archive is `<archive>/v094_vs_sft_v09_heldout_2026-11-11`. The local ComfyUI 0.43.0 queue produced **16/16 successful images** without altering the user's open workflow. Every new image has a JSON sidecar with exact graph, prompt ID, timing, hash and metrics; `metrics.csv` holds all 16 measurements. A post-run check confirmed the eight graph pairs match in text-encoding inputs, VAE, latent, sampler settings and preview path; checkpoint and LoRA application are the intended differences. The eight sheets are in `blind/*.jpg`; each also has full-resolution `blind/<scene>_A.png` and `_B.png` copies re-encoded with **no embedded graph metadata**. All 16 full-resolution copies were checked for empty PNG metadata. `blind/prompts.md` gives the prompt text and rating instructions, and `blind/ratings.csv` is an untouched form with 16 rows. The A/B mapping is separate in `blind_key.json` at the archive root.

The maintainer supplied eight A/B choices before the key was opened. AI-model reviews (GPT, Gemini and the assistant that ran the experiment) of these sheets are in [AI-model reviews](#ai-model-reviews-qualitative-not-measurements); they are not counted here.

For the person hypothesis, score **overall preference** separately from skin/face detail, hand anatomy, clothing texture and prompt fidelity. More visible texture alone does not establish greater realism or a better image. The four prompts per group provide only a first check of scene dependence; one seed per prompt, one resolution and one CFG/step setting cannot estimate a general model advantage. No defaults, model artifacts or frontend nodes were changed, and no commit or push was made.

### Decoded maintainer review

The maintainer's chat choices were recorded as `user_blind_ratings.csv` (SHA-256 `edcbdef4cfa0ead3492f927f16b7e6e186bec9bf2409936ee3670fe1f0778e84`), preserving the blank form. The user's “watch A” was assigned to `watch_macro`; `watchmaker B` was separately specified. No user rationale or numerical detail scores were supplied, so only overall preferences can be attributed to the maintainer.

| Scene | Group | A / B | Maintainer overall |
|---|---|---|---|
| Watchmaker | Person | SFT / baked | B = baked |
| Trumpeter | Person | SFT / baked | B = baked |
| Climber | Person | SFT / baked | B = baked |
| Flower seller | Person | Baked / SFT | B = SFT |
| Pocket-watch macro | No person | SFT / baked | A = SFT |
| Rainy alley | No person | SFT / baked | B = baked |
| Citrus still life | No person | Baked / SFT | B = SFT |
| Lunar observatory | No person | Baked / SFT | A = baked |

The maintainer prefers **baked v0.9.4 on 5/8** overall: **3/4 person** scenes and **2/4 nonperson** scenes. One seed per prompt and one reviewer: a hypothesis about people, not a portrait advantage.

| Variant | Images | Median seconds | Mean luma / 255 | Mean HSV saturation / 255 | Mean near-clip pixels |
|---|---:|---:|---:|---:|---:|
| Baked v0.9.4 | 8 | 4.87 | 95.9 | 91.4 | 6.6% |
| SFT + Turbo v0.9 | 8 | 4.60 | 96.7 | 74.1 | 2.2% |

Baked v0.9.4 has higher mean HSV saturation in **all eight pairs** and more near-clipped pixels in **seven of eight**. These are appearance measurements, not detail or quality scores; higher saturation and contrast can make texture look stronger without adding faithful structure. The execution times are session-specific, and baked was submitted first in every pair, so this is not a controlled speed comparison. A pixel difference or generic sharpness metric would also be confounded by different compositions and face sizes. The previous selected-prompt screen favored SFT in the maintainer's vote (5/8), whereas these new prompts favor baked (5/8): the direction changes with the test set. Do not infer a global winner from either eight-image screen. Further work on the person hypothesis should separately score facial texture **and** facial realism/anatomy on more prompts and seeds before changing a default or making a broad quality claim.

**Maintainer's trumpeter note:** one of the hands in **A (SFT + v0.9)** looks strange to them, and that defect decided their vote for **B (baked v0.9.4)**. A qualitative judgment, not a counted hand-anatomy failure rate.

## Quick practical comparison: baked v0.9.4 versus SFT + Turbo v0.7 (2026-11-11)

The maintainer asked whether a comparison against the earlier Turbo v0.7 would be equitable, since v0.9 is unreleased and only they can run it. The installed `MOSS/HaDeS-MOSS-Turbo-8steps-v0.7.safetensors` is the documented 8-step v0.7 file, not the experimental `v0.7a` checkpoints. Its embedded metadata specifies **MOSS-Image SFT, strength 1.0, 8 steps, CFG 1, Euler/simple and shift 5**, matching the settings used for baked v0.9.4. The local README lists this v0.7 as a released LoRA. This is a fair **practical pipeline comparison at shared settings**. It cannot isolate whether differences come from v0.9.4's underlying checkpoint, its baked Turbo weights or the change from Turbo v0.7 to v0.9. It also cannot establish the maintainer's proposed long-run consistency advantage for v0.9 without a larger predefined sample and a defect rubric.

To make a short, directly paired test, `tools/moss_v094_vs_v07_eval.py` reuses the exact **eight baked v0.9.4 PNGs, prompts and seeds** from the preceding held-out experiment and generates only the eight missing SFT + v0.7 images. All 16 compared images use the same text encoder, VAE, 1152×1536 latent, prompt text, negative text, seed, 8 steps, CFG 1, Euler/simple, shift 5, denoise 1 and SLG off. Only the checkpoint and LoRA application differ. The archive is `<archive>/v094_vs_sft_v07_2026-11-11`; `manifest.json` records the v0.7 file hash and its metadata. All **8/8 new runs succeeded** through local ComfyUI 0.43.0, with a free-queue check before each submission and no change to the user's active workflow. The eight reference images were hash-checked and not regenerated. A post-run check confirmed the eight paired graphs match in text-encoding inputs, VAE, latent, sampler parameters and preview path. `metrics.csv` holds 16 rows; each new PNG has an exact graph, prompt ID, image hash, timing and metrics sidecar.

The eight randomized A/B sheets and 16 full-resolution A/B copies are under `blind/`. All blind PNGs were re-encoded and checked to contain **no graph metadata**. The folder also contains `prompts.md` and an untouched `ratings.csv`; the mapping is separate in `blind_key.json` at the archive root. Because the baked images were already shown in the preceding experiment, returning reviewers can recognize them: **the rating is only partially blind**, even with A/B labels. Do not open `blind_key.json`, variant-named PNGs or `metrics.csv` before rating if preserving the remaining blinding matters.

AI-model reviews (GPT, Gemini and the assistant that ran the experiment) of these sheets are in [AI-model reviews](#ai-model-reviews-qualitative-not-measurements); they are not counted here.

### Decoded maintainer votes

The maintainer's A/B choices were transcribed from chat to `user_blind_ratings.csv` (SHA-256 `3f224bd29ae071c7293967d5d0400541b1c6c6a5937a40ab2990968795ad7465`). Their “watch B” refers to `watch_macro`; `watchmaker B` was separately specified.

| Scene | A / B mapping | Maintainer |
|---|---|---|
| Watchmaker | SFT v0.7 / baked | Baked |
| Trumpeter | SFT v0.7 / baked | Baked |
| Climber | Baked / SFT v0.7 | Baked |
| Flower seller | Baked / SFT v0.7 | SFT v0.7 |
| Pocket-watch macro | Baked / SFT v0.7 | SFT v0.7 |
| Rainy alley | SFT v0.7 / baked | Baked |
| Citrus still life | Baked / SFT v0.7 | Baked |
| Lunar observatory | SFT v0.7 / baked | Baked |

The maintainer prefers **baked v0.9.4 on 6/8** against SFT + v0.7 (5/8 against SFT + v0.9 on the same prompts in the previous round). They report that **trumpeter A (SFT + v0.7)** appears to have six fingers on one hand, which decided that vote; the count is not independently verified. The result supports baked v0.9.4 as a competitive practical alternative to the documented v0.7 pipeline in this short, partially blind screen. It does not establish model-wide superiority or a long-run consistency advantage; that needs a larger predefined prompt / seed series with explicit defect counts.

| Variant | Images | Median seconds | Mean luma / 255 | Mean HSV saturation / 255 | Mean near-clip pixels |
|---|---:|---:|---:|---:|---:|
| Baked v0.9.4 | 8 | 4.87 | 95.9 | 91.4 | 6.6% |
| SFT + Turbo v0.7 | 8 | 4.38 | 89.9 | 75.7 | 3.5% |

The color numbers describe appearance, not fidelity or realism. The baked timings were recorded in the earlier run and the SFT + v0.7 timings in this run, so the median difference is not a controlled speed result. No default, model file or user workflow was changed for this comparison.

## HaDeS MOSS v0.9.4 Turbo outside 8 steps / CFG 1: steps, CFG and café bias (2026-11-11)

`tools/moss_v094_range_eval.py`, archive `<archive>/v094_range_2026-11-11`, 84 images, all successful. Stock ComfyUI nodes, `HaDeS_MOSS_v0.9.4_Turbo.safetensors`, the Qwen3.5 int8 ConvRot text encoder and FLUX.2 VAE, euler / simple, shift 5, denoise 1. The eight held-out scenes and seeds of the v0.9.4 versus SFT + Turbo v0.9 comparison, 1152 x 1536; the 8-step images reproduce that comparison's baked images pixel for pixel (difference 0.0 on all eight). One seed per scene: what follows holds for these scenes, not as a rate.

**Steps** (CFG 1, 8 scenes, medians):

| Steps | Seconds | Mean luma / 255 | Mean HSV saturation / 255 | Near-clip pixels | Laplacian std | High-frequency energy |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.5 | 95.4 | 85.8 | 5.2% | 15.8 | 1.43% |
| 8 | 4.0 | 100.1 | 87.8 | 5.9% | 18.7 | 1.67% |
| 12 | 5.8 | 101.2 | 90.1 | 8.0% | 20.5 | 2.29% |
| 20 | 9.5 | 107.3 | 93.7 | 9.1% | 24.7 | 3.22% |
| 30 | 14.0 | 108.6 | 93.0 | 10.9% | 27.5 | 3.72% |
| 50 | 23.0 | 109.2 | 94.2 | 11.7% | 29.6 | 4.27% |

- No image broke or burned from 6 to 50 steps (by eye, all 48). More steps add texture and fine detail (Laplacian and high-frequency energy rise at every step count, 50 steps about 2.6x the high-frequency share of 8) and brighten slightly; the near-clipped share doubles.
- The image does not converge: the mean pixel change from one step count to the next stays 8-52 / 255 even from 30 to 50, against 36-57 between 8 and 50. People keep their pose and identity; backgrounds, objects and wide scenes change (the observatory widens and rebuilds its room from 20 steps). Choosing a step count chooses an image, as at 8-16 steps in the CFG / step screen above.
- Text: `OBSERVATORY 7` right at every step count; the alley sign read `CAFE24` at 6, only `24` at 8, and `CAFE 24` from 12 to 50.

**CFG** (20 steps; watchmaker, trumpeter, pocket watch, rainy alley; medians):

| CFG | Seconds | Mean luma / 255 | Mean HSV saturation / 255 | Near-clip pixels | Laplacian std |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 9.8 | 86.7 | 93.6 | 8.1% | 18.7 |
| 1.5 | 18.5 | 78.7 | 98.3 | 15.3% | 21.9 |
| 2.0 | 18.5 | 76.8 | 104.3 | 16.8% | 21.5 |
| 2.5 | 18.5 | 72.0 | 112.7 | 18.6% | 22.0 |
| 3.0 | 18.5 | 66.9 | 112.9 | 19.7% | 23.9 |

- No burn up to CFG 3 on these four (by eye); each step up darkens and saturates the image, and near-clipped pixels more than double from 1 to 3. Above CFG 1 every step runs the model twice (1.9x the time). Above 1.5 the prompt's "navy apron" spread to the whole outfit of the watchmaker; `CAFE 24` stayed right at every CFG.

**Café bias** (`photograph of a woman sitting in a café`, seeds 0-15, 1024 x 1024, 8 steps, CFG 1, the protocol of the 2026-11-09 counts): by eye, **9 of 16 clearly East Asian, 2 ambiguous** (about 9-11 / 16). Same counter and protocol: Base 25 steps about 2-4 / 16, the BeTa Turbo v0.9 LoRA live on Base about 12 / 16 with euler. v0.9.4 keeps most of the bias of the BeTa turbo baked into it; the other two LoRAs do not remove it. One counter, by eye, one prompt.

## Turbo v1.0: neutral people prompts against the café bias (2026-11-12)

**Question.** Turbo v0.9 (BeTa) drifted to East Asian faces on the neutral café prompt (about 12 / 16, Base about 2-4). Its training prompts named an ethnicity for 86 of 115 people prompts, and only 29 of 1024 left the appearance open. DMD follows the teacher's dominant mode. Do the teacher's own regression pairs on neutral people prompts keep Base's variety?

**Recipe.** `tools/moss_dmd_train.py` with v0.9's settings: Base as teacher and student, int4 text encoder, 512 px, 4 steps, rank 2 on layers 24-41, 3000 updates, `--reg-weight 0.05`, `--fake-updates 5`. Two things changed:
- **Prompts:** `tools/moss_turbo_v10_prompts.py` replaced 230 no-person prompts of v0.9 with photographs of people whose ethnicity is not given, so 234 of 1024 prompts are neutral people prompts.
- **Regression pairs:** `--reg-pairs 1024`, one teacher pair per prompt.

The run finished with 3000 updates and 0 skipped. Run folder: `<archive>/moss_turbo_runs/dmd512_base_v10`.

**Café bias** (v0.9 protocol: Base + LoRA at 1.0, 8 steps, euler / simple, CFG 1, 1024 x 1024, seeds 0-15). Clearly East Asian faces, counted by eye by one counter:

| Checkpoint | 600 | 1000 | 1400 | 2000 | 3000 |
|---|---:|---:|---:|---:|---:|
| Clearly East Asian (ambiguous) | 7 (up to 10) | 6 (up to 9) | 9 (up to 10) | 7 (up to 8), 2 with no face | 8, 2 with no face |

v0.9 counted about 8-9 early and about 11-12 at the end. In v1.0 the count stays flat through training instead of growing. At 3000 updates it is about 8 / 16, still above Base. One prompt, one counter.

**Quality** (Base + LoRA, 8 steps, euler / simple, CFG 1). Test set:
- the eight held-out scenes (1152 x 1536, their seeds);
- the two text prompts of `moss_eval` (1024 x 1024, seeds 1000-1002).

Medians of 14 images. Archive: `<archive>/moss_eval/turbo_v10/quality`.

| LoRA | Mean luma / 255 | Mean HSV saturation / 255 | Near-clip pixels | Laplacian std | High-frequency energy |
|---|---:|---:|---:|---:|---:|
| Turbo v0.9 (BeTa) | 109.7 | 89.3 | 4.5% | 14.9 | 1.49% |
| v1.0, 2000 updates | 110.6 | 95.6 | 9.7% | 15.7 | 1.65% |
| v1.0, 3000 updates | 119.6 | 74.2 | 6.1% | 13.3 | 1.22% |

- **Text:** the bakery sign and the three-line poster were right on all 6 seeds with every LoRA. The small text in the held-out scenes (`CAFE 24`, `OBSERVATORY 7`) was right with v0.9 and with 3000 updates. With 2000 updates both were wrong.
- **Look:** 3000 updates is brighter and less saturated than v0.9, and slightly softer.
- **Maintainer verdict:** after testing 3000 updates on their own prompts, the maintainer judged it the best turbo LoRA trained so far. It becomes **Turbo v1.0**. This is a judgement, not a blind vote.

### Turbo v1.0 at CFG 1.5: the guidance and the negative, separated (2026-11-12)

**Setup.** Same 14 jobs and settings as the quality table above, with v1.0 at 3000 updates. Three arms:
- CFG 1;
- CFG 1.5 with the held-out negative;
- CFG 1.5 with an empty negative.

At CFG 1 ComfyUI never evaluates the negative: `comfy/samplers.py` skips the uncond pass when `cfg == 1`. So the CFG 1 arm is the quality run's images. Going from CFG 1 to 1.5 therefore adds two things: the guidance, and the negative.
- CFG 1 against 1.5 with the empty negative isolates the guidance.
- 1.5 empty against 1.5 with the negative isolates the negative.

Archive: `<archive>/moss_eval/turbo_v10/cfg`.

| Arm | Mean luma / 255 | Mean HSV saturation / 255 | Near-clip pixels | Laplacian std | High-frequency energy |
|---|---:|---:|---:|---:|---:|
| CFG 1 | 119.6 | 74.2 | 6.1% | 13.3 | 1.22% |
| CFG 1.5, empty negative | 102.5 | 104.9 | 9.9% | 16.3 | 1.81% |
| CFG 1.5, negative | 98.2 | 105.5 | 10.4% | 16.1 | 1.80% |

- **The guidance moves the image, not the negative.** Mean pixel change per image:
  - CFG 1 against 1.5 empty (the guidance): 17-34 / 255, median 28.
  - 1.5 empty against 1.5 with the negative: 3-17 / 255, median 10.

  With the negative or without it, CFG 1.5 gives the same composition with small changes in detail. The guidance darkens and saturates the image and sharpens it, about to where v0.9 sits at CFG 1. The negative only darkens it a little more.
- **Text:** right on all 14 jobs in all three arms: the bakery sign and the poster on every seed, and `CAFE 24` and `OBSERVATORY 7`. At CFG 1 none of these 14 was wrong, so this test cannot show a text gain at CFG 1.5. The broken "E" of "EUROPA" in the maintainer's example at CFG 1 (fixed at 1.5) was one seed of a prompt outside this set.
- **Cost:** above CFG 1 every step runs the model twice.

### Turbo 8-steps SFT v1.0 (2026-11-23)

**Training.** The Base v1.0 recipe rerun from zero with SFT BF16 as teacher and student using `tools/moss_dmd_train.py`. Same 1,024 prompts, encoder and seed (0); the saved prompt embeddings matched the Base run byte for byte. SFT generated its own 1,024 regression targets at 25 steps, CFG 3 and negative-pass layer-22 skip. Generator: 512 x 512, 4 steps, CFG 1, rank 2 in layers 24-41; regression weight 0.05 and five fake-score updates per generator update. The archived report for `<archive>/moss_turbo_runs/dmd512_sft_v10` records 3,000 updates, zero skipped. The EMA and `lora_u03000.safetensors` agree on all 216 tensors. Release export: 144 LoRA matrices rounded to BF16, 72 FP32 alpha scalars preserved, 3,564,296 bytes.

**Comparison protocol.** ComfyUI output `HaDeS/compare/lora_compare_00164_.png` through `lora_compare_00208_.png`. Both branches use the same SFT BF16 model, Qwen3.5-9B int8 ConvRot encoder, FLUX.2 VAE, seed 1200, Euler / simple, shift 5, denoise 1 and SLG off. A loads the released Base v1.0 LoRA; B loads the SFT u3000 FP32 training export, SHA256 `d761fe7ca0fbeCD524af07f17a6db65a3c9e34e8db73db4f026532ba0054a073`. The BF16 export was checked tensor by tensor against the rounded source.

**Main test.** 21 distinct scenes at 1344 x 1792, 8 steps, CFG 1, strength 1.0 (sheets 165-168, 175, 182, 188-202). Metrics use the image panels only. Luma is the Rec. 709-weighted sum of 8-bit RGB; HSV saturation is on the 0-255 scale. Extreme pixels have luma below 5 or above 250. Laplacian is the standard deviation of the four-neighbour luma Laplacian.

| LoRA, both on SFT | Mean luma | Mean HSV saturation | Median extreme pixel share | Median Laplacian |
|---|---:|---:|---:|---:|
| Base v1.0 | 106.0 | 75.9 | 0.92% | 16.04 |
| SFT v1.0 | 96.1 | 90.7 | 2.21% | 15.17 |

- SFT was darker in 21/21 pairs, more saturated in 17/21, and had more extreme pixels in 20/21. Renders retained similar compositions in most pairs by eye. Both LoRAs spelled `OPEN`, `VINYL ONLY`, `THE JUNGLE EATS US ALL!` and the main *Oaken Valley* title correctly in the inspected examples (166, 193, 195, 196).
- Step sweep: four scenes were coherent at 4 steps (169-171, 173). In the ceramicist sweep, both were softer at 2 steps; texture recovered through 8 and changed more gradually through 20 (172-181).
- CFG sweep at strength 1.0: on the studio portrait, both darkened from CFG 1 to 2.5 and SFT was darker throughout (182-187). At strength 0.5 the two LoRAs gave visually close results at 8 steps, CFG 2.5 on the forest and two portraits (206-208). At 15 steps, strength 0.5 and CFG 1, sheets 203-205 repeat the same A / B forest pair.
- Scope: one seed, no SFT-without-LoRA baseline, no appearance-bias test for this release, and no isolated negative-prompt test. The image metrics above are from the FP32 training export.

## MOSS KSampler denoise_mode: core against exact on img2img (2026-11-12)

**Question.** At 4-8 steps, does `exact` give a denoise that moves smoothly where the core schedule jumps, and does it cost image quality?

**Setup.** Img2img through `MOSS KSampler`. Archive with script, inputs and records: `<archive>/moss_eval/denoise_exact`.
- **Model:** MOSS-Image Base + Turbo 8-steps v1.0 at 1.0, euler / simple, CFG 1, shift 5, seed 7.
- **Prompt:** the same prompt as the input image.
- **Inputs:** four 1152 x 1536 images:
  - two photographs rendered by HaDeS MOSS v0.9.4 (watchmaker, rainy alley);
  - two images from the maintainer: an anime knight and a flat "VISIT EUROPA" poster.
- **Grid:** 4 and 8 steps, denoise 0.3-0.9 in steps of 0.1, both modes: 112 images, all successful. The start sigma is read from the KSampler's own `info`.

**Start sigma and mean distance to the input** (RGB MAE / 255; watchmaker, alley, knight, poster):

| Denoise | 4 steps, core | 4 steps, exact | 8 steps, core | 8 steps, exact |
|---:|---|---|---|---|
| 0.3 | 0.690: 11 / 22 / 12 / 7 | 0.682: 11 / 22 / 12 / 7 | 0.690: 12 / 23 / 13 / 7 | 0.682: 12 / 23 / 13 / 7 |
| 0.5 | 0.833: 17 / 32 / 19 / 11 | 0.833: same image | 0.833: 18 / 33 / 20 / 12 | 0.833: same image |
| 0.6 | 0.909: 23 / 42 / 28 / 18 | 0.882: 20 / 38 / 24 / 15 | 0.889: 22 / 41 / 26 / 16 | 0.882: 21 / 40 / 25 / 16 |
| 0.7 | **0.952**: 33 / 52 / 39 / 31 | 0.921: 25 / 44 / 30 / 19 | 0.930: 28 / 47 / 34 / 23 | 0.921: 27 / 46 / 31 / 20 |
| 0.8 | **0.952: the 0.7 image again** | 0.952: 33 / 52 / 39 / 31 | 0.952: 35 / 55 / 41 / 32 | 0.952: 34 / 55 / 40 / 32 |
| 0.9 | **1.000: pure noise** | 0.978: 53 / 58 / 48 / 38 | **1.000: pure noise** | 0.978: 54 / 62 / 48 / 40 |

**Up to 0.5 the two modes are the same.** At 0.4 and 0.5 the start sigma is identical, and at 0.3 it is 0.690 against 0.682. The mode matters only above 0.5.

**Core jumps, as ComfyUI's schedule code predicts:**
- At 4 steps, 0.7 and 0.8 give pixel-identical images.
- At **0.9 the core starts from pure noise at 4 and at 8 steps** (`int(8 / 0.9) = 8`), so the input is ignored:
  - The poster became a new, washed-out poster.
  - At 8 steps its title broke: "VISIT" was cut off.

**Exact is continuous.** Distance to the input rose with every 0.1 step, on all four inputs and at both step counts. At the same denoise, 4 and 8 steps landed within about 4 / 255 of each other.

**At 0.9 exact (sigma 0.978), the input is mostly gone:**
- The poster kept its layout, palette and text.
- The knight kept the character and the setting in a new pose.
- The photographs became new scenes; the alley lost its sign.

**Quality.**
- **At the same start sigma**, the two modes give the same Laplacian sharpness and no artifacts by eye.
- **At 4 steps, 0.6-0.7 exact** fills the gap the core skips, with clean images.
- **Side observation:** the input alley's sign read only "24"; denoise 0.7-0.8 rewrote it as "CAFE 24" in both modes.

**Limits.** One seed, one model setup, four inputs. Image quality was judged by eye, not voted.

## HaDeS MOSS v0.9.4 Turbo: source and baked recipe (2026-11-11)

The maintainer supplied the complete construction workflow after the comparison rounds. Its active path is **`mossImage_base_bf16.safetensors` (MOSS Image Base) → MOSS Multi-LoRA → MOSS Model Saver**. The visually dimmed LoRA Extractor/Merger branches in the screenshot do not feed the Multi-LoRA input and are not part of this saved model's stated recipe. This corrects any possible reading of the SFT comparison arm as the baked model's source: **v0.9.4 starts from Base, not SFT**. The maintainer reports that both the BF16 and int8 ConvRot v0.9.4 files use the following recipe; the screenshots directly show the BF16 saving path, while the exact int8 conversion path is not shown.

| Active input, in node order | Multi-LoRA strength | Provenance |
|---|---:|---|
| `mossImage_base_bf16.safetensors` | — | Source checkpoint shown in the active MOSS Loader. |
| `MOSS/data_expanded.safetensors` | 0.30 | Maintainer describes this as an expanded-dataset LoRA trained with SFT; its file header identifies `moss_image` but does not name an SFT checkpoint. Applied here to Base. |
| `MOSS/HaDeS_MOSS_color_detailer_v0.2.safetensors` | 0.60 | A pre-merged color/detail LoRA. Its own header names `HaDeS_test_colour_v01.safetensors` and `moss_context_detaile_tweaker_20261018_rank2.safetensors`, with **inner merge weights 0.6 and 0.75** respectively. These are distinct from the outer 0.60 application strength. |
| `MOSS/HaDeS-MOSS-BeTa-Turbo-8steps-v0.9.safetensors` | 1.00 | Private Turbo v0.9 used at 8 inference steps in the measured comparisons. Its header says `steps: 4`; the filename and actual inference setting say 8. The header alone cannot resolve the training-step convention. |

The BF16 Model Saver path in the screenshot specifies **bfloat16**, `nearest` rounding, metadata enabled and author `HaDeS`. Local safetensors headers confirm `modelspec.architecture=moss_image`, 42 layers and `hades/precision=bfloat16` for `HaDeS_MOSS_v0.9.4_Turbo.safetensors`; the companion `HaDeS_MOSS_v0.9.4_Turbo_int8_Convrot.safetensors` header declares `int8 ConvRot (Comfy-Org layout)` and 42 layers. Neither final header contains `hades/recipe`: the Model Saver recipe field is blank in the supplied screenshot, and the saver writes that metadata key only for nonempty recipe text. The recipe above is therefore recorded **externally**, not embedded in either model file. The files were read only; no model was re-saved or modified.

Source screenshots are archived at `<archive>/v094_recipe_2026-11-11/`: `base_and_recipe_workflow.png` (SHA-256 `32082384c1b52d49f6a86fbf049d50d00b7b0202776c7f974e75ea76fb36329f`) and `multilora_recipe.png` (SHA-256 `16361a040e76d27f28193d279ea1e26c694eb4498543eef18494120912889250`). The measured quality comparisons above used the **BF16 v0.9.4 file**; they do not establish visual parity with the int8 ConvRot companion. The maintainer considers v0.9.4 publishable as an alternative to SFT. That is a release decision supported by the short practical comparisons, not a claim of universal image-quality superiority.

## MOSS Image Saver: selected-stage provenance and UI validation (2026-11-11)

The Image Saver now accepts optional `primary_stage_metadata` from a second Suite Telemetry. The existing `metadata` input still describes the whole pipeline; the selected stage supplies A1111 parameters, filename tokens and model/LoRA hashes. The stage must contain exactly one MOSS KSampler and at most one Loader and Prompt Enhancer. Without the new connection, the earlier first-KSampler behavior remains. A direct sampler-only stage is accepted but emits a note that its model information is missing. The PNG keeps the full `hades_moss` list and adds `hades_moss_primary_stage` only when a stage is selected. This is manual stage selection; pixel-only postprocessing has no MOSS KSampler stage to select.

An isolated CPU ComfyUI **0.43.0 / frontend 1.57.4** instance on port 8190 used a model-free two-stage workflow. It was saved and reloaded in the frontend. In the generated PNG, `parameters` names the refine prompt, **12 steps, seed 22 and model `refine_ui`**; the filename is `stage_ui_refine_ui_22_12_00001_.png`. The full metadata has six node records and the selected stage has three. The embedded workflow is present. The model names are synthetic and no diffusion model was loaded or rendered; the solid-color image tests provenance and UI behavior only. Evidence and the reusable editor/API workflows are in `<archive>/stage_metadata_validation/`.

| Frontend mode | Actual check | Result |
|---|---|---|
| Classic canvas, HaDeS widgets | New stage socket and global socket connected; execution, cached output, preview, style switch, zoom and manual resize | Passed |
| Classic canvas, standard ComfyUI widgets | Same connections, execution/cached output, preview and manual resize | Passed after fixing native preview hiding |
| Nodes 2.0 / Vue | Connections, editable filename field, execution, preview, zoom context and manual resize; saved workflow survived reload | Passed |

The style-switch regression was found in the real frontend: the Suite's `addCustomWidget` hook had also registered ComfyUI's later native image preview as a decorative HaDeS widget, so switching to standard widgets hid it. The hook now tracks only Suite widgets named `hades_*`; a JS regression test guards the native preview. Python suite: **66 tests passed, 1 opt-in CUDA test skipped**. JS widget suite: **5 passed**. `git diff --check` reported no whitespace errors. No full diffusion run, genuine two-sampler image comparison or hash check on real model files was performed in this UI smoke test; the unit test exercises resource selection and hashes with controlled files. The isolated workflow and generated PNGs are retained under `<archive>`; the user's active workflow and settings were untouched. No commit or push was made.

## Pair Guard and Recipe Preview: development-node validation (2026-11-11)

*Both nodes were removed later the same day: the pair check became `tools/moss_pair_check.py` (for saved images), and Recipe Preview repeated what the `info` outputs and the Metadata Reader show. The record below is kept as it was.*

`MOSS Pair Guard` and `MOSS Recipe Preview` are V3 development nodes. Pair Guard checks matching RGB batch geometry and, when both stage metadata inputs are supplied, compares the declared KSampler controls. A different model or LoRA is permitted and reported; an unequal seed, prompt or other declared control stops the downstream comparison. Without metadata it reports geometry-only coverage, and `require_metadata` can make this an error. It does not prove that the info strings belong to the image tensors. Recipe Preview passes the trimmed text to Model Saver's `recipe` input, validates any Suite trailers, and optionally compares them with the model's attached recipe. It does not inspect weights or save a file. Its `include_metadata` switch is an advisory mirror of Model Saver's switch, not a live connection to it.

The model-free editor workflow [`examples/dev/HaDeS MOSS Suite - Pair Guard and Recipe Preview (Dev).json`](../examples/dev/HaDeS%20MOSS%20Suite%20-%20Pair%20Guard%20and%20Recipe%20Preview%20(Dev).json) was generated from the actual `/object_info` of the isolated server (9 nodes, 7 links). The synthetic 64×64 images and declared seed 1200 passed through Pair Guard to Image Audit; the candidate seed changed to 1201 produced the expected `controls do not match: seed` error in the frontend, and restoring 1200 restored the pass. The separate Recipe Preview displayed its entire dummy recipe and confirmed that one `HADES_META` trailer would be passed. The strings name illustrative files; no model or LoRA was loaded, sampled or saved.

| Frontend mode | Actual check | Result |
|---|---|---|
| Classic canvas, HaDeS widgets | Connected image/metadata sockets, editable source strings, queue execution and cached reports; both nodes enlarged manually and viewed at 54% zoom | Passed; reports remained within node bounds and were read-only. |
| Classic canvas, standard ComfyUI widgets | Switched the Suite setting, executed the same graph, resized both nodes, checked text wrapping and reports | Passed; reports stayed visible after the style change. |
| Nodes 2.0 / Vue | Executed graph, edited metadata seed and `include_metadata`, checked expected failure and updated advisory report, resized both nodes, then saved and reloaded the workflow | Passed; input values, sockets and sizes survived reload. Reports reappeared after executing the reloaded graph. |

Validation ran on a separate **ComfyUI 0.43.0 / frontend 1.57.4** CPU instance with only this Suite enabled. Its API request/result, server object definitions and the frontend-saved workflow are under `<archive>/pair_recipe_validation/`; the user's active ComfyUI workflow and settings were not changed. The reports have native read-only output widgets: legacy `readOnly` and Vue `options.read_only`, with reserved layout height, wrapping and internal scrolling. Cached successful results were checked. After a failed queue item, ComfyUI can still display the previous successful report while marking the node as errored; users should read the queue error for the failed attempt. The test did not run real diffusion, verify provenance against source images or write a model file. Python development-node tests: **20 passed**; core suite: **66 passed, 1 opt-in CUDA test skipped**; JS widget tests: **5 passed**. No commit or push was made.

## QwenLook LoRA and HaDeS MOSS v1.0 Turbo (2026-11-13)

**QwenLook.** A text-to-image LoRA trained with `tools/moss_edit_train.py --t2i` on 1833 images that Qwen-Image-Edit 2509 (Lightning 8-step LoRA) drew from MOSS-Image SFT scenes, each captioned with its scene prompt plus the added object; rank 32, 1000 steps at 1024, one LoRA on SFT and one on Base. Runs: `<archive>/moss_edit_runs/qwenlook_{sft,base}1024`.

**Distance to no LoRA** (6 unseen prompts x 3 seeds, 30 steps, CFG 4, 1024; pixel L1 x 100 at 64 px; `<archive>/moss_eval/qwenlook_cmp`): SFT 6.3 at strength 0.5 / 10 at 1.0; Base 14.8 / 16.9. Crude 64 px variety across seeds: SFT 15.7 → 14.7, Base 17.8 → 18.2. The LoRA teaches the SFT look, so it moves Base much more than SFT.

**Base vs SFT with Turbo v1.0, with and without QwenLook** (10 varied prompts, 1 seed, 8 steps, CFG 1 and 1.5, 1344 x 1792; `<archive>/moss_eval/turbo_trio`). Pixel L1 x 100, CFG 1:

| Pair | 64 px | 16 px |
|---|---:|---:|
| Base vs SFT, Turbo only | 14.05 | 11.30 |
| Base vs SFT, Turbo + QwenLook | 12.73 | 10.17 |
| Base: QwenLook on / off | 14.52 | 11.79 |
| SFT: QwenLook on / off | 10.04 | 7.55 |

QwenLook brings Base and SFT only about 10 % closer, prompt-dependent (watercolor 12 → 7, render 17 → 11; macro 9 → 12). CFG 1.5 gives the same pattern. An eye reading of the sheets had overstated this as "nearly the same".

**QwenLook on / off on Base + Turbo v1.0** (16 photograph + 4 illustration prompts, 1 seed, CFG 1, 1344 x 1792; `<archive>/moss_eval/ql_onoff`):

| | Without | With |
|---|---:|---:|
| Laplacian variance, photos (sharper in 16 / 16) | 39.1 | 77.1 |
| High-frequency share (> 1/4 Nyquist), photos | 1.88 % | 3.71 % |
| Laplacian variance, illustrations (4 / 4) | 113.6 | 271.2 |
| Qwen3.5-9B "real photograph?" yes, photos | 15 / 16 | 2 / 16 |
| Qwen3.5-9B "real photograph?" yes, illustrations | 0 / 4 | 0 / 4 |
| Qwen3.5-9B detail score 1-10, photos | 8.31 | 8.75 |

The VLM judge is not calibrated on real photographs. Maintainer's blind pairs (criterion: which looks more realistic, any visible error disqualifies): without 12, with 8 (photos 10 / 6). The maintainer reports that several votes went against QwenLook only for small errors in that seed and finds QwenLook more realistic overall; by eye the with-LoRA images are crisper, more saturated and more polished (fewer photographic choices such as motion blur).

**Baked v1.0** (Base + Turbo v1.0 1.0 + QwenLook Base 1.0, MOSS Model Saver, BF16, stochastic). Against the live LoRAs on the 20 prompts above: same compositions and sharpness (Laplacian 77.7 vs 77.1, HF 3.73 vs 3.71 %; distance to without-QwenLook 14.68 vs 14.33), small details differ (L1 4.7 at 64 px, max 7.4; a pistol in one hand gone). Renders are deterministic (a live re-render differs by 0.0), so the difference comes from the bake's rounding at 8 steps. The released file (`HaDeS_MOSS_v1.0_Turbo.safetensors`, header title and recipe rewritten, tensors unchanged) re-renders the release candidate pixel-identically.

**Blind pairs, v0.9.4 Turbo vs v1.0** (20 new prompts, 1 seed, 8 steps, CFG 1, 1344 x 1792, left / right at random; `<archive>/moss_eval/blind_rc1`): **maintainer v1.0 16 / v0.9.4 4**. The assistant's vote is under AI-model reviews.

**Café bias, v1.0** ("photograph of a woman sitting in a café", 16 seeds, 8 steps, CFG 1, 1152 x 1536; `<archive>/moss_eval/cafe_rc1`): about 1 East Asian face in 16 by eye (v0.9.4: 9-11). The 16 women look much alike: young, long wavy brown hair, smiling at the camera.

**Juggernaut Moss v1** (Kandoo/Juggernaut-Moss, `juggernaut_moss_v1_bf16.safetensors`, SHA-256 `99f2920a…` matching the author's repo; a full finetune of Base by its weights: norms identical to Base, linears about 0.4 % from Base). One seed per prompt, by eye: with Turbo v1.0 at 8 steps (10 prompts) and at the author's settings against Base + QwenLook (30 steps, CFG 3, SLG block 22, 5 photo prompts), Juggernaut gives more candid, photographic framing; Base + QwenLook follows the prompt more closely (the nurse prompt came out black and white, seen from behind, in both Juggernaut runs). A rank-256 LoRA of Juggernaut minus Base keeps 25.6 % of the SVD energy and closes about a third of the pixel gap (Base vs Juggernaut 10.94 → 7.19 at strength 1, 5 prompts, with Turbo v1.0).

## Prompt Enhancer `detailed` mode and random prompts (2026-11-13)

**detailed vs caption** (6 short ideas, enhancer seed 0, temperature 0.7, rendered with v1.0 rc1 at 8 steps, CFG 1, image seed 4242; `<archive>/moss_eval/enhancer_detailed`): caption 21-86 words, detailed 164-226. The first version turned "anime girl on a rooftop at sunset" into an editorial photograph with a 35 mm lens; after the medium rule, 6 / 6 ideas kept their medium in the first words (anime x 3 seeds, manga, watercolor; "a lighthouse in a storm", with no medium, became an oil painting). Not compared blind; the maintainer's impression is that detailed prompts improve the images a lot.

**Random prompts** (detailed mode, empty idea, seeds 0-19, rendered with rc1; `<archive>/moss_eval/enhancer_random`): 17 different media out of 26 in 20 draws, prompts 166-243 words, varied people and places. Places repeated by chance in these seeds (Mongolian steppe 5, Lagos 3); over 2000 seeds each place is drawn 78-125 times. Golden-hour light dominates many images, a model preference not set by the draw.

**Every mode on one short idea** (2026-11-20, for the third video tutorial). Idea `an old fisherman on a dock`; MOSS-Image Base, Qwen3.5 9B int8, FLUX.2 VAE, Turbo 8-steps v1.0 (8 steps, CFG 1, euler / simple), 1024 x 1024, image seed 1013; enhancer temperature 0.7, seeds 0-5 unless said; clean ComfyUI on port 8189 with `--use-sage-attention`, PyTorch 2.9.1+cu130, RTX 5090, models warm; `I:/HaDeS_MOSS/videos/prompt_enhancer/measure` (results.json, sheets). By eye, one image seed:

- **caption:** 56-118 words (6 seeds), all photographs; the enhancer added headwear (cap or straw hat) to 5 / 6 texts and 5 / 6 images.
- **detailed:** 173-238 words (9 seeds). With no medium in the idea it chose an oil painting in 8 / 9 seeds (one black-and-white photograph), and the images were paintings. With `a photograph of an old fisherman on a dock`, 6 / 6 texts began as a documentary photograph (35 mm lens, f/2.8) and 6 / 6 images were photographs, 168-189 words. One of the 9 texts changed the subject's age ("a young man in his late twenties" for "an old fisherman").
- **Random prompts** (detailed, empty idea): 6 seeds, 5 different media (ukiyo-e print, food photograph, studio portrait photograph, cinematic still x 2, 3D animated still), 196-236 words.
- **Image input** (the first video's fisherman, idea empty, caption): 64-69 words, 4 / 4 texts and images kept the white beard, brown wool sweater, white net and the birds (seagulls in 3 texts, "birds" in one). With the idea `the same man in winter, snow on the dock`: 4 / 4 texts and images added snow and kept the man.
- **Tags** (`1boy, old man, white beard, fishing net, pier, sunrise`, caption, 4 seeds): prose of 18-61 words, an adult man in all four; seeds 0 and 1 gave the same 18-word sentence.
- **Negative `hat`** (caption, the same 6 seeds): every text stated the absence instead of leaving it out (`wearing no hat`, `his head bare without a hat`, `No hat is present on his head`), and 0 / 6 images showed headwear, against 5 / 6 without the negative. At CFG 1 the sampler does not use the negative conditioning, so this is the written prompt's doing. One idea, one image seed.
- **Time:** the answer alone took 1.6 s for caption and 2.2-2.5 s for detailed (3 seeds each, no image rendered); a full run with the 8-step image took 4.1-5.7 s.

**Corrections** (2026-11-21, the same setup). The detailed oil-painting prompt of enhancer seed 2 (189 words) pasted back as the idea, detailed mode, enhancer seeds 0-5, with a note after a blank line; texts checked by word, images by eye:

| Note after the pasted prompt | Text applies it | Image shows it | The rest of the scene |
|---|---|---|---|
| none (control) | — | — | 5 / 6 images nearly identical to the original; 1 text made the man "a young adult figure" |
| `Corrections: the fisherman is a young man.` | 6 / 6 ("a young man"), but 5 / 6 kept "his white beard" | about 3 / 6 younger | kept |
| `Correcciones: el pescador es un hombre joven.` (Spanish) | 6 / 6, 4 / 6 kept "his white beard" | about 4 / 6 younger | kept |
| `Corrections: the fisherman is a young man with a short dark beard.` | 6 / 6 (white beard left in 2) | 6 / 6 a younger man with a dark beard | kept |
| `Corrections: the fisherman wears a hat and there is a cat in the image.` | 6 / 6 hat and cat (212-248 words) | 6 / 6 hat and cat | kept |

- The note was never copied into the prompt (no "Correction" in any of the 30 texts); the Spanish note came out in English.
- A correction that contradicts other words of the pasted prompt can leave them in ("young man" with "his white beard"); naming the conflicting detail too fixed it here. One prompt, one image seed.

## HaDeS MOSS v1.0 Turbo int8 ConvRot (2026-11-14)

`HaDeS_MOSS_v1.0_Turbo.safetensors` saved by MOSS Model Saver as int8 ConvRot (Comfy-Org layout): 5.41 GB, SHA256 `e0707a51…ca7344`, the same keys, dtypes and shapes as the v0.9.4 int8 file. Against the BF16 file, 10 prompts of `prompts_40_variados.txt` (0, 6, 10, 14, 20, 25, 27, 30, 34, 35), stock nodes, seed 1001, 1344 x 1792, 8 steps, CFG 1, euler / simple, official negative (`<archive>/moss_eval/v10_int8`):

| Measure | Value |
|---|---|
| Mean absolute difference, 64 px thumbnail | 2.87 % (1.30-4.17) |
| Same, 16 px (composition) | 1.83 % (1.03-2.60) |
| Same, full size | 12.5 / 255 |
| Laplacian variance, BF16 / int8 (mean) | 918 / 862 |
| int8 re-render vs first render | 0 (3 images) |

- Same compositions in 10 / 10; small details change (a staff appears, a figure moves, invented beetle labels differ). "OPEN" and "VISIT THE FJORDS" right in both.
- Time per image in this session (whole graph, models cached, the two files alternated twice on 3 prompts): int8 about 3.4 s, BF16 about 6.6 s. This is the opposite of the October measurement above (int8 slower); not investigated, so not claimed on the model card.

## MOSS Text Check: reader on drawn text (2026-11-17)

75 images (backgrounds from the user's own Forge images, about 1 MP), one layout text box each, text drawn with PIL so the right answer is known: 8 Windows fonts, 28 to 110 px, on a coloured plate (60 %) or straight on the picture with an outline. Read by `MOSS Text Check` with `qwen3.5_9b_int8_convrot`, max_tokens 96, through ComfyUI.

| Group | Cases | Expected verdict | Result |
|---|---|---|---|
| Text drawn as expected | 40 | exact | 40 exact |
| One letter changed in the drawn text | 20 | close | 20 close (read as drawn, never corrected) |
| Nothing drawn in the box | 15 | missing | 15 missing |

1.2 s per case on an RTX 5090. The first run gave the 20 typo cases "outside its box" through a rule bug (a rounded similarity compared with an unrounded one); the verdicts above are recomputed from the same readings with the fixed rule, which is the one in the node and has a regression test.

Not measured in this set: sizes under 28 px (one case), non-Latin scripts, the w4a8 and hybrid text encoders. A drawn font is easier to read than generated lettering, so these numbers are an upper bound for generated images.

### Text painted by MOSS-Image (2026-11-17)

20 prompts asking for a given text (shop signs, posters, neon, a book cover, a t-shirt, a bus display, a cake, a wine label...), Base + Turbo 8-steps v1.0, 8 steps, CFG 1, euler / simple, 1024², seeds 2000-2019; `expected_text` mode (whole image). What MOSS really painted was then read by eye (the assistant) and compared with the node's verdict.

- MOSS painted the asked text in 18 of 20 images; one book title came out as "THE SILENT GAADEN", and one wine label cut "Rouge" at the image edge.
- **19 of 20 verdicts right**: the typo was caught ("close", read as painted).
- The wrong verdict: the reader completed the cut "Roug(e)" and the node said "exact".
- The first run gave three more wrong verdicts, all from the node's own matching. A text painted over three lines (GOOD / VIBES / ONLY, Happy / Birthday / Ana, Summer Festival / 2026) was only tried over two joined lines. The node now joins up to four consecutive read lines, and the 19/20 above is recomputed from the same readings with that rule (regression test `test_text_split_over_three_lines_is_exact`).
- 3.9 s per image including the 8-step generation.

Not measured: sizes under 28 px (one drawn case), non-Latin scripts, layout boxes on generated images, the w4a8 and hybrid text encoders. A text cut by the image edge can be completed by the reader.

## LoRA atlas: block recipes on three LoRAs (2026-11-17)

`tools/moss_lora_atlas.py`, three LoRAs trained on Base. Base template: 25 steps, CFG 2.5, er_sde / simple, 1024².

**Calibration.** On 3 prompts x 2 seeds: no LoRA, the full LoRA, each 6-layer group alone, and each group left out.

**Recipes.** Built from the calibration numbers only:
- **full**: every group at 1;
- **style**: the two groups with the most style over layout;
- **keep_comp**: without the group whose removal takes the most layout change;
- **half**: every group at 0.5.

**Validation.** On 3 other prompts x 2 seeds. Each recipe is scored by the share of the full LoRA's change it keeps in style (texture, color, edges) and in layout (32 px luminance thumbnail).

| LoRA | Recipe | Style kept | Layout kept | Ratio |
|---|---|---|---|---|
| Forge alpha s1500 (anime style) | without 30-35 | 0.87 | 0.94 | 0.93 |
| | without 24-29 | 0.76 | 0.79 | 0.97 |
| | half | 0.78 | 0.87 | 0.90 |
| Manara v02 s1400 (comic style, private) | without 24-29 | 0.82 | 0.72 | **1.14** |
| | style (12-17 + 24-29) | 0.73 | 0.71 | 1.03 |
| | half | 0.55 | 0.65 | 0.85 |
| User identity s600 | without 24-29 | 0.95 | 0.83 | 1.14 |
| | without 30-35 | 1.06 | 0.91 | 1.16 |
| | half | 0.87 | 0.73 | 1.19 |

**Removing layers 24-29 took away the most layout change** in Forge (0.21, with 30-35 at 0.22) and in Manara (0.21).

**Blind human checks (the maintainer):**
- **Manara**, 6 comic sheets of none / full / without 24-29 / half. Picked as "most Manara", and the same as "like most": full 3, without 24-29 2, half 1, none 0.
- **Identity**, 12 sheets (the 6 identity prompts x 2 seeds), resemblance scored 0-3, with the ArcFace similarity to the centroid of the 40 training photos (InsightFace buffalo_l, used locally for measurement only):

| Recipe | Eye, mean 0-3 | Panels ≥ 2 | ArcFace |
|---|---|---|---|
| full | 1.75 | 7/12 | 0.55 |
| without 24-29 | 0.25 | 1/12 | 0.14 |
| without 30-35 | 0.08 | 0/12 | 0.38 |
| half | 0.08 | 0/12 | 0.31 |
| no LoRA | — | — | 0.01 |

ArcFace and the eye correlate at 0.57 over 48 panels and agree on the winner. The eye says "it's me" only from about 0.5.

**Conclusion.** Layers 24-29 carry much of what a LoRA adds: layout for the two style LoRAs, the face for the identity one. No block recipe gave the user something that a strength change or the full LoRA does not:
- with a style LoRA, the user wants the whole style;
- with an identity LoRA, any cut loses the face.

The atlas stays a research tool in `tools/`, not a Suite node. The metrics are low-level statistics, and the numbers hold for these LoRAs, prompts and seeds only.

## MOSS Text Check and MOSS Compare Sampler: three-view frontend acceptance (2026-11-18)

Both 0.8.1 nodes were run with real models in the maintainer's ComfyUI **0.43.0 / frontend 1.57.4** (RTX 5090), from a separate workflow tab: MOSS Loader (`mossImage_base_bf16`, `qwen3.5_9b_int8_convrot`, `flux2-vae`) as model A, a merge loaded with the stock `Load Diffusion Model` as model B, MOSS Turbo LoRA (Turbo 8-steps v1.0 at 0.5, 15 steps, CFG 2.5), a 768 x 768 latent, the prompt asking for a sign that reads "HaDeS", Text Check on the sampler's `images` with `expected_text` HaDeS, and Suite Telemetry fed by both `info` outputs. The settings were restored to classic canvas / HaDeS widgets afterwards; sheets are in `output/HaDeS/acceptance_0.8.1/`.

| View | Actual check | Result |
|---|---|---|
| Classic canvas, HaDeS widgets | Queue from the frontend; sheet saved; KSampler widgets dimmed under the turbo; Telemetry read both nodes | Passed; Text Check `1/1 exact` |
| Classic canvas, standard ComfyUI widgets | All widgets of both nodes shown; changed seed and `label_B`; queue | Passed; `exact`; `label_B` on the sheet |
| Nodes 2.0 / Vue | Changed seed and `blind`; queue; workflow serialized and reloaded | Passed; `exact`; panels numbered 1 and 2 with the key in `info` only; Turbo, Compare Sampler and Text Check values and the 13 links survived the reload |

One defect was found and fixed: a branch without a model name (the stock loader carries no recipe) let the other branch's model go in the sheet footer as if every panel used it, so the merge read as `mossImage_base_bf16`, and in blind mode the footer named a model. The Compare Sheet now puts the model in the footer only when every branch names the same one, and the Compare Sampler labels a model without a recipe `unknown model`. After a restart the same graph gave `A: mossImage_base_bf16 + Turbo x0.5`, `B: unknown model + Turbo x0.5` and a footer without a model. Python suite: **99 tests passed, 1 opt-in CUDA test skipped** (one new test for this case); JS widget tests: **5 passed**. One prompt and one seed per view: this checks the UI and the wiring, not image quality or reading accuracy (see the Text Check sections above).

## Model Saver nearest and live LoRAs (2026-11-18)

Bake: `preview_test` (SFT + 30% Base in layers 24-29, bfloat16) + Turbo 8-steps v1.0 x0.5 + forge_mega_s5250 x0.8, compared with the same
LoRAs loaded live in the running server (15 steps, CFG 2.5, euler / simple, 896x1152, two prompts, seed 5102). RTX 5090, ComfyUI
`lora_compute_dtype` float16.

| Weights in the file | Mean pixel difference to the live render (0-255), two prompts |
|---|---|
| Saver 0.8.1 `nearest` run inside the server (core lazy saving) | 37.8 / 33.7 |
| float32 sum of the patches, rounded once | 39.6 / 32.9 |
| weight cast to float16, patches applied, rounded once (ComfyUI's whole-in-VRAM load) | 40.3 / 33.8 |
| patches computed in bfloat16 (ComfyUI's on-the-fly path) | 17.1 / 15.2 |

- With Turbo at 15 steps the sampler is chaotic: two files that differ only by rounding render about 21/255 apart, so only bit-identical
  weights reproduce a live render pixel for pixel. A repeated live render gave 0.00.
- Run alone (`load_models_gpu`, model whole in VRAM), both the 0.8.1 Saver and the fixed one write exactly the float16-path weights:
  0 differing elements in 12 of the 168 patched tensors, against `ModelPatcher.patch_weight_to_device`. Inside the server the 0.8.1
  Saver's file differed from that (6.5% of the patched elements differed from the float32 sum): the core's lazy saving follows how much
  of the model is in VRAM. The fixed Saver computes the float16 path itself, so the file is the same either way.
- The server's live renders were closest to the bfloat16 on-the-fly path: with the 9.75 GB text encoder also loaded, part of the model
  seems to be patched on the fly, so which weights a live render uses depends on the memory state. Not measured further.
- Test: `test_saver_nearest_matches_live_lora_weights` (two stacked LoRAs, float16 LoRA compute forced) checks the file against
  `ModelPatcher.patch_weight_to_device`. The 0.8.1 Saver passes it too (on the CPU nothing is patched on the fly); it guards the new path.

## Layout mode: Prompt Enhancer boxes and MOSS placement (2026-11-19)

Eight ideas written in Spanish, the way the maintainer writes them, each stating where 3 or 4 elements go ("un dragón enorme arriba a la derecha", "la firma \"HaDeS\" abajo a la izquierda"): 28 stated elements, 7 of them with two directions (e.g. top right). `MOSS Prompt Enhancer` in `layout` mode wrote the boxes (`qwen3.5_9b_int8_convrot`, 1024 x 1536 canvas, temperature 0.7, seed 1); the draft was compiled by `MOSS Layout Compiler` and sampled with Bulle d'Encre (`HaDeS_MOSS_preview_beta`, seed 1200, 15 steps, CFG 2.5, euler / sgm_uniform), the boxes not edited.

| Check | Result |
|---|---|
| Box centre in the stated region (left / right < 0.45 / > 0.55 of the width, top / bottom of the height, centre 0.3-0.7), by script | **28 / 28** (the script first counted 27: its keyword for the book matched the cat's "bookshelf"; the book box is in the lower centre) |
| Element drawn inside its box, by eye on the overlay sheet | **26 / 28**; both misses in one image: the astronaut (a 64 px wide, 500 px tall box) drawn lower, the rover (a box above the ground line) drawn lower |
| Texts asked for (CAFÉ, MAR DEL NORTE, HaDeS, LA ÚLTIMA NOCHE, PRÓXIMAMENTE) | the boxes held the exact words; MOSS rendered all, with one extra line "LA" above LA ÚLTIMA NOCHE |
| Items dropped by the fitting (no usable box) | 0 |

The enhancer also adds elements the idea did not give: in a first run through the Layout Editor's DRAFT LAYOUT button, a signature box with invented text ("J. M. 2023") when the idea asked for "una firma" without saying what it reads. One seed per idea and one enhancer seed: individual outcomes, not a rate. Files: `I:/HaDeS_MOSS/layout_test` on the maintainer's machine (drafts, images, overlay sheet).

**The same 8 drafts with HaDeS MOSS v1.0 Turbo** (2026-11-20; boxes unchanged, its card's settings: 8 steps, CFG 1, euler / simple, seed 1200, 1024 x 1536): **26 / 28** elements drawn inside their box, by eye on the overlay sheet. The misses: the astronaut (the same 64 px wide box) drawn wider than it, centred to its right, and the dog beside the fisherman, drawn left of its box. The five texts were rendered as asked, again with the extra line "LA" above LA ÚLTIMA NOCHE. One seed per draft. Files: `I:/HaDeS_MOSS/layout_test_v10`.

The two Layout Editor buttons were checked once in the classic canvas view (ComfyUI 0.43.0 / frontend 1.57.4): DRAFT LAYOUT ran only the Prompt Enhancer and the editor (history outputs `2`, `4`; partial target `4`) and imported its 5 boxes; after one box was moved, GENERATE ran the graph with the Prompt Enhancer served from the cache and the moved box unchanged, and the moon was drawn inside it. Nodes 2.0 / Vue view not checked.

## Linked width / height into the Prompt Enhancer (YouTube report, 2026-11-23)

Report: in the styled widgets a link could not be dropped on the Prompt Enhancer's `width` / `height`, and with standard widgets the size
did not reach the layout. Clean 8189, frontend 1.57.4, MOSS Latent 16:9 at 1 MP (1360x768) linked into the Enhancer
(`I:/HaDeS_MOSS/layout_test/linked_size_check.py`, real runs through the API, int8 text encoder, seed 1):

| Mode | Before (0.9.5) | After (0.9.6) |
|---|---|---|
| `layout`: draft and Layout Compiler | 1360x768 | 1360x768 |
| `caption`: draft | `{"caption"}` (no size) | `{"caption", "width": 1360, "height": 768}` |
| `caption`: Layout Compiler latent | 1024x1024 | 1360x768 |

Dropping the Latent's `width` output on the Enhancer's `width` row with the mouse:

| View | Before | After |
|---|---|---|
| Classic canvas, HaDeS widgets | connects | connects |
| Classic canvas, standard widgets | connects | connects |
| Nodes 2.0 / Vue, standard widgets | connects | not re-checked (the change only adds a marker to styled rows) |
| Nodes 2.0 / Vue, HaDeS widgets | **does not connect** | connects |

Cause in Nodes 2.0: the frontend finds the drop target from the DOM (the `.lg-node-widget` row under the pointer and the
`[data-slot-key]` inside it); a styled row has no input of its own. Each styled row now gets a hidden marker with its backing input's
slot key. Not checked: subgraphs, save / reload with the link, zoom.

## Layout Editor DRAFT LAYOUT / GENERATE: acceptance in four views (2026-11-19, after audit 0.8.1)

ComfyUI 0.43.0, frontend 1.57.4, RTX 5090. The `Layout from a Sentence` example with the stock SFT file; the buttons pressed with the mouse in the built-in browser pane, the queue state read from `/queue` and `/history`.

| View | DRAFT LAYOUT | Edit | GENERATE |
|---|---|---|---|
| Classic canvas, HaDeS widgets | With a GENERATE job already running and the enhancer seed changed, the DRAFT LAYOUT job queued behind it with its own `prompt_id`; the canvas got that job's draft (5 boxes), not the running job's older one. Pressed again with the mouse: its draft imported | a box moved | the run used the moved box; enhancer from the cache; Text Check `exact` on "HaDeS" |
| Classic canvas, standard widgets | buttons shown under the draft's JSON; the draft imported into the JSON box | the JSON's text item edited to "DONIA" | the run used "DONIA"; Text Check `exact`, *case or punctuation differ* (read "Donia") |
| Nodes 2.0 (Vue), HaDeS widgets | canvas and buttons drawn in the Vue node; draft imported | a box dragged with the mouse (x 64 → 240) | the run used the dragged box; Text Check `exact` |
| Nodes 2.0 (Vue), standard widgets | Vue buttons under the JSON; draft imported into the textarea | the textarea's text item edited to "MOSS" | the run used "MOSS"; Text Check `exact`, *case or punctuation differ* |

Saved and reloaded: the editor's `widgets_values` hold only the draft (the buttons are not saved) and the reloaded draft is the same. An editor added and removed again cleared its run state. In Vue, an empty image preview under the editor came from another open workflow tab whose node 3 was a Load Image (the frontend keeps node outputs by node id across tabs); with that tab closed it did not appear. One workflow, one run per view: this checks the UI and the queue handling, not image quality.

## Bulle d'Encre: style phrases and text-rendering factors (2026-11-19)

Bulle d'Encre (`HaDeS_MOSS_preview_beta`, BF16), 1024 x 1024, 15 steps, CFG 2.5, euler / sgm_uniform, negative `worst quality, bad anatomy, jpeg artifact, colorful`, plain CLIPTextEncode. Images and data: `I:/HaDeS_MOSS/style_text_tests` on the maintainer's machine.

**Style phrases.** Each phrase was put in front of four neutral scenes (a woman reading at a café, an old fisherman, a fox in snow, a lantern market), 2 seeds, against the same scene and seed with no phrase. *Move* is the CLIP-ViT-H image distance (1 - cosine) to the no-phrase image, in units of the distance between the two seeds of a scene with no phrase (0.115); *coherence* is the mean cosine between the 8 shift vectors (high: the phrase pushes every scene the same way, i.e. a consistent look).

| Phrase | Move (seed gaps) | Lowest move | Coherence | By eye (seed 101) |
|---|---:|---:|---:|---|
| black and white manga | 3.27 | 0.269 | 0.45 | grey-tone manga in all four |
| vintage poster | 2.94 | 0.160 | 0.24 | not a style: a framed poster or invented text ("RODOLES CIEGORE…") added in the scene |
| pencil sketch | 2.89 | 0.202 | 0.46 | graphite sketch in all four |
| 1950s comic book | 2.76 | 0.200 | 0.43 | bolder ink and period faces |
| ink line art | 2.76 | 0.211 | 0.40 | black line art on white |
| flat cel shading | 2.60 | 0.155 | 0.38 | flatter colour, cleaner shapes |
| European comic illustration | 2.54 | 0.157 | 0.43 | stronger ink outlines and saturation |
| watercolor painting | 2.37 | 0.119 | 0.35 | lighter, softer colour; not a clear watercolour look |
| anime style | 2.33 | 0.077 | 0.31 | close to the default |
| oil painting | 2.01 | 0.027 | 0.27 | close to the default |
| 3D render | 1.21 | 0.029 | 0.14 | close to the default |
| photograph | 0.65 | 0.011 | 0.07 | almost no change: the default of these neutral prompts is already a soft realistic illustration |

The phrases that change the medium (monochrome manga, pencil, ink) move furthest and most consistently; colour comic phrases (1950s comic, European comic, flat cel) move about as far in colour; anime, oil painting, 3D render and photograph barely move this model from its default. One phrase at the start of a short prompt, two seeds: a ranking for prompting, not a measure of style quality.

**Text factors.** A bakery storefront at dusk with a hanging sign, 4 seeds per condition, read by `MOSS Text Check` (expected text anywhere). Writing conditions: 3 lengths (1, 3, 6 words) x 3 ways of writing (`reads "TEXT"`, `reads TEXT`, `with the words TEXT painted on it`) x 2 cases (upper, Title Case); count conditions: 1, 2 or 3 quoted texts (sign, chalkboard, window poster).

| Factor | Exact (literal) |
|---|---|
| 1 / 3 / 6 words | 24/24, 24/24, 23/24 |
| quoted / bare / "the words" | 24/24, 23/24, 24/24 |
| UPPER / Title Case | 36/36, 35/36 (Title Case rendered as written) |
| 1 / 2 / 3 texts | 4/4, 8/8, 11/12 |

94 of 96 texts were exact and literal. The two misses: a 6-word Title Case text written bare repeated a word ("Baked Baked"), and with three texts the poster's "SALE" was missing. At this size and with up to 6 words, none of these factors made a measurable difference on Bulle d'Encre; longer texts, smaller text, more texts and other models were not tested.

## Compare Sheet peak RAM (2026-11-19, audit 0.8.1 #9)

`MOSS Compare Sheet` with 4 panels, batch 4, 1344 x 1792 random images (441 MB of inputs), saving the 4 PNGs, one fresh CPU process per version; peak working set (psutil `peak_wset`, Windows) minus the process memory once the inputs were made.

| Version | Peak added by `compare()` | Of which the `IMAGE` output | Time |
|---|---:|---:|---:|
| Before (every sheet kept, then a float copy of each, then the stack) | 1 112 MB | 475 MB | 2.7 s |
| After (each sheet built, saved and copied into the preallocated output in turn) | 555 MB | 475 MB | 2.5 s |

The output tensor and the four PNG files are bit-identical between the two versions (SHA-256). The Compare Sampler builds its sheet through the same code. One size and batch measured; the saving scales with batch size and sheet area.

## AI-model reviews (qualitative, not measurements)

Preferences given by AI models on the blind sheets of the four 2026-11-11 experiments above: GPT and Gemini (screenshots supplied by the maintainer) and the assistant that ran the experiments (sealed before each key was opened, but it had seen earlier images, so never fully blind). They are kept for the record, with their tables as decoded. They are **impressions, not measurements**: the criteria differ and are not verifiable from screenshots, AI judgments may be correlated, and the same pairs were judged repeatedly. No conclusion, default or model card in this repository relies on them.

### CFG / step screen of baked v0.9.4

An assistant review is saved separately in `assistant_ratings.csv`: one first and one second preference per sheet, recorded by A–H letter before consulting the key. It is **not strictly blind** because the assistant had seen labelled technical contact sheets beforehand. The maintainer's archived blank template remains untouched. The setting mapping was opened only after the maintainer supplied their ratings.

| Scene / seed | Maintainer favorite(s) | Assistant first | Assistant second |
|---|---|---|---|
| Airlock / original | D = 12 / 1.2 | A = 8 / 1.2 | D = 12 / 1.2 |
| Airlock / 1200 | A = 12 / 1.0 | A = 12 / 1.0 | B = 12 / 1.2 |
| Trawler portrait / original | E = 8 / 2.0 | B = 8 / 1.0 | C = 8 / 1.2 |
| Trawler portrait / 1200 | C = 8 / 1.5; G = 12 / 1.5 (tie) | F = 16 / 1.0 | E = 12 / 1.0 |
| Ceramic texture / original | F = 8 / 2.0 | G = 12 / 1.0 | H = 12 / 1.2 |
| Ceramic texture / 1200 | C = 12 / 1.2 | B = 16 / 1.0 | A = 12 / 1.5 |
| Comic text / original | C = 8 / 1.2 | D = 8 / 1.5 | H = 8 / 2.0 |
| Comic text / 1200 | F = 8 / 2.0 | D = 12 / 1.5 | F = 8 / 2.0 |

#### Third reviewer: Gemini's sheet preferences (2026-11-11)

The maintainer supplied a screenshot of Gemini's eight A–H choices, archived as `<archive>/baked_turbo_v094_2026-11-11/gemini_blind_rating.png` (SHA-256 `ea98c0b2384ae91472790fb0b40a50f454b0b3c20b107ce73e6d14a49e2f3708`). The screenshot describes a review of lighting quality, texture realism and overall sharpness. The maintainer calls it blind; the screenshot alone cannot verify what other context Gemini saw. It records one favorite per sheet, with no numerical scores or systematic prompt-accuracy / artifact fields. The reasons below are brief summaries of the screenshot, not independently verified image properties.

| Scene / seed | Gemini letter = steps / CFG | Stated reason (summary) | Same image as maintainer favorite? | In assistant top two? |
|---|---|---|---|---|
| Airlock / original | C = 8 / 1.5 | Volumetric light and atmosphere | No | No |
| Airlock / 1200 | C = 8 / 1.0 | Floor grilles and panel definition | No | No |
| Trawler portrait / original | C = 8 / 1.2 | Raincoat texture and water reflections | No | Yes, second |
| Trawler portrait / 1200 | E = 12 / 1.0 | Facial light and sea background | No (maintainer tied C/G) | Yes, second |
| Ceramic texture / original | C = 8 / 1.0 | Gloss and water marks on green enamel | No | No |
| Ceramic texture / 1200 | E = 12 / 1.0 | Fabric and clay nuances | No | No |
| Comic text / original | E = 8 / 1.0 | Typography composition and brick contrast | No | No |
| Comic text / 1200 | B = 12 / 1.0 | Table light and hand sharpness | No | No |

Decoded counts: **8/CFG 1: 3**, **12/CFG 1: 3**, **8/CFG 1.2: 1**, **8/CFG 1.5: 1**. Thus Gemini chose CFG 1 on **6/8 sheets** and CFG 2 on none. There is **0/8 exact image agreement** with the maintainer's favorites (including the tied sheet), **0/8 first-choice agreement** with the assistant and **2/8** Gemini choices in the assistant's top two. This is substantial reviewer disagreement under different, incompletely specified criteria; it is not a quality or reliability rate for any setting. Do not pool these choices as interchangeable votes or infer a universally preferred CFG.

Gemini's preference for CFG 1 reinforces its role as the held-out **reference**, while the maintainer's three 8/CFG 2 selections still justify testing that more saturated option as a challenger. The earlier shortlist of 8/CFG 2 and 12/CFG 1.2 remains **provisional and maintainer-led**, not a consensus across reviewers. The held-out review should require separate scores for prompt fidelity, artifacts and overall preference under the same criteria for every reviewer before deciding whether extra steps or CFG are useful. No additional generation or default change was made for this third review.

#### Fourth reviewer: separate GPT sheet preferences (2026-11-11)

The maintainer also supplied a screenshot of another GPT review, archived as `<archive>/baked_turbo_v094_2026-11-11/gpt_blind_rating.png` (SHA-256 `9348e17619efc05bbee8d92670413a84afc5656ef303c7df13d6ebaf093dab52`). This is **a separate review** from the assistant's earlier `assistant_ratings.csv`. The screenshot says its choices were made without knowing the setting-to-letter key and lists composition, lighting, visual coherence, detail and text legibility as criteria. The screenshot does not independently establish the reviewer's full context or whether every criterion was applied consistently.

| Scene / seed | Separate GPT letter = steps / CFG | Same image as maintainer favorite? | Same as Gemini? | In assistant top two? |
|---|---|---|---|---|
| Airlock / original | H = 12 / 1.5 | No | No | No |
| Airlock / 1200 | C = 8 / 1.0 | No | Yes | No |
| Trawler portrait / original | C = 8 / 1.2 | No | Yes | Yes, second |
| Trawler portrait / 1200 | C = 8 / 1.5 | Yes, tied favorite | No | No |
| Ceramic texture / original | H = 12 / 1.2 | No | No | Yes, second |
| Ceramic texture / 1200 | C = 12 / 1.2 | Yes | No | No |
| Comic text / original | D = 8 / 1.5 | No | No | Yes, first |
| Comic text / 1200 | E = 8 / 1.5 | No | No | No |

Decoded counts: **8/CFG 1.5: 3**, **12/CFG 1.2: 2**, and one each for **8/CFG 1**, **8/CFG 1.2** and **12/CFG 1.5**. Thus this reviewer chose CFG 1.2–1.5 on **7/8 sheets** and CFG 2 on none. Exact image agreement is **2/8 with the maintainer** (counting the tied favorite), **2/8 with Gemini**, and **1/8 with the earlier assistant's first choices**; three choices occur in the earlier assistant's top two. The screenshot also warns against selecting a visually striking comic image if it distorts the title text. That is a stated evaluation concern, not a verified defect count from this vote.

The four reviews favor different images and were not collected with a common scoring rubric: the maintainer chose 8/CFG 2 most often; Gemini chose CFG 1 on six sheets; this GPT favored moderate CFG; and the earlier assistant review was not fully blind. Their disagreement is evidence that the first screen cannot determine one global best setting by preference alone. Keep 8/CFG 1 as a reproducible reference and retain 8/CFG 2 and 12/CFG 1.2 only as **test candidates**, not recommended defaults. A held-out round with shared scoring criteria and explicit text / artifact checks remains the next measurement; no new generation or model change was made for this review.

### Baked v0.9.4 versus SFT/Base + Turbo v0.9

The assistant's eight first and second choices are in `assistant_blind_ratings.csv` (SHA-256 `ca03c3959839ff4b49f3bdf17caa1313441a96b152e63228c1c3eeb0339a08fe`), sealed before this key was opened; it had seen the baked images in the earlier screen. Assistant: SFT first on **7/8**, Base **1/8**, baked **0/8**.

The maintainer supplied two screenshots with one A/B/C choice per sheet, in order **GPT-text** and **Gemini**. The originals are archived as `gpt_text_blind_rating.png` (SHA-256 `c2c5115046f7b01dd745d1120b56cda383090738065e1674112c2f65176d4fac`) and `gemini_blind_rating.png` (SHA-256 `14b2793d376e802399450bc8427b29b09612015f70d0589ec1dccad3550b9a9f`) in the comparison archive. Their letters were transcribed to `additional_blind_ratings.csv` (SHA-256 `f61dc99e3dea8f6f2fbaf02c638e011d7ad5b7c8a4b4cfe77cfb37470fe6bbb5`). The screenshots describe qualitative judgments; neither provides standardized prompt-accuracy or artifact scores. The exact context supplied to either reviewer cannot be verified from the screenshots alone.

| Scene / seed | Maintainer | Assistant | GPT-text | Gemini | Sheet result |
|---|---|---|---|---|---|
| Airlock / original | SFT | SFT | baked | Base | SFT plurality 2/4 |
| Airlock / 1200 | SFT | SFT | baked | SFT | SFT 3/4 |
| Trawler portrait / original | baked | SFT | baked | baked | Baked 3/4 |
| Trawler portrait / 1200 | baked | Base | baked | baked | Baked 3/4 |
| Ceramic texture / original | baked | SFT | baked | baked | Baked 3/4 |
| Ceramic texture / 1200 | SFT | SFT | baked | SFT | SFT 3/4 |
| Comic text / original | SFT | SFT | SFT | SFT | SFT 4/4 |
| Comic text / 1200 | SFT | SFT | SFT | SFT | SFT 4/4 |

| Reviewer | Baked v0.9.4 | SFT + v0.9 | Base + v0.9 |
|---|---:|---:|---:|
| Maintainer | 3 | 5 | 0 |
| Assistant | 0 | 7 | 1 |
| GPT-text | 6 | 2 | 0 |
| Gemini | 3 | 4 | 1 |
| **Descriptive total** | **12** | **18** | **2** |

The maintainer and Gemini agree on **7/8 exact images**; GPT-text and the maintainer agree on **5/8**. Seven sheets have at least a 3/4 majority; the original-seed airlock has only a 2/4 plurality. Both comic sheets are unanimous for SFT + v0.9. The baked model wins 3/4 votes on both trawler sheets and the original-seed ceramic sheet. These are preferences among the **same eight triplets**, not 32 independent image tests; the assistant also had previous exposure to baked images. The reviewers used different qualitative criteria, and the screenshots cannot establish consistent blinding or scorer calibration. The added votes strengthen the scene-dependent observation, **not a universal quality ranking**. The conclusion remains limited to four selected prompts, two seeds each, 1152×1536, 8 steps and CFG 1. No additional rendering or model/default change was made for this tally.

### Held-out eight scenes, baked v0.9.4 versus SFT + Turbo v0.9

The assistant reviewed the eight blind sheets and full-resolution person images, then saved one overall preference per scene and a **separate face-texture-detail preference** for the four person scenes in `assistant_blind_ratings.csv` (SHA-256 `8e71057497d749f52424cbfe04bfa52846d8b491f4d5ad787feb08bd8019a0fb`). This file was sealed **before opening the A/B key**. The assistant had seen baked-model images from earlier experiments, so the review is not perfectly blind. The maintainer subsequently supplied eight independent A/B choices, after which the key was opened. The decoded results are below; additional reviewers should avoid the key, variant-named PNGs, `metrics.csv` and the decoded section until they have voted.

| Scene | Maintainer overall | Assistant overall | Assistant face-texture detail |
|---|---|---|---|
| Watchmaker | B = baked | A = SFT | B = baked |
| Trumpeter | B = baked | A = SFT | B = baked |
| Climber | B = baked | A = SFT | B = baked |
| Flower seller | B = SFT | B = SFT | A = baked |
| Pocket-watch macro | A = SFT | A = SFT | — |
| Rainy alley | B = baked | A = SFT | — |
| Citrus still life | B = SFT | B = SFT | — |
| Lunar observatory | A = baked | B = SFT | — |

The assistant prefers SFT + v0.9 on **8/8** overall but marks baked v0.9.4 as having **more visible facial texture on 4/4 person scenes**: texture and whole-image preference can point different ways.

The maintainer supplied separate screenshots of **GPT, then Gemini**, each with one A/B choice and qualitative explanation per scene. The originals are archived in this experiment as `gpt_blind_rating.png` (SHA-256 `0ad2e421816d049d7f284a2026aca86478d9b11297191bad61f75d724bfb44c9`) and `gemini_blind_rating.png` (SHA-256 `e938968c50dc7774c1ac15d13f803b22fd56f642858e6c3922b18f65ee33fc2b`). Their letters were transcribed to `additional_blind_ratings.csv` (SHA-256 `12633065a14b969ab9e2cbd54db9741b5e422941536189bdcb9329ae94c10191`). The screenshots state or imply blind selection; they do not establish each reviewer's entire context, nor a shared numerical rubric.

| Scene | Maintainer | Assistant overall | GPT | Gemini | Per-sheet result |
|---|---|---|---|---|---|
| Watchmaker | Baked | SFT | Baked | Baked | Baked 3/4 |
| Trumpeter | Baked | SFT | SFT | SFT | SFT 3/4 |
| Climber | Baked | SFT | Baked | Baked | Baked 3/4 |
| Flower seller | SFT | SFT | Baked | Baked | Tie 2/2 |
| Pocket-watch macro | SFT | SFT | SFT | SFT | SFT 4/4 |
| Rainy alley | Baked | SFT | Baked | Baked | Baked 3/4 |
| Citrus still life | SFT | SFT | SFT | Baked | SFT 3/4 |
| Lunar observatory | Baked | SFT | Baked | SFT | Tie 2/2 |

| Reviewer | Baked v0.9.4 | SFT + v0.9 | Baked person scenes | Baked nonperson scenes |
|---|---:|---:|---:|---:|
| Maintainer | 5/8 | 3/8 | 3/4 | 2/4 |
| Assistant overall | 0/8 | 8/8 | 0/4 | 0/4 |
| GPT | 5/8 | 3/8 | 3/4 | 2/4 |
| Gemini | 5/8 | 3/8 | 3/4 | 2/4 |

GPT and Gemini agree on **6/8 exact images** and on **all four person scenes**. In their descriptions, both favor the baked watchmaker and climber for visible face/gear texture, but the SFT trumpeter for more coherent pose and instrument. They also favor the baked flower seller, while the maintainer and assistant prefer SFT overall there. This fits the assistant's earlier **4/4 baked face-texture** assessment while showing that texture and whole-image quality can lead to different votes. The pocket-watch macro is the only unanimous sheet; flower seller and observatory are tied. Across the three newly supplied reviewers, baked receives **15/24 preferences**; adding the assistant's eight SFT preferences yields **15 baked versus 17 SFT**. These totals are descriptive only: the same eight image pairs were judged repeatedly, reviewers' criteria differ, and AI judgments may be correlated. The three supplied reviews each prefer baked on **3/4 people** versus **2/4 nonperson** scenes, which strengthens the *scene-dependent preference hypothesis* but does not establish a population-level portrait advantage or prove physically faithful facial detail. No new rendering or default change was made for these added votes.

### Baked v0.9.4 versus SFT + Turbo v0.7

The assistant recorded one overall preference per scene and a separate face-texture-detail choice for each person scene in `assistant_partial_blind_ratings.csv` (SHA-256 `646c8b586f42902075b30a1f3f6dd1ee03c42523c1bd16e534ae1f6bfe6476ab`) before opening the new key. The assistant also recognizes images from the prior round. A fresh-scene validation would be needed if this quick screen suggests a surprising result; the older LoRA and familiar images limit what one vote can establish. No commit or push was made.

GPT and Gemini screenshots were archived as `gpt_blind_rating.png` (SHA-256 `442c1a70ebd2c5fa1fe96754505063dea4067817ca1cd39d52982f95872b0c2f`) and `gemini_blind_rating.png` (SHA-256 `ee8fcc51be099c33c9dcb85c45f822dc794358cdb3e03c17ead0d2795135a45`); their letters are in `additional_blind_ratings.csv` (SHA-256 `76e42f7cffef084f77416c50231aa008ba4477d0cd519ef472c34b29606aa941`). The screenshots provide qualitative reasons but cannot establish whether the reviewers recognized images from the earlier round.

| Scene | A / B mapping | Maintainer | GPT | Gemini | Assistant overall | Assistant face texture |
|---|---|---|---|---|---|---|
| Watchmaker | SFT v0.7 / baked | Baked | SFT v0.7 | Baked | SFT v0.7 | Baked |
| Trumpeter | SFT v0.7 / baked | Baked | SFT v0.7 | SFT v0.7 | SFT v0.7 | Baked |
| Climber | Baked / SFT v0.7 | Baked | Baked | Baked | SFT v0.7 | Baked |
| Flower seller | Baked / SFT v0.7 | SFT v0.7 | Baked | Baked | SFT v0.7 | Baked |
| Pocket-watch macro | Baked / SFT v0.7 | SFT v0.7 | SFT v0.7 | SFT v0.7 | SFT v0.7 | — |
| Rainy alley | SFT v0.7 / baked | Baked | Baked | Baked | Baked | — |
| Citrus still life | Baked / SFT v0.7 | Baked | Baked | Baked | SFT v0.7 | — |
| Lunar observatory | SFT v0.7 / baked | Baked | Baked | SFT v0.7 | SFT v0.7 | — |

| Reviewer | Baked v0.9.4 first | SFT + v0.7 first |
|---|---:|---:|
| Maintainer | **6/8** | 2/8 |
| GPT | **5/8** | 3/8 |
| Gemini | **5/8** | 3/8 |
| Assistant overall | 1/8 | **7/8** |

Rainy alley is unanimous for baked v0.9.4; pocket-watch macro is unanimous for SFT + v0.7. Citrus and climber give baked a 3/4 majority, and trumpeter gives SFT a 3/4 majority. Watchmaker, flower seller and observatory split 2/2. Three reviewers individually favor baked on these familiar pairs, while the assistant favors SFT; **17 versus 15** total votes across all four reviewers would be a misleading headline because the same eight pairs are repeatedly judged under different criteria. In the immediately preceding v0.9 comparison on these same prompts, maintainer/GPT/Gemini each selected baked 5/8; replacing v0.9 with v0.7 did not produce a decisive separation in their short preference tallies. It is not a direct blind head-to-head between the two LoRAs.

### v0.9.4 Turbo vs v1.0 (2026-11-13)

The assistant voted on the 20 blind pairs and saved its verdicts before opening the key: v1.0 11, v0.9.4 5, tie 4. It agreed with the maintainer on 10 of 20 pairs. Not counted in the model card.

## Experimental quantization — MOSS Base repeatability pilot (2026-11-21)

RTX 5090, driver 621.01, PyTorch 2.9.1+cu130, SDPA, ComfyUI 0.43.0
(426f5eea52a7ece1dd599fff4cb6cd1f352ddd24), Suite worktree exp/quant based on
6c8d227ebf9c56854b77a29ade8a2269534969ad. Lab root: I:/HaDeS_MOSS/quant_lab.
Eight local checkpoint paths, exact lengths and SHA-256 digests are recorded in
phase0/model_inventory.json. The Base BF16 file is 10,164,655,808 bytes, the hybrid
encoder 6,724,098,468 bytes and the Flux2 VAE 336,213,556 bytes.

tools/moss_quant_repeat.py called stock nodes directly in a separate process after
the isolated HTTP startup attempt on port 8190 timed out. Graph and evidence:
phase0/base_offline_01/{graph.json,measurement.json,repeat_0.png,repeat_1.png}.
Use the ComfyUI venv Python with -B and arguments --core H:/Comfy_UI_V61/ComfyUI
--prompts I:/HaDeS_MOSS/prompts/prompts_40_variados.txt --out <new-run-directory>.
Prompt file SHA-256: 242fd546d84709a92cdc8a5d9fa2f6b42f20d5ecf68f12c27f6a53d6e1d32f8e.
Graph file SHA-256: 339fd6aa34e956fca6a16e0c768935c7c507a30f38d21f157ad955f4788544a4.

First prompt (Sardinian shepherd), seed 1001, 1024 square, 25 steps, CFG 2.5,
er_sde/simple, default model shift, SLG single block 22 over 0–50%, Suite evaluation
negative. Both calls executed KSampler and VAEDecode, reusing the loaded objects,
conditioning and initial empty latent. Setup with conditioning took 33.12 s.
Sampling, decoding and conversion to uint8 RGB took 17.19 s then 14.09 s.
Peak PyTorch allocated CUDA memory was 17,813,873,664 bytes in each run (allocator
scope). RGB SHA-256 b2716b6599c430534107302f980d9239487bcfa955f8f5ee3935b170fc16fcdd
in both images. Changed pixels: 0 of 1,048,576. This establishes within-process
repeatability for one prompt and seed. Cold restarts and other prompts remain untested.

Historical cross-check: the ten BF16/int8 PNG pairs in I:/HaDeS_MOSS/moss_eval/v10_int8
yield mean full-size RGB MAE 12.545/255 (pair range 7.052–26.126), recomputed as the
arithmetic mean of per-pair absolute RGB differences. Those files use v1.0 Turbo at
1344 x 1792, 8 steps, CFG 1, euler/simple. The historical 2.87% at 64 px belongs
to this Turbo test. The precise historical thumbnail resampling procedure remains
unconfirmed. The four planned native Base candidates still require verified
checkpoints and measurements before setting quality gates.

## Native NVFP4 block projections on MOSS Base — 2026-11-21

RTX 5090, driver 621.01, PyTorch 2.9.1+cu130, ComfyUI 0.43.0 and SDPA, same core and Suite revisions as the Base pilot. All paths below are relative to I:/HaDeS_MOSS/quant_lab. This is a native baseline for evaluating the experimental representation.

Conversion: tools/moss_quant_nvfp4.py, source phase0/model_inventory.json entry mossImage_base_bf16.safetensors. Default Comfy NVFP4 quantizer, nearest rounding, per-matrix global and per-16 block scales, dynamic input scales. Exported qkv, o, ffn_up_gate and ffn_down in all 42 blocks (168 matrices). The 211 other tensors retained their dtype, shape and raw contents, verified against the source. Target: phase1/models/moss_base_nvfp4_blocks.safetensors, SHA256 d59ce5a741fc8642509ae3610b007b834d94b747e5538187ca8a45159c8c4fb9. The 3,327,469,496 bytes include 2,378,170,368 weight codes, 297,271,968 scale bytes, 651,935,808 literal bytes, 3,024 format-marker bytes and an 88,328-byte header. See phase1/size_audit.json.

tools/moss_quant_grid.py rendered each checkpoint in its own fresh process with the stock UNETLoader and KSampler. Positive prompts 0,6,10,14,20,25,27,30,34,35 from the pilot prompt file, one seed (1001), 1024 square, 25 steps, CFG 2.5, er_sde/simple, Base SLG block 22 through 50%, hybrid Qwen and Flux2 VAE. Only the UNet filename differs within a pair. BF16 prompt 0 also matched the pilot's RGB hash across process restarts. Source graphs, PNGs, RGB hashes and run records: phase1/bf16_01 and phase1/nvfp4_01.

| Measure | Base BF16 | NVFP4 |
|---|---:|---:|
| Complete checkpoint bytes | 10,164,655,808 | 3,327,469,496 |
| First sampling + decode + finite check + RGB conversion, seconds | 16.069 | 9.842 |
| Same, median of next nine, seconds | 14.657 | 7.665 |
| Peak CUDA bytes allocated by PyTorch during measured runs | 17,813,873,664 | 11,039,570,944 |
| Completed comfy_kitchen.scaled_mm_nvfp4 calls across ten samplings | 0 | 83,480 |
| UNETLoader return in separate fresh process, seconds (one trial) | 2.738 | 0.092 |

Sampling timings exclude prompt encoding, model setup and image writes. The loader-only times exclude subsequent materialization/device upload, use different storage volumes, and the OS file cache was uncontrolled. They do not establish a cold-storage loading speedup. Operator instrumentation counted successful returns on the stock path; low-level kernel identities were not profiled.

tools/moss_quant_compare.py: RGB8 bicubic square thumbnails with Pillow, L1 expressed as percentage of 255. At 64 px mean 9.103%, median 8.188%, range 4.329–19.075%. At 16 px mean 7.298%, range 3.695–14.262%. Full-resolution RGB MAE mean 26.905 levels out of 255, range 12.064–62.020. The four-panel detective manga (27) is the largest deviation across all three scales; the diner (14) is the smallest. Mean luma Laplacian variance (Suite four-neighbour filter) rises from 514.16 to 1763.83. This is a signal of high-frequency change and requires visual interpretation. The thumbnail scores are specific to this Base protocol.

MOSS Text Check (hybrid Qwen, greedy, seed 0) read both requested strings literally in both variants: OPEN (14) and VISIT THE FJORDS (34). This tests the requested strings only. Human preference and composition review are pending. phase1/nvfp4_comparison_01 contains summary.json, pairs.csv, text_check.json, the two loader timings, blind_pairs.jpg and the separate A-side key. The 1024 px PNGs in each render directory are available for full-size inspection.

### Blind quality review by HaDeS — 2026-11-21

On the ten-pair sheet, HaDeS marked 00B, 06A, 10B, 14B, 20A, 25B, 27A, 30B, 34B and 35B as very poor quality. He judged the better-looking alternatives barely adequate overall. Decoding the recorded key assigns nine of the ten marked images to the NVFP4 candidate. The exception is 34B (BF16, fjord poster). These are paired visual judgments on one seed and this sheet; the reviewer gave no numerical scores. Even 14B (NVFP4), the pair with the lowest thumbnail difference, was marked very poor despite its exact text reading. This NVFP4 recipe falls short of the project's quality objective in this review. Retain its size and timing as comparison data. Absolute reference quality merits checking at native resolution before setting the acceptance threshold.

Review scale clarification from HaDeS: each 1024 x 1024 render was displayed as a 256 x 256 JPEG sheet tile, potentially smaller when the viewer fits the entire sheet. The 9/10 result describes choices at that viewing scale. The observation that even the preferred images looked marginal applies to those previews; absolute quality at native resolution remains unassessed. phase1/nvfp4_comparison_01/parejas_1024.zip contains the 20 original PNGs under the same A/B labels, with no resampling or re-encoding. The original key has already been disclosed.

### Native-resolution visual verdict from HaDeS — 2026-11-21

After reviewing the original 1024 x 1024 images, HaDeS marked all ten NVFP4 renders as unacceptable and judged the BF16 renders correct. He reported illegible text and artifacts that sometimes made the depicted content hard to recognize. The 10/10 count applies to the overall visual verdict; separate counts for individual defects were not supplied. This resolves the question left open by the thumbnail review, including pair 34.

MOSS Text Check matched the two requested strings (OPEN and VISIT THE FJORDS) with its automated reader. Those matches measure string recognition on two images. Human legibility and scene recognizability are assessed separately. Exclude this specific 168-projection NVFP4 Base recipe from quality candidates and retain its measurements as a rejected baseline. Scope: these ten prompts, seed 1001 and the recorded Base workflow.

## Native ConvRot W4A4 block projections on MOSS Base — 2026-11-21

Paired against the same phase1/bf16_01 references as the NVFP4 experiment above. Same ten prompts, seed 1001, 1024 square, er_sde/simple, 25 steps, CFG 2.5, Base SLG, hybrid Qwen and Flux2 VAE on RTX 5090. For every pair the saved graphs matched after removing the model filename. tools/moss_quant_nvfp4.py --format convrot_w4a4 used the installed TensorCoreConvRotW4A4Layout defaults (rotation groups of 256, quantization groups of 64, int4 multiplication, stochastic rounding off). All 168 block projections converted. The other 211 tensors matched source bytes, shapes and dtypes.

Checkpoint phase1/models/moss_base_convrot_w4a4_blocks.safetensors, SHA256 9866ee1cee8f77aae2de26e6c0d6500b2c628dd2404f4f59e8da489508f11014. Exact size 3,035,514,568 bytes. Audit: phase1/convrot_size_audit.json. Counted 83,480 completed convrot_w4a4_linear calls over ten samplings. First sampling + decode + finite check + RGB conversion 11.10 s, median of remaining nine 9.39 s; peak allocated CUDA bytes 10,739,221,504. Loader return 0.027 s with uncontrolled OS cache and deferred materialization; this excludes first generation.

RGB thumbnail L1 means at 64/16 px: 17.553% / 15.844%. Full-resolution mean absolute RGB difference: 47.841 levels of 255. Largest at all three scales: prompt 00 (30.554%, 30.306%, 78.157/255). Mean Laplacian variance 92.51 versus 514.16 in BF16. Text Check exact 0/2: no reading for OPEN, VISIT THE FOREST for VISIT THE FJORDS. BF16 exact 2/2. This recipe saves 291,954,928 bytes versus NVFP4, with higher mean image deviation and longer warm sampling. HaDeS subsequently judged this ConvRot W4A4 recipe visually worse overall than the previously rejected NVFP4 recipe on this test. No individual pair votes were reported for ConvRot. Exclude this recipe from the quality shortlist and retain its measurements as a baseline.

Evidence: phase1/convrot_01 (graphs, PNGs and timings), phase1/convrot_comparison_01 (metrics, text readback and private A-side key). Workspace root convrot_w4a4_parejas_1024.zip contains the 20 original full-resolution PNGs with per-pair randomized labels; every archived image was byte-verified against the corresponding original.

## Native W4A8 on MOSS Base — 2026-11-21

On RTX 5090, using the same Base protocol, ten prompts, seed 1001, 25 steps, CFG 2.5, er_sde/simple, SLG and 1024-square BF16 references as the preceding experiment. Converted the same 168 projections using stock AsymW4A8Int8Layout (four-bit symmetric weight quantization with codebook, group size 16, ConvRot group size 256, eight-bit activations). No prompt calibration. The 211 other tensors match BF16 byte for byte. The complete file phase1/models/moss_base_w4a8_blocks.safetensors occupies 3,332,833,584 bytes, SHA256 7b68d419a65b53f217b06fc5f2b1b7909dfbb253fac91093def38f57279d8e4e. Breakdown in phase1/w4a8_size_audit.json.

Stock loader returned after 0.104 s in a fresh process with uncontrolled OS cache and deferred materialization. Sampling + decoding + finite check + RGB conversion: 9.387 s first image, 7.607 s median warm. PyTorch peak allocated CUDA memory 11,011,812,352 bytes. The stock W4A8 operator completed 83,480 calls during sampling. Full per-image graphs and timings in phase1/w4a8_01.

Paired RGB L1 at 64 px: mean 10.420%, range 5.351–14.228%; at 16 px: mean 8.915%. Full-resolution MAE mean 30.104/255, worst 52.635/255 (manga, prompt 27). Mean luma Laplacian variance 236.72 vs BF16 514.16. Text Check returned literal matches for both OPEN and VISIT THE FJORDS. Mean RGB deviation is above NVFP4 and below ConvRot W4A4; native-resolution human assessment remains pending. Evidence in phase1/w4a8_comparison_01. All 20 anonymized PNGs in w4a8_parejas_1024.zip at the workspace root matched the corresponding source bytes.

### Native-resolution W4A8 review by HaDeS — 2026-11-21

HaDeS ranked W4A8 above NVFP4 visually, citing better scene coherence and readable lettering where NVFP4 was illegible. He found W4A8 too bright and below the project quality target. Keep it as the strongest visually reviewed four-bit baseline on these ten scenes. The reviewer gave an overall assessment without per-image votes.

Follow-up on brightness: mean full-resolution RGB8 luma, computed from decoded PNG values as 0.2126 R + 0.7152 G + 0.0722 B with equal weight per image, was 99.39/255 for BF16, 91.16 for NVFP4 and 114.15 for W4A8. W4A8 exceeded BF16 in 9/10 scenes, by +14.76 levels on average (paired range −5.52 to +27.17). This measures the resulting images, including composition changes. Paired values: phase1/w4a8_comparison_01/mean_luma.csv.

## Native W6A8 on MOSS Base — 2026-11-21

Same RTX 5090, 168 projection matrices, ten prompts and seed 1001 as W4A8 above. The saved per-pair graphs differ only in the DiT filename. Stock AsymW4A8Int8Layout, bits=6, codebook=False, default symmetric weights, 16-weight scale groups, rotation groups of 256 and six-bit scale search. The checkpoint records w6a8_int8. Verified packed int8 row lengths of 3K/4 for all 168 matrices and all 211 literal tensors byte for byte against BF16. File phase1/models/moss_base_w6a8_blocks.safetensors is 4,521,890,208 bytes, SHA256 4c6533ef6532f420ba53d9ce195189243b3518d79ea0134fd1eef46a0c27d7ca. Size audit in phase1/w6a8_size_audit.json.

The loader returned in 0.102 seconds (one fresh process, uncontrolled OS cache, before materialization). First sampling/decoding/finite check/RGB conversion took 9.713 s, median of nine warm runs 7.593 s. Peak PyTorch allocation 12,181,939,200 bytes. 83,480 successful calls to the shared W4/W6 linear operation (instrumentation field w4a8_mm_completed). Full run records: phase1/w6a8_01.

Mean paired RGB L1 at 64/16 px was 4.053% / 3.094%; full-size MAE 12.451/255. All ten 64 px differences fell relative to W4A8. Largest deviation was the macro watch, prompt 06 (10.367% at 64 px, 28.323/255 full-size). Mean luma Laplacian variance 497.99 vs BF16 514.16. On the full-size PNGs, mean luma was 102.08 vs BF16 99.39 and W4A8 114.15. Paired W6A8 minus BF16 was +2.70 on average, brighter in 8/10, spanning -4.18 to +8.27. Text Check matched OPEN and VISIT THE FJORDS literally. In the full-resolution review, HaDeS judged the visual loss minimal and all ten W6A8 images very similar to BF16. For this MOSS Base workflow and seed, W6A8 is the accepted visual benchmark. The target for the experimental format is comparable fidelity to BF16 at fewer than 4,521,890,208 bytes for the complete DiT checkpoint. Results and per-scene luma: phase1/w6a8_comparison_01. Workspace root w6a8_parejas_1024.zip contains all 20 original PNGs, byte-verified against their blinded assignments.
## Experimental codec pilot protocol (2026-11-21, before candidate scoring)

Deployment targets established from the accepted W6A8 Base run: complete DiT under 4,521,890,208 bytes, comparable full-resolution fidelity in human review, literal text correct on both historical cases. Report candidate mean and worst L1 at 64 px alongside W6A8 (4.053%, 10.367%), plus full-size error, luma, memory and speed. These are comparison points. Only complete-checkpoint image tests can establish acceptance.

First mechanism test: fixed 128 sampled output rows per matrix from the four projections of blocks 0/20/41, full input width. Fit dictionaries to separate rows. Calibration scenes 1 and 21; validation scenes 8 and 32; indices from prompts_40_variados.txt. Final acceptance prompts stay unused. Both candidates use exactly the same dictionaries, base codes, residual proposals and number of corrections, changing only the ranking. Primary readout: paired relative Frobenius error of the projection outputs to BF16 weights, FP32 accumulation, per matrix and validation scene. Advance to block-level testing if the equal-weight mean improves on both scenes. Report regressions per matrix. No claim about final image quality or superiority to W6A8 from this test.

### Vector codec v0 paired pilot results (2026-11-21)

Hardware and workflow: RTX 5090, BF16 Base, hybrid Qwen, pilot graph in phase0/base_offline_01/graph.json, seed 1001, 1024 square, 25 steps, CFG 2.5, er_sde/simple, SLG. tools/moss_codec_capture.py forced projection forwards for hooks in an isolated process. Final captures in phase2/capture_fullspan use projection invocation indices 0, 16, 32 and 48 of 50, with 48 evenly spaced image tokens per invocation (192 total). A preliminary first-half capture in phase2/capture_01 was excluded. Calibration: Havana street and Venice watercolor. Validation: snow leopard and pixel-art harbour. No decoded-image comparison in this pilot.

tools/moss_codec_pilot.py (seed 1121) used a 128-row disjoint dictionary-fit sample and a 128-row measured sample per matrix; the full column width is retained. Two 256-entry FP16 codebooks per matrix, four weights per vector, FP16 row RMS scale, 8-bit main index, one bitmap bit per group and an 8-bit residual index on exactly one quarter of groups. Main/residual dictionaries fit by ten Lloyd iterations each on 16,384 training vectors per matrix. The weight criterion ranks reduction in squared weight error, the activation criterion ranks reduction in local quadratic error using empirical 4-by-4 uncentered second moments per input group. Both account for BF16 decode rounding. Selection neglects correlations between groups; evaluation uses the complete sampled-row matmul, including cross-group contributions. Metric: norm of output difference / norm of BF16 weight reference output, with identical FP32 operands/accumulation and TF32 disabled; unweighted mean across matrix/scene cells.

| Held-out scene | Weight ranking | Activation ranking | Improved matrices |
|---|---:|---:|---:|
| Snow leopard | 19.2488% | 16.8910% | 12/12 |
| Pixel-art harbour | 19.1958% | 17.1433% | 12/12 |
| Equal-weight mean | 19.2223% | 17.0172% | 24/24 paired cells |

Relative error reduction 11.472%. Mean reduction by projection family: qkv 7.706%, attention output 11.342%, FFN up/gate 16.275%, FFN down 8.840%. The 24 cells share model weights and scenes; they are not independent replications. Both archives in phase2/vector_pilot_01 occupy exactly 2,357,208 bytes for 6,684,672 sampled weights (2.8210 bits/weight including header, dictionaries, scales and bitmap). tools/moss_vector_codec.py round-tripped each saved matrix bit-exactly against the encoder reconstruction on CPU and GPU. CPU checks also covered non-divisible widths, empty/all/partial masks and truncated bitmaps. Row mappings, run SHA-256 values and raw paired results in results.json; paired_errors.csv is the compact output.

Decision: activation-based selection clears the predeclared two-scene pilot gate. Proceed to complete-block tests against both the weight-only ranking and the accepted W6A8 reference before scaling up. The pilot does not quantify full-checkpoint size, accumulated denoiser error, image fidelity, or generation speed.

### Complete-block comparison with native W6A8 (2026-11-21)

RTX 5090, same core, BF16 Base pilot graph, hybrid encoder, 1024 square, seed 1001, 25 steps, CFG 2.5, er_sde/simple and SLG. Frozen pilot codebooks and calibration activations; full rows and columns in the four projections of blocks 0, 20 and 41. Global per-matrix ranking, fixed 25% corrections. Both archives cover 339,738,624 weights and measure 117,032,112 bytes. phase2/full_blocks_01/weights.safetensors SHA256 7df589ca8178e6e689e50195482439a9b71c1c430b3b7e617a48f42f4a403936; activations.safetensors SHA256 2b4d1c219cf8cf72ca102908a864be82183f4f7c453a6303887fc07dd3d06c84. Conversion: tools/moss_codec_full_blocks.py. Complete-model size remains extrapolated.

tools/moss_codec_capture.py --blocks saved the complete input, modulation, positional embedding, optional mask and output for 24 BF16 calls (two validation scenes, four invocation indices 0/16/32/48 and three blocks) in phase2/block_inputs_01. No downstream propagation of candidate errors. tools/moss_codec_block_replay.py re-executed stock MOSS blocks. BF16 reproduced all recorded outputs bit-exactly. W6A8 used the accepted checkpoint's actual packed weights and completed four native linear operations per call. Dense HaDeS weights were read from each archive. The numerator for both errors is the Frobenius norm of (candidate output minus reference output); denominators are the BF16 output norm and the BF16 update norm (reference output minus original input), respectively. Equal weight per recorded call.

| Block | Output error: weights / activations / W6A8 | Update error: weights / activations / W6A8 |
|---|---|---|
| 0 | 22.936 / 23.036 / 0.557% | 36.684 / 35.199 / 1.622% |
| 20 | 3.620 / 3.139 / 0.299% | 31.138 / 27.005 / 2.578% |
| 41 | 59.611 / 61.052 / 2.125% | 70.116 / 71.833 / 2.501% |
| Mean | 32.661 / 33.087 / 0.952% | 45.543 / 44.846 / 1.840% |

Native W6A8 was closer in all 24 calls. Activation ranking improved over weight ranking in 10/24 calls; its average residual-update improvement did not produce an average block-output improvement. Diagnostic run --isolate-final: only one activation-ranked projection quantized at a time, other three BF16. Block 41 mean output error: qkv 11.512%, o 14.771%, ffn_up_gate 42.265%, ffn_down 53.568%. These isolated errors do not add. Raw paired measurements: phase2/full_blocks_01/block_errors.csv and block41_isolation.csv. Stop full-checkpoint conversion of uniform v0. A revised codec should protect the FFN projections and compete with native W6A8 on these block tests, then verify generalization on fresh scenes. This experiment reports block errors, not visual quality.

### Mixed HaDeS v0 attention + native W6A8 FFN replay (2026-11-21)

Exploratory test on the same RTX 5090, core, frozen BF16 block captures (`phase2/block_inputs_01`, 24 calls across blocks 0, 20, 41, two development validation scenes `val_pixel` and `val_wildlife`, invocation indices 0/16/32/48), and checkpoints (`mossImage_base_bf16.safetensors`, `phase1/models/moss_base_w6a8_blocks.safetensors`, `phase2/full_blocks_01/activations.safetensors`). Adapted `tools/moss_codec_block_replay.py` (`--mixed --serialize-mixed-dir`) to keep BF16 norms from `source`, load dense BF16-decoded `proj.qkv` and `proj.o` from `activations.safetensors`, and load native W6A8 `proj.ffn_up_gate` and `proj.ffn_down` from `w6`.

Verification: all 24 calls finite; `bf16` control reproduced `data["y"]` bit-exactly (`0.000%` error, `0` native `w4a8_int8_linear` calls/call); `w6` dispatched `4` native calls/call; `mixed_attn_hades_ffn_w6` dispatched exactly `2` native FFN calls/call.

| Variant | Block 0 out / upd | Block 20 out / upd | Block 41 out / upd | Mean out / upd (24 calls) |
|---|---:|---:|---:|---:|
| BF16 | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% |
| Native W6A8 | 0.433% / 0.441% | 0.299% / 2.578% | 2.125% / 2.501% | 0.952% / 1.840% |
| Mixed (HaDeS v0 attn + W6A8 FFN) | 11.111% / 11.309% | 1.624% / 14.047% | 19.147% / 22.612% | 10.628% / 15.989% |

By scene (mean across 12 calls): `val_pixel` output 10.876% / update 16.154% (W6A8: 0.958% / 1.832%); `val_wildlife` output 10.379% / update 15.824% (W6A8: 0.946% / 1.849%).

Serialized 12-projection archive `phase2/mixed_attn_hades_ffn_w6_01/mixed_blocks.safetensors`: 223,291,540 bytes total (39,026,688 B HaDeS v0 attention + 184,258,692 B W6A8 FFN + 6,160 B header/metadata), vs 276,418,824 tensor bytes for the 12 W6A8 projections (92,160,132 B attention + 184,258,692 B FFN) and 117,032,112 B for uniform HaDeS v0 (`activations.safetensors`). Extrapolating the 3-block tensor size (x14) plus 651,935,808 literal bytes yields ~3,777,931,128 bytes (excluding full-model header; extrapolated, not generated).

Decision (pre-declared gate: mean output error <= 3.0% and block 41 <= 5.0%): failed. Protecting FFN with W6A8 reduced mean output error from 33.087% to 10.628%, but HaDeS v0 attention alone remains ~11x above W6A8 (0.952%). Do not scale this v0 attention representation to full-model or fresh-scene image tests; redesign the attention representation as well. Evidence: `phase2/mixed_attn_hades_ffn_w6_01/{mixed_block_errors.csv,size_breakdown.json,mixed_blocks.safetensors,replay.log}`.

### Vector codec v1 screening: subgroup scales and 2D/4D capacity (2026-11-22)

Exploratory screen via `tools/moss_codec_v1_screen.py` on the same RTX 5090, core, frozen BF16 inputs (`phase2/block_inputs_01`, 24 calls across blocks 0, 20, 41), calibration captures (`phase2/capture_fullspan`, seed 1121), and controls (`bf16`, `w6`). Four v1 variants tested across all 12 projections of blocks 0, 20, and 41 (256-entry FP16 codebooks, 10 Lloyd iterations on 16,384 vectors from 256 sampled rows per matrix, BF16 norms preserved):

- `v1_g64_4d_r25`: subgroup RMS scale `g=64`, 4D base + 25% 4D residual selected by activations (`127,458,696` B; extrapolated 42 blocks: `2,436,251,712` B).
- `v1_g64_4d_r100`: subgroup RMS scale `g=64`, 4D base + 100% 4D residual (`180,541,736` B; extrapolated: `3,179,429,952` B).
- `v1_g64_2d_r100`: subgroup RMS scale `g=64`, 2D 1-stage (`180,502,456` B; extrapolated: `3,178,913,856` B; equal-byte control to `v1_g64_4d_r100`).
- `v1_g16_2d_r100`: subgroup RMS scale `g=16`, 2D 1-stage (`212,353,008` B; extrapolated: `3,624,820,800` B).

| Variant | Block 0 out / upd | Block 20 out / upd | Block 41 out / upd | Mean out / upd (24 calls) | 3-block file bytes |
|---|---:|---:|---:|---:|---:|
| `bf16` | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | — |
| `w6` (native W6A8) | 0.433% / 0.441% | 0.299% / 2.578% | 2.125% / 2.501% | 0.952% / 1.840% | 276,418,824 (tensors) |
| `v1_g64_4d_r25` | 14.489% / 14.749% | 3.399% / 29.269% | 41.448% / 48.762% | 19.779% / 30.926% | 127,458,696 |
| `v1_g64_4d_r100` | 13.223% / 13.460% | 2.735% / 23.482% | 41.267% / 48.651% | 19.075% / 28.531% | 180,541,736 |
| `v1_g64_2d_r100` | 24.236% / 24.670% | 3.009% / 25.789% | 57.906% / 68.229% | 28.384% / 39.562% | 180,502,456 |
| `v1_g16_2d_r100` | 14.450% / 14.708% | 2.204% / 18.875% | 36.685% / 43.214% | 17.779% / 25.599% | 212,353,008 |

Verification: all 24 calls finite per variant; `bf16` bit-exact (`0` native calls); `w6` `4` native calls/call; v1 variants `0` native calls/call and finite round-trip decode.

Findings and decision: (1) Replacing row-wide RMS scale with `g=64` subgroup scale alone cut mean output error from `33.087%` (v0) to `19.779%`. (2) At equal bytes (`~180.5 MB`), 2-stage 4D RVQ (`19.075%`) beat 1-stage 2D VQ (`28.384%`). (3) All four v1 designs failed the pre-declared `<= 3.0%` mean / `<= 5.0%` block-41 gate because 256-entry 2D/4D codebooks only provide ~4-bit-equivalent resolution (~16 levels/weight) on unrotated weights, far below W6A8's 64 levels/weight. Next step: test higher-resolution representations (e.g., 5–6 bit scalar/rotated quantization or 3-stage / larger-codebook VQ with orthogonal rotation) against W6A8 on `phase2/block_inputs_01`. Evidence: `phase2/v1_screen_01/{v1_block_errors.csv,v1_size_summary.json,*.safetensors}` and `phase2/v1_screen_01.log`.

### Hadamard-rotated (FWHT-64) v2 screening on blocks 0, 20, 41 (2026-11-22)

Exploratory screen via `tools/moss_codec_v2_screen.py` on the same RTX 5090, core, frozen BF16 inputs (`phase2/block_inputs_01`, 24 calls), and controls (`bf16`, `w6`). Applied an orthogonal 64x64 Sylvester Hadamard rotation (`FWHT-64`, zero file overhead) per subgroup of 64 weights before encoding, comparing 4D 2-stage RVQ (`v2_fwht64_4d_s2`, seed 1121, `4.25 bpw`) against bit-packed symmetric scalar quantization at 4, 5, and 6 bits (`v2_fwht64_q4` at `4.25 bpw`, `v2_fwht64_q5` at `5.25 bpw`, `v2_fwht64_q6` at `6.25 bpw`), with exact round-trip pack/unpack and BF16 norms preserved.

| Variant | Block 0 out / upd | Block 20 out / upd | Block 41 out / upd | Mean out / upd (24 calls) | 3-block file bytes (extrap. 42 bl.) |
|---|---:|---:|---:|---:|---|
| `bf16` | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | — |
| `w6` (native W6A8) | 0.433% / 0.441% | 0.299% / 2.578% | 2.125% / 2.501% | 0.952% / 1.840% | 276,418,824 B tensors (4,521,890,208 B) |
| `v2_fwht64_4d_s2` | 3.411% / 3.472% | 1.357% / 11.723% | 9.523% / 11.193% | 4.764% / 8.796% | 180,541,688 B (~3,179,429,952 B) |
| `v2_fwht64_q4` | 0.885% / 0.900% | 1.178% / 10.202% | 8.720% / 10.249% | 3.594% / 7.117% | 180,489,024 B (~3,178,741,824 B) |
| `v2_fwht64_q5` | 0.682% / 0.694% | 0.560% / 4.841% | 3.971% / 4.664% | **1.737% / 3.400%** | 222,956,352 B (~3,773,284,416 B) |
| `v2_fwht64_q6` | 0.368% / 0.375% | 0.312% / 2.708% | 2.139% / 2.516% | **0.940% / 1.866%** | 265,423,680 B (~4,367,827,008 B) |

Verification: all 24 calls finite per variant; `bf16` bit-exact (`0` native calls); `w6` `4` native calls/call; v2 variants `0` native calls/call and finite round-trip decode.

Findings and decision: (1) Adding `FWHT-64` alone cut 4D 2-stage RVQ mean output error from `19.075%` (`v1_g64_4d_r100`) to `4.764%` at identical bytes. (2) At equal bytes (`4.25 bpw`, `~180.5 MB`), rotated 4-bit scalar (`3.594%`) beat rotated 4D RVQ (`4.764%`) without codebooks. (3) Both `v2_fwht64_q5` (`1.737%` mean, `3.971%` block 41, `~3.77 GB` extrapolated) and `v2_fwht64_q6` (`0.940%` mean, `2.139%` block 41, matching `w6`'s `0.952%` at `~4.37 GB` extrapolated) pass the pre-declared `<= 3.0%` mean / `<= 5.0%` block-41 gate. Next step: verify generalization on fresh unseen scenes and blocks before full-model conversion and image testing. Evidence: `phase2/v2_screen_01/{v2_block_errors.csv,v2_size_summary.json,*.safetensors}` and `phase2/v2_screen_01.log`.

### Fresh-scene generalization of FWHT-64 5-bit and 6-bit (2026-11-22)

Confirmatory out-of-sample test on 3 unseen prompts from `I:/HaDeS_MOSS/prompts/prompts_40_variados.txt` (`fresh_portrait=2`, `fresh_bw=12`, `fresh_ukiyoe=22`; unused in `phase1` and `phase2/capture_fullspan`). `tools/moss_codec_capture.py --blocks` recorded 36 BF16 block calls (3 scenes x 3 blocks x 4 invocations) in `phase2/block_inputs_fresh_01`. `tools/moss_codec_v2_screen.py --replay-from phase2/v2_screen_01` replayed the already-serialized `v2_fwht64_q5.safetensors` and `v2_fwht64_q6.safetensors` archives alongside `bf16` and `w6`.

| Variant | Block 0 out / upd | Block 20 out / upd | Block 41 out / upd | Mean out / upd (36 calls) | Per-scene out (portrait / bw / ukiyoe) |
|---|---:|---:|---:|---:|---|
| `bf16` | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% / 0.000% |
| `w6` (native W6A8) | 0.444% / 0.452% | 0.305% / 2.725% | 2.159% / 2.567% | 0.969% / 1.915% | 1.030% / 0.910% / 0.967% |
| `v2_fwht64_q5` | 0.640% / 0.652% | 0.576% / 5.131% | 3.958% / 4.696% | **1.725% / 3.493%** | 1.780% / 1.710% / 1.684% |
| `v2_fwht64_q6` | 0.419% / 0.427% | 0.336% / 3.001% | 2.129% / 2.528% | **0.961% / 1.985%** | 0.977% / 0.958% / 0.948% |

Verification: all 36 calls finite per variant; `bf16` bit-exact (`0` native calls); `w6` `4` native calls/call; `v2_fwht64_q5` and `v2_fwht64_q6` `0` native calls/call.

Decision: both candidates generalize cleanly to unseen scenes, clearing the pre-declared confirmation gate (`q5` `1.725% <= 3.0%`, block 41 `3.958% <= 5.0%`; `q6` `0.961% <= 1.5%`, matching `w6`'s `0.969%`). Advance to Phase 3: convert complete MOSS Base BF16 checkpoints (including `v2_fwht64_q5`, `~3.77 GB` extrapolated) and evaluate full-model image fidelity on Base and Turbo. Evidence: `phase2/v2_fresh_01/v2_block_errors.csv`, `phase2/v2_fresh_01.log`, `phase2/block_inputs_fresh_01.log`.

## Phase 3: Complete MOSS Base FWHT-64 5-bit checkpoint and 10-scene evaluation (2026-11-22)

Same RTX 5090, core, hybrid Qwen, Flux2 VAE, ten Base prompts (`0,6,10,14,20,25,27,30,34,35`), seed 1001, 1024 square, 25 steps, CFG 2.5, er_sde/simple, Base SLG, paired against `phase1/bf16_01`. Conversion: `tools/moss_codec_v2_convert.py --bits 5` quantized all 168 block projections (`FWHT-64`, `g=64`, symmetric 5-bit packed `[-15, 15]`) and preserved all 211 other tensors byte-for-byte. Packed archive `phase3/models/moss_base_fwht64_q5_blocks.safetensors`: **3,773,344,112 bytes** (SHA-256 `608f91dd737c28af3a7b1632e2495c5e8b688f377832de89b126b7abe3185631`; 2,972,712,960 B packed + 148,635,648 B scales + 651,935,808 B literals + 59,696 B header/metadata), saving **748,546,096 bytes** vs W6A8 (`4,521,890,208` B). Round-trip decode verified bit-exact per matrix against the in-memory reconstruction; for Phase 3 format-quality isolation before implementing a deferred loader, `tools/moss_quant_grid.py` rendered the dense BF16 reconstruction (`moss_base_fwht64_q5_dense_bf16.safetensors`, 10,164,655,808 B).

`tools/moss_quant_compare.py` vs `phase1/bf16_01`:
- **L1 @ 64 px:** mean **6.856%** (median 6.768%, range 3.251%–11.158%, worst prompt 27) — vs W6A8 `4.053%`, NVFP4 `9.103%`, W4A8 `10.420%`.
- **L1 @ 16 px:** mean **6.084%** (range 2.715%–9.498%).
- **Full-size RGB MAE:** mean **19.653/255** (median 18.432/255, range 8.904–39.761/255) — vs W6A8 `12.451/255`, NVFP4 `26.905/255`, W4A8 `30.104/255`.
- **Mean luma Laplacian variance:** **431.25** (vs BF16 `514.16`, W6A8 `497.99`).
- **Mean full-size RGB8 luma:** **86.78/255** vs BF16 `99.39/255` (`-12.61` levels on average, darker in 10/10 scenes, range `-24.66` to `-3.44`).
- **MOSS Text Check:** exact literal match **2/2** (`OPEN` on 14, `VISIT THE FJORDS` on 34).

Evidence: `phase3/fwht64_q5_size_audit.json`, `phase3/fwht64_q5_01`, `phase3/fwht64_q5_comparison_01/{summary.json,pairs.csv,text_check.json,blind_pairs.jpg,blind_key_A.json}`, and workspace root `fwht64_q5_parejas_1024.zip` (SHA-256 `703c3f4ddbcb961f186352e5939862a8d73ebe44166f0583debdabbe086c9235`) for HaDeS blind native-resolution review.

### Blind native-resolution review by HaDeS — 2026-11-22

After inspecting the 10 full-resolution pairs in `fwht64_q5_parejas_1024.zip`, HaDeS judged the pairs very close ("son muy cercanos") and requested the key (`blind_key_A.json` disclosed afterwards: `A = BF16` on `0, 6, 20, 27, 30`; `A = FWHT-64 5-bit` on `10, 14, 25, 34, 35`). Next: test `FWHT-64 5-bit` on `HaDeS_MOSS_v1.0_Turbo` (8 steps, CFG 1) and validate the pure-PyTorch `MODEL` loader (Phase 4: load time, VRAM, and reconstruction cost).

### Higher-resolution (`1152x1536`) Base evaluation of FWHT-64 5-bit (2026-11-22)

Same ten Base prompts, seed 1001, 25 steps, CFG 2.5, er_sde/simple, Base SLG, hybrid Qwen, Flux2 VAE, and dense BF16-decoded `FWHT-64 5-bit` (`moss_base_fwht64_q5_dense_bf16.safetensors`), rendered at `1152x1536` (`phase3/graph_1152x1536.json`) against fresh BF16 renders (`phase3/bf16_1152x1536_01`).

- **L1 @ 64 px:** mean **8.301%** (median 8.539%, range 4.782%–12.344%, max prompt 34).
- **L1 @ 16 px:** mean **7.592%** (median 8.212%, range 3.838%–11.940%).
- **Full-size RGB MAE:** mean **23.249/255** (median 22.792/255, range 12.498–39.919/255).
- **Mean luma Laplacian variance:** **700.72** (vs BF16 `737.81`).
- **Mean full-size RGB8 luma:** **82.68/255** vs BF16 `100.40/255` (`-17.72` levels on average, darker in 10/10 scenes, range `-29.92` to `-9.06`).
- **MOSS Text Check:** exact literal match **2/2** (`OPEN` on 14, `VISIT THE FJORDS` on 34).

Blinded full-resolution 1152x1536 pairs: `fwht64_q5_parejas_1152x1536.zip` (SHA-256 `85dd5a010e469b691ab4d7f19dcd4174400b7812e2b9dd3d7c49f4d7e91bc784`); private key kept unopened in `phase3/fwht64_q5_comparison_1152x1536_01/blind_key_A.json` pending HaDeS review.

Human review (HaDeS, 2026-11-22): judged the `1152x1536` pairs quite close ("igual que antes, son bastante cercanas"); `blind_key_A.json` remains unopened. Next: test `FWHT-64 5-bit` on `HaDeS_MOSS_v1.0_Turbo` (8 steps, CFG 1) and implement/benchmark the pure-PyTorch `MODEL` loader (Phase 4).

### HaDeS MOSS v1.0 Turbo FWHT-64 5-bit evaluation at 1024x1024 (2026-11-22)

Converted `HaDeS_MOSS_v1.0_Turbo.safetensors` (`10,164,656,216` B, SHA-256 `cfd7e0db798616954112604d28bcc55bf1adf56851d07f112fc786876c541e9e`) with `tools/moss_codec_v2_convert.py --bits 5` to `phase3/models/moss_v1_turbo_fwht64_q5_blocks.safetensors`: **3,773,343,976 bytes** (SHA-256 `5ff7ed8bb416f8ac361e22160ff82281bde26b633271395d3ac6199ccd5ab48d`; 168 quantized matrices + 210 literal tensors). Rendered the ten `phase1` prompts (`1024x1024`, seed 1001, 8 steps, CFG 1.0, `euler`/`simple`, hybrid Qwen, Flux2 VAE) using the dense BF16 reconstruction (`phase3/turbo_fwht64_q5_01`) vs `HaDeS_MOSS_v1.0_Turbo.safetensors` (`phase3/turbo_bf16_01`).

- **L1 @ 64 px:** mean **6.404%** (median 7.072%, range 2.984%–9.152%, max prompt 6).
- **L1 @ 16 px:** mean **4.597%** (median 4.527%, range 2.391%–7.627%).
- **Full-size RGB MAE:** mean **22.120/255** (median 22.451/255, range 12.022–35.056/255).
- **Mean luma Laplacian variance:** **1502.19** (vs Turbo BF16 `1594.84`).
- **Mean full-size RGB8 luma:** **131.48/255** vs Turbo BF16 `132.58/255` (`-1.10` levels on average, range `-6.70` to `+9.54`, brighter in 3/10).
- **MOSS Text Check:** exact literal match **2/2** (`OPEN` on 14, `VISIT THE FJORDS` on 34).

Blinded 1024x1024 pairs: `turbo_fwht64_q5_parejas_1024.zip` (SHA-256 `b5c8e50996eab9c5e10e38d315ac122dab3b7448b2476ca00afccd27d5cc297a`); private key kept unopened in `phase3/turbo_fwht64_q5_comparison_01/blind_key_A.json` pending HaDeS review.

Human review (HaDeS, 2026-11-22): accepted `FWHT-64 5-bit` for its file size (`3.77 GB`) on `HaDeS_MOSS_v1.0_Turbo` ("Estoy conforme. Es una calidad aceptable para el peso."). `blind_key_A.json` remains unopened. Ready for Phase 4 (`MODEL` loader in pure PyTorch, VRAM/load/decode benchmarks, and LoRA regression checks).

## Native mixed-precision priority (W4A8 + W6A8, no custom loader)

### Native W4A8 vs W6A8 sensitivity and mixability screen (2026-11-22)

Exploratory screen via `tools/moss_native_mix_screen.py` on the 36 unseen BF16 block calls (`phase2/block_inputs_fresh_01`, blocks 0, 20, 41), mixing `asym_w4a8_int8` (`phase1/models/moss_base_w4a8_blocks.safetensors`) and `w6a8_int8` (`phase1/models/moss_base_w6a8_blocks.safetensors`) inside a single `Block` with `comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)` and BF16 norms.

| Variant | Block 0 out / upd | Block 20 out / upd | Block 41 out / upd | Mean out / upd |
|---|---:|---:|---:|---:|
| `bf16` | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% |
| `w6` (all W6A8) | 0.444% / 0.452% | 0.305% / 2.725% | 2.159% / 2.567% | 0.969% / 1.915% |
| `w4` (all W4A8) | 1.293% / 1.318% | 0.817% / 7.263% | 5.970% / 7.085% | 2.693% / 5.222% |
| `w4_attn_w6_ffn` | 1.133% / 1.154% | 0.529% / 4.719% | 5.332% / 6.340% | 2.331% / 4.071% |
| `w6_attn_w4_ffn` | 1.079% / 1.099% | 0.683% / 6.072% | 3.418% / 4.044% | **1.727% / 3.739%** |

Block 41 single-projection W4A8 isolation (other three W6A8, 12 calls): `qkv` **4.258%**, `o` **3.801%**, `ffn_up_gate` **2.795%**, `ffn_down` **2.673%**.

Verification and findings: all mixed W4A8+W6A8 blocks loaded cleanly and dispatched exactly `4` native `w4a8_int8_linear` calls per block call. Contrary to `v0` (where FFN dominated), in native `AsymW4A8Int8Layout` **attention (`qkv`, `o`) is more sensitive than FFN**. Because attention is only 1/3 of each block's weights, `w6_attn_w4_ffn` extrapolates to **~3.73 GB** across 42 blocks (leaving ~70 MB under the `~3.80 GB` target to also protect late-block FFN if needed) while matching `FWHT-64 5-bit`'s block error (`1.727%` vs `1.725%`). Evidence: `phase4_native_mix/sensitivity_01/native_mix_errors.csv` and `phase4_native_mix/sensitivity_01.log`.

### Native mixed-precision v1 (`w6_attn + w4_ffn + ffn_w6_38_41`) on MOSS Base 1024x1024 (2026-11-22)

Built `phase4_native_mix/models/moss_base_native_mix_v1.safetensors` via `tools/moss_native_mix_convert.py` (**3,804,677,908 bytes**, SHA-256 `f60057040276935bad18d1450f7c08995168499ef99b6b759b7a5b775f03f2e1`; 92 `w6a8_int8` tensors = 1,535,920,104 B, 76 `asym_w4a8_int8` tensors = 1,616,722,692 B, 211 literal tensors = 651,935,808 B, 99,304 B header/metadata including `hades/recipe`). Loaded directly with stock `UNETLoader` and rendered the ten `phase1` Base prompts (`1024x1024`, seed 1001, 25 steps, CFG 2.5, `er_sde/simple`, Base SLG) against `phase1/bf16_01`:

- **Native kernel dispatch:** **`83,480`** completed `w4a8_int8_linear` calls across the 10 samplings (100% native path).
- **Speed & VRAM:** first sampling `8.944 s`, median warm **`7.034 s`** (vs BF16 `14.657 s`, **2.08x faster**); peak allocated CUDA memory **`11,462,130,688` B** (vs BF16 `17,813,873,664` B, **-6.35 GB**).
- **L1 @ 64 px:** mean **6.731%**, median **5.693%** (vs W4A8 `10.420%`, FWHT-64 5-bit `6.856%` / median `6.768%`, W6A8 `4.053%`).
- **L1 @ 16 px:** mean **5.424%**, median **4.787%** (vs FWHT-64 5-bit `6.084%`).
- **Full-size RGB MAE:** mean **20.108/255**, median **17.189/255** (vs W4A8 `30.104/255`, FWHT-64 5-bit `19.653/255`).
- **Mean luma Laplacian variance:** **361.87** (vs BF16 `514.16`, W4A8 `236.72`).
- **Mean full-size RGB8 luma:** **107.18/255** vs BF16 `99.39/255` (`+7.79` levels, brighter in 10/10, range `+2.81` to `+22.92` — halving W4A8's `+14.76` shift and avoiding FWHT-64 5-bit's `-12.61` darkening).
- **MOSS Text Check:** exact literal match **2/2** (`OPEN` on 14, `VISIT THE FJORDS` on 34).

Blinded 1024x1024 pairs: `native_mix_base_parejas_1024.zip` (SHA-256 `6ca57b383d03b1cfc16e507ad0872c7017eb26e32c9fb1077b61423e14f50df1`); private key kept unopened in `phase4_native_mix/base_1024_comparison_01/blind_key_A.json` pending HaDeS review.

Human review (HaDeS, 2026-11-22) and verification: in blind review of `native_mix_base_parejas_1024.zip`, both models appear equal in brightness/darkness by eye (`blind_key_A.json` remains unopened). `H:/Comfy_UI_V61/ComfyUI/models/diffusion_models/HMS_base_v1.safetensors` (3,804,677,908 B) is a byte-identical copy of `phase4_native_mix/models/moss_base_native_mix_v1.safetensors` (both SHA-256 `f60057040276935bad18d1450f7c08995168499ef99b6b759b7a5b775f03f2e1`). Per-scene RGB8 mean luma (`phase4_native_mix/base_1024_comparison_01/mean_luma.csv`, `0.2126 R + 0.7152 G + 0.0722 B`): `00` 59.7 vs 65.7 (`+6.04`), `06` 54.0 vs 63.0 (`+9.00`), `10` 40.2 vs 45.1 (`+4.94`), `14` 26.3 vs 31.1 (`+4.84`), `20` 79.6 vs 86.9 (`+7.28`), `25` 124.5 vs 147.4 (`+22.92`), `27` 184.5 vs 187.8 (`+3.31`), `30` 85.5 vs 97.5 (`+11.95`), `34` 146.2 vs 149.0 (`+2.81`), `35` 193.4 vs 198.2 (`+4.83`) — brighter in 10/10, mean `+7.79`, median `+5.49` (scene 25 inflates the mean; small steady bias, not noticeable to HaDeS by eye; monitored for calibration, non-blocking). Open item noted: Laplacian variance `361.87` vs BF16 `514.16` (-30%; W6A8 `497.99`, FWHT-5 `431.25`).

### Native mixed-precision v1 on HaDeS MOSS v1.0 Turbo 1024x1024 (2026-11-22)

Converted `HaDeS_MOSS_v1.0_Turbo.safetensors` via `tools/moss_native_mix_convert.py --protected-ffn-blocks 38-41` to `phase4_native_mix/models/moss_v1_turbo_native_mix_v1.safetensors`: **3,804,678,060 bytes** (SHA-256 `8ac061064f95968799c022f39943db6e0a4309d53f15fa9a83801fd56e99238a`). Loaded via stock `UNETLoader` and rendered the ten `phase1` prompts (`1024x1024`, seed 1001, 8 steps, CFG 1.0, `euler`/`simple`, hybrid Qwen, Flux2 VAE) against `phase3/turbo_bf16_01`:

- **Native kernel dispatch:** **`13,440`** completed `w4a8_int8_linear` calls across 10 samplings (100% native path).
- **Speed & VRAM:** first sampling `5.783 s`, median warm **`1.266 s`** (vs Turbo BF16 `2.543 s`, **2.01x faster**); peak allocated CUDA memory **`11,462,130,688` B** (vs BF16 `17,813,873,664` B, **-6.35 GB**).
- **L1 @ 64 px:** mean **6.181%**, median **5.981%** (vs FWHT-64 5-bit `6.404%` / median `7.072%`).
- **L1 @ 16 px:** mean **4.618%**, median **4.770%**.
- **Full-size RGB MAE:** mean **21.219/255**, median **20.509/255** (vs FWHT-64 5-bit `22.120/255`).
- **Mean luma Laplacian variance:** **1251.36** (vs Turbo BF16 `1594.84`).
- **Mean full-size RGB8 luma** (`phase4_native_mix/turbo_1024_comparison_01/mean_luma.csv`): mean **`+2.83`**, median **`+1.42`**, brighter in `8/10` (`00` 132.6 vs 134.7 `+2.08`; `06` 99.5 vs 99.9 `+0.43`; `10` 51.7 vs 56.3 `+4.57`; `14` 64.7 vs 62.5 `-2.18`; `20` 134.4 vs 130.0 `-4.42`; `25` 136.4 vs 147.3 `+10.94`; `27` 182.5 vs 184.2 `+1.65`; `30` 133.7 vs 147.0 `+13.35`; `34` 182.9 vs 183.6 `+0.74`; `35` 207.4 vs 208.6 `+1.20`).
- **MOSS Text Check:** exact literal match **2/2** (`OPEN` on 14, `VISIT THE FJORDS` on 34).

Blinded 1024x1024 pairs: `native_mix_turbo_parejas_1024.zip` (SHA-256 `e0f7c1f3ccff5abe8d8dc2024734d5508e0d38738f390b6d1e36536770edb817`); private key kept unopened in `phase4_native_mix/turbo_1024_comparison_01/blind_key_A.json` pending HaDeS review.

### Protected FFN sweep in Base: `ffn30` (`30-41`) and `ffn20` (`20-41`) (2026-11-22)

Built `phase4_native_mix/models/moss_base_native_mix_ffn30.safetensors` (`--protected-ffn-blocks 30-41`, **3,955,670,084 bytes**, SHA-256 `90f83f76399c01666d5b7a49fcadc90b0e4bc830a07b8cb2fa48b01aa3171a44`) and `moss_base_native_mix_ffn20.safetensors` (`--protected-ffn-blocks 20-41`, **4,144,410,304 bytes**, SHA-256 `ced7665ead90185259bd3a2d0c259888888617ad29014598ae9e3fabcfde2e28`), and rendered the ten `phase1` Base prompts (`1024x1024`, seed 1001, 25 steps, CFG 2.5, `er_sde/simple`, Base SLG) via stock `UNETLoader` (`83,480` `w4a8_int8_linear` calls each) against `phase1/bf16_01`:

| Mix | Protected FFN | Size (bytes) | L1 @ 64 px | MAE / 255 | Luma diff (mean / med) | Laplacian var | Warm s | Peak VRAM |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `v1` | `38-41` | 3,804,677,908 | 6.731% | 20.108 | +7.79 / +5.49 (10/10) | 361.87 | 7.034 | 11.462 GB |
| `ffn30` | `30-41` | 3,955,670,084 | 5.327% | 16.644 | +3.01 / +2.73 (9/10) | 408.39 | 7.067 | 11.613 GB |
| `ffn20` | `20-41` | 4,144,410,304 | 4.725% | 14.657 | +2.13 / +2.08 (8/10) | 432.04 | 7.087 | 11.802 GB |
| `W6A8` | `0-41` | 4,521,890,208 | 4.053% | 12.451 | +2.70 / — (8/10) | 497.99 | 7.593 | 12.182 GB |

Per-scene `mean_luma.csv` and `summary.json` saved in `phase4_native_mix/base_ffn30_comparison_01` and `phase4_native_mix/base_ffn20_comparison_01` (no blind ZIPs generated yet).

### Native Mix v1 vs ffn30: text (2026-11-22)

Converted `HaDeS_MOSS_v1.0_Turbo.safetensors` with `--protected-ffn-blocks 30-41` to `phase4_native_mix/models/moss_v1_turbo_native_mix_ffn30.safetensors` (**3,955,670,236 bytes**, SHA-256 `ffdae75512091d90447855034c6ac19c3b9cc2fb820e56a4f8d70c5c7fac96c5`). Rendered 8 text prompts (`phase5_text/prompts_text.txt`, 11 expected strings per seed) across 3 seeds (`1001, 1002, 1003`, `1024x1024`, hybrid Qwen, Flux2 VAE) on Base (25 steps, CFG 2.5, `er_sde/simple`, SLG) and Turbo (8 steps, CFG 1.0, `euler/simple`) for `bf16`, `v1`, and `ffn30` (144 images total), scored with MOSS Text Check (`tools/moss_text_score_sheets.py`):

| Family | Model | Exact | Close | Missing | Outside box | Failures |
|---|---|---:|---:|---:|---:|---|
| Base | `bf16` | 33 | 0 | 0 | 0 | — |
| Base | `v1` (`38-41`) | 33 | 0 | 0 | 0 | — |
| Base | `ffn30` (`30-41`) | 32 | 1 | 0 | 0 | `p04/s1003`: `THE LAST NIGHT` -> `THE NIGHT` (`close`) |
| Turbo | `bf16` | 33 | 0 | 0 | 0 | — |
| Turbo | `v1` (`38-41`) | 33 | 0 | 0 | 0 | — |
| Turbo | `ffn30` (`30-41`) | 33 | 0 | 0 | 0 | — |

`ffn30` does not improve text rendering over `v1` (`v1` achieves `66/66` exact across Base and Turbo vs `65/66` for `ffn30`). Comparison sheets (`BF16 | v1 | ffn30`, 3 seeds per prompt): `phase5_text/sheets/{base,turbo}_p00..07.jpg`. Evidence: `phase5_text/text_summary.json`, `phase5_text/text_details.csv`.

## HMS v1 calibration — 2026-11-22

HMS v1 is the uncalibrated native ffn30 recipe. Four calibration prompts (zero-based IDs 1, 21, 3, 23) per family with independent BF16 Base and Turbo traces. Each trace covers 42 blocks × four projections, 48 image positions at four invocation indices per batch element. Full per-projection counts and indices: phase6_calib/run01/{base,turbo}/cal_*.json. Base historical cross-check (24 scene/projection combinations): relative differences in ConvRot activation second moments below 9e-8. For the SLG-skipped block, indices were scaled to actual calls. Runtime and rotation checks: phase6_calib/run01/check.log.

C1: hold ffn30 per-channel scales fixed, fit W4 LUT using activation-energy-weighted Lloyd (300,000 sampled weights, seed 0, 25 iterations); score original and adjacent FP8 group scales plus clips from 0.80 through 1.00 by 0.05 on the decoded INT8 grid. Core BF16 dequant versus explicit reconstruction checked elementwise. Validation on 36 BF16 block calls (three held-out scenes × three blocks × four invocations), arithmetic mean of relative L2 norms per case:

| Variant | Block output | Residual update |
|---|---:|---:|
| BF16 | 0 | 0 |
| W6A8 | 0.00969264 | 0.01914777 |
| ffn30 | 0.01306800 | 0.03246051 |
| C1 | 0.01353111 | 0.03219174 |

C1 increases mean output error by 3.54%, failing the required 10% reduction. Four successful native linear calls per quantized block case. Raw: phase6_calib/run01/c1_replay.csv. No human verdict on C1.

C2 (same frozen C1 channel/group scales and codebooks): GPTQ in rotated coordinates, full uncentered FP32 second moment from 768 calibration rows per projection, 1% mean-diagonal damping, sequential columns in blocks of 128. Twelve calibrated matrices verified against eager dequant. On the same 36 held-out block calls, C2 output error **0.01115878**, update error **0.02910035**, a **14.61% relative improvement in output** over ffn30. C2 passes the block gate; W6A8 remains better on average. Raw cases: phase6_calib/c2_pilot01/c2_replay.csv. C1 checkpoints retain exact ffn30 lengths (3,955,670,084 B Base, 3,955,670,236 B Turbo) and tensor headers.

C2 vs the historical W6A8 anchor, same ten Base scenes / seed 1001: L1@64 4.02444% vs 4.05346%, full-resolution RGB MAE 12.79611 vs 12.45086 per 255. At 3,955,670,084 vs 4,521,890,208 bytes, C2 saves 566,220,124 bytes (12.52% of W6A8) and gives similar mean L1@64 on this set. W6A8 still has lower mean block output error and slightly lower full-resolution MAE. This comparison does not establish perceptual equivalence. Sources: phase1/w6a8_comparison_01/summary.json, phase1/w6a8_size_audit.json and phase6_calib/visual_c2_01/base_grid_c2_score/summary.json.

Base C2 Text Check exceptions (all close, zero missing): p00/1003 FRESH BREAD DAILY -> FRESH BREADAILY; p04/1001 and p04/1003 THE LAST NIGHT -> THE NIGHT; p05/1003 MAIN STREET -> MAIN STRET. The p04/1003 miss also occurred in ffn30. These are automated readings. Turbo retains 33/33; all three lost exact matches relative to ffn30 are in Base. Per-image OCR output: phase6_calib/visual_c2_01/base_text_c2_score/text_check.json.

### 2026-11-22T03:24:16+01:00 — Cierre de la calibración HMS v1

| Familia | Serie | Variante | L1@64 % | L1@16 % | MAE /255 | Var. Lap. | Δ luma media / mediana | Más claras | Text Check | s caliente | VRAM pico B | Ops. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | grid | ffn30 | 5.327 | 3.826 | 16.644 | 408.393 | 3.010 / 2.733 | 9/10 | 2/2 | 7.067 | 11613117440 | 83480 |
| base | grid | c1 | 4.816 | 3.721 | 15.177 | 561.735 | 4.873 / 5.467 | 9/10 | 2/2 | 7.157 | 11613117440 | 83480 |
| base | grid | c2 | 4.024 | 2.953 | 12.796 | 521.640 | 1.297 / 0.555 | 5/10 | 2/2 | 7.161 | 11613117440 | 83480 |
| base | text | ffn30 | 8.091 | 6.388 | 22.701 | 94.057 | -0.422 / 1.292 | 15/24 | 32/33 | 7.369 | 11613117440 | 200352 |
| base | text | c1 | 7.708 | 6.023 | 21.815 | 85.185 | 3.949 / 3.785 | 17/24 | 33/33 | 6.931 | 11613117440 | 200352 |
| base | text | c2 | 7.588 | 6.044 | 21.398 | 95.634 | 1.189 / -1.736 | 7/24 | 29/33 | 6.747 | 11613117440 | 200352 |
| turbo | grid | ffn30 | 6.441 | 4.642 | 22.397 | 1553.046 | 2.217 / 2.076 | 8/10 | 2/2 | 1.629 | 11613117440 | 13440 |
| turbo | grid | c1 | 6.248 | 4.410 | 21.785 | 1589.016 | 1.386 / 1.063 | 7/10 | 2/2 | 1.640 | 11613117440 | 13440 |
| turbo | grid | c2 | 5.519 | 3.872 | 19.352 | 1549.248 | 1.200 / 0.312 | 8/10 | 2/2 | 1.606 | 11613117440 | 13440 |
| turbo | text | ffn30 | 5.871 | 3.946 | 19.450 | 658.028 | 5.009 / 5.394 | 22/24 | 33/33 | 1.271 | 11613117440 | 32256 |
| turbo | text | c1 | 5.617 | 3.880 | 18.551 | 710.728 | 4.310 / 4.980 | 21/24 | 33/33 | 1.175 | 11613117440 | 32256 |
| turbo | text | c2 | 4.666 | 3.089 | 15.936 | 709.281 | 1.021 / 1.633 | 16/24 | 33/33 | 1.171 | 11613117440 | 32256 |

C2: mejora relativa en bloques 14.61 %. Luma mediana Turbo de texto +1.633 niveles RGB8 frente al límite de +2,5. Text Check combinado 62/66. Diferencias de L1@64 en puntos porcentuales (C2 menos ffn30): {'base_grid': -1.3023865145016336, 'base_text': -0.5030786186002176, 'turbo_grid': -0.9220728656045756, 'turbo_text': -1.204589311853213}. Igualdad del pico asignado en las cuatro comparaciones: True. Los tiempos son medianas tras descartar la primera imagen; sus variaciones se publican sin imponer un margen a posteriori.

- base c1: 3,955,670,084 B, SHA-256 3cd21fd1d2ac80403625ecddacc3265c1ad9139a565627cceddda423987ad834
- base c2: 3,955,670,084 B, SHA-256 ad00abc18f0d7e628fed7ede5d782a5775e91e03d555091fbf86dc7a2eaa414a
- turbo c1: 3,955,670,236 B, SHA-256 78aecfc1318a55a49142a17023e8dc2bcb0d479752f3e1eaeb4b50d536252bee
- turbo c2: 3,955,670,236 B, SHA-256 0bb80e39537c086402d9b28ff6e89bba597a51f65b7bacc7e5114e7c6f92893c

Comandos: phase6_calib/run01/*.ps1, c2_pilot01/stage.ps1, c2_full01/stage.ps1 y visual_c1_01/stage.ps1, visual_c2_01/stage.ps1 (estos cuatro bajo phase6_calib). Cada score/cases.csv registra métricas por PNG, text_check.json las lecturas y runs.json los tiempos y recuentos. Modelo y referencias tienen el mismo encabezado tensorial por familia. Evidencias y hashes: phase6_calib/evidence_hashes.csv, phase6_calib/model_inventory.json y phase6_calib/decision.json. No se ha realizado revisión humana de C1 ni C2. Resultados limitados a los grafos y semillas registrados.

Decisión: mantener ffn30 como referencia HMS v1. No se activa la revisión ciega con los criterios acordados. Conservar C1/C2 como experimentos.

### HMS v1 C2 text, 10 seeds (2026-11-22)

Follow-up Base text evaluation on 10 fresh seeds (`2001–2010`) across the 8 prompts of `phase5_text/prompts_text.txt` (11 expected strings per seed = 110 texts per model; 240 images total in `phase7_text_seeds/{base_bf16,base_ffn30,base_c2}`; 1024x1024, 25 steps, CFG 2.5, `er_sde/simple`, Base SLG, hybrid Qwen, Flux2 VAE), scored with MOSS Text Check (`tools/moss_text_score_fails.py`):

| Model | Exact | Close | Missing | Outside box | Non-exact readings |
|---|---:|---:|---:|---:|---|
| `BF16` | 107 | 1 | 2 | 0 | `p00_s2006` (`FRESH BREAD DAIYY`), `p03_s2002` (`COFFEE LUGUT`), `p04_s2007` (`THE LAST`) |
| `ffn30` | 107 | 1 | 2 | 0 | `p02_s2010` (`OPEN 24`), `p03_s2002` (`COFFEE LVER`), `p04_s2007` (`LAST`) |
| `C2` | 102 | 7 | 1 | 0 | `p00_s2007` (`FRESH BREAD DALEY`), `p02_s2010` (`OPEN HOURS`), `p03_s2010` (`COFFEE TEA 2.50`, `TEA 2.50`), `p04_s2002` (`THE LAST LOTERN#FT# NIGHT`), `p04_s2004` (`A.ROSSI`), `p04_s2007` (`THE LAST THEN`), `p05_s2001` (`AIN STREET`) |

`ffn30` matches `BF16` (`107/110` exact), while Base `C2` drops to `102/110` (`131/143` combined across `1001–1003` and `2001–2010` vs `139/143` for `ffn30` and `140/143` for `BF16`). Nine side-by-side comparison sheets (`BF16 | ffn30 | C2`) for all non-exact cases are saved in `phase7_text_seeds/fails/`. Evidence: `phase7_text_seeds/text_summary.json`, `phase7_text_seeds/text_details.csv`.

### HMS v1 C2t (2026-11-22)

Calibrated GPTQ (`C2t`) using the four C2 scenes (`1, 21, 3, 23`) plus three quoted-text prompts (`16, 29, 37` from `prompts_40_variados.txt`, excluded from all evaluation sets) across all 42 blocks for Base and Turbo (`phase8_c2t/models/moss_base_hms_v1_c2t.safetensors`: 3,955,670,084 B, SHA-256 `81632cf79adc26c4dc1f86a13f9cc94bd0cb2442ec29a3fe5616d0cf2f69da89`; `moss_v1_turbo_hms_v1_c2t.safetensors`: 3,955,670,236 B, SHA-256 `2a984578776d38f0f12a2dde3b9a19df2a1689c896307ba5a26a2f7c55dee5c8`), keeping the exact `ffn30` layout, C1 grids, and `hades/recipe` metadata.

| Family | Series | Variant | L1@64 % | MAE /255 | Var. Lap. | Δ luma mean / median | Text Check | Warm s | Peak VRAM B | Ops |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Base | 10-scene grid | `C2t` | 4.130 | 13.003 | 547.92 | +1.444 / +0.490 (8/10) | 2/2 | 7.180 | 11,613,117,440 | 83,480 |
| Turbo | 10-scene grid | `C2t` | 5.160 | 18.379 | 1544.95 | -0.168 / +0.074 (5/10) | 2/2 | 1.308 | 11,613,117,440 | 13,440 |
| Base | text `1001–1003` | `C2t` | 8.035 | 22.353 | 96.76 | -6.422 / -2.343 (7/24) | **33/33** | 7.314 | 11,613,117,440 | 200,352 |
| Base | text `2001–2010` | `C2t` | 6.496 | 19.023 | 185.18 | -2.844 / -0.552 (33/80) | **106/110** | 7.168 | 11,613,117,440 | 667,840 |
| Turbo | text `2001–2010` | `C2` | 5.029 | 17.010 | 701.05 | +1.649 / +1.738 (56/80) | **108/110** | 1.251 | 11,613,117,440 | 107,520 |
| Turbo | text `2001–2010` | `C2t` | 6.125 | 20.106 | 674.18 | +0.433 / +0.307 (48/80) | **108/110** | 1.250 | 11,613,117,440 | 107,520 |

Gates (`TAREAS_C2T.md`):
- Base text combined (`1001–1003` + `2001–2010`, 143 texts): **`C2t` 139/143 exact** (vs `C2` 131/143, `ffn30` 139/143, `BF16` 140/143) — **PASS** (`>= 138/143`).
- Base 10-scene fidelity: `L1@64` `4.130%` vs `C2` `4.024%` (`+0.106` pp <= `+0.3` pp) and median luma `+0.490` within `±2.5` — **PASS**.
- Turbo text (`2001–2010`, 110 texts): `BF16` 108/110, `C2` 108/110, `C2t` 108/110 — **PASS** (`>= BF16 - 2`).

Sixteen 4-column failure sheets (`BF16 | ffn30 | C2 | C2t`) for Base non-exact cases are saved in `phase8_c2t/fails/` (plus 2 Turbo sheets in `phase8_c2t/fails_turbo_200x/fails/`). Evidence: `phase8_c2t/{base_grid_c2t_score,turbo_grid_c2t_score,base_text_100x_score,base_text_200x_score,turbo_text_c2_200x_score,turbo_text_c2t_200x_score}/summary.json` and `phase8_c2t/fails_*/text_summary.json`.

### HMS v1: decision (2026-11-22)

Human review (HaDeS, blind, `hms_c2t_base_parejas_1024.zip` and `hms_c2t_turbo_parejas_1024.zip`, ten scenes each against BF16, keys
unopened): "practically equal"; C2t judged the clear winner over the earlier candidates. **HMS v1 is fixed as C2t:** the `ffn30` layout
(attention W6A8, FFN of blocks 30-41 W6A8, FFN of blocks 0-29 W4A8, all ComfyUI / comfy-kitchen native formats, loaded by the stock
`UNETLoader`) with GPTQ in the ConvRot basis on fixed C1 grids, calibrated on prompts 1, 21, 3, 23 (scenes) and 16, 29, 37 (quoted text)
of `prompts_40_variados.txt`. Base `moss_base_hms_v1_c2t.safetensors` 3,955,670,084 B (SHA-256 `81632cf7...da89`), HaDeS MOSS v1.0 Turbo
`moss_v1_turbo_hms_v1_c2t.safetensors` 3,955,670,236 B (SHA-256 `2a984578...e5c8`); `hades/recipe` metadata names the method and the
calibration prompts. Summary against BF16 on this protocol (RTX 5090): Base L1@64 4.130 % (W6A8 4.053 %), luma median +0.49, Laplacian
547.9 (BF16 514.2), text 139/143 (BF16 140/143); Turbo L1@64 5.160 %, luma median +0.07, text 108/110 (= BF16); about 2x faster than
BF16 and 11.6 GB peak allocated VRAM instead of 17.8 GB. Lesson: calibration without text prompts lost text (C2 131/143 in Base);
the calibration set must include quoted text. Not yet measured: transfer of the recipe to finetunes and merges, other GPUs, the light
pack (HMS + hybrid text encoder) against the official files.

Reproduction (the lab's `phase8_c2t/run_all.ps1`, condensed): `moss_native_mix_convert.py --source <BF16> --protected-ffn-blocks 30-41`
(the `ffn30` template), `moss_codec_capture.py --hms-diagonal` then `--hms-rows` on the calibration prompts, `moss_hms_calibrate.py`
(C1 scales and codebook), `moss_hms_gptq.py` (GPTQ shards), `moss_hms_assemble.py` (the final checkpoint with `hades/recipe`);
evaluation with `moss_quant_grid.py`, `moss_hms_score.py` and `moss_text_eval_batch.py`.

### HMS v1 on SFT and Bulle d'Encre (2026-11-22)

Applied the exact `HMS v1` (`C2t` recipe: `ffn30` layout + `C1` + `GPTQ` with the 7 calibration prompts `1, 21, 3, 23, 16, 29, 37` from `prompts_40_variados.txt`, `--expected-scenes 7`) to `mossImage_sft_bf16.safetensors` (`moss_sft`, 25 steps, CFG 2.5, `er_sde/simple`, no SLG) and `HaDeS_MOSS_preview_beta.safetensors` (`bulle`, 15 steps, CFG 2.5, `euler/simple`, SLG), evaluating each against its own BF16 reference on the 10 `phase1` scenes (seed 1001) and 8 text prompts × 10 seeds (`2001–2010`, 110 expected texts):

- `phase9_sft_bulle/models/moss_sft_hms_v1.safetensors`: **3,955,670,084 B** (SHA-256 `a33dc49aa831a9451085ec3ac0a9ab6fdde0aba9a1eb2860d37b73864b5a0c79`).
- `phase9_sft_bulle/models/bulle_hms_v1.safetensors`: **3,955,670,404 B** (SHA-256 `c936f1c159c1cbec8396e6f377dc55e7c66179db23a6443106301c49ce4ee572`).

| Model | Series | L1@64 % | L1@16 % | MAE /255 | Var. Lap. (HMS / BF16) | Δ luma mean / median | Text exact (HMS / BF16) | Warm s | Peak VRAM B | Ops |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| `moss_sft` | 10-scene grid | 4.723 | 3.272 | 15.501 | 2003.35 / 2079.49 | +1.680 / +0.699 (8/10) | 2/2 | 7.296 | 11,613,117,440 | 84,000 |
| `moss_sft` | text `2001–2010` | 5.934 | 4.163 | 18.371 | 819.44 / 872.23 | +1.029 / +0.536 (46/80) | **106/110** vs **105/110** | 7.217 | 11,613,117,440 | 672,000 |
| `bulle` | 10-scene grid | 4.542 | 2.852 | 17.734 | 1595.01 / 1573.50 | +0.474 / -0.139 (5/10) | 2/2 | 4.521 | 11,613,117,440 | 50,080 |
| `bulle` | text `2001–2010` | 5.107 | 3.504 | 17.313 | 823.58 / 808.15 | +1.541 / +0.517 (47/80) | **110/110** vs **108/110** | 4.513 | 11,613,117,440 | 400,640 |

Both models pass all gates (`text >= BF16 - 2`, `|median luma| <= 2.5`; reference `L1@64` vs Base `4.130%` and Turbo `5.160%`), confirming that the `HMS v1` (`C2t`) recipe transfers cleanly to both `SFT` and `Bulle d'Encre` without tuning. Blind 1024x1024 pairs: `hms_sft_parejas_1024.zip` (SHA-256 `b64f43f44b4fb52c26ef588c334f814cd9cfd435d6e3f8185943f7004a8a8875`) and `hms_bulle_parejas_1024.zip` (SHA-256 `57ddebb262c7cf82bc0baa1f35c9ee0eaee6d077cc688e1e79961cb3e44d6eed`); keys kept unopened in `phase9_sft_bulle/{moss_sft,bulle}/blind_key_A.json`. Failure sheets (`BF16 | HMS`): `phase9_sft_bulle/{moss_sft,bulle}/text_fails/fails/`.

Human review (HaDeS, blind, both ZIPs, keys unopened): "excellent, both". HMS v1 is approved on all four models (Base, SFT, HaDeS MOSS v1.0 Turbo, Bulle d'Encre). Note: Bulle d'Encre was evaluated with SLG (the graph came from Base's); BF16 and HMS used the same graph, so the comparison holds, but its usual settings have no SLG. SFT and Bulle d'Encre used C1 on all seven calibration prompts (`moss_hms_calibrate.py --expected-scenes 7`); Base and Turbo used C1 on the first four and GPTQ on seven.

### Model Saver: HMS v1 precision (0.9.5, 2026-11-22)

Validation of the Model Saver's HMS v1 export running inside a real ComfyUI instance (clean server on port 8189, `--use-sage-attention`) through the API (`phase10_saver/validate_real.py`). The exported file is bit-identical to the lab's SFT HMS v1 file (942/942 tensors).

**Saver performance (server peak resources):**

| Job | Save time | Peak RAM | Peak VRAM | Target file |
|---|---:|---:|---:|---|
| `base_hms` | 560 s | 19,461 MiB | 24,599 MiB | `base_hms_00001_.safetensors` |
| `merge_hms` | 494 s | 23,367 MiB | 26,083 MiB | `merge_hms_00001_.safetensors` |
| `merge_bf16` | 28 s | 30,051 MiB | 23,669 MiB | `merge_bf16_00001_.safetensors` |

**Quality metrics (offline scoring against respective BF16 references):**

| Model / Merge | Series | L1@64 % | L1@16 % | MAE /255 | Var. Lap. (HMS / BF16) | Δ luma mean / median | Text exact (HMS / BF16) | Warm s | Peak VRAM B | Ops |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| `base_hms` | 10-scene grid | 4.122 | 3.060 | 13.134 | 876.23 / 514.16 | +2.329 / +1.772 (7/10) | 2/2 | 7.633 | 11,613,117,440 | 83,480 |
| `base_hms` | text `2001–2010` | 6.723 | 5.288 | 19.594 | 199.41 / 187.09 | -3.358 / -1.307 (30/80) | **109/110** vs **107/110** | 7.634 | 11,613,117,440 | 667,840 |
| `merge_hms` | 10-scene grid | 4.492 | 3.211 | 14.862 | 1096.50 / 1735.17 | -2.341 / -2.782 (2/10) | 2/2 | 7.917 | 11,613,117,440 | 84,000 |
| `merge_hms` | text `2001–2010` | 5.939 | 4.216 | 17.965 | 701.90 / 742.77 | -1.152 / -2.151 (25/80) | **107/110** vs **108/110** | 7.669 | 11,613,117,440 | 672,000 |

Notes:
- `base_hms` is scored against the lab's `phase1/bf16_01` (10 scenes) and `phase7_text_seeds/base_bf16` (text seeds).
- `merge_hms` is scored directly against `merge_bf16` saved in the same session.
- On `merge_hms`, scene 27 (4-panel manga page) exhibits a composition shift (MAE 40.77, L1@64 10.51%) rather than blur, preserving sharp inked line art.
- Not hidden: on the merge (SFT with half of Base in blocks 18-35, the sixth video's test recipe) the 10-scene median luma (-2.78) is just past the lab's +-2.5 gate and the grid's mean Laplacian drops 1735 -> 1097, mostly scene 27's composition shift (12586 -> 6733 in the first run; the other scenes lose 5-25 %). Text stays within the gate (107/110 vs 108/110). The four published HMS v1 models (Base, SFT, v1.0 Turbo, Bulle d'Encre) all pass every gate; an HMS file of a merge should be checked by eye against its BF16 save.
- UI (frontend 1.57.4, clean 8189): the Saver with `HMS v1` selected shows the `clip` input and `hms_steps`, `hms_cfg`, `hms_sampler`, `hms_scheduler` with their defaults in the classic canvas with HaDeS widgets, with the standard-widgets option and in Nodes 2.0 / Vue. Not checked: zoom, manual resize and save / reload of a workflow with the new inputs.

## MOSS Control (color / depth) on Base, SFT and Turbo (2026-11-24, for the ninth video tutorial)

Clean ComfyUI on port 8189 (`--use-sage-attention`, RTX 5090), `qwen3.5_9b_int8_convrot`, `flux2-vae`, 1024 x 1024. Adapter: `moss-color-depth-control.safetensors` (`darask0/MOSS-Image-color-depth-control`, trained on Base, loaded via `MOSS Multi-LoRA` at strength 1.0). Depth map extracted from `fisherman_video1.png` with `DepthAnythingV2Preprocessor` (`depth_anything_v2_vitl.pth`, resolution 1024). Prompt (knight scene, keeping the fisherman's pose): *"Oil painting of a medieval knight in weathered steel armor mending a chainmail cloak on a stone castle parapet at sunset, warm torches, sea in the background."* Scripts, timings and contact sheets in `I:/HaDeS_MOSS/videos/controlnet/measure`.

- **Base vs SFT and Turbo 8-steps** (`color_mode = off (depth only)`, `strength 1.0`, `0.0–1.0`; timings are `execution_start` → `execution_success` median of 3 warm runs on seeds 1020–1022; visual inspection on seeds 1013–1015):

| Config | Steps / CFG / Sampler | Warm times (s) | Median (s) | By eye (pose and prompt) |
|---|---|---|---|---|
| `mossImage_base_bf16` | 25 / 2.5 / `er_sde` / `simple` | 14.12, 14.14, 14.21 | 14.14 | Clean; follows hands, torso, head tilt and cloak pose |
| `mossImage_sft_bf16` | 25 / 2.5 / `er_sde` / `simple` | 14.19, 14.21, 14.51 | 14.21 | Clean; same pose fidelity, slightly warmer contrast / cloak |
| `Base` + `HaDeS-MOSS-Turbo-8steps-v1.0` | 8 / 1.0 / `euler` / `simple` | 3.30, 3.29, 3.28 | 3.29 | Clean (4.3× faster); keeps depth pose and armor detail |
| `SFT` + `HaDeS-MOSS-Turbo-8steps-sft-v1.0` | 8 / 1.0 / `euler` / `simple` | 3.29, 3.24, 3.26 | 3.26 | Clean (4.4× faster); keeps pose and warm sunset palette |
| `SFT` + `HaDeS-MOSS-Turbo-8steps-v0.7` | 8 / 1.2 / `euler` / `simple` | 5.28, 5.30, 5.31 | 5.30 | Clean; CFG > 1 runs two passes per step |

- **`color_mode`, `end_percent` and `strength`** (`Base` + `Turbo v1.0`, 8 steps, CFG 1.0, `euler / simple`, seeds 1013–1014; `color_map` connected to `fisherman_video1.png`; MAE / 255 at 512 x 512 against the input photo):

| Setting | MAE vs input (1013 / 1014) | By eye |
|---|---|---|
| `off (depth only)`, `strength 1.0`, `end 1.0` | 57.94 / 61.98 | Closed-visor knight helmet; colors and textures follow the prompt |
| `flat regions` (`8` colors, `blur 4`), `end 1.0` | 53.58 / 59.04 | Guides light/dark regions and warm-left / sea-right layout without copying the face |
| `as given` (photo on `color_map`), `end 1.0` | 48.45 / 58.08 | Copies photo texture (seed 1013 opens the visor and reproduces the fisherman's face/beard); meant for flat segmentation maps, not photos |
| `off (depth only)`, `end 0.75` | 58.39 / 62.46 | Keeps pose while loosening fine contour adherence in the last steps |
| `flat regions`, `end 0.75` | 53.73 / 59.24 | Same overall layout with slightly softer adherence to the input regions |
| `off (depth only)`, `strength 0.7`, `end 1.0` | 59.69 / 60.59 | Shifts torso/head alignment slightly away from the depth map |

- **Caption + Control + Telemetry flow** (`Base` + `Turbo v1.0`, `MOSS Prompt Enhancer` in `caption` mode with `fisherman_video1.png` connected, `flat regions`, seed 1013): 13.33 s on the first run (including Qwen vision encoding). `Suite Telemetry` (`Loader`, `Multi-LoRA`, `Prompt Enhancer`, `Latent`, `Control`, `KSampler`) reports `No findings: the connected nodes are consistent with what was measured.` Not measured here: `moss_image_sft_pose_control_lora`, non-square aspect ratios, and multi-resolution ControlNet passes.

## Img2Img: `denoise_mode` `core` vs `exact` on the tutorial fisherman (2026-11-24, for the tenth video tutorial)

Clean ComfyUI on port 8189 (`--use-sage-attention`, RTX 5090), `mossImage_base_bf16`, `qwen3.5_9b_int8_convrot`, `flux2-vae`, `HaDeS-MOSS-Turbo-8steps-v1.0` (8 steps, CFG 1.0, `euler / simple`, 1024 x 1024), input `fisherman_video1.png` through `VAEEncode`. Prompt: *"Impressionist oil painting of an old fisherman mending a net on a wooden dock at stormy night, glowing lantern light, thick impasto brushstrokes, dramatic dark clouds."* Scripts, timings and contact sheet in `I:/HaDeS_MOSS/videos/img2img/measure`.

| Denoise | `core` `start_sigma` | `core` median (s) | `core` MAE / 255 (1013 / 1014 / 1015) | `exact` `start_sigma` | `exact` median (s) | `exact` MAE / 255 (1013 / 1014 / 1015) |
|---:|---:|---:|---|---:|---:|---|
| 0.5 | 0.8333 | 3.06 | 14.24 / 16.26 / 17.20 | 0.8333 | 3.15 | 14.30 / 16.43 / 17.10 |
| 0.6 | 0.8891 | 3.06 | 16.33 / 20.17 / 21.32 | 0.8824 | 3.14 | 15.82 / 19.93 / 20.96 |
| 0.7 | 0.9305 | 3.47 | 17.09 / 27.77 / 31.91 | 0.9211 | 3.15 | 16.87 / 25.55 / 28.77 |
| 0.8 | 0.9524 | 3.27 | 26.92 / 32.99 / 37.07 | 0.9524 | 3.24 | 27.26 / 32.75 / 36.54 |
| 0.9 | **1.0000** | 3.13 | 78.32 / 85.75 / 87.03 (ignores input) | **0.9783** | 3.18 | 59.33 / 72.05 / 62.48 (keeps centered subject) |

- Consistent with `MOSS KSampler denoise_mode: core against exact on img2img` (2026-11-12): up to `0.8`, both modes preserve the fisherman in the foreground (adding the glowing lantern and storm clouds at `0.8`). At `0.9`, `core` starts at `1.0000` (pure noise, MAE 78.3–87.0) and replaces the composition with a distant figure, whereas `exact` starts at `0.9783` and keeps the centered fisherman on the dock with the lantern and night sky.

## HMS v1 vs BF16 live comparison on Base (2026-11-25, for the eleventh video tutorial)

Clean ComfyUI on port 8189 (`--use-sage-attention`, RTX 5090), `qwen3.5_9b_int8_convrot`, `flux2-vae`, 1024 x 1024, 25 steps, CFG 2.5, `er_sde / simple`, seed 1013–1014. Compared `mossImage_base_bf16.safetensors` (`10.2 GB`) against `MOSS-Image_Base_HMS_v1.safetensors` (`3.96 GB`, `3,955,670,084 B`) through `UNETLoader` and `MOSS Compare Sampler`. Scripts and sheets in `I:/HaDeS_MOSS/videos/hms_v1/measure`.

- **Single-model sampling times** (warm runs, seeds 1013 / 1014): `BF16` **13.83 s / 13.83 s**; `HMS v1` **6.22 s / 6.24 s** (**2.22× faster**).
- **Fisherman similarity** (`MAE / 255` at 512 x 512 between BF16 and HMS v1): **9.0** (seed 1013) and **8.0** (seed 1014).
- **Compare Sampler sheets** (both models in one node, seed 1013): fisherman `21.00 s`; harbor sign (`"HARBOR COFFEE"` / `"OPEN DAILY"`) `21.64 s`, every word spelled right in both `BF16` and `HMS v1`.
