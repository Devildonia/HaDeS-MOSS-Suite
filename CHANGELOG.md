# Changelog

All notable changes to the **HaDeS MOSS Suite** are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed
- **Nodes 2.0: a link dropped on a styled (HaDeS) field connects.** In the Vue node view a link dragged onto a styled row (for example the Prompt Enhancer's `width`) went nowhere; only the standard widgets took it. Each styled row now carries its hidden backing input, as the classic canvas already did. Checked by dragging in the classic canvas (HaDeS and standard widgets) and in Nodes 2.0 (both).
- **Caption mode keeps the canvas size.** With `caption` / `detailed` / `as written`, the Prompt Enhancer's `width` / `height` (typed or linked, e.g. from MOSS Latent) did not reach the Layout Compiler, whose latent stayed 1024x1024. The caption draft now carries `width` / `height`; the Compiler uses them, and the Layout Editor's NEW LAYOUT FROM THIS CAPTION starts on that canvas. A caption draft without a size (older Suites, written by hand) still gives 1024x1024. Measured: MOSS Latent 16:9 at 1 MP linked into the Enhancer gives a 1360x768 latent in caption mode (was 1024x1024) and in layout mode (unchanged).

## [0.9.5] - 2026-11-23

### Added
- **Model Saver: HMS v1 precision.** Saves any MOSS-Image model (merges included) in HMS v1 with the lab's recipe (C1 + GPTQ calibrated on the model itself with the seven fixed prompts; needs the `clip` input). Bit-identical to the lab's SFT HMS v1 file (942/942 tensors); in a real ComfyUI on an RTX 5090, 8-10 minutes per model; MOSS-Image Base in HMS L1@64 4.12 % from BF16, text exact 109/110 (BF16 107/110); a merge lands slightly darker than its BF16 save (`tools/RESULTS.md`, "Model Saver: HMS v1 precision"). Base and SFT in HMS v1 on Hugging Face: [DeVilDoNia/MOSS-Image-HMS](https://huggingface.co/DeVilDoNia/MOSS-Image-HMS).
- **HMS v1 quantization recipe (tools, not in the Registry package):** MOSS-Image Base and HaDeS MOSS v1.0 Turbo in about 3.96 GB with the stock `UNETLoader` and ComfyUI's native W4A8 / W6A8 kernels: attention and the FFN of blocks 30-41 in W6A8, the other FFN in W4A8, GPTQ-calibrated on scenes and quoted text. Measured against BF16 (`tools/RESULTS.md`, "HMS v1: decision"): Base L1@64 4.13 % (W6A8 4.05 %), text 139/143 (BF16 140/143); Turbo text 108/110 (= BF16); about 2x faster and 11.6 GB peak VRAM instead of 17.8 GB; judged practically equal to BF16 in a blind review. Tools: `moss_native_mix_convert.py`, `moss_codec_capture.py`, `moss_hms_calibrate.py`, `moss_hms_gptq.py`, `moss_hms_assemble.py`, with the evaluation helpers and the research tools of the experiment (native baselines, rejected codecs).

## [0.9.4] - 2026-11-22

### Fixed
- **Model Block Merger and Multi-Model Merger, a saved merge's recipe left out a model loaded with Load Diffusion Model** (or another core loader): only a model from the MOSS Loader was named, so SFT merged with Base from Load Diffusion Model was saved with a recipe naming SFT alone and the Metadata Reader showed only SFT. A merger now follows the cable of an input that carries no Suite recipe up the workflow to its file, as the MOSS KSampler does since 0.9.3, and records it like a MOSS Loader's model. Checked in a clean ComfyUI: the test recipe saved by the Model Saver and read back names `mossImage_sft_bf16` and `mossImage_base_bf16`.
- **MOSS Compare Sampler and Compare Sheet labelled a model from Load Diffusion Model as "unknown model"** (or by its slot letter): the Compare Sampler now follows a model's cable up the workflow to its file (through LoRA nodes and mergers) when the model carries no Suite recipe, and the Compare Sheet takes the model from the KSampler's traced model files when the branch has no MOSS Loader. Checked in a clean ComfyUI: SFT, Base, a merge and Bulle d'Encre read `mossImage_sft_bf16`, `mossImage_base_bf16`, `merge: ...` and `HaDeS_MOSS_preview_beta`; a Compare Sheet of two Base seeds `mossImage_base_bf16 · seed 1013` / `· seed 1014`.

### Changed
- **Layout from a Sentence example**, rebuilt for the eighth video tutorial: the fisherman poster on HaDeS MOSS v1.0 Turbo (was a night street on SFT), the title "NORTH SEA" and the signature "HaDeS" in quotes, enhancer seed 2, image seed 1013, 8 steps, CFG 1, with MOSS Text Check's report on a text preview and HaDeS MOSS v1.0 Turbo's download link. Run in a clean ComfyUI with the measured draft: pixel-identical to the video. Measured (`tools/RESULTS.md`, "Layout and Text Check on a poster"): over 6 enhancer seeds the title, the fisherman and the signature in their region 6 / 6, the sun its own box 3 / 6; over 8 image seeds title, sun and signature in their boxes 8 / 8 and both texts `exact` 8 / 8; the sun moved to the horizon followed its box 8 / 8; Text Check 0.7-1.0 s.
- **Compare Models example**, rebuilt for the seventh video tutorial: SFT, Base, the sixth video's test recipe and Bulle d'Encre on one Compare Sampler sheet (the fisherman, seed 1013), a muted group with Bulle d'Encre's extracted LoRA at four strengths through four Multi-LoRAs, and a muted Compare Sheet with two seeds. Run in a clean ComfyUI: the sheet matches the video's (max pixel difference 1). Measured (`tools/RESULTS.md`, "Compare models and LoRAs"): each column is the MOSS KSampler's image; the sheet takes the same time and memory as four KSamplers; the LoRA's distance to Bulle d'Encre off 0.86, x0.5 0.74, x0.8 0.59, x1.0 0.26.
- **Merge, Save and Telemetry example**, rebuilt for the sixth video tutorial: a test recipe made for the video, not a recommendation (SFT with half of Base in layers 18-35), on the first video's fisherman (seed 1013), with the Model Saver taking its recipe from the merger's info, a Metadata Reader and a SLERP Multi-Model Merger (muted), and a note with the measurements. Run in a clean ComfyUI: the image is pixel-identical to the video's. Measured (`tools/RESULTS.md`, "Merge, save and telemetry: a test recipe"): the recipe lands between SFT and Base in the same render time; East Asian faces on two neutral prompts, SFT 10 / 16 and 16 / 16, the recipe ~3 / 16 and ~13-14 / 16, Base ~1-2 / 16 and ~6 / 16; the bf16 file (10.16 GB) renders the live merge's image, int8 ConvRot (5.41 GB) faster with small changes; SLERP 0.5 about the Block Merger at 0.5.

### Added
- **`tools/moss_video.py layout_text`:** the eighth tutorial (a poster laid out from a sentence, a box moved, the text read back).
- **`tools/moss_video.py compare`:** the seventh tutorial (four models on one sheet, a LoRA at four strengths, blind mode, the Compare Sheet).
- **`tools/moss_video.py merge_save`:** the sixth tutorial (the Block Merger, the bias counts, Telemetry, the Model Saver in bf16 and int8, the files loaded back, the Metadata Reader, the Multi-Model Merger).

## [0.9.3] - 2026-11-21

### Fixed
- **MOSS Image Saver, the Model field named the wrong model** when the KSampler's model did not come from the first MOSS Loader (a Loader kept only for its text encoder and VAE, a model merged in the workflow, a second stage): the field and its hash came from the first Loader in Telemetry. The MOSS KSampler now follows its `model` cable up the workflow, through LoRA nodes and mergers, to the loaders that read a file (MOSS Loader, Load Diffusion Model, Load Checkpoint) and records them in its `info`; the Saver names that file with its hash, or `Model: merge: a + b` with no model hash for a merge made in the workflow, and `%model%` follows. Older KSampler infos fall back to the first Loader. Checked in a clean ComfyUI: the LoRA Toolkit example now writes `Model: merge: mossImage_sft_bf16 + mossImage_base_bf16`, Basic Generation and Turbo 8 steps + Upscale x2 still `Model: mossImage_base_bf16` with its hash. Telemetry's report shows it as `model=a + b`.
- **MOSS Image Saver, a LoRA used in two stages was listed twice** in `Lora hashes` (the turbo on the generation and on the upscale pass); each file is listed once.

### Changed
- **LoRA Toolkit example**, rebuilt for the fourth video tutorial around Bulle d'Encre (was a placeholder LoRA and finetune): Metadata Reader on Bulle d'Encre, its base rebuilt with the Block Merger, the Extractor at rank 64, the Multi-LoRA with the extracted LoRA and a Turbo 8-steps v1.0 row at 0.5, the Merger and the Resize, with Bulle d'Encre's download link and a note giving the steps in order. Run in a clean ComfyUI as it opens and with the Multi-LoRA on: both images are pixel-identical to the measured ones. Measured (`tools/RESULTS.md`, "The LoRA toolkit on Bulle d'Encre"): rank 64 (252 MB, 97.0 %) brings Bulle d'Encre back at 0.19 of a seed-to-seed distance (rank 32 0.33, rank 128 0.22); with the Turbo row at 0.5, 8 steps / CFG 1 keep the scene and look in 2.2 s against 8.0 s; layers 24-29 and 30-35 each carry about three quarters of the change; the merged file is A + 0.5 B within 0.24 % per layer; rank 32 resize (126 MB) stays within the noise, dynamic 0.90 (40 MB) drifts.
- **Prompt Enhancer example**, rebuilt for the third video tutorial: MOSS-Image Base with Turbo 8-steps v1.0 (was SFT at 25 steps with a bookshop idea), the first video's fisherman in `detailed` mode with "a photograph of", its own enhancer seed, Suite Telemetry reading the turbo too, and a note with every mode's measurements. Run in a clean ComfyUI: the image is pixel-identical to the video's. Measured (`tools/RESULTS.md`, "Every mode on one short idea"): caption 56-118 words, detailed 168-238; with no medium in the idea detailed chose an oil painting in 8 / 9 seeds, and "a photograph of" gave 6 / 6 photographs; the negative `hat` made the enhancer write "no hat" and 0 / 6 images had one (5 / 6 without it); an answer takes 1.6-2.5 s. The note also explains corrections: the detailed prompt pasted back as the idea with "Corrections: ..." at the end (measured: a hat and a cat 6 / 6 with the rest of the scene kept; "a young man" kept "white beard" in 5 / 6 texts and about 3 / 6 images looked younger, naming the beard too gave 6 / 6; a Spanish note applied in 6 / 6 texts, written in English).
- **Turbo 8 steps + Upscale x2 example**, rebuilt for the second video tutorial: MOSS-Image Base with Turbo 8-steps v1.0 (was SFT with v0.7) on the first video's fisherman (seed 1013); the MOSS pass is on, all three images go through the MOSS Image Saver with Suite Telemetry's metadata, and Telemetry reads both samplers and the turbo. Run in a clean ComfyUI: the 1024, x2 and x2 + MOSS pass images are pixel-identical to the video's. Measured on Base + v1.0 (`tools/RESULTS.md`): 2.9 s at 1024, 1.1 s for RealESRGAN x2, 10.7-11.1 s for the pass at 2048 (denoise 0.25); by eye on 100 % crops the pass adds skin pores, beard hairs and sharper net threads, and 0.40 redraws more.

### Added
- **Metadata example** (fifth video tutorial): the Turbo 8 steps + Upscale x2 workflow with a Generate stage and a Refine stage Telemetry on the Savers' `primary_stage_metadata`. Run in a clean ComfyUI: the 1024 and x2 PNGs carry the generation's settings, the refined PNG `Size: 2048x2048, Denoising strength: 0.25`, each with no note and with `hades_moss` and `hades_moss_primary_stage`. Measured (`tools/RESULTS.md`, "What the MOSS Image Saver writes"): the first save with hashes 5.7 s for the 10 GB model, the next 0.06 s; what `hashes` off and `embed_workflow` off leave in the PNG.
- **`tools/moss_video.py metadata`:** the fifth tutorial (the PNG's text read back on camera and lit field by field, the PNG dropped back into ComfyUI, two stages, a merge, privacy). The overlay's `show()` no longer loses its layer to a `hide()` issued just before.
- **`tools/moss_video.py lora_toolkit`:** the fourth tutorial (the whole toolkit from an empty canvas, the 100 s extraction fast-forwarded; the 3 images on camera are pixel-identical to the measured ones). Recording now turns the node search's preview off and sets a new node's values the moment it is created, so no default file name from the user's folders is drawn.
- **`tools/moss_video.py prompt_enhancer`:** the third tutorial (every Prompt Enhancer mode on the fisherman: as written, caption, detailed, random prompts, the negative, tags an image as input and corrections after the image; the 11 images on camera are pixel-identical to the measured ones). The recording ComfyUI gets its own `--input-directory` (`comfy_input`), so no file of the user's input folder can appear in a Load Image list.
- **`tools/moss_video.py upscale_telemetry`:** the second tutorial (the first video's workflow rebuilt with the mechanical parts fast-forwarded, Suite Telemetry and its report, RealESRGAN x2 and the MOSS pass, with 100 % crops and the times from ComfyUI's history). `--captions none` records without burned captions and writes only the EN / ES .srt files.

## [0.9.2] - 2026-11-20

### Added
- **Basic Generation example:** the six nodes of the first video tutorial (Base at 25 steps, then Turbo 8-steps v1.0 at 8 in the `turbo` input), with a note; made by `tools/make_examples.py` with the official file names and download links. Suite Telemetry feeds the Image Saver's `metadata`, so the PNG carries the prompt, settings and model / LoRA hashes as A1111 parameters (the graph built in the video leaves `metadata` empty and its PNG has none). Run in a clean ComfyUI: the 8-step image is pixel-identical to the video's, and with the turbo node switched off the KSampler's own 25 steps run.
- **`tools/moss_video.py basic_generation`:** builds that workflow on an empty canvas (node search by double-click, a click drops the new node, wires dragged slot to slot, list widgets set by code so no file list opens), runs it at 25 and 8 steps and shows both images and 100 % crops with the times ComfyUI's history recorded (13.4 s and 3.2 s on an RTX 5090). `--resolution 1440p` records a 2560 x 1440 page (Chrome's screencast ignores the device scale), with the cursor, overlays and captions scaled to match; `--until N` stops a test take after line N.

### Fixed
- **The red frame of a missing model stays after picking the right file in a Suite widget:** the Suite's widgets set the value and called its callback but not `node.onWidgetChanged`, the hook from which the ComfyUI frontend clears a node's errors, so the frame stayed until a reload (running the workflow did not clear it either). `setWidgetValue` now calls it as LiteGraph does for its own widgets. Checked in a clean ComfyUI with the Basic Generation example (official file names, local files named differently): the Loader's frame (3171 red pixels) is gone after the pick, while a value set without the hook keeps its frame; a widget test covers the call.

## [0.9.1] - 2026-11-20

### Added
- **CI: the published package checked as a user gets it** (suggested by audit 0.8.1): `comfy node pack` builds the zip, `tests/check_package.py` checks its files and loads it in a clean ComfyUI (no git checkout), which must register exactly the distributed nodes, serve their frontend, log no import failure and know every example's nodes and widget counts. Run locally on the 0.9.0 tree, with a stand-in for comfy node pack (git-tracked files minus .comfyignore) and a clean CPU ComfyUI 0.43.0: 37 files, 23 nodes, 4 frontend files, 10 examples, all checks passed.

### Fixed
- **MOSS Compare Sheet / Compare Sampler memory** (audit 0.8.1 #9): the sheets of a batch are built, saved and copied into the output one at a time. Measured with 4 panels, batch 4 at 1344 x 1792: the peak RAM added falls from 1 112 MB to 555 MB, of which 475 MB is the `IMAGE` output itself; the output and the saved PNGs are bit-identical (`tools/RESULTS.md`).
- **MOSS Layout Editor, an unfinished box:** a box without its caption (drawn on the canvas, outlined in red) stopped the editor itself, so not even DRAFT LAYOUT could bring a new draft to replace it. The editor now runs, says in `info` what is missing and still offers `draft_in`; the Layout Compiler refuses to generate from it, and its message names the box: *box 5 in the Layout Editor: select it and write its caption, or delete it*.
- **MOSS Layout Editor, picking a box:** a click takes the smallest box under the pointer (the topmost of equal sizes). A box covering the whole canvas, drawn on top, made the boxes inside it impossible to select, move or resize; it is now picked where no other box is.
- **MOSS Text Check with a caption prompt:** a caption draft (the Prompt Enhancer in `caption` / `detailed` mode) on `layout` stopped the workflow with *nothing to check*. Its texts in quotes (`a sign that reads "OPEN"`) are now the texts looked for; with no text asked for at all, the node says so, `passed` is true and the workflow goes on.

## [0.9.0] - 2026-11-19

### Added
- **MOSS Layout Editor: `DRAFT LAYOUT` and `GENERATE` buttons** under the canvas. `DRAFT LAYOUT` (shown when `draft_in` is connected) runs only the nodes the editor needs, e.g. the Prompt Enhancer in `layout` mode, without the sampler, and the boxes it writes replace the canvas at once; move, resize or delete them, then `GENERATE` runs the whole workflow with them (the same as Run). Any other run still only offers `draft_in` with `IMPORT`. The editor is now an output node, which ComfyUI needs for that partial run (the same as Run selected).
- **Example workflows:** `Layout from a Sentence` (Prompt Enhancer layout → DRAFT LAYOUT → Layout Editor → GENERATE → Text Check) and `Compare Models` (Compare Sampler, with a muted Compare Sheet group). Sampler Metadata is in the Turbo example and LoRA Block Weight (muted) and Metadata Reader in the LoRA Toolkit, so every node of the pack is in an example.

### Changed
- **Behaviour an existing workflow may notice:** MOSS Compare Sampler's `images` output now carries every image of every batch (it gave one per model); MOSS Text Check no longer passes an empty read (a workflow that relied on `passed` can now stop there, rightly); a Compare Sheet footer leaves out what not every panel carries; MOSS Model Saver `nearest` files hold the weights of a whole-in-VRAM load. Details under Fixed.
- **Example workflows** regenerated from the current node definitions (the `info` outputs, Turbo's `enabled` switch and the newer widgets were missing in some).

### Fixed
- **Audit 0.8.1** (findings and changes in `docs/AUDIT_0.8.1.md`):
  - **MOSS Compare Sheet / Compare Sampler footer:** a value goes in the footer only when every panel carries it. With one branch without `info` (an image from disk or another pack), the other branch's seed, steps and prompt were written in the footer as if the whole sheet shared them; they now stay on that panel's label.
  - **MOSS Text Check:** an empty read no longer matches visible text (`!!!` read as nothing was `exact` and `passed`); a text made only of punctuation is compared as written. Each item says whether the literal text matched, and the report adds *case or punctuation differ* to an `exact` that is not literal. A thin box (1024 x 4) is upscaled at most to 2048 px on its long side before the reader (it made a 98304 x 384 tensor). The verdicts measured on 75 + 20 texts do not change.
  - **MOSS Model Saver:** the file name is reserved (`O_EXCL`) when the weights are ready, so two exports to the same folder and prefix (two ComfyUI instances) get two numbers; the counter was read before the weights were computed and the later export replaced the earlier one.
  - **MOSS Layout Editor, DRAFT LAYOUT:** imports the draft of its own run only, by the `prompt_id` the server gave it; a job already in the queue no longer cancels the import, and an earlier run of the editor no longer imports its older draft. The editor's event listeners are removed with the node. Checked in the classic view with HaDeS and standard widgets and in Nodes 2.0 (Vue) with both, with a job running when DRAFT LAYOUT was pressed (`tools/RESULTS.md`).
  - **MOSS Layout Editor, standard widgets:** DRAFT LAYOUT and GENERATE buttons under the draft's JSON, where the canvas buttons are hidden.
  - **MOSS Compare Sampler, `images`:** every image of every batch (A's batch first); it gave only the first image of each model.
  - **LoRA Merger / Resize / Extractor:** a failed export removes the empty file reserved for it; it stayed in `models/loras` under the final name.
  - **Suite Telemetry:** an incomplete `info` is named and left out, and the checks run on the others; it replaced every finding with one warning.
  - **MOSS LoRA Resize:** a LoRA of full differences only is told apart ("no rank to reduce") from one that does not apply to the model.
  - **Dev Tools import (git checkout):** a broken `moss_dev_nodes.py` is logged with its traceback instead of hidden; the rest of the pack loads. Absent, as in the Registry package, it stays silent.
- **Suite Telemetry, canvas check:** with a Layout Editor in the pipeline, the canvas of a Prompt Enhancer in `layout` mode that wrote the editor's first draft is no longer compared: the editor's canvas (resizable) is the one used. It gave a false *Canvas mismatch* error when the editor's canvas was resized, or before DRAFT LAYOUT had brought the draft in. The editor, the Layout Compiler, the latent and the KSampler are still compared.
- **MOSS Model Saver, `rounding` nearest:** computes the patched weights itself, the way ComfyUI does for a model loaded whole in VRAM (the weight cast to the GPU's LoRA compute dtype, the patches applied, rounded once), so the file no longer depends on what else is in VRAM. It used the core's lazy saving, whose result depends on how much of the model is loaded: run alone it gave exactly these weights, but in a running server (text encoder loaded too) it gave a file that rendered visibly different from the live LoRAs at the same seed. Checked on an RTX 5090: 0 elements differ from ComfyUI's whole-in-VRAM load in the 12 of 168 patched tensors compared (`tools/RESULTS.md`). Live renders made while the model is only partly in VRAM use LoRAs applied another way and can still differ a little from the file; the tooltip says so.

## [0.8.1] - 2026-11-18

### Added
- **`MOSS Text Check`** (new node): reads the text in a generated image with MOSS-Image's own text encoder (Qwen3.5-9B, whose file holds the vision tower; no other model is loaded) and compares it with what was asked: each layout text item inside its box, or expected texts anywhere. Verdicts exact / close / missing / outside its box, a mask of the boxes that failed for a retry, `passed`, and `info`. On 75 images with text drawn in common fonts (28-110 px) every verdict was right (exact 40/40, a one-letter typo 20/20 close, empty box 15/15 missing; `tools/RESULTS.md`); on 20 texts painted by MOSS itself 19 verdicts were right (the miss: a word cut by the image edge read as whole). A text split over several lines is matched joined.
- **`MOSS Compare Sheet` blind mode:** `blind` shuffles the panels and numbers them 1 to 4 without names; the key goes to `info` and the PNG, never onto the sheet. `blind_seed` sets the order.
- **`tools/moss_lora_atlas.py`:** per-group profile of a LoRA (each 6-layer group alone and left out), block recipes built from it, and their check on other prompts. Measured on three LoRAs (`tools/RESULTS.md`, LoRA atlas): layers 24-29 carry much of what a LoRA adds, but no recipe beat the full LoRA or a plain strength change for the user, so it stays a research tool.
- **`tools/moss_edit_train.py --hold-out-ids`:** holds out the given scenes instead of a random draw, so two runs can be compared on the same held-out images.
- **`MOSS Compare Sampler`** (new node): samples up to four models with the same prompt, latent, seed and settings (each as the MOSS KSampler would, turbo included) and saves the labelled sheet, with one `info` for Suite Telemetry, which checks every branch from it. Labels come from the recipe each model carries, not from the wiring: a Multi-LoRA switched off shows the bare model, a merge shows `merge: a + b`.
- **MOSS Turbo LoRA:** an `enabled` switch (optional, last, default on, so saved workflows keep their widget order). Off, the MOSS KSampler applies no turbo and samples with its own steps, CFG, sampler, scheduler and SLG, and draws those widgets at full strength again; before, its settings only came back by unplugging the cable or setting strength 0. The turbo node keeps its strength and settings, and its `info` says it is off.

### Changed
- **MOSS Turbo LoRA:** `strength` moves in steps of 0.01 (was 0.05), to try values such as 0.22 when the turbo is used at low strength over a LoRA; 1.0 stays the only strength measured.

### Fixed
- **`MOSS Compare Sheet` labels:** a branch with the turbo on and no LoRA of its own was labelled with the turbo alone; it now shows the model file and then the turbo. A merge's name holding ' + ' is no longer repeated on its label. A model named by only some branches no longer goes in the footer as if every panel used it (found in the three-view acceptance: a merge loaded with the stock loader was shown under the Base name); the Compare Sampler labels a model without a recipe `unknown model`.

Audit of 0.8.0 (every finding checked first; details and tests in `docs/AUDIT_0.8.0.md`):
- **`MOSS KSampler` turbo input:** a LoRA that changes no weight of the model (made for another model) is now an error; before, the few-step settings still ran and the metadata named the turbo. Strength 0 means no turbo.
- **`MOSS KSampler` denoise:** denoise 0 no longer crashes (the latent passes through); in `exact` mode the reported start sigma and steps are those of the schedule actually run (they differed with schedulers that start below sigma max).
- **`MOSS Image Saver`:** with several sampling stages in `metadata` and no `primary_stage_metadata`, the node now says that the A1111 parameters come from the first stage and the LoRA hashes from all of them.
- **Saving weights:** Model Saver, LoRA Extractor, Merger and Resize write to a temporary file and rename it at the end, so an interrupted save leaves neither a partial file nor a lost one; a free name is reserved at once.
- **`MOSS Compare Sheet`:** settings equal in every branch go in the footer once (model, steps, CFG, sampler / scheduler, size); what differs (model, seed, steps...) goes on each panel's label. One LoRA on two checkpoints no longer gives two identical labels.
- **Prompt Enhancer layout mode:** an answer whose items all lack a usable box keeps a draft with them in `dropped` for the Layout Editor; a dropped text item without text comes back as an object.
- **Multi-Model Merger TIES:** keeps exactly the planned share of entries when magnitudes tie at the cut (a constant tensor was zeroed); Model Stock no longer rejects weights summing to 0.
- **Validation:** Multi-LoRA reads `"on": "false"` and rejects unknown group names; Control rejects start_percent ≥ end_percent; Metadata Reader caches headers, rejects a non-object header and no longer calls FP8 files int8 ConvRot; Telemetry and Compare Sheet ignore malformed HADES_META trailers instead of crashing.
- **`tools/moss_pair_check.py`:** compares the schedule actually run (denoise mode, start sigma, schedule steps); its test runs in CI.
- **`tools/moss_edit_train.py`:** a pair may list several captions (`captions`), one drawn per step.

## [0.8.0] - 2026-11-15

A new node, `MOSS Compare Sheet`, for comparing LoRAs or checkpoints inside the Suite; the Layout Editor places layout items the Prompt Enhancer dropped.

### Added
- **`MOSS Compare Sheet`:** up to four images (A to D) side by side in one PNG, each labelled with the LoRA its branch used, read from that branch's `info` (Multi-LoRA rows with their strength, LoRA Block Weight, the KSampler's turbo, else the Loader's model), with seed and prompt underneath (from the KSampler and Prompt Enhancer infos). Labels and footer can be typed instead. Images of other sizes are scaled to image A's height. Saves to `MOSS/compare/` (or only previews), keeps the workflow in the PNG and the labels in a `hades_moss_compare` chunk, and returns the sheet and an `info` for Suite Telemetry. Comparing checkpoints of one LoRA no longer needs stitch nodes or a script.
- **Dropped layout items can be placed again:** in `layout` mode the `MOSS Prompt Enhancer` used to leave out, with only a note in `info`, the items whose answer had no box or a box outside the canvas. It now keeps them in a `dropped` list in its draft (label, caption, text and the reason). After IMPORT, the `MOSS Layout Editor` shows one row per dropped item under the canvas: PLACE, then drag on the canvas, and the item joins the layout with its label, caption and text; DISCARD removes it. The `MOSS Layout Compiler` accepts `dropped` and leaves it out of the model prompt; its `info` and the editor's say how many are still unplaced. Suggested by a user.

### Changed
- **README:** the introduction lists the models and LoRAs made with the suite (HaDeS MOSS v1.0 Turbo, Turbo 8-steps v1.0 for MOSS-Image Base, the earlier SFT turbo LoRAs), and `MOSS Turbo LoRA` names v1.0 and says to leave the `turbo` input empty with HaDeS MOSS v1.0 Turbo; it used to describe only the SFT LoRAs. Prompt Enhancer modes listed one by one, its status line under Interface, all tools in `tools/` listed.
- **Example workflows:** the loaders now name the `qwen3.5_9b_int8_convrot` text encoder (and its Comfy-Org download link) instead of `qwen3.5_9b_w4a8_gptq`. The int8 file is the one used in every test on the HaDeS MOSS v1.0 Turbo card; the w4a8 file still loads (pick it in the loader for less VRAM).

### Fixed
- **`MOSS Prompt Enhancer` status line:** in `detailed` mode it said CAPTION; it now says DETAILED, or "DETAILED · random prompt from the seed" with an empty idea and no image. The unused `width` / `height` rows are dimmed in `detailed` mode as in `caption`.
- **Dev tools:** the scripts in `tools/` have one statement per line (the Registry's validator warns about `;`, and will reject it later).

## [0.7.5] - 2026-11-14

The Prompt Enhancer gets a `detailed` mode and random prompts.

### Added
- **`detailed` mode in `MOSS Prompt Enhancer`:** turns a short idea into a longer prompt (about 120-200 words) written like a photographer's or illustrator's brief: where the light comes from, lens and shot type for photographs, technique and materials for other media, concrete and slightly imperfect details, no quality tags. Unlike `caption`, it may add setting, props and story details beyond the idea; it keeps the idea's subject, medium, counts, exclusions and visible wording. Offered as another option, not measured against `caption`. The `caption` instructions are unchanged. The idea's medium is kept and named first, with no camera words outside photographs (a first version turned an anime idea into a photograph).
- **Random prompts in `MOSS Prompt Enhancer`:** `detailed` mode with an empty `idea` and no `image` invents a prompt. The seed draws a medium (26), a subject area (20) and a place / time (20) from fixed lists, so prompts vary by construction rather than drifting to the model's favourite scenes; `info` names the draw.

## [0.7.4] - 2026-11-13

Registry release: the listing of 0.7.3 still named two nodes removed in that version.

### Added
- **`tools/moss_turbo_v10_prompts.py`:** writes the training prompts of Turbo v1.0. It takes the 1024 prompts of Turbo v0.9 and replaces 230 of the no-person prompts with photographs of people whose ethnicity is not given, against the café bias.
- **Turbo v1.0 measurements in `tools/RESULTS.md`:**
  - On the neutral café prompt, about 8 / 16 East Asian faces, against about 11-12 for v0.9, and the count stays flat through training.
  - Text is right as often as with v0.9.
  - Images are brighter and less saturated than v0.9 (median luma 120 against 110, saturation 74 against 89).
- **Edit benchmark in `tools/RESULTS.md`:** 24 edit instructions on 8 SFT references, 3 seeds, chinoll's MOSS-edit, HQ armV5 and no edit LoRA, scored blind by the maintainer.
  - Clean removals: armV5 11 / 21, chinoll 7 / 21, none 0 / 21.
  - Other instructions done: armV5 46 / 51, chinoll 42 / 51; chinoll changes less of the rest of the image.
  - The Prompt Enhancer's edit mode renders pixel-identical to the core `TextEncodeMOSSImageEdit` on four checked cases.
- **`tools/moss_edit_pairs.py`:** builds removal pairs for an own edit LoRA. B is a clean MOSS SFT scene; an editor (Qwen-Image-Edit 2509 with the Lightning 8-step LoRA, or a MOSS edit LoRA) adds an object; only the changed area is pasted back onto B, so A equals B outside it. The pair is A + "Remove the X." → B. Qwen3.5-9B checks that the object is present in A and absent in B. Also writes review sheets.
- **`tools/moss_edit_train.py`:** trains a MOSS-Image edit LoRA from zero on those pairs (flow matching, shift 5, Moss-Image-Edit reference format, gradient checkpointing: about 20 GB and 3.1 s / step at 512 px with accumulation 4). Holds out whole scenes, writes previews, resumable checkpoints and a report.
  - Overfit check on 74 pilot pairs, 500 steps: the four training previews remove their object cleanly; on two unseen scenes, one clean removal and one partial. A pilot, not a result.
- **Edit LoRA tools, second round:**
  - `moss_edit_pairs.py check` also asks how many of the object are in A; a usable pair has exactly one (Qwen-Image-Edit sometimes adds two). On the pilot it dropped 5 pairs, all with two or more of the object. Requests to ComfyUI retry on a dropped connection.
  - New modes: `propose` (scenes from a prompt list, three objects per scene proposed by Qwen3.5-9B from offered kinds, the edit benchmark's objects excluded, a position each), `scenes` (B for that plan) and `combine` (objects added separately to the same B pasted together, each one removed with the others kept; on the 1000-scene run, 482 extra pairs from 241 scenes, no GPU).
  - `moss_edit_train.py`: `--hold-out-objects` (whole objects out of training), `--vary-instruction` (Remove / Erase / Get rid of ...), `--t2i` (plain text-to-image LoRA on images and captions).
- **First own removal LoRA, v0.1 (not published):** 3000 steps at 512 px on 1384 pairs of 62 objects. In ComfyUI it removes cleanly at 512 and copies the reference at 1 MP: 0 / 21 on the benchmark's removal tasks at 1 MP. At 512 it removed only objects close to its 62 training objects (a sign 3 / 3, a car 1 / 3; mug, lamp post, floor lamp, dock and desks 0 / 3). Next: train at 1024 on far more kinds of objects.

### Changed
- **`pyproject.toml` description:** no longer names MOSS Pair Guard and MOSS Recipe Preview (removed in 0.7.3) or the repository-only Dev Tools, so the Comfy Registry listing matches the package.
- **MOSS KSampler, `denoise_mode` tooltip:** now gives the measured result instead of "not measured yet". On img2img with Turbo v1.0 at 4 and 8 steps (112 images, `tools/RESULTS.md`):
  - up to denoise 0.5 the two modes give the same image;
  - above 0.5, `exact` moves smoothly, while the core mode repeats an image and, at 0.9, starts from pure noise.

## [0.7.3] - 2026-11-11

First Comfy Registry release since 0.6.0: it also carries 0.6.1, 0.7.0, 0.7.1 and 0.7.2 below.

### Added
- **MOSS Prompt Enhancer, tag prompts:** Danbooru count tags (`1girl`, `1boy`, `2girls`, `multiple_boys`) are written as adults ("a young woman", "a young man") unless the idea states an age; written as "girl", a recycled `1girl` prompt rendered a childlike figure. Checked through ComfyUI on four tag prompts x two seeds: 8 / 8 adult wording, `elderly` kept.
- **MOSS KSampler `denoise_mode`:** `exact` makes denoise continuous at few steps (a point on the model's noise-time axis, every step run from there with the scheduler's spacing); the default `steps / denoise (core)` keeps the stock schedule. `info` records the start sigma and, in core mode, the steps the core actually builds.
- **`tools/moss_pair_check.py`:** checks that two images saved by the MOSS Image Saver (or two Telemetry json files) are a fair A / B pair: same size, sampling settings and prompt; a different model or LoRA is listed. Replaces the Pair Guard node.

- **`tools/moss_v094_range_eval.py`:** HaDeS MOSS v0.9.4 Turbo at 6-50 steps and CFG 1-3, and the café bias count; results in `tools/RESULTS.md` and the v0.9.4 model card.

### Changed
- **Every node has an `info` output with HADES_META, read by Suite Telemetry:** MOSS Turbo LoRA, MOSS Sampler Metadata, MOSS Image Saver and MOSS Suite Telemetry get an `info` output (added last, so saved workflows keep their links). Telemetry lists them by name instead of counting them as foreign inputs, and a test keeps every node's trailer and Telemetry's names in step.
- **Dev Tools are repository-only:** `moss_dev_nodes.py`, `web/moss_dev_tools.js` and `examples/dev/` are left out of the Comfy Registry package; the Suite loads without them. `pyproject.toml` no longer lists `web` and `examples` under `includes`: comfy-cli forces listed folders in past `.comfyignore`, and both are packaged anyway as git-tracked files.
- **Removed MOSS Pair Guard and MOSS Recipe Preview** (Dev Tools, repository-only): the pair check is now `tools/moss_pair_check.py`, for saved images; Recipe Preview repeated what the `info` outputs and the Metadata Reader show. Their Dev example is removed; the Dev Tools README section is rewritten in the Suite's format.
- **Suite Telemetry, denoise:** a warning when a core-mode denoise below 1 runs the full schedule from pure noise (e.g. 0.9 at 4 steps: the input is ignored), and a note when step rounding gives the same image across a range of denoise values. From ComfyUI's own schedule code; reported independently by lilting.ch.
- **Suite Telemetry, LoRA energy:** the 97.9% energy warning for LoRA Resize / Merger is now a note with both measurements: a style LoRA drifted at about 90% energy, while few-step LoRAs cut to rank 2 rendered the full LoRA's images, so energy alone does not predict the image. LoRA Extractor keeps a warning, at its own measured threshold (95.9%).
- **`tools/RESULTS.md`:** GPT, Gemini and assistant preferences on the blind sheets moved to a section marked qualitative; the experiments' conclusions cite only the maintainer's votes and the metrics. The v0.9.4 Turbo model card likewise counts only the author's votes.

### Fixed
- **Dev Tools report widget:** the read-only report no longer fails when the frontend gives the widget no `inputEl` (Nodes 2.0 / Vue).
- **Docs:** `tools/RESULTS.md` and the README no longer carry local drive paths, chat attachment identifiers or an internal working note.

### Tests
- int8 ConvRot model + LoRA saved as int8: the LoRA'd layers are requantized, the untouched ones are written byte for byte.

## [0.7.2] - 2026-11-11

### Added
- **MOSS Pair Guard:** verifies that reference and candidate image batches have matching geometry and pair counts before running image audits. Optional stage-specific Telemetry inputs enforce matching sampling controls (seed, steps, CFG, sampler, scheduler, flow shift, denoise, size, SLG, prompt/negative) while allowing model/LoRA differences.
- **MOSS Recipe Preview:** previews the exact recipe text that `MOSS Model Saver` will embed as `hades/recipe`, validates `HADES_META` trailers, and cross-checks with the model patcher's attached recipe.
- **Dev examples organization:** model-free development workflows now reside under `examples/dev/` (`Dev Tools`, `Pair Guard and Recipe Preview`, `Stage Metadata`), keeping `examples/` dedicated to generation workflows; `tools/make_examples.py` updated accordingly.

## [0.7.1] - 2026-11-11

### Added
- **Image Saver stage metadata:** an optional `primary_stage_metadata` input takes a stage-specific Telemetry JSON. The selected stage supplies A1111 settings, model/LoRA hashes and filename tokens; the full pipeline remains in `hades_moss`, with the selected stage in a separate PNG chunk. Existing workflows retain their first-KSampler behavior.
- **Model Card:** complete model card and sample manifest for `HaDeS_MOSS_v0.9.4_Turbo` under `tools/model_cards/`.
- **Baked Turbo measurement:** `tools/moss_baked_turbo_eval.py` archives paired CFG / step sweeps, exact graphs, image metrics and blind comparison sheets on the experiment drive. First results are in `tools/RESULTS.md`.

### Fixed
- **Interface:** standard ComfyUI widgets toggle keeps native output previews (such as core Save Image / Image Saver previews) visible.
- **MOSS Loader:** model filename metadata no longer collides with the model patcher argument when recording a loader recipe. The collision prevented the loader from completing; a regression test covers the metadata contract.

## [0.7.0] - 2026-11-11

### Added
- **Dev Tools:** `MOSS Layout Regions` makes binary masks from selected layout boxes; `MOSS Image Audit` measures paired images globally and inside / outside a mask, with per-image JSON, color deltas and a difference map. No model loading or added dependencies. Includes a model-free example and geometry / numerical / frontend regressions.
- **Edit-LoRA measurement tool:** `tools/moss_edit_eval.py` generates controlled references, compares full / keep-only / leave-out layer configurations through a running ComfyUI, and archives images, hashes, graphs, metrics and contact sheets on the experiment drive. Protocol and observations: `tools/RESULTS.md` (MOSS-edit layer ablation, 2026-11-10).

### Fixed
- **Dev Tools interface:** reports remain visible with standard ComfyUI widgets and are read-only in Nodes 2.0 / Vue as well as the classic canvas. Both nodes were exercised in all three Suite views; acceptance evidence is in `tools/RESULTS.md`.
- **LoRA export:** factors and full diffs on the same weight are added before saving; diff-only merges finish with compression energy marked as not applicable. Merger and Resize protect source files by filesystem identity, including Windows filename case and hard links.
- **Image Saver:** ComfyUI's `--disable-metadata` now suppresses every embedded text chunk, including A1111 parameters and `hades_moss`.
- **Model merges:** zero-weight models no longer amplify TIES / DARE-TIES. SLERP uses linear interpolation for zero-length and nearly antiparallel inputs. Recipe overrides and Multi-LoRA JSON reject non-finite strengths.
- **Model Saver:** quantized inputs are converted to the requested int8 ConvRot layout with matching scales and configuration; existing matching int8 weights and scales are preserved.
- **Recipes:** merger info includes complete ratios / weights, seed and sources known to the Suite; the Metadata Reader follows these nested recipes and invalidates its cache when a source changes or appears.
- **Interface:** the MOSS classic-widget switch leaves other suites untouched and preserves individual node preferences.

### Tests
- Added real-core export / reload regressions, Windows source-file identity checks, frontend setting tests and opt-in CUDA export checks. CI now includes ComfyUI v0.43.0 on Linux and Windows, plus upstream master on Linux. Local measurements and remaining limits: `docs/AUDIT_0.6.1.md`.

## [0.6.1] - 2026-11-10

Edit mode in the MOSS Prompt Enhancer.

### Added
- **MOSS Prompt Enhancer, edit mode:** `edit_format` (`off`, `Moss-Image-Edit`, `ai-toolkit`, `ai-toolkit (kv_cache off)`, the core's formats), `vae` and `edit_megapixels`: with an image-edit LoRA in the Multi-LoRA (e.g. `MOSS-edit` / `Moss-Image-Edit`, or an `ai-toolkit` edit / pose LoRA), the image becomes the reference latent of both the positive and negative outputs, replacing two core `Text Encode MOSS-Image Edit` nodes. Use `as written` for a direct edit instruction (e.g. on `MOSS-edit`, which also runs cleanly with `HaDeS MOSS Turbo 8-steps` on the KSampler's `turbo` input) or `caption` to have the model write the full target description from the image and the idea. The reference is resized to `edit_megapixels` with its aspect kept (16 px grid) and `width` / `height` out give its size for the MOSS Latent. Shown in the status line and `info`.

## [0.6.0] - 2026-11-10

MOSS Control (color / depth), and fixes for MOSS Latent and the MOSS Image Saver.

### Added
- **MOSS Control (color / depth):** color / depth control for a MOSS-Image control adapter loaded as a LoRA, on the Prompt Enhancer's conditioning: maps resized in pixels to the latent before the VAE, `color_mode` off (depth only) / flat regions (a photo reduced to flat colors) / as given, strength and step range, a `maps` preview and `info`. Suite Telemetry notes a strength other than 1.0 and `as given`; the Metadata Reader names a control adapter (widened `video_embedder`) and says it is applied at full strength whatever the LoRA strength.
- **Example workflow "Control (color, depth)":** a depth map through MOSS Control with the adapter in the Multi-LoRA. The other examples gain the MOSS Turbo LoRA's cfg / sampler / scheduler widgets at their defaults (no change in output).

### Fixed
- **MOSS Latent:** `custom_width` / `custom_height` were ignored unless `aspect_ratio` was `custom`: editing them now switches it to `custom`, and when they are linked from another node they are used whatever the combo says.
- **MOSS Image Saver:** without its `metadata` input connected the node did not run and no image was saved. `metadata` is optional now: the images are saved with the workflow and the prompt texts given, and a note says the sampling settings are missing.

## [0.5.0] - 2026-11-09

The MOSS Metadata Reader, recipes stored in saved models and LoRAs, and a turbo input that can run above CFG 1.

### Changed
- **MOSS Turbo LoRA:** `cfg`, `sampler_name` and `scheduler` widgets; the MOSS KSampler's turbo input samples at them instead of fixed CFG 1, euler / simple. The defaults are those settings, so saved workflows render as before. Above CFG 1 the negative prompt takes part and each step runs twice (about 1.9x the time). Measured with one seed per prompt (`tools/RESULTS.md`, "Turbo LoRAs above CFG 1"): Turbo 8-steps at 1.5 rendered 10 hard prompts on SFT and Base without burning; on a merge with a turbo baked in, 1.25–1.5 brought back prompt details lost at 1.0, saturation rose with every step and 2.5 looked overdone; texture-heavy prompts looked better at 1.0–1.2. The KSampler's status line shows the turbo node's sampler and CFG.
- **Suite Telemetry:** a turbo LoRA above CFG 1 or a few-step LoRA in Multi-LoRA above CFG 1 is now a note with what was measured (it was a warning for Multi-LoRA); a note for a turbo LoRA with a sampler other than euler / er_sde (simple); a Multi-LoRA few-step LoRA below CFG 1 stays a warning.
- **`tools/moss_dmd_train.py`:** `--prompts-file`, a JSON list of training prompts in place of the built-in generator.

### Added
- **MOSS Model Saver:** an optional `recipe` input (text, or the `info` of the MOSS Loader, Multi-LoRA or a merger), written into the file as `hades/recipe`, so a model with merges and LoRAs baked in keeps how it was made. Empty, the file is as before.
- **MOSS LoRA Extractor:** optional `tuned_name` and `base_name`, written into the LoRA as `hades/model_tuned` / `hades/model_base`.
- **MOSS Metadata Reader:** shows a LoRA's and / or a diffusion model's safetensors metadata and what its tensors show (size, dtypes, LoRA kind, rank and alpha, the MOSS-Image layers and modules touched, MOSS-Image or not, parameters, fused q/k/v, int8 ConvRot), reading only the file header. For LoRAs from the Suite's Merger, Resize and Extractor it follows the `hades/*` recipe into the source files still in `models/loras` (also when moved to another subfolder) and gives each original file's weight in the result. For models it shows the Model Saver's `hades/recipe` and follows the model (MOSS Loader) and LoRAs (Multi-LoRA, LoRA Block Weight) named in it into their own recipes; for extracted LoRAs, the model names when the Extractor was given them. Optional SHA256. Outputs `info` (with HADES_META) and `metadata_json`.

### Fixed
- **MOSS Model Saver:** a `recipe` connected with `include_metadata` off was dropped without a word; `info` now says it was not saved.
- **MOSS Metadata Reader:** a model recipe listed Multi-LoRA and LoRA Block Weight LoRAs at their strength without their group weights, so a LoRA applied to some layer groups only read as applied everywhere; the groups are shown now.
- **Suite Telemetry:** a Metadata Reader `info` was counted as an input without a HaDeS MOSS trailer.

## [0.4.1] - 2026-11-07

The MOSS Image Saver.

### Added
- **MOSS Image Saver:** PNG images with the generation settings as A1111 `parameters` (prompt, negative prompt, steps, sampler, CFG, seed, size, model and LoRAs with SHA256 hashes, cached per session) read from Suite Telemetry's `json`, plus ComfyUI's workflow; filename tokens `%seed%`, `%model%` and others. The MOSS Loader's info now carries the model and text encoder file names, and the Prompt Enhancer's its prompt and negative text, for it; Telemetry's report leaves those out.
- **Example workflows:** saved through the MOSS Image Saver, with the Prompt Enhancer's info in Telemetry.

## [0.4.0] - 2026-11-07

New nodes MOSS Loader, Multi-LoRA, Layout Editor and Prompt Enhancer, a shared seed on MOSS Latent, stochastic rounding in the Model Saver, example workflows rebuilt on the Suite's own nodes, and the fixes of the pre-release audit (`docs/AUDIT_0.4.0.md`).

### Added
- **Comfy Registry icon:** `Icon` in `[tool.comfy]`, so the Suite's card shows `web/icon.png`.
- **MOSS Model Saver:** a `rounding` input for `bfloat16` output, `stochastic` by default: weights with merge / LoRA patches are rounded up or down with probability by distance (seeded per weight, so the same inputs give the same file); `nearest` is ComfyUI's own saving. Unpatched weights are copied unchanged either way. Measured with two LoRAs whose changes are smaller than a bfloat16 step: rounded to nearest the file kept about 80–85% of their effect, stochastic all of it (`tools/RESULTS.md`, "Baking subtle LoRAs into bfloat16"). The chosen rounding is written in the file's metadata and the info output. Files saved before were rounded to nearest.
- **MOSS Prompt Enhancer:** a positive / negative prompt box for the sampler (conditioning outputs, an `as written` mode with no generation, the official template's negative when empty) that also writes a detailed prompt, in the suite's canvas style (status line for the mode, rows the mode does not use faded) (a caption, or a pixel Layout draft compiled by the Layout Compiler's code) from a short idea and / or an image, with MOSS-Image's own text encoder, so no other model is loaded. Instructions adapted from OpenMOSS' official `moss-image-prompt-enhancer` (Apache-2.0). Works around a comfy-kitchen 0.2.40 decode-kernel error that stops text generation on PyTorch builds below cu130 (in the core `TextGenerate` too), only while the node generates. Tried on one seed per case (`tools/RESULTS.md`), not measured. Its seed starts on `fixed`, so the same idea writes the same prompt until you change it.
- **MOSS Latent:** a `seed` widget (with control after generate, starting on `fixed`) and a `seed` output at the end, to share one seed between the MOSS KSampler, a second sampler and the Prompt Enhancer so the whole image reproduces from the Suite; shown in the status line and `info`. Workflows saved before keep their outputs and get seed 0. The example workflows wire it to their KSamplers.
- **MOSS Loader:** diffusion model, text encoder (always type `moss_image`) and VAE in one node, as the three core loaders load them, with clear errors for a non-MOSS model, a text encoder that is not MOSS-Image's or a VAE without 128 latent channels, and an `info` output (files, precision in memory, the attention the file sets). Suite Telemetry reads it: it names the files and notes an int8 model with its measured cost or a `weight_dtype` conversion.
- **MOSS Multi-LoRA:** any number of LoRAs in one node, one row each with on / off, strength (drag, click to type, right-click for 1.0) and a foldable weight per group of 6 layers (LoRA Block Weight's code, now shared); a LoRA picker with a search box; a master switch that passes the model through. Few-step LoRAs are recognised from their file's metadata (the HaDeS MOSS Turbo files), tagged with their steps and checked by Suite Telemetry against the KSampler's steps and CFG. No group presets: none has been measured.
- **MOSS Layout Editor:** a visual editor for layout drafts. Boxes on a canvas of the draft's size, colored by label, moved, resized and drawn with the mouse on the 16 px grid; the selected box's label, caption, text and pixels edited under the canvas; canvas size changes scale the boxes; boxes still missing a caption or text outlined in red. Optional `draft_in` (imported with a button, e.g. from the Prompt Enhancer) and a background `image` (from Load Image at once, otherwise after a run) with a button to match the canvas to its aspect ratio. Its `draft` goes to the Layout Compiler; Suite Telemetry includes its canvas in the size check. Example workflow "Layout Editor".
- **`tools/moss_slider_train.py`:** trains image-free slider LoRAs on MOSS-Image: from a pair of prompt suffixes (Concept Sliders on flow matching), or with `--pairs` from image pairs the model renders itself (`--pair-kind blur` for detail, `colour` for colour), optionally on an SFT / Base merge computed in memory.

### Changed
- **Example workflows:** all load with the MOSS Loader, encode their prompts with the Prompt Enhancer (`as written`) and share the MOSS Latent's seed with their samplers; Telemetry reads the loader too. The LoRA Toolkit uses the Multi-LoRA instead of LoRA Block Weight. New example "Prompt Enhancer" (caption mode, one seed for the writing and the sampling, optional Load Image).

### Fixed
- **MOSS Model Saver, float32:** weights with merge / LoRA patches were rounded to bfloat16 before reaching the float32 file (the core's lazy saving applies patches in the model's own dtype), so changes smaller than a bfloat16 step were lost while the metadata said `rounding: none`. They are now computed in float32.
- **LoRAs for another model:** LoRA Block Weight and the LoRA tools loaded a LoRA none of whose keys match MOSS-Image as nothing (Block Weight reported it as applied). Now an error names the file, in the new Multi-LoRA too, whose `info` gives the number of weights each LoRA changes.
- **Suite Telemetry:** the canvas-mismatch error fired when a latent and a KSampler differed in size without any layout (img2img, an upscale pass); it now compares canvases only with a layout in the pipeline, and includes the Prompt Enhancer's canvas in `layout` mode.
- **README:** Telemetry warns about a LoRA below 97.9% energy (the measured threshold), not 90%.
- **`__version__`** in the package said 0.2.1 in the 0.3.0 release.

## [0.3.0] - 2026-11-06

Turbo 8-steps v0.7, a `steps` widget on MOSS Turbo LoRA and an example workflow with a 2x upscale.

### Added
- **MOSS Turbo LoRA:** a `steps` widget (default 4, the count Turbo and Turbo Diverse were trained on; 8 for Turbo 8-steps; both measured), so a few more or fewer steps can be tried; the MOSS KSampler uses it in place of the fixed 4, and its status line shows it. Workflows saved before keep 4.
- **Turbo 8-steps v0.7** (`HaDeS-MOSS-Turbo-8steps-v0.7.safetensors`, Hugging Face): a third turbo LoRA, for 8 steps. In blind pairs at 8 steps it beat akhaliq's 4-step LoRA 18 to 5 and matched lodestones' Base / Kromoss 8-step LoRA run on SFT at 1024 (10 to 7); lodestones' wins at 2048. Turbo and Turbo Diverse stay the 4-step files.
- **Suite Telemetry:** a note when a turbo LoRA runs at a step count other than 4 or 8 (the measured ones); the denoise note, the KSampler `turbo` tooltip and the README now cite the Turbo 8-steps pass at denoise 0.25 / 0.40 on a 2x-upscaled image.
- **Example workflow "Turbo 8 steps + Upscale x2":** generation with Turbo 8-steps v0.7, a 2x upscale with RealESRGAN x2 and an optional MOSS pass at 2048 (Turbo 8-steps, denoise 0.25), each in a group switched from the group's right-click menu, with a note. `tools/make_examples.py` now writes groups, notes and fixed positions.
- **tools/moss_dmd_train.py:** `--fake-updates N`, DMD2's two time-scale rule (N fake-score updates per generator update, the extra ones on fresh generator samples). With 5, seed variety held from update 1,000 to 2,000 where it collapsed without (v0.7a, `tools/RESULTS.md`).

### Documentation
- **tools/RESULTS.md:** v0.7a (two time-scale updates) checkpoint by checkpoint to update 3,000; the few-step LoRAs at 8 steps in blind comparisons (v0.7a against akhaliq's and lodestones' LoRAs, cut and full) and their colour against SFT at 25 steps.

## [0.2.1] - 2026-11-05

Fixes from the 0.2.0 audit (`docs/AUDIT_0.2.0.md`); 0.2.0 never reached the Comfy Registry.

### Added
- **MOSS Sampler Metadata:** a `turbo` output, the turbo LoRA file and strength the KSampler used (empty without one), so image-saver metadata can name it. Appended after the existing outputs; saved links keep their slots.
- **Tests:** Telemetry's turbo notes (strength other than 1.0, denoise below 1.0) and the metadata `turbo` output.

### Changed
- **MOSS KSampler (canvas):** with the turbo input connected, the rows it overrides (steps, cfg, sampler, scheduler, SLG) are drawn faded; they stay editable.
- **MOSS Layout Compiler:** the info trailer's `items` now counts every item, with `text_items` for those with text (it counted text items only, while its text said "N items, M with text").

### Fixed
- **Hugging Face model card:** the Turbo (update 1,600) cells for 2048 x 2048, 768 x 1344, the explicit prompts and the 2048 timing came from other checkpoints; measured on the file now, with a 2048 small-text row (Turbo 1 / 3, akhaliq and Turbo Diverse 2 / 3) and the 16:9 sign. `tools/RESULTS.md` records the same.
- **pyproject.toml:** punctuation of the description.

### Documentation
- **README:** the turbo LoRAs' 2048 small-text result and what was only checked to run (other base models, img2img); links to `tools/*.py` point at GitHub, since the Registry package leaves those files out.

## [0.2.0] - 2026-11-05

Turbo: our own few-step LoRAs for MOSS-Image SFT (4 steps, CFG 1, 3.4 MB each, on Hugging Face) and a turbo input on the MOSS KSampler.

### Documentation
- **README / `tools/RESULTS.md`:** upscaling a MOSS latent needs the core `Upscale Latent By`; `Upscale Latent` assumes a x8 VAE and doubles the image size on MOSS.

### Added
- **MOSS Turbo LoRA** and a `turbo` input on **MOSS KSampler**: the KSampler applies the few-step LoRA (as the stock `LoraLoaderModelOnly`) and samples at 4 steps, CFG 1, euler / simple with SLG off, the settings it was trained and measured at; its info and status line show it, and Telemetry notes a strength other than 1.0 or denoise below 1.0. Made for HaDeS MOSS Turbo and Turbo Diverse v0.6 ([Hugging Face](https://huggingface.co/DeVilDoNia/HaDeS-MOSS-Turbo), 3.4 MB each), trained with `tools/moss_dmd_train.py`.
- **Example workflow "Turbo":** MOSS Turbo LoRA into the MOSS KSampler on SFT, with the LoRA's Hugging Face download link; run end to end in ComfyUI like the other examples.
- **`tools/moss_eval.py`:** an `attention` mode that times images under the attention backend the server runs and compares runs pixel by pixel. `tools/RESULTS.md` records that every measurement used SageAttention, and compares it with PyTorch attention and int8 files in speed and images.
- **`tools/moss_eval.py`:** `progressive` mode (two-resolution sampling through stock `KSamplerAdvanced`, with re-noise and backtracking), `turbo` mode (few-step LoRAs with the usual and the bypass loader) and `turbo_groups` mode (which layer groups of a few-step LoRA carry its effect). Results in `tools/RESULTS.md`: two-resolution sampling saves about 15% cleanly at 1024 x 1024; the DMD2 8-step and the 4-step LoRAs render 5.6x / 9.8x faster, lose about half their change to bf16 rounding in the early layers with the usual loader, and do their work in layers 24–41.
- **`tools/moss_eval.py`:** `turbo_files` mode (renders LoRA files derived from a few-step LoRA, any prompt set or size, against the original). Results in `tools/RESULTS.md`: cut to layers 24–41 at rank 2 (3.6 MB), both few-step LoRAs render their originals' images, text and layout prompts (54 of 54 spelled right) and 2048 x 2048 included; their directions are not readable from the SFT weights' own spectrum.
- **`tools/moss_activations.py`** and **`tools/moss_bias_turbo.py`:** record what a reduced few-step LoRA reads from the late-layer activations, and replace it by fixed per-step biases calibrated from those activations. Results in `tools/RESULTS.md`: on the qkv weights the change is nearly constant over tokens and images; with no LoRA, the calibrated biases on layers 24–41 turn SFT at 4 steps / CFG 1 into clean images and correct text on held-out seeds.
- **`tools/moss_bias_fit.py`:** fits those per-step biases with no LoRA, against SFT at 25 steps as the teacher; `moss_bias_turbo.py --biases` tests them, `--calib-only` saves calibrated ones. Results in `tools/RESULTS.md`: after 19 minutes of fitting at 512 x 512, SFT at 4 steps goes from broken to coherent images with legible text, below the LoRA's quality. `--teacher-lora` / `--blend` / `--init-biases` fit them against the few-step LoRA, or a blend of it and SFT, as the teacher: from zero in 5 minutes the biases reach the LoRA's quality with no LoRA loaded, and they keep it down to a half-LoRA teacher, losing it below. `--student-lora` fits biases on top of a few-step LoRA (`moss_bias_turbo.py --biases-on-lora` tests them): on our DMD LoRA, half-SFT biases varied the compositions but added haze and broke the lettering on a poster in 3 of 6 seeds.
- **`tools/moss_dmd_train.py`:** trains a few-step LoRA by DMD on one GPU (akhaliq's recipe with one shared base model and hooked LoRAs), with atomic checkpoints every N updates and automatic resume. Results in `tools/RESULTS.md`: a rank-2 LoRA on layers 24–41, trained from zero in 30 minutes at 512 x 512, renders the test prompts at 1024 x 1024 in 4 steps at the published 4-step LoRA's quality by eye, text included, with less variety across seeds. `--init-lora`, `--fake-warmup` and `--checkpoint` (per-block gradient checkpointing: 13.3 GB peak at 1024 x 1024) continue a LoRA at a larger size: 600 more updates at 1024 x 1024 kept the quality and lowered the variety across seeds. `--reg-pairs` / `--reg-weight` add DMD v1's regression term (latent mean squared error to teacher images rendered from fixed noises before training): it brought the seed variety past the published LoRA's, with ghosting and doubled letters at 2,000 updates; 300 more updates of DMD alone cleaned them and brought back one composition per prompt. Kept on through training at a fifth of that weight (0.05), the regression term gave clean images, every string right and the published LoRA's variety; the same LoRA holds at 2048 x 2048, on 16:9 and 9:16 canvases and on layout JSON, and is 3.4 MB in bf16 with identical images.
- **`tools/moss_eval.py`:** `turbo_files` takes the bias prompts in `--prompts` and a non-square canvas with `--height`.

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
