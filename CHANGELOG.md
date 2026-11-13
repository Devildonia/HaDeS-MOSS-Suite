# Changelog

All notable changes to the **HaDeS MOSS Suite** are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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
