# Audit 0.8.0: findings and fixes

External audit of `main` at `75ef989` (16 November 2026, overall 8.0 / 10). Each finding was checked against the code before
changing anything; every fix has a regression test in `tests/`. Tests after the fixes: suite 89 passed (1 CUDA skipped), Dev Tools 15,
pair check 5, edit eval 4, widgets 5.

| # | Finding | Verdict | Change | Test |
|---|---|---|---|---|
| 1 | Turbo applied the few-step settings (and named the LoRA in metadata) even when the LoRA changed no weight; strength 0 did the same | **Real** | The KSampler checks the turbo LoRA with the same key match as Multi-LoRA and stops with an error for a LoRA of another model; strength 0 now means no turbo (the node's own settings, no turbo in `info`) | `test_turbo_checks_the_lora_and_strength_zero` |
| 2 | Image Saver with several stages in `metadata` and no `primary_stage_metadata` took the first KSampler / Loader but the LoRA hashes of every stage | **Real** | The node now shows a note (UI and `info`) naming the number of KSamplers / Loaders and pointing to `primary_stage_metadata`. Behaviour kept for existing workflows. A1111 `Lora hashes` carry no strength by format; the strength stays in the `hades_moss` chunk | `test_two_stages_without_primary_warn` |
| 3 | Model Saver and the LoRA tools wrote straight to the final name; the free name was chosen apart from the write | **Real** | `write_atomic`: write to `<name>.<pid>.part` in the same folder, rename onto the final name at the end, remove the temporary on failure. Without overwrite, the free name is created with `O_EXCL` (`reserve_path`) | `test_write_atomic_keeps_the_old_file_on_failure` |
| 4 | `denoise = 0` raised `ZeroDivisionError` building the metadata | **Real** | The latent passes through as in the core; `info` records start sigma 0 and `steps_run: 0` | `test_denoise_zero_passes_the_latent_through` |
| 5 | `exact` reported the requested point as start sigma; schedulers that start below sigma max (`ddim_uniform`) actually start lower | **Real** | Start sigma and steps come from the schedule handed to the sampler; `steps_run` when it differs from `steps` | `test_exact_start_sigma_is_the_schedule_actually_run` |
| 6 | Compare Sheet: one LoRA on two checkpoints gave identical labels; the footer took the first branch's seed | **Real** | Settings equal in every branch go in the footer once (model, steps, CFG, sampler / scheduler, size, denoise); what differs (model, seed, steps...) goes on that panel's label; different prompts are said, not hidden | `test_same_lora_on_two_models_and_settings_per_branch` |
| 7 | `moss_pair_check.py` compared the requested denoise but not the schedule actually run; CI did not run its test | **Real** | Compares `denoise_mode`, `start_sigma`, `schedule_steps`, `steps_run`; old PNGs missing them are reported unverified; test added to CI | `test_effective_schedule_is_compared` |
| 8 | Layout answers with no usable box raised before the draft could reach the editor; a dropped text item without text came back as `text: ""` | **Real** | The draft keeps `items: []` and `dropped`; the summary caption is encoded meanwhile with a note; a text item without text is dropped as `object`; the compiler's error says to place the dropped items first | `test_layout_answer_without_any_box_keeps_a_draft` |
| 9 | Metadata Reader re-read the same header for every reference (4,095 reads in a synthetic tree) | **Real** (synthetic) | Headers cached by path, size and modification time; copies returned | covered by the existing reader tests |
| 10 | TIES trim dropped every entry tied at the cut (`[1, 1, 1, 1]`, density 0.5 → all zero) | **Real** | Keeps exactly `k` entries by a stable sort | `test_ties_trim_keeps_exactly_k_with_ties` |
| 11 | Cached Loader / Multi-LoRA branches vs Image Saver hashing the file on disk when a file is replaced under the same name | **Risk, not reproduced** | Not changed. ComfyUI re-runs a loader when its inputs change, not when the file under the same name changes; the hash is of the file on disk at save time. Documented here; replacing a model file during a session should be followed by a ComfyUI restart or a cache clear | — |

## Minor findings

| Finding | Verdict | Change |
|---|---|---|
| `HADES_META []` crashed Compare Sheet; incomplete `ksampler` trailer crashed Telemetry; safetensors header root `[]` raised `TypeError` | Real | `parse_info_meta` accepts only objects with a string `node`; Telemetry reports an incomplete trailer as a warning instead of stopping; the reader rejects a non-object header with a clear error. Test `test_info_meta_must_be_an_object` |
| Metadata Reader called an FP8 file "int8 ConvRot" (any `comfy_quant` key) | Real | Also requires int8 tensors |
| Control: `start_percent > end_percent` encoded the maps and applied nothing | Real | Error before encoding |
| Multi-LoRA: `"on": "false"` stayed on; a misspelt group key was dropped silently | Real | String booleans read; unknown group keys are an error. Test `test_multi_lora_row_validation` |
| Model Stock rejected weights summing to 0 although it ignores weights | Real | The check no longer applies to Model Stock |
| Compare Sheet memory at 4 × 4096² (~770 MiB float32) | Estimate, not measured | Not changed; normal sheets are far smaller |
| Prompt Enhancer swaps a global comfy-kitchen function during generation | Real only with overlapping calls | Not changed: ComfyUI executes prompts one at a time |
| Research tools reuse outputs by file name; resume loads checkpoints with `weights_only=False` | Real, tooling only | Not changed; resume only checkpoints you made |

## Not done here

- Frontend acceptance of Compare Sheet and the Layout Editor's IMPORT / PLACE / DISCARD in the three views (classic HaDeS, classic standard widgets, Nodes 2.0) is still pending: it needs a person in the browser.
- The ZIP is not yet exercised as a clean install in CI.
