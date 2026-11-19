# Audit 0.8.1: findings and fixes

External audit of `main` at `15dfa20` (19 November 2026, overall 8.2 / 10), which already held the `Unreleased` changes (Layout Editor
buttons, Model Saver nearest, the Telemetry canvas fix). Each finding was checked against the code before changing anything; every
code fix has a regression test in `tests/`. Tests after the fixes: suite 106 passed (1 CUDA skipped), Dev Tools 15, pair check 5,
edit eval 4, widgets 5. The Layout Editor
buttons were then accepted in four views with real models (`tools/RESULTS.md`, "Layout Editor DRAFT LAYOUT / GENERATE: acceptance").

| # | Finding | Verdict | Change | Test |
|---|---|---|---|---|
| 1 | Compare Sheet put one branch's seed, steps and prompt in the footer of the whole sheet when the other branch had no `info` | **Real** | Every branch counts in `split_common`, also one without settings; the prompt goes in the footer only when every branch carries the same one | `test_footer_only_with_what_every_branch_carries` (and the updated `test_labels_from_branch_infos_and_saved_sheet`) |
| 2 | Text Check gave `exact` to `!!!` against an empty read (both comparison forms empty); `HaDeS` / `hades` and `12.50` / `12 50` are `exact` | **Real** | Empty read against visible text: 0; punctuation-only text compared as written. `exact` stays the folded comparison the measurements used, and each item has `literal`; the report says *case or punctuation differ* | `test_empty_read_never_passes_and_literal_flag` |
| 3 | Model Saver read the counter before computing the weights and did not reserve the name; two instances could write the same file. The 0.8.0 audit row said the Saver used `reserve_path` | **Real** (two writers on one folder) | The name is reserved with `O_EXCL` right before the write, from the counter on; the 0.8.0 row was right for the LoRA tools only | `test_two_saves_with_the_same_counter_get_two_files` |
| 4 | DRAFT LAYOUT: a boolean flag, cleared by any job's end and consumed by any later `onExecuted`; three global listeners per editor never removed | **Real** (with a busy queue) | The `prompt_id` of the request whose partial targets are this editor is read from the API while the command queues it; only that prompt's `executed` message imports; listeners removed in `onRemoved` | browser acceptance with a job running (`tools/RESULTS.md`) |
| 5 | Text Check upscaled a thin crop before the reader's 1 MP limit (1024 x 4 → 98304 x 384) | **Real** (thin boxes) | Upscale factor capped so the long side stays at most 2048 px | `test_thin_box_upscale_is_bounded` |
| 6 | Compare Sampler `images` gave `im[:1]` per branch while the sheets showed whole batches | **Real** | Every image, A's batch first; tooltip and README say so | `test_images_output_keeps_every_batch_item` |
| 7 | A failed LoRA export left the empty reserved `.safetensors` under its final name | **Real** | `reserved(path)` removes an empty placeholder when the work that fills it fails (Merger, Resize, Extractor, Model Saver); a file with content is never touched | added to `test_write_atomic_keeps_the_old_file_on_failure` |
| 8 | One incomplete `info` replaced every Telemetry finding with one warning | **Real** | Each trailer is checked alone first; a broken one is named and left out, the checks run on the rest | added to `test_sizes_compared_only_with_a_layout` |
| 9 | Compare Sheet keeps several copies of a large batch in RAM | **Real** | Measured, then fixed the same day: sheets built, saved and copied into a preallocated output one at a time. 4 panels x batch 4 at 1344 x 1792: peak added 1 112 MB → 555 MB (475 MB of it the `IMAGE` output); output and PNGs bit-identical (`tools/RESULTS.md`) | `test_batch_sheets_built_one_at_a_time_match_their_files` |
| 10 | LoRA Resize said "no layer applies" for a LoRA of full diffs only | **Real** | Its own error: only full differences, no rank to reduce | `test_resize_of_a_diff_only_lora_says_why` |
| 11 | `__init__.py` hid any `ImportError` of the Dev Tools | **Real** | Absent module (Registry package): silent. Present but failing: logged with its traceback, the rest of the pack loads | — (import path) |
| 12 | CHANGELOG summarized the Saver check as "0 elements differ" without the 12 of 168 tensors | **Real** (wording) | CHANGELOG now names the 12 of 168 tensors compared, as README and RESULTS already did | — |
| 13 | Layout Editor buttons not accepted in Vue nor with standard widgets; standard widgets showed only the JSON | **Real** | Standard-widget buttons added under the JSON; accepted in four views (classic / Vue × HaDeS / standard widgets), save and reload included | `tools/RESULTS.md` |
| 14 | CI tests the checkout, not the published ZIP in a clean install | **Real, not a bug** | Done the same day (`1560dad`): a `package` CI job builds the zip with `comfy node pack`, checks its files and loads it in a clean ComfyUI (v0.43.0 and master) | `tests/check_package.py` |

The auditor's "70 passed" was a read-only run in their environment; here the suite runs 106 tests.
