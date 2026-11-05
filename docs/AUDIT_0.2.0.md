# HaDeS MOSS Suite 0.2.0 — audit (2026-11-05)

Scope: every node module, the canvas script, tests, CI, Registry package, README / CHANGELOG / tooltips against
`tools/RESULTS.md`, the turbo node live in ComfyUI 0.43.0 (SageAttention, RTX 5090), and the Hugging Face release
`DeVilDoNia/HaDeS-MOSS-Turbo`.

## Verified

| Check | Result |
|---|---|
| Unit tests (`tests/test_moss_suite.py`, CPU) | 23 / 23 pass |
| CI on GitHub (latest ComfyUI) | green on every commit of 2026-11-04 / 05, 0.2.0 included |
| Canvas scripts (`node --check`) | both valid |
| Turbo node vs stock `LoraLoaderModelOnly` + `KSampler` | pixel-identical, 8 images, both LoRAs, KSampler widgets left at 25 / 2.5 / er_sde / SLG on |
| Turbo edge cases live | batch 2, strength 0, int8 finetune as base, 1344 x 768, img2img at denoise 0.6: all run |
| Telemetry turbo notes | fire for strength 0 and denoise 0.6, silent otherwise |
| Example workflows | all 4 run end to end (`make_examples.py --run`), telemetry clean |
| Canvas UI | MOSS Turbo LoRA draws its banner and rows; KSampler status reads `TURBO · euler / simple · 4 steps · CFG 1.0` when wired |
| Tooltip numbers vs RESULTS.md | Block Merger group shares (all 10), LoRA energy threshold (97.9 / 90.4–91.1%), layout line-break counts, shift and SLG claims, turbo claims: all match |
| Registry package (tracked files minus `.comfyignore`) | nodes, `web/`, 4 examples, `tools/RESULTS.md`; no tests, tools or weights |
| Hugging Face files | both LoRAs and images match the local files by sha256, text files by size; card renders, license apache-2.0, base model linked |

## Findings

| # | Severity | Where | Finding | Fix (status) |
|---|---|---|---|---|
| 1 | Medium | HF model card | Turbo (update 1,600) cells for 2048 / 768 x 1344, explicit prompts and seconds were filled from other checkpoints, not measured on the file. Now measured: explicit 18 / 18, 2048 and 768 x 1344 clean, 6.9 s at 2048 — but the 2048 sign's small line is right in 1 / 3 (akhaliq 2 / 3, Diverse 2 / 3). | Fixed: card cells measured on the file, 2048 small-text and 16:9 sign rows added, uploaded to Hugging Face. |
| 2 | Medium | `tools/RESULTS.md` | The update-1,600 wide test above is not recorded. | Fixed: recorded in RESULTS.md. |
| 3 | Low | tests | The two Telemetry turbo notes (strength, denoise) have no unit test. | Fixed: `test_turbo_notes`, plus the metadata `turbo` output in the turbo test. |
| 4 | Low | MOSS Sampler Metadata | Reports steps 4 / CFG 1 for a turbo run but not the turbo LoRA, so image-saver metadata cannot name it. | Fixed: `turbo` output appended (file:strength, empty without turbo). |
| 5 | Low | MOSS KSampler UI | With turbo wired, the steps / CFG / sampler / scheduler / SLG widgets still show their own values; only the status line says TURBO. | Fixed: the overridden rows are drawn faded, still editable; checked in the canvas. |
| 6 | Low | Telemetry | A turbo LoRA on a model other than SFT (e.g. an int8 finetune) runs without any note, though only SFT was measured. The KSampler cannot see which model it got. | Documented in the README's turbo section. |
| 7 | Cosmetic | `pyproject.toml` | Description misses commas ("...single-stream DiT a model saver..."). | Fixed. |
| 8 | Cosmetic | Layout Compiler info | HADES_META `items` counts text items only, while the text line says "N items, M with text". | Fixed: `items` counts every item, `text_items` the ones with text (Telemetry reads neither). |
| 9 | Info | README on the Registry page | Relative links to `tools/*.py` point at files the Registry package leaves out (pre-existing since 0.1.0). | Fixed: absolute GitHub links. |

No correctness bug was found in the node code. All findings fixed after the audit (CHANGELOG, [Unreleased]).
