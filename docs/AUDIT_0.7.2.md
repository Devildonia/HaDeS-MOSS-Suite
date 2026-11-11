# Audit of 0.7.2 and corrections — 2026-11-11

Scope: the whole repository at `53d1789` (0.7.2) before a Comfy Registry release, covering bugs, silent bugs and the project's working rules (no unmeasured claims, no network calls or new dependencies, no AI attribution, CHANGELOG under `[Unreleased]`, at most one Registry release a day, an `info` output with HADES_META on every node, nothing private or internal in what ships).

## What held

- All tests passed before any change: 66 Suite tests (1 opt-in CUDA test skipped), 20 Dev Tools tests, 4 edit-evaluation tests, the JavaScript widget tests.
- No network requests or new dependencies in runtime code. No AI attribution in the commit history. Version strings agree (`__init__.py`, `pyproject.toml`, CHANGELOG).
- int8 ConvRot model + LoRA → int8 save: the LoRA'd layers are requantized from the patched weight and untouched layers are written byte for byte. (An earlier "0 tensors" alarm came from the audit harness reusing a state dict that `load_diffusion_model_state_dict` empties, not from the Saver.)
- Recipe attachments survive patcher clones; the Image Saver's `primary_stage_metadata` selection behaves as documented.

## Findings and corrections

| # | Finding | Correction |
|---|---|---|
| 1 | `tools/RESULTS.md` shipped an internal working instruction ("wait for the user's further manual tests…"), chat attachment identifiers and 15 local drive paths, one of them a user profile folder. | The instruction is now a plain status line. The attachment identifiers are removed. Paths read `<archive>/…` (the maintainer's local measurement archive) and `<ComfyUI>/…`, with a key under the title. Measurements are unchanged. |
| 2 | The README gave a local drive path as a tool default. | It now points to `--out`. |
| 3 | The CHANGELOG had no `[Unreleased]` section. | Restored, and these corrections are listed there. |
| 4 | Release cadence: the CHANGELOG has 0.6.0 and 0.6.1 on 2026-11-10 and 0.7.0, 0.7.1 and 0.7.2 on 2026-11-11. | No code change. Before the next Registry release, check which of these were published. If one was published on 2026-11-11, the next release waits until 2026-11-12. |
| 5 | `web/moss_dev_tools.js` set `report.inputEl` unconditionally; a frontend that gives the widget no `inputEl` (Nodes 2.0 / Vue) would throw while building the node. | Guarded. |
| 6 | No test covered int8 + LoRA → int8 saving. | Regression test added; it checks the patched layer's int8 weight and scale changed and untouched layers are byte-identical. `load_int8` takes a file name, because Windows keeps a loaded file locked. |
| 7 | The Dev Tools (measurement nodes) shipped in the Registry package. Also, `[tool.comfy] includes = ["web", "examples"]` made comfy-cli force those folders in past `.comfyignore`. | `moss_dev_nodes.py`, `web/moss_dev_tools.js` and `examples/dev/` are in `.comfyignore`. `includes` is empty, because both folders are git-tracked and packaged anyway. `__init__.py` and `moss_suite.js` load the Dev Tools only when present. The README says they are repository-only. |
| 8 | Six nodes broke the HADES_META rule. MOSS Turbo LoRA, Sampler Metadata, Image Saver and Suite Telemetry had no `info` output. Pair Guard and Recipe Preview reports had no trailer, so Telemetry counted them as foreign. | The four nodes get an `info` output, added last so saved workflows keep their links. Pair Guard and Recipe Preview add the trailer to `report`; Recipe Preview's trailer comes last, after the recipe's own trailers. Telemetry's `NODE_NAMES` lists all 24 nodes. A new test reads every `format_info` / `format_model_info` call in the source and requires the set of node names to equal `NODE_NAMES`. |

## Verification

- The Registry package was built with comfy-cli's own `zip_files` and the pyproject configuration. It holds 33 files: no Dev Tools, tests or tools except `tools/RESULTS.md`. Extracted and imported on its own in the ComfyUI 0.43.0 venv, it registers 20 nodes with `moss_dev_nodes = None`.
- Final runs: 68 Suite tests (1 opt-in CUDA test skipped), 20 Dev Tools tests, 4 edit-evaluation tests and 5 JavaScript tests, all passing; `git diff --check` is clean.
- Live frontend (ComfyUI 0.43.0, frontend 1.57.4, classic canvas): `HaDeS.MOSSSuiteUI` registers with the new dynamic import. Image Audit and Layout Regions still build their `measurement_report` widget, and a missing module rejects as the guard expects. The server was running Python code older than 0.7.2, so the new `info` outputs are covered by the tests only, not by a live run.

## Not changed

- The measurement scripts in `tools/` keep their local default output folders. They are repository-only developer tools and take `--out`.
- No image-quality or speed claims are added or changed.
