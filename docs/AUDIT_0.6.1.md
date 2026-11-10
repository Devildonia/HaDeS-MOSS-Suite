# Audit corrections — 2026-11-10

Scope agreed with the maintainer: audit items **1, 2, 3, 5, 6, 7, 8, 16, 17, 18, 19 and 20**. Items 4, 9–15 are unchanged. Existing node names, inputs and output socket types are preserved.

## Method

Before production edits, all 49 existing Python tests passed against the installed ComfyUI 0.43.0 core. New regression tests then reproduced the accepted failures. Exports are saved to a temporary test directory and loaded back through the actual ComfyUI loaders. No production model files are used.

| Audit item | Evidence and correction |
|---|---|
| 1 | A low-rank delta plus a full diff on the same weight reloaded with only the diff. Export now writes their sum as a single full diff; the test compares the reloaded delta against both contributions. The affected weight is no longer rank-compressed, so its file representation may be larger. |
| 2 | A saved PNG contained private prompt text even with `disable_metadata=True`. The test now checks that the saved PNG has no text chunks for either workflow-embedding setting. Returned parameter text remains available. |
| 3 | Differently capitalized Windows names bypassed the source guard. Resize and Merger now check filesystem identity. Windows case-alias and hard-link tests confirm rejection and preservation of the source. |
| 5 | With one active task vector of ones and weights `[1, 0]`, TIES returned twos. TIES and DARE-TIES now normalize by the number of nonzero-weight models; the regression expects ones. The existing trimming policy is unchanged. |
| 6 | Diff-only merges saved a file and then raised `ZeroDivisionError`. They now complete and reload correctly. Compression energy is JSON `null`, and Telemetry skips its compression warning for that case. |
| 7 | FP8 → int8 export retained FP8 configuration. The tests check output tensor dtype and configuration, reload the file and compare its patched weight. Another test exercises actual lazy quantized parameter pieces. Matching int8 weights and scales are checked for exact preservation. |
| 8 | NaN, infinity and overflowing numeric text were accepted in strengths / recipes. These inputs now fail at the parsing boundary. Finite extrapolation values remain supported. |
| 16 | Antiparallel SLERP inputs collapsed at intermediate ratios. Zero-length and nearly collinear inputs now use linear interpolation, with exact endpoints. Ordinary spherical interpolation retains its tensor calculation. |
| 17 | Merger info omitted settings needed to reconstruct the merge. It now records ratios / weights, seed and nested source recipes on patcher clones. Sources are identified through the Suite Loader and subsequent Suite merger / LoRA nodes; external loaders remain unidentified. Old recipes are still readable. |
| 18 | The global UI switch selected unrelated HaDeS nodes and deleted individual preferences. JavaScript tests exercise the actual setting functions and confirm isolation and preference restoration. |
| 19 | Updating a referenced LoRA left the reader's cache key unchanged. Dependency fingerprints now include referenced models / LoRAs. Tests cover changed leaves, stable repeated reads and a previously missing model becoming available through a nested merger recipe. |
| 20 | CI previously checked only latest ComfyUI on Linux. It now includes the v0.43.0 baseline on Linux and Windows, retains upstream master, and runs frontend behavior tests. CUDA export checks are opt-in locally. |

## Measurements and limits

Local environment: Windows, installed ComfyUI 0.43.0, PyTorch 2.9.1+cu130. The quantized export checks also passed using CUDA on the tiny model, including FP8 → int8 with a patch and matching int8 preservation.

Final default run: 64 Python tests, 63 passed and the opt-in CUDA test skipped; both JavaScript behavior tests passed. A separate run with `HADES_MOSS_TEST_CUDA=1` passed all 10 saver / sampler tests, including the CUDA test.

The first SLERP guard added unconditional norm calculations. A CUDA microbenchmark exposed that overhead, so the final guard only checks zero norms when cosine similarity is zero. On 2048×2048 float32 tensors, five alternating rounds of 30 calls after warm-up gave median times of **0.235 ms before** and **0.231 ms after**. The normal-input outputs were identical in this check (maximum absolute difference 0). This is a microbenchmark, not a full-model speed claim.

JavaScript behavior tests use a small simulated frontend environment, not a live browser. The new GitHub Actions matrix has not been executed from this local session. No new full-model image-quality, low-VRAM or cross-accelerator measurements are claimed.
