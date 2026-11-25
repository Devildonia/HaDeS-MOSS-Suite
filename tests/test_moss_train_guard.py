"""Tests of tools/moss_train_guard.py (audit 0.9.6 F01 / F09): one run per output folder, and only the run it was started with.
Run from tests/: H:/Comfy_UI_V61/ComfyUI/venv/Scripts/python.exe test_moss_train_guard.py"""
import argparse
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import moss_train_guard as guard  # noqa: E402


def args(**kw):
    base = {"unet": "base.safetensors", "lr": 1e-4, "seed": 0, "updates": 3000, "save_every": 200, "out": "x", "checkpoint": False}
    return argparse.Namespace(**{**base, **kw})


class TestTrainGuard(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="hades_guard_")
        self.prompts = os.path.join(self.out, "..", f"prompts_{os.getpid()}.json")
        with open(self.prompts, "w", encoding="utf-8") as f:
            json.dump(["a cat", "a dog"], f)

    def test_the_lock_is_exclusive_and_released(self):
        with guard.run_lock(self.out):
            with self.assertRaises(SystemExit):
                with guard.run_lock(self.out):
                    pass
        with guard.run_lock(self.out):  # released after the run
            pass
        self.assertFalse(os.path.exists(os.path.join(self.out, "TRAIN_RUNNING")))

    def test_a_resume_must_be_the_same_run(self):
        guard.seal_run(self.out, args(), {"prompts_file": self.prompts})
        guard.seal_run(self.out, args(), {"prompts_file": self.prompts})  # same run: resumes
        guard.seal_run(self.out, args(updates=4000, save_every=100), {"prompts_file": self.prompts})  # operational: allowed
        with self.assertRaisesRegex(SystemExit, "args.lr"):
            guard.seal_run(self.out, args(lr=5e-5), {"prompts_file": self.prompts})
        with self.assertRaisesRegex(SystemExit, "args.unet"):
            guard.seal_run(self.out, args(unet="sft.safetensors"), {"prompts_file": self.prompts})
        with open(self.prompts, "w", encoding="utf-8") as f:
            json.dump(["a cat", "a horse"], f)  # one prompt changed in the file: another dataset
        with self.assertRaisesRegex(SystemExit, "files.prompts_file"):
            guard.seal_run(self.out, args(), {"prompts_file": self.prompts})

    def test_folders_are_identified_by_their_listing(self):
        d = tempfile.mkdtemp(prefix="hades_pairs_")
        open(os.path.join(d, "a.png"), "wb").close()
        first = guard.file_identity(d)
        with open(os.path.join(d, "b.png"), "wb") as f:
            f.write(b"x")
        self.assertNotEqual(first, guard.file_identity(d))

    def test_per_index_cache_keys_follow_the_prompt(self):
        self.assertNotEqual(guard.prompt_tag("a cat"), guard.prompt_tag("a dog"))
        self.assertEqual(guard.prompt_tag("a cat"), guard.prompt_tag("a cat"))

    def test_atomic_save_leaves_no_temporary_file(self):
        path = os.path.join(self.out, "cache.pt")
        guard.save_torch_atomic({"x": 1}, path)
        self.assertEqual(sorted(os.listdir(self.out)), ["cache.pt"])


if __name__ == "__main__":
    unittest.main()
