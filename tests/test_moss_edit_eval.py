"""Offline checks for the edit experiment; no ComfyUI queue or model loading."""
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("moss_edit_eval", pathlib.Path(__file__).resolve().parents[1] / "tools" / "moss_edit_eval.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TestEditExperiment(unittest.TestCase):
    def args(self, out):
        return types.SimpleNamespace(out=out, unet="sft.safetensors", clip="text.safetensors", vae="flux2.safetensors",
                                     lora="MOSS/MOSS-edit.safetensors", size=1024, prompt_style="caption", turbo="",
                                     steps=30, cfg=4.0, sampler="euler", scheduler="normal",
                                     reference_inputs={"still_life": "hades_moss_edit_eval/still.png [temp]", "portrait": "hades_moss_edit_eval/person.png [temp]"})

    def test_ablations_change_only_model_patches(self):
        args = self.args(pathlib.Path("experiment"))
        case = bench.CASES["mug_blue"]
        full = bench.edit_graph(args, case, "full", 1000)
        for name in bench.configurations():
            graph = bench.edit_graph(args, case, name, 1000)
            for key in ("clip", "vae", "latent", "ref", "pos", "neg", "dec", "out"):
                self.assertEqual(graph[key], full[key])
            a, b = dict(graph["ks"]["inputs"]), dict(full["ks"]["inputs"])
            a.pop("model")
            b.pop("model")
            self.assertEqual(a, b)
        self.assertEqual(full["latent"]["class_type"], "EmptyFlux2LatentImage")
        self.assertEqual(full["ks"]["inputs"]["denoise"], 1.0)
        self.assertEqual(full["pos"]["inputs"]["edit_format"], "Moss-Image-Edit")
        self.assertEqual(full["neg"]["inputs"]["images.image_0"], ["ref", 0])

    def test_leave_out_and_keep_only_are_complements(self):
        configs = bench.configurations()
        for group in bench.GROUPS:
            suffix = group.removeprefix("layers_")
            keep, leave = configs["in_" + suffix], configs["out_" + suffix]
            self.assertEqual(sum(keep.values()), 1)
            self.assertEqual(sum(leave.values()), 6)
            self.assertTrue(all(keep[g] + leave[g] == 1 for g in bench.GROUPS))

    def test_region_box_uses_image_width_and_height(self):
        mask = bench.region_mask((200, 100), [0.25, 0.1, 0.75, 0.9])
        self.assertEqual(mask.shape, (100, 200))
        self.assertEqual(int(mask.sum()), 100 * 80)
        self.assertTrue(mask[10, 50] and mask[89, 149])
        self.assertFalse(mask[9, 50] or mask[90, 149])
        self.assertEqual(int(bench.region_mask((200, 100), None).sum()), 0)

    def test_changed_image_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "result.png"
            graph, identities = {"test": True}, {"model": "abc"}
            fingerprint = bench.digest(json.dumps({"graph": graph, "models": identities}, sort_keys=True).encode())
            path.write_bytes(b"modified")
            bench.write_json(path.with_suffix(".json"), {"fingerprint": fingerprint, "image_sha256": bench.digest(b"original")})
            with patch.object(bench, "request", side_effect=AssertionError("must not enqueue")):
                with self.assertRaisesRegex(ValueError, "changed"):
                    bench.run_graph(self.args(path.parent), graph, path, identities)


if __name__ == "__main__":
    unittest.main(verbosity=2)
