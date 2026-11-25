"""Offline checks for tools/moss_pair_check.py; no ComfyUI."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

spec = importlib.util.spec_from_file_location("moss_pair_check", pathlib.Path(__file__).resolve().parents[1] / "tools" / "moss_pair_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def stage(seed=11, model="base.safetensors", prompt="A cup"):
    return [{"node": "loader", "model_file": model},
            {"node": "prompt_enhancer", "prompt": prompt, "negative": "blurry"},
            {"node": "ksampler", "seed": seed, "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
             "flow_shift": 5.0, "denoise": 1.0, "width": 32, "height": 16, "slg": "off"}]


class TestPairCheck(unittest.TestCase):
    def test_model_difference_is_listed_not_rejected(self):
        out = pc.check_pair((stage(model="base.safetensors"), (32, 16)), (stage(model="sft.safetensors"), (32, 16)))
        self.assertEqual(out["mismatches"], [])
        self.assertEqual(len(out["differences"]), 1)
        self.assertIn("prompt", out["matched"])

    def test_settings_prompt_and_size_mismatches(self):
        self.assertIn("seed", pc.check_pair((stage(), None), (stage(seed=12), None))["mismatches"][0])
        self.assertEqual(pc.check_pair((stage(), None), (stage(prompt="A teapot"), None))["mismatches"], ["prompt differs"])
        self.assertEqual(pc.check_pair((stage(), None), (stage(prompt="A teapot"), None), compare_prompt=False)["mismatches"], [])
        self.assertIn("its KSampler 32x16", pc.check_pair((stage(), (64, 16)), (stage(), (32, 16)))["mismatches"][0])

    def test_effective_schedule_is_compared(self):
        """Audit 0.8.0 #7: the same denoise in core and exact mode runs a different schedule."""
        a, b = stage(), stage()
        a[-1].update(denoise=0.9, denoise_mode="core", start_sigma=1.0, schedule_steps=8)
        b[-1].update(denoise=0.9, denoise_mode="exact", start_sigma=0.978)
        mism = pc.check_pair((a, None), (b, None))["mismatches"]
        self.assertTrue(any(m.startswith("denoise_mode") for m in mism) and any(m.startswith("start_sigma") for m in mism))
        old = stage()  # saved before the Suite recorded the schedule
        self.assertIn("denoise_mode (one side was saved before the Suite recorded it)", pc.check_pair((old, None), (b, None))["unverified"])

    def test_one_stage_per_side(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pc.check_pair((stage() + stage()[-1:], None), (stage(), None))

    def test_reads_image_saver_png_and_telemetry_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = PngInfo()
            info.add_text("hades_moss", json.dumps(stage(seed=1)))
            info.add_text("hades_moss_primary_stage", json.dumps(stage()))  # the selected stage wins
            png = pathlib.Path(tmp, "a.png")
            Image.new("RGB", (32, 16)).save(png, pnginfo=info)
            tel = pathlib.Path(tmp, "b.json")
            tel.write_text(json.dumps({"nodes": stage(), "findings": []}), encoding="utf-8")
            self.assertEqual(pc.main([str(png), str(tel)]), 0)
            plain = pathlib.Path(tmp, "c.png")
            Image.new("RGB", (32, 16)).save(plain)
            with self.assertRaisesRegex(ValueError, "no hades_moss chunk"):
                pc.load_side(plain)

    def test_verdict_tells_equality_from_evidence(self):
        """Audit 0.9.6 F16: 'fair pair' was printed with fields left unverified."""
        full = pc.check_pair((stage(), None), (stage(), None))
        self.assertEqual(pc.verdict(full)[1], 0)  # no prompt-schedule fields recorded: fair on what was recorded
        self.assertTrue(pc.verdict(full)[0].startswith("fair on what was recorded"))
        self.assertEqual(pc.verdict(full, strict=True)[1], 2)
        thin = [{"node": "ksampler", "seed": 1, "steps": 8}]
        text, code = pc.verdict(pc.check_pair((thin, None), (thin, None), compare_prompt=False))
        self.assertEqual(code, 2)
        self.assertIn("NOT ENOUGH EVIDENCE", text)
        self.assertEqual(pc.verdict(pc.check_pair((stage(seed=1), None), (stage(seed=2), None)))[1], 1)

    def test_reads_two_branches_of_a_compare_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = PngInfo()
            info.add_text("hades_moss_compare", json.dumps({"labels": ["A", "B"], "branches": {"A": stage(), "B": stage(model="sft.safetensors")}}))
            png = pathlib.Path(tmp, "sheet.png")
            Image.new("RGB", (80, 20)).save(png, pnginfo=info)
            metas, size = pc.load_side(f"{png}:B")
            self.assertIsNone(size)
            self.assertEqual(metas[0]["model_file"], "sft.safetensors")
            self.assertEqual(pc.main([f"{png}:A", f"{png}:B"]), 0)
            with self.assertRaisesRegex(ValueError, "no branch C"):
                pc.load_side(f"{png}:C")


if __name__ == "__main__":
    unittest.main(verbosity=2)
