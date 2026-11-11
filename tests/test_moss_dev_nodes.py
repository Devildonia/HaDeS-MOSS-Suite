"""CPU geometry / numerical tests against real ComfyUI schemas; no models or images on disk."""
import importlib
import json
import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT.parent)]
TEST_ARGV, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
pack = importlib.import_module(ROOT.name)
dev = importlib.import_module(ROOT.name + ".moss_dev_nodes")
common = importlib.import_module(ROOT.name + ".moss_common")
telemetry = importlib.import_module(ROOT.name + ".moss_telemetry_nodes")


def draft():
    return json.dumps({"width": 32, "height": 16, "bbox_coord_space": "abs", "bbox_format": "xyxy",
                       "summary": {"caption": "Geometry control"}, "items": [
                           {"label": "object", "caption": "left", "bbox": [1.2, 2.8, 8.1, 10.2]},
                           {"label": "text", "caption": "overlap", "text": "test", "bbox": [6, 4, 15, 12]},
                           {"label": "character", "caption": "edge", "bbox": [28, 12, 32, 16]}]})


def audit(a, b, **kwargs):
    output = dev.HaDeSMossImageAudit.execute(a, b, **kwargs)
    return output, json.loads(output.args[2])


class TestLayoutRegions(unittest.TestCase):
    def test_fractional_union_and_preview(self):
        out = dev.HaDeSMossLayoutRegions.execute(draft(), "2,1-2")
        expected = np.zeros((1, 16, 32), dtype=np.float32)
        expected[:, 2:11, 1:9] = 1
        expected[:, 4:12, 6:15] = 1
        np.testing.assert_array_equal(out.args[0].numpy(), expected)
        np.testing.assert_array_equal(out.args[1].numpy(), np.repeat(expected[..., None], 3, axis=-1))
        record = json.loads(out.args[2])
        self.assertEqual(record["union_pixels"], int(expected.sum()))
        self.assertEqual([r["index"] for r in record["regions"]], [1, 2])
        self.assertEqual(common.parse_info_meta(out.args[3])["node"], "layout_regions")

    def test_padding_clips_at_all_canvas_edges(self):
        out = dev.HaDeSMossLayoutRegions.execute(draft(), "1,3", 100)
        self.assertTrue(torch.all(out.args[0] == 1))
        for region in json.loads(out.args[2])["regions"]:
            self.assertEqual(region["mask_bbox_xyxy"], [0, 0, 32, 16])

    def test_pixel_edges_match_compiler_near_integer_boundaries(self):
        data = json.loads(draft())
        data["items"][0]["bbox"] = [1.99999999999, 2.8, 8.00000000001, 10.2]
        prompt = json.loads(dev.compile_layout(data)[0])
        y0, x0, y1, x1 = prompt["items"][0]["bbox"]
        expected = [x0 + 16, y0 + 8, x1 + 16, y1 + 8]
        output = dev.HaDeSMossLayoutRegions.execute(json.dumps(data), "1")
        self.assertEqual(json.loads(output.args[2])["regions"][0]["mask_bbox_xyxy"], expected)

    def test_invalid_selection_does_not_silently_choose_boxes(self):
        for selection in ("", "0", "4", "2-1", "1,", "-1", "1.0", "1-999999999"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                dev.HaDeSMossLayoutRegions.execute(draft(), selection)

    def test_caption_and_wrong_coordinate_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "layout draft"):
            dev.HaDeSMossLayoutRegions.execute('{"caption":"test"}')
        data = json.loads(draft())
        data["bbox_format"] = "yxyx"
        with self.assertRaisesRegex(ValueError, "bbox_format"):
            dev.HaDeSMossLayoutRegions.execute(json.dumps(data))


class TestImageAudit(unittest.TestCase):
    def test_identical_images(self):
        a = torch.rand(2, 16, 32, 3)
        out, record = audit(a, a)
        self.assertEqual(torch.count_nonzero(out.args[0]).item(), 0)
        for row in record["results"]:
            self.assertNotIn("inside", row)
            self.assertEqual(row["global"]["mae_255"], 0)
            self.assertEqual(row["global"]["changed_percent"], 0)
            self.assertEqual(row["global"]["hsv_saturation_delta"], 0)

    def test_known_change_with_region_and_independent_numpy(self):
        a = torch.zeros(1, 2, 4, 3)
        b = a.clone()
        b[0, 0, 1, 0] = 1
        mask = torch.zeros(1, 2, 4)
        mask[:, 0, :2] = 1
        a_before, b_before, mask_before = a.clone(), b.clone(), mask.clone()
        out, record = audit(a, b, region_mask=mask, difference_gain=10)
        row = record["results"][0]
        expected = np.abs(b.numpy() - a.numpy()) * 255
        self.assertAlmostEqual(row["global"]["mae_255"], float(expected.mean()), places=5)
        self.assertAlmostEqual(row["inside"]["mae_255"], 255 / 6)
        self.assertEqual(row["inside"]["changed_percent"], 50)
        self.assertEqual(row["outside"]["mae_255"], 0)
        self.assertAlmostEqual(row["inside"]["luma_delta_255"], 255 * 0.2126 / 2, places=5)
        self.assertAlmostEqual(row["inside"]["hsv_saturation_delta"], 0.5)
        self.assertEqual(out.args[0].max().item(), 1)
        self.assertTrue(torch.equal(a, a_before) and torch.equal(b, b_before) and torch.equal(mask, mask_before))

    def test_threshold_is_strict_and_gain_does_not_change_numbers(self):
        a = torch.zeros(1, 1, 2, 3)
        b = a.clone()
        b[0, 0, 0, 0], b[0, 0, 1, 0] = 0.25, 0.5
        _, first = audit(a, b, change_threshold=63.75, difference_gain=0)
        _, second = audit(a, b, change_threshold=63.75, difference_gain=50)
        self.assertEqual(first["results"], second["results"])
        self.assertEqual(first["results"][0]["global"]["changed_percent"], 50)

    def test_empty_regions_are_null_not_zero(self):
        a = torch.zeros(1, 2, 4, 3)
        for value, empty in ((0, "inside"), (1, "outside")):
            _, record = audit(a, a, region_mask=torch.full((1, 2, 4), value))
            self.assertIsNone(record["results"][0][empty])

    def test_batches_are_paired_and_mask_broadcast_is_explicit(self):
        a = torch.zeros(2, 2, 4, 3)
        b = a.clone()
        b[1] = 1
        _, shared = audit(a, b, region_mask=torch.ones(1, 2, 4))
        self.assertEqual(shared["results"][0]["inside"]["mae_255"], 0)
        self.assertEqual(shared["results"][1]["inside"]["mae_255"], 255)
        masks = torch.stack((torch.zeros(2, 4), torch.ones(2, 4)))
        _, paired = audit(a, b, region_mask=masks)
        self.assertIsNone(paired["results"][0]["inside"])
        self.assertIsNone(paired["results"][1]["outside"])

    def test_fractional_masks_use_declared_binary_boundary(self):
        a = torch.zeros(1, 1, 2, 3)
        _, record = audit(a, a, region_mask=torch.tensor([[[0.49, 0.5]]]))
        self.assertEqual(record["results"][0]["inside"]["pixels"], 1)

    def test_no_implicit_resize_or_image_broadcast(self):
        a = torch.zeros(2, 2, 4, 3)
        for b in (a[:1], torch.zeros(2, 4, 2, 3), torch.zeros(2, 2, 4, 4)):
            with self.assertRaisesRegex(ValueError, "identical"):
                audit(a, b)
        for mask in (torch.zeros(2, 4), torch.zeros(3, 2, 4), torch.zeros(1, 4, 2)):
            with self.assertRaisesRegex(ValueError, "mask must"):
                audit(a, a, region_mask=mask)

    def test_invalid_values_fail_clearly(self):
        a = torch.zeros(1, 2, 4, 3)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            b = a.clone()
            b[0, 0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "finite RGB"):
                audit(a, b)
        for kwargs in ({"change_threshold": float("nan")}, {"change_threshold": -1}, {"difference_gain": float("inf")},
                       {"region_mask": torch.full((1, 2, 4), float("nan"))}):
            with self.assertRaises(ValueError):
                audit(a, a, **kwargs)

    def test_regions_to_audit_round_trip(self):
        mask = dev.HaDeSMossLayoutRegions.execute(draft(), "3").args[0]
        a = torch.zeros(1, 16, 32, 3)
        b = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        _, record = audit(a, b, region_mask=mask)
        self.assertEqual(record["results"][0]["inside"]["mae_255"], 255)
        self.assertEqual(record["results"][0]["outside"]["mae_255"], 0)


class TestRegistration(unittest.TestCase):
    def test_v3_schemas_and_legacy_pack_registration(self):
        for node_id, cls in dev.NODE_CLASS_MAPPINGS.items():
            self.assertIs(pack.NODE_CLASS_MAPPINGS[node_id], cls)
            info = cls.GET_NODE_INFO_V1()
            self.assertEqual(info["category"], "HaDeS/MOSS Dev Tools")
            self.assertTrue(info["output_node"])
            self.assertTrue(info["output"])
        text = dev.HaDeSMossLayoutRegions.execute(draft(), "1").args[3]
        result = telemetry.HaDeSMossSuiteTelemetry().audit("dev test", info_1=text)
        self.assertIn("Layout Regions", result["result"][0])


def stage(seed=11, model="base.safetensors", prompt="A cup"):
    return json.dumps({"nodes": [
        {"node": "loader", "model_file": model},
        {"node": "prompt_enhancer", "prompt": prompt, "negative": "blurry"},
        {"node": "ksampler", "seed": seed, "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
         "flow_shift": 5.0, "denoise": 1.0, "width": 32, "height": 16, "slg": "off"}]})


class TestPairGuard(unittest.TestCase):
    def test_pass_through_and_allowed_model_difference(self):
        a, b = torch.zeros(2, 16, 32, 3), torch.ones(2, 16, 32, 3)
        out = dev.HaDeSMossPairGuard.execute(a, b, stage(model="base.safetensors"), stage(model="sft.safetensors"))
        self.assertIs(out.args[0], a)
        self.assertIs(out.args[1], b)
        record = json.loads(out.args[3])
        self.assertEqual(record["status"], "pass")
        self.assertEqual(record["batch_pairs"], 2)
        self.assertEqual(len(record["allowed_differences"]), 1)
        self.assertIn("PASS", out.args[2])

    def test_shape_metadata_and_prompt_mismatches_fail(self):
        a = torch.zeros(1, 16, 32, 3)
        with self.assertRaisesRegex(ValueError, "image shape mismatch"):
            dev.HaDeSMossPairGuard.execute(a, torch.zeros(1, 32, 16, 3))
        with self.assertRaisesRegex(ValueError, "one-sided"):
            dev.HaDeSMossPairGuard.execute(a, a, stage(), None)
        with self.assertRaisesRegex(ValueError, "seed"):
            dev.HaDeSMossPairGuard.execute(a, a, stage(), stage(seed=12))
        with self.assertRaisesRegex(ValueError, "prompt"):
            dev.HaDeSMossPairGuard.execute(a, a, stage(), stage(prompt="A teapot"))
        with self.assertRaisesRegex(ValueError, "image width"):
            dev.HaDeSMossPairGuard.execute(a, a, stage().replace('"width": 32', '"width": 64'), stage())

    def test_missing_metadata_is_explicit_and_multistage_rejected(self):
        a = torch.zeros(1, 16, 32, 3)
        record = json.loads(dev.HaDeSMossPairGuard.execute(a, a).args[3])
        self.assertFalse(record["metadata_checked"])
        self.assertTrue(record["unverified"])
        with self.assertRaisesRegex(ValueError, "require_metadata"):
            dev.HaDeSMossPairGuard.execute(a, a, require_metadata=True)
        multi = json.loads(stage())
        multi["nodes"].append(multi["nodes"][-1])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            dev.HaDeSMossPairGuard.execute(a, a, json.dumps(multi), stage())
        with self.assertRaisesRegex(ValueError, "invalid Suite trailer"):
            dev.HaDeSMossPairGuard.execute(a, a, "HADES_META []", stage())


class TestRecipePreview(unittest.TestCase):
    def test_preserves_full_recipe_and_checks_attachment(self):
        attachment = {"node": "multi_lora", "loras": [{"name": "turbo", "strength": 1.0}]}
        source = common.format_info("Multi-LoRA | turbo", **attachment)

        class Model:
            def get_attachment(self, name):
                self_name = name
                assert self_name == "hades_moss_recipe"
                return attachment

        out = dev.HaDeSMossRecipePreview.execute(source, model=Model())
        self.assertEqual(out.args[0], source)
        self.assertTrue(json.loads(out.args[2])["model_attachment_match"])
        self.assertIn("turbo", out.args[1])

    def test_empty_malformed_and_metadata_off(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            dev.HaDeSMossRecipePreview.execute(" ")
        with self.assertRaisesRegex(ValueError, "malformed"):
            dev.HaDeSMossRecipePreview.execute("notes\nHADES_META {broken")
        with self.assertRaisesRegex(ValueError, "malformed"):
            dev.HaDeSMossRecipePreview.execute("notes\nHADES_META []")
        out = dev.HaDeSMossRecipePreview.execute("hand-written recipe", include_metadata=False)
        record = json.loads(out.args[2])
        self.assertFalse(record["will_embed_if_saver_matches"])
        self.assertIn("No structured", out.args[1])


if __name__ == "__main__":
    unittest.main(argv=TEST_ARGV, verbosity=2)
