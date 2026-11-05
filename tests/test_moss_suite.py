"""HaDeS MOSS Suite tests against the real ComfyUI core, on a tiny synthetic MOSS-Image model (CPU, a few MB).

Run from a ComfyUI checkout with this pack in custom_nodes:
    python custom_nodes/<this pack>/tests/test_moss_suite.py
"""
import importlib
import json
import os
import sys
import tempfile
import unittest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PACK_DIR))
TEST_ARGV, sys.argv = sys.argv, [sys.argv[0], "--cpu"]  # ComfyUI parses sys.argv on import; unittest gets the real arguments

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import safetensors.torch  # noqa: E402
import comfy.lora  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
from comfy.ldm.moss_image.model import MOSSImage  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

PACK = os.path.basename(PACK_DIR)
common = importlib.import_module(f"{PACK}.moss_common")
mergers = importlib.import_module(f"{PACK}.moss_merger_nodes")
loras = importlib.import_module(f"{PACK}.moss_lora_nodes")
pipeline = importlib.import_module(f"{PACK}.moss_pipeline_nodes")
telemetry = importlib.import_module(f"{PACK}.moss_telemetry_nodes")
saver = importlib.import_module(f"{PACK}.moss_saver_nodes")

DIM, LAYERS = 256, 6  # 2 heads of 128; 256 keeps every linear a multiple of the ConvRot group size
TMP = tempfile.mkdtemp(prefix="hades_moss_tests_")
folder_paths.folder_names_and_paths["loras"] = ([TMP], {".safetensors"})
folder_paths.set_output_directory(TMP)
sampler = importlib.import_module(f"{PACK}.moss_sampler_nodes")


def tiny_state_dict(seed):
    """A MOSS-Image state dict (fused q/k/v) with random weights: dim 256, 6 layers, the real 4096-wide text projection and 128 latent channels."""
    g = torch.Generator().manual_seed(seed)
    model = MOSSImage(dim=DIM, ffn_dim=2 * DIM, num_heads=DIM // 128, num_layers=LAYERS, fused_qkv=True, operations=comfy.ops.disable_weight_init)
    return {k: (torch.ones(v.shape) if "norm" in k else torch.randn(v.shape, generator=g) * 0.02).to(torch.bfloat16)
            for k, v in model.state_dict().items()}


def load(sd):
    return comfy.sd.load_diffusion_model_state_dict({k: v.clone() for k, v in sd.items()})


def load_int8(sd):
    """The state dict saved and loaded back in Comfy-Org's int8 ConvRot layout (layer linears int8, int8 attention)."""
    path = os.path.join(TMP, "q.safetensors")
    out = {}
    for k, v in sd.items():
        if saver._INT8_LAYER_LINEAR_RE.match(k):
            qt = QuantizedTensor.from_float(v, "TensorWiseINT8Layout", is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
            out.update(qt.state_dict(k))
            out[f"{k[:-len('.weight')]}.comfy_quant"] = saver._json_tensor(saver._INT8_QUANT_CONF)
        else:
            out[k] = v
    for i in range(LAYERS):
        out[f"layers.{i}.comfy_attention.config"] = saver._json_tensor(saver._INT8_ATTENTION_CONF)
    safetensors.torch.save_file(out, path)
    return comfy.sd.load_diffusion_model(path)


def stored(model, key, value):
    """`value` rounded to the dtype the loaded model stores `key` in (float32 on CPU, so no rounding there), back in float32."""
    return value.to(comfy.utils.get_attr(model.model.diffusion_model, key).dtype).float()


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def patched_weight(model, key):
    """The weight `key` (no prefix) as the core computes it from the model's patches, in float32."""
    pk = f"diffusion_model.{key}"
    return common.key_patches_weight(common.key_patches(model, pk), "cpu", pk) if pk in model.patches else \
        comfy.utils.get_attr(model.model.diffusion_model, key).float()


class TestLayoutCompiler(unittest.TestCase):
    # Expected strings: this compiler's output, checked byte for byte against OpenMOSS' scripts/compile_prompt.py on 2026-11-03
    # (layout.md example, fractional boxes on 1360x768, caption drafts); the box below is the one layout.md documents.
    DOC_EXAMPLE = {
        "width": 1536, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
        "summary": {"meta": ["product campaign"], "canvas": "single", "caption": "A ceramic cup campaign."},
        "items": [{"label": "object", "caption": "An ivory ceramic cup.", "bbox": [176, 224, 800, 864]},
                  {"label": "text", "caption": "Supporting copy on two lines.", "text": "一杯温暖\n一刻从容", "bbox": [928, 512, 1376, 656]}],
    }

    def test_layout_md_box(self):
        prompt, w, h, texts = pipeline.compile_layout(json.loads(json.dumps(self.DOC_EXAMPLE)))
        out = json.loads(prompt)
        self.assertEqual(out["items"][0]["bbox"], [-288, -592, 352, 32])  # layout.md: "conversion yields [-288,-592,352,32]"
        self.assertEqual((out["bbox_coord_space"], out["bbox_format"], w, h), ("abs_centralized", "yxyx", 1536, 1024))
        self.assertNotIn("width", out)
        self.assertIn("一杯温暖\\n一刻从容", prompt)  # line break kept as the JSON escape, characters unescaped
        self.assertEqual(texts[0][1:], (448, 144))

    def test_fractional_boxes_round_outward(self):
        d = {"width": 1360, "height": 768, "bbox_coord_space": "abs", "bbox_format": "xyxy", "summary": {"caption": "c"},
             "items": [{"label": "object", "caption": "o", "bbox": [160.4, 63.7, 863.2, 544.9]}]}
        self.assertEqual(json.loads(pipeline.compile_layout(d)[0])["items"][0]["bbox"], [63 - 384, 160 - 680, 545 - 384, 864 - 680])

    def test_caption_draft_passes_through(self):
        self.assertEqual(pipeline.compile_layout({"caption": "A plain caption."})[0], "A plain caption.")

    def test_rejections(self):
        cases = {
            "outside": lambda d: d["items"][0].__setitem__("bbox", [0, 0, 2000, 100]),
            "not multiple of 16": lambda d: d.__setitem__("width", 1000),
            "unknown field": lambda d: d["items"][0].__setitem__("id", 3),
            "text without text": lambda d: d["items"][1].pop("text"),
            "centered space": lambda d: d.__setitem__("bbox_coord_space", "abs_centralized"),
            "bad label": lambda d: d["items"][0].__setitem__("label", "logo"),
        }
        for name, mutate in cases.items():
            d = json.loads(json.dumps(self.DOC_EXAMPLE))
            mutate(d)
            with self.subTest(name), self.assertRaises(ValueError):
                pipeline.compile_layout(d)

    def test_node_latent_and_duplicate_keys(self):
        node = pipeline.HaDeSMossLayoutCompiler()
        prompt, latent, w, h, info = node.compile(json.dumps(self.DOC_EXAMPLE), 2)
        self.assertEqual(tuple(latent["samples"].shape), (2, 128, 64, 96))
        self.assertEqual(common.parse_info_meta(info)["multiline"], [[2, 448, 144]])
        with self.assertRaises(ValueError):
            node.compile('{"caption": "a", "caption": "b"}', 1)


class TestLatent(unittest.TestCase):
    def test_sizes(self):
        node = pipeline.HaDeSMossLatent()
        for aspect, mp, size in (("1:1", 1.0, (1024, 1024)), ("16:9", 1.0, (1360, 768)), ("1:1", 4.0, (2048, 2048))):
            latent, w, h, _ = node.generate(aspect, mp, 1024, 1024, 1)
            self.assertEqual((w, h), size)
            self.assertEqual(tuple(latent["samples"].shape), (1, 128, h // 16, w // 16))
        self.assertEqual(node.generate("custom", 1.0, 1010, 600, 1)[1:3], (1008, 608))


class TestMergers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sd_a, cls.sd_b = tiny_state_dict(1), tiny_state_dict(2)
        cls.a, cls.b = load(cls.sd_a), load(cls.sd_b)

    def test_ratio_zero_adds_no_patch(self):
        m, merged, n, quantized, scale = common.apply_block_merge(self.a, self.b, [0.0] * LAYERS, dict.fromkeys(common.PARTS, 0.0))
        self.assertEqual((merged, len(m.patches), n, quantized, scale), (0, 0, LAYERS, [], None))

    def test_ratio_one_and_half(self):
        for ratio in (1.0, 0.5):
            m = common.apply_block_merge(self.a, self.b, [ratio] * LAYERS, dict.fromkeys(common.PARTS, ratio))[0]
            for key in ("layers.3.proj.qkv.weight", "text_embedder.proj.weight", "shared_block_modulation.bias"):
                expected = stored(self.a, key, self.sd_a[key].float() + (self.sd_b[key].float() - self.sd_a[key].float()) * ratio)
                with self.subTest(ratio=ratio, key=key):
                    torch.testing.assert_close(patched_weight(m, key), expected, rtol=0, atol=1e-6)

    def test_group_ratios_follow_layers(self):
        ratios = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        m = common.apply_block_merge(self.a, self.b, ratios, dict.fromkeys(common.PARTS, 0.0))[0]
        torch.testing.assert_close(patched_weight(m, "layers.2.proj.o.weight"), self.sd_b["layers.2.proj.o.weight"].float())
        torch.testing.assert_close(patched_weight(m, "layers.1.proj.o.weight"), self.sd_a["layers.1.proj.o.weight"].float())

    def test_fp16_fixed_rescale(self):
        """model2 = model1 with every residual write x 1/16 (as miaomiaoHaremMOSS_baseV11_fp16): detected, merged back to model1."""
        sd = {k: v.clone() for k, v in self.sd_a.items()}
        for k in sd:
            if common._RESIDUAL_WRITER_RE.match(k):
                sd[k] = (sd[k].float() / 16).to(torch.bfloat16)
            elif k.startswith("shared_block_modulation."):
                rows = sd[k].float().clone()
                for c in common._MODULATION_GATES:
                    rows.view(6, -1, *rows.shape[1:])[c] /= 16
                sd[k] = rows.to(torch.bfloat16)
        fixed = load(sd)
        m, _, _, _, scale = common.apply_block_merge(self.a, fixed, [1.0] * LAYERS, dict.fromkeys(common.PARTS, 1.0))
        self.assertAlmostEqual(scale, 16.0, places=3)
        for key in ("video_embedder.weight", "text_embedder.blocks.1.ffn_down.bias", "shared_block_modulation.weight"):
            with self.subTest(key=key):
                torch.testing.assert_close(patched_weight(m, key), self.sd_a[key].float(), rtol=0, atol=2e-3)

    def test_quantized_input_is_dequantized(self):
        q = load_int8(self.sd_b)
        key = "layers.4.proj.ffn_down.weight"
        self.assertIsInstance(comfy.utils.get_attr(q.model.diffusion_model, key), QuantizedTensor)
        m, _, _, quantized, _ = common.apply_block_merge(self.a, q, [1.0] * LAYERS, dict.fromkeys(common.PARTS, 1.0))
        self.assertEqual(quantized, ["model2"])
        deq = stored(self.a, key, comfy.utils.get_attr(q.model.diffusion_model, key).dequantize().float().to(torch.bfloat16).float())
        torch.testing.assert_close(patched_weight(m, key), deq, rtol=0, atol=1e-6)

    def test_multi_merge_task_arithmetic(self):
        node = mergers.HaDeSMossMultiModelMerger()
        m, info = node.merge(self.b, load(tiny_state_dict(3)), "Task Arithmetic", 0.5, 1.0, 0, 1.0, 1.0, base_model=self.a)
        key = "layers.0.proj.ffn_up_gate.weight"
        sd_c = tiny_state_dict(3)
        base = self.sd_a[key].float()
        expected = stored(self.b, key, base + ((self.sd_b[key].float() - base) + (sd_c[key].float() - base)) / 2)
        torch.testing.assert_close(patched_weight(m, key), expected, rtol=0, atol=1e-6)
        self.assertEqual(common.parse_info_meta(info)["method"], "Task Arithmetic")


class TestLoras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sd = tiny_state_dict(4)
        cls.model = load(cls.sd)
        g = torch.Generator().manual_seed(5)
        r = 4
        cls.split, cls.fused = {}, {}
        for i in range(LAYERS):
            for p in ("q", "k", "v"):  # kohya, split q/k/v
                cls.split[f"lora_unet_layers_{i}_proj_{p}.lora_down.weight"] = torch.randn(r, DIM, generator=g) * 0.1
                cls.split[f"lora_unet_layers_{i}_proj_{p}.lora_up.weight"] = torch.randn(DIM, r, generator=g) * 0.1
                cls.split[f"lora_unet_layers_{i}_proj_{p}.alpha"] = torch.tensor(2.0)
            cls.fused[f"diffusion_model.layers.{i}.proj.qkv.lora_A.weight"] = torch.randn(r, DIM, generator=g) * 0.1  # PEFT, fused
            cls.fused[f"diffusion_model.layers.{i}.proj.qkv.lora_B.weight"] = torch.randn(3 * DIM, r, generator=g) * 0.1
            cls.fused[f"diffusion_model.layers.{i}.proj.o.lora_A.weight"] = torch.randn(r, DIM, generator=g) * 0.1
            cls.fused[f"diffusion_model.layers.{i}.proj.o.lora_B.weight"] = torch.randn(DIM, r, generator=g) * 0.1
        safetensors.torch.save_file(cls.split, os.path.join(TMP, "split.safetensors"))
        safetensors.torch.save_file(cls.fused, os.path.join(TMP, "fused.safetensors"))

    def core_delta(self, name, key, strength=1.0):
        patches = loras.load_lora_patches(name, self.model)
        ps = [(strength, p, 1.0, k[1] if isinstance(k, tuple) else None, None) for k, p in patches.items() if (k[0] if isinstance(k, tuple) else k) == key]
        return comfy.lora.calculate_weight(ps, torch.zeros(comfy.utils.get_attr(self.model.model, key).shape), key)

    def test_factors_match_core(self):
        key = "diffusion_model.layers.2.proj.qkv.weight"
        for name in ("split.safetensors", "fused.safetensors"):
            factors, diffs = loras.lora_factors(name, self.model, 0.7, "cpu")
            up, down = factors[key]
            with self.subTest(name):
                torch.testing.assert_close(up @ down, self.core_delta(name, key, 0.7), rtol=1e-5, atol=1e-6)
                self.assertEqual(up.shape[1], 12 if name == "split.safetensors" else 4)  # split q/k/v stack to 3 x rank

    def test_compress_is_exact_at_full_rank(self):
        up, down = loras.lora_factors("split.safetensors", self.model, 1.0, "cpu")[0]["diffusion_model.layers.0.proj.qkv.weight"]
        cu, cd, kept = loras.compress(up, down, 1000, 0.0)
        torch.testing.assert_close(cu @ cd, up @ down, rtol=1e-4, atol=1e-6)
        self.assertAlmostEqual(kept, 1.0, places=5)

    def test_block_weight_scales_groups(self):
        node = loras.HaDeSMossLoraBlockWeight()
        full = dict.fromkeys(mergers.GROUPS, 1.0)
        m, info = node.apply(self.model, "fused.safetensors", 1.0, 1.0, **{**full, mergers.GROUPS[2]: 0.0})
        counts = common.parse_info_meta(info)["counts"]
        self.assertEqual(sum(counts.values()), 2 * LAYERS)
        # layer 2 sits in group 2 on a 6-layer model (proportional split onto 7 groups): no patch there, layer 3 patched
        self.assertNotIn("diffusion_model.layers.2.proj.o.weight", m.patches)
        self.assertIn("diffusion_model.layers.3.proj.o.weight", m.patches)

    def test_merger_and_resize_files_load_in_core(self):
        merged_path = loras.HaDeSMossLoraMerger().merge(self.model, "split.safetensors", 1.0, "fused.safetensors", 1.0, 64, "merged", overwrite_existing=True)["result"][0]
        key = "diffusion_model.layers.1.proj.qkv.weight"
        expected = self.core_delta("split.safetensors", key) + self.core_delta("fused.safetensors", key)
        # rank 64 >= 12 + 4 keeps the merge exact; what is left is the bfloat16 storage of the factors
        self.assertLess(rel_err(self.core_delta(os.path.basename(merged_path), key), expected), 1e-2)
        resized = loras.HaDeSMossLoraResize().resize(self.model, "split.safetensors", 2, "resized", overwrite_existing=True)["result"][1]
        self.assertLess(common.parse_info_meta(resized)["energy_kept"], 1.0)

    def test_extractor_recovers_low_rank_change(self):
        sd = {k: v.clone() for k, v in self.sd.items()}
        g = torch.Generator().manual_seed(6)
        delta = (torch.randn(DIM, 3, generator=g) @ torch.randn(3, DIM, generator=g)) * 0.01
        sd["layers.5.proj.o.weight"] = (sd["layers.5.proj.o.weight"].float() + delta).to(torch.bfloat16)
        tuned = load(sd)
        path, info = loras.HaDeSMossLoraExtractor().extract(tuned, self.model, 8, "extracted", overwrite_existing=True)["result"]
        meta = common.parse_info_meta(info)
        self.assertEqual(meta["layers"], 1)
        self.assertAlmostEqual(meta["energy_kept"], 1.0, places=3)
        key = "diffusion_model.layers.5.proj.o.weight"
        # the stored delta also holds the bfloat16 rounding of the tuned weight, which rank 8 leaves out
        self.assertLess(rel_err(self.core_delta(os.path.basename(path), key), delta), 5e-2)


class TestSaverAndSampler(unittest.TestCase):
    def test_int8_model_saved_as_bf16_holds_no_int8(self):
        q = load_int8(tiny_state_dict(7))
        for precision, dtype in (("bfloat16", "BF16"), ("float32", "F32")):
            path = saver.HaDeSMossModelSaver().save(q, "tests/save", precision, False)["result"][1]
            with safetensors.safe_open(path, "pt") as f:
                keys = list(f.keys())
                with self.subTest(precision):
                    self.assertEqual({f.get_slice(k).get_dtype() for k in keys if k.endswith("proj.qkv.weight")}, {dtype})
                    self.assertFalse([k for k in keys if k.endswith(("comfy_quant", "weight_scale", "comfy_attention.config"))])
                    deq = comfy.utils.get_attr(q.model.diffusion_model, "layers.2.proj.qkv.weight").dequantize().float().to(torch.bfloat16).float()
                    torch.testing.assert_close(f.get_tensor("layers.2.proj.qkv.weight").float(), deq, rtol=0, atol=0)

    def test_int8_output_matches_comfy_org_layout(self):
        path = saver.HaDeSMossModelSaver().save(load(tiny_state_dict(8)), "tests/save", "int8 ConvRot (Comfy-Org layout)", False)["result"][1]
        with safetensors.safe_open(path, "pt") as f:
            keys = set(f.keys())
            self.assertEqual(f.get_slice("layers.0.proj.qkv.weight").get_dtype(), "I8")
            self.assertEqual(f.get_slice("text_embedder.proj.weight").get_dtype(), "BF16")
            self.assertTrue({"layers.0.proj.qkv.weight_scale", "layers.0.proj.qkv.comfy_quant", "layers.5.comfy_attention.config"} <= keys)

    def test_image_output_without_vae_is_reported(self):
        prompt = {"9": {"class_type": "HaDeSMossKSampler", "inputs": {}}, "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 1]}}}
        with self.assertRaisesRegex(ValueError, "no VAE"):
            sampler.HaDeSMossKSampler().sample(None, None, None, None, 0, 1, 1.0, "euler", "simple", 1.0, 0.0, "off", prompt=prompt, unique_id="9")

    def test_turbo_applies_lora_and_its_settings(self):
        g = torch.Generator().manual_seed(9)
        name = "layers.3.proj.o"
        lora = {f"diffusion_model.{name}.lora_up.weight": torch.randn(DIM, 2, generator=g), f"diffusion_model.{name}.lora_down.weight": torch.randn(2, DIM, generator=g),
                f"diffusion_model.{name}.alpha": torch.tensor(2.0)}
        safetensors.torch.save_file(lora, os.path.join(TMP, "turbo_test.safetensors"))
        turbo = sampler.HaDeSMossTurbo().pick("turbo_test.safetensors", 1.0)[0]
        seen = {}

        def fake_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
            seen.update(steps=steps, cfg=cfg, sampler=sampler_name, scheduler=scheduler, patched=sorted(model.patches))
            return ({"samples": torch.zeros(1, 128, 4, 4)},)

        original, sampler.nodes.common_ksampler = sampler.nodes.common_ksampler, fake_ksampler
        try:
            info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 2.5, "er_sde", "beta", 1.0, 0.0,
                                                      "block 22, all steps", turbo=turbo)[2]
        finally:
            sampler.nodes.common_ksampler = original
        self.assertEqual({k: seen[k] for k in ("steps", "cfg", "sampler", "scheduler")}, {"steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"})
        self.assertEqual(seen["patched"], [f"diffusion_model.{name}.weight"])
        meta = common.parse_info_meta(info)
        self.assertEqual((meta["turbo"], meta["slg"], meta["steps"]), ("turbo_test.safetensors", "off", 4))
        values = sampler.HaDeSMossSamplerMetadata().read(info)["result"]
        self.assertEqual((values[1], values[2], values[7]), (4, 1.0, "turbo_test.safetensors:1"))
        # the turbo node's steps widget replaces the default 4; the KSampler's own steps (25) stay ignored
        sampler.nodes.common_ksampler = fake_ksampler
        try:
            info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 2.5, "er_sde", "beta", 1.0, 0.0,
                                                      "off", turbo=sampler.HaDeSMossTurbo().pick("turbo_test.safetensors", 1.0, 6)[0])[2]
        finally:
            sampler.nodes.common_ksampler = original
        self.assertEqual((seen["steps"], common.parse_info_meta(info)["steps"]), (6, 6))


class TestTelemetry(unittest.TestCase):
    def test_turbo_notes(self):
        f = common.format_info
        base = dict(width=1024, height=1024, steps=4, cfg=1.0, flow_shift=5.0, sampler="euler", scheduler="simple", seed=0, slg="off")
        cases = {"measured": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0), 0),
                 "strength": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=0.5), 1),
                 "denoise": (dict(denoise=0.6, turbo="t.safetensors", turbo_strength=1.0), 1),
                 "steps": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, steps=6), 1),
                 "8 steps": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, steps=8), 0),
                 "no turbo, denoise": (dict(denoise=0.6), 0),
                 "no turbo, 25 steps": (dict(denoise=1.0, steps=25), 0)}
        for name, (extra, notes) in cases.items():
            with self.subTest(name):
                data = telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=f("x", "ksampler", **{**base, **extra}))["result"][1]
                self.assertEqual(sum("Turbo" in x["message"] for x in json.loads(data)["findings"]), notes)

    def test_findings(self):
        f = common.format_info
        infos = [f("x", "layout", width=1024, height=1024, items=1, multiline=[[2, 832, 192]]),
                 f("x", "latent", width=1360, height=768, batch=1),
                 f("x", "multi_model_merger", method="TIES", models=2, density=0.5, scale=1.0, modified_tensors=1, quantized=["model_2"], rescaled=[]),
                 f("x", "lora_resize", saved_path="p", rank=16, layers=1, energy_kept=0.85),
                 "foreign text"]
        report, data, summary = telemetry.HaDeSMossSuiteTelemetry().audit("T", **{f"info_{i + 1}": s for i, s in enumerate(infos)})["result"]
        levels = [x["level"] for x in json.loads(data)["findings"]]
        self.assertEqual((levels.count("error"), levels.count("warning"), levels.count("note")), (1, 3, 1))
        self.assertIn("1 input(s) without a HaDeS MOSS trailer ignored", report)

    def test_clean_pipeline(self):
        f = common.format_info
        infos = [f("x", "latent", width=1024, height=1024, batch=1),
                 f("x", "ksampler", width=1024, height=1024, steps=25, cfg=2.5, flow_shift=5.0, sampler="er_sde", scheduler="simple", seed=1, denoise=1.0, slg="off")]
        self.assertEqual(json.loads(telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=infos[0], info_2=infos[1])["result"][1])["findings"], [])


if __name__ == "__main__":
    unittest.main(argv=TEST_ARGV, verbosity=2)
