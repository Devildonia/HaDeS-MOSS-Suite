"""HaDeS MOSS Suite tests against the real ComfyUI core, on a tiny synthetic MOSS-Image model (CPU, a few MB).

Run from a ComfyUI checkout with this pack in custom_nodes:
    python custom_nodes/<this pack>/tests/test_moss_suite.py
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

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
enhancer = importlib.import_module(f"{PACK}.moss_enhancer_nodes")
metadata = importlib.import_module(f"{PACK}.moss_metadata_nodes")
control = importlib.import_module(f"{PACK}.moss_control_nodes")

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


def load(sd, dtype=None):
    """dtype=torch.bfloat16 keeps the weights in bfloat16 in memory, as on a GPU (on the CPU ComfyUI holds them in float32)."""
    return comfy.sd.load_diffusion_model_state_dict({k: v.clone() for k, v in sd.items()}, model_options={"dtype": dtype} if dtype else {})


def load_int8(sd, name="q"):
    """The state dict saved and loaded back in Comfy-Org's int8 ConvRot layout (layer linears int8, int8 attention), as TMP/<name>.safetensors
    (a name of its own when the test reads the file back: Windows keeps a loaded file locked)."""
    path = os.path.join(TMP, f"{name}.safetensors")
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
            "dropped not a list": lambda d: d.__setitem__("dropped", "an item"),
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
            latent, w, h, _, _ = node.generate(aspect, mp, 1024, 1024, 1)
            self.assertEqual((w, h), size)
            self.assertEqual(tuple(latent["samples"].shape), (1, 128, h // 16, w // 16))
        self.assertEqual(node.generate("custom", 1.0, 1010, 600, 1)[1:3], (1008, 608))
        # width / height linked from another node: used even with an aspect ratio picked, as the user expects
        prompt = {"5": {"inputs": {"custom_width": ["3", 0], "custom_height": 768}}}
        self.assertEqual(node.generate("1:1", 1.0, 1536, 768, 1, prompt=prompt, unique_id="5")[1:3], (1536, 768))
        self.assertEqual(node.generate("1:1", 1.0, 1536, 768, 1, prompt={"5": {"inputs": {}}}, unique_id="5")[1:3], (1024, 1024))

    def test_seed_passes_through(self):
        out = pipeline.HaDeSMossLatent().generate("1:1", 1.0, 1024, 1024, 1, seed=123456789012)
        self.assertEqual(out[4], 123456789012)
        self.assertEqual(common.parse_info_meta(out[3])["seed"], 123456789012)


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

    def test_zero_weight_does_not_amplify_ties(self):
        base = torch.zeros(4)
        for method in ("TIES", "DARE-TIES"):
            gs = [torch.Generator().manual_seed(i) for i in range(2)]
            actual = mergers.multi_merge_tensor(method, base, [torch.ones(4), torch.full((4,), 3.0)], [1.0, 0.0], 1.0, 1.0, gs)
            torch.testing.assert_close(actual, torch.ones(4))

    def test_slerp_degenerate_and_regular_inputs(self):
        a = torch.tensor([1.0, 0.0])
        for b in (-a, torch.zeros_like(a), a):
            for t in (0.0, 0.25, 1.0):
                torch.testing.assert_close(mergers.slerp(a, b, t), a + (b - a) * t)
        b = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(mergers.slerp(a, b, 0.5), torch.full((2,), 2 ** -0.5))

    def test_recipe_rejects_nonfinite_ratios(self):
        for value in ("nan", "inf", "-inf", "1e999"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                mergers.parse_recipe(",".join([value] + ["0"] * (LAYERS + 2)), LAYERS)

    def test_merger_info_keeps_settings_and_sources(self):
        a, b = self.a.clone(), self.b.clone()
        a.set_attachments("hades_moss_recipe", {"node": "loader", "model_file": "source_a.safetensors"})
        b.set_attachments("hades_moss_recipe", {"node": "loader", "model_file": "source_b.safetensors"})
        ratios = [0.123456] * LAYERS
        m, info = mergers.HaDeSMossModelBlockMerger().merge(a, b, 0.2, 0.3, 0.4,
                                                          recipe_override=",".join(map(str, ratios + [0.2, 0.3, 0.4])))
        meta = common.parse_info_meta(info)
        self.assertEqual(meta["layer_ratios"], ratios)
        self.assertEqual(meta["part_ratios"], {"text_embedder": 0.2, "time_modulation": 0.3, "io": 0.4})
        self.assertEqual(meta["sources"]["model2"]["model_file"], "source_b.safetensors")
        self.assertEqual(m.get_attachment("hades_moss_recipe"), meta)
        _, info = mergers.HaDeSMossMultiModelMerger().merge(m, b, "DARE", 0.5, 0.75, 123, 0.25, 1.0, base_model=a)
        meta = common.parse_info_meta(info)
        self.assertEqual((meta["seed"], meta["weights"]), (123, [0.25, 1.0]))
        self.assertEqual(meta["sources"]["model_1"]["layer_ratios"], ratios)
        self.assertEqual(meta["sources"]["base_model"]["model_file"], "source_a.safetensors")

    def test_merger_names_models_from_core_loaders(self):
        """A model from a core loader carries no recipe: the merger walks the workflow to its file, so the saved recipe names it
        (Base through UNETLoader was missing from a saved SFT x Base merge); what a model carries still comes first."""
        a, b = self.a.clone(), self.b.clone()
        a.set_attachments("hades_moss_recipe", {"node": "loader", "model_file": "moss_loader_sft.safetensors"})
        graph = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "sft.safetensors", "weight_dtype": "default"}},
                 "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors", "weight_dtype": "default"}},
                 "3": {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["1", 0], "model2": ["2", 0]}},
                 "4": {"class_type": "HaDeSMossMultiModelMerger", "inputs": {"model_1": ["1", 0], "model_2": ["2", 0]}},
                 "5": {"class_type": "ModelMergeSimple", "inputs": {"model1": ["1", 0], "model2": ["2", 0], "ratio": 0.5}},
                 "6": {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["5", 0], "model2": ["2", 0]}}}
        _, info = mergers.HaDeSMossModelBlockMerger().merge(a, b, 0.0, 0.0, 0.0, prompt=graph, unique_id="3", **dict.fromkeys(mergers.GROUPS, 0.5))
        sources = common.parse_info_meta(info)["sources"]
        self.assertEqual(sources["model1"]["model_file"], "moss_loader_sft.safetensors")
        self.assertEqual(sources["model2"], {"node": "loader", "model_file": "base.safetensors", "from": "workflow"})
        _, info = mergers.HaDeSMossMultiModelMerger().merge(self.a.clone(), b, "SLERP", 0.5, 1.0, 0, 1.0, 1.0, prompt=graph, unique_id="4")
        sources = common.parse_info_meta(info)["sources"]
        self.assertEqual([sources[k]["model_file"] for k in ("model_1", "model_2")], ["sft.safetensors", "base.safetensors"])
        _, info = mergers.HaDeSMossModelBlockMerger().merge(self.a.clone(), b, 0.0, 0.0, 0.0, prompt=graph, unique_id="6",
                                                            **dict.fromkeys(mergers.GROUPS, 0.5))
        self.assertEqual(common.parse_info_meta(info)["sources"]["model1"], {"node": "files", "model_files": ["sft.safetensors", "base.safetensors"]})
        _, info = mergers.HaDeSMossModelBlockMerger().merge(self.a.clone(), b, 0.0, 0.0, 0.0, **dict.fromkeys(mergers.GROUPS, 0.5))
        self.assertEqual(common.parse_info_meta(info)["sources"], {"model1": None, "model2": None})  # no workflow (a script): unknown


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

    def test_multi_lora_matches_block_weight(self):
        groups = {**dict.fromkeys(mergers.GROUPS, 1.0), mergers.GROUPS[2]: 0.0, mergers.GROUPS[4]: 0.5}
        single = loras.HaDeSMossLoraBlockWeight().apply(self.model, "fused.safetensors", 0.7, 1.0, **groups)[0]
        rows = [{"name": "fused.safetensors", "on": True, "strength": 0.7, "groups": groups},
                {"name": "split.safetensors", "on": False, "strength": 1.0}, {"name": ""}]
        m, info = loras.HaDeSMossMultiLora().apply(self.model, True, json.dumps(rows))
        self.assertEqual(set(m.patches), set(single.patches))
        for k in single.patches:
            self.assertEqual([p[0] for p in m.patches[k]], [p[0] for p in single.patches[k]])
        meta = common.parse_info_meta(info)
        self.assertEqual([r["name"] for r in meta["loras"]], ["fused.safetensors"])
        self.assertEqual(meta["loras"][0]["groups"], {mergers.GROUPS[2]: 0.0, mergers.GROUPS[4]: 0.5})
        self.assertIs(loras.HaDeSMossMultiLora().apply(self.model, False, json.dumps(rows))[0], self.model)
        with self.assertRaises(ValueError):
            loras.HaDeSMossMultiLora().apply(self.model, True, "[{")

    def test_lora_for_another_model_is_an_error(self):
        """A LoRA none of whose keys match MOSS-Image (here SD-style keys) used to load as nothing, reported as applied."""
        foreign = {"lora_unet_input_blocks_1_1_proj_in.lora_down.weight": torch.randn(4, 320), "lora_unet_input_blocks_1_1_proj_in.lora_up.weight": torch.randn(320, 4)}
        safetensors.torch.save_file(foreign, os.path.join(TMP, "foreign.safetensors"))
        with self.assertRaisesRegex(ValueError, "changes no weight"):
            loras.HaDeSMossMultiLora().apply(self.model, True, json.dumps([{"name": "foreign.safetensors"}]))
        with self.assertRaisesRegex(ValueError, "changes no weight"):
            loras.HaDeSMossLoraBlockWeight().apply(self.model, "foreign.safetensors", 1.0, 1.0, **dict.fromkeys(mergers.GROUPS, 1.0))
        info = loras.HaDeSMossMultiLora().apply(self.model, True, json.dumps([{"name": "fused.safetensors"}]))[1]
        self.assertEqual(common.parse_info_meta(info)["loras"][0]["weights"], 2 * LAYERS)

    def test_multi_lora_names_few_step_files(self):
        meta = {"settings": "4 steps, CFG 1, euler / simple, strength 1.0, shift 5"}
        safetensors.torch.save_file(self.fused, os.path.join(TMP, "turbo.safetensors"), metadata=meta)
        self.assertEqual(loras.few_step_settings(meta)[0], 4)
        self.assertIsNone(loras.few_step_settings({"settings": "30 steps, CFG 4"}))
        self.assertIsNone(loras.few_step_settings({"settings": "6 steps, CFG 1.5, euler"}))
        self.assertEqual(loras.few_step_settings({"settings": "8 steps, CFG 1.0, euler"})[0], 8)
        rows = [{"name": "turbo.safetensors", "strength": 1.0}, {"name": "fused.safetensors", "strength": 0.5}]
        info = loras.HaDeSMossMultiLora().apply(self.model, True, json.dumps(rows))[1]
        self.assertEqual(common.parse_info_meta(info)["turbo"], [{"name": "turbo.safetensors", "steps": 4, "strength": 1.0}])
        sampler_info = common.format_info("x", "ksampler", width=1024, height=1024, steps=30, cfg=4.0, flow_shift=5.0, slg="off")
        findings = json.loads(telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=info, info_2=sampler_info)["result"][1])["findings"]
        self.assertTrue(any("trained for 4 steps" in f["message"] for f in findings))

    def test_layout_editor_outputs_a_compilable_draft(self):
        # an output node: the editor's DRAFT LAYOUT button runs the graph only up to it (partial execution)
        self.assertTrue(pipeline.HaDeSMossLayoutEditor.OUTPUT_NODE)
        node = pipeline.HaDeSMossLayoutEditor()
        out = node.edit(pipeline.EXAMPLE_DRAFT, draft_in='{"caption": "x"}')
        draft, width, height, info = out["result"]
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(out["ui"]["draft_in"], ['{"caption": "x"}'])
        self.assertEqual(pipeline.compile_layout(json.loads(draft))[0], pipeline.compile_layout(json.loads(pipeline.EXAMPLE_DRAFT))[0])
        self.assertEqual(common.parse_info_meta(info)["items"], 2)
        empty_caption = json.loads(pipeline.EXAMPLE_DRAFT)
        empty_caption["items"].append({"label": "object", "caption": "", "bbox": [0, 0, 64, 64]})
        # an unfinished box no longer stops the editor (DRAFT LAYOUT must be able to replace it): info says what is missing and
        # draft_in still goes to the UI; the Layout Compiler refuses it, naming the box (user report 2026-11-19)
        out = node.edit(json.dumps(empty_caption), draft_in='{"caption": "new"}')
        self.assertEqual(out["ui"]["draft_in"], ['{"caption": "new"}'])
        meta = common.parse_info_meta(out["result"][3])
        self.assertIn("items[2].caption", meta["problem"])
        self.assertIn("not ready to generate", out["result"][3])
        with self.assertRaisesRegex(ValueError, r"items\[2\]\.caption must be nonempty text \(box 3 in the Layout Editor"):
            pipeline.HaDeSMossLayoutCompiler().compile(out["result"][0], 1)
        for broken in ('{"caption": "a cat"}', "{not json"):
            out = node.edit(broken, draft_in="x")
            self.assertEqual(out["ui"]["draft_in"], ["x"])
            self.assertIn("problem", common.parse_info_meta(out["result"][3]))

    def test_merger_and_resize_files_load_in_core(self):
        merged_path = loras.HaDeSMossLoraMerger().merge(self.model, "split.safetensors", 1.0, "fused.safetensors", 1.0, 64, "merged", overwrite_existing=True)["result"][0]
        key = "diffusion_model.layers.1.proj.qkv.weight"
        expected = self.core_delta("split.safetensors", key) + self.core_delta("fused.safetensors", key)
        # rank 64 >= 12 + 4 keeps the merge exact; what is left is the bfloat16 storage of the factors
        self.assertLess(rel_err(self.core_delta(os.path.basename(merged_path), key), expected), 1e-2)
        resized = loras.HaDeSMossLoraResize().resize(self.model, "split.safetensors", 2, "resized", overwrite_existing=True)["result"][1]
        self.assertLess(common.parse_info_meta(resized)["energy_kept"], 1.0)

    def test_factors_and_full_diff_on_same_weight_are_added(self):
        key = "diffusion_model.layers.0.proj.o.weight"
        prefix = key[:-len(".weight")]
        diff = torch.full((DIM, DIM), 0.125)
        safetensors.torch.save_file({f"{prefix}.diff": diff}, os.path.join(TMP, "full_diff.safetensors"))
        path, info = loras.HaDeSMossLoraMerger().merge(self.model, "fused.safetensors", 1.0, "full_diff.safetensors", 0.5,
                                                     64, "mixed_diff", overwrite_existing=True)["result"]
        expected = self.core_delta("fused.safetensors", key) + diff * 0.5
        torch.testing.assert_close(self.core_delta(os.path.basename(path), key), expected.to(torch.bfloat16).float())
        with safetensors.safe_open(path, "pt") as f:
            self.assertNotIn(f"{prefix}.lora_up.weight", f.keys())
            self.assertIn(f"{prefix}.diff", f.keys())

    def test_resize_of_a_diff_only_lora_says_why(self):
        """Audit 0.8.1: a LoRA of full diffs only was rejected as 'no LoRA layer applies to this model'."""
        prefix = "diffusion_model.layers.0.proj.o"
        safetensors.torch.save_file({f"{prefix}.diff": torch.full((DIM, DIM), 0.125)}, os.path.join(TMP, "only_diff.safetensors"))
        with self.assertRaisesRegex(ValueError, "only full weight differences"):
            loras.HaDeSMossLoraResize().resize(self.model, "only_diff.safetensors", 4, "only_diff_resized", overwrite_existing=True)

    def test_diff_only_merge_loads_and_reports_no_compression(self):
        key = "diffusion_model.layers.0.proj.o.weight"
        prefix = key[:-len(".weight")]
        for name, value in (("diff_a", 0.125), ("diff_b", 0.25)):
            safetensors.torch.save_file({f"{prefix}.diff": torch.full((DIM, DIM), value)}, os.path.join(TMP, name + ".safetensors"))
        path, info = loras.HaDeSMossLoraMerger().merge(self.model, "diff_a.safetensors", 1.0, "diff_b.safetensors", 1.0,
                                                     4, "diff_only", overwrite_existing=True)["result"]
        torch.testing.assert_close(self.core_delta(os.path.basename(path), key), torch.full((DIM, DIM), 0.375))
        self.assertIsNone(common.parse_info_meta(info)["energy_kept"])
        self.assertEqual(telemetry.diagnose([common.parse_info_meta(info)]), [])

    def test_source_alias_cannot_be_overwritten(self):
        source = os.path.join(TMP, "alias_source.safetensors")
        shutil.copyfile(os.path.join(TMP, "fused.safetensors"), source)
        alias = os.path.join(TMP, "source_alias.safetensors")
        if not os.path.exists(alias):
            os.link(source, alias)
        with open(source, "rb") as f:
            before = f.read()
        with self.assertRaisesRegex(ValueError, "overwrite.*source"):
            loras.HaDeSMossLoraResize().resize(self.model, "alias_source.safetensors", 4, "source_alias", overwrite_existing=True)
        with self.assertRaisesRegex(ValueError, "overwrite.*source"):
            loras.HaDeSMossLoraMerger().merge(self.model, "alias_source.safetensors", 1.0, "split.safetensors", 1.0,
                                             8, "source_alias", overwrite_existing=True)
        with open(source, "rb") as f:
            self.assertEqual(f.read(), before)

    @unittest.skipUnless(os.name == "nt", "Windows case-insensitive file identity")
    def test_source_different_case_cannot_be_overwritten(self):
        shutil.copyfile(os.path.join(TMP, "fused.safetensors"), os.path.join(TMP, "case_source.safetensors"))
        with self.assertRaisesRegex(ValueError, "overwrite.*source"):
            loras.HaDeSMossLoraResize().resize(self.model, "case_source.safetensors", 4, "CASE_SOURCE", overwrite_existing=True)
        with self.assertRaisesRegex(ValueError, "overwrite.*source"):
            loras.HaDeSMossLoraMerger().merge(self.model, "case_source.safetensors", 1.0, "split.safetensors", 1.0,
                                             8, "CASE_SOURCE", overwrite_existing=True)

    def test_multi_lora_rejects_nonfinite_strengths(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for fields in ({"strength": value}, {"groups": {mergers.GROUPS[0]: value}}, {"groups": {"other_layers": value}}):
                with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "finite"):
                    loras.parse_lora_rows(json.dumps([{"name": "fused.safetensors", **fields}]))

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
    def test_fp8_to_int8_export_loads_with_requested_layout(self):
        sd = tiny_state_dict(21)
        key = "layers.0.proj.qkv.weight"
        qt = QuantizedTensor.from_float(sd.pop(key), "TensorCoreFP8E4M3Layout", scale="recalculate")
        sd.update(qt.state_dict(key))
        sd[key[:-len(".weight")] + ".comfy_quant"] = saver._json_tensor({"format": "float8_e4m3fn"})
        model = comfy.sd.load_diffusion_model_state_dict(sd)
        self.assertIsInstance(comfy.utils.get_attr(model.model.diffusion_model, key), QuantizedTensor)
        model.add_patches({"diffusion_model." + key: ("diff", (torch.full(qt.shape, 0.01),))})
        expected = patched_weight(model, key)
        path = saver.HaDeSMossModelSaver().save(model, "tests/fp8_to_int8", saver.PRECISIONS[1], True)["result"][1]
        with safetensors.safe_open(path, "pt") as f:
            self.assertEqual(f.get_slice(key).get_dtype(), "I8")
            config = json.loads(bytes(f.get_tensor(key[:-len(".weight")] + ".comfy_quant").tolist()))
            self.assertEqual(config, saver._INT8_QUANT_CONF)
        reloaded = comfy.sd.load_diffusion_model(path)
        actual = comfy.utils.get_attr(reloaded.model.diffusion_model, key).dequantize().float()
        self.assertLess(rel_err(actual.cpu(), expected.cpu()), 0.03)

    @unittest.skipUnless(os.environ.get("HADES_MOSS_TEST_CUDA") == "1", "opt-in CUDA export checks")
    def test_quantized_exports_on_cuda(self):
        self.assertTrue(torch.cuda.is_available(), "HADES_MOSS_TEST_CUDA=1 needs CUDA")
        with patch.object(comfy.model_management, "get_torch_device", return_value=torch.device("cuda")):
            self.test_fp8_to_int8_export_loads_with_requested_layout()
            self.test_int8_to_int8_keeps_existing_quantized_weights()

    def test_fp8_lazy_pieces_are_converted_to_int8(self):
        state_dict = comfy.model_patcher.ModelPatcher.model_state_dict_for_saving
        pieces = []

        def lazy_state(model, dm, prefix):
            sd = state_dict(model, dm, prefix)
            for k, w in common.model_weights(model).items():
                if not isinstance(w, QuantizedTensor):
                    continue
                caster = comfy.model_patcher.LazyCastingQuantizedParam(model, prefix + k)
                for group_key in w.state_dict(k):
                    if group_key in sd:
                        sd[group_key] = comfy.model_patcher.LazyCastingParamPiece(caster, prefix + group_key, sd[group_key])
                        pieces.append(group_key)
            return sd

        with patch.object(comfy.model_patcher.ModelPatcher, "model_state_dict_for_saving", lazy_state):
            self.test_fp8_to_int8_export_loads_with_requested_layout()
        self.assertTrue(pieces)

    def test_int8_to_int8_keeps_existing_quantized_weights(self):
        model = load_int8(tiny_state_dict(22))
        before = model.model.diffusion_model.state_dict()
        path = saver.HaDeSMossModelSaver().save(model, "tests/int8_copy", saver.PRECISIONS[1], False)["result"][1]
        after = safetensors.torch.load_file(path)
        for k in before:
            if k.endswith(("proj.qkv.weight", "proj.o.weight", "proj.ffn_up_gate.weight", "proj.ffn_down.weight", "weight_scale", "comfy_quant")):
                torch.testing.assert_close(before[k], after[k], rtol=0, atol=0)

    def test_int8_with_lora_saved_as_int8_requantizes_only_patched_layers(self):
        key = "layers.2.proj.o"
        g = torch.Generator().manual_seed(1)
        lora = {f"diffusion_model.{key}.lora_up.weight": torch.randn(DIM, 4, generator=g) * 0.05,
                f"diffusion_model.{key}.lora_down.weight": torch.randn(4, DIM, generator=g) * 0.05, f"diffusion_model.{key}.alpha": torch.tensor(4.0)}
        q = load_int8(tiny_state_dict(31), "q_lora")
        m = comfy.sd.load_lora_for_models(q, None, lora, 1.0, 0)[0]
        path = saver.HaDeSMossModelSaver().save(m, "tests/int8_lora", saver.PRECISIONS[1], False)["result"][1]
        raw_in = safetensors.torch.load_file(os.path.join(TMP, "q_lora.safetensors"))
        out = safetensors.torch.load_file(path)
        self.assertFalse(torch.equal(raw_in[f"{key}.weight"], out[f"{key}.weight"]))  # the LoRA is in the saved int8 weight
        self.assertFalse(torch.equal(raw_in[f"{key}.weight_scale"], out[f"{key}.weight_scale"]))
        self.assertEqual(out[f"{key}.weight"].dtype, torch.int8)
        for k in ("layers.3.proj.o.weight", "layers.3.proj.o.weight_scale", "layers.2.proj.qkv.weight"):  # untouched: byte for byte
            torch.testing.assert_close(raw_in[k], out[k], rtol=0, atol=0)

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

    def test_stochastic_round_is_unbiased_and_seeded(self):
        x = torch.full((200000,), 1.0 + 0.3 * 2 ** -7)  # 30% of the way from 1.0 to the next bfloat16 value
        r = common.stochastic_round_bf16(x, 1).float()
        self.assertEqual(set(r.unique().tolist()), {1.0, 1.0 + 2 ** -7})
        self.assertAlmostEqual(r.mean().item(), x[0].item(), delta=2 ** -7 * 0.01)
        torch.testing.assert_close(common.stochastic_round_bf16(x, 1), common.stochastic_round_bf16(x, 1), rtol=0, atol=0)
        y = torch.tensor([-0.75 * 2 ** -7 - 1.0, 3.0e38, float("inf"), float("nan")])
        out = common.stochastic_round_bf16(y, 2).float()
        self.assertTrue(out[0].item() in (-1.0, -1.0 - 2 ** -7))
        self.assertTrue(torch.isfinite(out[1]) and out[2].item() == float("inf") and torch.isnan(out[3]))

    def test_saver_stochastic_keeps_a_sub_step_lora(self):
        """A LoRA change of a quarter bfloat16 step: rounded to nearest it vanishes, stochastic rounding keeps it on average,
        float32 output keeps it exactly. Weights without patches are written unchanged."""
        sd = tiny_state_dict(9)
        key = "layers.1.proj.ffn_up_gate.weight"
        signs = torch.randint(0, 2, sd[key].shape, generator=torch.Generator().manual_seed(3)) * 2 - 1
        sd[key] = (signs * 0.02).to(torch.bfloat16)  # one magnitude, so one bfloat16 step for every weight
        w = sd[key].float()
        step = 2.0 ** (torch.floor(torch.log2(w.abs()[0, 0])).item() - 7)
        up = torch.rand(w.shape[0], 1, generator=torch.Generator().manual_seed(1)) * 2 - 1
        down = torch.rand(1, w.shape[1], generator=torch.Generator().manual_seed(2)) * 2 - 1
        scale = 0.25 * step  # |change| at most a quarter step
        lora = {f"diffusion_model.{key[:-len('.weight')]}.lora_up.weight": up * scale,
                f"diffusion_model.{key[:-len('.weight')]}.lora_down.weight": down,
                f"diffusion_model.{key[:-len('.weight')]}.alpha": torch.tensor(1.0)}
        delta = (up @ down) * scale
        coef = {}
        for rounding, precision in (("nearest", "bfloat16"), ("stochastic", "bfloat16"), ("float32", "float32")):
            m, _ = comfy.sd.load_lora_for_models(load(sd, torch.bfloat16), None, lora, 1.0, 0)
            path = saver.HaDeSMossModelSaver().save(m, "tests/round", precision, False, rounding="nearest" if rounding == "float32" else rounding)["result"][1]
            with safetensors.safe_open(path, "pt") as f:
                got = f.get_tensor(key).float() - w
                coef[rounding] = (got.flatten() @ delta.flatten() / (delta.flatten() @ delta.flatten())).item()
                if precision == "bfloat16":
                    torch.testing.assert_close(f.get_tensor("layers.2.proj.qkv.weight"), sd["layers.2.proj.qkv.weight"], rtol=0, atol=0)
                else:  # float32: the patched weight exactly, not rounded to bfloat16 first
                    torch.testing.assert_close(got, delta, rtol=1e-4, atol=1e-7)
        self.assertLess(coef["nearest"], 0.5)
        self.assertAlmostEqual(coef["stochastic"], 1.0, delta=0.1)

    def test_saver_nearest_matches_live_lora_weights(self):
        """Two stacked LoRAs baked with rounding nearest: the file holds the weights ComfyUI computes when it loads the model
        with the LoRAs (ModelPatcher.patch_weight_to_device), element for element. float16 LoRA compute, as on most GPUs."""
        dev = comfy.model_management.get_torch_device()
        with patch.dict(comfy.model_management.LORA_COMPUTE_DTYPES, {dev: torch.float16}):
            self._saver_nearest_matches_live(dev)

    def _saver_nearest_matches_live(self, dev):
        sd = tiny_state_dict(10)
        g = torch.Generator().manual_seed(4)
        keys = ("layers.1.proj.ffn_up_gate.weight", "layers.2.proj.qkv.weight")
        m = load(sd, torch.bfloat16)
        for strength in (0.5, 0.8):
            lora = {}
            for key in keys:
                base = f"diffusion_model.{key[:-len('.weight')]}"
                lora[f"{base}.lora_up.weight"] = torch.randn(sd[key].shape[0], 4, generator=g) * 0.01
                lora[f"{base}.lora_down.weight"] = torch.randn(4, sd[key].shape[1], generator=g)
                lora[f"{base}.alpha"] = torch.tensor(4.0)
            m, _ = comfy.sd.load_lora_for_models(m, None, lora, strength, 0)
        # the weights loading would give, taken before the Saver loads (and so patches) the model
        live = {key: m.patch_weight_to_device(f"diffusion_model.{key}", device_to=dev, return_weight=True).to("cpu", torch.bfloat16) for key in keys}
        path = saver.HaDeSMossModelSaver().save(m, "tests/live_nearest", "bfloat16", False, rounding="nearest")["result"][1]
        with safetensors.safe_open(path, "pt") as f:
            for key in keys:
                torch.testing.assert_close(f.get_tensor(key), live[key], rtol=0, atol=0)
            torch.testing.assert_close(f.get_tensor("layers.3.proj.qkv.weight"), sd["layers.3.proj.qkv.weight"], rtol=0, atol=0)

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
        self.assertEqual(common.parse_info_meta(values[8])["node"], "sampler_metadata")
        # the turbo node's steps widget replaces the default 4; the KSampler's own steps (25) stay ignored
        sampler.nodes.common_ksampler = fake_ksampler
        try:
            info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 2.5, "er_sde", "beta", 1.0, 0.0,
                                                      "off", turbo=sampler.HaDeSMossTurbo().pick("turbo_test.safetensors", 1.0, 6)[0])[2]
        finally:
            sampler.nodes.common_ksampler = original
        self.assertEqual((seen["steps"], common.parse_info_meta(info)["steps"]), (6, 6))
        # the turbo node's CFG, sampler and scheduler replace CFG 1, euler / simple
        sampler.nodes.common_ksampler = fake_ksampler
        try:
            info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 2.5, "euler", "beta", 1.0, 0.0, "off",
                                                      turbo=sampler.HaDeSMossTurbo().pick("turbo_test.safetensors", 1.0, 12, 1.5, "er_sde", "simple")[0])[2]
        finally:
            sampler.nodes.common_ksampler = original
        self.assertEqual({k: seen[k] for k in ("steps", "cfg", "sampler", "scheduler")}, {"steps": 12, "cfg": 1.5, "sampler": "er_sde", "scheduler": "simple"})
        self.assertEqual((common.parse_info_meta(info)["cfg"], common.parse_info_meta(info)["sampler"]), (1.5, "er_sde"))


    def test_turbo_checks_the_lora_and_strength_zero(self):
        """Audit 0.8.0 #1: a LoRA for another model must not switch on the few-step settings; strength 0 means no turbo."""
        safetensors.torch.save_file({"lora_unet_other_model.lora_up.weight": torch.ones(4, 2), "lora_unet_other_model.lora_down.weight": torch.ones(2, 4)},
                                    os.path.join(TMP, "turbo_foreign.safetensors"))
        seen = {}

        def fake_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
            seen.update(steps=steps, cfg=cfg)
            return ({"samples": torch.zeros(1, 128, 4, 4)},)

        original, sampler.nodes.common_ksampler = sampler.nodes.common_ksampler, fake_ksampler
        try:
            with self.assertRaisesRegex(ValueError, "changes no weight"):
                sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 2.5, "er_sde", "simple", 1.0, 0.0, "off",
                                                   turbo=sampler.HaDeSMossTurbo().pick("turbo_foreign.safetensors", 1.0)[0])
            self.assertEqual(seen, {})
            info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 2.5, "er_sde", "simple", 1.0, 0.0, "off",
                                                      turbo=sampler.HaDeSMossTurbo().pick("turbo_foreign.safetensors", 0.0)[0])[2]
        finally:
            sampler.nodes.common_ksampler = original
        meta = common.parse_info_meta(info)
        self.assertEqual((seen["steps"], seen["cfg"], meta["steps"], "turbo" in meta), (25, 2.5, 25, False))

    def test_turbo_switched_off_samples_with_the_ksampler_settings(self):
        """enabled off: no LoRA applied (even a foreign one), the KSampler's own steps / CFG, no turbo in the metadata."""
        seen = {}

        def fake_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
            seen.update(steps=steps, cfg=cfg, sampler=sampler_name)
            return ({"samples": torch.zeros(1, 128, 4, 4)},)

        turbo, tinfo = sampler.HaDeSMossTurbo().pick("turbo_foreign.safetensors", 0.25, 15, 2.5, "euler", "simple", enabled=False)
        original, sampler.nodes.common_ksampler = sampler.nodes.common_ksampler, fake_ksampler
        try:
            info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, None, 7, 25, 3.0, "er_sde", "simple", 1.0, 0.0, "off",
                                                      turbo=turbo)[2]
        finally:
            sampler.nodes.common_ksampler = original
        meta = common.parse_info_meta(info)
        self.assertEqual((seen, meta["steps"], "turbo" in meta), ({"steps": 25, "cfg": 3.0, "sampler": "er_sde"}, 25, False))
        self.assertIs(common.parse_info_meta(tinfo)["enabled"], False)
        self.assertIn("off: the KSampler uses its own settings", tinfo)
        # enabled is added after cfg / sampler_name / scheduler, not in place of them (a second "optional" key once dropped them)
        self.assertEqual(list(sampler.HaDeSMossTurbo.INPUT_TYPES()["optional"]), ["cfg", "sampler_name", "scheduler", "enabled"])

    def test_denoise_zero_passes_the_latent_through(self):
        """Audit 0.8.0 #4: denoise 0 used to raise ZeroDivisionError while building the metadata."""
        def fake_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
            return (latent,)

        original, sampler.nodes.common_ksampler = sampler.nodes.common_ksampler, fake_ksampler
        try:
            for mode in (sampler.DENOISE_CORE, sampler.DENOISE_EXACT):
                info = sampler.HaDeSMossKSampler().sample(load(tiny_state_dict(10)), None, None, {"samples": torch.zeros(1, 128, 4, 4)}, 7, 4, 1.0,
                                                          "euler", "simple", 0.0, 0.0, "off", denoise_mode=mode)[2]
                meta = common.parse_info_meta(info)
                self.assertEqual((meta["start_sigma"], meta["steps_run"], "schedule_steps" in meta), (0.0, 0, False))
        finally:
            sampler.nodes.common_ksampler = original


class TestImageSaver(unittest.TestCase):
    def test_two_stages_without_primary_warn(self):
        """Audit 0.8.0 #2: two KSamplers in metadata and no primary_stage_metadata is ambiguous; the node says so."""
        isaver = importlib.import_module(f"{PACK}.moss_image_saver_nodes")
        f = common.format_info
        tel = json.dumps({"nodes": [common.parse_info_meta(f("x", "ksampler", seed=1, steps=4, cfg=1.0, sampler="euler", scheduler="simple",
                                                             width=64, height=64, flow_shift=5.0)),
                                    common.parse_info_meta(f("x", "ksampler", seed=22, steps=12, cfg=1.0, sampler="euler", scheduler="simple",
                                                             width=64, height=64, flow_shift=5.0))]})
        out = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/two_stage", False, False, metadata=tel, positive="A cat.")
        self.assertIn("2 KSamplers", out["ui"]["text"][0])
        self.assertIn("primary_stage_metadata", common.parse_info_meta(out["result"][1])["notes"][0])

    def test_disable_metadata_removes_all_png_text(self):
        from PIL import Image
        isaver = importlib.import_module(f"{PACK}.moss_image_saver_nodes")
        meta = common.format_info("private", "prompt_enhancer", prompt="PRIVATE", negative="PRIVATE NEGATIVE")
        for embed in (False, True):
            with patch.object(isaver.args, "disable_metadata", True):
                out = isaver.HaDeSMossImageSaver().save(torch.zeros(1, 8, 8, 3), "tests/private", False, embed,
                                                       metadata=meta, prompt={"private": "value"}, extra_pnginfo={"workflow": {"private": True}})
            item = out["ui"]["images"][0]
            with Image.open(os.path.join(TMP, item["subfolder"], item["filename"])) as png:
                self.assertEqual(png.text, {})
            self.assertIn("PRIVATE", out["result"][0])

    def test_png_carries_a1111_parameters_and_hashes(self):
        from PIL import Image
        isaver = importlib.import_module(f"{PACK}.moss_image_saver_nodes")
        f = common.format_info
        safetensors.torch.save_file({"x": torch.ones(3)}, os.path.join(TMP, "style.safetensors"))
        lora_hash = isaver.file_sha256(os.path.join(TMP, "style.safetensors"))[:10]
        infos = [f("x", "prompt_enhancer", mode="caption", width=1024, height=1024, image=False, decode_fallback=False, notes=[],
                   prompt="A red fox in snow.", negative="blurry"),
                 f("x", "latent", width=1024, height=1024, batch=1, seed=7),
                 f("x", "multi_lora", enabled=True, loras=[{"name": "style.safetensors", "strength": 0.8, "weights": 12}], turbo=[]),
                 f("x", "ksampler", width=1024, height=1024, steps=25, cfg=2.5, flow_shift=5.0, sampler="er_sde", scheduler="simple",
                   seed=7, denoise=1.0, slg="off")]
        tel = telemetry.HaDeSMossSuiteTelemetry().audit("T", **{f"info_{i + 1}": s for i, s in enumerate(infos)})["result"][1]
        out = isaver.HaDeSMossImageSaver().save(torch.rand(2, 64, 64, 3), "tests/img_%seed%", True, True, metadata=tel,
                                                prompt={"1": {}}, extra_pnginfo={"workflow": {"nodes": []}})
        params = out["result"][0]
        self.assertTrue(params.startswith("A red fox in snow.\nNegative prompt: blurry\nSteps: 25, Sampler: er_sde, Schedule type: simple, "
                                          "CFG scale: 2.5, Seed: 7, Size: 1024x1024"), params)
        self.assertIn(f'Lora hashes: "style: {lora_hash}"', params)
        self.assertIn(f'"lora:style": "{lora_hash}"', params)
        self.assertEqual(len(out["ui"]["images"]), 2)
        first = out["ui"]["images"][0]
        self.assertTrue(first["filename"].startswith("img_7_"))
        png = Image.open(os.path.join(folder_paths.get_output_directory(), first["subfolder"], first["filename"]))
        self.assertEqual(png.text["parameters"], params)
        self.assertEqual(json.loads(png.text["workflow"]), {"nodes": []})
        # one info connected directly instead of Telemetry
        self.assertIn("Seed: 7", isaver.HaDeSMossImageSaver().save(torch.rand(1, 64, 64, 3), "tests/img", False, False, metadata=infos[3])["result"][0])
        with self.assertRaisesRegex(ValueError, "connect Suite Telemetry"):
            isaver.HaDeSMossImageSaver().save(torch.rand(1, 64, 64, 3), "tests/img", False, False, metadata="not metadata")
        # no metadata connected: the image is still saved, with the prompt text given and a note
        out = isaver.HaDeSMossImageSaver().save(torch.rand(1, 64, 64, 3), "tests/nometa", False, False, positive="A cat.")
        self.assertEqual(len(out["ui"]["images"]), 1)
        self.assertTrue(out["result"][0].startswith("A cat."))
        self.assertIn("metadata not connected", out["ui"]["text"][0])

    def test_primary_stage_keeps_full_pipeline_and_uses_only_selected_resources(self):
        from PIL import Image
        isaver = importlib.import_module(f"{PACK}.moss_image_saver_nodes")
        f = common.format_info
        files = {}
        for name, value in (("base.safetensors", 1), ("refine.safetensors", 2),
                            ("base_lora.safetensors", 3), ("refine_lora.safetensors", 4)):
            path = os.path.join(TMP, name)
            safetensors.torch.save_file({"x": torch.full((3,), value)}, path)
            files[name] = path
        def loader(name):
            return f("x", "loader", model=name, model_file=name + ".safetensors", precision="bfloat16",
                     vae="flux2-vae.safetensors", text_encoder="qwen.safetensors")
        def stage(prompt, name, lora, seed, steps, cfg, size):
            return [f("x", "prompt_enhancer", prompt=prompt, negative="blurry"), loader(name),
                    f("x", "multi_lora", enabled=True, loras=[{"name": lora + ".safetensors", "strength": 1.0}], turbo=[]),
                    f("x", "ksampler", width=size, height=size, steps=steps, cfg=cfg, flow_shift=5.0, sampler="euler",
                      scheduler="simple", seed=seed, denoise=1.0, slg="off")]
        base = stage("Base scene.", "base", "base_lora", 11, 8, 1.0, 1024)
        refine = stage("Refined scene.", "refine", "refine_lora", 22, 12, 1.5, 1536)
        audit = telemetry.HaDeSMossSuiteTelemetry().audit
        full_json = audit("All", **{f"info_{i + 1}": item for i, item in enumerate(base + refine)})["result"][1]
        refine_json = audit("Refine", **{f"info_{i + 1}": item for i, item in enumerate(refine)})["result"][1]
        paths = lambda folder, name: files.get(name)
        with patch.object(isaver.folder_paths, "get_full_path", side_effect=paths):
            saved = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/%model%_%seed%_%steps%", True, True,
                                                       metadata=full_json, primary_stage_metadata=refine_json,
                                                       prompt={"1": {}}, extra_pnginfo={"workflow": {"nodes": []}})
            legacy = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/legacy_%model%_%seed%", True, False,
                                                        metadata=full_json)
        params = saved["result"][0]
        self.assertTrue(params.startswith("Refined scene.\nNegative prompt: blurry\nSteps: 12, Sampler: euler, Schedule type: simple, "
                                          "CFG scale: 1.5, Seed: 22, Size: 1536x1536"), params)
        self.assertIn(f"Model hash: {isaver.file_sha256(files['refine.safetensors'])[:10]}", params)
        self.assertIn(f"refine_lora: {isaver.file_sha256(files['refine_lora.safetensors'])[:10]}", params)
        self.assertNotIn("base_lora", params)
        self.assertIn("refine_22_12_", saved["ui"]["images"][0]["filename"])
        item = saved["ui"]["images"][0]
        with Image.open(os.path.join(TMP, item["subfolder"], item["filename"])) as png:
            self.assertEqual(json.loads(png.text["hades_moss"]), json.loads(full_json)["nodes"])
            self.assertEqual(json.loads(png.text["hades_moss_primary_stage"]), json.loads(refine_json)["nodes"])
            self.assertEqual(json.loads(png.text["workflow"]), {"nodes": []})
        self.assertIn("Base scene.", legacy["result"][0])
        self.assertIn("Seed: 11", legacy["result"][0])
        self.assertIn("legacy_base_11_", legacy["ui"]["images"][0]["filename"])
        item = legacy["ui"]["images"][0]
        with Image.open(os.path.join(TMP, item["subfolder"], item["filename"])) as png:
            self.assertNotIn("hades_moss_primary_stage", png.text)
        with self.assertRaisesRegex(ValueError, "exactly one KSampler"):
            isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/invalid", False, False,
                                              primary_stage_metadata=full_json)
        sampler_only = refine[-1]
        partial = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/partial_%seed%", False, False,
                                                    metadata=full_json, primary_stage_metadata=sampler_only)
        self.assertIn("Seed: 22", partial["result"][0])
        self.assertNotIn("Model:", partial["result"][0])
        self.assertNotIn("Lora hashes:", partial["result"][0])
        self.assertIn("primary stage has no MOSS Loader info", partial["ui"]["text"][0])
        with patch.object(isaver.args, "disable_metadata", True):
            private = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/private_stage", False, False,
                                                        metadata=full_json, primary_stage_metadata=refine_json)
        item = private["ui"]["images"][0]
        with Image.open(os.path.join(TMP, item["subfolder"], item["filename"])) as png:
            self.assertEqual(png.text, {})


    def test_ksampler_traces_the_files_its_model_comes_from(self):
        """The Model field: the KSampler walks its model input up to the loaders (core or Suite), through LoRA nodes and mergers,
        and skips other outputs of the same nodes (a MOSS Loader's CLIP is not the model)."""
        sampler = importlib.import_module(f"{PACK}.moss_sampler_nodes")
        graph = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "sft.safetensors", "weight_dtype": "default"}},
                 "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors", "weight_dtype": "default"}},
                 "3": {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["1", 0], "model2": ["2", 0], "layers_24_29": 0.3}},
                 "4": {"class_type": "HaDeSMossMultiLora", "inputs": {"model": ["3", 0], "enabled": True, "loras": "[]"}},
                 "5": {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["4", 0], "positive": ["7", 0]}},
                 "6": {"class_type": "HaDeSMossLoader", "inputs": {"unet_name": "moss.safetensors", "clip_name": "te.safetensors"}},
                 "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 1], "text": "x"}},
                 "8": {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["6", 0]}},
                 "9": {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["1", 0]}},
                 "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ckpt.safetensors"}},
                 "11": {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["10", 0]}}}
        self.assertEqual(sampler.model_files(graph, "5"), [{"file": "sft.safetensors", "folder": "diffusion_models"},
                                                           {"file": "base.safetensors", "folder": "diffusion_models"}])
        self.assertEqual(sampler.model_files(graph, "8"), [{"file": "moss.safetensors", "folder": "diffusion_models"}])
        self.assertEqual(sampler.model_files(graph, "11"), [{"file": "ckpt.safetensors", "folder": "checkpoints"}])
        graph["12"] = {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["1", 0], "model2": ["1", 0]}}
        graph["13"] = {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["12", 0]}}
        self.assertEqual(sampler.model_files(graph, "13"), [{"file": "sft.safetensors", "folder": "diffusion_models"}])
        self.assertEqual(sampler.model_files(None, "5"), [])
        self.assertEqual(sampler.model_files(graph, "404"), [])
        if "HaDeSMossLoader" in sampler.nodes.NODE_CLASS_MAPPINGS:  # registered in ComfyUI: its CLIP output is not the model
            graph["14"] = {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["6", 1]}}
            self.assertEqual(sampler.model_files(graph, "14"), [])

    def test_saver_names_the_sampled_model_and_lists_each_lora_once(self):
        """The Model field comes from the KSampler's model, not the first MOSS Loader (a Loader used only for its text encoder);
        a merge made in the workflow is named after its sources with no model hash; a LoRA in two stages is listed once."""
        isaver = importlib.import_module(f"{PACK}.moss_image_saver_nodes")
        f = common.format_info
        files = {}
        for name, value in (("loader_unet.safetensors", 1), ("sampled.safetensors", 2), ("turbo.safetensors", 3)):
            files[name] = os.path.join(TMP, name)
            safetensors.torch.save_file({"x": torch.full((3,), value)}, files[name])
        loader = f("x", "loader", model="loader_unet", model_file="loader_unet.safetensors", vae="v.safetensors", text_encoder="te.safetensors")
        ks = lambda **kw: f("x", "ksampler", **{**dict(width=64, height=64, steps=8, cfg=1.0, flow_shift=5.0, sampler="euler",
                                                       scheduler="simple", seed=3, denoise=1.0, slg="off", turbo="turbo.safetensors",
                                                       turbo_strength=1.0), **kw})
        enh = f("x", "prompt_enhancer", prompt="A fisherman.", negative="")
        audit = telemetry.HaDeSMossSuiteTelemetry().audit
        paths = lambda folder, name: files.get(name)
        one = audit("T", info_1=enh, info_2=loader, info_3=ks(model_files=[{"file": "sampled.safetensors", "folder": "diffusion_models"}]),
                    info_4=ks(denoise=0.25))["result"][1]
        merge = audit("T", info_1=enh, info_2=loader, info_3=ks(model_files=[{"file": "sft.safetensors", "folder": "diffusion_models"},
                                                                             {"file": "base.safetensors", "folder": "diffusion_models"}]))["result"][1]
        legacy = audit("T", info_1=enh, info_2=loader, info_3=ks())["result"][1]
        with patch.object(isaver.folder_paths, "get_full_path", side_effect=paths):
            p1 = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/%model%", True, False, metadata=one)
            p2 = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/%model%", True, False, metadata=merge)
            p3 = isaver.HaDeSMossImageSaver().save(torch.rand(1, 8, 8, 3), "tests/%model%", True, False, metadata=legacy)
        params = p1["result"][0]
        self.assertIn(f"Model hash: {isaver.file_sha256(files['sampled.safetensors'])[:10]}, Model: sampled,", params)
        self.assertNotIn("loader_unet", params)
        self.assertTrue(p1["ui"]["images"][0]["filename"].startswith("sampled_"))
        turbo_hash = isaver.file_sha256(files["turbo.safetensors"])[:10]
        self.assertIn(f'Lora hashes: "turbo: {turbo_hash}"', params)  # two KSamplers with the same turbo: once
        params = p2["result"][0]
        self.assertIn("Model: merge: sft + base,", params)
        self.assertNotIn("Model hash", params)
        self.assertNotIn('"model"', params)
        self.assertIn("merge made in the workflow", p2["ui"]["text"][0])
        self.assertTrue(p2["ui"]["images"][0]["filename"].startswith("merge_"))
        self.assertIn("Model: loader_unet,", p3["result"][0])  # a KSampler info without model_files: the first Loader, as before
        report = audit("T", info_1=ks(model_files=[{"file": "sft.safetensors", "folder": "diffusion_models"},
                                                   {"file": "base.safetensors", "folder": "diffusion_models"}]))["result"][0]
        self.assertIn("model=sft.safetensors + base.safetensors", report)

class TestControl(unittest.TestCase):
    class FakeVAE:
        latent_channels = 128

        def __init__(self):
            self.sizes = []

        def encode(self, img):
            self.sizes.append(tuple(img.shape))
            return torch.full((img.shape[0], 128, img.shape[1] // 16, img.shape[2] // 16), float(img.mean()))

    def test_maps_resized_in_pixels_and_depth_only(self):
        vae, latent = self.FakeVAE(), {"samples": torch.zeros(1, 128, 48, 32)}  # 512 x 768
        cond = [[torch.zeros(1, 4, 8), {}]]
        pos, neg, maps, info = control.HaDeSMossControl().apply(cond, cond, vae, latent, "off (depth only)", 8, 4.0, 1.0, 0.0, 1.0,
                                                                 depth_map=torch.rand(1, 1536, 1024, 3), color_map=torch.rand(1, 1536, 1024, 3))
        self.assertEqual(vae.sizes, [(1, 768, 512, 3)])  # only depth encoded, at the latent size
        concat = pos[0][1]["concat_latent_image"]
        self.assertEqual(tuple(concat.shape), (1, 256, 48, 32))
        self.assertEqual(float(concat[:, :128].abs().max()), 0.0)  # color off is a zero latent
        self.assertIn("concat_latent_image", neg[0][1])
        self.assertEqual(common.parse_info_meta(info)["maps"], "depth")

    def test_flat_regions_reduce_colors(self):
        out = control.flat_regions(torch.rand(1, 64, 64, 3), 4, 2.0)
        self.assertLessEqual(len(torch.unique(out.reshape(-1, 3), dim=0)), 4)

    def test_needs_a_map(self):
        with self.assertRaisesRegex(ValueError, "connect a depth_map"):
            control.HaDeSMossControl().apply([], [], self.FakeVAE(), {"samples": torch.zeros(1, 128, 4, 4)}, "flat regions", 8, 4.0, 1.0, 0.0, 1.0)


class TestCanvasCheck(unittest.TestCase):
    def test_sizes_compared_only_with_a_layout(self):
        f = common.format_info
        ks = dict(steps=25, cfg=2.5, flow_shift=5.0, sampler="er_sde", scheduler="simple", seed=1, denoise=1.0, slg="off")
        audit = lambda *infos: json.loads(telemetry.HaDeSMossSuiteTelemetry().audit("T", **{f"info_{i + 1}": s for i, s in enumerate(infos)})["result"][1])["findings"]
        # an upscale pass: latent 1024, KSampler 2048, no layout -> no error
        self.assertEqual([x for x in audit(f("x", "latent", width=1024, height=1024, batch=1), f("x", "ksampler", width=2048, height=2048, **ks))
                          if x["level"] == "error"], [])
        # the Prompt Enhancer's layout canvas against the KSampler's -> error
        enh = f("x", "prompt_enhancer", mode="layout", width=1024, height=1536, image=False, decode_fallback=False, notes=[])
        self.assertTrue(any(x["level"] == "error" for x in audit(enh, f("x", "ksampler", width=1024, height=1024, **ks))))
        # with a Layout Editor, its canvas is the one used: the enhancer that wrote the first draft is not compared
        editor = f("x", "layout_editor", width=1024, height=1024, items=2, dropped=0)
        self.assertEqual([x for x in audit(enh, editor, f("x", "ksampler", width=1024, height=1024, **ks)) if x["level"] == "error"], [])
        # the editor's canvas against the KSampler's is still checked
        self.assertTrue(any(x["level"] == "error" for x in audit(enh, editor, f("x", "ksampler", width=1024, height=1536, **ks))))
        # one incomplete trailer is named and left out; the other checks still run (audit 0.8.1)
        broken = "x\nHADES_META " + json.dumps({"node": "ksampler", "steps": 25})
        found = audit(editor, f("x", "latent", width=512, height=512, batch=1), broken)
        self.assertTrue(any(x["level"] == "warning" and "KSampler info is incomplete" in x["message"] for x in found))
        self.assertTrue(any(x["level"] == "error" and "Canvas mismatch" in x["message"] for x in found))
        # caption mode carries no canvas
        cap = f("x", "prompt_enhancer", mode="caption", width=1024, height=1536, image=False, decode_fallback=False, notes=[])
        self.assertEqual([x for x in audit(cap, f("x", "ksampler", width=1024, height=1024, **ks)) if x["level"] == "error"], [])


class TestLoader(unittest.TestCase):
    def test_model_name_metadata_does_not_conflict_with_patcher(self):
        class Patcher:
            def set_attachments(self, key, value):
                self.key, self.value = key, value

        patcher = Patcher()
        info = common.format_model_info("loader", "loader", patcher, model="m.safetensors", model_file="MOSS/m.safetensors")
        self.assertEqual(patcher.key, "hades_moss_recipe")
        self.assertEqual(patcher.value["model"], "m.safetensors")
        self.assertEqual(common.parse_info_meta(info)["model_file"], "MOSS/m.safetensors")

    def test_precision_and_telemetry_note(self):
        loader = importlib.import_module(f"{PACK}.moss_loader_nodes")
        sd = tiny_state_dict(9)
        self.assertIn(loader.model_precision(load(sd)), ("bf16", "fp32"))  # the precision in memory: fp32 when ComfyUI runs on the CPU
        q = loader.model_precision(load_int8(sd))
        self.assertTrue(q.startswith("int8 ("), q)
        info = common.format_info("x", "loader", model="m.safetensors", precision=q, weight_dtype="default", attention="comfy_kitchen_int8",
                                  layers=LAYERS, text_encoder="te.safetensors", vae="flux2-vae.safetensors")
        findings = json.loads(telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=info)["result"][1])["findings"]
        self.assertTrue(any("57% slower" in f["message"] for f in findings))


class TestCompareSheet(unittest.TestCase):
    def test_labels_from_branch_infos_and_saved_sheet(self):
        from PIL import Image
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        f = common.format_info
        multi = f("x", "multi_lora", enabled=True, loras=[{"name": "MOSS\\manara\\style_s1500.safetensors", "strength": 1.0}], turbo=[])
        blockw = f("x", "lora_block_weight", lora="style_s1600.safetensors", strength_model=0.8, weights="")
        tel = json.dumps({"nodes": [common.parse_info_meta(f("x", "ksampler", seed=1001, steps=8, cfg=1.0)),
                                    common.parse_info_meta(f("x", "prompt_enhancer", prompt="A terrace at sunset."))]})
        out = cmp.HaDeSMossCompareSheet().compare(torch.rand(1, 64, 48, 3), "tests/compare", True, True,
                                                   image_B=torch.rand(1, 32, 24, 3), image_C=torch.rand(1, 64, 48, 3),
                                                   info_A=multi, info_B=blockw, info_C=tel, label_C="", footer="")
        meta = common.parse_info_meta(out["result"][1])
        # only C carries settings and a prompt: they stay on its label, the footer names nothing A and B may not share (audit 0.8.1 #1)
        self.assertEqual(meta["labels"], ["A: style_s1500", "B: style_s1600 x0.8", "C: seed 1001 · 8 steps · CFG 1"])
        self.assertEqual(meta["label_sources"], ["info", "info", "info"])
        self.assertEqual(meta["footer"], "")
        self.assertTrue(any("scaled" in n for n in meta["notes"]))
        sheet = out["result"][0]
        self.assertEqual(sheet.shape[0], 1)
        self.assertGreater(sheet.shape[2], 3 * 48)
        item = out["ui"]["images"][0]
        with Image.open(os.path.join(TMP, item["subfolder"], item["filename"])) as png:
            self.assertIn("hades_moss_compare", png.text)
            self.assertEqual(png.size, (sheet.shape[2], sheet.shape[1]))

    def test_batch_sheets_built_one_at_a_time_match_their_files(self):
        """Audit 0.8.1 #9: sheets are now built, saved and copied into the output one at a time; each output sheet must still be
        exactly its saved PNG (the uint8 sheet / 255)."""
        import numpy as np
        from PIL import Image
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        out = cmp.HaDeSMossCompareSheet().compare(torch.rand(3, 40, 32, 3), "tests/compare_batch", True, False,
                                                   image_B=torch.rand(3, 40, 32, 3), footer="seed 1")
        sheets, items = out["result"][0], out["ui"]["images"]
        self.assertEqual((sheets.shape[0], len(items)), (3, 3))
        for b, item in enumerate(items):
            with Image.open(os.path.join(TMP, item["subfolder"], item["filename"])) as png:
                expected = torch.from_numpy(np.asarray(png.convert("RGB"), dtype=np.float32) / 255.0)
            torch.testing.assert_close(sheets[b], expected, rtol=0, atol=0)

    def test_manual_label_and_preview_only(self):
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        with patch.object(cmp.folder_paths, "get_temp_directory", return_value=TMP):
            out = cmp.HaDeSMossCompareSheet().compare(torch.rand(2, 32, 32, 3), "tests/compare_preview", False, False,
                                                       image_B=torch.rand(1, 32, 32, 3), label_A="base", footer="seed 7")
        meta = common.parse_info_meta(out["result"][1])
        self.assertEqual(meta["labels"], ["A: base", "B"])
        self.assertEqual(meta["files"], [])
        self.assertEqual(out["result"][0].shape[0], 1)
        self.assertTrue(any("batch sizes differ" in n for n in meta["notes"]))
        self.assertEqual(out["ui"]["images"][0]["type"], "temp")


    def test_same_lora_on_two_models_and_settings_per_branch(self):
        """Audit 0.8.0 #6: one LoRA on two checkpoints gave two identical labels; settings that differ go on each label."""
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        f = common.format_info

        def branch(model, seed):
            return json.dumps({"nodes": [common.parse_info_meta(f("x", "loader", model_file=model)),
                                         common.parse_info_meta(f("x", "multi_lora", loras=[{"name": "shared.safetensors", "strength": 1.0}])),
                                         common.parse_info_meta(f("x", "ksampler", seed=seed, steps=20, cfg=1.0, sampler="euler", scheduler="simple",
                                                                  width=64, height=64))]})
        out = cmp.HaDeSMossCompareSheet().compare(torch.rand(1, 32, 32, 3), "tests/compare_models", False, False,
                                                   image_B=torch.rand(1, 32, 32, 3), info_A=branch("base.safetensors", 11),
                                                   info_B=branch("sft.safetensors", 22), footer="")
        meta = common.parse_info_meta(out["result"][1])
        self.assertEqual(meta["labels"], ["A: shared · base · seed 11", "B: shared · sft · seed 22"])
        self.assertEqual(meta["footer"], "20 steps | CFG 1 | euler / simple | 64x64")
        bad = cmp.HaDeSMossCompareSheet().compare(torch.rand(1, 32, 32, 3), "tests/compare_bad", False, False, info_A="HADES_META []")
        self.assertEqual(common.parse_info_meta(bad["result"][1])["labels"], ["A"])

    def test_footer_only_with_what_every_branch_carries(self):
        """Audit 0.8.1 #1: B without info (an image from disk) let A's seed, steps and prompt into the footer of the whole sheet."""
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        f = common.format_info
        info_A = json.dumps({"nodes": [common.parse_info_meta(f("x", "ksampler", seed=123, steps=15, cfg=2.5)),
                                     common.parse_info_meta(f("x", "prompt_enhancer", prompt="A lighthouse."))]})
        out = cmp.HaDeSMossCompareSheet().compare(torch.rand(1, 32, 32, 3), "tests/compare_one_info", False, False,
                                                   image_B=torch.rand(1, 32, 32, 3), info_A=info_A, footer="")
        meta = common.parse_info_meta(out["result"][1])
        self.assertEqual(meta["footer"], "")
        self.assertEqual(meta["labels"], ["A: seed 123 · 15 steps · CFG 2.5", "B"])
        # both branches with the same settings and prompt: in the footer once
        out = cmp.HaDeSMossCompareSheet().compare(torch.rand(1, 32, 32, 3), "tests/compare_two_info", False, False,
                                                   image_B=torch.rand(1, 32, 32, 3), info_A=info_A, info_B=info_A, footer="")
        self.assertEqual(common.parse_info_meta(out["result"][1])["footer"], "seed 123 | 15 steps | CFG 2.5 | A lighthouse.")

    def test_model_in_footer_only_when_every_branch_names_it(self):
        """0.8.1 acceptance: a branch without a model (a stock loader) left the other branch's model in the footer for the whole sheet."""
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        f = common.format_info
        ks = common.parse_info_meta(f("x", "ksampler", seed=1, steps=15, cfg=2.5, sampler="euler", scheduler="simple", width=64, height=64))
        info_A = json.dumps({"nodes": [ks, common.parse_info_meta(f("x", "loader", model_file="base.safetensors"))]})
        info_B = json.dumps({"nodes": [ks]})
        out = cmp.HaDeSMossCompareSheet().compare(torch.rand(1, 32, 32, 3), "tests/compare_partial", False, False,
                                                   image_B=torch.rand(1, 32, 32, 3), info_A=info_A, info_B=info_B, footer="")
        meta = common.parse_info_meta(out["result"][1])
        self.assertEqual(meta["labels"], ["A: base", "B"])
        self.assertEqual(meta["footer"], "seed 1 | 15 steps | CFG 2.5 | euler / simple | 64x64")

    def test_blind_shuffles_numbers_and_keeps_the_key_in_info(self):
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        f = common.format_info
        imgs = {f"image_{s}": torch.full((1, 16, 16, 3), k / 4.0) for k, s in enumerate("BCD", 1)}
        infos = {f"info_{s}": f("x", "lora_block_weight", lora=f"lora_{s}.safetensors", strength_model=1.0, weights="") for s in "ABCD"}
        run = lambda seed: cmp.HaDeSMossCompareSheet().compare(torch.zeros(1, 16, 16, 3), "tests/blind", False, False, footer="",
                                                                blind=True, blind_seed=seed, **imgs, **infos)
        out = run(5)
        meta = common.parse_info_meta(out["result"][1])
        key = meta["blind_key"]
        self.assertEqual(sorted(key), ["1", "2", "3", "4"])
        self.assertEqual(sorted(key.values()), [f"{s}: lora_{s}" for s in "ABCD"])
        self.assertNotEqual(list(key.values()), [f"{s}: lora_{s}" for s in "ABCD"])  # seed 5 does not keep the order
        # the panel shown as 1 is the image of the branch the key names, and the same seed gives the same order
        first = "ABCD".index(key["1"][0])
        self.assertAlmostEqual(float(out["result"][0][0, 50, 8, 0]), first / 4.0, places=2)
        self.assertEqual(common.parse_info_meta(run(5)["result"][1])["blind_key"], key)
        self.assertNotIn("blind_key", common.parse_info_meta(cmp.HaDeSMossCompareSheet().compare(
            torch.zeros(1, 16, 16, 3), "tests/blind", False, False, footer="", **imgs, **infos)["result"][1]))


class TestCompareSampler(unittest.TestCase):
    class Model:
        def __init__(self, recipe):
            self.recipe = recipe

        def get_attachment(self, key):
            return self.recipe if key == "hades_moss_recipe" else None

    def run_sampler(self, models, turbo=None, batch=1, **kw):
        cmp = importlib.import_module(f"{PACK}.moss_compare_nodes")
        f = common.format_info
        calls = []

        def fake_sample(_self, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise, flow_shift, slg,
                        vae=None, turbo=None, denoise_mode=None, **_):
            calls.append((model, seed, steps, cfg))
            on = turbo is not None and turbo["strength"] != 0 and turbo.get("enabled", True)
            t = dict(turbo=turbo["lora_name"], turbo_strength=turbo["strength"]) if on else {}
            info = f("x", "ksampler", seed=seed, steps=steps, cfg=cfg, sampler=sampler_name, scheduler=scheduler, width=64, height=64,
                     denoise=1.0, flow_shift=5.0, slg="off", **t)
            return ({"samples": None}, torch.full((batch, 16, 16, 3), len(calls) / 8.0), info)

        named = {f"model_{s}": m for s, m in zip("BCD", models[1:])}
        with patch.object(cmp.HaDeSMossKSampler, "sample", fake_sample):
            out = cmp.HaDeSMossCompareSampler().compare(models[0], None, None, None, None, 7, 15, 2.5, "euler", "simple", "tests/csampler",
                                                         False, False, turbo=turbo, **named, **kw)
        return out, calls

    def test_labels_come_from_what_each_model_carries(self):
        loader = lambda name: {"node": "loader", "model": name, "model_file": f"MOSS/{name}", "precision": "bf16", "weight_dtype": "default"}
        lora_on = {"node": "multi_lora", "enabled": True, "loras": [{"name": "MOSS/forge_mega/forge_mega_s5250.safetensors", "strength": 1.0}],
                   "turbo": [], "sources": {"model": loader("mossImage_base_bf16.safetensors")}}
        merge = {"node": "multi_model_merger", "sources": {"model_1": loader("a.safetensors"), "model_2": loader("b.safetensors"),
                                                             "base_model": loader("base.safetensors")}}
        # A: LoRA on; B: Multi-LoRA off (the loader's model passed through); C: a merge; D: a model from outside the Suite
        models = [self.Model(lora_on), self.Model(loader("mossImage_base_bf16.safetensors")), self.Model(merge), self.Model(None)]
        with patch.object(importlib.import_module(f"{PACK}.moss_compare_nodes").folder_paths, "get_temp_directory", return_value=TMP):
            out, calls = self.run_sampler(models, footer="")
        meta = common.parse_info_meta(out["result"][2])
        self.assertEqual([c[1:] for c in calls], [(7, 15, 2.5)] * 4)  # every model sampled with the same seed and settings
        self.assertEqual(meta["labels"], ["A: forge_mega_s5250 · mossImage_base_bf16", "B: mossImage_base_bf16", "C: merge: a + b", "D: unknown model"])
        self.assertTrue(any("model_D carries no HaDeS MOSS recipe" in n for n in meta["notes"]))
        self.assertEqual(tuple(out["result"][1].shape), (4, 16, 16, 3))
        self.assertEqual([m["node"] for m in meta["branches"]["A"]], ["multi_lora", "ksampler", "loader"])

    def test_images_output_keeps_every_batch_item(self):
        """Audit 0.8.1: the sheets showed whole batches but `images` gave only the first image of each model."""
        loader = {"node": "loader", "model": "base.safetensors", "model_file": "base.safetensors", "precision": "bf16", "weight_dtype": "default"}
        with patch.object(importlib.import_module(f"{PACK}.moss_compare_nodes").folder_paths, "get_temp_directory", return_value=TMP):
            out, _ = self.run_sampler([self.Model(loader), self.Model(loader)], batch=3, footer="")
        images = out["result"][1]
        self.assertEqual(tuple(images.shape), (6, 16, 16, 3))
        self.assertEqual([round(float(images[i, 0, 0, 0]) * 8) for i in range(6)], [1, 1, 1, 2, 2, 2])  # A's batch first, then B's
        self.assertEqual(out["result"][0].shape[0], 3)  # one sheet per batch index

    def test_turbo_on_and_off_and_one_cable_to_telemetry(self):
        loader = {"node": "loader", "model": "base.safetensors", "model_file": "base.safetensors", "precision": "bf16", "weight_dtype": "default"}
        sampler = importlib.import_module(f"{PACK}.moss_sampler_nodes")
        turbo, _ = sampler.HaDeSMossTurbo().pick("MOSS/turbo/T8.safetensors", 0.25, 15, 2.5, "euler", "simple")
        with patch.object(importlib.import_module(f"{PACK}.moss_compare_nodes").folder_paths, "get_temp_directory", return_value=TMP):
            on, _ = self.run_sampler([self.Model(loader), self.Model(loader)], turbo=turbo, footer="")
            off, _ = self.run_sampler([self.Model(loader), self.Model(loader)], turbo=dict(turbo, enabled=False), footer="")
        self.assertEqual(common.parse_info_meta(on["result"][2])["labels"], ["A: base + T8 x0.25", "B: base + T8 x0.25"])
        self.assertEqual(common.parse_info_meta(off["result"][2])["labels"], ["A: base", "B: base"])
        report = telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=on["result"][2])["result"]
        findings = json.loads(report[1])["findings"]
        self.assertIn("Compare Sampler", report[0])
        self.assertEqual(sum("strength 0.25" in x["message"] for x in findings), 1)  # two branches, the same note once


class TestTextCheck(unittest.TestCase):
    def setUp(self):
        self.tc = importlib.import_module(f"{PACK}.moss_textcheck_nodes")

    def test_compare_form_and_similarity(self):
        tc = self.tc
        self.assertEqual(tc.norm("  Northern-Lights! "), "northern lights")
        self.assertEqual(tc.similarity("NORTHERN LIGHTS", "northern lights"), 1.0)
        self.assertAlmostEqual(tc.similarity("NORTHERN", "NORTHEN"), 1 - 1 / 8)
        self.assertEqual(tc.verdict(1.0), "exact")
        self.assertEqual(tc.verdict(0.7), "close")
        self.assertEqual(tc.verdict(0.2), "missing")
        self.assertEqual(tc.clean_answer("NONE."), "")
        self.assertEqual(tc.clean_answer('Text: "OPEN"'), "OPEN")

    def test_layout_boxes_are_read_and_failures_masked(self):
        tc = self.tc
        layout = json.dumps({"width": 512, "height": 256, "items": [
            {"label": "text", "caption": "title", "text": "NORTHERN LIGHTS", "bbox": [0, 0, 256, 64]},
            {"label": "text", "caption": "sign", "text": "OPEN", "bbox": [256, 128, 512, 256]},
            {"label": "object", "caption": "a lamp", "bbox": [0, 128, 128, 256]}]})
        seen = []

        def reader(clip, image, question, max_tokens):
            seen.append((question, tuple(image.shape)))
            if question == tc.READ_ALL:
                return "NORTHERN LIGHTS\nOPEN"
            return "NORTHERN LIGHTS" if len(seen) == 1 else "NONE"
        # the image is twice the layout canvas: boxes scale with it
        rows, mask, lines = tc.check(None, torch.rand(1, 512, 1024, 3), layout, "", 64, reader)
        self.assertEqual([r["verdict"] for r in rows], ["exact", "outside its box"])
        self.assertEqual(rows[1]["read_elsewhere"], "OPEN")
        self.assertGreaterEqual(min(seen[0][1][1:3]), 384)  # small crops are upscaled for the reader
        self.assertEqual(float(mask[0, 400, 800]), 1.0)  # the failed OPEN box, in image pixels
        self.assertEqual(float(mask[0, 50, 100]), 0.0)   # the exact title box stays black

    def test_empty_read_never_passes_and_literal_flag(self):
        """Audit 0.8.1 #2: '!!!' against an empty read was exact (both comparison forms empty), so passed was True."""
        tc = self.tc
        self.assertEqual(tc.similarity("!!!", ""), 0.0)
        self.assertEqual(tc.similarity("!!!", "!!!"), 1.0)
        self.assertLess(tc.similarity("!!!", "?"), 1.0)
        self.assertEqual(tc.similarity("HaDeS", ""), 0.0)
        layout = json.dumps({"width": 64, "height": 64, "items": [{"label": "text", "caption": "x", "text": "!!!", "bbox": [0, 0, 64, 32]},
                                                                  {"label": "text", "caption": "y", "text": "12.50", "bbox": [0, 32, 64, 64]}]})
        reads = iter(["NONE", "12 50", "12 50"])
        rows, mask, _ = tc.check(None, torch.rand(1, 64, 64, 3), layout, "", 64, lambda *a: next(reads))
        self.assertEqual(rows[0]["verdict"], "missing")
        self.assertEqual(float(mask[0, 10, 10]), 1.0)
        # the comparison form folds punctuation: exact, flagged as not literal and said so in the report
        self.assertEqual((rows[1]["verdict"], rows[1]["literal"]), ("exact", False))
        node = tc.HaDeSMossTextCheck()
        clip = type("C", (), {"generate": None})()
        check = tc.check
        with patch.object(tc, "check", lambda clip, image, layout, expected, mt: check(clip, image, layout, expected, mt, lambda *a: "12 50")):
            report = node.run(torch.rand(1, 64, 64, 3), clip, expected_text="12.50")["result"][0]
        self.assertIn("case or punctuation differ", report)

    def test_thin_box_upscale_is_bounded(self):
        """Audit 0.8.1: a 1024 x 4 box was upscaled to 98304 x 384 before the reader's 1 MP limit."""
        crop, _ = self.tc.crop_box(torch.rand(1, 64, 1024, 3), (0, 30, 1024, 34), (1024, 64), margin=0.0)
        self.assertLessEqual(max(crop.shape[1:3]), 2048)
        crop, _ = self.tc.crop_box(torch.rand(1, 64, 64, 3), (0, 0, 64, 32), (64, 64), margin=0.0)
        self.assertEqual(min(crop.shape[1:3]), 384)  # small boxes still reach the reader's minimum

    def test_a_typo_in_its_box_is_close_not_outside(self):
        """Measured case: the box and the whole image both read 'Hello Worly'; the box also holding other text."""
        tc = self.tc
        layout = json.dumps({"width": 64, "height": 64, "items": [{"label": "text", "caption": "x", "text": "Hello World", "bbox": [0, 0, 64, 32]}]})
        reader = lambda clip, image, question, max_tokens: "RELI STOR.\nHello Worly" if question == tc.READ_BOX else "Hello Worly"
        rows, mask, _ = tc.check(None, torch.rand(1, 64, 64, 3), layout, "", 64, reader)
        self.assertEqual(rows[0]["verdict"], "close")
        self.assertEqual(rows[0]["read"], "Hello Worly")
        self.assertEqual(float(mask[0, 10, 10]), 1.0)

    def test_text_split_over_three_lines_is_exact(self):
        """Measured case: MOSS painted GOOD / VIBES / ONLY on three lines of a t-shirt."""
        tc = self.tc
        reader = lambda clip, image, question, max_tokens: "GOOD\nVIBES\nONLY\nsize M"
        rows, _, _ = tc.check(None, torch.rand(1, 64, 64, 3), "", "GOOD VIBES ONLY", 64, reader)
        self.assertEqual(rows[0]["verdict"], "exact")

    def test_expected_text_without_layout_and_errors(self):
        tc = self.tc
        reader = lambda clip, image, question, max_tokens: "CAFE LUNA\nOPEN 24H"
        rows, mask, _ = tc.check(None, torch.rand(1, 64, 64, 3), "", "Cafe Luna\nCLOSED", 64, reader)
        self.assertEqual([r["verdict"] for r in rows], ["exact", "missing"])
        self.assertEqual(float(mask.sum()), 0.0)
        # a caption draft with no quoted text: nothing to read, no error, the workflow goes on (user report 2026-11-19)
        rows, mask, lines = tc.check(None, torch.rand(1, 64, 64, 3), json.dumps({"caption": "a cup"}), "", 64, reader)
        self.assertEqual((rows, float(mask.sum()), lines), ([], 0.0, []))
        clip = type("C", (), {"generate": None})()
        out = tc.HaDeSMossTextCheck().run(torch.rand(1, 64, 64, 3), clip, layout=json.dumps({"caption": "a cup"}))["result"]
        self.assertTrue(out[2])
        self.assertIn("nothing to check", out[0])
        self.assertTrue(common.parse_info_meta(out[3])["nothing_to_check"])
        # a caption draft (detailed / caption mode): its quoted texts are the texts asked for, anywhere in the image
        draft = json.dumps({"caption": 'A bakery whose sign reads "Cafe Luna", a chalkboard with “OPEN 24H” and "!!" in pencil.'})
        self.assertEqual(tc.quoted_texts(draft), ["Cafe Luna", "OPEN 24H"])
        rows, _, _ = tc.check(None, torch.rand(1, 64, 64, 3), draft, "", 64, reader)
        self.assertEqual([(r["text"], r["verdict"]) for r in rows], [("Cafe Luna", "exact"), ("OPEN 24H", "exact")])
        with self.assertRaisesRegex(ValueError, "cannot generate text"):
            tc.HaDeSMossTextCheck().run(torch.rand(1, 64, 64, 3), object(), expected_text="X")


class TestAuditFixes(unittest.TestCase):
    def test_ties_trim_keeps_exactly_k_with_ties(self):
        """Audit 0.8.0 #10: a magnitude threshold dropped every entry tied at the cut."""
        self.assertEqual(mergers.ties_trim(torch.ones(4), 0.5).tolist(), [1.0, 1.0, 0.0, 0.0])
        t = torch.tensor([3.0, -1.0, 2.0, -5.0])
        self.assertEqual(mergers.ties_trim(t, 0.5).tolist(), [3.0, 0.0, 0.0, -5.0])
        self.assertEqual(mergers.ties_trim(t, 1.0).tolist(), t.tolist())

    def test_write_atomic_keeps_the_old_file_on_failure(self):
        """Audit 0.8.0 #3: a failed save must not leave a partial file nor destroy the previous one."""
        path = os.path.join(TMP, "atomic.safetensors")
        with open(path, "w") as fh:
            fh.write("old")

        def boom(tmp):
            with open(tmp, "w") as fh:
                fh.write("partial")
            raise OSError("disk full")
        with self.assertRaises(OSError):
            common.write_atomic(path, boom)
        self.assertEqual(open(path).read(), "old")
        self.assertEqual([n for n in os.listdir(TMP) if n.startswith("atomic.") and n.endswith(".part")], [])
        # a failed export removes its empty placeholder; a file with content is left alone (audit 0.8.1)
        empty = common.reserve_path(lambda i: os.path.join(TMP, f"failed_{i}.safetensors"))
        with self.assertRaises(RuntimeError):
            with common.reserved(empty):
                raise RuntimeError("compression failed")
        self.assertFalse(os.path.exists(empty))
        with self.assertRaises(RuntimeError):
            with common.reserved(path):
                raise RuntimeError("later failure")
        self.assertEqual(open(path).read(), "old")
        a = common.reserve_path(lambda i: os.path.join(TMP, f"reserved_{i}.bin"))
        b = common.reserve_path(lambda i: os.path.join(TMP, f"reserved_{i}.bin"))
        self.assertNotEqual(a, b)

    def test_two_saves_with_the_same_counter_get_two_files(self):
        """Audit 0.8.1 #3: the counter was read from the folder before the weights were ready, so two exports to the same prefix
        (two ComfyUI instances) could pick the same name and the second replaced the first."""
        model = load_int8(tiny_state_dict(22))
        real = saver.folder_paths.get_save_image_path
        frozen = lambda prefix, out_dir, *a: (*real(prefix, out_dir, *a)[:2], 1, *real(prefix, out_dir, *a)[3:])
        with patch.object(saver.folder_paths, "get_save_image_path", frozen):  # both see counter 1, as two instances would
            a = saver.HaDeSMossModelSaver().save(model, "tests/same_counter", saver.PRECISIONS[1], False)["result"][1]
            b = saver.HaDeSMossModelSaver().save(model, "tests/same_counter", saver.PRECISIONS[1], False)["result"][1]
        self.assertNotEqual(a, b)
        self.assertGreater(os.path.getsize(a), 0)
        self.assertGreater(os.path.getsize(b), 0)

    def test_layout_answer_without_any_box_keeps_a_draft(self):
        """Audit 0.8.0 #8: every item without a box raised before the draft could reach the Layout Editor."""
        draft, notes = enhancer.fit_draft({"summary": {"caption": "A poster."}, "items": [{"label": "text", "caption": "Title", "text": "HELLO"},
                                                                                          {"label": "text", "caption": "Subtitle"}]}, 512, 512)
        self.assertEqual(draft["items"], [])
        self.assertEqual([d["label"] for d in draft["dropped"]], ["text", "object"])  # a text item with no text comes back as object
        with self.assertRaisesRegex(ValueError, "Layout Editor"):
            pipeline.compile_layout(draft)

    def test_multi_lora_row_validation(self):
        """Audit 0.8.0 minor: "on": "false" stayed on, a misspelt group key was silently dropped."""
        self.assertEqual(loras.parse_lora_rows('[{"name": "x", "on": "false"}]')[0]["on"], False)
        with self.assertRaisesRegex(ValueError, "unknown group"):
            loras.parse_lora_rows('[{"name": "x", "groups": {"layers_00_5": 1.0}}]')

    def test_info_meta_must_be_an_object(self):
        self.assertIsNone(common.parse_info_meta("HADES_META []"))
        self.assertIsNone(common.parse_info_meta('HADES_META {"x": 1}'))


class TestDenoiseSchedule(unittest.TestCase):
    def setUp(self):
        from comfy.model_sampling import ModelSamplingDiscreteFlow
        self.ms = ModelSamplingDiscreteFlow()
        self.ms.set_parameters(shift=5.0, multiplier=1000)

    def test_core_start_sigmas_match_comfy_schedule(self):
        # 4 steps, shift 5, simple: int(steps / denoise) steps, the last 4 run
        expected = {0.3: (13, 0.690), 0.5: (8, 0.833), 0.6: (6, 0.909), 0.7: (5, 0.952), 0.8: (5, 0.952), 0.9: (4, 1.0)}
        for d, (n, sigma) in expected.items():
            self.assertEqual(sampler.core_schedule_steps(4, d), n)
            self.assertAlmostEqual(sampler.start_sigma(self.ms, "euler", "simple", 4, d, sampler.DENOISE_CORE), sigma, places=3)

    def test_exact_is_continuous_and_full_at_one(self):
        full = sampler.full_sigmas(self.ms, "euler", "simple", 4)
        torch.testing.assert_close(sampler.exact_sigmas(self.ms, "euler", "simple", 4, 1.0), full, atol=1e-6, rtol=0)
        starts = [float(sampler.exact_sigmas(self.ms, "euler", "simple", 4, d)[0]) for d in (0.6, 0.7, 0.8, 0.89, 0.9)]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(set(round(s, 4) for s in starts)), 5)  # 0.7 and 0.8 differ, unlike the core schedule
        self.assertAlmostEqual(starts[3], 0.976, places=3)
        for d in (0.3, 0.7, 0.9):
            s = sampler.exact_sigmas(self.ms, "euler", "simple", 4, d)
            self.assertEqual(len(s), 5)
            self.assertEqual(float(s[-1]), 0.0)
            self.assertTrue(bool((s[:-1] > s[1:]).all()))
            self.assertAlmostEqual(float(s[0]), sampler.start_sigma(self.ms, "euler", "simple", 4, d, sampler.DENOISE_EXACT), places=4)

    def test_ksampler_modes_reach_the_sampler_and_info(self):
        seen = {}

        def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kw):
            seen["sigmas"] = kw.get("sigmas")
            return torch.zeros_like(latent_image)

        def fake_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
            seen["denoise"] = denoise
            return ({"samples": torch.zeros(1, 128, 4, 4)},)

        originals = sampler.comfy.sample.sample, sampler.nodes.common_ksampler
        sampler.comfy.sample.sample, sampler.nodes.common_ksampler = fake_sample, fake_ksampler
        try:
            run = lambda mode: common.parse_info_meta(sampler.HaDeSMossKSampler().sample(
                load(tiny_state_dict(12)), None, None, {"samples": torch.zeros(1, 128, 4, 4)}, 3, 4, 1.0, "euler", "simple", 0.7, 0.0, "off",
                denoise_mode=mode)[2])
            exact, core = run(sampler.DENOISE_EXACT), run(sampler.DENOISE_CORE)
        finally:
            sampler.comfy.sample.sample, sampler.nodes.common_ksampler = originals
        self.assertEqual(len(seen["sigmas"]), 5)
        self.assertAlmostEqual(float(seen["sigmas"][0]), exact["start_sigma"], places=4)
        self.assertEqual((exact["denoise_mode"], round(exact["start_sigma"], 3)), ("exact", 0.921))
        self.assertNotIn("schedule_steps", exact)
        self.assertEqual(seen["denoise"], 0.7)
        self.assertEqual((core["denoise_mode"], core["schedule_steps"], round(core["start_sigma"], 3)), ("core", 5, 0.952))

    def test_discard_penultimate_samplers_keep_length(self):
        self.assertEqual(len(sampler.exact_sigmas(self.ms, "dpm_2", "simple", 4, 0.7)), 5)

    def test_exact_start_sigma_is_the_schedule_actually_run(self):
        """Audit 0.8.0 #5: with schedulers starting below sigma max (ddim_uniform) the reported start was the requested point."""
        for sched in ("ddim_uniform", "simple", "beta"):
            s = sampler.exact_sigmas(self.ms, "euler", sched, 4, 0.7)
            self.assertAlmostEqual(sampler.start_sigma(self.ms, "euler", sched, 4, 0.7, sampler.DENOISE_EXACT), float(s[0]), places=5)


class TestTelemetry(unittest.TestCase):
    def test_every_trailer_is_known_to_telemetry(self):
        """The Suite's rule: every node writes a HADES_META trailer and Suite Telemetry names it instead of counting it as foreign."""
        import ast
        import glob
        names = set()
        for path in glob.glob(os.path.join(PACK_DIR, "moss_*.py")):
            for call in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) in ("format_info", "format_model_info") and len(call.args) > 1:
                    if isinstance(call.args[1], ast.Constant):  # moss_common's own wrapper passes a variable
                        names.add(call.args[1].value)
        self.assertEqual(names, set(telemetry.NODE_NAMES))  # no unknown trailer, no stale name

    def test_turbo_notes(self):
        f = common.format_info
        base = dict(width=1024, height=1024, steps=4, cfg=1.0, flow_shift=5.0, sampler="euler", scheduler="simple", seed=0, slg="off")
        cases = {"measured": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0), 0),
                 "strength": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=0.5), 1),
                 "denoise": (dict(denoise=0.6, turbo="t.safetensors", turbo_strength=1.0), 1),
                 "steps": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, steps=6), 1),
                 "8 steps": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, steps=8), 0),
                 "cfg 1.5": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, cfg=1.5), 1),
                 "er_sde": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, sampler="er_sde"), 0),
                 "dpmpp_2m / karras": (dict(denoise=1.0, turbo="t.safetensors", turbo_strength=1.0, sampler="dpmpp_2m", scheduler="karras"), 1),
                 "no turbo, cfg 2.5": (dict(denoise=1.0, steps=25, cfg=2.5), 0),
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
        report, data, summary, info = telemetry.HaDeSMossSuiteTelemetry().audit("T", **{f"info_{i + 1}": s for i, s in enumerate(infos)})["result"]
        self.assertEqual(common.parse_info_meta(info)["error"], 1)
        levels = [x["level"] for x in json.loads(data)["findings"]]
        self.assertEqual((levels.count("error"), levels.count("warning"), levels.count("note")), (1, 2, 2))  # resize energy: a note
        self.assertIn("1 input(s) without a HaDeS MOSS trailer ignored", report)

    def test_energy_rules(self):
        f = common.format_info
        audit = lambda info: [(x["level"], x["message"][:14]) for x in json.loads(telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=info)["result"][1])["findings"]]
        self.assertEqual(audit(f("x", "lora_extractor", saved_path="p", rank=32, layers=1, energy_kept=0.906)), [("warning", "LoRA Extractor")])
        self.assertEqual(audit(f("x", "lora_extractor", saved_path="p", rank=64, layers=1, energy_kept=0.959)), [])
        self.assertEqual(audit(f("x", "lora_merger", saved_path="p", rank=2, layers=1, energy_kept=0.56)), [("note", "LoRA Merger: 5")])

    def test_denoise_schedule_rules(self):
        f = common.format_info
        base = dict(width=1024, height=1024, cfg=1.0, flow_shift=5.0, sampler="euler", scheduler="simple", seed=1, slg="off")
        def findings(**extra):
            data = telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=f("x", "ksampler", **{**base, **extra}))["result"][1]
            return [(x["level"], x["message"]) for x in json.loads(data)["findings"] if "denoise" in x["message"] and "Turbo" not in x["message"]]
        (level, msg), = findings(steps=4, denoise=0.9, denoise_mode="core", schedule_steps=4, start_sigma=1.0)
        self.assertEqual(level, "warning")
        self.assertIn("pure noise", msg)
        (level, msg), = findings(steps=4, denoise=0.7, denoise_mode="core", schedule_steps=5, start_sigma=0.9524)
        self.assertEqual(level, "note")
        self.assertIn("above 0.67 up to 0.80", msg)
        self.assertEqual(findings(steps=25, denoise=0.5, denoise_mode="core", schedule_steps=50), [])  # a 0.01-wide band: no note
        self.assertEqual(findings(steps=4, denoise=0.7, denoise_mode="exact", start_sigma=0.921), [])
        self.assertEqual(findings(steps=4, denoise=0.7), [])  # infos written before 0.7.3 have no schedule_steps

    def test_clean_pipeline(self):
        f = common.format_info
        infos = [f("x", "latent", width=1024, height=1024, batch=1),
                 f("x", "ksampler", width=1024, height=1024, steps=25, cfg=2.5, flow_shift=5.0, sampler="er_sde", scheduler="simple", seed=1, denoise=1.0, slg="off")]
        self.assertEqual(json.loads(telemetry.HaDeSMossSuiteTelemetry().audit("T", info_1=infos[0], info_2=infos[1])["result"][1])["findings"], [])


class FakeClip:
    """Stands in for the moss_image CLIP: records the chat text and answers with a fixed string."""
    def __init__(self, answer):
        self.answer, self.seen = answer, None

    def tokenize(self, text, images=None):
        if text.startswith("<|im_start|>"):  # the chat sent to generate(); later calls encode the conditionings
            self.seen = (text, images)
        return {"text": text}

    def generate(self, tokens, **kwargs):
        self.kwargs = kwargs
        return [0]

    def decode(self, ids):
        return self.answer

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", tokens["text"]]]


class TestPromptEnhancer(unittest.TestCase):
    def test_count_tags_read_as_adults_in_every_mode(self):
        # Danbooru 1girl / 1boy say nothing about age; "girl" in prose read as a child to MOSS-Image
        for mode in ("caption", "detailed", "layout"):
            for has_image in (False, True):
                text = enhancer.instructions(mode, has_image)
                self.assertIn("1girl", text)
                self.assertIn('"a young woman" or "a young man", never "girl", "boy" or "child", unless the idea itself states an age', text)

    def test_caption_parsing(self):
        self.assertEqual(enhancer.parse_caption('```json\n{"caption": "A red fox in snow."}\n```'), "A red fox in snow.")
        self.assertEqual(enhancer.parse_caption('Caption: "A red fox in snow."'), "A red fox in snow.")
        self.assertEqual(enhancer.parse_caption('<think>\nhmm {"x": 1}\n</think>\n\n{"caption": "A fox."}'), "A fox.")

    def test_detailed_mode_expands_and_caption_keeps_to_the_idea(self):
        detailed, caption = enhancer.instructions("detailed", False), enhancer.instructions("caption", False)
        self.assertIn("120 to 200 words", detailed)
        # a first try turned "anime girl on a rooftop" into an editorial photograph
        self.assertIn("The medium comes first and never changes", detailed)
        self.assertNotIn("Keep a minimalist brief minimalist", detailed)
        self.assertIn("Keep a minimalist brief minimalist", caption)
        self.assertNotIn("120 to 200 words", caption)
        self.assertIn('{"caption": "..."}', detailed)  # same answer format, parsed like caption
        out = enhancer.HaDeSMossPromptEnhancer().enhance(FakeClip('{"caption": "A long detailed fox."}'), "a fox", "detailed", 1024, 1024, 0.7, 0, 256)
        self.assertEqual(out["result"][2], "A long detailed fox.")

    def test_empty_caption_is_an_error(self):
        # an empty caption used to be encoded as the raw JSON answer
        for answer in ('{"caption": ""}', "   "):
            with self.assertRaisesRegex(ValueError, "no caption"):
                enhancer.HaDeSMossPromptEnhancer().enhance(FakeClip(answer), "a fox", "caption", 1024, 1024, 0.7, 0, 256)

    def test_layout_draft_is_fitted_and_compiles(self):
        answer = ('Here is the draft:\n```json\n{"width": 512, "height": 512, "summary": {"caption": "A poster.", "canvas": "single", "id": 3},'
                  '"items": [{"label": "text", "caption": "Bold title.", "text": "HELLO", "bbox": [100.4, -20, 1100, 200.6]},'
                  '{"label": "object", "caption": "No box."},'
                  '{"label": "thing", "caption": "A sun.", "bbox_2d": [300, 300, 200, 400]}]}\n```')
        node = enhancer.HaDeSMossPromptEnhancer()
        clip = FakeClip(answer)
        pos, neg, prompt, draft, w, h, info = node.enhance(clip, "a poster", "layout", 1024, 768, 0.0, 1, 512)["result"]
        d = json.loads(draft)
        self.assertEqual((w, h, d["width"], d["height"]), (1024, 768, 1024, 768))
        self.assertEqual([it["bbox"] for it in d["items"]], [[100, 0, 1024, 201], [200, 300, 300, 400]])
        self.assertEqual([it["label"] for it in d["items"]], ["text", "object"])
        self.assertNotIn("id", d["summary"])
        self.assertEqual(json.loads(prompt)["items"][0]["text"], "HELLO")
        self.assertIn("left out", info)
        # the item without a box is kept for the Layout Editor, and the compiler leaves it out of the model prompt
        self.assertEqual(d["dropped"], [{"label": "object", "caption": "No box.", "reason": "no box in the answer"}])
        self.assertNotIn("No box.", prompt)
        self.assertNotIn("dropped", json.loads(prompt))
        self.assertIn("Canvas: width 1024, height 768.", clip.seen[0])
        self.assertFalse(clip.kwargs["do_sample"])  # temperature 0 is greedy
        self.assertEqual(pos, [["cond", prompt]])
        self.assertEqual(neg, [["cond", enhancer.DEFAULT_NEGATIVE]])

    def test_caption_mode_and_image_tokens(self):
        clip = FakeClip('{"caption": "An old fisherman mending nets at dawn."}')
        image = torch.rand(2, 1500, 1000, 3)
        pos, neg, prompt, draft, *_ = enhancer.HaDeSMossPromptEnhancer().enhance(clip, "", "caption", 1024, 1024, 0.7, 5, 256, image=image,
                                                                                 negative="blurry")["result"]
        self.assertEqual(prompt, "An old fisherman mending nets at dawn.")
        self.assertEqual(json.loads(draft), {"caption": prompt})
        text, images = clip.seen
        self.assertEqual(text.count(enhancer.VISION_BLOCK), 1)
        self.assertEqual(len(images), 1)
        self.assertLessEqual(images[0].shape[1] * images[0].shape[2], 1024 * 1024)
        self.assertEqual(images[0].shape[0], 1)
        self.assertEqual(neg, [["cond", "blurry"]])
        self.assertIn("Keep out of the image:\nblurry", text)

    def test_as_written_is_a_plain_prompt_box(self):
        clip = FakeClip("must not be used")
        pos, neg, prompt, draft, w, h, info = enhancer.HaDeSMossPromptEnhancer().enhance(clip, "a red fox", "as written", 1010, 1024, 0.7, 0, 256)["result"]
        self.assertEqual((pos, neg, prompt), ([["cond", "a red fox"]], [["cond", enhancer.DEFAULT_NEGATIVE]], "a red fox"))
        self.assertEqual((w, h), (1008, 1024))
        self.assertFalse(hasattr(clip, "kwargs"))  # nothing generated

    def test_edit_mode_adds_the_reference(self):
        class EditClip(FakeClip):
            def tokenize(self, text, images=None):
                return {"text": text, "images": len(images or [])}

            def encode_from_tokens_scheduled(self, tokens):
                return [[tokens["text"], {"images": tokens["images"]}]]

        class FakeVAE:
            def encode(self, img):
                self.size = tuple(img.shape)
                return torch.zeros(1, 128, img.shape[1] // 16, img.shape[2] // 16)

        image, vae = torch.rand(1, 600, 1200, 3), FakeVAE()  # 2:1
        for fmt, vision in (("Moss-Image-Edit", False), ("ai-toolkit", True)):
            with self.subTest(fmt):
                pos, neg, prompt, draft, w, h, info = enhancer.HaDeSMossPromptEnhancer().enhance(
                    EditClip("unused"), "her hair blonde", "as written", 1024, 1024, 0.7, 0, 256, image=image, edit_format=fmt, vae=vae)["result"]
                self.assertEqual((w, h), (1456, 720))  # about 1 MP at 2:1, 16 px grid
                self.assertEqual(vae.size, (1, 720, 1456, 3))
                for cond in (pos, neg):
                    self.assertEqual(cond[0][1]["reference_latents_method"], enhancer.EDIT_FORMATS[fmt][0])
                    self.assertEqual(tuple(cond[0][1]["reference_latents"][0].shape), (1, 128, 45, 91))
                    self.assertEqual(cond[0][0].startswith("Picture 1:"), vision)
                self.assertEqual(common.parse_info_meta(info)["edit_format"], fmt)
        with self.assertRaisesRegex(ValueError, "needs the image"):
            enhancer.HaDeSMossPromptEnhancer().enhance(EditClip("x"), "x", "as written", 1024, 1024, 0.7, 0, 256, edit_format="ai-toolkit")

    def test_needs_idea_or_image(self):
        with self.assertRaisesRegex(ValueError, "idea"):
            enhancer.HaDeSMossPromptEnhancer().enhance(FakeClip(""), "  ", "caption", 1024, 1024, 0.7, 0, 256)

    def test_detailed_with_no_idea_writes_a_random_prompt(self):
        clip = FakeClip('{"caption": "An invented scene."}')
        out = enhancer.HaDeSMossPromptEnhancer().enhance(clip, "  ", "detailed", 1024, 1024, 0.7, 7, 256)
        self.assertEqual(out["result"][2], "An invented scene.")
        medium, subject, place = enhancer.surprise_brief(7)
        chat_text = clip.seen[0]
        self.assertIn(f"Medium: {medium}", chat_text)
        self.assertIn(f"Place and time: {place}", chat_text)
        self.assertIn("No idea is given", chat_text)
        self.assertEqual(common.parse_info_meta(out["result"][6])["random_brief"], [medium, subject, place])
        # the seed decides the brief, so the same seed gives the same draw and seeds spread over the lists
        self.assertEqual(enhancer.surprise_brief(7), enhancer.surprise_brief(7))
        self.assertGreater(len({enhancer.surprise_brief(s)[0] for s in range(40)}), 12)
        # with an idea, no draw and no surprise instructions
        clip2 = FakeClip('{"caption": "A fox."}')
        enhancer.HaDeSMossPromptEnhancer().enhance(clip2, "a fox", "detailed", 1024, 1024, 0.7, 7, 256)
        self.assertNotIn("No idea is given", clip2.seen[0])

    def test_decode_fallback_only_on_stride_errors(self):
        try:
            import comfy_kitchen
            import comfy_kitchen.flash_attention as fa
            comfy_kitchen.flash_attention_decode_gqa, fa._decode_gqa_torch
        except (ImportError, AttributeError):
            self.skipTest("comfy-kitchen without the GQA decode kernel")
        native, torch_impl = comfy_kitchen.flash_attention_decode_gqa, fa._decode_gqa_torch
        calls = []

        def refusing(*a, **k):
            calls.append("native")
            raise RuntimeError("Unsupported Flash Attention tensor strides")
        comfy_kitchen.flash_attention_decode_gqa = refusing
        fa._decode_gqa_torch = lambda q, k, v, l, c: (calls.append("torch") or ("out", "lse"))
        try:
            with enhancer.decode_fallback() as state:
                self.assertEqual(comfy_kitchen.flash_attention_decode_gqa(1, 2, 3, 4), "out")
                self.assertEqual(comfy_kitchen.flash_attention_decode_gqa(1, 2, 3, 4, return_lse=True), ("out", "lse"))
            self.assertTrue(state["fallback"])
            self.assertEqual(calls, ["native", "torch", "torch"])  # after the first refusal the native kernel is not retried
            self.assertIs(comfy_kitchen.flash_attention_decode_gqa, refusing)  # restored on exit
        finally:
            comfy_kitchen.flash_attention_decode_gqa, fa._decode_gqa_torch = native, torch_impl


class TestMetadataReader(unittest.TestCase):
    def test_recipe_dependencies_invalidate_reader(self):
        folder_paths.folder_names_and_paths["diffusion_models"] = ([TMP], {".safetensors"})
        leaf, root = "dependency_leaf.safetensors", "dependency_root.safetensors"
        self.lora_file(os.path.join(TMP, leaf), 2, {})
        self.lora_file(os.path.join(TMP, root), 2, {"hades/source_lora": leaf, "hades/target_rank": "2"})
        reader = metadata.HaDeSMossMetadataReader
        before = reader.IS_CHANGED(root, metadata.NONE)
        self.lora_file(os.path.join(TMP, leaf), 3, {"changed": "yes"})
        self.assertNotEqual(before, reader.IS_CHANGED(root, metadata.NONE))
        self.assertEqual(reader.IS_CHANGED(root, metadata.NONE), reader.IS_CHANGED(root, metadata.NONE))
        model = "dependency_model.safetensors"
        source = {"node": "loader", "model_file": "missing_source.safetensors"}
        merge = common.format_info("merge", "model_block_merger", layer_ratios=[0.5] * LAYERS,
                                   part_ratios=dict.fromkeys(common.PARTS, 0.5), sources={"model1": source, "model2": source})
        safetensors.torch.save_file(tiny_state_dict(23), os.path.join(TMP, model), metadata={"hades/recipe": merge})
        before = reader.IS_CHANGED(metadata.NONE, model)
        safetensors.torch.save_file(tiny_state_dict(24), os.path.join(TMP, "missing_source.safetensors"))
        self.assertNotEqual(before, reader.IS_CHANGED(metadata.NONE, model))
        report, data = reader().read(metadata.NONE, model)["result"]
        operation = json.loads(data)["model"]["recipe"]["parts"][0]
        self.assertEqual(operation["settings"]["layer_ratios"], [0.5] * LAYERS)
        self.assertEqual(operation["parts"][0]["model"], "missing_source.safetensors")
        self.assertIn("model_block_merger", report)

    @staticmethod
    def lora_file(path, rank, meta):
        sd = {}
        for i in (2, 3):
            p = f"diffusion_model.layers.{i}.proj.o"
            sd.update({f"{p}.lora_up.weight": torch.zeros(DIM, rank), f"{p}.lora_down.weight": torch.zeros(rank, DIM), f"{p}.alpha": torch.tensor(float(rank))})
        safetensors.torch.save_file(sd, path, metadata=meta)

    def test_lora_model_and_recipe(self):
        """A resize of a merge whose first source was moved to a subfolder: the recipe is followed to both originals."""
        folder_paths.folder_names_and_paths["diffusion_models"] = ([TMP], {".safetensors"})
        os.makedirs(os.path.join(TMP, "moved"), exist_ok=True)
        self.lora_file(os.path.join(TMP, "moved", "meta_a.safetensors"), 4, {"settings": "8 steps"})
        self.lora_file(os.path.join(TMP, "meta_b.safetensors"), 2, {})
        self.lora_file(os.path.join(TMP, "meta_ab.safetensors"), 6, {"hades/lora_a": "meta_a.safetensors", "hades/lora_b": "meta_b.safetensors",
                                                                     "hades/weights": "0.5,1.0", "hades/rank": "6"})
        self.lora_file(os.path.join(TMP, "meta_r.safetensors"), 3, {"hades/source_lora": "meta_ab.safetensors", "hades/target_rank": "3",
                                                                    "hades/dynamic_energy": "0.0"})
        safetensors.torch.save_file(tiny_state_dict(10), os.path.join(TMP, "meta_model.safetensors"))
        info, js = metadata.HaDeSMossMetadataReader().read("meta_r.safetensors", "meta_model.safetensors")["result"]
        d = json.loads(js)
        lt, mt = d["lora"]["tensors"], d["model"]["tensors"]
        self.assertEqual((lt["rank_max"], lt["layers"], lt["layer_modules"], lt["alphas"]), (3, [2, 3], ["proj.o"], [3.0]))
        self.assertEqual(common.parse_info_meta(info)["recipe"], {os.path.join("moved", "meta_a.safetensors"): 0.5, "meta_b.safetensors": 1.0})
        self.assertEqual((mt["moss_image"], len(mt["layers"]), mt["fused_qkv"], mt["int8_convrot"]), (True, LAYERS, True, False))

    def test_model_recipe_is_saved_and_followed(self):
        """The Model Saver writes its recipe input; the reader follows the Loader's model and the Multi-LoRA's LoRAs named in it."""
        self.test_lora_model_and_recipe()  # the meta_* files
        recipe = "\n".join([common.format_info("HaDeS MOSS Loader | 6 layers", "loader", model_file="meta_model.safetensors"),
                             common.format_info("HaDeS MOSS Multi-LoRA | 1 of 1 on", "multi_lora", enabled=True,
                                                loras=[{"name": "meta_r.safetensors", "strength": 0.5, "groups": {"layers_00_05": 0.0}}], turbo=[])])
        path, info = saver.HaDeSMossModelSaver().save(load(tiny_state_dict(11)), "tests/recipe", "bfloat16", True, recipe=recipe)["result"][1:]
        self.assertTrue(common.parse_info_meta(info)["recipe"])
        name = os.path.relpath(path, TMP)
        text, js = metadata.HaDeSMossMetadataReader().read(metadata.NONE, name)["result"]
        r = json.loads(js)["model"]["recipe"]
        self.assertEqual(r["text"], recipe)
        loader_part, lora_part = r["parts"]
        self.assertEqual((loader_part["model"], "text" in loader_part), ("meta_model.safetensors", False))
        self.assertEqual(metadata.recipe_leaves(lora_part), {os.path.join("moved", "meta_a.safetensors"): 0.25, "meta_b.safetensors": 0.5})
        self.assertIn("model meta_model.safetensors  (no recipe in the file)", text)
        self.assertIn("[groups: layers_00_05 0]", text)
        # a recipe with include_metadata off is not written, and the info says so
        info = saver.HaDeSMossModelSaver().save(load(tiny_state_dict(11)), "tests/nometa", "bfloat16", False, recipe=recipe)["result"][2]
        self.assertIn("recipe NOT saved", info)
        self.assertTrue(common.parse_info_meta(text)["model_recipe"])
        # without a recipe the file says so
        path = saver.HaDeSMossModelSaver().save(load(tiny_state_dict(11)), "tests/norecipe", "bfloat16", True)["result"][1]
        self.assertIn("Recipe: none in the file", metadata.HaDeSMossMetadataReader().read(metadata.NONE, os.path.relpath(path, TMP))["result"][0])

    def test_extractor_writes_model_names(self):
        sd = tiny_state_dict(12)
        base = load(sd)
        sd["layers.1.proj.o.weight"] = (sd["layers.1.proj.o.weight"].float() + 0.01).to(torch.bfloat16)
        path = loras.HaDeSMossLoraExtractor().extract(load(sd), base, 4, "meta_extracted", overwrite_existing=True,
                                                      tuned_name="tuned.safetensors", base_name="base.safetensors")["result"][0]
        text = metadata.HaDeSMossMetadataReader().read(os.path.basename(path), metadata.NONE)["result"][0]
        self.assertIn("extracted from tuned.safetensors - base.safetensors", text)

    def test_errors(self):
        with open(os.path.join(TMP, "not_safetensors.safetensors"), "wb") as f:
            f.write(b"PK\x03\x04 not a safetensors file")
        with self.assertRaisesRegex(ValueError, "not a safetensors file"):
            metadata.HaDeSMossMetadataReader().read("not_safetensors.safetensors", metadata.NONE)
        with self.assertRaisesRegex(ValueError, "pick a LoRA"):
            metadata.HaDeSMossMetadataReader().read(metadata.NONE, metadata.NONE)


if __name__ == "__main__":
    unittest.main(argv=TEST_ARGV, verbosity=2)
