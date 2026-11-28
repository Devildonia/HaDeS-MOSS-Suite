import math
import os
import time

import torch
import torch.nn.functional as F

import comfy.model_management
import folder_paths

from .moss_common import format_info

# encoder size, heads, layers, the four layers the head reads (1-based, as transformers numbers its stages), DPT neck channels, fusion
ARCH = {
    "depth_anything_v2_vitl.pth": ("Depth-Anything-V2-Large", 1024, 16, 24, [5, 12, 18, 24], [256, 512, 1024, 1024], 256),
    "depth_anything_v2_vitb.pth": ("Depth-Anything-V2-Base", 768, 12, 12, [3, 6, 9, 12], [96, 192, 384, 768], 128),
    "depth_anything_v2_vits.pth": ("Depth-Anything-V2-Small", 384, 6, 12, [3, 6, 9, 12], [48, 96, 192, 384], 64),
}
CKPTS = list(ARCH)
INPUT_SIZE, MULTIPLE = 518, 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WEIGHTS_HINT = ("MOSS Depth Map reads the Depth Anything V2 weights (depth_anything_v2_vitl.pth, 1.3 GB, from the Hugging Face repo "
                "depth-anything/Depth-Anything-V2-Large) from models/depth_anything, or from the folder comfyui_controlnet_aux keeps them in "
                "if that pack downloaded them. Nothing is downloaded here.")
_MODELS = {}  # ckpt -> (file identity, model); one model per checkpoint, kept on the CPU between runs

if "depth_anything" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("depth_anything", os.path.join(folder_paths.models_dir, "depth_anything"))


def weights_path(ckpt):
    """The weights file: models/depth_anything (or a folder added for it in extra_model_paths), else where comfyui_controlnet_aux
    downloads it (its ckpts folder, or AUX_ANNOTATOR_CKPTS_PATH). Read as a file only: no code of that pack is imported. None if absent."""
    path = folder_paths.get_full_path("depth_anything", ckpt)
    if path:
        return path
    repo = ARCH[ckpt][0]
    roots = [os.environ.get("AUX_ANNOTATOR_CKPTS_PATH", "")]
    roots += [os.path.join(base, "comfyui_controlnet_aux", "ckpts") for base in folder_paths.get_folder_paths("custom_nodes")]
    for root in roots:
        p = os.path.join(root, "depth-anything", repo, ckpt) if root else ""
        if p and os.path.isfile(p):
            return p
    return None


def file_identity(path):
    st = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns)


V5_ATTENTION = ((".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
                (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj."))


def convert_state_dict(sd, layers, target_keys=None):
    """The original Depth Anything V2 checkpoint's keys as transformers' DepthAnythingForDepthEstimation names them (fused q / k / v
    split; the DPT fusion stages in transformers' order, deepest first). transformers 5 renamed DINOv2's attention (q_proj ... o_proj):
    with the installed model's `target_keys` the names follow whichever it uses."""
    out = _convert(sd)
    if target_keys is not None and "backbone.encoder.layer.0.attention.q_proj.weight" in target_keys:
        renamed = {}
        for k, v in out.items():
            for old, new in V5_ATTENTION:
                k = k.replace(old, new)
            renamed[k] = v
        out = renamed
    return out


def _convert(sd):
    out = {}
    fusion = {1: 3, 2: 2, 3: 1, 4: 0}
    for k, v in sd.items():
        n = None
        if k.startswith("pretrained."):
            r = k[len("pretrained."):]
            if r in ("cls_token", "mask_token"):
                n = f"backbone.embeddings.{r}"
            elif r == "pos_embed":
                n = "backbone.embeddings.position_embeddings"
            elif r.startswith("patch_embed.proj."):
                n = "backbone.embeddings.patch_embeddings.projection." + r.split(".")[-1]
            elif r.startswith("norm."):
                n = "backbone.layernorm." + r.split(".")[-1]
            elif r.startswith("blocks."):
                _, i, rest = r.split(".", 2)
                base = f"backbone.encoder.layer.{i}."
                if rest.startswith("attn.qkv."):
                    kind = rest.split(".")[-1]
                    for j, name in enumerate(("query", "key", "value")):
                        out[f"{base}attention.attention.{name}.{kind}"] = v.chunk(3, dim=0)[j].clone()
                    continue
                n = base + {"attn.proj.weight": "attention.output.dense.weight", "attn.proj.bias": "attention.output.dense.bias",
                            "ls1.gamma": "layer_scale1.lambda1", "ls2.gamma": "layer_scale2.lambda1"}.get(rest, rest)
        elif k.startswith("depth_head."):
            r = k[len("depth_head."):]
            p = r.split(".")
            if p[0] == "projects":
                n = f"neck.reassemble_stage.layers.{p[1]}.projection.{p[2]}"
            elif p[0] == "resize_layers":
                n = f"neck.reassemble_stage.layers.{p[1]}.resize.{p[2]}"
            elif p[0] == "scratch" and p[1].startswith("layer") and p[1].endswith("_rn"):
                n = f"neck.convs.{int(p[1][5]) - 1}.{p[2]}"
            elif p[0] == "scratch" and p[1].startswith("refinenet"):
                base = f"neck.fusion_stage.layers.{fusion[int(p[1][-1])]}."
                if p[2] == "out_conv":
                    n = base + f"projection.{p[3]}"
                else:
                    unit = {"resConfUnit1": "residual_layer1", "resConfUnit2": "residual_layer2"}[p[2]]
                    n = base + f"{unit}.convolution{p[3][-1]}.{p[4]}"
            elif r.startswith("scratch.output_conv1."):
                n = "head.conv1." + p[-1]
            elif r.startswith("scratch.output_conv2.0."):
                n = "head.conv2." + p[-1]
            elif r.startswith("scratch.output_conv2.2."):
                n = "head.conv3." + p[-1]
        if n is None:
            raise RuntimeError(f"MOSS Depth Map: unexpected weight {k!r}; this is not a Depth Anything V2 checkpoint the node knows.")
        out[n] = v
    return out


def build_model(ckpt, path):
    """transformers' Depth Anything (a dependency of ComfyUI itself) with the original checkpoint's weights; its DINOv2 attention scales
    once (the clean map of tools/RESULTS.md). Every weight must match: a missing or extra one is an error, not a silent default."""
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config

    _, hidden, heads, layers, out_indices, neck, fusion = ARCH[ckpt]
    backbone = Dinov2Config(hidden_size=hidden, num_attention_heads=heads, num_hidden_layers=layers, intermediate_size=4 * hidden,
                            patch_size=MULTIPLE, image_size=INPUT_SIZE, layerscale_value=1.0, out_indices=out_indices,
                            apply_layernorm=True, reshape_hidden_states=False,
                            attn_implementation="sdpa")  # DINOv2's memory-efficient attention (the depth head has none)
    # max_depth left at its default: a relative map is rescaled to 0-255 anyway, and newer transformers accept only an int there
    config = DepthAnythingConfig(backbone_config=backbone, reassemble_hidden_size=hidden, patch_size=MULTIPLE, neck_hidden_sizes=neck,
                                 fusion_hidden_size=fusion, head_hidden_size=32, depth_estimation_type="relative")
    model = DepthAnythingForDepthEstimation(config)
    sd = convert_state_dict(torch.load(path, map_location="cpu", weights_only=True), layers, set(model.state_dict()))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MOSS Depth Map: {os.path.basename(path)} does not fit Depth Anything V2 {ARCH[ckpt][0]} "
                           f"(missing {list(missing)[:4]}, unexpected {list(unexpected)[:4]}).")
    return model.eval()


def get_model(ckpt, path):
    """The model for this file, reloaded when the file under that name changes and the old one dropped (audit 0.9.9 M4)."""
    ident = file_identity(path)
    cached = _MODELS.get(ckpt)
    if cached is not None and cached[0] == ident:
        return cached[1]
    _MODELS.pop(ckpt, None)
    model = build_model(ckpt, path)
    _MODELS[ckpt] = (ident, model)
    return model


def multiple_of(x, minimum):
    """Depth Anything V2's resize: the nearest multiple of 14, not below `minimum`."""
    y = int(round(x / MULTIPLE) * MULTIPLE)
    return y if y >= minimum else int(math.ceil(x / MULTIPLE) * MULTIPLE)


def depth_map(model, img, resolution, device):
    """One image (H x W x 3, 0-1) -> its depth map (resolution on the shorter side, 3 channels, near is white), with the original
    pipeline's steps: the shorter side to 518 (multiple of 14, bicubic), ImageNet normalisation, depth back to the image size
    (bilinear), min-max to 0-255 in whole steps, then to the requested size (bicubic up, area down)."""
    h, w = int(img.shape[0]), int(img.shape[1])
    scale = max(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = multiple_of(scale * h, INPUT_SIZE), multiple_of(scale * w, INPUT_SIZE)
    x = img[:, :, :3].permute(2, 0, 1)[None].to(device, torch.float32)
    x = (torch.round(x.clamp(0, 1) * 255) / 255)  # the 8-bit image the original reads
    x = F.interpolate(x, (nh, nw), mode="bicubic", align_corners=False)
    x = (x - torch.tensor(MEAN, device=device)[:, None, None]) / torch.tensor(STD, device=device)[:, None, None]
    depth = model(pixel_values=x).predicted_depth
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).floor()
    k = float(resolution) / float(min(h, w))
    th, tw = int(round(h * k)), int(round(w * k))
    if (th, tw) != (h, w):
        mode = {"mode": "bicubic", "align_corners": False} if k > 1 else {"mode": "area"}
        depth = F.interpolate(depth[None, None], (th, tw), **mode)[0, 0]
    depth = torch.round(depth).clamp(0, 255) / 255.0
    return depth[:, :, None].expand(-1, -1, 3).float().cpu()


class HaDeSMossDepthMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photo or render whose pose you want to keep."}),
                "ckpt_name": (CKPTS, {"default": CKPTS[0], "tooltip": "Depth Anything V2 weights, from models/depth_anything or "
                                      "comfyui_controlnet_aux's ckpts folder. vitl (Large) is the one measured."}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64,
                                       "tooltip": "Size of the map's shorter side. The model itself reads the image at 518 px on its shorter "
                                                  "side whatever this value; 1024 is the measured setting."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_map", "info")
    FUNCTION = "make"
    CATEGORY = "HaDeS/MOSS Pipeline"
    DESCRIPTION = ("A depth map for MOSS Control (near is white) with Depth Anything V2, its attention scaled once. "
                   "comfyui_controlnet_aux 1.1.6's own node scales it twice and gives a blur with no outline on this install. "
                   "The model is transformers' (installed with ComfyUI) with the original weights; no other node pack's code is used, "
                   "and nothing is downloaded.")

    def make(self, image, ckpt_name, resolution):
        path = weights_path(ckpt_name)
        if path is None:
            raise RuntimeError(f"MOSS Depth Map: {ckpt_name} not found. " + WEIGHTS_HINT)
        device = comfy.model_management.get_torch_device()
        start = time.perf_counter()
        model = get_model(ckpt_name, path)
        model.to(device)
        try:
            with torch.no_grad():
                maps = [depth_map(model, img, int(resolution), device) for img in image]
        finally:
            model.to("cpu")
            comfy.model_management.soft_empty_cache()
        seconds = time.perf_counter() - start
        out = torch.stack(maps)
        h, w = int(out.shape[1]), int(out.shape[2])
        text = (f"HaDeS MOSS Depth Map | {ckpt_name} | {len(maps)} image(s) | {w} x {h} | {seconds:.1f} s\n"
                "Attention scaled once (comfyui_controlnet_aux's own node scales it twice on this install). Near is white.")
        info = format_info(text, "depth_map", ckpt=ckpt_name, resolution=int(resolution), images=len(maps), width=w, height=h,
                           attention="single scale")
        return (out, info)


NODE_CLASS_MAPPINGS = {"HaDeSMossDepthMap": HaDeSMossDepthMap}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossDepthMap": "MOSS Depth Map"}
