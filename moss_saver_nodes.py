import datetime
import json
import os
import re

import torch
import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.quant_ops
import comfy.utils

from .moss_common import count_layers, format_info, key_patches, key_patches_float, key_patches_weight, model_weights, stochastic_round_bf16

# The layout of Comfy-Org's moss_image_*_int8_convrot files: the four linears of every layer in int8 with ConvRot,
# int8 attention in every layer, everything else (text_embedder, modulations, embedders, head, norms) in bfloat16.
_INT8_LAYER_LINEAR_RE = re.compile(r"^layers\.\d+\.proj\.(qkv|q|k|v|o|ffn_up_gate|ffn_down)\.weight$")
_INT8_QUANT_CONF = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
_INT8_ATTENTION_CONF = {"attention": "comfy_kitchen_int8"}

PRECISIONS = ["bfloat16", "int8 ConvRot (Comfy-Org layout)", "float32"]
ROUNDINGS = ["stochastic", "nearest"]


def _json_tensor(obj):
    return torch.tensor(list(json.dumps(obj).encode("utf-8")), dtype=torch.uint8)


class HaDeSMossModelSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "diffusion_models/MOSS_HaDeS_Merged", "tooltip": "Subfolder and file name prefix inside ComfyUI/output."}),
                "precision": (PRECISIONS, {"tooltip": "bfloat16: same as the original files. int8 ConvRot: the layout of Comfy-Org's *_int8_convrot files "
                                                      "(layer linears in int8 with Hadamard rotation and int8 attention, the rest in bfloat16), "
                                                      "about half the size. float32: uncompressed."}),
                "include_metadata": ("BOOLEAN", {"default": True, "tooltip": "Write ModelSpec title / author and the HaDeS precision and layer count into the safetensors header."}),
            },
            "optional": {
                "custom_author": ("STRING", {"default": "HaDeS"}),
                "custom_model_title": ("STRING", {"default": ""}),
                "rounding": (ROUNDINGS, {"default": ROUNDINGS[0], "tooltip":
                    "bfloat16 output, weights with merge / LoRA patches. stochastic: each one rounded up or down with probability by "
                    "distance (seeded per weight, so the file is reproducible); keeps changes smaller than a bfloat16 step. nearest: "
                    "ComfyUI's own saving; such changes are partly lost. Measured with two subtle LoRAs baked into a merge: nearest kept "
                    "about 80-85% of their effect, stochastic all of it, with renders as close to the float32 result as two equally "
                    "valid roundings are to each other (tools/RESULTS.md). Unpatched weights are copied unchanged either way."}),
                "recipe": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Optional. How this model was made, written into the file as hades/recipe (with include_metadata). Link the info "
                    "output of the MOSS Loader, Multi-LoRA or a merger here (several joined with the core Concatenate node), or type notes. "
                    "The MOSS Metadata Reader shows it and follows the models and LoRAs it names."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "saved_path", "info")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Saves a MOSS-Image MODEL with its merge / LoRA patches baked in, loadable by the stock Load Diffusion Model node."

    def save(self, model, filename_prefix, precision, include_metadata, custom_author="HaDeS", custom_model_title="", rounding="stochastic", recipe=""):
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        os.makedirs(full_output_folder, exist_ok=True)
        out_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")

        # Same path as comfy.sd.save_checkpoint: with dynamic VRAM the patches are applied at cast time and never
        # live in the raw state_dict, so the weights go through the lazy patching casters.
        comfy.model_management.load_models_gpu([model])
        lazy_sd = model.model_state_dict_for_saving(model.model.diffusion_model, "diffusion_model.")

        quant_device = comfy.model_management.get_torch_device()
        int8 = precision.startswith("int8")
        dtype = torch.float32 if precision == "float32" else torch.bfloat16
        # Preserve matching int8 ConvRot pieces; every other quantized layout is converted with its patches.
        quantized = {k[:-len(".weight")] for k, w in model_weights(model).items() if k.endswith(".weight") and isinstance(w, comfy.quant_ops.QuantizedTensor)}
        preserved = set()
        if int8:
            for module in quantized:
                conf_key = f"{module}.comfy_quant"
                if _INT8_LAYER_LINEAR_RE.match(f"{module}.weight") and conf_key in lazy_sd:
                    conf = json.loads(bytes(lazy_sd[conf_key].to("cpu").tolist()))
                    if all(conf.get(k) == v for k, v in _INT8_QUANT_CONF.items()):
                        preserved.add(module)
        stochastic = rounding == "stochastic" and dtype == torch.bfloat16 and not int8
        rounding_used = "stochastic" if stochastic else "none (float32)" if dtype == torch.float32 else "nearest"
        # float32 output: patched weights computed in float32 here; the core's lazy saving applies patches in the model's own
        # dtype (bfloat16), which would round them before they reach the float32 file
        exact = dtype == torch.float32

        def baked(pk):
            w32 = key_patches_float(key_patches(model, pk), quant_device, pk)
            return (w32 if exact else stochastic_round_bf16(w32, comfy.utils.string_to_seed(pk))).cpu().contiguous()

        def quantize(k, tensor):
            qt = comfy.quant_ops.QuantizedTensor.from_float(tensor.to(quant_device, torch.bfloat16), "TensorWiseINT8Layout",
                                                           is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
            return {**{qk: qv.cpu() for qk, qv in qt.state_dict(k).items()},
                    f"{k[:-len('.weight')]}.comfy_quant": _json_tensor(_INT8_QUANT_CONF)}

        sd = {}
        for k, v in lazy_sd.items():
            module = k.rsplit(".", 1)[0]
            pk = f"diffusion_model.{k}"
            if module in quantized - preserved:
                if k == f"{module}.weight":
                    if int8 and _INT8_LAYER_LINEAR_RE.match(k):
                        sd.update(quantize(k, key_patches_float(key_patches(model, pk), quant_device, pk)))
                    elif (stochastic or exact) and pk in model.patches:
                        sd[k] = baked(pk)
                    else:
                        sd[k] = key_patches_weight(key_patches(model, pk), quant_device, pk).to(dtype).cpu().contiguous()
                continue  # scales and quantization configuration must belong to the output layout
            if k.endswith("comfy_attention.config"):
                continue  # int8 output gets the Comfy-Org setting below; float output keeps no quantized attention
            if module in preserved:
                sd[k] = v.to("cpu").contiguous()
                continue
            if (stochastic or exact) and pk in model.patches:
                # the core's saving rounds bfloat16 to nearest (its stochastic_rounding() is stochastic for fp8 only)
                sd[k] = baked(pk)
                continue
            tensor = v.to("cpu").contiguous()
            if isinstance(v, comfy.model_patcher.LazyCastingParamPiece) or not torch.is_floating_point(tensor):
                sd[k] = tensor
            elif int8 and _INT8_LAYER_LINEAR_RE.match(k):
                sd.update(quantize(k, tensor))
            else:
                sd[k] = tensor.to(dtype)

        n_layers = count_layers(sd.keys())
        if int8:
            for i in range(n_layers):
                sd[f"layers.{i}.comfy_attention.config"] = _json_tensor(_INT8_ATTENTION_CONF)

        metadata = None
        if include_metadata:
            metadata = {
                "format": "pt",
                "modelspec.architecture": "moss_image",
                "modelspec.sai_model_spec": "1.0.0",
                "modelspec.title": custom_model_title or filename,
                "modelspec.author": custom_author or "HaDeS",
                "hades/precision": precision,
                "hades/rounding": rounding_used,
                "hades/layers": str(n_layers),
                "hades/timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            if recipe.strip():
                metadata["hades/recipe"] = recipe.strip()
        comfy.utils.save_torch_file(sd, out_path, metadata=metadata)

        size_gb = os.path.getsize(out_path) / 1024 ** 3
        info = format_info(f"HaDeS MOSS Model Saver | {os.path.basename(out_path)} | {precision}{', stochastic rounding' if stochastic else ''} | "
                           f"{size_gb:.2f} GB | {n_layers} layers" + (" | recipe saved" if metadata and "hades/recipe" in metadata else "")
                           + (" | recipe NOT saved: include_metadata is off" if recipe.strip() and not include_metadata else ""),
                           "model_saver", saved_path=out_path, precision=precision, n_layers=n_layers, rounding=rounding_used,
                           recipe=bool(metadata and "hades/recipe" in metadata))
        return {"ui": {"saved_path": [out_path], "info": [info]}, "result": (model, out_path, info)}


NODE_CLASS_MAPPINGS = {"HaDeSMossModelSaver": HaDeSMossModelSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossModelSaver": "MOSS Model Saver"}
