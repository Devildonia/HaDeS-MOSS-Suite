import json
import math
import os
import re
import struct
from collections import Counter

import folder_paths

from .moss_common import HADES_META_PREFIX, format_info
from .moss_merger_nodes import GROUPS
from .moss_image_saver_nodes import file_sha256

NONE = "(none)"
_MAX_HEADER = 100 * 1024 ** 2  # safetensors caps its header at 100 MB too
_MAX_DEPTH = 12
_SHOWN_CHARS = 240  # metadata values longer than this are cut in the info text (metadata_json keeps them whole)

_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "lora_unet_", "transformer.", "base_model.model.")
_LAYER_RE = re.compile(r"(?:^|[._])layers[._](\d+)(?=[._]|$)")
_LAYER_MODULES = ("proj.qkv", "proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down", "norm1", "norm2", "q_norm", "k_norm")
_OTHER_MODULES = ("text_embedder", "time_embedding", "video_embedder", "shared_block_modulation", "final_norm_video", "final_linear_video",
                  "video_head_modulation")
_LORA_DOWN_RE = re.compile(r"[._](?:lora_down|lora_A)(?:\.[^.]+)?\.weight$")


def _name_re(name):
    """A module name matched with '.' or '_' between its words, as kohya files write `proj_ffn_up_gate` for `proj.ffn_up_gate`."""
    return re.compile(r"(?:^|[._])" + "[._]".join(re.escape(p) for p in re.split(r"[._]", name)) + r"(?=[._]|$)")


_LAYER_MODULE_RES = [(n, _name_re(n)) for n in _LAYER_MODULES]
_OTHER_MODULE_RES = [(n, _name_re(n)) for n in _OTHER_MODULES]


def read_header(path):
    """(metadata, tensor entries) from a safetensors header: only the first bytes of the file are read, never the weights."""
    with open(path, "rb") as f:
        raw = f.read(8)
        n = struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
        if not 2 <= n <= _MAX_HEADER:
            raise ValueError(f"HaDeS MOSS Metadata Reader: '{os.path.basename(path)}' is not a safetensors file (only safetensors headers are read).")
        try:
            header = json.loads(f.read(n))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"HaDeS MOSS Metadata Reader: '{os.path.basename(path)}' has an unreadable safetensors header ({e}).") from e
    meta = header.pop("__metadata__", None) or {}
    return {str(k): str(v) for k, v in meta.items()}, header, 8 + n


def _scalar(path, start, entry):
    """The value of a one-element tensor (a kohya alpha), read from its few bytes; None for dtypes it does not decode."""
    a, b = entry["data_offsets"]
    fmt = {"F32": "<f", "F16": "<e", "F64": "<d"}.get(entry["dtype"])
    if b - a > 8 or (fmt is None and entry["dtype"] != "BF16"):
        return None
    with open(path, "rb") as f:
        f.seek(start + a)
        raw = f.read(b - a)
    if entry["dtype"] == "BF16":
        return struct.unpack("<f", b"\x00\x00" + raw)[0] if len(raw) == 2 else None
    return struct.unpack(fmt, raw)[0] if len(raw) == struct.calcsize(fmt) else None


def _strip(key):
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def tensor_summary(tensors):
    """What the tensor names, shapes and dtypes say, whatever the metadata claims."""
    keys = [_strip(k) for k in tensors]
    layers = sorted({int(m.group(1)) for k in keys if (m := _LAYER_RE.search(k))})
    modules = [n for n, r in _LAYER_MODULE_RES if any(r.search(k) for k in keys if _LAYER_RE.search(k))]
    other = [n for n, r in _OTHER_MODULE_RES if any(r.search(k) for k in keys)]
    numel = sum(math.prod(t.get("shape") or [1]) for k, t in tensors.items()
                if not k.endswith(("comfy_quant", "weight_scale", "comfy_attention.config")))
    return {"tensors": len(tensors), "dtypes": dict(Counter(t.get("dtype", "?") for t in tensors.values()).most_common()),
            "layers": layers, "layer_modules": modules, "other_modules": other, "elements": numel}


def lora_summary(path, tensors, start):
    s = tensor_summary(tensors)
    ranks = [t["shape"][0] for k, t in tensors.items() if _LORA_DOWN_RE.search(k) and len(t.get("shape") or []) >= 2]
    kinds = Counter()
    for k in tensors:
        if _LORA_DOWN_RE.search(k):
            kinds["LoRA (" + ("PEFT lora_A / lora_B" if "lora_A" in k else "lora_up / lora_down") + ")"] += 1
        elif k.endswith(".diff") or k.endswith(".diff_b"):
            kinds["full diff"] += 1
        elif "hada_w1_a" in k:
            kinds["LoHa"] += 1
        elif "lokr_w1" in k:
            kinds["LoKr"] += 1
        elif k.endswith("video_embedder.weight") and len(tensors[k].get("shape") or []) == 2 and tensors[k]["shape"][1] > 128:
            kinds[f"control adapter: video_embedder widened to {tensors[k]['shape'][1]} input channels (replaced at full strength, "
                  "whatever the LoRA strength)"] += 1
    alphas = sorted({round(v, 4) for k, t in tensors.items() if k.endswith(".alpha") and (v := _scalar(path, start, t)) is not None})
    s.update(kinds=dict(kinds), rank_min=min(ranks) if ranks else None, rank_max=max(ranks) if ranks else None,
             alphas=alphas[:8] + (["..."] if len(alphas) > 8 else []))
    return s


def model_summary(tensors):
    s = tensor_summary(tensors)
    keys = [_strip(k) for k in tensors]
    s.update(moss_image=any(k.startswith("shared_block_modulation") for k in keys),
             fused_qkv=any(k.endswith("proj.qkv.weight") for k in keys),
             int8_convrot=any(k.endswith("comfy_quant") for k in keys),
             text_embedder_blocks=len({k.split(".")[2] for k in keys if k.startswith("text_embedder.blocks.")}))
    return s


def _weights(text):
    try:
        a, b = (float(x) for x in str(text).split(","))
        return a, b
    except ValueError:
        return None


def find_file(folder, name):
    """(name in the folder's list, path) for a name as a metadata key wrote it; a file moved to another subfolder since is found by its file
    name when exactly one file has it. (None, None) when it is not there."""
    path = folder_paths.get_full_path(folder, name)
    if path is not None:
        return name, path
    base = re.split(r"[\\/]", name)[-1].lower()
    hits = [n for n in folder_paths.get_filename_list(folder) if re.split(r"[\\/]", n)[-1].lower() == base]
    return (hits[0], folder_paths.get_full_path(folder, hits[0])) if len(hits) == 1 else (None, None)


def recipe(name, weight=1.0, chain=()):
    """The LoRA's recipe from the hades/ keys the Suite's Merger and Resize write, followed into the source files while they are in models/loras.
    A source replaced since under the same name is read as it is now."""
    node = {"lora": name, "weight": weight}
    found, path = find_file("loras", name)
    if path is None:
        return {**node, "missing": True}
    if found != name:
        node["found_as"] = found
    if os.path.abspath(path) in chain or len(chain) >= _MAX_DEPTH:
        return {**node, "loop": True}
    try:
        meta = read_header(path)[0]
    except (OSError, ValueError):
        return {**node, "unreadable": True}
    chain = chain + (os.path.abspath(path),)
    if "hades/lora_a" in meta and "hades/lora_b" in meta and (w := _weights(meta.get("hades/weights"))):
        node.update(op="merge", rank=meta.get("hades/rank"), sources=[recipe(meta["hades/lora_a"], weight * w[0], chain),
                                                                     recipe(meta["hades/lora_b"], weight * w[1], chain)])
    elif "hades/source_lora" in meta:
        dyn = meta.get("hades/dynamic_energy", "0")
        node.update(op="resize", rank=meta.get("hades/target_rank"), dynamic_energy=dyn, sources=[recipe(meta["hades/source_lora"], weight, chain)])
    elif "hades/energy_kept" in meta:
        node.update(op="extract", rank=meta.get("hades/rank"), energy_kept=meta["hades/energy_kept"],
                    tuned=meta.get("hades/model_tuned"), base=meta.get("hades/model_base"))
    return node


def recipe_lines(node, depth=0):
    pad = "  " * depth
    head = f"{pad}{node['weight']:+.4g} × {node['lora']}" + (f" (now {node['found_as']})" if node.get("found_as") else "")
    if node.get("groups"):
        head += "  [groups: " + ", ".join(f"{g} {w:g}" for g, w in node["groups"].items()) + "]"
    if node.get("missing"):
        return [head + "  (not in models/loras)"]
    if node.get("loop"):
        return [head + "  (loop or too deep, not followed)"]
    if node.get("unreadable"):
        return [head + "  (unreadable)"]
    op = node.get("op")
    if op == "merge":
        head += f"  = merge, rank {node['rank']}"
    elif op == "resize":
        head += f"  = resize to rank {node['rank']}" + (f", dynamic energy {node['dynamic_energy']}" if node["dynamic_energy"] not in ("0", "0.0") else "")
    elif op == "extract":
        models = f"{node['tuned'] or '?'} - {node['base'] or '?'}" if node.get("tuned") or node.get("base") else "two models"
        head += f"  = extracted from {models}, rank {node['rank']}, energy kept {float(node['energy_kept']):.1%}"
    return [head] + [line for s in node.get("sources", []) for line in recipe_lines(s, depth + 1)]


def recipe_leaves(node, out=None):
    """Weight of each original file in the end result (merge weights multiplied; a file used twice adds up)."""
    out = {} if out is None else out
    if node.get("sources"):
        for s in node["sources"]:
            recipe_leaves(s, out)
    else:
        name = node.get("found_as", node["lora"])
        out[name] = out.get(name, 0.0) + node["weight"]
    return out


def info_metas(text):
    """Every HADES_META trailer in a text that may join several info outputs."""
    out = []
    for line in str(text).splitlines():
        if line.startswith(HADES_META_PREFIX):
            try:
                out.append(json.loads(line[len(HADES_META_PREFIX):]))
            except json.JSONDecodeError:
                pass
    return out


def model_recipe_parts(m, chain, depth=0):
    """Follow a saved node recipe, including the input recipes captured by mergers."""
    if not isinstance(m, dict) or depth >= _MAX_DEPTH:
        return []
    kind = m.get("node")
    parts = []
    if kind == "loader" and m.get("model_file"):
        return [model_recipe(m["model_file"], chain)]
    for source in (m.get("sources") or {}).values():
        parts += model_recipe_parts(source, chain, depth + 1)
    if kind in ("model_block_merger", "multi_model_merger"):
        return [{"operation": kind, "settings": {k: v for k, v in m.items() if k not in ("node", "sources")}, "parts": parts}]
    if kind == "multi_lora":
        parts += [{**recipe(r["name"], float(r.get("strength", 1.0))), **({"groups": r["groups"]} if r.get("groups") else {})}
                  for r in m.get("loras", []) if r.get("name")]
    elif kind == "lora_block_weight" and m.get("lora"):
        groups = {g: w for g, w in zip(GROUPS, m.get("weights") or []) if w != 1.0}
        if m.get("other_layers", 1.0) != 1.0:
            groups["other_layers"] = m["other_layers"]
        parts.append({**recipe(m["lora"], float(m.get("strength_model", 1.0))), **({"groups": groups} if groups else {})})
    return parts


def model_recipe(name, chain=()):
    """A model's hades/recipe (written by the MOSS Model Saver), with the models and LoRAs its info trailers name followed into their own
    recipes: the MOSS Loader's model file, the Multi-LoRA's and LoRA Block Weight's LoRAs with their strengths."""
    node = {"model": name}
    found, path = find_file("diffusion_models", name)
    if path is None:
        return {**node, "missing": True}
    if found != name:
        node["found_as"] = found
    if os.path.abspath(path) in chain or len(chain) >= _MAX_DEPTH:
        return {**node, "loop": True}
    try:
        meta = read_header(path)[0]
    except (OSError, ValueError):
        return {**node, "unreadable": True}
    text = meta.get("hades/recipe", "").strip()
    if not text:
        return node
    chain = chain + (os.path.abspath(path),)
    parts = []
    for m in info_metas(text):
        parts += model_recipe_parts(m, chain)
    return {**node, "text": text, "parts": parts}


def model_recipe_lines(node, depth=0):
    pad = "  " * depth
    if "operation" in node:
        lines = [f"{pad}{node['operation']}: {json.dumps(node['settings'], ensure_ascii=False)}"]
        for part in node["parts"]:
            lines += model_recipe_lines(part, depth + 1) if "lora" not in part else recipe_lines(part, depth + 1)
        return lines
    head = f"{pad}model {node['model']}" + (f" (now {node['found_as']})" if node.get("found_as") else "")
    for flag, note in (("missing", "not in models/diffusion_models"), ("loop", "loop or too deep, not followed"), ("unreadable", "unreadable")):
        if node.get(flag):
            return [f"{head}  ({note})"]
    if "text" not in node:
        return [f"{head}  (no recipe in the file)"]
    lines = [f"{head}  recipe as saved:"]
    lines += [f"{pad}  | {line}" for line in node["text"].splitlines() if line.strip() and not line.startswith(HADES_META_PREFIX)]
    for part in node["parts"]:
        lines += model_recipe_lines(part, depth + 1) if "lora" not in part else recipe_lines(part, depth + 1)
    return lines


def _shown(v):
    v = " ".join(str(v).split())
    return v if len(v) <= _SHOWN_CHARS else v[:_SHOWN_CHARS] + f"... ({len(v)} chars)"


def _size(n):
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.2f} MB"


def _layers_text(layers):
    if not layers:
        return "none"
    runs, start = [], layers[0]
    for a, b in zip(layers, layers[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return f"{len(layers)} ({', '.join(runs)})"


def _common_lines(d):
    s = d["tensors"]
    lines = [f"File: {d['file']} | {_size(d['size_bytes'])} | {s['tensors']} tensors | "
             + ", ".join(f"{k} {v}" for k, v in s["dtypes"].items())]
    if d.get("sha256"):
        lines.append(f"SHA256: {d['sha256']} (AutoV2 {d['sha256'][:10]})")
    return lines


def _metadata_lines(meta, shown_above=()):
    if not meta:
        return ["Metadata: none in the file"]
    head = f"Metadata ({len(meta)} keys" + (f"; {', '.join(shown_above)} above" if shown_above else "") + "):"
    return [head] + [f"  {k}: {_shown(v)}" for k, v in sorted(meta.items()) if k not in shown_above]


def lora_report(d):
    s = d["tensors"]
    lines = ["== LoRA =="] + _common_lines(d)
    kinds = ", ".join(f"{k} {v}" for k, v in s["kinds"].items()) or "no LoRA layer found"
    rank = "" if s["rank_max"] is None else (f" | rank {s['rank_max']}" if s["rank_min"] == s["rank_max"] else f" | rank {s['rank_min']}-{s['rank_max']} (per layer)")
    lines.append(f"Measured: {kinds}{rank}" + (f" | alpha {', '.join(map(str, s['alphas']))}" if s["alphas"] else ""))
    lines.append(f"Layers: {_layers_text(s['layers'])} | modules: {', '.join(s['layer_modules']) or 'none'}"
                 + (f" | outside the layers: {', '.join(s['other_modules'])}" if s["other_modules"] else ""))
    if not s["layers"] and not s["other_modules"]:
        lines.append("No MOSS-Image key name found (layers.N, text_embedder, ...): this LoRA is probably for another model.")
    r = d["recipe"]
    if r.get("op"):
        lines.append("Recipe (from the hades/ metadata of each file):")
        lines += recipe_lines(r, 1)
        if r.get("sources"):
            lines.append("Original files in the result: " + ", ".join(f"{w:+.4g} × {n}" for n, w in recipe_leaves(r).items()))
    return lines + _metadata_lines(d["metadata"])


def model_report(d):
    s = d["tensors"]
    lines = ["== Model =="] + _common_lines(d)
    if s["moss_image"]:
        lines.append(f"Measured: MOSS-Image, {len(s['layers'])} layers, {s['elements'] / 1e9:.2f} B parameters | "
                     f"q/k/v {'fused (proj.qkv)' if s['fused_qkv'] else 'split'} | {'int8 ConvRot' if s['int8_convrot'] else 'not quantized'} | "
                     f"text embedder blocks {s['text_embedder_blocks']}")
    else:
        lines.append(f"Measured: no shared_block_modulation key, so not a MOSS-Image diffusion model ({s['elements'] / 1e9:.2f} B elements)")
    r = d["recipe"]
    if "text" not in r:
        lines.append("Recipe: none in the file (the MOSS Model Saver writes one from its recipe input)")
        return lines + _metadata_lines(d["metadata"])
    lines.append("Recipe (hades/recipe, with the models and LoRAs it names followed into their files):")
    lines += model_recipe_lines(r, 1)[1:]
    return lines + _metadata_lines(d["metadata"], ("hades/recipe",))


def describe(folder, name, sha256):
    path = folder_paths.get_full_path_or_raise(folder, name)
    meta, tensors, start = read_header(path)
    d = {"file": name, "path": path, "size_bytes": os.path.getsize(path), "metadata": meta}
    if sha256:
        d["sha256"] = file_sha256(path)
    if folder == "loras":
        d["tensors"] = lora_summary(path, tensors, start)
        d["recipe"] = recipe(name)
    else:
        d["tensors"] = model_summary(tensors)
        d["recipe"] = model_recipe(name)
    return d


class HaDeSMossMetadataReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": ([NONE] + folder_paths.get_filename_list("loras"), {"tooltip": "LoRA in models/loras, or (none)."}),
                "model_name": ([NONE] + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "Diffusion model in models/diffusion_models, or (none)."}),
            },
            "optional": {
                "sha256": ("BOOLEAN", {"default": False, "tooltip": "Also hash the files (reads them whole: seconds for a 10 GB model; "
                                                                    "cached while the file is unchanged, shared with the MOSS Image Saver)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("info", "metadata_json")
    FUNCTION = "read"
    OUTPUT_NODE = True
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = ("Shows what a LoRA or model file says about itself (its safetensors metadata) and what its tensors show (size, dtypes, layers, "
                   "LoRA rank, MOSS-Image layout). Reads the header only, not the weights. For LoRAs made with the Suite's Merger, Resize and "
                   "Extractor, and models saved with a recipe by the MOSS Model Saver, it follows the recipe into the source files that are "
                   "still in models/.")

    @classmethod
    def IS_CHANGED(cls, lora_name, model_name, sha256=False):
        out = []
        seen = set()

        def visit(node):
            if "lora" in node or "model" in node:
                folder = "loras" if "lora" in node else "diffusion_models"
                name = node.get("found_as", node.get("lora", node.get("model")))
                _, path = find_file(folder, name)
                key = (folder, name, path)
                if key in seen:
                    return
                seen.add(key)
                st = os.stat(path) if path and os.path.exists(path) else None
                out.append((*key, st.st_size, st.st_mtime_ns) if st else key)
            for child in node.get("sources", []) + node.get("parts", []):
                visit(child)

        for folder, name in (("loras", lora_name), ("diffusion_models", model_name)):
            if name == NONE:
                continue
            visit(recipe(name) if folder == "loras" else model_recipe(name))
        return json.dumps([out, sha256])

    def read(self, lora_name, model_name, sha256=False):
        if lora_name == NONE and model_name == NONE:
            raise ValueError("HaDeS MOSS Metadata Reader: pick a LoRA, a model or both.")
        data, lines = {}, ["HaDeS MOSS Metadata Reader"]
        if lora_name != NONE:
            data["lora"] = describe("loras", lora_name, sha256)
            lines += lora_report(data["lora"])
        if model_name != NONE:
            data["model"] = describe("diffusion_models", model_name, sha256)
            lines += model_report(data["model"])
        lt = data.get("lora", {}).get("tensors", {})
        info = format_info("\n".join(lines), "metadata_reader", lora=lora_name if lora_name != NONE else None,
                           model=model_name if model_name != NONE else None, lora_rank=lt.get("rank_max"),
                           recipe=recipe_leaves(data["lora"]["recipe"]) if data.get("lora", {}).get("recipe", {}).get("op") else None,
                           model_recipe="text" in data.get("model", {}).get("recipe", {}))
        out = json.dumps(data, indent=1, ensure_ascii=False)
        return {"ui": {"text": [info]}, "result": (info, out)}


NODE_CLASS_MAPPINGS = {"HaDeSMossMetadataReader": HaDeSMossMetadataReader}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossMetadataReader": "MOSS Metadata Reader"}
