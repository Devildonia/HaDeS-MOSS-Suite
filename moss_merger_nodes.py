from .moss_common import PARTS, apply_block_merge, format_info

N_GROUPS = 7  # groups of 6 layers on the 42-layer model; other depths are split proportionally
RATIO = {"min": -1.0, "max": 2.0, "step": 0.01}


def group_name(g):
    lo, hi = g * 6, g * 6 + 5
    return f"layers_{lo:02d}_{hi:02d}"


GROUPS = [group_name(g) for g in range(N_GROUPS)]

# Share of the SFT -> Base change each group carries alone, min-max over layout / edges / texture / color (tools/RESULTS.md)
MEASURED = {
    "layers_00_05": "0.17-0.40", "layers_06_11": "0.10-0.28", "layers_12_17": "0.11-0.31", "layers_18_23": "0.20-0.48",
    "layers_24_29": "0.44-0.76 (strongest)", "layers_30_35": "0.37-0.65", "layers_36_41": "0.17-0.39",
    "text_embedder": "0.12-0.32", "time_modulation": "0.17-0.31", "io": "0.07-0.20 (weakest)",
}


def tooltip(name, what):
    return f"{what} 0.0 = Model 1, 1.0 = Model 2. Alone it carries {MEASURED[name]} of the SFT -> Base change (tools/RESULTS.md)."


def layer_group(i, n_layers):
    return i * N_GROUPS // n_layers


def parse_recipe(text, n_layers):
    """CSV of one ratio per layer followed by text_embedder, time_modulation, io."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != n_layers + len(PARTS):
        raise ValueError(f"HaDeS MOSS Merger: recipe_override needs {n_layers + len(PARTS)} comma-separated values "
                         f"({n_layers} layers, then {', '.join(PARTS)}), got {len(parts)}.")
    values = [float(p) for p in parts]
    return values[:n_layers], dict(zip(PARTS, values[n_layers:]))


class HaDeSMossModelBlockMerger:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"model1": ("MODEL",), "model2": ("MODEL",)}
        for g in GROUPS:
            required[g] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip(g, f"Ratio for {g.replace('_', ' ', 1).replace('_', '-')}.")})
        required["text_embedder"] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip("text_embedder", "Projection of the Qwen3.5 hidden states and its 2 text blocks.")})
        required["time_modulation"] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip("time_modulation", "time_embedding and shared_block_modulation (the one modulation shared by every layer).")})
        required["io"] = ("FLOAT", {"default": 0.5, **RATIO, "tooltip": tooltip("io", "video_embedder (latent in) and the output head (final norm, linear and modulation).")})
        return {
            "required": required,
            "optional": {
                "recipe_override": ("STRING", {"default": "", "tooltip": "Optional per-layer recipe: one ratio per layer (42), then text_embedder, time_modulation, io. Replaces the sliders when set."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "merge"
    CATEGORY = "HaDeS/MOSS Mergers"
    DESCRIPTION = "Merges two MOSS-Image models per layer group, lazily: no merged copy in RAM, the merge is computed when the model loads."

    def merge(self, model1, model2, text_embedder, time_modulation, io, recipe_override="", **groups):
        n_layers = len(model1.model.diffusion_model.layers)
        if recipe_override.strip():
            layer_ratios, part_ratios = parse_recipe(recipe_override, n_layers)
            mode = "recipe"
        else:
            layer_ratios = [groups[GROUPS[layer_group(i, n_layers)]] for i in range(n_layers)]
            part_ratios = {"text_embedder": text_embedder, "time_modulation": time_modulation, "io": io}
            mode = "groups"
        m, merged, n_layers = apply_block_merge(model1, model2, layer_ratios, part_ratios)
        return (m, format_info(f"HaDeS MOSS Model Merged ({mode}) | {n_layers} layers | modified tensors: {merged}",
                               "model_block_merger", mode=mode, n_layers=n_layers, modified_tensors=merged))


NODE_CLASS_MAPPINGS = {"HaDeSMossModelBlockMerger": HaDeSMossModelBlockMerger}
NODE_DISPLAY_NAME_MAPPINGS = {"HaDeSMossModelBlockMerger": "MOSS Model Block Merger"}
