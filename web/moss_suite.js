import { app } from "../../scripts/app.js";
import {
    HADES_THEME,
    applyNodeFrameStyle,
    createSpectralBannerWidget,
    createSpectralSliderWidget,
    createStyledComboWidget,
    createStyledTextRowWidget,
    createStyledToggleWidget,
    finalizeHadesNode,
    hideBackingWidget,
    setGlobalClassicUI,
    setWidgetValue,
} from "./moss_widgets.js";

// =============================================================================
// HADES MOSS SUITE UI
// =============================================================================
// Every widget of a HaDeS MOSS node gets a styled row from its type (combo, float slider, int / text row, toggle);
// multi-line text keeps ComfyUI's editor. Tooltips come from the Python INPUT_TYPES, where the measurements live.

// Layer groups keep the depth hues of the HaDeS Anima Suite chart; the parts outside the layers are neutral.
const GROUP_COLORS = {
    layers_00_05: "#6d8fd6", layers_06_11: "#5fa3b5", layers_12_17: "#8a84cf", layers_18_23: "#62a88a",
    layers_24_29: "#a786c4", layers_30_35: "#c49a5a", layers_36_41: "#c07a98",
    text_embedder: "#8b93a1", time_modulation: "#7d8a7a", io: "#9a8f80", other_layers: "#8b93a1",
};
const GROUPS = Object.keys(GROUP_COLORS).filter((k) => k.startsWith("layers_"));
const PARTS = ["text_embedder", "time_modulation", "io"];

const SHORT_TAGS = {
    unet_name: "MODEL", lora_name: "LORA", lora_a: "LORA A", lora_b: "LORA B", sampler_name: "SAMPLER", strength_model: "STRENGTH",
    filename_prefix: "FILE", export_filename: "FILE", flow_shift: "SHIFT", aspect_ratio: "ASPECT", megapixels: "MPIX",
    custom_width: "WIDTH", custom_height: "HEIGHT", batch_size: "BATCH", recipe_override: "RECIPE", include_metadata: "METADATA",
    overwrite_existing: "OVERWRITE", dynamic_energy: "ENERGY", target_rank: "RANK", custom_author: "AUTHOR", custom_model_title: "TITLE",
    report_title: "TITLE", time_modulation: "TIME MOD", text_embedder: "TEXT EMB", other_layers: "OTHER",
};

// Text in a slider row next to its tag; the share each merge group carries alone of the SFT -> Base change (tools/RESULTS.md)
const SLIDER_LABELS = {
    layers_00_05: "carries 0.17–0.40", layers_06_11: "carries 0.10–0.28", layers_12_17: "carries 0.11–0.31", layers_18_23: "carries 0.20–0.48",
    layers_24_29: "carries 0.44–0.76", layers_30_35: "carries 0.37–0.65", layers_36_41: "carries 0.17–0.39",
    text_embedder: "carries 0.12–0.32", time_modulation: "carries 0.17–0.31", io: "carries 0.07–0.20",
    flow_shift: "0 = model (5.0)",
};

function tagFor(name) {
    if (SHORT_TAGS[name]) return SHORT_TAGS[name];
    const m = name.match(/^layers_(\d+)_(\d+)$/);
    if (m) return `L${m[1]}–${m[2]}`;
    return name.replace(/_/g, " ").toUpperCase();
}

// Seed + control_after_generate as HaDeS rows; the native widgets stay (hidden) so ComfyUI's control logic and serialization work unchanged.
function addSeedRows(node, wSeed, wControl) {
    hideBackingWidget(wSeed);
    hideBackingWidget(wControl);
    const seedLinked = () => node.getSlotFromWidget(wSeed)?.link != null;
    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        origOnConnectionsChange?.apply(this, arguments);
        this.setDirtyCanvas(true, true);
    };
    node.addCustomWidget(createStyledTextRowWidget(node, wSeed, {
        tag: "SEED", numeric: true, promptTitle: "Seed", isDisabled: seedLinked, disabledText: "From the linked seed input",
    }));
    if (wControl) {
        node.addCustomWidget(createStyledComboWidget(node, wControl, {
            tag: "AFTER RUN", isDisabled: seedLinked,
            tooltip: "<strong>Control after generate</strong><br>fixed, increment, decrement or randomize the seed after each queued run. Inactive while the seed input is linked.",
        }));
    }
}

// Styled rows for every widget, in the node's own order. `sliders` overrides the visual range of FLOAT widgets.
function styleWidgets(node, { sliders = {}, colors = {}, dimmed = null } = {}) {
    for (const w of [...(node.widgets || [])]) {
        if (w._isHadesBacking || w._isHadesCustom || w.name === "control_after_generate") continue;
        const color = colors[w.name] || GROUP_COLORS[w.name];
        const isDimmed = dimmed ? () => dimmed(w.name) : null;
        const opts = w.options || {};
        if (w.name === "seed" || w.name === "noise_seed") {
            addSeedRows(node, w, node.widgets.find((x) => x.name === "control_after_generate"));
        } else if (w.type === "combo") {
            hideBackingWidget(w);
            node.addCustomWidget(createStyledComboWidget(node, w, { tag: tagFor(w.name), cleanFileNames: /_name$|^lora_[ab]$/.test(w.name), marker: color, isDimmed }));
        } else if (w.type === "toggle") {
            hideBackingWidget(w);
            node.addCustomWidget(createStyledToggleWidget(node, w, { tag: tagFor(w.name), label: w.name.replace(/_/g, " "), onText: "ON", offText: "OFF" }));
        } else if (w.type === "number" && opts.precision !== 0) {  // FLOAT; the frontend gives INT widgets precision 0
            hideBackingWidget(w);
            const range = sliders[w.name] || [opts.min ?? 0, opts.max ?? 1];
            node.addCustomWidget(createSpectralSliderWidget(node, w, {
                tag: tagFor(w.name), label: (node.type === "HaDeSMossModelBlockMerger" || !(w.name in GROUP_COLORS)) ? (SLIDER_LABELS[w.name] ?? "") : "",
                color: color || HADES_THEME.accent, marker: color || null,
                min: opts.min ?? -1e9, max: opts.max ?? 1e9, visMin: range[0], visMax: range[1], step: opts.step2 ?? 0.01, defaultVal: w.value, isDimmed,
            }));
        } else if (w.type === "number" || ((w.type === "text" || w.type === "string") && !opts.multiline)) {
            hideBackingWidget(w);
            const numeric = w.type === "number";
            node.addCustomWidget(createStyledTextRowWidget(node, w, {
                tag: tagFor(w.name), numeric, promptTitle: w.name.replace(/_/g, " "), color: HADES_THEME.accent, emptyPlaceholder: "— empty —", isDimmed,
            }));
        }
    }
}

function banner(node, config) {
    node.addCustomWidget(createSpectralBannerWidget(node, { accentColor: HADES_THEME.accent, ...config }));
}

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const num = (node, name, fallback = 0) => Number(widget(node, name)?.value ?? fallback);

// Bars for the 42 layers (proportional groups) and the three parts, or the per-layer recipe when one is set.
function blockMergerBars(node) {
    const recipe = String(widget(node, "recipe_override")?.value || "").split(",").map((s) => s.trim()).filter(Boolean).map(Number);
    if (recipe.length === 45 && recipe.every((v) => !Number.isNaN(v))) {
        return recipe.map((r, i) => ({ ratio: r, color: i < 42 ? GROUP_COLORS[GROUPS[Math.floor((i * 7) / 42)]] : GROUP_COLORS[PARTS[i - 42]] }));
    }
    const bars = [];
    for (let i = 0; i < 42; i++) {
        const g = GROUPS[Math.floor((i * 7) / 42)];
        bars.push({ ratio: num(node, g, 0.5), color: GROUP_COLORS[g] });
    }
    for (const p of PARTS) bars.push({ ratio: num(node, p, 0.5), color: GROUP_COLORS[p] });
    return bars;
}

function setRatios(node, values) {
    const recipe = widget(node, "recipe_override");
    if (recipe) setWidgetValue(recipe, "", node);
    for (const [name, v] of Object.entries(values)) setWidgetValue(widget(node, name), v, node);
    node.setDirtyCanvas(true, true);
}

const all = (v) => Object.fromEntries([...GROUPS, ...PARTS].map((k) => [k, v]));

const INSTALLERS = {
    HaDeSMossModelBlockMerger(node) {
        banner(node, {
            title: "LAYERS 0–41 · TEXT · TIME · IO",
            getBars: () => blockMergerBars(node),
            actions: [
                { label: "M1", tooltip: "<strong>Model 1</strong><br>Every group and part at 0.0.", onClick: () => setRatios(node, all(0.0)) },
                { label: "50/50", tooltip: "<strong>Even blend</strong><br>Every group and part at 0.5.", onClick: () => setRatios(node, all(0.5)) },
                {
                    label: "24–35 M2",
                    tooltip: "<strong>Layers 24–35 from Model 2</strong><br>The groups that carried most of the SFT → Base change (tools/RESULTS.md); the rest stays Model 1.",
                    onClick: () => setRatios(node, { ...all(0.0), layers_24_29: 1.0, layers_30_35: 1.0 }),
                },
                { label: "M2", tooltip: "<strong>Model 2</strong><br>Every group and part at 1.0.", onClick: () => setRatios(node, all(1.0)) },
            ],
        });
        styleWidgets(node, { sliders: all([0, 1]) });
    },
    HaDeSMossLoraBlockWeight(node) {
        banner(node, {
            title: "LORA STRENGTH PER LAYER GROUP",
            getBars: () => {
                const strength = num(node, "strength_model", 1);
                const vals = [...GROUPS, "other_layers"].map((g) => num(node, g, 1) * strength);
                const top = Math.max(1, ...vals.map(Math.abs));
                return vals.map((v, i) => ({ ratio: Math.abs(v) / top, color: GROUP_COLORS[[...GROUPS, "other_layers"][i]] }));
            },
        });
        styleWidgets(node, { sliders: { ...Object.fromEntries([...GROUPS, "other_layers"].map((g) => [g, [0, 2]])), strength_model: [0, 2] } });
    },
    HaDeSMossKSampler(node) {
        // a connected turbo input overrides these in the backend: draw them faded (still editable for when it is unplugged)
        const TURBO_OVERRIDES = new Set(["steps", "cfg", "sampler_name", "scheduler", "slg"]);
        const turboLinked = () => node.inputs?.find((i) => i.name === "turbo")?.link != null;
        // the steps widget of the MOSS Turbo LoRA node feeding the turbo input (4, its default, if it cannot be read)
        const turboSteps = () => {
            const slot = node.inputs?.findIndex((i) => i.name === "turbo") ?? -1;
            const src = slot >= 0 ? node.getInputNode?.(slot) : null;
            const steps = src && widget(src, "steps")?.value;
            return Number.isFinite(steps) ? steps : 4;
        };
        banner(node, {
            title: "MOSS KSAMPLER",
            getStatusText: () => {
                const shift = ` · shift ${num(node, "flow_shift") > 0 ? num(node, "flow_shift") : "model"}`;
                // a connected turbo input overrides steps / CFG / sampler / scheduler / SLG in the backend
                if (turboLinked()) return `TURBO · euler / simple · ${turboSteps()} steps · CFG 1.0${shift}`;
                return `${widget(node, "sampler_name")?.value} / ${widget(node, "scheduler")?.value} · ${num(node, "steps")} steps · CFG ${num(node, "cfg").toFixed(1)}`
                    + `${shift}${widget(node, "slg")?.value !== "off" ? " · SLG" : ""}`;
            },
        });
        styleWidgets(node, { sliders: { cfg: [1, 8], denoise: [0, 1], flow_shift: [0, 12] }, dimmed: (name) => TURBO_OVERRIDES.has(name) && turboLinked() });
    },
    HaDeSMossLatent(node) {
        banner(node, {
            title: "MOSS LATENT",
            getStatusText: () => {
                const aspect = widget(node, "aspect_ratio")?.value;
                if (aspect === "custom") return `${num(node, "custom_width")} × ${num(node, "custom_height")} · custom`;
                const [aw, ah] = String(aspect).split(":").map(Number);
                const unit = Math.sqrt((num(node, "megapixels", 1) * 1024 * 1024) / (aw * ah));
                const snap = (v) => Math.max(16, Math.round(v / 16) * 16);
                return `${snap(aw * unit)} × ${snap(ah * unit)} · ${aspect} · 128-ch latent`;
            },
        });
        styleWidgets(node, { sliders: { megapixels: [0.25, 4] } });
    },
};

// Nodes without a specific installer: title banner and styled rows.
const GENERIC = {
    HaDeSMossMultiModelMerger: { title: "MULTI-MODEL MERGE", sliders: { density: [0, 1], scale: [0, 2], weight_1: [0, 2], weight_2: [0, 2], weight_3: [0, 2], weight_4: [0, 2] } },
    HaDeSMossModelSaver: { title: "MODEL SAVER" },
    HaDeSMossLoraExtractor: { title: "LORA EXTRACTOR" },
    HaDeSMossLoraMerger: { title: "LORA MERGER", sliders: { weight_a: [0, 2], weight_b: [0, 2], dynamic_energy: [0, 1] } },
    HaDeSMossLoraResize: { title: "LORA RESIZE", sliders: { dynamic_energy: [0, 1] } },
    HaDeSMossLayoutCompiler: { title: "LAYOUT COMPILER" },
    HaDeSMossSamplerMetadata: { title: "SAMPLER METADATA" },
    HaDeSMossTurbo: { title: "TURBO LORA · CFG 1", sliders: { strength: [0, 2] } },
    HaDeSMossSuiteTelemetry: { title: "SUITE TELEMETRY" },
};
for (const [cls, { title, sliders = {} }] of Object.entries(GENERIC)) {
    INSTALLERS[cls] = (node) => {
        banner(node, { title });
        styleWidgets(node, { sliders });
    };
}

app.registerExtension({
    name: "HaDeS.MOSSSuiteUI",
    settings: [
        {
            id: "HaDeS.MOSS.ClassicUI",
            category: ["HaDeS MOSS Suite", "Appearance", "Classic widgets"],
            name: "Use standard ComfyUI widgets on HaDeS MOSS nodes",
            type: "boolean",
            defaultValue: false,
            tooltip: "Hides the HaDeS canvas widgets and shows the plain ComfyUI inputs on every HaDeS MOSS node.",
            onChange(value) {
                setGlobalClassicUI(Boolean(value));
            },
        },
    ],
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const install = INSTALLERS[nodeData.name];
        if (!install) return;
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);
            this.serialize_widgets = true;
            applyNodeFrameStyle(this);
            install(this);
            finalizeHadesNode(this, 380);
            // room for the draft editor; a saved size wins when a workflow is loaded
            if (nodeData.name === "HaDeSMossLayoutCompiler") this.setSize([460, Math.max(this.size[1], 520)]);
        };
    },
});
