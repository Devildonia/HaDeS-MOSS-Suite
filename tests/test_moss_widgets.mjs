import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

function setup() {
    const app = { graph: { _nodes: [] } };
    const context = vm.createContext({ app, window: { addEventListener() {} }, Image: class {}, setTimeout });
    const source = readFileSync(new URL("../web/moss_widgets.js", import.meta.url), "utf8")
        .replace('import { app } from "../../scripts/app.js";', "")
        .replace(/^export /gm, "")
        .replace('new URL("./icon.png", import.meta.url).href', '"icon.png"');
    vm.runInContext(source, context);
    const node = (type, flags = {}) => ({ type, flags, size: [400, 200],
        computeSize: () => [400, 200], setSize() {}, setDirtyCanvas() {} });
    return { app, context, node };
}

test("global classic override preserves individual preferences", () => {
    const { app, context, node } = setup();
    const classic = node("HaDeSMossLatent", { classic_ui: true });
    const styled = node("HaDeSMossKSampler");
    app.graph._nodes = [classic, styled];
    context.setGlobalClassicUI(true);
    context.setGlobalClassicUI(false);
    assert.equal(classic.flags.classic_ui, true);
    assert.equal(context.isNodeClassicUI(classic), true);
    assert.equal(context.isNodeClassicUI(styled), false);
});

test("global switch leaves other HaDeS suites untouched", () => {
    const { app, context, node } = setup();
    const other = node("HaDeSAnimaLoader", { classic_ui: true });
    other._hadesCustomWidgets = [];
    other._hasHadesFrameStyle = true;
    other.setSize = () => assert.fail("MOSS resized another suite's node");
    app.graph._nodes = [other];
    context.setGlobalClassicUI(true);
    context.setGlobalClassicUI(false);
    assert.equal(other.flags.classic_ui, true);
});

test("dev reports reserve layout height, wrap text and do not serialize measured output", () => {
    const report = { inputEl: { style: {} }, options: {} };
    let oldCalls = 0, repaints = 0;
    const node = { onExecuted() { oldCalls++; }, setDirtyCanvas() { repaints++; } };
    const context = vm.createContext({ app: {}, ComfyWidgets: { STRING: () => ({ widget: report }) } });
    const source = readFileSync(new URL("../web/moss_dev_tools.js", import.meta.url), "utf8")
        .replace(/^import .*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context);
    context.addMeasurementReport(node);
    assert.equal(report.options.getMinHeight(), 180);
    assert.equal(report.inputEl.style.minHeight, "0");
    assert.equal(report.inputEl.style.overflowWrap, "anywhere");
    assert.equal(report.inputEl.style.overflow, "auto");
    assert.equal(report.inputEl.readOnly, true);
    assert.equal(report.options.read_only, true);
    assert.equal(report.options.serialize, false);
    assert.equal(report.serializeValue(), undefined);
    node.onExecuted({ text: ["first", "second"] });
    assert.equal(report.value, "first\nsecond");
    node.onExecuted({});
    assert.equal(report.value, "");
    assert.equal(oldCalls, 2);
    assert.equal(repaints, 2);
});

test("measurement reports stay visible through standard / HaDeS widget toggles", () => {
    const { context, node } = setup();
    const report = { name: "measurement_report", inputEl: { style: {} }, options: {}, _isHadesCustom: true,
        hidden: false, computeSize: () => [380, 180] };
    const banner = { computeSize: () => [380, 28], draw() {} };
    const target = node("HaDeSMossImageAudit");
    target._hadesCustomWidgets = [banner, report];
    target.widgets = [report];
    context.ComfyWidgets = { STRING: () => ({ widget: report }) };
    const source = readFileSync(new URL("../web/moss_dev_tools.js", import.meta.url), "utf8")
        .replace(/^import .*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context);
    context.addMeasurementReport(target);
    context.setNodeClassicUI(target, true, true);
    assert.equal(banner.hidden, true);
    assert.equal(report.hidden, false);
    assert.equal(report.computeSize()[1], 180);
    context.setNodeClassicUI(target, false, true);
    assert.equal(banner.hidden, false);
    assert.equal(report.hidden, false);
    assert.equal(report.options.read_only, true);
});

test("style switch keeps native output previews visible", () => {
    const { context, node } = setup();
    const target = node("HaDeSMossImageSaver");
    target.widgets = [];
    target.addCustomWidget = function (widget) { this.widgets.push(widget); return widget; };
    context.applyNodeFrameStyle(target);
    const banner = { name: "hades_banner_test", computeSize: () => [380, 28], draw() {} };
    const preview = { name: "image", computeSize: () => [380, 180], draw() {} };
    target.addCustomWidget(banner);
    target.addCustomWidget(preview);
    context.setNodeClassicUI(target, true, true);
    assert.equal(banner.hidden, true);
    assert.equal(preview.hidden, undefined);
    assert.equal(preview.computeSize()[1], 180);
    assert.equal(preview._isHadesCustom, undefined);
    context.setNodeClassicUI(target, false, true);
    assert.equal(preview.hidden, undefined);
});

test("layout editor picks the smallest box under the pointer, so a full-canvas box does not hide the others", () => {
    const source = readFileSync(new URL("../web/moss_layout_editor.js", import.meta.url), "utf8");
    const fn = source.match(/export function pickBox[\s\S]*?\n}\r?\n/)[0].replace("export ", "");
    const pickBox = vm.runInNewContext(`(${fn.replace("function pickBox", "function")})`);
    const background = [0, 0, 1024, 1024], moon = [700, 100, 900, 300];
    assert.equal(pickBox([moon, background], 800, 200), 0);  // the background drawn on top of the moon
    assert.equal(pickBox([background, moon], 800, 200), 1);
    assert.equal(pickBox([moon, background], 50, 900), 1);   // where only the background is
    assert.equal(pickBox([moon, background], 2000, 2000), -1);
    assert.equal(pickBox([[0, 0, 10, 10], [0, 0, 10, 10]], 5, 5), 1);  // equal sizes: the topmost
});

test("a value picked in a HaDeS widget reaches node.onWidgetChanged, so the frontend clears a missing model's red frame", () => {
    const { context, node } = setup();
    const n = node("HaDeSMossLoader");
    const calls = [];
    n.onWidgetChanged = (name, value, old, widget) => calls.push([name, value, old, widget]);
    const w = { name: "unet_name", value: "moss_image_base_bf16.safetensors", callback() {} };
    context.setWidgetValue(w, "mossImage_base_bf16.safetensors", n);
    assert.equal(w.value, "mossImage_base_bf16.safetensors");
    assert.deepEqual(calls, [["unet_name", "mossImage_base_bf16.safetensors", "moss_image_base_bf16.safetensors", w]]);
});

test("Nodes 2.0: a styled row carries its backing input's slot key, so a dropped link connects (YouTube report 2026-11-22)", () => {
    const { context } = setup();
    const made = [];
    context.document = { createElement: () => { const el = { dataset: {}, style: {}, remove() { row.children.splice(row.children.indexOf(el), 1); } }; made.push(el); return el; } };
    const row = { children: [], appendChild(el) { this.children.push(el); }, querySelector() { return this.children.find((c) => c.dataset.hadesDrop) ?? null; } };
    const canvasEl = { closest: (sel) => (sel === ".lg-node-widget" ? row : null) };
    const node = { id: 7, inputs: [{ name: "clip" }, { name: "idea", widget: { name: "idea" } }, { name: "width", widget: { name: "width" } }] };
    context.markVueDropTarget(canvasEl, node, { name: "width" });
    assert.equal(row.children.length, 1);
    assert.equal(row.children[0].dataset.slotKey, "7-in-2");
    assert.equal(row.children[0].style.display, "none");
    context.markVueDropTarget(canvasEl, node, { name: "width" }); // redraws reuse the marker
    assert.equal(made.length, 1);
    context.markVueDropTarget(canvasEl, node, { name: "gone" }); // no input behind the row: no marker
    assert.equal(row.children.length, 0);
});

test("a size saved in the standard view survives a reload; a taller HaDeS size does not grow a standard node", () => {
    const { context } = setup();
    const node = (size, loaded) => ({ size: [...size], _hadesLoadedSize: loaded, setSize(s) { this.size = [...s]; } });
    const big = node([380, 216], [500, 400]);
    context.keepLoadedSize(big);
    assert.deepEqual(big.size, [500, 400]);
    const small = node([380, 216], [300, 100]);  // smaller than the widgets need: the minimum wins
    context.keepLoadedSize(small);
    assert.deepEqual(small.size, [380, 216]);
    const hadesSaved = node([380, 216], null);  // saved in the HaDeS style: no size to keep
    context.keepLoadedSize(hadesSaved);
    assert.deepEqual(hadesSaved.size, [380, 216]);
});

function layoutQueue() {
    const source = readFileSync(new URL("../web/moss_layout_editor.js", import.meta.url), "utf8").replace(/\r\n/g, "\n");
    const start = source.indexOf("export const draftWaiters"), end = source.indexOf("export function attachDraftRun");
    const timers = [];
    const ctx = vm.createContext({
        setTimeout: (fn, ms) => { timers.push({ fn, ms, live: true }); return timers.length; },
        clearTimeout: (t) => { if (t && timers[t - 1]) timers[t - 1].live = false; },
    });
    vm.runInContext("let hookedApi = null;\n" + source.slice(start, end).replace(/^export /gm, "").replace("let hookedApi = null;\n", "")
        + "\nthis.draftWaiters = draftWaiters; this.hookDraftQueue = hookDraftQueue; this.waitForDraft = waitForDraft;", ctx);
    let n = 0;
    const api = { queuePrompt: async (number, data, options) => ({ prompt_id: `p${++n}` }) };
    return { ctx, api, timers };
}

test("two DRAFT LAYOUT editors share one queue hook and each gets its own request (audit 0.9.9 M7)", async () => {
    const { ctx, api } = layoutQueue();
    const orig = api.queuePrompt;
    ctx.hookDraftQueue(api);
    const hooked = api.queuePrompt;
    assert.notEqual(hooked, orig);
    ctx.hookDraftQueue(api);  // a second editor: no second wrapper around the first
    assert.equal(api.queuePrompt, hooked);
    const got = [];
    ctx.waitForDraft("3", (id) => got.push(["3", id]));
    ctx.waitForDraft("7", (id) => got.push(["7", id]));
    await api.queuePrompt(0, {}, { partialExecutionTargets: ["7"] });
    await api.queuePrompt(0, {}, { partialExecutionTargets: ["12:3"] });  // a node inside a subgraph
    await api.queuePrompt(0, {}, {});  // an ordinary run: no editor waits for it
    assert.deepEqual(got, [["7", "p1"], ["3", "p2"]]);
    assert.equal(ctx.draftWaiters.size, 0);
});

test("a DRAFT LAYOUT request that never reaches the queue stops waiting after 15 s", async () => {
    const { ctx, api, timers } = layoutQueue();
    ctx.hookDraftQueue(api);
    const got = [];
    ctx.waitForDraft("3", (id) => got.push(id));
    assert.equal(timers.at(-1).ms, 15000);
    timers.at(-1).fn();
    assert.equal(ctx.draftWaiters.size, 0);
    await api.queuePrompt(0, {}, { partialExecutionTargets: ["3"] });
    assert.deepEqual(got, []);
    ctx.waitForDraft("3", (id) => got.push(id));  // asked again: the older timer does not drop the new wait
    timers[0].fn();
    assert.equal(ctx.draftWaiters.size, 1);
});

test("Multi-LoRA's few-step badge is read again after a failure and after a minute (audit 0.9.9 B2)", async () => {
    const source = readFileSync(new URL("../web/moss_multilora.js", import.meta.url), "utf8").replace(/\r\n/g, "\n");
    const code = source.slice(source.indexOf("const fewStepCache"), source.indexOf("// ---- searchable LoRA picker"));
    let now = 1000, answer = { ok: false }, calls = 0;
    const api = { fetchApi: async () => { calls++; return answer.ok ? { ok: true, json: async () => answer.meta } : { ok: false, status: 500 }; } };
    const ctx = vm.createContext({ api, Date: { now: () => now } });
    vm.runInContext(code.replace("export function", "function") + "\nthis.fewStepSteps = fewStepSteps;", ctx);
    const node = { setDirtyCanvas() {} };
    const settle = () => new Promise((r) => setImmediate(r));
    assert.equal(ctx.fewStepSteps("t.safetensors", node), null);
    await settle();
    assert.equal(calls, 1);
    assert.equal(ctx.fewStepSteps("t.safetensors", node), null);  // a failed read is kept for 10 s only
    assert.equal(calls, 1);
    now += 10001;
    answer = { ok: true, meta: { settings: "8 steps, CFG 1, euler / simple" } };
    ctx.fewStepSteps("t.safetensors", node);
    await settle();
    assert.equal(calls, 2);
    assert.equal(ctx.fewStepSteps("t.safetensors", node), 8);
    answer = { ok: true, meta: { settings: "a style LoRA" } };  // the file replaced under the same name
    now += 30000;
    assert.equal(ctx.fewStepSteps("t.safetensors", node), 8);  // still within the minute
    now += 30001;
    assert.equal(ctx.fewStepSteps("t.safetensors", node), 8);  // read again; the old answer meanwhile
    await settle();
    assert.equal(calls, 3);
    assert.equal(ctx.fewStepSteps("t.safetensors", node), null);
});
