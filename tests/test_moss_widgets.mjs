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
