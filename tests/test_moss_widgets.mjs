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
