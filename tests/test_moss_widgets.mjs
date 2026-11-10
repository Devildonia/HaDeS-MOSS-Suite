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
