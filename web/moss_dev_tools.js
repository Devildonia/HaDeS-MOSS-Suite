import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

export function addMeasurementReport(node) {
    const report = ComfyWidgets.STRING(node, "measurement_report", ["STRING", { multiline: true }], app).widget;
    // This native output widget is content, not a HaDeS replacement for a standard input.
    report._isHadesCustom = false;
    if (node._hadesCustomWidgets) node._hadesCustomWidgets = node._hadesCustomWidgets.filter(w => w !== report);
    report.inputEl.readOnly = true;
    Object.assign(report.inputEl.style, { minHeight: "0", boxSizing: "border-box", whiteSpace: "pre-wrap", overflowWrap: "anywhere", overflow: "auto", resize: "none" });
    // Tell the node layout about the space; a CSS min-height alone overflows at small node sizes / zoom levels.
    report.options.getMinHeight = () => 180;
    report.options.read_only = true; // Vue renders its own textarea rather than the legacy inputEl.
    report.options.serialize = false;
    report.serialize = false;
    report.serializeValue = () => undefined;
    const previous = node.onExecuted;
    node.onExecuted = function (message) {
        previous?.apply(this, arguments);
        report.value = (message?.text ?? []).join("\n");
        this.setDirtyCanvas(true, true);
    };
    return report;
}
