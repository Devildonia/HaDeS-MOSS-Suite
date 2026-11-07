import { api } from "../../scripts/api.js";
import {
    HADES_FONT,
    HADES_THEME,
    drawRoundedRect,
    drawRowLabel,
    drawRowSurface,
    hideHadesTooltip,
    setWidgetValue,
    showHadesTooltip,
} from "./moss_widgets.js";

// =============================================================================
// MOSS LAYOUT EDITOR: the draft's boxes on a canvas of its size, edited with the mouse
// =============================================================================
// The draft lives in the node's hidden `draft` widget (the Layout Compiler's schema, top-left xyxy pixels). Every edit
// rewrites it; boxes snap to the 16 px latent grid.

export const LABEL_COLORS = { object: "#6d8fd6", character: "#c49a5a", text: "#a786c4" };
const LABELS = Object.keys(LABEL_COLORS);
const GRID = 16;
const MIN_BOX = 32;
const ROW_H = 28;
const MAX_CANVAS_H = 560;
const HANDLE = 7;

const snap = (v) => Math.round(v / GRID) * GRID;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const shown = (s) => String(s ?? "").replace(/\n/g, " ⏎ ");

function parseDraft(text) {
    try {
        const d = JSON.parse(text || "");
        if (d && typeof d === "object" && Array.isArray(d.items) && Number(d.width) > 0 && Number(d.height) > 0) return { draft: d };
        if (d && typeof d === "object" && typeof d.caption === "string") return { caption: d.caption };
        return { error: "not a layout draft (width, height, summary, items)" };
    } catch (e) {
        return { error: "the draft is not valid JSON" };
    }
}

function newDraft(caption = "") {
    return { width: 1024, height: 1024, bbox_coord_space: "abs", bbox_format: "xyxy", summary: { canvas: "single", caption }, items: [] };
}

function itemProblem(item) {
    if (!String(item.caption ?? "").trim()) return "needs a caption";
    if (item.label === "text" && !String(item.text ?? "").trim()) return "needs its text";
    return null;
}

// ---- small DOM text editor (multi-line, Ctrl+Enter to apply) ----
let editorEl = null;
function closeTextEditor() {
    editorEl?._cleanup?.();
    editorEl?.remove();
    editorEl = null;
}

export function openTextEditor(event, title, value, onApply, hint = "") {
    closeTextEditor();
    const T = HADES_THEME;
    const el = document.createElement("div");
    editorEl = el;
    Object.assign(el.style, {
        position: "fixed", zIndex: 10000, width: "420px", display: "flex", flexDirection: "column", gap: "6px", padding: "8px",
        background: T.bgNode, border: `1px solid ${T.borderActive}`, borderRadius: "6px", boxShadow: "0 8px 28px #000a",
        font: HADES_FONT.text, color: T.textPrimary,
    });
    el.style.left = `${Math.max(8, Math.min(event.clientX, window.innerWidth - 430))}px`;
    el.style.top = `${Math.max(8, Math.min(event.clientY, window.innerHeight - 240))}px`;
    const head = document.createElement("div");
    head.textContent = title.toUpperCase();
    Object.assign(head.style, { font: HADES_FONT.label, letterSpacing: "0.6px", color: T.textMuted });
    const area = document.createElement("textarea");
    area.value = value ?? "";
    area.rows = 5;
    Object.assign(area.style, {
        resize: "vertical", padding: "6px 8px", background: T.bgControl, color: T.textPrimary, border: `1px solid ${T.borderSubtle}`,
        borderRadius: "4px", outline: "none", font: HADES_FONT.text,
    });
    const foot = document.createElement("div");
    foot.textContent = hint || "Ctrl+Enter applies · Esc cancels";
    Object.assign(foot.style, { font: HADES_FONT.small, color: T.textMuted });
    el.append(head, area, foot);
    document.body.appendChild(el);
    const apply = () => { const v = area.value; closeTextEditor(); onApply(v); };
    area.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); apply(); }
        else if (e.key === "Escape") { e.preventDefault(); closeTextEditor(); }
    });
    // clicking outside applies, like leaving a text field
    const outside = (e) => { if (!el.contains(e.target)) apply(); };
    setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
    el._cleanup = () => document.removeEventListener("pointerdown", outside, true);
    area.focus();
    area.select();
}

// ---- background image: a Load Image node upstream at once, otherwise the image the last run saw ----
function upstreamImageUrl(node) {
    const slot = node.inputs?.findIndex((i) => i.name === "image") ?? -1;
    if (slot < 0 || node.inputs[slot].link == null) return null;
    const src = node.getInputNode?.(slot);
    if (!src) return null; // shown even when the Load Image is muted: the background is only a guide on the canvas
    const name = src?.widgets?.find((w) => w.name === "image")?.value;
    if (!name || typeof name !== "string") return null;
    const parts = name.replace(/\\/g, "/").split("/");
    const file = parts.pop();
    return api.apiURL(`/view?filename=${encodeURIComponent(file)}&type=input&subfolder=${encodeURIComponent(parts.join("/"))}`);
}

export function createLayoutEditorWidget(node, backing) {
    const state = { sel: -1, tool: "object", drag: null, hover: null, showBg: true };
    const images = new Map(); // url -> HTMLImageElement
    const imageFor = (url) => {
        if (!url) return null;
        if (!images.has(url)) {
            const img = new Image();
            img.onload = () => node.setDirtyCanvas(true, true);
            img.src = url;
            images.set(url, img);
        }
        const img = images.get(url);
        return img.complete && img.naturalWidth ? img : null;
    };
    const bgUrl = () => upstreamImageUrl(node) || node._hadesLayoutBgUrl || null;

    const read = () => parseDraft(backing.value);
    const write = (draft) => {
        setWidgetValue(backing, JSON.stringify(draft, null, 2), node);
        fit();
    };
    // the node is exactly as tall as its content: the canvas height follows the width and the draft's aspect ratio
    const fit = () => {
        const h = node.computeSize()[1];
        if (Math.abs(node.size[1] - h) > 1) node.setSize([node.size[0], h]);
        node.setDirtyCanvas(true, true);
    };
    const origOnResize = node.onResize;
    node.onResize = function (size) {
        origOnResize?.apply(this, arguments);
        size[1] = this.computeSize()[1];
    };

    // canvas rectangle for a widget width and draft size
    const frame = (widgetWidth, y, d) => {
        const maxW = widgetWidth - 24;
        let w = maxW;
        let h = (w * d.height) / d.width;
        if (h > MAX_CANVAS_H) { h = MAX_CANVAS_H; w = (h * d.width) / d.height; }
        return { x: 12 + (maxW - w) / 2, y, w, h, s: w / d.width };
    };

    const resizeCanvas = (d, W, H) => {
        W = Math.max(256, snap(W)); H = Math.max(256, snap(H));
        const kx = W / d.width, ky = H / d.height;
        for (const it of d.items) {
            if (!Array.isArray(it.bbox)) continue;
            const [x1, y1, x2, y2] = it.bbox;
            it.bbox = [clamp(snap(x1 * kx), 0, W - GRID), clamp(snap(y1 * ky), 0, H - GRID), 0, 0];
            it.bbox[2] = clamp(Math.max(snap(x2 * kx), it.bbox[0] + GRID), GRID, W);
            it.bbox[3] = clamp(Math.max(snap(y2 * ky), it.bbox[1] + GRID), GRID, H);
        }
        d.width = W; d.height = H;
    };

    const detailRows = (d) => (state.sel >= 0 && d.items[state.sel] ? 4 : 1);

    return {
        name: "hades_layout_editor",
        _hadesBackingWidget: backing,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _zones: [],

        computeSize(width) {
            // LiteGraph asks with the node's minimum width; the canvas is drawn at the node's real width, so size it from that
            width = Math.max(width || 0, node.size?.[0] || 0);
            const p = read();
            if (!p.draft) return [width, ROW_H * 2 + 8];
            return [width, ROW_H + frame(width, 0, p.draft).h + 8 + detailRows(p.draft) * ROW_H + 4];
        },

        draw(ctx, n, widgetWidth, y) {
            const p = read();
            const zones = [];
            const x = 12, w = widgetWidth - 24;
            ctx.save();
            ctx.textBaseline = "middle";

            if (!p.draft) {
                drawRowSurface(ctx, x, y + 2, w, 24);
                ctx.font = HADES_FONT.text;
                ctx.fillStyle = p.error ? HADES_THEME.danger : HADES_THEME.textSecondary;
                ctx.textAlign = "left";
                ctx.fillText(p.error ? `Draft: ${p.error}` : "Caption draft: no boxes yet", x + 10, y + 14.5);
                const by = y + ROW_H + 2;
                drawRoundedRect(ctx, x, by, w, 24, 4);
                ctx.setLineDash([3, 3]);
                ctx.strokeStyle = HADES_THEME.borderSubtle;
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.font = HADES_FONT.label;
                ctx.textAlign = "center";
                ctx.fillStyle = HADES_THEME.textSecondary;
                ctx.fillText(p.caption !== undefined ? "NEW LAYOUT FROM THIS CAPTION" : "START A NEW LAYOUT", x + w / 2, by + 12.5);
                zones.push({ kind: "new", x, y: by, w, h: 24 });
                this._zones = zones;
                ctx.restore();
                return;
            }
            const d = p.draft;
            if (state.sel >= d.items.length) state.sel = -1;

            // toolbar: label to draw with (also the legend), canvas size, import, background
            let tx = x;
            const ty = y + 2;
            ctx.font = HADES_FONT.label;
            for (const label of LABELS) {
                const tw = ctx.measureText(label.toUpperCase()).width + 22;
                const active = state.tool === label;
                drawRoundedRect(ctx, tx, ty + 3, tw, 18, 9);
                ctx.fillStyle = active ? `${LABEL_COLORS[label]}40` : HADES_THEME.bgControl;
                ctx.fill();
                ctx.strokeStyle = active ? LABEL_COLORS[label] : HADES_THEME.borderSubtle;
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(tx + 9, ty + 12, 3, 0, Math.PI * 2);
                ctx.fillStyle = LABEL_COLORS[label];
                ctx.fill();
                ctx.fillStyle = active ? HADES_THEME.textPrimary : HADES_THEME.textSecondary;
                ctx.textAlign = "left";
                ctx.fillText(label.toUpperCase(), tx + 15, ty + 12.5);
                zones.push({ kind: "tool", label, x: tx, y: ty, w: tw, h: 24 });
                tx += tw + 4;
            }
            const right = [];
            const bg = bgUrl() && imageFor(bgUrl());
            if (bg) right.push({ kind: "match", text: "MATCH IMAGE" }, { kind: "bg", text: state.showBg ? "BG ON" : "BG OFF" });
            if (node._hadesDraftIn) right.push({ kind: "import", text: "IMPORT" });
            right.push({ kind: "size", text: `${d.width} × ${d.height}` });
            let rx = x + w;
            for (const b of right.reverse()) {
                ctx.font = b.kind === "size" ? HADES_FONT.value : HADES_FONT.label;
                const bw = ctx.measureText(b.text).width + 14;
                rx -= bw;
                drawRoundedRect(ctx, rx, ty + 3, bw, 18, 3);
                ctx.fillStyle = b.kind === "import" ? `${HADES_THEME.accent}30` : HADES_THEME.bgControl;
                ctx.fill();
                ctx.strokeStyle = b.kind === "import" ? HADES_THEME.accent : HADES_THEME.borderSubtle;
                ctx.stroke();
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.textAlign = "center";
                ctx.fillText(b.text, rx + bw / 2, ty + 12.5);
                zones.push({ kind: b.kind, x: rx, y: ty, w: bw, h: 24 });
                rx -= 4;
            }

            // canvas
            const f = frame(widgetWidth, y + ROW_H, d);
            ctx.save();
            ctx.beginPath();
            ctx.rect(f.x, f.y, f.w, f.h);
            ctx.clip();
            ctx.fillStyle = HADES_THEME.bgControl;
            ctx.fillRect(f.x, f.y, f.w, f.h);
            if (bg && state.showBg) {
                ctx.globalAlpha = 0.6;
                ctx.drawImage(bg, f.x, f.y, f.w, f.h);
                ctx.globalAlpha = 1;
            }
            // 64 px guide grid (every fourth latent cell)
            ctx.strokeStyle = "#ffffff0d";
            ctx.lineWidth = 1;
            for (let gx = 64; gx < d.width; gx += 64) { ctx.beginPath(); ctx.moveTo(f.x + gx * f.s + 0.5, f.y); ctx.lineTo(f.x + gx * f.s + 0.5, f.y + f.h); ctx.stroke(); }
            for (let gy = 64; gy < d.height; gy += 64) { ctx.beginPath(); ctx.moveTo(f.x, f.y + gy * f.s + 0.5); ctx.lineTo(f.x + f.w, f.y + gy * f.s + 0.5); ctx.stroke(); }

            const boxes = d.items.map((it, i) => (state.drag?.i === i && state.drag.box ? state.drag.box : it.bbox));
            d.items.forEach((it, i) => {
                const b = boxes[i];
                if (!Array.isArray(b) || b.length !== 4) return;
                const color = LABEL_COLORS[it.label] || HADES_THEME.textMuted;
                const bx = f.x + b[0] * f.s, by = f.y + b[1] * f.s, bw = (b[2] - b[0]) * f.s, bh = (b[3] - b[1]) * f.s;
                const sel = i === state.sel;
                const problem = itemProblem(it);
                ctx.fillStyle = `${color}${sel ? "33" : "1f"}`;
                ctx.fillRect(bx, by, bw, bh);
                ctx.lineWidth = sel ? 2 : 1.25;
                if (problem) ctx.setLineDash([4, 3]);
                ctx.strokeStyle = problem ? HADES_THEME.danger : color;
                ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
                ctx.setLineDash([]);
                // label tag + content
                ctx.save();
                ctx.beginPath();
                ctx.rect(bx + 2, by + 2, bw - 4, bh - 4);
                ctx.clip();
                ctx.font = HADES_FONT.label;
                ctx.fillStyle = color;
                ctx.textAlign = "left";
                ctx.fillText(`${i + 1} ${it.label.toUpperCase()}`, bx + 6, by + 10);
                ctx.font = HADES_FONT.text;
                ctx.fillStyle = HADES_THEME.textPrimary;
                const body = it.label === "text" && it.text ? `“${shown(it.text)}”` : shown(it.caption);
                const lines = String(problem && !body ? problem : body).split(" ");
                let line = "", ly = by + 24;
                for (const word of lines) {
                    const test = line ? `${line} ${word}` : word;
                    if (ctx.measureText(test).width > bw - 12 && line) {
                        ctx.fillText(line, bx + 6, ly);
                        line = word;
                        ly += 13;
                        if (ly > by + bh - 6) break;
                    } else line = test;
                }
                if (ly <= by + bh - 6) ctx.fillText(line, bx + 6, ly);
                ctx.restore();
                if (sel) {
                    ctx.fillStyle = HADES_THEME.textPrimary;
                    for (const [hx, hy] of [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]]) ctx.fillRect(hx - 3.5, hy - 3.5, 7, 7);
                }
            });
            if (state.drag?.kind === "draw" && state.drag.box) {
                const b = state.drag.box;
                ctx.setLineDash([4, 3]);
                ctx.strokeStyle = LABEL_COLORS[state.tool];
                ctx.lineWidth = 1.5;
                ctx.strokeRect(f.x + b[0] * f.s, f.y + b[1] * f.s, (b[2] - b[0]) * f.s, (b[3] - b[1]) * f.s);
                ctx.setLineDash([]);
            }
            ctx.restore();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.lineWidth = 1;
            ctx.strokeRect(f.x - 0.5, f.y - 0.5, f.w + 1, f.h + 1);
            zones.push({ kind: "canvas", x: f.x, y: f.y, w: f.w, h: f.h, f, boxes });

            // rows under the canvas: the selected box, or the summary
            let ry = f.y + f.h + 8;
            const row = (kind, tag, text, marker, problem = false) => {
                drawRowSurface(ctx, x, ry + 2, w, 24);
                drawRowLabel(ctx, x, ry + 2, 24, 74, tag, marker);
                ctx.save();
                ctx.beginPath();
                ctx.rect(x + 80, ry, w - 88, ROW_H);
                ctx.clip();
                ctx.font = HADES_FONT.text;
                ctx.textAlign = "left";
                ctx.fillStyle = problem ? HADES_THEME.danger : text ? HADES_THEME.textPrimary : HADES_THEME.textMuted;
                ctx.fillText(text || "— click to write —", x + 80, ry + 14.5);
                ctx.restore();
                zones.push({ kind, x, y: ry + 2, w, h: 24 });
                ry += ROW_H;
            };
            const it = d.items[state.sel];
            if (it) {
                const color = LABEL_COLORS[it.label];
                // label row: the three labels to pick from, order and delete at the right
                drawRowSurface(ctx, x, ry + 2, w, 24);
                drawRowLabel(ctx, x, ry + 2, 24, 74, `BOX ${state.sel + 1}/${d.items.length}`, color);
                let lx = x + 80;
                ctx.font = HADES_FONT.label;
                for (const label of LABELS) {
                    const lw = ctx.measureText(label.toUpperCase()).width + 14;
                    drawRoundedRect(ctx, lx, ry + 6, lw, 16, 8);
                    ctx.fillStyle = it.label === label ? `${LABEL_COLORS[label]}40` : "transparent";
                    ctx.fill();
                    ctx.strokeStyle = it.label === label ? LABEL_COLORS[label] : HADES_THEME.borderSubtle;
                    ctx.stroke();
                    ctx.fillStyle = it.label === label ? HADES_THEME.textPrimary : HADES_THEME.textMuted;
                    ctx.textAlign = "center";
                    ctx.fillText(label.toUpperCase(), lx + lw / 2, ry + 14.5);
                    zones.push({ kind: "label", label, x: lx, y: ry + 2, w: lw, h: 24 });
                    lx += lw + 4;
                }
                const actions = [["front", "TO FRONT"], ["delete", "DELETE"]];
                let ax = x + w - 4;
                for (const [kind, text] of actions.reverse()) {
                    const aw = ctx.measureText(text).width + 12;
                    ax -= aw;
                    ctx.fillStyle = kind === "delete" ? HADES_THEME.danger : HADES_THEME.textSecondary;
                    ctx.textAlign = "center";
                    ctx.fillText(text, ax + aw / 2, ry + 14.5);
                    zones.push({ kind, x: ax, y: ry + 2, w: aw, h: 24 });
                    ax -= 4;
                }
                ry += ROW_H;
                row("caption", "CAPTION", shown(it.caption), color, !String(it.caption ?? "").trim());
                row("text", "TEXT", shown(it.text), color, it.label === "text" && !String(it.text ?? "").trim());
                row("bbox", "BOX PX", Array.isArray(it.bbox) ? `${it.bbox.join(", ")}   (${it.bbox[2] - it.bbox[0]} × ${it.bbox[3] - it.bbox[1]})` : "", color);
            } else {
                row("summary", "IMAGE", shown(d.summary?.caption), null, !String(d.summary?.caption ?? "").trim());
            }
            this._zones = zones;
            ctx.restore();
        },

        mouse(event, pos, n) {
            const [mx, my] = pos;
            const type = event.type;
            const p = read();
            const hit = this._zones.find((z) => mx >= z.x && mx <= z.x + z.w && my >= z.y && my <= z.y + z.h);
            const canvasZone = this._zones.find((z) => z.kind === "canvas");

            if (type === "pointermove" || type === "mousemove") {
                const g = state.drag;
                if (g && canvasZone && p.draft) {
                    const { f } = canvasZone;
                    const d = p.draft;
                    const px = clamp((mx - f.x) / f.s, 0, d.width), py = clamp((my - f.y) / f.s, 0, d.height);
                    const dx = snap(px - g.startX), dy = snap(py - g.startY);
                    const [x1, y1, x2, y2] = g.orig;
                    if (g.kind === "move") {
                        const bw = x2 - x1, bh = y2 - y1;
                        const nx = clamp(x1 + dx, 0, d.width - bw), ny = clamp(y1 + dy, 0, d.height - bh);
                        g.box = [nx, ny, nx + bw, ny + bh];
                    } else if (g.kind === "resize") {
                        const b = [x1, y1, x2, y2];
                        if (g.corner.includes("l")) b[0] = clamp(snap(x1 + dx), 0, x2 - MIN_BOX);
                        if (g.corner.includes("r")) b[2] = clamp(snap(x2 + dx), x1 + MIN_BOX, d.width);
                        if (g.corner.includes("t")) b[1] = clamp(snap(y1 + dy), 0, y2 - MIN_BOX);
                        if (g.corner.includes("b")) b[3] = clamp(snap(y2 + dy), y1 + MIN_BOX, d.height);
                        g.box = b;
                    } else if (g.kind === "draw") {
                        const ax = snap(g.startX), ay = snap(g.startY), bx = snap(px), by = snap(py);
                        g.box = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
                    }
                    if (g.box && (g.box[0] !== x1 || g.box[1] !== y1 || g.box[2] !== x2 || g.box[3] !== y2)) g.moved = true;
                    n.setDirtyCanvas(true, false);
                    return true;
                }
                if (hit?.kind === "size") showHadesTooltip("<strong>Canvas size</strong><br>Click to type width × height (multiples of 16). Boxes scale with it.", event.clientX, event.clientY, HADES_THEME.accent);
                else if (hit?.kind === "tool") showHadesTooltip(`<strong>${hit.label}</strong><br>Drag on an empty part of the canvas to draw a new ${hit.label} box.`, event.clientX, event.clientY, LABEL_COLORS[hit.label]);
                else if (hit?.kind === "import") showHadesTooltip("<strong>Import</strong><br>Replaces this draft with the one at the draft_in input (from the last run).", event.clientX, event.clientY, HADES_THEME.accent);
                else if (hit?.kind === "match") showHadesTooltip("<strong>Match image</strong><br>Sets the canvas to the background image's aspect ratio at about the current area, snapped to 16 px.", event.clientX, event.clientY, HADES_THEME.accent);
                else hideHadesTooltip();
                return false;
            }

            if (type === "pointerup" || type === "mouseup") {
                const g = state.drag;
                state.drag = null;
                if (!g || !p.draft) return Boolean(g);
                const d = p.draft;
                if (g.kind === "draw") {
                    const b = g.box;
                    if (b && b[2] - b[0] >= MIN_BOX && b[3] - b[1] >= MIN_BOX) {
                        d.items.push({ label: state.tool, caption: "", ...(state.tool === "text" ? { text: "" } : {}), bbox: b });
                        state.sel = d.items.length - 1;
                        write(d);
                    } else {
                        state.sel = -1;
                        fit();
                    }
                } else if (g.moved && d.items[g.i]) {
                    d.items[g.i].bbox = g.box;
                    write(d);
                }
                n.setDirtyCanvas(true, true);
                return true;
            }

            if ((type !== "pointerdown" && type !== "mousedown") || !hit) return false;
            hideHadesTooltip();
            if (hit.kind === "new") {
                write(newDraft(p.caption ?? ""));
                return true;
            }
            if (!p.draft) return false;
            const d = p.draft;
            const it = d.items[state.sel];
            const edit = (title, value, apply, hint) => openTextEditor(event, title, value, (v) => { apply(v); write(d); }, hint);

            switch (hit.kind) {
                case "tool":
                    state.tool = hit.label;
                    n.setDirtyCanvas(true, false);
                    return true;
                case "size":
                    edit("Canvas width × height", `${d.width} x ${d.height}`, (v) => {
                        const m = v.match(/(\d+)\s*[x×*, ]\s*(\d+)/i);
                        if (m) resizeCanvas(d, Number(m[1]), Number(m[2]));
                    }, "e.g. 1024 x 1536 · snapped to 16 px · Ctrl+Enter applies");
                    return true;
                case "match": {
                    const img = imageFor(bgUrl());
                    if (img) {
                        const area = d.width * d.height;
                        const r = img.naturalWidth / img.naturalHeight;
                        resizeCanvas(d, Math.sqrt(area * r), Math.sqrt(area / r));
                        write(d);
                    }
                    return true;
                }
                case "bg":
                    state.showBg = !state.showBg;
                    n.setDirtyCanvas(true, false);
                    return true;
                case "import": {
                    const imported = parseDraft(node._hadesDraftIn);
                    if (imported.draft || imported.caption !== undefined) {
                        state.sel = -1;
                        setWidgetValue(backing, imported.draft ? JSON.stringify(imported.draft, null, 2) : node._hadesDraftIn, node);
                        fit();
                    }
                    return true;
                }
                case "label":
                    it.label = hit.label;
                    if (hit.label === "text" && it.text === undefined) it.text = "";
                    write(d);
                    return true;
                case "front":
                    d.items.push(d.items.splice(state.sel, 1)[0]);
                    state.sel = d.items.length - 1;
                    write(d);
                    return true;
                case "delete":
                    d.items.splice(state.sel, 1);
                    state.sel = -1;
                    write(d);
                    return true;
                case "caption":
                    edit(`Box ${state.sel + 1} caption`, it.caption, (v) => { it.caption = v.trim(); }, "What is in the box: subject, look, for text the type style · Ctrl+Enter applies");
                    return true;
                case "text":
                    edit(`Box ${state.sel + 1} text`, it.text, (v) => {
                        const t = v.replace(/\r/g, "").trim();
                        if (t) it.text = t; else delete it.text;
                    }, "The literal words to render; a new line here is a line break in the image · Ctrl+Enter applies");
                    return true;
                case "bbox":
                    edit(`Box ${state.sel + 1} x1, y1, x2, y2 (pixels)`, it.bbox.join(", "), (v) => {
                        const nums = v.split(/[^\d.-]+/).filter(Boolean).map(Number);
                        if (nums.length === 4 && nums.every(Number.isFinite)) {
                            const [x1, y1, x2, y2] = nums;
                            const b = [clamp(Math.min(x1, x2), 0, d.width), clamp(Math.min(y1, y2), 0, d.height), clamp(Math.max(x1, x2), 0, d.width), clamp(Math.max(y1, y2), 0, d.height)];
                            if (b[2] > b[0] && b[3] > b[1]) it.bbox = b.map(Math.round);
                        }
                    }, "Exact pixels, not snapped · Ctrl+Enter applies");
                    return true;
                case "summary":
                    d.summary = d.summary || {};
                    edit("Whole image", d.summary.caption, (v) => { d.summary.caption = v.trim(); }, "Medium, setting, lighting, palette of the whole image · Ctrl+Enter applies");
                    return true;
                case "canvas": {
                    const { f, boxes } = hit;
                    const px = (mx - f.x) / f.s, py = (my - f.y) / f.s;
                    // a handle of the selected box first, then the topmost box under the pointer
                    const sb = boxes[state.sel];
                    if (sb) {
                        const corners = { tl: [sb[0], sb[1]], tr: [sb[2], sb[1]], bl: [sb[0], sb[3]], br: [sb[2], sb[3]] };
                        for (const [corner, [cx, cy]] of Object.entries(corners)) {
                            if (Math.abs(cx * f.s - px * f.s) <= HANDLE && Math.abs(cy * f.s - py * f.s) <= HANDLE) {
                                state.drag = { kind: "resize", corner, i: state.sel, orig: [...sb], startX: px, startY: py, moved: false };
                                return true;
                            }
                        }
                    }
                    for (let i = boxes.length - 1; i >= 0; i--) {
                        const b = boxes[i];
                        if (Array.isArray(b) && px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3]) {
                            state.sel = i;
                            state.drag = { kind: "move", i, orig: [...b], startX: px, startY: py, moved: false };
                            fit();
                            return true;
                        }
                    }
                    state.drag = { kind: "draw", orig: [0, 0, 0, 0], startX: px, startY: py, moved: false };
                    return true;
                }
            }
            return false;
        },
    };
}

export function layoutEditorStatus(backing) {
    const p = parseDraft(backing.value);
    if (!p.draft) return p.error ? `DRAFT ERROR · ${p.error}` : "CAPTION DRAFT · no boxes";
    const d = p.draft;
    const count = (l) => d.items.filter((i) => i.label === l).length;
    const todo = d.items.filter(itemProblem).length;
    const parts = [`${d.width} × ${d.height}`, `${d.items.length} boxes`];
    for (const l of LABELS) if (count(l)) parts.push(`${count(l)} ${l}`);
    if (todo) parts.push(`${todo} to fill in`);
    return parts.join(" · ");
}
