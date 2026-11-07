import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    HADES_FONT,
    HADES_THEME,
    clampNum,
    drawRoundedRect,
    drawRowSurface,
    formatCleanLabel,
    hideHadesTooltip,
    setWidgetValue,
    showHadesTooltip,
} from "./moss_widgets.js";

// =============================================================================
// MOSS MULTI-LORA: a list of LoRA rows stored as JSON in the node's hidden `loras` widget
// =============================================================================
// Row: on / off pill · LoRA name (click: searchable picker) · strength [− 1.00 +] · layer groups · remove.
// The groups panel holds one weight per group of 6 layers plus the parts outside them; the backend multiplies them by the strength.

const ROW_H = 28;
const PANEL_H = 44;
const FOOT_H = 28;
const STRENGTH = { min: -10, max: 10, step: 0.05, visMin: 0, visMax: 2 };
const GROUP_WEIGHT = { min: -2, max: 3, step: 0.05 };

// ---- few-step LoRAs, read from the file's own metadata ("4 steps, CFG 1, ..."), as the backend does ----
const fewStepCache = new Map(); // name -> steps | null | "pending"
const STEPS_RE = /^\s*(\d+)\s+steps?\b/i;
const CFG1_RE = /\bCFG 1(\.0+)?(?![.\d])/;
// LoRA names come from the workflow's JSON, which may be someone else's: never insert them into the tooltip as HTML
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function fewStepSteps(name, node) {
    if (!name) return null;
    if (fewStepCache.has(name)) {
        const v = fewStepCache.get(name);
        return v === "pending" ? null : v;
    }
    fewStepCache.set(name, "pending");
    api.fetchApi(`/view_metadata/loras?filename=${encodeURIComponent(name)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((meta) => {
            const settings = String(meta?.settings ?? "");
            const m = settings.match(STEPS_RE);
            fewStepCache.set(name, m && CFG1_RE.test(settings) ? Number(m[1]) : null);
            node.setDirtyCanvas(true, true);
        })
        .catch(() => fewStepCache.set(name, null));
    return null;
}

// ---- searchable LoRA picker (DOM) ----
let pickerEl = null;

function closePicker() {
    if (!pickerEl) return;
    pickerEl._cleanup?.();
    pickerEl.remove();
    pickerEl = null;
}

async function fetchLoraNames() {
    try {
        const r = await api.fetchApi("/models/loras");
        if (r.ok) return await r.json();
    } catch {}
    return [];
}

async function openLoraPicker(event, current, onPick) {
    closePicker();
    const names = await fetchLoraNames();
    const T = HADES_THEME;
    const el = document.createElement("div");
    pickerEl = el;
    Object.assign(el.style, {
        position: "fixed", zIndex: 10000, width: "420px", maxHeight: "440px", display: "flex", flexDirection: "column",
        background: T.bgNode, border: `1px solid ${T.borderActive}`, borderRadius: "6px", boxShadow: "0 8px 28px #000a",
        font: HADES_FONT.text, color: T.textPrimary, overflow: "hidden",
    });
    const left = Math.min(event.clientX, window.innerWidth - 430);
    const top = Math.min(event.clientY, window.innerHeight - 450);
    el.style.left = `${Math.max(8, left)}px`;
    el.style.top = `${Math.max(8, top)}px`;

    const input = document.createElement("input");
    input.placeholder = `Search ${names.length} LoRAs: words in any order`;
    Object.assign(input.style, {
        margin: "8px", padding: "6px 8px", background: T.bgControl, color: T.textPrimary, border: `1px solid ${T.borderSubtle}`,
        borderRadius: "4px", outline: "none", font: HADES_FONT.text,
    });
    const list = document.createElement("div");
    Object.assign(list.style, { overflowY: "auto", padding: "0 4px 6px" });
    el.append(input, list);
    document.body.appendChild(el);

    let shown = [];
    let active = 0;
    const render = () => {
        const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
        shown = names.filter((n) => words.every((w) => n.toLowerCase().includes(w))).slice(0, 400);
        if (active >= shown.length) active = Math.max(0, shown.length - 1);
        list.replaceChildren(...shown.map((n, i) => {
            const item = document.createElement("div");
            const parts = n.split(/[\\/]/);
            const file = formatCleanLabel(parts.pop());
            const folder = parts.join("/");
            Object.assign(item.style, {
                padding: "4px 8px", borderRadius: "3px", cursor: "pointer", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                background: i === active ? T.bgControlHover : "transparent", borderLeft: `2px solid ${n === current ? T.accent : "transparent"}`,
            });
            item.title = n;
            item.innerHTML = "";
            if (folder) {
                const f = document.createElement("span");
                f.textContent = `${folder}/`;
                f.style.color = T.textMuted;
                item.append(f);
            }
            item.append(document.createTextNode(file));
            item.addEventListener("pointerenter", () => { active = i; paint(); });
            item.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); choose(n); });
            return item;
        }));
        if (!shown.length) {
            const empty = document.createElement("div");
            empty.textContent = names.length ? "No LoRA matches" : "No LoRAs in models/loras";
            Object.assign(empty.style, { padding: "6px 8px", color: T.textMuted });
            list.append(empty);
        }
    };
    const paint = () => {
        [...list.children].forEach((c, i) => { if (shown[i]) c.style.background = i === active ? T.bgControlHover : "transparent"; });
        list.children[active]?.scrollIntoView({ block: "nearest" });
    };
    const choose = (n) => {
        closePicker();
        onPick(n);
    };
    input.addEventListener("input", () => { active = 0; render(); });
    input.addEventListener("keydown", (e) => {
        e.stopPropagation(); // keep ComfyUI's shortcuts out of the search box
        if (e.key === "ArrowDown") { active = Math.min(shown.length - 1, active + 1); paint(); e.preventDefault(); }
        else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); paint(); e.preventDefault(); }
        else if (e.key === "Enter") { if (shown[active]) choose(shown[active]); e.preventDefault(); }
        else if (e.key === "Escape") { closePicker(); e.preventDefault(); }
    });
    const outside = (e) => { if (!el.contains(e.target)) closePicker(); };
    setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
    el._cleanup = () => document.removeEventListener("pointerdown", outside, true);

    const start = names.indexOf(current);
    if (start >= 0) active = start;
    render();
    paint();
    input.focus();
}

// ---- the list widget ----
function readRows(backing) {
    try {
        const rows = JSON.parse(backing.value || "[]");
        return Array.isArray(rows) ? rows.filter((r) => r && typeof r === "object") : [];
    } catch {
        return [];
    }
}

function isCustomGroups(row, keys) {
    return keys.some((k) => row.groups?.[k] !== undefined && Number(row.groups[k]) !== 1);
}

export function createMultiLoraWidget(node, backing, { groupKeys, groupColors, isEnabled }) {
    const groupLabel = (k) => {
        const m = k.match(/^layers_(\d+)_(\d+)$/);
        return m ? `${Number(m[1])}–${Number(m[2])}` : "other";
    };
    const open = new Set(); // rows with the groups panel showing (UI only)

    const write = (rows) => {
        setWidgetValue(backing, JSON.stringify(rows), node);
        fit();
    };
    const fit = () => {
        const h = node.computeSize()[1];
        node.setSize([node.size[0], h + 8]);
        node.setDirtyCanvas(true, true);
    };
    const promptNumber = (title, value, range, event, apply) => {
        app.canvas?.prompt(title, String(value), (val) => {
            const parsed = parseFloat(val);
            if (!Number.isNaN(parsed)) apply(clampNum(parsed, range.min, range.max, 2));
        }, event);
    };

    return {
        name: "hades_multilora_list",
        _hadesBackingWidget: backing,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _zones: [],
        _drag: null,
        _hover: null,

        computeSize(width) {
            const rows = readRows(backing);
            return [width, rows.length * ROW_H + [...open].filter((i) => i < rows.length).length * PANEL_H + FOOT_H];
        },

        draw(ctx, n, widgetWidth, y) {
            const rows = readRows(backing);
            const x = 12;
            const w = widgetWidth - 24;
            const zones = [];
            ctx.save();
            if (!isEnabled()) ctx.globalAlpha = 0.45;
            ctx.textBaseline = "middle";
            let cy = y;

            rows.forEach((row, i) => {
                const on = row.on !== false;
                const rowY = cy + 2;
                const h = 24;
                const mid = rowY + h / 2 + 0.5;
                drawRowSurface(ctx, x, rowY, w, h, this._hover === i);
                ctx.save();
                if (!on) ctx.globalAlpha *= 0.5;

                // on / off pill
                const pill = { x: x + 6, y: rowY + 6, w: 22, h: 12 };
                drawRoundedRect(ctx, pill.x, pill.y, pill.w, pill.h, 6);
                ctx.fillStyle = on ? HADES_THEME.accent : HADES_THEME.bgControl;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.borderSubtle;
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(on ? pill.x + pill.w - 6 : pill.x + 6, pill.y + 6, 4, 0, Math.PI * 2);
                ctx.fillStyle = on ? HADES_THEME.bgNode : HADES_THEME.textMuted;
                ctx.fill();
                zones.push({ kind: "toggle", i, x: pill.x - 4, y: rowY, w: pill.w + 8, h });

                // right side: remove, groups, strength
                const rm = { x: x + w - 20, w: 18 };
                const gb = { x: rm.x - 22, w: 20 };
                const st = { x: gb.x - 100, w: 96 };
                ctx.font = HADES_FONT.small;
                ctx.textAlign = "center";
                ctx.fillStyle = HADES_THEME.textMuted;
                ctx.fillText("✕", rm.x + rm.w / 2, mid);
                zones.push({ kind: "remove", i, x: rm.x, y: rowY, w: rm.w, h });

                const custom = isCustomGroups(row, groupKeys);
                drawRoundedRect(ctx, gb.x + 2, rowY + 5, gb.w - 4, h - 10, 3);
                ctx.fillStyle = open.has(i) ? HADES_THEME.bgControlHover : HADES_THEME.bgControl;
                ctx.fill();
                // seven tiny bars: the group weights at a glance
                const gw = (gb.w - 8) / groupKeys.length;
                groupKeys.forEach((k, gi) => {
                    const v = Math.max(0, Math.min(2, Number(row.groups?.[k] ?? 1)));
                    const bh = Math.max(1, (v / 2) * (h - 14));
                    ctx.fillStyle = custom ? groupColors[k] || HADES_THEME.accent : HADES_THEME.textMuted;
                    ctx.fillRect(gb.x + 4 + gi * gw, rowY + h - 7 - bh, Math.max(1, gw - 0.6), bh);
                });
                zones.push({ kind: "groups", i, x: gb.x, y: rowY, w: gb.w, h });

                const strength = Number(row.strength ?? 1);
                drawRoundedRect(ctx, st.x, rowY + 3, st.w, h - 6, 3);
                ctx.fillStyle = HADES_THEME.bgControl;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.borderSubtle;
                ctx.stroke();
                ctx.font = HADES_FONT.text;
                ctx.fillStyle = HADES_THEME.textMuted;
                ctx.fillText("−", st.x + 8, mid);
                ctx.fillText("+", st.x + st.w - 8, mid);
                ctx.font = HADES_FONT.value;
                ctx.fillStyle = strength === 0 ? HADES_THEME.textMuted : HADES_THEME.textPrimary;
                ctx.fillText(strength.toFixed(2), st.x + st.w / 2, mid);
                zones.push({ kind: "dec", i, x: st.x, y: rowY, w: 17, h });
                zones.push({ kind: "inc", i, x: st.x + st.w - 17, y: rowY, w: 17, h });
                zones.push({ kind: "strength", i, x: st.x + 17, y: rowY, w: st.w - 34, h });

                // name, with the few-step tag when the file's metadata gives one
                const nameX = pill.x + pill.w + 8;
                let nameW = st.x - nameX - 6;
                const steps = fewStepSteps(row.name, node);
                if (steps) {
                    ctx.font = HADES_FONT.label;
                    const tag = `${steps} STEPS`;
                    const tw = ctx.measureText(tag).width + 10;
                    const tx = st.x - 6 - tw;
                    drawRoundedRect(ctx, tx, rowY + 6, tw, h - 12, 3);
                    ctx.strokeStyle = HADES_THEME.warn;
                    ctx.stroke();
                    ctx.fillStyle = HADES_THEME.warn;
                    ctx.textAlign = "center";
                    ctx.fillText(tag, tx + tw / 2, mid);
                    nameW = tx - nameX - 6;
                }
                ctx.save();
                ctx.beginPath();
                ctx.rect(nameX, rowY, Math.max(0, nameW), h);
                ctx.clip();
                ctx.font = HADES_FONT.text;
                ctx.textAlign = "left";
                ctx.fillStyle = row.name ? HADES_THEME.textPrimary : HADES_THEME.textMuted;
                let label = row.name ? formatCleanLabel(row.name) : "— pick a LoRA —";
                if (ctx.measureText(label).width > nameW) {
                    while (label.length > 2 && ctx.measureText(label + "…").width > nameW) label = label.slice(0, -1);
                    label += "…";
                }
                ctx.fillText(label, nameX, mid);
                ctx.restore();
                zones.push({ kind: "name", i, x: nameX, y: rowY, w: Math.max(0, nameW), h });
                ctx.restore();
                cy += ROW_H;

                // groups panel: one cell per group of layers + other
                if (open.has(i)) {
                    const keys = [...groupKeys, "other_layers"];
                    const py = cy;
                    const cellW = (w - 8) / keys.length;
                    drawRoundedRect(ctx, x + 10, py, w - 10, PANEL_H - 4, 4);
                    ctx.fillStyle = HADES_THEME.bgControl;
                    ctx.fill();
                    keys.forEach((k, gi) => {
                        const cx = x + 12 + gi * cellW;
                        const v = Number(row.groups?.[k] ?? 1);
                        ctx.fillStyle = groupColors[k] || HADES_THEME.textMuted;
                        ctx.fillRect(cx + 3, py + 4, cellW - 8, 2);
                        ctx.textAlign = "center";
                        ctx.font = HADES_FONT.label;
                        ctx.fillStyle = HADES_THEME.textMuted;
                        ctx.fillText(groupLabel(k), cx + cellW / 2 - 1, py + 14);
                        ctx.font = HADES_FONT.value;
                        ctx.fillStyle = v === 1 ? HADES_THEME.textSecondary : HADES_THEME.textPrimary;
                        ctx.fillText(v.toFixed(2), cx + cellW / 2 - 1, py + 29);
                        zones.push({ kind: "group", i, key: k, x: cx, y: py, w: cellW, h: PANEL_H - 4 });
                    });
                    cy += PANEL_H;
                }
            });

            // footer: add a LoRA
            const fy = cy + 2;
            drawRoundedRect(ctx, x, fy, w, 24, 4);
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = this._hover === "add" ? HADES_THEME.borderActive : HADES_THEME.borderSubtle;
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = HADES_FONT.label;
            ctx.textAlign = "center";
            ctx.fillStyle = HADES_THEME.textSecondary;
            ctx.fillText("+  ADD LORA", x + w / 2, fy + 12.5);
            zones.push({ kind: "add", x, y: fy, w, h: 24 });

            this._zones = zones;
            ctx.restore();
        },

        mouse(event, pos, n) {
            const [mx, my] = pos;
            const hit = this._zones.find((z) => mx >= z.x && mx <= z.x + z.w && my >= z.y && my <= z.y + z.h);
            const type = event.type;
            const rows = readRows(backing);

            if (type === "pointermove" || type === "mousemove") {
                if (this._drag) {
                    const d = this._drag;
                    const dx = d.vertical ? d.startY - my : mx - d.startX;
                    if (Math.abs(dx) > 2) d.moved = true;
                    if (d.moved) {
                        const range = d.key ? GROUP_WEIGHT : STRENGTH;
                        const sens = d.key ? 2 / 80 : (STRENGTH.visMax - STRENGTH.visMin) / 140;
                        const next = clampNum(Math.round((d.startVal + dx * sens) / range.step) * range.step, range.min, range.max, 2);
                        const r = readRows(backing);
                        if (!r[d.i]) return true;
                        if (d.key) r[d.i].groups = { ...(r[d.i].groups || {}), [d.key]: next };
                        else r[d.i].strength = next;
                        setWidgetValue(backing, JSON.stringify(r), n);
                    }
                    return true;
                }
                const hover = hit ? (hit.kind === "add" ? "add" : hit.i) : null;
                if (hover !== this._hover) {
                    this._hover = hover;
                    n.setDirtyCanvas(true, false);
                }
                if (hit?.kind === "name" && rows[hit.i]?.name) {
                    const steps = fewStepSteps(rows[hit.i].name, n);
                    showHadesTooltip(`<strong>${escapeHtml(rows[hit.i].name)}</strong>${steps ? `<br>Few-step LoRA: its metadata says ${steps} steps at CFG 1.` : ""}<br>Click to pick another.`,
                        event.clientX, event.clientY, HADES_THEME.accent);
                } else if (hit?.kind === "groups") {
                    showHadesTooltip("<strong>Strength per layer group</strong><br>Opens the weights for each group of 6 layers and for the parts outside them; each multiplies the row's strength.",
                        event.clientX, event.clientY, HADES_THEME.accent);
                } else if (hit?.kind === "group") {
                    showHadesTooltip(`<strong>${hit.key === "other_layers" ? "Outside the layers" : `Layers ${groupLabel(hit.key)}`}</strong><br>Drag up / down, click to type, right-click for 1.0.`,
                        event.clientX, event.clientY, groupColors[hit.key] || HADES_THEME.accent);
                } else if (hit?.kind === "strength") {
                    showHadesTooltip("<strong>Strength</strong><br>Drag sideways, click to type, right-click for 1.0.", event.clientX, event.clientY, HADES_THEME.accent);
                } else {
                    hideHadesTooltip();
                }
                return false;
            }

            if (type === "pointerleave" || type === "mouseleave") {
                this._hover = null;
                hideHadesTooltip();
                return false;
            }

            if (type === "pointerup" || type === "mouseup") {
                const d = this._drag;
                this._drag = null;
                if (d && !d.moved) {
                    const range = d.key ? GROUP_WEIGHT : STRENGTH;
                    const title = d.key ? (d.key === "other_layers" ? "Outside the layers" : `Layers ${groupLabel(d.key)}`) : "Strength";
                    promptNumber(`${title} (${range.min} to ${range.max})`, d.startVal, range, event, (v) => {
                        const r = readRows(backing);
                        if (!r[d.i]) return;
                        if (d.key) r[d.i].groups = { ...(r[d.i].groups || {}), [d.key]: v };
                        else r[d.i].strength = v;
                        write(r);
                    });
                }
                return Boolean(d);
            }

            if ((type !== "pointerdown" && type !== "mousedown") || !hit) return false;
            hideHadesTooltip();
            const row = rows[hit.i];
            const right = event.button === 2;

            switch (hit.kind) {
                case "add":
                    openLoraPicker(event, null, (name) => write([...readRows(backing), { name, on: true, strength: 1.0 }]));
                    return true;
                case "toggle":
                    row.on = row.on === false;
                    write(rows);
                    return true;
                case "remove": {
                    rows.splice(hit.i, 1);
                    const shifted = [...open].filter((k) => k !== hit.i).map((k) => (k > hit.i ? k - 1 : k));
                    open.clear();
                    shifted.forEach((k) => open.add(k));
                    write(rows);
                    return true;
                }
                case "groups":
                    if (right) {
                        delete row.groups;
                        write(rows);
                    } else {
                        open.has(hit.i) ? open.delete(hit.i) : open.add(hit.i);
                        fit();
                    }
                    return true;
                case "name":
                    openLoraPicker(event, row.name, (name) => {
                        const r = readRows(backing);
                        if (r[hit.i]) { r[hit.i].name = name; write(r); }
                    });
                    return true;
                case "dec":
                case "inc":
                    row.strength = clampNum(Number(row.strength ?? 1) + (hit.kind === "inc" ? STRENGTH.step : -STRENGTH.step), STRENGTH.min, STRENGTH.max, 2);
                    write(rows);
                    return true;
                case "strength":
                    if (right) { row.strength = 1.0; write(rows); return true; }
                    this._drag = { i: hit.i, startX: mx, startY: my, startVal: Number(row.strength ?? 1), moved: false };
                    return true;
                case "group":
                    if (right) {
                        row.groups = { ...(row.groups || {}), [hit.key]: 1.0 };
                        write(rows);
                        return true;
                    }
                    this._drag = { i: hit.i, key: hit.key, vertical: true, startX: mx, startY: my, startVal: Number(row.groups?.[hit.key] ?? 1), moved: false };
                    return true;
            }
            return false;
        },
    };
}

// Banner status: how many rows are on, and the few-step LoRAs with their steps.
export function multiLoraStatus(node, backing, isEnabled) {
    const rows = readRows(backing).filter((r) => r.name);
    if (!isEnabled()) return `OFF · model passed through · ${rows.length} in the list`;
    const on = rows.filter((r) => r.on !== false && Number(r.strength ?? 1) !== 0);
    const turbo = on.map((r) => fewStepSteps(r.name, node)).filter(Boolean);
    const parts = [`${on.length} of ${rows.length} on`];
    if (turbo.length === 1) parts.push(`few-step LoRA · ${turbo[0]} steps, CFG 1`);
    else if (turbo.length > 1) parts.push(`${turbo.length} few-step LoRAs stacked`);
    return parts.join(" · ");
}
