import { app } from "../../scripts/app.js";

// =============================================================================
// HADES UI - DESIGN TOKENS & CORE CANVAS HELPERS
// =============================================================================
// Derived from the HaDeS Anima Suite widget library. One neutral surface scale and a single accent. Green / amber / red are reserved for state (ok / warning / error);
// the legacy hue names map onto that small palette so every node shares the same restrained look.
const ACCENT = "#9bbf6a";  // moss green; HaDeS Anima Suite uses blue
export const HADES_THEME = {
    bgNode: "#171a16",
    bgPanel: "#1e221c",
    bgControl: "#151813",
    bgControlHover: "#252b22",
    borderSubtle: "#2b3127",
    borderActive: "#9bbf6a66",
    textPrimary: "#e5e9e0",
    textSecondary: "#a3ab9a",
    textMuted: "#6e7566",
    accent: ACCENT,
    // ok is teal, not green, so a state never reads as the accent
    ok: "#5fb3a6",
    warn: "#d6a24e",
    danger: "#d9706a",
    blue: ACCENT,
    cyan: ACCENT,
    purple: ACCENT,
    emerald: "#5fb3a6",
    amber: "#d6a24e",
    rose: "#d9706a",
};

const FRAME_COLOR = "#20251d";  // node title bar

export const HADES_FONT = {
    label: "600 9px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
    text: "11px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
    value: "11px ui-monospace, 'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
    small: "500 9.5px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
};

// Shared row furniture: flat row surface + quiet uppercase label column (+ optional data-color marker).
function drawRowSurface(ctx, x, y, w, h, highlight = false) {
    drawRoundedRect(ctx, x, y, w, h, 4);
    ctx.fillStyle = HADES_THEME.bgPanel;
    ctx.fill();
    ctx.strokeStyle = highlight ? HADES_THEME.borderActive : HADES_THEME.borderSubtle;
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawRowLabel(ctx, x, y, h, labelW, text, marker = null) {
    if (marker) {
        drawRoundedRect(ctx, x + 4, y + 6, 3, h - 12, 1.5);
        ctx.fillStyle = marker;
        ctx.fill();
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, labelW, h);
    ctx.clip();
    ctx.font = HADES_FONT.label;
    if ("letterSpacing" in ctx) ctx.letterSpacing = "0.6px";
    ctx.fillStyle = HADES_THEME.textMuted;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(String(text).toUpperCase(), x + (marker ? 12 : 9), y + h * 0.5 + 0.5);
    ctx.restore();
}

// Snap a free-form drag value onto the widget step grid (relative to min) before clamping.
function snapToStep(value, min, step) {
    if (!step) return value;
    return min + Math.round((value - min) / step) * step;
}

// Shared layout finalizer: enforce a real minimum width (computeSize drives LiteGraph's resize handle),
// keep saved workflow sizes, and never shrink below the content height. Call after all widgets are added.
export function finalizeHadesNode(node, minW) {
    const origComputeSize = node.computeSize;
    node.computeSize = function () {
        const sz = origComputeSize.apply(this, arguments);
        sz[0] = Math.max(sz[0], minW);
        return sz;
    };
    const origOnResize = node.onResize;
    node.onResize = function (size) {
        origOnResize?.apply(this, arguments);
        size[0] = Math.max(size[0], minW);
    };
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        origOnConfigure?.apply(this, arguments);
        this.size[0] = Math.max(this.size[0], minW);
        this.size[1] = Math.max(this.size[1], this.computeSize()[1]);
    };
    // Frontend-driven value changes (control_after_generate, links) only fire the backing widget callback.
    for (const w of node.widgets || []) {
        if (!w._isHadesBacking) continue;
        const origCallback = w.callback;
        w.callback = function () {
            const r = origCallback?.apply(this, arguments);
            redrawVueWidgets(node);
            return r;
        };
    }
    const sz = node.computeSize();
    node.setSize([Math.max(minW, node.size?.[0] || 0), sz[1] + 8]);
}

export function formatCleanLabel(rawName, emptyText = "— None —") {
    if (!rawName || rawName === "None") return emptyText;
    const base = String(rawName).split(/[\\/]/).pop();
    return base.replace(/\.(safetensors|ckpt|pt|bin)$/i, "");
}

export function clampNum(val, min, max, decimals = 2) {
    const num = Number(val);
    if (Number.isNaN(num)) return min;
    const clamped = Math.max(min, Math.min(max, num));
    const factor = 10 ** decimals;
    return Math.round(clamped * factor) / factor;
}

// Vue nodes paint each legacy widget into its own <canvas> (WidgetLegacy), sized from getBoundingClientRect(),
// which includes the graph zoom. Re-fit it to its CSS layout width so drawn bounds match pointer offsets at any zoom.
function wrapVueWidgetDraw(widget) {
    const draw = widget.draw;
    widget.draw = function (ctx, node, width, y, height) {
        const el = ctx.canvas;
        if (LiteGraph.vueNodesMode && el !== app.canvas?.canvas) {
            el._hadesWidget = this;
            if (el.clientWidth && Math.abs(el.clientWidth - width) > 1) {
                const ratio = el.width / width;
                width = el.clientWidth;
                this.width = width;
                el.width = width * ratio;
                ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            }
        }
        return draw.call(this, ctx, node, width, y, height);
    };
}

// WidgetLegacy only repaints on resize or the widget's own callback; our rows read hidden backing widgets.
function redrawVueWidgets(node) {
    for (const w of node._hadesCustomWidgets || []) w.triggerDraw?.();
}

export function setWidgetValue(widget, value, node) {
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value, app.canvas, node, [0, 0], null);
    if (node?.graph) {
        node.graph._version = (node.graph._version || 0) + 1;
    }
    node?.setDirtyCanvas?.(true, true);
}

export function drawRoundedRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
}

// =============================================================================
// FLOATING HUD TOOLTIP ENGINE
// =============================================================================
let _hadesTooltipEl = null;

if (typeof window !== "undefined" && !window._hadesTooltipGlobalInit) {
    window._hadesTooltipGlobalInit = true;
    // Capture phase: Vue node widget canvases stop pointer event propagation.
    window.addEventListener("pointerdown", () => hideHadesTooltip(), { passive: true, capture: true });
    window.addEventListener("wheel", () => hideHadesTooltip(), { passive: true });

    window.addEventListener(
        "pointermove",
        (e) => {
            const canvas = app?.canvas;
            if (!canvas) return;
            let widgets, localX, localY;
            if (LiteGraph.vueNodesMode) {
                // Each widget has its own canvas there, drawn at y = 0, so bounds are canvas-local.
                const vueWidget = e.target?._hadesWidget;
                if (!vueWidget) {
                    hideHadesTooltip();
                    return;
                }
                widgets = [vueWidget];
                localX = e.offsetX;
                localY = e.offsetY;
            } else {
                const node = canvas.node_over;
                const graphMouse = canvas.graph_mouse;
                if (!node?.widgets || !graphMouse) {
                    hideHadesTooltip();
                    return;
                }
                widgets = node.widgets;
                localX = graphMouse[0] - node.pos[0];
                localY = graphMouse[1] - node.pos[1];
            }

            for (const w of widgets) {
                if (w.hidden) continue;

                // 1. Check banner action buttons
                if (Array.isArray(w._btnBounds)) {
                    for (const btn of w._btnBounds) {
                        if (
                            btn.tooltip &&
                            localX >= btn.x &&
                            localX <= btn.x + btn.w &&
                            localY >= btn.y &&
                            localY <= btn.y + btn.h
                        ) {
                            showHadesTooltip(btn.tooltip, e.clientX, e.clientY, btn.color || HADES_THEME.emerald);
                            return;
                        }
                    }
                }

                // 2. Check banner HUD pills
                if (Array.isArray(w._pillBounds)) {
                    for (const pill of w._pillBounds) {
                        if (
                            pill.tooltip &&
                            localX >= pill.x &&
                            localX <= pill.x + pill.w &&
                            localY >= pill.y &&
                            localY <= pill.y + pill.h
                        ) {
                            showHadesTooltip(pill.tooltip, e.clientX, e.clientY, pill.color || HADES_THEME.cyan);
                            return;
                        }
                    }
                }

                // 2b. Check banner HUD box / entire banner box
                if (w.tooltip && (w._hudBounds || w._bannerBounds)) {
                    const bb = w._hudBounds || w._bannerBounds;
                    if (localX >= bb.x && localX <= bb.x + bb.w && localY >= bb.y && localY <= bb.y + bb.h) {
                        showHadesTooltip(w.tooltip, e.clientX, e.clientY, w.accentColor || HADES_THEME.emerald);
                        return;
                    }
                }

                // 3. Check custom widgets (combos, sliders, toggles)
                const b = w._rowBounds || w._bounds;
                const tt = w.tooltipHtml || w.tooltip;
                if (b && tt) {
                    if (localX >= b.x && localX <= b.x + b.w && localY >= b.y && localY <= b.y + b.h) {
                        showHadesTooltip(tt, e.clientX, e.clientY, w.tagColor || w.color || HADES_THEME.emerald);
                        return;
                    }
                }
            }

            hideHadesTooltip();
        },
        { passive: true, capture: true }
    );
}

export function showHadesTooltip(htmlContent, clientX, clientY, accentColor = HADES_THEME.emerald) {
    if (!htmlContent || typeof document === "undefined") return;
    if (!_hadesTooltipEl) {
        _hadesTooltipEl = document.createElement("div");
        _hadesTooltipEl.id = "hades-moss-floating-tooltip";
        Object.assign(_hadesTooltipEl.style, {
            position: "fixed",
            zIndex: "99999",
            pointerEvents: "none",
            backgroundColor: "#1c2019",
            color: "#e4e7ec",
            padding: "8px 12px",
            borderRadius: "6px",
            fontFamily: "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
            fontSize: "11px",
            lineHeight: "1.45",
            maxWidth: "320px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.08)",
            transition: "opacity 0.12s ease-out, transform 0.12s ease-out",
            opacity: "0",
            transform: "translateY(4px)",
            display: "none",
        });
        document.body.appendChild(_hadesTooltipEl);
    }
    _hadesTooltipEl.innerHTML = htmlContent;
    _hadesTooltipEl.style.borderLeft = `2px solid ${HADES_THEME.accent}`;
    _hadesTooltipEl.style.display = "block";

    const pad = 14;
    const vpW = window.innerWidth || 1920;
    const vpH = window.innerHeight || 1080;
    let targetX = clientX + 14;
    let targetY = clientY + 14;
    if (targetX + 330 > vpW) {
        targetX = Math.max(pad, clientX - 330);
    }
    if (targetY + 120 > vpH) {
        targetY = Math.max(pad, clientY - 100);
    }

    _hadesTooltipEl.style.left = `${targetX}px`;
    _hadesTooltipEl.style.top = `${targetY}px`;
    requestAnimationFrame(() => {
        if (_hadesTooltipEl) {
            _hadesTooltipEl.style.opacity = "1";
            _hadesTooltipEl.style.transform = "translateY(0)";
        }
    });
}

export function hideHadesTooltip() {
    if (_hadesTooltipEl && _hadesTooltipEl.style.display !== "none") {
        _hadesTooltipEl.style.opacity = "0";
        _hadesTooltipEl.style.transform = "translateY(4px)";
        setTimeout(() => {
            if (_hadesTooltipEl && _hadesTooltipEl.style.opacity === "0") {
                _hadesTooltipEl.style.display = "none";
            }
        }, 120);
    }
}

// =============================================================================
// CLASSIC COMFYUI THEME / VANILLA TOGGLE ENGINE
// =============================================================================
// The HaDeS.MOSS.ClassicUI setting calls setGlobalClassicUI with its stored value when it is registered, before any node exists
export function isNodeClassicUI(node) {
    if (window._hadesMossGlobalClassicUI) return true;
    return Boolean(node?.flags?.classic_ui);
}

export function setNodeClassicUI(node, enableClassic, skipResize = false, isUserOverride = false) {
    if (!node) return;
    
    // Only persist on node.flags if explicitly triggered by user context menu on this node
    if (isUserOverride) {
        node.flags = node.flags || {};
        if (enableClassic) {
            node.flags.classic_ui = true;
        } else {
            delete node.flags.classic_ui;
        }
    }

    const isClassic = Boolean(enableClassic);

    // Save previous spectral size before shrinking to classic
    if (isClassic && !node._hadesSavedSpectralSize && node.size) {
        node._hadesSavedSpectralSize = [node.size[0], node.size[1]];
    }

    // 1. Toggle custom widgets (banners, styled combos, sliders)
    for (const cw of node._hadesCustomWidgets || []) {
        if (isClassic) {
            cw.hidden = true;
            if (cw.computeSize && !cw._origComputeSize) cw._origComputeSize = cw.computeSize;
            if (cw.draw && !cw._origDraw) cw._origDraw = cw.draw;
            cw.computeSize = () => [0, -4];
            cw.draw = () => {};
        } else {
            cw.hidden = false;
            if (cw._origComputeSize) cw.computeSize = cw._origComputeSize;
            if (cw._origDraw) cw.draw = cw._origDraw;
        }
    }

    // 2. Toggle backing widgets (standard ComfyUI inputs)
    for (const w of node.widgets || []) {
        if (w._isHadesBacking) {
            if (isClassic) {
                w.hidden = false;
                if (w._origComputeSize) w.computeSize = w._origComputeSize;
                else delete w.computeSize;
                if (w._origDraw) w.draw = w._origDraw;
                else delete w.draw;
            } else {
                w.hidden = true;
                w.computeSize = () => [0, -4];
                w.draw = () => {};
            }
        }
    }

    // 3. Node frame styling
    if (isClassic) {
        node.color = node._defaultHadesColor || undefined;
        node.bgcolor = node._defaultHadesBgcolor || undefined;
    } else {
        node.color = FRAME_COLOR;
        node.bgcolor = HADES_THEME.bgNode;
    }

    // 4. Resize and repaint
    if (!skipResize) {
        const sz = node.computeSize();
        if (isClassic) {
            node.setSize([Math.max(sz[0], 240), sz[1] + 6]);
        } else {
            const targetW = node._hadesSavedSpectralSize ? Math.max(node._hadesSavedSpectralSize[0], sz[0]) : Math.max(sz[0], 260);
            const targetH = node._hadesSavedSpectralSize ? Math.max(node._hadesSavedSpectralSize[1], sz[1] + 8) : sz[1] + 8;
            node.setSize([targetW, targetH]);
            node._hadesSavedSpectralSize = null;
        }
        node.setDirtyCanvas?.(true, true);
        app?.canvas?.setDirty?.(true, true);
    }
}

export function setGlobalClassicUI(enableGlobalClassic) {
    window._hadesMossGlobalClassicUI = Boolean(enableGlobalClassic);

    const graph = app?.graph || window?.app?.graph;
    if (graph?._nodes) {
        for (const n of graph._nodes) {
            if (n._hadesCustomWidgets || n._hasHadesFrameStyle || (n.type && n.type.startsWith("HaDeSMoss"))) {
                if (!enableGlobalClassic && n.flags?.classic_ui) {
                    delete n.flags.classic_ui;
                }
                const shouldBeClassic = enableGlobalClassic ? true : Boolean(n.flags?.classic_ui);
                setNodeClassicUI(n, shouldBeClassic, false, false);
            }
        }
    }
    app?.canvas?.setDirty?.(true, true);
    graph?.setDirtyCanvas?.(true, true);
}

export function hideBackingWidget(widget) {
    // A second call would save the hidden stubs as the originals and break the vanilla toggle
    if (!widget || widget._isHadesBacking) return;
    if (widget.computeSize && !widget._origComputeSize) widget._origComputeSize = widget.computeSize;
    if (widget.draw && !widget._origDraw) widget._origDraw = widget.draw;
    widget._isHadesBacking = true;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
}

const HADES_ICON_IMG = new Image();
HADES_ICON_IMG.src = new URL("./icon.png", import.meta.url).href;
HADES_ICON_IMG.onload = () => {
    app?.canvas?.setDirty?.(true, true);
};

export function enableNodeHeaderIcon(node, options = {}) {
    if (!node) return;
    if (!node._default_hades_title) {
        node._default_hades_title = node.title || node.constructor?.title || "";
    }
    const stripPrefix = options.stripPrefix ?? true;

    // 1. Right-click context menu option to show/hide icon on demand
    const origGetExtraMenuOptions = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, menuOptions) {
        if (typeof origGetExtraMenuOptions === "function") {
            origGetExtraMenuOptions.call(this, canvas, menuOptions);
        }
        const currentFlags = this.flags || {};
        const defTitle = this._default_hades_title || this.constructor?.title || "";
        const curTitle = this.getTitle?.() ?? this.title ?? "";
        const isDefault = (
            curTitle.replace(/^HaDeS\s+/i, "").trim() === defTitle.replace(/^HaDeS\s+/i, "").trim()
        );
        const isVisible = currentFlags.show_hades_icon !== undefined
            ? Boolean(currentFlags.show_hades_icon)
            : isDefault;

        if (Array.isArray(menuOptions)) {
            const isClassic = isNodeClassicUI(this);
            const subOptions = [
                {
                    content: isClassic ? "Use HaDeS widgets" : "Use standard ComfyUI widgets",
                    callback: () => {
                        setNodeClassicUI(this, !isClassic, false, true);
                    },
                },
                {
                    content: isVisible ? "Hide Header Icon" : "Show Header Icon",
                    callback: () => {
                        this.flags = this.flags || {};
                        this.flags.show_hades_icon = !isVisible;
                        this.setDirtyCanvas?.(true, true);
                    },
                },
            ];

            menuOptions.push({
                content: "HaDeS MOSS Suite",
                has_submenu: true,
                submenu: {
                    options: subOptions,
                },
                callback: (value, options, event, parentMenu) => {
                    if (parentMenu && parentMenu.current_submenu) return;
                    const ContextMenu = window.LiteGraph?.ContextMenu || parentMenu?.constructor;
                    if (ContextMenu) {
                        new ContextMenu(subOptions, { event, parentMenu });
                    }
                },
            });
        }
    };

    // 2. Custom header text & icon rendering
    const origOnDrawTitleText = node.onDrawTitleText;
    node.onDrawTitleText = function (ctx, title_height, size, scale, titleFontStyle, selected) {
        if (isNodeClassicUI(this)) {
            if (typeof origOnDrawTitleText === "function") {
                origOnDrawTitleText.call(this, ctx, title_height, size, scale, titleFontStyle, selected);
                return;
            }
            const h = title_height || 30;
            ctx.font = titleFontStyle || this.titleFontStyle || "600 12px Inter, system-ui, sans-serif";
            ctx.fillStyle = selected ? "#ffffff" : (this.constructor.title_text_color || "#e5e7eb");
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(this.title || this.type, 10, Math.round(-h * 0.5));
            return;
        }

        if (typeof origOnDrawTitleText === "function") {
            origOnDrawTitleText.call(this, ctx, title_height, size, scale, titleFontStyle, selected);
            return;
        }

        ctx.font = titleFontStyle || this.titleFontStyle || "600 12px Inter, system-ui, sans-serif";
        let rawTitle = this.getTitle?.() ?? this.title ?? `❌ ${this.type}`;
        if (stripPrefix) {
            rawTitle = String(rawTitle).replace(/^HaDeS\s+/i, "");
        }
        if (this.pinned) rawTitle += " 📌";

        const defTitle = this._default_hades_title || this.constructor?.title || "";
        const isDefaultName = (
            rawTitle.trim() === defTitle.replace(/^HaDeS\s+/i, "").trim()
            || rawTitle.trim() === defTitle.trim()
        );

        const currentFlags = this.flags || {};
        const showIcon = currentFlags.show_hades_icon !== undefined
            ? Boolean(currentFlags.show_hades_icon)
            : isDefaultName;

        const h = title_height || 30;
        const iconSize = Math.max(14, Math.min(20, Math.round(h * 0.6)));
        const iconY = Math.round(-h + (h - iconSize) * 0.5);
        const iconX = Math.round(h * 0.75);

        let textX = iconX;
        if (showIcon) {
            if (HADES_ICON_IMG.complete && HADES_ICON_IMG.naturalWidth > 0) {
                ctx.drawImage(HADES_ICON_IMG, iconX, iconY, iconSize, iconSize);
            }
            textX = iconX + iconSize + 6;
        }

        const maxTextW = size[0] - textX - 10;
        let displayTitle = rawTitle;

        if (this.collapsed) {
            displayTitle = rawTitle.substring(0, 18);
        } else if (maxTextW > 20 && ctx.measureText(displayTitle).width > maxTextW) {
            while (displayTitle.length > 2 && ctx.measureText(displayTitle + "…").width > maxTextW) {
                displayTitle = displayTitle.slice(0, -1);
            }
            displayTitle += "…";
        }

        ctx.fillStyle = selected
            ? "#ffffff"
            : (this.constructor.title_text_color || "#e5e7eb");
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(displayTitle, textX, Math.round(-h * 0.5));
    };
}

export function applyNodeFrameStyle(node) {
    if (!node) return;
    node._hasHadesFrameStyle = true;
    if (!node._defaultHadesColor && node.color !== FRAME_COLOR) {
        node._defaultHadesColor = node.color;
    }
    if (!node._defaultHadesBgcolor && node.bgcolor !== HADES_THEME.bgNode) {
        node._defaultHadesBgcolor = node.bgcolor;
    }

    if (!node._hadesCustomWidgetHooked) {
        node._hadesCustomWidgetHooked = true;
        const origAddCustomWidget = node.addCustomWidget;
        node.addCustomWidget = function (widget) {
            if (widget) {
                widget._isHadesCustom = true;
                if (widget.draw) wrapVueWidgetDraw(widget);
                if (!widget._origComputeSize) widget._origComputeSize = widget.computeSize;
                if (!widget._origDraw) widget._origDraw = widget.draw;
                this._hadesCustomWidgets = this._hadesCustomWidgets || [];
                this._hadesCustomWidgets.push(widget);
            }
            return origAddCustomWidget.apply(this, arguments);
        };
        const origSetDirtyCanvas = node.setDirtyCanvas;
        node.setDirtyCanvas = function () {
            redrawVueWidgets(this);
            return origSetDirtyCanvas.apply(this, arguments);
        };
        // A link dropped on a styled row targets its hidden backing widget's input, not the first input of that type
        const origGetSlotFromWidget = node.getSlotFromWidget;
        node.getSlotFromWidget = function (widget) {
            return origGetSlotFromWidget.call(this, widget?._hadesBackingWidget ?? widget);
        };
        // Hidden backing widgets are skipped by the layout, so their input sockets sat at the top of the node.
        // Give them their styled row's position (centered on the 24px row surface) and re-place the sockets.
        const origArrange = node.arrange;
        node.arrange = function () {
            origArrange.apply(this, arguments);
            let moved = false;
            for (const w of this.widgets || []) {
                const backing = w._hadesBackingWidget;
                if (!backing || w.y == null || backing.y === w.y + 4) continue;
                backing.y = w.y + 4;
                moved = true;
            }
            if (moved) origArrange.apply(this, arguments);
        };
    }

    node.color = FRAME_COLOR;
    node.bgcolor = HADES_THEME.bgNode;
    enableNodeHeaderIcon(node);

    if (isNodeClassicUI(node)) {
        setTimeout(() => {
            setNodeClassicUI(node, true);
        }, 1);
    }

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (typeof origOnConfigure === "function") origOnConfigure.apply(this, arguments);
        if (isNodeClassicUI(this)) {
            setNodeClassicUI(this, true);
        }
    };
}

// =============================================================================
// REUSABLE HADES CANVAS WIDGET COMPONENTS
// =============================================================================

export function createSpectralBannerWidget(node, config) {
    const {
        title,
        accentColor = HADES_THEME.blue,
        getStatusText = null,
        getStatusColor = null,
        getBars = null,
        drawCustomHUD = null,
        customHeight = null,
        actions = [],
        tooltip = config.tooltip || "",
    } = config;

    const hasGraph = typeof getBars === "function" || typeof drawCustomHUD === "function";
    const widgetHeight = customHeight || (hasGraph ? 62 : 28);

    return {
        name: `hades_banner_${title}`,
        type: "custom",
        value: "",
        tooltip,
        accentColor,
        options: { serialize: false },
        serialize: false,
        _btnBounds: [],
        _pillBounds: [],
        _hudBounds: null,
        _bannerBounds: null,

        computeSize(width) {
            return [width, widgetHeight];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = widgetHeight - 4;
            const topY = y + 2;

            this._bannerBounds = { x, y: topY, w, h };

            ctx.save();

            drawRoundedRect(ctx, x, topY, w, h, 6);
            ctx.fillStyle = HADES_THEME.bgPanel;
            ctx.fill();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.lineWidth = 1;
            ctx.stroke();

            const labelText = getStatusText ? getStatusText() : title;
            if (labelText) {
                const ledColor = getStatusColor ? getStatusColor() : accentColor;
                ctx.beginPath();
                ctx.arc(x + 11, topY + 12, 3, 0, Math.PI * 2);
                ctx.fillStyle = ledColor;
                ctx.fill();

                ctx.font = HADES_FONT.small;
                ctx.fillStyle = HADES_THEME.textSecondary;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(labelText, x + 20, topY + 12);
            }

            const btnH = 16;
            const btnGap = 4;
            this._btnBounds = [];

            ctx.font = HADES_FONT.small;
            ctx.textAlign = "center";

            if (!labelText && actions.length > 0) {
                const maxAvailableW = w - 8;
                const isNarrow = w < 280;
                const btnGap = isNarrow ? 2 : 4;
                ctx.font = isNarrow ? "500 9px Inter, system-ui, sans-serif" : HADES_FONT.small;

                let naturalTotal = 0;
                const naturalWidths = actions.map((act) => {
                    const baseW = Math.round(ctx.measureText(act.label).width + (isNarrow ? 6 : 8));
                    const nw = Math.max(isNarrow ? 22 : 28, baseW);
                    naturalTotal += nw;
                    return nw;
                });

                const totalGaps = (actions.length - 1) * btnGap;
                const availableForButtons = maxAvailableW - totalGaps;
                const scale = naturalTotal > availableForButtons ? availableForButtons / naturalTotal : 1.0;

                let cursorLeft = x + 4;
                for (let i = 0; i < actions.length; i++) {
                    const act = actions[i];
                    const btnW = Math.max(20, Math.floor(naturalWidths[i] * scale));
                    const bx = cursorLeft;
                    const by = topY + 4;

                    const isActive = Boolean(act.isActive?.(n));
                    drawRoundedRect(ctx, bx, by, btnW, btnH, 3);
                    ctx.fillStyle = isActive ? `${HADES_THEME.accent}26` : HADES_THEME.bgControl;
                    ctx.fill();
                    ctx.strokeStyle = isActive ? HADES_THEME.borderActive : HADES_THEME.borderSubtle;
                    ctx.stroke();

                    let lbl = act.label;
                    if (ctx.measureText(lbl).width > btnW - 3) {
                        if (lbl === "Base/Rae") lbl = "Base";
                        else if (lbl === "Aesthetic") lbl = "Aesth";
                        else if (lbl === "Beta57") lbl = "B57";
                        else if (lbl === "Turbo") lbl = "Turb";
                        else if (lbl === "TDM 4s") lbl = "4s";
                        else if (lbl === "HiRes 1.5x") lbl = "1.5x";
                        else if (lbl === "Painterly") lbl = "Paint";
                        else if (lbl === "Base→Turbo") lbl = "B→T";
                        else if (lbl === "1.0x Polish") lbl = "1.0x";
                        else if (lbl === "Safe 1.75x") lbl = "Safe";
                        else if (lbl === "1.35→1.75x") lbl = "1.75x";
                        else if (lbl === "1.5→2.0x") lbl = "2.0x";
                        else if (lbl === "1.5x Lock") lbl = "Lock";
                        else if (lbl === "Turbo S3") lbl = "Turbo";
                        else if (lbl === "BF16 Master") lbl = "BF16";
                        else if (lbl === "ConvRot INT8") lbl = btnW < 52 ? "INT8" : "CR-INT8";
                        else if (lbl === "ConvRot INT4") lbl = btnW < 52 ? "INT4" : "CR-INT4";
                        else if (lbl === "Diffusers HF") lbl = btnW < 54 ? "HF" : "Diffusers";
                    }

                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(bx + 1, by, btnW - 2, btnH);
                    ctx.clip();
                    ctx.fillStyle = isActive ? HADES_THEME.textPrimary : HADES_THEME.textSecondary;
                    ctx.fillText(lbl, bx + btnW * 0.5, by + btnH * 0.5 + 0.5);
                    ctx.restore();

                    this._btnBounds.push({ x: bx, y: by, w: btnW, h: btnH, onClick: act.onClick, tooltip: act.tooltip, color: act.color });
                    cursorLeft = bx + btnW + btnGap;
                }
            } else {
                let cursorRight = x + w - 5;
                for (let i = actions.length - 1; i >= 0; i--) {
                    const act = actions[i];
                    const btnW = Math.max(32, ctx.measureText(act.label).width + 10);
                    const bx = cursorRight - btnW;
                    const by = topY + 4;

                    const isActive = Boolean(act.isActive?.(n));
                    drawRoundedRect(ctx, bx, by, btnW, btnH, 3);
                    ctx.fillStyle = isActive ? `${HADES_THEME.accent}26` : HADES_THEME.bgControl;
                    ctx.fill();
                    ctx.strokeStyle = isActive ? HADES_THEME.borderActive : HADES_THEME.borderSubtle;
                    ctx.stroke();
                    ctx.fillStyle = isActive ? HADES_THEME.textPrimary : HADES_THEME.textSecondary;
                    ctx.fillText(act.label, bx + btnW * 0.5, by + btnH * 0.5 + 0.5);

                    this._btnBounds.push({ x: bx, y: by, w: btnW, h: btnH, onClick: act.onClick, tooltip: act.tooltip, color: act.color });
                    cursorRight = bx - btnGap;
                }
            }

            const hasTopRow = Boolean(labelText) || actions.length > 0;
            if (hasGraph) {
                const graphX = x + 6;
                const graphY = hasTopRow ? topY + 24 : topY + 4;
                const graphW = w - 12;
                const graphH = hasTopRow ? h - 28 : h - 8;

                drawRoundedRect(ctx, graphX, graphY, graphW, graphH, 4);
                ctx.fillStyle = HADES_THEME.bgControl;
                ctx.fill();

                this._hudBounds = { x: graphX, y: graphY, w: graphW, h: graphH };
                this._pillBounds = [];
                if (typeof drawCustomHUD === "function") {
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(graphX, graphY, graphW, graphH);
                    ctx.clip();
                    drawCustomHUD(ctx, { x: graphX, y: graphY, w: graphW, h: graphH }, n, this._pillBounds);
                    ctx.restore();
                } else if (typeof getBars === "function") {
                    const bars = getBars() || [];
                    const midY = graphY + graphH * 0.5;
                    ctx.strokeStyle = HADES_THEME.borderSubtle;
                    ctx.beginPath();
                    ctx.moveTo(graphX + 2, midY);
                    ctx.lineTo(graphX + graphW - 2, midY);
                    ctx.stroke();

                    if (bars.length > 0) {
                        const pad = 4;
                        const availW = graphW - pad * 2;
                        const stepW = availW / bars.length;
                        const barW = Math.max(2, stepW - 1.5);

                        for (let i = 0; i < bars.length; i++) {
                            const b = bars[i];
                            const norm = Math.max(0, Math.min(1, Number(b.ratio) || 0));
                            const bh = Math.max(2, norm * (graphH - 6));
                            const bx = graphX + pad + i * stepW;
                            const by = graphY + graphH - 3 - bh;

                            ctx.fillStyle = b.color || accentColor;
                            ctx.fillRect(bx, by, barW, bh);
                        }
                    }
                }
            }

            ctx.restore();
        },

        mouse(event, pos, n) {
            if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
            const [mx, my] = pos;
            for (const btn of this._btnBounds) {
                if (mx >= btn.x && mx <= btn.x + btn.w && my >= btn.y && my <= btn.y + btn.h) {
                    btn.onClick?.(n);
                    n.setDirtyCanvas(true, true);
                    return true;
                }
            }
            return false;
        },
    };
}

export function createStyledComboWidget(node, backingWidget, config = {}) {
    const {
        tag = "MODE",
        tagColor = config.color || HADES_THEME.purple,
        tagWidth = null,
        label = "",
        cleanFileNames = false,
        marker = null,
        // stored value -> shown text; the backing widget keeps the stored value so workflows stay compatible
        displayNames = null,
        isDisabled = null,
        isDimmed = null, // drawn faded but still editable (e.g. a value another input overrides)
        tooltip = config.tooltip || backingWidget?.tooltip || backingWidget?.options?.tooltip || "",
    } = config;

    const plainTooltip = typeof tooltip === "string" ? tooltip.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "";

    return {
        name: `hades_combo_${backingWidget?.name || label}`,
        _hadesBackingWidget: backingWidget,
        type: "custom",
        value: "",
        tooltip: plainTooltip,
        tooltipHtml: tooltip,
        options: { serialize: false, tooltip: plainTooltip },
        serialize: false,
        _bounds: null,
        _isHovered: false,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;

            ctx.save();
            if (isDisabled?.() || isDimmed?.()) ctx.globalAlpha = 0.45;
            drawRowSurface(ctx, x, rowY, w, h, this._isHovered);
            ctx.font = HADES_FONT.label;
            const tagW = tagWidth || Math.max(52, Math.round(ctx.measureText(tag).width + 18));
            drawRowLabel(ctx, x, rowY, h, tagW, tag, marker);
            ctx.textBaseline = "middle";

            const rawVal = backingWidget?.value ?? "";
            const valText = cleanFileNames ? formatCleanLabel(rawVal) : String(displayNames?.[rawVal] ?? rawVal);

            const textStartX = x + tagW + 10;
            const textMaxW = w - tagW - 26;

            ctx.save();
            ctx.beginPath();
            ctx.rect(textStartX, rowY, textMaxW, h);
            ctx.clip();

            ctx.font = HADES_FONT.text;
            ctx.textAlign = "left";
            if (label) {
                ctx.fillStyle = HADES_THEME.textSecondary;
                ctx.fillText(`${label}: `, textStartX, rowY + h * 0.5 + 0.5);
                const lw = ctx.measureText(`${label}: `).width;
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(valText, textStartX + lw, rowY + h * 0.5 + 0.5);
            } else {
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(valText, textStartX, rowY + h * 0.5 + 0.5);
            }
            ctx.restore();

            ctx.fillStyle = HADES_THEME.textMuted;
            ctx.font = HADES_FONT.small;
            ctx.textAlign = "center";
            ctx.fillText("▾", x + w - 10, rowY + h * 0.5 + 0.5);

            this._bounds = { x, y: rowY, w, h };
            ctx.restore();
        },

        mouse(event, pos, n) {
            if (!this._bounds || !backingWidget) {
                return false;
            }
            const [mx, my] = pos;
            const b = this._bounds;
            const isInside = mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h;

            if (event.type === "pointermove" || event.type === "mousemove") {
                if (isInside && tooltip) {
                    this._isHovered = true;
                    showHadesTooltip(tooltip, event.clientX, event.clientY, tagColor);
                } else if (this._isHovered) {
                    this._isHovered = false;
                    hideHadesTooltip();
                }
                return false;
            }

            if (event.type === "pointerleave" || event.type === "mouseleave") {
                if (this._isHovered) {
                    this._isHovered = false;
                    hideHadesTooltip();
                }
                return false;
            }

            if ((event.type === "pointerdown" || event.type === "mousedown") && isInside) {
                hideHadesTooltip();
                this._isHovered = false;
                if (isDisabled?.()) return true;
                const values = backingWidget.options?.values || [];
                const entries = displayNames ? values.map((v) => ({ content: displayNames[v] ?? v, value: v })) : values;
                new LiteGraph.ContextMenu(entries, {
                    event,
                    title: label || backingWidget.name,
                    className: "dark",
                    scale: Math.max(1, app.canvas?.ds?.scale || 1),
                    callback: (selected) => {
                        if (selected !== undefined) {
                            setWidgetValue(backingWidget, displayNames ? selected.value : selected, n);
                        }
                    },
                });
                return true;
            }
            return false;
        },
    };
}

export function createSpectralSliderWidget(node, backingWidget, config = {}) {
    const {
        tag = "VAL",
        rangeTag = config.range || "",
        label = backingWidget?.name || "",
        color = HADES_THEME.blue,
        tagWidth = null,
        min = -10.0,
        max = 10.0,
        visMin = config.min ?? 0.0,
        visMax = config.max ?? 1.0,
        step = 0.05,
        defaultVal = config.defVal ?? 0.5,
        decimals = 2,
        isInt = false,
        zeroLabel = null,
        marker = null,
        isDimmed = null, // drawn faded but still editable (e.g. a value another input overrides)
        tooltip = config.tooltip || backingWidget?.tooltip || backingWidget?.options?.tooltip || "",
    } = config;

    const plainTooltip = typeof tooltip === "string" ? tooltip.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "";

    return {
        name: `hades_slider_${backingWidget?.name || label}`,
        _hadesBackingWidget: backingWidget,
        type: "custom",
        value: "",
        tooltip: plainTooltip,
        tooltipHtml: tooltip,
        options: { serialize: false, tooltip: plainTooltip },
        serialize: false,
        _zones: null,
        _rowBounds: null,
        _isHovered: false,
        _dragStartX: null,
        _dragStartVal: null,
        _didDrag: false,
        _dragRail: false,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;

            const curVal = clampNum(backingWidget?.value ?? defaultVal, min, max, isInt ? 0 : decimals);

            ctx.save();
            if (isDimmed?.()) ctx.globalAlpha = 0.45;
            drawRowSurface(ctx, x, rowY, w, h, this._isHovered);

            const isNarrow = w < 280;
            const hasLabel = Boolean(label);
            ctx.font = HADES_FONT.label;
            const tagW = tagWidth || Math.max(isNarrow ? 44 : 52, Math.round(ctx.measureText(rangeTag || tag).width + 18));
            drawRowLabel(ctx, x, rowY, h, tagW, rangeTag || tag, marker);
            ctx.textBaseline = "middle";

            // Right numeric box [ −  val  + ]
            const ctrlW = hasLabel ? (isNarrow ? 96 : 114) : (isNarrow ? 74 : 82);
            const ctrlX = x + w - ctrlW - 3;
            const ctrlY = rowY + 3;
            const ctrlH = h - 6;
            const btnW = isNarrow ? 15 : 17;

            drawRoundedRect(ctx, ctrlX, ctrlY, ctrlW, ctrlH, 3);
            ctx.fillStyle = HADES_THEME.bgControl;
            ctx.fill();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.stroke();

            const trackX = ctrlX + btnW;
            const trackW = ctrlW - btnW * 2;
            const norm = Math.max(0, Math.min(1, (curVal - visMin) / (visMax - visMin || 1)));


            ctx.font = HADES_FONT.text;
            ctx.textAlign = "center";
            ctx.fillStyle = HADES_THEME.textMuted;
            ctx.fillText("−", ctrlX + btnW * 0.5, ctrlY + ctrlH * 0.5 + 0.5);
            ctx.fillText("+", ctrlX + ctrlW - btnW * 0.5, ctrlY + ctrlH * 0.5 + 0.5);

            ctx.font = HADES_FONT.value;
            ctx.fillStyle = HADES_THEME.textPrimary;
            const valStr = zeroLabel && curVal === 0 ? zeroLabel : (isInt ? String(Math.round(curVal)) : curVal.toFixed(decimals));
            ctx.fillText(valStr, trackX + trackW * 0.5, ctrlY + ctrlH * 0.5 + 0.5);

            let railStartX = 0;
            let railW = 0;

            if (hasLabel) {
                const labelX = x + tagW + 7;
                const labelMaxW = ctrlX - labelX - 5;
                ctx.save();
                ctx.beginPath();
                ctx.rect(labelX, rowY, Math.max(0, labelMaxW), h);
                ctx.clip();
                ctx.font = HADES_FONT.text;
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.textAlign = "left";

                let displayLabel = label;
                if (labelMaxW > 12 && ctx.measureText(displayLabel).width > labelMaxW) {
                    if (displayLabel.includes("ModelSpec")) {
                        displayLabel = "ModelSpec Header";
                        if (ctx.measureText(displayLabel).width > labelMaxW) displayLabel = "ModelSpec";
                    }
                    if (ctx.measureText(displayLabel).width > labelMaxW) {
                        while (displayLabel.length > 2 && ctx.measureText(displayLabel + "…").width > labelMaxW) {
                            displayLabel = displayLabel.slice(0, -1);
                        }
                        displayLabel += "…";
                    }
                }
                ctx.fillText(displayLabel, labelX, rowY + h * 0.5 + 0.5);
                ctx.restore();
            } else {
                // Precision Rail in center
                railStartX = x + tagW + 8;
                const railEndX = ctrlX - 8;
                railW = railEndX - railStartX;

                if (railW > 14) {
                    const railH = 3;
                    const railY = rowY + Math.round((h - railH) * 0.5);

                    // Track background rail
                    drawRoundedRect(ctx, railStartX, railY, railW, railH, 1.5);
                    ctx.fillStyle = HADES_THEME.borderSubtle;
                    ctx.fill();

                    // Active progress fill
                    const fillRailW = norm * railW;
                    if (fillRailW > 0.5) {
                        drawRoundedRect(ctx, railStartX, railY, fillRailW, railH, 1.5);
                        ctx.fillStyle = HADES_THEME.accent;
                        ctx.fill();
                    }

                    // Precision thumb indicator
                    const thumbX = Math.round(railStartX + fillRailW);
                    const thumbY = rowY + h * 0.5;
                    ctx.beginPath();
                    ctx.arc(thumbX, thumbY, 4.5, 0, Math.PI * 2);
                    ctx.fillStyle = HADES_THEME.textPrimary;
                    ctx.fill();
                    ctx.strokeStyle = HADES_THEME.bgControl;
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            }

            this._rowBounds = { x, y: rowY, w, h };
            this._zones = {
                rowY,
                h,
                hasRail: !hasLabel && railW > 14,
                rail: (!hasLabel && railW > 14) ? { x: railStartX, w: railW } : null,
                dec: { x: ctrlX, w: btnW },
                numBox: { x: trackX, w: trackW },
                track: (!hasLabel && railW > 14) ? { x: railStartX, w: railW } : { x: trackX, w: trackW },
                inc: { x: ctrlX + ctrlW - btnW, w: btnW },
            };

            ctx.restore();
        },

        mouse(event, pos, n) {
            if (!this._zones || !backingWidget) return false;
            const [mx, my] = pos;
            const z = this._zones;
            const rb = this._rowBounds;
            const isInside = rb ? (mx >= rb.x && mx <= rb.x + rb.w && my >= rb.y && my <= rb.y + rb.h) : false;

            if (event.type === "pointerdown" || event.type === "mousedown") {
                hideHadesTooltip();
                this._isHovered = false;
                if (my < z.rowY || my > z.rowY + z.h) return false;

                if (event.button === 2) {
                    setWidgetValue(backingWidget, defaultVal, n);
                    return true;
                }

                if (mx >= z.dec.x && mx <= z.dec.x + z.dec.w) {
                    const next = clampNum(Number(backingWidget.value) - step, min, max, isInt ? 0 : decimals);
                    setWidgetValue(backingWidget, next, n);
                    return true;
                }

                if (mx >= z.inc.x && mx <= z.inc.x + z.inc.w) {
                    const next = clampNum(Number(backingWidget.value) + step, min, max, isInt ? 0 : decimals);
                    setWidgetValue(backingWidget, next, n);
                    return true;
                }

                if (z.hasRail && z.rail && mx >= z.rail.x - 3 && mx <= z.rail.x + z.rail.w + 3) {
                    this._dragRail = true;
                    const ratio = Math.max(0, Math.min(1, (mx - z.rail.x) / z.rail.w));
                    const rawVal = visMin + ratio * (visMax - visMin);
                    const stepped = snapToStep(rawVal, min, step);
                    const next = clampNum(stepped, min, max, isInt ? 0 : decimals);
                    setWidgetValue(backingWidget, next, n);
                    return true;
                }

                if (mx >= z.track.x && mx <= z.track.x + z.track.w) {
                    this._dragStartX = mx;
                    this._dragStartVal = Number(backingWidget.value) ?? defaultVal;
                    this._didDrag = false;
                    return true;
                }
            } else if (event.type === "pointermove" || event.type === "mousemove") {
                if (isInside && tooltip && !this._dragStartX && !this._dragRail) {
                    this._isHovered = true;
                    showHadesTooltip(tooltip, event.clientX, event.clientY, color);
                } else if (this._isHovered && (!isInside || this._dragStartX || this._dragRail)) {
                    this._isHovered = false;
                    hideHadesTooltip();
                }

                if (this._dragRail && z?.rail) {
                    const ratio = Math.max(0, Math.min(1, (mx - z.rail.x) / z.rail.w));
                    const rawVal = visMin + ratio * (visMax - visMin);
                    const stepped = snapToStep(rawVal, min, step);
                    const next = clampNum(stepped, min, max, isInt ? 0 : decimals);
                    setWidgetValue(backingWidget, next, n);
                    return true;
                }

                if (this._dragStartX !== null) {
                    const dx = mx - this._dragStartX;
                    if (Math.abs(dx) > 2) {
                        this._didDrag = true;
                        const sensitivity = isInt ? (visMax - visMin) / 160 : (visMax - visMin) / 140;
                        const rawNext = this._dragStartVal + dx * sensitivity;
                        const stepped = snapToStep(rawNext, min, step);
                        const next = clampNum(stepped, min, max, isInt ? 0 : decimals);
                        setWidgetValue(backingWidget, next, n);
                    }
                    return true;
                }
            } else if (event.type === "pointerleave" || event.type === "mouseleave") {
                if (this._isHovered) {
                    this._isHovered = false;
                    hideHadesTooltip();
                }
                return false;
            } else if (event.type === "pointerup" || event.type === "mouseup") {
                this._dragRail = false;
                if (this._dragStartX !== null) {
                    const wasDrag = this._didDrag;
                    this._dragStartX = null;
                    this._dragStartVal = null;
                    this._didDrag = false;

                    if (!wasDrag && app.canvas) {
                        app.canvas.prompt(
                            `${rangeTag || tag} (${min} to ${max})`,
                            String(backingWidget.value),
                            (val) => {
                                const parsed = isInt ? parseInt(val, 10) : parseFloat(val);
                                if (!Number.isNaN(parsed)) {
                                    const next = clampNum(parsed, min, max, isInt ? 0 : decimals);
                                    setWidgetValue(backingWidget, next, n);
                                }
                            },
                            event
                        );
                    }
                    return true;
                }
            }
            return false;
        },
    };
}

export function createStyledTextRowWidget(node, backingWidget, config = {}) {
    const {
        tag = "FILE",
        tagWidth = null,
        color = HADES_THEME.emerald,
        promptTitle = "Enter Value",
        suffix = "",
        emptyPlaceholder = "— None —",
        numeric = false,
        onEdit = null,
        displayValue = null,
        isDisabled = null,
        isDimmed = null, // drawn faded but still editable (e.g. a value another input overrides)
        disabledText = "",
        tooltip = config.tooltip || backingWidget?.tooltip || backingWidget?.options?.tooltip || "",
    } = config;

    const plainTooltip = typeof tooltip === "string" ? tooltip.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "";

    return {
        name: `hades_text_${backingWidget?.name || tag}`,
        _hadesBackingWidget: backingWidget,
        type: "custom",
        value: "",
        tooltip: plainTooltip,
        tooltipHtml: tooltip,
        tagColor: color,
        options: { serialize: false, tooltip: plainTooltip },
        serialize: false,
        _bounds: null,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;
            const disabled = Boolean(isDisabled?.());

            ctx.save();
            if (disabled || isDimmed?.()) ctx.globalAlpha = 0.45;
            drawRowSurface(ctx, x, rowY, w, h);

            const isNarrow = w < 280;
            const tagW = tagWidth || (isNarrow ? 48 : 56);
            drawRowLabel(ctx, x, rowY, h, tagW, tag);
            ctx.textBaseline = "middle";

            let sfxW = 0;
            if (suffix) {
                ctx.font = HADES_FONT.small;
                sfxW = ctx.measureText(suffix).width + 10;
                ctx.fillStyle = HADES_THEME.textMuted;
                ctx.textAlign = "right";
                ctx.fillText(suffix, x + w - 6, rowY + h * 0.5 + 0.5);
            }

            const rawVal = String(backingWidget?.value ?? "").trim();
            const displayVal = displayValue ? displayValue(rawVal) : suffix ? rawVal.replace(new RegExp(`${suffix}$`, "i"), "") : rawVal;

            ctx.save();
            ctx.beginPath();
            ctx.rect(x + tagW + 8, rowY, w - tagW - sfxW - 14, h);
            ctx.clip();
            const shown = disabled ? disabledText : displayVal;
            ctx.font = shown && !disabled ? HADES_FONT.text : `italic ${HADES_FONT.text}`;
            ctx.fillStyle = shown && !disabled ? HADES_THEME.textPrimary : HADES_THEME.textMuted;
            ctx.textAlign = "left";
            ctx.fillText(shown || emptyPlaceholder, x + tagW + 10, rowY + h * 0.5 + 0.5);
            ctx.restore();

            this._bounds = { x, y: rowY, w, h };
            ctx.restore();
        },

        mouse(event, pos, n) {
            if ((event.type !== "pointerdown" && event.type !== "mousedown") || !this._bounds || !backingWidget) {
                return false;
            }
            const [mx, my] = pos;
            const b = this._bounds;
            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                if (isDisabled?.()) return true;
                if (onEdit) {
                    onEdit(event, n);
                    return true;
                }
                if (app.canvas) {
                    app.canvas.prompt(
                        promptTitle,
                        String(backingWidget.value || ""),
                        (val) => {
                            if (numeric) {
                                const parsed = Number.parseInt(String(val ?? "").trim(), 10);
                                if (!Number.isNaN(parsed)) setWidgetValue(backingWidget, Math.max(0, parsed), n);
                                return;
                            }
                            setWidgetValue(backingWidget, String(val ?? "").trim(), n);
                        },
                        event
                    );
                }
                return true;
            }
            return false;
        },
    };
}

export function createStyledToggleWidget(node, backingWidget, config = {}) {
    const {
        tag = "FLAG",
        tagWidth = null,
        color = HADES_THEME.emerald,
        label = backingWidget?.name || "",
        onText = "ENABLED",
        offText = "DISABLED",
        tooltip = config.tooltip || backingWidget?.tooltip || backingWidget?.options?.tooltip || "",
    } = config;

    const plainTooltip = typeof tooltip === "string" ? tooltip.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "";

    return {
        name: `hades_toggle_${backingWidget?.name || label}`,
        _hadesBackingWidget: backingWidget,
        type: "custom",
        value: "",
        tooltip: plainTooltip,
        tooltipHtml: tooltip,
        options: { serialize: false, tooltip: plainTooltip },
        serialize: false,
        _bounds: null,
        _isHovered: false,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;
            const isOn = Boolean(backingWidget?.value);

            ctx.save();
            drawRowSurface(ctx, x, rowY, w, h, this._isHovered);

            const isNarrow = w < 280;
            ctx.font = HADES_FONT.label;
            const tagW = tagWidth || Math.max(isNarrow ? 44 : 52, Math.round(ctx.measureText(tag).width + 18));
            drawRowLabel(ctx, x, rowY, h, tagW, tag);
            ctx.textBaseline = "middle";

            ctx.font = HADES_FONT.label;
            const currentPillText = isOn ? onText : offText;
            const naturalPillW = Math.round(ctx.measureText(currentPillText).width + (isNarrow ? 10 : 14));
            const pillW = Math.max(isNarrow ? 66 : 76, naturalPillW);
            const pillX = x + w - pillW - 3;
            const pillY = rowY + 3;
            const pillH = h - 6;

            if (label) {
                const labelStartX = x + tagW + (isNarrow ? 6 : 10);
                const labelMaxW = Math.max(0, pillX - labelStartX - 6);

                if (labelMaxW > 8) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(labelStartX, rowY, labelMaxW, h);
                    ctx.clip();

                    ctx.font = HADES_FONT.text;
                    ctx.fillStyle = HADES_THEME.textPrimary;
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";

                    let displayLabel = label;
                    if (ctx.measureText(displayLabel).width > labelMaxW && displayLabel.includes(" (")) {
                        const simplified = displayLabel.replace(/\s*\([^)]*\)/, "");
                        if (ctx.measureText(simplified).width <= labelMaxW) {
                            displayLabel = simplified;
                        }
                    }

                    if (ctx.measureText(displayLabel).width > labelMaxW) {
                        while (displayLabel.length > 2 && ctx.measureText(displayLabel + "...").width > labelMaxW) {
                            displayLabel = displayLabel.slice(0, -1);
                        }
                        displayLabel += "...";
                    }

                    ctx.fillText(displayLabel, labelStartX, rowY + h * 0.5 + 0.5);
                    ctx.restore();
                }
            }

            drawRoundedRect(ctx, pillX, pillY, pillW, pillH, 3);
            ctx.fillStyle = isOn ? `${HADES_THEME.accent}26` : HADES_THEME.bgControl;
            ctx.fill();
            ctx.strokeStyle = isOn ? HADES_THEME.borderActive : HADES_THEME.borderSubtle;
            ctx.stroke();

            ctx.font = HADES_FONT.label;
            ctx.fillStyle = isOn ? HADES_THEME.textPrimary : HADES_THEME.textMuted;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(currentPillText, pillX + pillW * 0.5, pillY + pillH * 0.5 + 0.5);

            this._bounds = { x, y: rowY, w, h };
            ctx.restore();
        },

        mouse(event, pos, n) {
            if (!this._bounds) return false;
            const [mx, my] = pos;
            const b = this._bounds;
            const isInside = mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h;

            if (event.type === "pointermove" || event.type === "mousemove") {
                if (isInside && tooltip) {
                    this._isHovered = true;
                    showHadesTooltip(tooltip, event.clientX, event.clientY, color);
                } else if (this._isHovered) {
                    this._isHovered = false;
                    hideHadesTooltip();
                }
                return false;
            }

            if (event.type === "pointerleave" || event.type === "mouseleave") {
                if (this._isHovered) {
                    this._isHovered = false;
                    hideHadesTooltip();
                }
                return false;
            }

            if ((event.type === "pointerdown" || event.type === "mousedown") && isInside && backingWidget) {
                hideHadesTooltip();
                this._isHovered = false;
                setWidgetValue(backingWidget, !backingWidget.value, n);
                return true;
            }
            return false;
        },
    };
}
