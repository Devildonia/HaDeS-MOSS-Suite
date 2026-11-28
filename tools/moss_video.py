"""Records a workflow walkthrough from a running ComfyUI and edits it with subtitles, for the YouTube explainer videos.

A headless Chromium (Playwright) opens ComfyUI at 1920 x 1080 and plays a scene: it loads an example workflow, moves a drawn cursor,
types, clicks the Suite's canvas buttons and drags boxes, while Chrome's screencast saves every frame with its time. Each step
says its caption (English and Spanish) and waits (a draft, a render) are marked fast; the edit then speeds those up with a small
badge, burns the captions in and writes .srt files in both languages for YouTube's caption tracks.

With --voice en|es each caption is also spoken: before recording, Qwen3-TTS (Base, voice clone from --ref, timbre only) reads every
line of the scene in ComfyUI's Python, where qwen-tts is installed; the scene then waits for each line to end before the next one,
counted in video time, so voices never overlap even across fast parts. Voices are cached by text, language and reference.

    python tools/moss_video.py layout_test --out I:/HaDeS_MOSS/videos --voice en --ref I:/HaDeS_MOSS/videos/voice/ref_norm.wav

Needs Playwright with Chromium and ffmpeg (with libass) on PATH. Talks only to the given ComfyUI URL (default localhost).
Runs on the ComfyUI queue: a scene that drafts or generates uses the GPU like a click in the browser would."""
import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
W, H = 1920, 1080  # the page and the video; 1440p is a 2560 x 1440 page (Chrome's screencast ignores device scale)

# a drawn cursor (headless Chromium shows none) with a ring on each press
CURSOR_JS = r"""
(() => {
  const add = () => {
    if (document.getElementById('__hv_cursor')) return;
    const c = document.createElement('div');
    c.id = '__hv_cursor';
    const k = Math.max(1, innerWidth / 1920);
    c.innerHTML = '<svg width="' + 28 * k + '" height="' + 28 * k + '" viewBox="0 0 28 28"><path d="M3 2 L3 22 L8.5 17 L12.5 26 L16 24.5 L12 15.8 L19.5 15.8 Z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, {position: 'fixed', left: '0', top: '0', zIndex: 2147483647, pointerEvents: 'none',
      transform: 'translate(960px, 540px)', filter: 'drop-shadow(0 2px 3px rgba(0,0,0,.45))'});
    document.body.appendChild(c);
    const ring = (x, y) => {
      const r = document.createElement('div');
      Object.assign(r.style, {position: 'fixed', left: (x - 22 * k) + 'px', top: (y - 22 * k) + 'px', width: 44 * k + 'px', height: 44 * k + 'px',
        borderRadius: '50%', border: '3px solid #ffcc33', zIndex: 2147483646, pointerEvents: 'none', opacity: '0.95',
        transition: 'transform .45s ease-out, opacity .45s ease-out', transform: 'scale(.35)'});
      document.body.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.15)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 600);
    };
    window.addEventListener('pointermove', (e) => { c.style.transform = `translate(${e.clientX - 3}px, ${e.clientY - 2}px)`; }, true);
    window.addEventListener('pointerdown', (e) => ring(e.clientX, e.clientY), true);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""

# helpers in the page: node-local point -> screen, and a smooth camera move that fits some nodes
HELPERS_JS = r"""
window.__hv = {
  app() { return window.comfyAPI?.app?.app || window.app; },
  node(id) { return this.app().graph.getNodeById(id); },
  screen(id, lx, ly) {
    const a = this.app(), n = this.node(id), r = a.canvas.canvas.getBoundingClientRect();
    const p = a.canvas.ds.convertOffsetToCanvas([n.pos[0] + lx, n.pos[1] + ly]);
    return [r.left + p[0], r.top + p[1]];
  },
  rect(id) {
    const n = this.node(id), t = (window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30);
    return [n.pos[0], n.pos[1] - t, n.pos[0] + n.size[0], n.pos[1] + n.size[1]];
  },
  // graph point -> screen, and a node slot's point on screen (output when out is true)
  g2s(x, y) {
    const a = this.app(), r = a.canvas.canvas.getBoundingClientRect(), p = a.canvas.ds.convertOffsetToCanvas([x, y]);
    return [r.left + p[0], r.top + p[1]];
  },
  slot(id, name, out) {
    const n = this.node(id), list = out ? n.outputs : n.inputs, i = list.findIndex((s) => s.name === name);
    if (i < 0) throw new Error(`${n.type} has no ${out ? 'output' : 'input'} ${name}`);
    const p = n.getConnectionPos(!out, i);
    return this.g2s(p[0], p[1]);
  },
  // ids: node ids, or {box: [x1, y1, x2, y2]} in graph space (an empty canvas has no nodes to frame)
  target(ids, pad, top, bottom = 150) {
    const a = this.app(), r = a.canvas.canvas.getBoundingClientRect();
    let [x1, y1, x2, y2] = [Infinity, Infinity, -Infinity, -Infinity];
    if (ids.box) [x1, y1, x2, y2] = ids.box;
    else for (const id of ids) { const b = this.rect(id); x1 = Math.min(x1, b[0]); y1 = Math.min(y1, b[1]); x2 = Math.max(x2, b[2]); y2 = Math.max(y2, b[3]); }
    const availH = r.height - top - bottom - pad * 2, availW = r.width - pad * 2;
    const s = Math.min(availW / (x2 - x1), availH / (y2 - y1), 2.2);
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    return [s, [r.width / (2 * s) - cx, (top + pad + availH / 2) / s - cy]];
  },
  camera(ids, ms, pad = 60, top = 0) {
    const a = this.app(), ds = a.canvas.ds, r = a.canvas.canvas.getBoundingClientRect();
    const [s1, o1] = this.target(ids, pad, top), s0 = ds.scale, o0 = [...ds.offset];
    // the graph point at the viewport centre and the zoom (in log) are interpolated: one smooth pan and zoom
    const mx = r.width / 2, my = r.height / 2;
    const c0 = [mx / s0 - o0[0], my / s0 - o0[1]], c1 = [mx / s1 - o1[0], my / s1 - o1[1]];
    const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    return new Promise((done) => {
      const t0 = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - t0) / ms), e = ease(t);
        const s = Math.exp(Math.log(s0) + (Math.log(s1) - Math.log(s0)) * e);
        ds.scale = s;
        ds.offset[0] = mx / s - (c0[0] + (c1[0] - c0[0]) * e);
        ds.offset[1] = my / s - (c0[1] + (c1[1] - c0[1]) * e);
        a.canvas.setDirty(true, true);
        if (t < 1) requestAnimationFrame(step); else { ds.scale = s1; ds.offset[0] = o1[0]; ds.offset[1] = o1[1]; a.canvas.setDirty(true, true); done(); }
      };
      requestAnimationFrame(step);
    });
  },
  widget(id, name) { return this.node(id).widgets.find((w) => w.name === name); },
  set(id, values) {
    for (const [k, v] of Object.entries(values)) { const w = this.widget(id, k); if (w) { w.value = v; w.callback?.(v); } }
    this.node(id).setDirtyCanvas(true, true);
  },
  widgetPoint(id, name) {
    const n = this.node(id), w = this.widget(id, name);
    const y = (w.last_y ?? w.y ?? 0) + (w.computedHeight ?? 20) / 2;
    return this.screen(id, n.size[0] * 0.1, y);
  },
  // a layer over the whole page: the result image, its boxes, an end card
  show(html, ms = 600) {
    let el = document.getElementById('__hv_show');
    if (!el) {
      el = document.createElement('div'); el.id = '__hv_show';
      Object.assign(el.style, {zoom: String(Math.max(1, innerWidth / 1920)), position: 'fixed', inset: '0', zIndex: 2147483600, background: 'rgba(10,10,12,0.96)', display: 'flex',
        alignItems: 'flex-start', justifyContent: 'center', paddingTop: '50px', boxSizing: 'border-box', opacity: '0', transition: `opacity ${ms}ms ease`, fontFamily: 'Segoe UI, sans-serif'});
      document.body.appendChild(el);
    }
    clearTimeout(el._hvRemove);  // a hide() just before would remove the layer this show() reuses
    el.innerHTML = html; requestAnimationFrame(() => { el.style.opacity = '1'; });
  },
  hide(ms = 600) { const el = document.getElementById('__hv_show'); if (el) { el.style.transition = `opacity ${ms}ms ease`; el.style.opacity = '0'; el._hvRemove = setTimeout(() => el.remove(), ms + 50); } },
  boxes(on) { document.querySelectorAll('.__hv_box').forEach((b) => { b.style.opacity = on ? '1' : '0'; }); },
  zone(id, kind) {
    const w = this.node(id).widgets.find((w) => Array.isArray(w._zones));
    const z = w?._zones.find((z) => z.kind === kind);
    return z ? {x: z.x, y: z.y, w: z.w, h: z.h, f: z.f ?? null} : null;
  },
};
"""


class Recorder:
    """Chrome's screencast: one JPEG per painted frame, with the time it was painted."""

    def __init__(self, page, folder):
        self.page, self.folder, self.frames = page, folder, []

    async def start(self):
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", lambda p: asyncio.ensure_future(self._frame(p)))
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})

    async def _frame(self, p):
        name = f"f{len(self.frames):06d}.jpg"
        self.frames.append((p["metadata"]["timestamp"], name))
        (self.folder / name).write_bytes(base64.b64decode(p["data"]))
        with contextlib.suppress(Exception):
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})

    async def stop(self):
        # a still page sends no frames: the last frame lasts until the recording stops
        self.t_stop = time.time()
        await self.cdp.send("Page.stopScreencast")
        await asyncio.sleep(0.2)


class StopScene(Exception):
    """Raised by say() past --until: a test take of the first lines only."""


UNTIL = None


class Director:
    """The scene's tools: captions, fast parts, a cursor that moves like a hand, clicks, typing, the camera."""

    def __init__(self, page, lines, voice_lens, language="en"):
        self.page, self.captions, self.fast_parts, self.pos = page, [], [], (W / 2, H / 2)
        self.lines, self.voice_lens = lines, voice_lens
        self.language = language
        self.t0, self.saved, self.voice_end, self.spoken, self.titles = time.time(), 0.0, 0.0, [], []

    def now(self):
        """Time in the finished video: the recording time less what the fast parts will cut."""
        return time.time() - self.t0 - self.saved

    async def _after_voice(self, gap=0.35):
        while self.now() < self.voice_end + gap:
            await asyncio.sleep(0.02)

    async def say(self, i):
        """Caption i of the scene, and its voice when there is one; waits for the previous voice to end first."""
        await self._after_voice()
        await self.no_library()
        if UNTIL is not None and i > UNTIL:
            raise StopScene
        en, es = self.lines[i][:2]
        self.captions.append((time.time(), en, es))
        if i in self.voice_lens:
            self.spoken.append((time.time(), i))
            self.voice_end = self.now() + self.voice_lens[i]

    def title(self, text, sub, seconds):
        self.titles.append((time.time(), seconds, text, sub))

    async def no_library(self):
        """Stop when a side panel listing files is open (the Model Library shows the user's own model and LoRA names)."""
        if await self.js("() => [...document.querySelectorAll('*')].some(e => /^(Model Library|Models)$/.test(e.innerText?.trim() ?? '') && e.getBoundingClientRect().width > 0 && e.getBoundingClientRect().left < 400 && e.getBoundingClientRect().top < 120 && e.childElementCount === 0)"):
            raise SystemExit("a model list is open on screen; recording stopped")

    async def hover(self, x, y, ms=None):
        await self.move(x, y, ms)

    async def end(self, tail=0.6):
        await self._after_voice(0.5)
        self.captions.append((time.time(), "", ""))
        await asyncio.sleep(tail)

    @contextlib.asynccontextmanager
    async def fast(self, target=1.6):
        if getattr(self, "_in_fast", False):  # nested: the outer fast part counts
            yield
            return
        self._in_fast = True
        t0 = time.time()
        try:
            yield
        finally:
            self._in_fast = False
        t1 = time.time()
        self.fast_parts.append((t0, t1, target))
        self.saved += max(0.0, (t1 - t0) - target)

    async def hold(self, s):
        await asyncio.sleep(s)

    async def move(self, x, y, ms=None):
        x0, y0 = self.pos
        dist = math.hypot(x - x0, y - y0)
        ms = ms or min(1100, 350 + dist * 0.55)
        n = max(2, int(ms / 16))
        # ease in-out with a slight arc, like a hand on a mouse
        bend = min(60, dist * 0.08)
        nx, ny = (-(y - y0) / dist, (x - x0) / dist) if dist else (0, 0)
        for i in range(1, n + 1):
            t = i / n
            e = t * t * (3 - 2 * t)
            arc = math.sin(math.pi * t) * bend
            await self.page.mouse.move(x0 + (x - x0) * e + nx * arc, y0 + (y - y0) * e + ny * arc)
            await asyncio.sleep(ms / 1000 / n)
        self.pos = (x, y)

    async def click(self, x, y):
        await self.move(x, y)
        await asyncio.sleep(0.15)
        await self.page.mouse.down()
        await asyncio.sleep(0.09)
        await self.page.mouse.up()
        await asyncio.sleep(0.25)

    async def drag(self, x0, y0, x1, y1, ms=1200):
        await self.move(x0, y0)
        await asyncio.sleep(0.2)
        await self.page.mouse.down()
        await asyncio.sleep(0.15)
        await self.move(x1, y1, ms)
        await asyncio.sleep(0.15)
        await self.page.mouse.up()
        await asyncio.sleep(0.3)

    async def js(self, code, arg=None):
        return await self.page.evaluate(code, arg)

    async def camera(self, ids, ms=1400, pad=60, top=0):
        await self.js("([ids, ms, pad, top]) => window.__hv.camera(ids, ms, pad, top)", [ids, ms, pad, top])

    async def screen(self, node, lx, ly):
        return await self.js("([n, x, y]) => window.__hv.screen(n, x, y)", [node, lx, ly])


# ---- scenes ----

LAYOUT_TEST_LINES = [
    ("Write the scene in one sentence, saying where each thing goes.",
     "Escribe la escena en una frase, diciendo dónde va cada cosa."),
    ("DRAFT LAYOUT: the Prompt Enhancer turns the sentence into boxes, without generating the image.",
     "DRAFT LAYOUT: el Prompt Enhancer convierte la frase en cajas, sin generar la imagen."),
    ("Move or resize any box before generating.",
     "Mueve o cambia de tamaño cualquier caja antes de generar."),
]


async def layout_test(d: Director):
    """About 20 s: a sentence typed in the Prompt Enhancer, DRAFT LAYOUT, one box dragged."""
    enhancer, editor = 2, 3
    await d.js("""(id) => { const w = window.__hv.widget(id, 'draft');
        w.value = JSON.stringify({width: 1024, height: 1024, bbox_coord_space: 'abs', bbox_format: 'xyxy', summary: {canvas: 'single', caption: ''}, items: []}, null, 2);
        window.__hv.node(id).setDirtyCanvas(true, true); }""", editor)
    await d.hold(0.3)
    await d.camera([enhancer, editor], 10, pad=40, top=60)
    await d.hold(0.8)
    await d.say(0)
    box = await d.js("""(id) => { const w = window.__hv.widget(id, 'idea'); w.value = '';
        if (w.element) w.element.value = ''; const r = w.element.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }""", enhancer)
    await d.click(box[0] + 40, box[1] + 30)
    await d.page.keyboard.type("A man seen from behind on the left of a narrow street at night, a full moon in the upper right, "
                               "a small dog at the end of the street, a signature in the lower right corner.", delay=15)
    await d.hold(1.0)
    await d.say(1)
    z = await d.js("(id) => window.__hv.zone(id, 'draft_run')", editor)
    x, y = await d.screen(editor, z["x"] + z["w"] / 2, z["y"] + z["h"] / 2)
    before = await d.js("(id) => window.__hv.widget(id, 'draft').value", editor)
    await d.click(x, y)
    async with d.fast():
        for _ in range(900):
            done = await d.js("""([id, before]) => { const n = window.__hv.node(id), v = window.__hv.widget(id, 'draft').value;
                return v !== before && !n._hadesDraftRun?.pendingPrompt(); }""", [editor, before])
            if done:
                break
            await asyncio.sleep(0.2)
        else:
            raise SystemExit("the draft did not arrive in 3 minutes")
        await d.hold(0.4)
    await d.hold(1.2)
    await d.camera([editor], 1300, pad=30, top=60)
    await d.hold(0.6)
    await d.say(2)
    f = (await d.js("(id) => window.__hv.zone(id, 'canvas')", editor))["f"]
    items = json.loads(await d.js("(id) => window.__hv.widget(id, 'draft').value", editor))["items"]
    text = lambda i: (i.get("caption", "") + " " + i.get("text", "")).lower()
    pick = next((i for w in ("moon", "dog") for i in items if w in text(i)), items[0])
    b = pick["bbox"]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0, y0 = await d.screen(editor, f["x"] + cx * f["s"], f["y"] + cy * f["s"])
    tx = max(b[2] - b[0], 140) / 2 + 40  # towards the centre, staying inside the canvas
    x1, y1 = await d.screen(editor, f["x"] + (cx - min(180, cx - tx)) * f["s"], f["y"] + (cy + 60) * f["s"])
    await d.drag(x0, y0, x1, y1, 1400)
    await d.hold(1.2)
    await d.end()


SENTENCE = ('A man seen from behind on the left of a narrow street at night, a full moon in the upper right, '
            'a small dog at the end of the street, a signature "HaDeS" in the lower right corner.')

LAYOUT_SENTENCE_LINES = [
    ("Hi, I'm HaDeS. Everything in this video was made with the HaDeS MOSS Suite, a set of ComfyUI nodes I built to make "
     "creating images with MOSS-Image easier.",
     "Hola, soy HaDeS. Todo lo que vas a ver está hecho con HaDeS MOSS Suite, una colección de nodos para ComfyUI que he creado "
     "para facilitar la creación de imágenes con MOSS-Image."),
    ("Everything you see in this video is free to download: the workflow, the official HaDeS MOSS Suite repository and the links "
     "to the Suite's official models are all in the description.",
     "Todo lo que sale en este vídeo se puede descargar gratis: en la descripción tienes el workflow, el repositorio oficial de "
     "HaDeS MOSS Suite y los enlaces a los modelos oficiales de la Suite."),
    ("To install it, open the Manager, search for HaDeS MOSS and click Install. Mine shows it as installed because, obviously, "
     "it's already in my ComfyUI. After a restart, the nodes are ready.",
     "Para instalarla, abre el Manager, busca HaDeS MOSS y pulsa Install. A mí me sale como instalada porque, obviamente, ya la "
     "tengo en mi ComfyUI. Tras reiniciar, los nodos ya están listos."),
    ("It's designed to be simple for everyone, whether you've used ComfyUI for years or you just installed it.",
     "Está pensada para que sea sencilla para todo el mundo, tanto si llevas años con ComfyUI como si acabas de instalarlo."),
    ("The Suite covers the whole path: loading the model, writing the prompt, sampling, saving with metadata, and checking the result.",
     "La Suite cubre todo el camino: cargar el modelo, escribir el prompt, generar, guardar con metadatos y comprobar el resultado."),
    ("And everything is measured: every preset and recommended value comes from tests, and the results are published with the Suite.",
     "Y todo está medido: cada preset y cada valor recomendado sale de pruebas, y los resultados se publican con la Suite."),
    ("Today: Layout from a Sentence. You describe the picture in one sentence, and the Suite places each thing where you said.",
     "Hoy: Layout from a Sentence. Describes la imagen en una frase y la Suite coloca cada cosa donde dijiste."),
    ("Write the scene in one sentence, saying where each thing goes.",
     "Escribe la escena en una frase, diciendo dónde va cada cosa."),
    ("DRAFT LAYOUT: the Prompt Enhancer turns the sentence into boxes, without generating the image.",
     "DRAFT LAYOUT: el Prompt Enhancer convierte la frase en cajas, sin generar la imagen."),
    ("Each box is a character, an object or a text, with a short description of what goes inside.",
     "Cada caja es un personaje, un objeto o un texto, con una breve descripción de lo que va dentro."),
    ("Move or resize any box before generating.",
     "Mueve o cambia de tamaño cualquier caja antes de generar."),
    ("I'm generating with HaDeS MOSS v1.0 Turbo, my own model. It's MOSS-Image Base with a Turbo LoRA I trained myself baked in: "
     "it teaches the model to reach in 8 steps what the original models, Base and SFT, need about 25 steps for. And at CFG 1 "
     "each step runs the model once instead of twice. If you want to try it: link in the description!",
     "Estoy generando con HaDeS MOSS v1.0 Turbo, mi propio modelo. Es MOSS-Image Base con una LoRA Turbo que entrené yo mismo, "
     "integrada en el modelo: le enseña a llegar en 8 pasos a lo que los modelos originales, Base y SFT, necesitan unos 25 pasos. "
     "Y con CFG 1 cada paso ejecuta el modelo una vez en lugar de dos. Si quieres probarlo: ¡enlace en la descripción!"),
    ("When the layout looks right, press Generate. The whole workflow runs with your boxes.",
     "Cuando el layout te guste, pulsa Generate. Todo el workflow se ejecuta con tus cajas."),
    ("MOSS-Image draws each thing inside its box. In my tests, 26 of 28 elements landed inside their boxes.",
     "MOSS-Image dibuja cada cosa dentro de su caja. En mis pruebas, 26 de 28 elementos quedaron dentro de su caja."),
    ("If you made it to the end and you liked it, and you want to see more videos like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final, te ha gustado y quieres ver más contenido así, dale un buen like y suscríbete, ¡es gratis!"),
]

BOX_COLORS = {"character": "#e0a040", "object": "#5b8def", "text": "#b06ae0"}


async def layout_sentence(d: Director):
    """The full Layout from a Sentence video: introduction, Manager, the Suite, then the walkthrough with HaDeS MOSS v1.0 Turbo."""
    loader, enhancer, editor, sampler, saver = 1, 2, 3, 6, 8
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    await d.js("""([e, k, l]) => {
        window.__hv.set(e, {seed: 3, idea: ''});
        window.__hv.set(k, {seed: 1001, steps: 8, cfg: 1.0, sampler_name: 'euler', scheduler: 'simple'});
        const w = window.__hv.widget(e, 'idea'); if (w.element) w.element.value = '';
    }""", [enhancer, sampler, loader])
    await d.js("""(id) => { const w = window.__hv.widget(id, 'draft');
        w.value = JSON.stringify({width: 1024, height: 1024, bbox_coord_space: 'abs', bbox_format: 'xyxy', summary: {canvas: 'single', caption: ''}, items: []}, null, 2);
        window.__hv.node(id).setDirtyCanvas(true, true); }""", editor)
    await d.camera(ids, 10, pad=50, top=60)
    await d.hold(1.0)

    # 1. introduction, a slow drift over the whole workflow
    d.title("HaDeS MOSS Suite", "Layout from a Sentence", 6.5)
    drift = asyncio.ensure_future(d.camera(ids[:6], 9000, pad=80, top=60))
    await d.say(0)
    await d.say(1)
    await drift

    # 1c. the Manager
    await d.say(2)
    mgr = d.page.get_by_role("button", name="Manager").first
    b = await mgr.bounding_box()
    await d.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    await d.hold(1.2)
    cnm = await d.page.get_by_text("Custom Nodes Manager").first.bounding_box()
    await d.click(cnm["x"] + cnm["width"] / 2, cnm["y"] + cnm["height"] / 2)
    await d.page.wait_for_selector("input.cn-manager-keywords", timeout=30000)
    await d.hold(1.5)
    sb = await d.page.locator("input.cn-manager-keywords").bounding_box()
    await d.click(sb["x"] + 60, sb["y"] + sb["height"] / 2)
    await d.page.keyboard.type("HaDeS MOSS", delay=70)
    await d.page.keyboard.press("Enter")
    await d.hold(1.0)
    row = await d.page.get_by_text("HaDeS MOSS Suite", exact=False).last.bounding_box()
    await d.hover(row["x"] + row["width"] / 2, row["y"] + row["height"] / 2)
    await d._after_voice(0.2)
    await d.hold(0.6)
    # close the list, then the Manager menu: only buttons whose class says close, the last one drawn first (never by position)
    for _ in range(3):
        close = await d.page.evaluate("""() => {
            const inManager = (e) => { let p = e; while (p && p.getBoundingClientRect().width < 900) p = p.parentElement;
                return Boolean(p && p !== document.body && (p.querySelector('input.cn-manager-keywords') || /ComfyUI Manager V\\d/.test(p.innerText.slice(0, 200)))); };
            const v = [...document.querySelectorAll('button.cm-ui-close, button[class*="close"], .comfy-modal-close')]
            .map(e => [e, e.getBoundingClientRect()]).filter(([e, r]) => r.width > 0 && r.width < 60 && getComputedStyle(e).visibility !== 'hidden' && inManager(e));
            const last = v[v.length - 1]; return last ? [last[1].x + last[1].width / 2, last[1].y + last[1].height / 2] : null; }""")
        if not close:
            break
        await d.click(*close)
        await d.hold(0.7)
    if await d.page.locator("input.cn-manager-keywords:visible").count():
        raise SystemExit("the Manager list did not close")

    # 2-3. simple for everyone, the whole path: node search
    await d.move(1500, 1000)
    await d.say(3)
    await d.camera(ids, 1600, pad=60, top=60)
    await d.say(4)
    await d.move(1860, 700)  # off the widgets: a resting pointer shows their tooltips
    # the search box by its command; typing only into its own field (a stray "m" opens the Model Library)
    await d.js("() => window.__hv.app().extensionManager.command.execute('Workspace.SearchBox.Toggle')")
    await d.hold(0.8)
    focused = await d.js("() => { const a = document.activeElement; return Boolean(a && a.tagName === 'INPUT' && a.closest('[role=dialog], .p-dialog, .comfy-vue-node-search-container')); }")
    if not focused:
        raise SystemExit("the node search box did not open; nothing typed")
    await d.page.keyboard.type("MOSS", delay=110)
    await d._after_voice(0.1)
    await d.hold(0.5)
    await d.page.keyboard.press("Escape")
    await d.hold(0.5)
    await d.no_library()

    # 4. measured: the flow_shift tooltip
    await d.camera([sampler], 1300, pad=120, top=60)
    await d.say(5)
    x, y = await d.js("([id, n]) => window.__hv.widgetPoint(id, n)", [sampler, "flow_shift"])
    await d.hover(x, y)
    for k in range(6):  # small moves keep the pointer events coming until the tooltip shows
        await d.page.mouse.move(x + (k % 2) * 2, y)
        await asyncio.sleep(0.25)
    await d._after_voice(0.1)
    await d.move(x + 300, y - 200)

    # 5-6. today's topic, the sentence
    await d.camera([enhancer, editor], 1500, pad=40, top=60)
    await d.say(6)
    await d._after_voice(0.1)
    await d.say(7)
    box = await d.js("""(id) => { const r = window.__hv.widget(id, 'idea').element.getBoundingClientRect(); return [r.left, r.top]; }""", enhancer)
    await d.click(box[0] + 40, box[1] + 30)
    await d.page.keyboard.type(SENTENCE, delay=15)
    await d.hold(0.6)

    # 7. DRAFT LAYOUT
    await d.say(8)
    z = await d.js("(id) => window.__hv.zone(id, 'draft_run')", editor)
    x, y = await d.screen(editor, z["x"] + z["w"] / 2, z["y"] + z["h"] / 2)
    before = await d.js("(id) => window.__hv.widget(id, 'draft').value", editor)
    await d.click(x, y)
    async with d.fast():
        for _ in range(1500):
            if await d.js("""([id, before]) => { const n = window.__hv.node(id), v = window.__hv.widget(id, 'draft').value;
                    return v !== before && !n._hadesDraftRun?.pendingPrompt(); }""", [editor, before]):
                break
            await asyncio.sleep(0.2)
        else:
            raise SystemExit("the draft did not arrive in 5 minutes")
        await d.hold(0.4)
    await d.hold(0.8)
    await d.camera([editor], 1300, pad=30, top=60)

    # 8. the boxes: select the dog
    items = json.loads(await d.js("(id) => window.__hv.widget(id, 'draft').value", editor))["items"]
    text = lambda i: (i.get("caption", "") + " " + str(i.get("text", ""))).lower()
    find = lambda w: next((i for i in items if w in text(i)), None)
    dog, moon = find("dog"), find("moon")
    if dog is None or moon is None or dog is moon:
        raise SystemExit(f"the draft has no separate dog and moon boxes ({len(items)} boxes); try another enhancer seed")
    f = (await d.js("(id) => window.__hv.zone(id, 'canvas')", editor))["f"]
    at = lambda bx, by: d.screen(editor, f["x"] + bx * f["s"], f["y"] + by * f["s"])
    await d.say(9)
    b = dog["bbox"]
    await d.click(*(await at((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
    await d._after_voice(0.1)

    # 9. move the moon, enlarge the dog
    await d.say(10)
    b = moon["bbox"]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    await d.drag(*(await at(cx, cy)), *(await at(cx - 144, cy + 32)), 1300)
    b = dog["bbox"]
    await d.click(*(await at((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
    await d.drag(*(await at(b[2], b[3])), *(await at(b[2] + 48, b[3] + 48)), 900)
    await d.hold(0.6)

    # 10a. the model
    await d.camera([loader], 1300, pad=140, top=60)
    await d.say(11)
    await d.hold(3.5)
    await d.camera([sampler], 1300, pad=120, top=60)
    x, y = await d.js("([id, n]) => window.__hv.widgetPoint(id, n)", [sampler, "steps"])
    await d.move(x, y)
    await d.hold(2.0)
    await d.camera([editor], 1300, pad=30, top=60)
    await d._after_voice(0.1)

    # 10b. GENERATE
    await d.say(12)
    z = await d.js("(id) => window.__hv.zone(id, 'generate')", editor)
    x, y = await d.screen(editor, z["x"] + z["w"] / 2, z["y"] + z["h"] / 2)
    await d.click(x, y)
    async with d.fast(2.0):
        await d.hold(1.0)
        await d.camera(ids, 1500, pad=60, top=60)
        for _ in range(3000):
            q = await d.js("() => fetch('/queue').then(r => r.json()).then(q => q.queue_running.length + q.queue_pending.length)")
            if q == 0:
                break
            await asyncio.sleep(0.3)
        await d.hold(0.8)
    draft = json.loads(await d.js("(id) => window.__hv.widget(id, 'draft').value", editor))
    hist = await d.js("() => fetch('/history?max_items=1').then(r => r.json())")
    run = next(iter(hist.values()))
    img = run["outputs"][str(saver)]["images"][0]
    url = f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
    await d.camera([saver], 1200, pad=80, top=60)
    await d._after_voice(0.1)

    # 11. the result, its boxes over it
    await d.say(13)
    ih, iw = 820, 820 * draft["width"] / draft["height"]
    s = ih / draft["height"]
    boxes = "".join(
        f'<div class="__hv_box" style="position:absolute;left:{it["bbox"][0] * s}px;top:{it["bbox"][1] * s}px;'
        f'width:{(it["bbox"][2] - it["bbox"][0]) * s}px;height:{(it["bbox"][3] - it["bbox"][1]) * s}px;'
        f'border:3px solid {BOX_COLORS.get(it["label"], "#fff")};border-radius:4px;opacity:0;transition:opacity .5s ease;'
        f'box-shadow:0 0 0 1px rgba(0,0,0,.5)"></div>' for it in draft["items"])
    await d.js("(html) => window.__hv.show(html)", f'<div style="position:relative;width:{iw}px;height:{ih}px">'
               f'<img src="{url}" style="width:100%;height:100%;border-radius:6px;box-shadow:0 10px 50px rgba(0,0,0,.6)">{boxes}</div>')
    await d.move(1700, 980)
    await d.hold(1.8)
    await d.js("() => window.__hv.boxes(true)")
    await d.hold(3.2)
    await d.js("() => window.__hv.boxes(false)")
    await d._after_voice(0.1)

    # 12. goodbye
    await d.js("(html) => window.__hv.show(html)", f'<div style="display:flex;align-items:center;gap:80px">'
               f'<img src="{url}" style="height:{ih}px;border-radius:6px;box-shadow:0 10px 50px rgba(0,0,0,.6)">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black, sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI, sans-serif;color:#ffcc33;margin-top:22px">Layout from a Sentence</div>'
               '<div style="font:400 26px Segoe UI, sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(14)
    await d.end(1.5)


TURBO_UPSCALE_LINES = [
    ("Hi, I'm HaDeS. Today I'll show you my Turbo 8 Steps plus x2 Upscale workflow for the MOSS-Image model in ComfyUI.",
     "Hola, soy HaDeS. Hoy te enseño mi workflow Turbo de ocho pasos con escalado x2 para el modelo MOSS-Image en ComfyUI."),
    ("And how do I install it? Open the ComfyUI Manager, go to Custom Nodes Manager, search for HaDeS MOSS Suite, and click Install. "
     "After installation, restart ComfyUI so the nodes appear. Mine already shows it as installed because, of course, I already have it.",
     "¿Y cómo instalo esto? Abre el Manager de ComfyUI, entra en Custom Nodes Manager, busca HaDeS MOSS Suite y pulsa Install. "
     "Después de instalarla, reinicia ComfyUI para que aparezcan los nodos. A mí ya me sale instalada porque, obviamente, ya la tengo."),
    ("The HaDeS models and workflows I made are free to download. I've put their links in the description, along with the links for "
     "the base model and the upscaler.",
     "Los modelos HaDeS y los workflows que he creado se pueden descargar gratis. He dejado sus enlaces en la descripción, junto con "
     "los del modelo base y el upscaler."),
    ("For this example, I'm using my Turbo 8-steps v0.7 LoRA with MOSS-Image SFT, as shown in this workflow. A newer, improved version, "
     "v1.0, is also available.",
     "Para este ejemplo uso mi LoRA Turbo 8-steps v0.7 con MOSS-Image SFT, tal como aparece en este workflow. También está disponible "
     "una versión posterior y mejorada, la v1.0."),
    ("I built this workflow in three stages: generate the image, upscale it, and optionally refine the upscaled image with MOSS.",
     "He organizado este workflow en tres etapas: generar la imagen, escalarla y, si quiero, refinarla con MOSS."),
    ("I generate the first image at 1024 by 1024, with eight steps, CFG one, and Euler Simple.",
     "Genero la primera imagen a 1024 por 1024, con ocho steps, CFG uno y Euler Simple."),
    ("Now I use RealESRGAN x2 to take the image to 2048 by 2048. This adds an upscaling stage without running another MOSS pass.",
     "Ahora uso RealESRGAN x2 para llevar la imagen a 2048 por 2048. Así añado el escalado sin ejecutar otro pase de MOSS."),
    ("If I want to refine the upscaled image, I can activate this MOSS pass at denoise 0.25.",
     "Si quiero refinar la imagen escalada, puedo activar este pase de MOSS con denoise 0,25."),
    ("In my four-image test, I preferred the texture in the full-size crops. That's a visual result from a small test, not a guarantee "
     "for every image.",
     "En mi prueba con cuatro imágenes, preferí la textura de los recortes al cien por cien. Es el resultado visual de una prueba "
     "pequeña, no una garantía para todas las imágenes."),
    ("I can stop after the upscale, or add the MOSS pass when I want to refine the result.",
     "Puedo quedarme con el escalado o añadir el pase de MOSS cuando quiera refinar el resultado."),
    ("If you've made it to the end and enjoyed the video, and you'd like more short, easy ComfyUI tutorials, please leave a like and "
     "subscribe. Thank you for watching until the end.",
     "Si has llegado hasta el final y te ha gustado el vídeo, y quieres más tutoriales cortos y sencillos para ComfyUI, déjame un like "
     "y suscríbete. ¡Gracias por ver el vídeo hasta el final!"),
    ("Here are the same areas at one hundred percent. On the left, I doubled the original pixels. On the right is the RealESRGAN result.",
     "Aquí se ven las mismas zonas al cien por cien. A la izquierda, he duplicado los píxeles del original. A la derecha, está el resultado de RealESRGAN."),
]


async def turbo_comparison(d, images, labels, heading, crop=None, double_before=False):
    """Display full images or matching 700 px windows in 2048 px image space."""
    await d.js("""async ({images, labels, heading, crop, doubleBefore}) => {
        const size = 700;
        const columns = await Promise.all(images.map(async (src, i) => {
            const img = new Image();
            img.src = src;
            await img.decode();
            const doubled = i === 0 && doubleBefore;
            const expected = doubled ? 1024 : 2048;
            if (img.naturalWidth !== expected || img.naturalHeight !== expected)
                throw new Error('Unexpected image size for the Turbo comparison');

            const style = crop
                ? 'width:2048px;height:2048px;max-width:none;margin-left:-' + crop[0] + 'px;margin-top:-' + crop[1] + 'px;'
                : 'width:700px;height:700px;object-fit:contain;';
            img.style.cssText = 'display:block;' + style + (crop && doubled ? 'image-rendering:pixelated;' : '');
            return '<div><div style="margin-bottom:12px">' + labels[i] + '</div>' +
                '<div style="width:700px;height:700px;overflow:hidden">' + img.outerHTML + '</div></div>';
        }));
        window.__hv.show('<section id="__hv_turbo_compare" style="color:white;font:600 26px Segoe UI,sans-serif">' +
            '<div style="color:#ffcc33;font-size:32px;margin-bottom:18px">' + heading + '</div>' +
            '<div style="display:flex;gap:42px">' + columns.join('') + '</div></section>', 180);
        if (crop) {
            const imgs = [...document.querySelectorAll('#__hv_turbo_compare img')];
            if (imgs.length !== 2 || devicePixelRatio !== 1 || imgs.some(img => {
                const frame = img.parentElement.getBoundingClientRect(), box = img.getBoundingClientRect();
                return box.width !== 2048 || box.height !== 2048 || frame.width !== size || frame.height !== size
                    || Math.abs(frame.left - box.left - crop[0]) > 0.01 || Math.abs(frame.top - box.top - crop[1]) > 0.01;
            })) throw new Error('The crop is not rendered at 1:1');
        }
    }""", {"images": images, "labels": labels, "heading": heading, "crop": crop, "doubleBefore": double_before})


async def turbo_crop_sequence(d, images, labels, voice_line, double_before=False):
    es = d.language == "es"
    face = "Rostro · recorte al 100 %" if es else "Face · 100% crop"
    hands = "Manos y herramientas · recorte al 100 %" if es else "Hands and tools · 100% crop"
    await turbo_comparison(d, images, labels, face, (950, 124), double_before)
    await d.hold(0.2)
    await d.say(voice_line)
    await d.hold(max(2.6, d.voice_lens.get(voice_line, 6) / 2))
    await turbo_comparison(d, images, labels, hands, (535, 790), double_before)
    await d._after_voice(0.35)
    await d.hold(0.3)


async def turbo_render(d, output_ids):
    """Queue the graph and return image URLs for the requested output nodes."""
    await d.js("() => window.__hv.app().queuePrompt(0, 1)")
    async with d.fast(2.0):
        await d.hold(0.8)
        for _ in range(1200):
            if await d.js("() => fetch('/queue').then(r => r.json()).then(q => q.queue_running.length + q.queue_pending.length)") == 0:
                break
            await asyncio.sleep(0.25)
        else:
            raise SystemExit("the Turbo workflow did not finish in 5 minutes")
        await d.hold(0.6)
    return await d.js("""async (ids) => {
        const hist = await fetch('/history?max_items=1').then(r => r.json());
        const run = Object.values(hist)[0];
        return ids.map(id => {
            const im = run?.outputs?.[id]?.images?.[0];
            if (!im) throw new Error('The Turbo workflow did not save output ' + id);
            return '/view?' + new URLSearchParams({
                filename: im.filename, subfolder: im.subfolder || '', type: im.type || 'output',
            });
        });
    }""", output_ids)


async def turbo_upscale_x2(d: Director):
    """Show Turbo v0.7, RealESRGAN x2 and the optional MOSS refinement."""
    all_ids = list(range(1, 13))
    es = d.language == "es"
    heading = "Imágenes completas" if es else "Full images"
    first_labels = ["1024 × 1024 · " + ("original" if es else "generated"), "2048 × 2048 · RealESRGAN x2"]
    crop_labels = ["Original ×2 · " + ("píxeles duplicados" if es else "doubled pixels"), first_labels[1]]
    refined_labels = ["RealESRGAN x2", "x2 + MOSS · denoise " + ("0,25" if es else "0.25")]
    await d.camera(all_ids, 10, pad=50, top=60)
    await d.hold(0.8)
    d.title("HaDeS MOSS Suite", "Turbo 8 Steps + Upscale x2", 6.5)
    await d.say(0)

    await d.say(1)
    manager = d.page.get_by_role("button", name="Manager").first
    b = await manager.bounding_box()
    if not b:
        raise SystemExit("the ComfyUI Manager button is not visible")
    await d.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    cnm = await d.page.get_by_text("Custom Nodes Manager").first.bounding_box()
    await d.click(cnm["x"] + cnm["width"] / 2, cnm["y"] + cnm["height"] / 2)
    await d.page.wait_for_selector("input.cn-manager-keywords", timeout=30000)
    await d.hold(1.0)
    search = d.page.locator("input.cn-manager-keywords").first
    await search.fill("HaDeS MOSS")
    await d.hold(2.0)
    if not await d.page.get_by_text("HaDeS MOSS Suite", exact=False).count():
        raise SystemExit("the Manager search did not show HaDeS MOSS Suite")
    row = await d.page.get_by_text("HaDeS MOSS Suite", exact=False).last.bounding_box()
    await d.hover(row["x"] + row["width"] / 2, row["y"] + row["height"] / 2)
    await d._after_voice(0.2)
    await d.hold(0.6)
    for _ in range(3):
        close = await d.page.evaluate("""() => {
            const inManager = (e) => { let p = e; while (p && p.getBoundingClientRect().width < 900) p = p.parentElement;
                return Boolean(p && p !== document.body && (p.querySelector('input.cn-manager-keywords') || /ComfyUI Manager V\\d/.test(p.innerText.slice(0, 200)))); };
            const v = [...document.querySelectorAll('button.cm-ui-close')]
            .map(e => [e, e.getBoundingClientRect()]).filter(([e, r]) => r.width > 0 && r.width < 60 && getComputedStyle(e).visibility !== 'hidden' && inManager(e));
            const last = v[v.length - 1]; return last ? [last[1].x + last[1].width / 2, last[1].y + last[1].height / 2] : null; }""")
        if not close:
            break
        await d.click(*close)
        await d.hold(0.6)
    if await d.page.locator("input.cn-manager-keywords:visible").count():
        raise SystemExit("the Manager list did not close")

    await d.camera([1], 950, pad=28, top=45)
    await d.say(2)
    await d.say(3)
    await d.hold(3.0)
    await d.camera([4], 900, pad=36, top=55)
    await d._after_voice()
    await d.camera(all_ids, 1100, pad=45, top=50)
    await d.say(4)
    await d.hold(2.0)
    await d.camera([7, 8, 9], 850, pad=38, top=46)
    await d.hold(1.25)
    await d.camera([10, 11, 12], 850, pad=38, top=46)
    await d._after_voice()

    await d.camera([3, 4, 5], 950, pad=26, top=44)
    await d.say(5)
    await d._after_voice()
    base, upscale = await turbo_render(d, [6, 9])
    await d.camera([7, 8, 9], 950, pad=36, top=45)
    await d.say(6)
    await d._after_voice()
    await d.move(W - 70, H - 100)
    await turbo_comparison(d, [base, upscale], first_labels, heading, double_before=True)
    await d.hold(3.0)
    await turbo_crop_sequence(d, [base, upscale], crop_labels, 11, double_before=True)
    await d.js("() => window.__hv.hide(180)")

    # Change only the three in-memory nodes in the optional group.
    await d.camera([11], 1000, pad=30, top=45)
    await d.js("""() => {
        const a = window.__hv.app();
        for (const id of [10, 11, 12]) window.__hv.node(id).mode = 0;
        a.graph.setDirtyCanvas(true, true);
        a.canvas.setDirty(true, true);
    }""")
    await d.hover(*await d.js("() => window.__hv.widgetPoint(11, 'denoise')"), ms=450)
    await d.say(7)
    await d._after_voice()
    refined, = await turbo_render(d, [12])
    await d.move(W - 70, H - 100)
    await turbo_comparison(d, [upscale, refined], refined_labels, heading)
    await d.hold(3.0)
    await turbo_crop_sequence(d, [upscale, refined], refined_labels, 8)
    await d.js("() => window.__hv.hide(180)")
    await d.camera([9, 12], 1050, pad=36, top=45)
    await d.say(9)
    await d._after_voice()

    links = "Enlaces al workflow y los modelos en la descripción" if es else "Workflow and model links in the description"
    await d.js("(html) => window.__hv.show(html)", '<div style="color:#fff;max-width:1200px;text-align:left">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               '<div style="font:600 38px Segoe UI,sans-serif;color:#ffcc33;margin-top:18px">Turbo 8 Steps + Upscale x2</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:38px">' + links + '</div></div>')
    await d.say(10)
    await d.end(1.5)


# ---- basic generation from an empty canvas ----

BASIC_PROMPT = ("Photograph of an old fisherman with a grey beard mending a net on a wooden dock at sunrise, his face and weathered "
                "hands in view, thick wool sweater, seagulls in the background.")
BASIC_SEED = 1013  # picked on the clean recording instance (sage attention): good at 25 steps and with the turbo
BASIC_LORA = "MOSS\\turbo_v10\\HaDeS-MOSS-Turbo-8steps-v1.0.safetensors"
BASIC_LOADER = {"unet_name": "mossImage_base_bf16.safetensors", "clip_name": "qwen3.5_9b_int8_convrot.safetensors",
                "vae_name": "flux2-vae.safetensors"}
# where each node goes, in graph space: one row, the turbo under the latent
BASIC_POS = {"HaDeSMossLoader": (0, 0), "HaDeSMossPromptEnhancer": (440, 0), "HaDeSMossLatent": (940, 0),
             "HaDeSMossKSampler": (1380, 0), "HaDeSMossImageSaver": (1820, 0), "HaDeSMossTurbo": (940, 450)}
BASIC_BOX = [-260, -60, 2220, 800]  # room on the left: a double-click near the sidebar opens nothing
# 100 % crops (image pixels, 512 square, top-left corners) of the seed-1013 images: face and hands, 25 steps then turbo
BASIC_CROPS = [((214, 34), (464, 512)), ((250, 44), (280, 512))]

BASIC_LINES = [
    ("Hi, I'm HaDeS, and welcome to the first tutorial on this channel.",
     "Hola, soy HaDeS, y bienvenido al primer tutorial de este canal."),
    ("Today we'll build a MOSS-Image workflow in ComfyUI from a completely empty canvas, one node at a time.",
     "Hoy vamos a montar un workflow de MOSS-Image en ComfyUI desde un lienzo completamente vacío, nodo a nodo."),
    ("We'll make a first image at twenty-five steps, then add my Turbo LoRA and make the same image in only eight.",
     "Haremos una primera imagen a veinticinco pasos, luego añadiremos mi LoRA Turbo y haremos la misma imagen en solo ocho."),
    ("Everything I use is free: the workflow we build today, the HaDeS MOSS Suite, my models and the official MOSS-Image files. "
     "All the links are in the description.",
     "Todo lo que uso es gratis: el workflow que montamos hoy, HaDeS MOSS Suite, mis modelos y los archivos oficiales de MOSS-Image. "
     "Tienes todos los enlaces en la descripción."),
    # 4: install
    ("First, the nodes. Open the Manager, go to Custom Nodes Manager and search for HaDeS MOSS.",
     "Primero, los nodos. Abre el Manager, entra en Custom Nodes Manager y busca HaDeS MOSS."),
    ("Click Install and restart ComfyUI. Mine is already installed, because obviously I use it every day.",
     "Pulsa Install y reinicia ComfyUI. A mí ya me sale instalada porque, obviamente, la uso a diario."),
    ("If you prefer the very latest updates, you can also install it from the official GitHub repository. The link is in the "
     "description too.",
     "Si prefieres tener siempre las últimas actualizaciones, también puedes instalarla desde el repositorio oficial de GitHub. "
     "El enlace también está en la descripción."),
    # 7: the empty canvas
    ("So here's an empty workflow. To make an image, we need five things.",
     "Aquí tenemos un workflow vacío. Para hacer una imagen necesitamos cinco cosas."),
    ("A loader for the model, a prompt, an empty image to start from, a sampler that does the work, and a node that saves the result.",
     "Un loader para el modelo, un prompt, una imagen vacía de la que partir, un sampler que hace el trabajo y un nodo que guarda "
     "el resultado."),
    # 9: loader
    ("I double-click the empty canvas to open the node search, and type MOSS Loader.",
     "Hago doble clic en el lienzo vacío para abrir el buscador de nodos y escribo MOSS Loader."),
    ("MOSS-Image needs three files: the diffusion model, the Qwen text encoder and the Flux 2 VAE. This one node loads all three.",
     "MOSS-Image necesita tres archivos: el modelo de difusión, el text encoder Qwen y el VAE de Flux 2. Este nodo carga los tres.",
     "MOSS-Image needs three files: the diffusion model, the Qwen text encoder and the Flux two VAE. This one node loads all three."),
    ("For the model I pick MOSS-Image Base, the original model from the MOSS team.",
     "Como modelo elijo MOSS-Image Base, el modelo original del equipo de MOSS."),
    ("Then the int8 text encoder and the Flux 2 VAE. They're all in the Comfy-Org MOSS-Image repository.",
     "Después, el text encoder int8 y el VAE de Flux 2. Están todos en el repositorio MOSS-Image de Comfy-Org.",
     "Then the int eight text encoder and the Flux two VAE. They're all in the Comfy Org MOSS-Image repository."),
    # 13: prompt
    ("Next, the prompt. I add the MOSS Prompt Enhancer and leave it on \"as written\", so it uses my text exactly as I type it.",
     "Ahora el prompt. Añado el MOSS Prompt Enhancer y lo dejo en \"as written\", para que use mi texto tal cual lo escribo.",
     "Next, the prompt. I add the MOSS Prompt Enhancer and leave it on as written, so it uses my text exactly as I type it."),
    ("MOSS reads natural language, not lists of tags. So I just describe the picture in a sentence.",
     "MOSS lee lenguaje natural, no listas de etiquetas. Así que describo la imagen en una frase."),
    ("I connect the CLIP output of the loader to the enhancer. That's the text encoder.",
     "Conecto la salida CLIP del loader al enhancer. Eso es el text encoder.",
     "I connect the clip output of the loader to the enhancer. That's the text encoder."),
    # 16: latent
    ("Now the empty image. MOSS Latent sets the size: square, 1024 by 1024.",
     "Ahora la imagen vacía. MOSS Latent fija el tamaño: cuadrada, 1024 por 1024.",
     "Now the empty image. MOSS Latent sets the size: square, ten twenty-four by ten twenty-four."),
    ("It also holds the seed. I keep it fixed, so we can compare the same image later.",
     "También guarda la semilla. La dejo fija para poder comparar luego la misma imagen."),
    # 18: sampler
    ("The heart of the workflow is the MOSS KSampler. Let's wire it up.",
     "El corazón del workflow es el MOSS KSampler. Vamos a conectarlo.",
     "The heart of the workflow is the MOSS K Sampler. Let's wire it up."),
    ("The model from the loader, the positive and negative prompts from the enhancer, and the latent and seed from MOSS Latent.",
     "El modelo del loader, los prompts positivo y negativo del enhancer, y el latente y la semilla de MOSS Latent."),
    ("I also connect the VAE, so the sampler hands me the finished image directly.",
     "También conecto el VAE, para que el sampler me dé directamente la imagen terminada."),
    ("For Base, I use twenty-five steps, CFG two point five, euler and simple.",
     "Para Base uso veinticinco pasos, CFG dos coma cinco, euler y simple."),
    # 22: saver
    ("Last, the MOSS Image Saver. It saves the image with its generation settings written inside, in the format Civitai reads.",
     "Por último, el MOSS Image Saver. Guarda la imagen con sus ajustes de generación dentro, en el formato que lee Civitai.",
     "Last, the MOSS Image Saver. It saves the image with its generation settings written inside, in the format Civitai reads."),
    # 23: first run
    ("That's the whole workflow. Let's run it.",
     "Ese es el workflow entero. Vamos a lanzarlo."),
    ("And here's our fisherman. On my RTX 5090, twenty-five steps took about thirteen seconds.",
     "Y aquí está nuestro pescador. En mi RTX 5090, los veinticinco pasos han tardado unos trece segundos.",
     "And here's our fisherman. On my RTX fifty ninety, twenty-five steps took about thirteen seconds."),
    # 25: turbo
    ("Twenty-five steps looks great, but it takes time. This is where my Turbo LoRA comes in.",
     "Veinticinco pasos se ve muy bien, pero lleva su tiempo. Aquí entra mi LoRA Turbo.",
     "Twenty-five steps looks great, but it takes time. This is where my Turbo Lora comes in."),
    ("I add the MOSS Turbo LoRA node and pick Turbo 8-steps v1.0, which I trained myself for MOSS-Image Base.",
     "Añado el nodo MOSS Turbo LoRA y elijo Turbo 8-steps v1.0, que entrené yo mismo para MOSS-Image Base.",
     "I add the MOSS Turbo Lora node and pick Turbo eight steps, version one, which I trained myself for MOSS-Image Base."),
    ("It's a distilled LoRA: it learned from the full model how to draw in eight steps instead of twenty-five. "
     "And the file is only three and a half megabytes.",
     "Es una LoRA destilada: aprendió del modelo completo a dibujar en ocho pasos en lugar de veinticinco. "
     "Y el archivo pesa solo tres megas y medio.",
     "It's a distilled Lora: it learned from the full model how to draw in eight steps instead of twenty-five. "
     "And the file is only three and a half megabytes."),
    ("I plug it into the turbo input of the sampler. Now the sampler uses the turbo's own settings: eight steps, CFG one, euler and simple.",
     "La conecto a la entrada turbo del sampler. Ahora el sampler usa los ajustes del turbo: ocho pasos, CFG uno, euler y simple."),
    ("At CFG one, each step runs the model once instead of twice. That's part of why it's so fast.",
     "Con CFG uno, cada paso ejecuta el modelo una vez en lugar de dos. Eso es parte de por qué va tan rápido."),
    # 30: second run
    ("Same prompt, same seed. Let's run it again.",
     "Mismo prompt, misma semilla. Volvemos a lanzarlo."),
    ("Eight steps: about three seconds.",
     "Ocho pasos: unos tres segundos."),
    ("Here they are side by side: twenty-five steps on the left, eight with the turbo on the right.",
     "Aquí están lado a lado: veinticinco pasos a la izquierda, ocho con el turbo a la derecha."),
    ("And up close, at full size. Not the same picture, but the same scene, with the skin, the beard and the wool all there, "
     "in about a quarter of the time.",
     "Y de cerca, a tamaño real. No es la misma imagen, pero sí la misma escena, con la piel, la barba y la lana, "
     "en más o menos una cuarta parte del tiempo."),
    # 34: tips
    ("A few tips. Set the seed to randomize when you want a new image on every run.",
     "Un par de consejos. Pon la semilla en randomize cuando quieras una imagen nueva en cada ejecución."),
    ("To go back to twenty-five steps, just switch the turbo node off. No need to rewire anything.",
     "Para volver a los veinticinco pasos, basta con apagar el nodo turbo. No hace falta tocar ningún cable."),
    ("And if you'd rather skip the LoRA, my model HaDeS MOSS v1.0 Turbo has this turbo already baked in. Load it and leave the "
     "turbo input empty.",
     "Y si prefieres no usar la LoRA, mi modelo HaDeS MOSS v1.0 Turbo ya lleva este turbo dentro. Cárgalo y deja vacía la entrada turbo.",
     "And if you'd rather skip the Lora, my model HaDeS MOSS version one Turbo has this turbo already baked in. Load it and leave "
     "the turbo input empty."),
    # 37: wrap-up
    ("That's it: six nodes, from an empty canvas to an image in eight steps.",
     "Y ya está: seis nodos, de un lienzo vacío a una imagen en ocho pasos."),
    ("I've saved this exact workflow. You can download it for free from the description and drop it straight into ComfyUI.",
     "He guardado este mismo workflow. Puedes descargarlo gratis desde la descripción y arrastrarlo directamente a ComfyUI."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

BASIC_BUILD_JS = r"""
async ([pos, loader, prompt, lora, seed]) => {
  const app = window.__hv.app(), LG = window.LiteGraph;
  app.graph.clear();
  const mk = (type) => { const n = LG.createNode(type); n.pos = [...pos[type]]; app.graph.add(n); return n; };
  const L = mk('HaDeSMossLoader'), P = mk('HaDeSMossPromptEnhancer'), T = mk('HaDeSMossLatent'),
        K = mk('HaDeSMossKSampler'), S = mk('HaDeSMossImageSaver'), U = mk('HaDeSMossTurbo');
  const set = (n, v) => { for (const [k, x] of Object.entries(v)) { const w = n.widgets.find(w => w.name === k); w.value = x; w.callback?.(x); } };
  set(L, loader); set(P, {idea: prompt, mode: 'as written'}); set(T, {seed});
  set(K, {steps: 2, cfg: 2.5, sampler_name: 'euler', scheduler: 'simple'}); set(U, {lora_name: lora, steps: 8, cfg: 1.0});
  const link = (a, out, b, inp) => a.connect(a.outputs.findIndex(s => s.name === out), b, b.inputs.findIndex(s => s.name === inp));
  link(L, 'CLIP', P, 'clip'); link(L, 'MODEL', K, 'model'); link(L, 'VAE', K, 'vae'); link(P, 'positive', K, 'positive');
  link(P, 'negative', K, 'negative'); link(T, 'latent', K, 'latent_image'); link(T, 'seed', K, 'seed'); link(K, 'image', S, 'images');
  const run = async () => { const pr = await app.graphToPrompt();
    const r = await (await fetch('/prompt', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt: pr.output})})).json();
    for (;;) { const h = await (await fetch('/history/' + r.prompt_id)).json(); if (h[r.prompt_id]?.status?.completed !== undefined) return h[r.prompt_id].status.status_str; await new Promise(f => setTimeout(f, 500)); } };
  const a = await run();
  link(U, 'turbo', K, 'turbo');
  const b = await run();
  app.graph.clear();
  return [a, b];
}
"""


async def basic_prepare(page):
    """Off camera: one short Base run and one turbo run load the model, the text encoder, the VAE and the LoRA."""
    print("warm-up runs:", await page.evaluate(BASIC_BUILD_JS, [BASIC_POS, BASIC_LOADER, BASIC_PROMPT, BASIC_LORA, 7]), flush=True)


async def basic_add(d: Director, type_, query, pos=None):
    """Double-click the canvas where the node goes (pos in graph space, BASIC_POS by default), type its name in the search
    box, Enter; returns the new node's id."""
    pos = list(pos or BASIC_POS[type_])
    gx, gy = pos
    x, y = await d.js("([x, y]) => window.__hv.g2s(x, y)", [gx + 40, gy + 20])
    before = await d.js("() => window.__hv.app().graph._nodes.length")
    await d.move(x, y)
    await asyncio.sleep(0.2)
    await d.page.mouse.dblclick(x, y)
    await d.hold(0.7)
    focused = await d.js("() => { const a = document.activeElement; return Boolean(a && a.tagName === 'INPUT' && a.closest('[role=dialog], .p-dialog, .comfy-vue-node-search-container')); }")
    if not focused:  # fall back on the search box's own command
        await d.js("() => window.__hv.app().extensionManager.command.execute('Workspace.SearchBox.Toggle')")
        await d.hold(0.7)
        focused = await d.js("() => { const a = document.activeElement; return Boolean(a && a.tagName === 'INPUT' && a.closest('[role=dialog], .p-dialog, .comfy-vue-node-search-container')); }")
    if not focused:
        raise SystemExit(f"the node search did not open for {query}; nothing typed")
    await d.page.keyboard.type(query, delay=75)
    await d.hold(0.9)
    await d.page.keyboard.press("Enter")
    await d.hold(0.4)
    await d.move(x + 6, y + 4, 300)  # the new node follows the pointer until a click drops it
    await d.click(x, y)
    await d.hold(0.3)
    node = await d.js("""([before, type, pos]) => { const g = window.__hv.app().graph;
        if (g._nodes.length !== before + 1) return null; const n = g._nodes[g._nodes.length - 1];
        if (n.type !== type) return 'wrong:' + n.type; if (n.flags?.ghost) return 'wrong: still following the pointer'; n.pos = [...pos]; g.setDirtyCanvas(true, true); return n.id; }""",
                      [before, type_, pos])
    if not node or str(node).startswith("wrong"):
        raise SystemExit(f"searching {query} did not add {type_} ({node})")
    await d.no_library()
    return node


async def basic_link(d: Director, a, out, b, inp, ms=1100):
    """Drag a wire from a's output to b's input, like a hand would; checks the link exists."""
    x0, y0 = await d.js("([id, n]) => window.__hv.slot(id, n, true)", [a, out])
    x1, y1 = await d.js("([id, n]) => window.__hv.slot(id, n, false)", [b, inp])
    await d.drag(x0, y0, x1, y1, ms)
    ok = await d.js("""([a, b, inp]) => { const n = window.__hv.node(b), s = n.inputs.find(s => s.name === inp);
        const l = s?.link != null && window.__hv.app().graph.links[s.link] ? window.__hv.app().graph.links[s.link] : null;
        return Boolean(l && String(l.origin_id) === String(a)); }""", [a, b, inp])
    if not ok:
        raise SystemExit(f"the wire {out} -> {inp} did not connect")


async def basic_point(d: Director, node, widget, values=None, ms=None):
    """Point at a widget (its value is set by code: a list is never opened, it would show the user's files)."""
    x, y = await d.js("""([id, name]) => { const n = window.__hv.node(id);
        const w = n.widgets.find(w => w.name === 'hades_combo_' + name || w.name === 'hades_text_' + name || w.name === 'hades_slider_' + name
            || w.name === 'hades_toggle_' + name) || n.widgets.find(w => w.name === name);
        const y = (w.last_y ?? w.y ?? 0) + (w.computedHeight ?? 24) / 2;
        return window.__hv.screen(id, n.size[0] * 0.45, y); }""", [node, widget])
    await d.move(x, y, ms)
    if values:
        await d.hold(0.3)
        await d.js("([id, v]) => window.__hv.set(id, v)", [node, values])


async def basic_run(d: Director, saver):
    """Press Run, fast-forward the render; returns (image url, seconds from execution start to success)."""
    run = d.page.locator("button:has-text('Run')").first
    b = await run.bounding_box()
    await d.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    async with d.fast(1.6):
        await d.hold(1.0)
        for _ in range(1200):
            if await d.js("() => fetch('/queue').then(r => r.json()).then(q => q.queue_running.length + q.queue_pending.length)") == 0:
                break
            await asyncio.sleep(0.25)
        else:
            raise SystemExit("the render did not finish in 5 minutes")
        await d.hold(0.5)
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    st = run["status"]
    if st.get("status_str") != "success":
        raise SystemExit(f"the render failed: {st}")
    stamps = {m[0]: m[1]["timestamp"] for m in st["messages"]}
    secs = (stamps["execution_success"] - stamps["execution_start"]) / 1000
    img = run["outputs"][str(saver)]["images"][0]
    url = f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
    print(f"render: {secs:.2f} s {img['filename']}", flush=True)
    return url, secs


def basic_crops_html(urls, labels):
    """Face and hands of both images at 100 %: 512 image pixels are 512 video pixels whatever the device scale."""
    side = 512 * 1920 / W
    cells = []
    for row in range(2):
        for k, url in enumerate(urls):
            x, y = BASIC_CROPS[k][row]
            cells.append(f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:6px;position:relative">'
                         f'<img src="{url}" style="position:absolute;width:{1024 * side / 512}px;height:{1024 * side / 512}px;'
                         f'left:{-x * side / 512}px;top:{-y * side / 512}px;max-width:none"></div>')
    head = "".join(f'<div style="color:#fff;font:600 24px Segoe UI,sans-serif;text-align:center">{t}</div>' for t in labels)
    return (f'<div style="display:grid;grid-template-columns:{side}px {side}px;gap:14px 28px">{head}{"".join(cells)}</div>')


async def basic_generation(d: Director):
    """The first tutorial: an empty canvas, six nodes added one by one, 25 steps on Base, then Turbo 8-steps v1.0."""
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "Your first MOSS-Image workflow, from scratch", 7.0)
    for i in range(4):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. the Manager (installed here: it lists the Suite with update / uninstall buttons)
    await d.say(4)
    mgr = d.page.get_by_role("button", name="Manager").first
    b = await mgr.bounding_box()
    await d.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    await d.hold(1.2)
    cnm = await d.page.get_by_text("Custom Nodes Manager").first.bounding_box()
    await d.click(cnm["x"] + cnm["width"] / 2, cnm["y"] + cnm["height"] / 2)
    await d.page.wait_for_selector("input.cn-manager-keywords", timeout=30000)
    await d.hold(1.2)
    sb = await d.page.locator("input.cn-manager-keywords").bounding_box()
    await d.click(sb["x"] + 60, sb["y"] + sb["height"] / 2)
    await d.page.keyboard.type("HaDeS MOSS", delay=70)
    await d.page.keyboard.press("Enter")
    await d.hold(1.0)
    row = await d.page.get_by_text("HaDeS MOSS Suite", exact=False).last.bounding_box()
    await d.hover(row["x"] + row["width"] / 2, row["y"] + row["height"] / 2)
    await d.say(5)
    btn = d.page.locator("button:has-text('Try update'), button:has-text('Uninstall')").first
    if await btn.count():
        bb = await btn.bounding_box()
        await d.hover(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
    await d.say(6)
    await d._after_voice(0.2)
    for _ in range(3):  # only buttons whose class says close, inside the Manager (never by position)
        close = await d.page.evaluate("""() => {
            const inManager = (e) => { let p = e; while (p && p.getBoundingClientRect().width < 900) p = p.parentElement;
                return Boolean(p && p !== document.body && (p.querySelector('input.cn-manager-keywords') || /ComfyUI Manager V\\d/.test(p.innerText.slice(0, 200)))); };
            const v = [...document.querySelectorAll('button.cm-ui-close, button[class*="close"], .comfy-modal-close')]
            .map(e => [e, e.getBoundingClientRect()]).filter(([e, r]) => r.width > 0 && r.width < 60 && getComputedStyle(e).visibility !== 'hidden' && inManager(e));
            const last = v[v.length - 1]; return last ? [last[1].x + last[1].width / 2, last[1].y + last[1].height / 2] : null; }""")
        if not close:
            break
        await d.click(*close)
        await d.hold(0.7)
    if await d.page.locator("input.cn-manager-keywords:visible").count():
        raise SystemExit("the Manager list did not close")

    # 3. the empty canvas: five things
    await d.move(W * 0.6, H * 0.62)
    await d.say(7)
    chips = ["Loader", "Prompt", "Latent", "Sampler", "Saver"]
    await d.js("""(labels) => { const el = document.createElement('div'); el.id = '__hv_chips';
        Object.assign(el.style, {zoom: String(Math.max(1, innerWidth / 1920)), position: 'fixed', left: '0', right: '0', top: '28%', display: 'flex', justifyContent: 'center', gap: '26px',
            zIndex: 2147483500, pointerEvents: 'none', transition: 'opacity .5s ease'});
        el.innerHTML = labels.map((t, i) => `<div style="opacity:0;transform:translateY(14px);transition:all .45s ease ${0.5 + i * 0.9}s;
            padding:16px 30px;border-radius:12px;background:#24242b;border:2px solid #ffcc33;color:#fff;font:600 30px Segoe UI,sans-serif">${t}</div>`).join(
            '<div style="color:#ffcc33;font:600 30px Segoe UI,sans-serif;align-self:center">→</div>');
        document.body.appendChild(el);
        requestAnimationFrame(() => requestAnimationFrame(() => el.querySelectorAll('div').forEach(c => { c.style.opacity = '1'; c.style.transform = 'none'; }))); }""", chips)
    await d.say(8)
    await d._after_voice(0.3)
    await d.js("() => { const el = document.getElementById('__hv_chips'); el.style.opacity = '0'; setTimeout(() => el.remove(), 600); }")
    await d.hold(0.6)

    # 4. MOSS Loader
    await d.say(9)
    loader = await basic_add(d, "HaDeSMossLoader", "MOSS Loader")
    await d.camera([loader], 1200, pad=200, top=60)
    await d.say(10)
    await d._after_voice(0.1)
    await d.say(11)
    await basic_point(d, loader, "unet_name", {"unet_name": BASIC_LOADER["unet_name"]})
    await d._after_voice(0.1)
    await d.say(12)
    await basic_point(d, loader, "clip_name", {"clip_name": BASIC_LOADER["clip_name"]})
    await d.hold(1.2)
    await basic_point(d, loader, "vae_name", {"vae_name": BASIC_LOADER["vae_name"]})

    # 5. the prompt
    await d.camera({"box": BASIC_BOX}, 1300, pad=60, top=60)
    await d.say(13)
    enhancer = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer")
    await d.camera([loader, enhancer], 1200, pad=50, top=60)
    await basic_point(d, enhancer, "mode", {"mode": "as written"})
    await d._after_voice(0.1)
    await d.say(14)
    box = await d.js("""(id) => { const w = window.__hv.widget(id, 'idea'); w.value = ''; if (w.element) w.element.value = '';
        const r = w.element.getBoundingClientRect(); return [r.left, r.top]; }""", enhancer)
    await d.click(box[0] + 40, box[1] + 30)
    if not await d.js("() => document.activeElement?.tagName === 'TEXTAREA'"):
        raise SystemExit("the prompt box is not focused; nothing typed")
    await d.page.keyboard.type(BASIC_PROMPT, delay=16)
    await d.move(W * 0.5, H * 0.85)
    await d._after_voice(0.1)
    await d.say(15)
    await basic_link(d, loader, "CLIP", enhancer, "clip")

    # 6. MOSS Latent
    await d.camera({"box": BASIC_BOX}, 1300, pad=60, top=60)
    await d.say(16)
    latent = await basic_add(d, "HaDeSMossLatent", "MOSS Latent")
    await d.camera([latent], 1200, pad=200, top=60)
    await basic_point(d, latent, "aspect_ratio", {"aspect_ratio": "1:1", "megapixels": 1.0})
    await d._after_voice(0.1)
    await d.say(17)
    await basic_point(d, latent, "seed", {"seed": BASIC_SEED})
    await d.hold(0.8)
    await basic_point(d, latent, "control_after_generate", {"control_after_generate": "fixed"})

    # 7. MOSS KSampler and its wires
    await d.camera({"box": BASIC_BOX}, 1300, pad=60, top=60)
    await d.say(18)
    sampler = await basic_add(d, "HaDeSMossKSampler", "MOSS KSampler")
    await d.camera([loader, enhancer, latent, sampler], 1300, pad=40, top=60)
    await d._after_voice(0.1)
    await d.say(19)
    await basic_link(d, loader, "MODEL", sampler, "model")
    await basic_link(d, enhancer, "positive", sampler, "positive")
    await basic_link(d, enhancer, "negative", sampler, "negative")
    await basic_link(d, latent, "latent", sampler, "latent_image")
    await basic_link(d, latent, "seed", sampler, "seed")
    await d._after_voice(0.1)
    await d.say(20)
    await basic_link(d, loader, "VAE", sampler, "vae")
    await d._after_voice(0.1)
    await d.camera([sampler], 1200, pad=160, top=60)
    await d.say(21)
    await basic_point(d, sampler, "steps", {"steps": 25})
    await d.hold(0.6)
    await basic_point(d, sampler, "cfg", {"cfg": 2.5})
    await d.hold(0.6)
    await basic_point(d, sampler, "sampler_name", {"sampler_name": "euler"})
    await d.hold(0.6)
    await basic_point(d, sampler, "scheduler", {"scheduler": "simple"})

    # 8. MOSS Image Saver
    await d.camera({"box": BASIC_BOX}, 1300, pad=60, top=60)
    await d.say(22)
    saver = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver")
    await d.camera([sampler, saver], 1200, pad=60, top=60)
    await basic_link(d, sampler, "image", saver, "images")
    await d._after_voice(0.1)

    # 9. first run, 25 steps
    await d.camera([loader, enhancer, latent, sampler, saver], 1300, pad=50, top=60)
    await d.say(23)
    await d._after_voice(0.1)
    url25, t25 = await basic_run(d, saver)
    await d.camera([saver], 1200, pad=120, top=60)
    await d.hold(0.8)
    await d.say(24)
    await d.js("(h) => window.__hv.show(h)", f'<div style="text-align:center"><img src="{url25}" style="height:760px;border-radius:6px">'
               f'<div style="color:#ffcc33;font:600 28px Segoe UI,sans-serif;margin-top:14px">MOSS-Image Base · 25 steps · {t25:.1f} s</div></div>')
    await d._after_voice(0.6)
    await d.js("() => window.__hv.hide()")
    await d.hold(0.6)

    # 10. the turbo
    await d.camera({"box": BASIC_BOX}, 1300, pad=60, top=60)
    await d.say(25)
    await d._after_voice(0.1)
    await d.say(26)
    turbo = await basic_add(d, "HaDeSMossTurbo", "MOSS Turbo LoRA")
    await d.camera([turbo], 1200, pad=180, top=60)
    await basic_point(d, turbo, "lora_name", {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0,
                                              "sampler_name": "euler", "scheduler": "simple"})
    await d._after_voice(0.1)
    await d.say(27)
    await d.hold(1.0)
    await basic_point(d, turbo, "steps")
    await d._after_voice(0.1)
    await d.camera([turbo, sampler], 1200, pad=60, top=60)
    await d.say(28)
    await basic_link(d, turbo, "turbo", sampler, "turbo")
    await d.camera([turbo], 1000, pad=180, top=60)
    await basic_point(d, turbo, "steps")
    await d.hold(0.6)
    await basic_point(d, turbo, "cfg")
    await d._after_voice(0.1)
    await d.say(29)
    await d._after_voice(0.1)

    # 11. second run, 8 steps
    await d.camera([loader, enhancer, latent, sampler, saver, turbo], 1300, pad=50, top=60)
    await d.say(30)
    await d._after_voice(0.1)
    url8, t8 = await basic_run(d, saver)
    await d.camera([saver], 1200, pad=120, top=60)
    await d.hold(0.8)
    await d.say(31)
    await d._after_voice(0.2)
    await d.say(32)
    side = 820
    await d.js("(h) => window.__hv.show(h)", '<div style="display:flex;gap:40px">' + "".join(
        f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px">'
        f'<div style="color:#ffcc33;font:600 26px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>'
        for u, t in ((url25, f"Base · 25 steps · {t25:.1f} s"), (url8, f"Base + Turbo 8-steps v1.0 · 8 steps · {t8:.1f} s"))) + "</div>")
    await d.move(W - 60, H - 90)
    await d._after_voice(0.4)
    await d.say(33)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               basic_crops_html([url25, url8], [f"25 steps · {t25:.1f} s", f"Turbo · 8 steps · {t8:.1f} s"]))
    await d._after_voice(0.8)
    await d.js("() => window.__hv.hide()")
    await d.hold(0.6)

    # 12. tips
    await d.camera([latent], 1200, pad=200, top=60)
    await d.say(34)
    await basic_point(d, latent, "control_after_generate")
    await d._after_voice(0.1)
    await d.camera([turbo], 1200, pad=180, top=60)
    await d.say(35)
    await basic_point(d, turbo, "enabled")
    await d._after_voice(0.1)
    await d.say(36)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:260px">'
               '<div style="font:900 64px Segoe UI Black,sans-serif">HaDeS MOSS v1.0 Turbo</div>'
               '<div style="font:600 32px Segoe UI,sans-serif;color:#ffcc33;margin-top:20px">MOSS-Image Base with Turbo 8-steps v1.0 baked in'
               ' · 8 steps · CFG 1</div><div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:36px">'
               'Link in the description</div></div>')
    await d._after_voice(0.4)
    await d.js("() => window.__hv.hide()")

    # 13. the whole workflow
    await d.camera([loader, enhancer, latent, sampler, saver, turbo], 1400, pad=50, top=60)
    await d.say(37)
    await d.move(W * 0.5, H * 0.9)
    await d._after_voice(0.1)
    await d.say(38)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Basic Generation.json</div></div>')
    await d._after_voice(0.3)

    # 14. goodbye
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:60px"><img src="{url8}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Your first MOSS-Image workflow</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(39)
    await d.say(40)
    await d.end(1.5)


basic_generation.prepare = basic_prepare


# ---- bigger images: upscale x2 + telemetry (the second tutorial) ----

UP_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("In the last video we built a MOSS-Image workflow from an empty canvas and made this fisherman in eight steps.",
     "En el vídeo anterior montamos un workflow de MOSS-Image desde un lienzo vacío e hicimos este pescador en ocho pasos."),
    ("Today we build it again, and then we make it twice as big, in two different ways. I'll show you which one I use, and why.",
     "Hoy lo volvemos a montar y después lo hacemos el doble de grande, de dos maneras distintas. Te enseño cuál uso yo y por qué."),
    ("We'll also meet Suite Telemetry, the node that checks your whole workflow and writes your settings into the image.",
     "Además conoceremos Suite Telemetry, el nodo que revisa todo tu workflow y escribe tus ajustes dentro de la imagen."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 5: quick rebuild
    ("Quick rebuild. If you missed the first video, every node is explained there.",
     "Montaje rápido. Si te perdiste el primer vídeo, ahí está explicado cada nodo."),
    ("MOSS Loader, with MOSS-Image Base, the int8 text encoder and the Flux 2 VAE.",
     "MOSS Loader, con MOSS-Image Base, el text encoder int8 y el VAE de Flux 2.",
     "MOSS Loader, with MOSS-Image Base, the int eight text encoder and the Flux two VAE."),
    ("The Prompt Enhancer, on \"as written\", with our fisherman.",
     "El Prompt Enhancer, en \"as written\", con nuestro pescador.",
     "The Prompt Enhancer, on as written, with our fisherman."),
    ("MOSS Latent, square, 1024, with the same seed as last time.",
     "MOSS Latent, cuadrado, 1024, con la misma semilla que la otra vez.",
     "MOSS Latent, square, ten twenty-four, with the same seed as last time."),
    ("The MOSS KSampler, with my Turbo 8-steps v1.0 plugged into its turbo input.",
     "El MOSS KSampler, con mi Turbo 8-steps v1.0 conectado a su entrada turbo.",
     "The MOSS K Sampler, with my Turbo eight steps, version one, plugged into its turbo input."),
    ("And the MOSS Image Saver at the end.",
     "Y el MOSS Image Saver al final."),
    # 11: Suite Telemetry
    ("Now something new: Suite Telemetry.",
     "Ahora algo nuevo: Suite Telemetry."),
    ("Almost every node in the Suite has an info output. It describes what that node really did: the files it loaded, the size, "
     "the seed, the steps.",
     "Casi todos los nodos de la Suite tienen una salida info. Describe lo que ese nodo ha hecho de verdad: los archivos que cargó, "
     "el tamaño, la semilla, los pasos."),
    ("I connect the info of the loader, the prompt, the latent, the sampler and the turbo. Telemetry reads them all together.",
     "Conecto el info del loader, del prompt, del latent, del sampler y del turbo. Telemetry los lee todos juntos."),
    ("Then it checks the workflow against what I've measured, and writes a report: errors, warnings and notes.",
     "Después revisa el workflow comparándolo con lo que he medido y escribe un informe: errores, avisos y notas."),
    ("For example, it knows the turbo was made for eight steps at CFG one. If your sampler uses other settings, it tells you.",
     "Por ejemplo, sabe que el turbo está hecho para ocho pasos con CFG uno. Si tu sampler usa otros ajustes, te lo dice."),
    ("And its json output goes into the Image Saver. That's what writes the prompt, the settings and the model hashes into the PNG.",
     "Y su salida json va al Image Saver. Es lo que escribe el prompt, los ajustes y los hashes de los modelos dentro del PNG.",
     "And its jason output goes into the Image Saver. That's what writes the prompt, the settings and the model hashes into the P N G."),
    ("So when you upload the image to Civitai, it shows exactly how you made it.",
     "Así, cuando subes la imagen a Civitai, se ve exactamente cómo la hiciste."),
    ("Let's run it.",
     "Vamos a lanzarlo."),
    ("Here's our fisherman, pixel for pixel the same image as in the last video. And here's the report: no findings. "
     "Everything matches what I've measured.",
     "Aquí está nuestro pescador, píxel a píxel la misma imagen que en el vídeo anterior. Y aquí está el informe: sin avisos. "
     "Todo cuadra con lo que he medido."),
    # 20: upscale model
    ("One thousand twenty-four pixels is fine for the screen, but sometimes you want it bigger. The first way is an upscale model.",
     "Mil veinticuatro píxeles está bien para la pantalla, pero a veces la quieres más grande. La primera manera es un modelo de escalado."),
    ("I add Load Upscale Model with RealESRGAN x2 plus. It goes in the upscale models folder; the link is in the description.",
     "Añado Load Upscale Model con RealESRGAN x2 plus. Va en la carpeta upscale_models; el enlace está en la descripción.",
     "I add Load Upscale Model with Real E S R GAN, times two plus. It goes in the upscale models folder; the link is in the description."),
    ("Then Upscale Image using Model, fed with the image from the sampler, and another Image Saver.",
     "Después Upscale Image using Model, con la imagen del sampler, y otro Image Saver."),
    ("It takes about one second and doubles the image to 2048. But nothing is redrawn: it only guesses the missing pixels.",
     "Tarda alrededor de un segundo y dobla la imagen a 2048. Pero no se redibuja nada: solo adivina los píxeles que faltan.",
     "It takes about one second and doubles the image to twenty forty-eight. But nothing is redrawn: it only guesses the missing pixels."),
    ("Up close, at full size, you can see it: the beard and the wool look smooth, almost painted.",
     "De cerca, a tamaño real, se nota: la barba y la lana se ven lisas, casi pintadas."),
    # 25: the MOSS pass
    ("The second way is to let MOSS redraw the upscaled image.",
     "La segunda manera es dejar que MOSS redibuje la imagen escalada."),
    ("VAE Encode turns the big image back into a latent. A second MOSS KSampler, with the same turbo, works on it.",
     "VAE Encode vuelve a convertir la imagen grande en latente. Un segundo MOSS KSampler, con el mismo turbo, trabaja sobre él.",
     "V A E Encode turns the big image back into a latent. A second MOSS K Sampler, with the same turbo, works on it."),
    ("The key is denoise: zero point two five. It keeps the picture and only redraws the fine detail.",
     "La clave es el denoise: cero coma veinticinco. Conserva la imagen y solo redibuja el detalle fino."),
    ("Let's run it. At 2048 this pass takes about eleven seconds.",
     "Vamos a lanzarlo. A 2048, este pase tarda unos once segundos.",
     "Let's run it. At twenty forty-eight this pass takes about eleven seconds."),
    ("And now, up close: skin pores, separate hairs in the beard and sharper threads in the net, where the upscaler only smoothed.",
     "Y ahora, de cerca: poros en la piel, los pelos de la barba separados y los hilos de la red más nítidos, donde el escalador "
     "solo suavizaba."),
    ("Telemetry notices this too: it adds a note that the turbo is running below denoise one. That's expected here.",
     "Telemetry también lo ve: añade una nota de que el turbo trabaja por debajo de denoise uno. Aquí es lo esperado."),
    # 31: which one, tips
    ("So which one do I use? For a quick bigger image, the upscaler alone. For the final picture, the MOSS pass.",
     "¿Y cuál uso? Para una imagen más grande rápida, el escalador solo. Para la imagen final, el pase MOSS."),
    ("If you want a little more detail, try denoise zero point four. It also changes the picture a bit more.",
     "Si quieres un poco más de detalle, prueba denoise cero coma cuatro. También cambia un poco más la imagen."),
    ("Your seed is still fixed, so you can try both without losing the image.",
     "La semilla sigue fija, así que puedes probar las dos sin perder la imagen."),
    ("One thing about metadata. With two samplers in the workflow, the PNG's settings come from the first one, the 1024 image. "
     "The Image Saver tells you so.",
     "Una cosa sobre los metadatos. Con dos samplers en el workflow, los ajustes del PNG salen del primero, el de la imagen de 1024. "
     "El Image Saver te lo avisa.",
     "One thing about metadata. With two samplers in the workflow, the P N G's settings come from the first one, the ten twenty-four "
     "image. The Image Saver tells you so."),
    ("You can choose which stage goes into the image, but metadata deserves a video of its own. I'll make one entirely about it, "
     "so stay tuned.",
     "Se puede elegir qué etapa va dentro de la imagen, pero los metadatos merecen un vídeo para ellos solos. Haré uno dedicado "
     "enteramente a ese tema, así que atento."),
    # 36: goodbye
    ("That's it: one image, made bigger two ways, with the whole recipe written inside the PNG.",
     "Y ya está: una imagen, agrandada de dos maneras, con toda la receta escrita dentro del PNG.",
     "That's it: one image, made bigger two ways, with the whole recipe written inside the P N G."),
    ("This exact workflow is free to download from the description.",
     "Este mismo workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

# where each node goes, in graph space: the first video's row, Telemetry under the sampler, the upscale to the right
UP_POS = {"loader": (0, 0), "enhancer": (440, 0), "latent": (940, 0), "sampler": (1380, 0), "saver": (1820, 0), "turbo": (940, 450),
          "telemetry": (1380, 640), "preview": (1820, 640), "upmodel": (2300, 0), "upscale": (2300, 150), "saver2": (2620, 0),
          "encode": (2300, 640), "sampler2": (2500, 640), "saver3": (2940, 640)}
UP_BOX = [-260, -60, 3380, 1240]
UP_UPSCALER = "RealESRGAN_x2plus.pth"
# 100 % crops at 2048 (top-left corners, 512 square), checked on the measured images
UP_CROPS = {"face": (790, 200), "hands": (620, 1340)}
UP_VIDEO1 = Path("I:/HaDeS_MOSS/videos/basic_generation/civitai/fisherman_turbo_v1.0_8steps_seed1013.png")
# denoise 0.40 is not run on camera: the measured image (same clean instance, same seed and settings; tools/RESULTS.md)
UP_D40 = "/view?filename=s3_d40_seed1013_00001_.png&subfolder=upscale_measure&type=output"

UP_WARM_JS = r"""
async (g) => {
  const r = await (await fetch('/prompt', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt: g})})).json();
  for (;;) { const h = await (await fetch('/history/' + r.prompt_id)).json(); if (h[r.prompt_id]?.status?.completed !== undefined) return h[r.prompt_id].status.status_str; await new Promise(f => setTimeout(f, 500)); }
}
"""


def up_api_graph(seed, denoise=0.25, full=True):
    """The pipeline in API form, for the off-camera warm-up (models, LoRA and upscaler loaded); full=False: the 1024 stage only."""
    ks = lambda latent, d: {"class_type": "HaDeSMossKSampler", "inputs": {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 1], "latent_image": latent, "seed": ["3", 4], "steps": 25,
        "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": d, "flow_shift": 0.0, "slg": "off",
        "vae": ["1", 2], "denoise_mode": "steps / denoise (core)", "turbo": ["4", 0]}}
    g = {"1": {"class_type": "HaDeSMossLoader", "inputs": dict(BASIC_LOADER)},
            "2": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {"clip": ["1", 1], "idea": BASIC_PROMPT, "mode": "as written",
                  "width": 1024, "height": 1024, "temperature": 0.7, "seed": 1, "max_tokens": 1024, "negative": ""}},
            "3": {"class_type": "HaDeSMossLatent", "inputs": {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024,
                  "custom_height": 1024, "batch_size": 1, "seed": seed}},
            "4": {"class_type": "HaDeSMossTurbo", "inputs": {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0,
                  "sampler_name": "euler", "scheduler": "simple", "enabled": True}},
            "5": ks(["3", 0], 1.0),
            "7": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UP_UPSCALER}},
            "8": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["7", 0], "image": ["5", 1]}},
            "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["1", 2]}},
            "11": ks(["10", 0], denoise),
            "13": {"class_type": "HaDeSMossSuiteTelemetry", "inputs": {"report_title": "HaDeS MOSS Suite Telemetry",
                   "info_1": ["1", 3], "info_2": ["2", 6], "info_3": ["3", 3], "info_4": ["5", 2], "info_5": ["4", 1]}},
            "6": {"class_type": "HaDeSMossImageSaver", "inputs": {"images": ["5", 1], "filename_prefix": "warmup/w", "hashes": True,
                  "embed_workflow": True, "metadata": ["13", 1]}},
            "9": {"class_type": "HaDeSMossImageSaver", "inputs": {"images": ["8", 0], "filename_prefix": "warmup/w", "hashes": True,
                  "embed_workflow": True, "metadata": ["13", 1]}},
            "12": {"class_type": "HaDeSMossImageSaver", "inputs": {"images": ["11", 1], "filename_prefix": "warmup/w", "hashes": True,
                  "embed_workflow": True, "metadata": ["13", 1]}}}
    if not full:
        g = {k: v for k, v in g.items() if k in ("1", "2", "3", "4", "5", "6", "13")}
    return g


async def up_prepare(page):
    """Off camera: the whole pipeline once at another seed loads the model, text encoder, VAE, LoRA and upscaler."""
    print("warm-up runs:", await page.evaluate(UP_WARM_JS, up_api_graph(7)), await page.evaluate(UP_WARM_JS, up_api_graph(8, full=False)),
          flush=True)


def up_crops_html(cols, rows):
    """Columns of 100 % crops: cols = [(url, label, pixelated)], rows = crop corners at 2048; a 1024 image is shown doubled."""
    side = 512 * 1920 / W
    full = 2048 * side / 512
    head = "".join(f'<div style="color:#fff;font:600 24px Segoe UI,sans-serif;text-align:center">{t}</div>' for _, t, _ in cols)
    cells = []
    for x, y in rows:
        for url, _, pix in cols:
            cells.append(f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:6px;position:relative">'
                         f'<img src="{url}" style="position:absolute;width:{full}px;height:{full}px;left:{-x * side / 512}px;'
                         f'top:{-y * side / 512}px;max-width:none;{"image-rendering:pixelated;" if pix else ""}"></div>')
    return (f'<div style="display:grid;grid-template-columns:repeat({len(cols)},{side}px);gap:14px 28px">{head}{"".join(cells)}</div>')


async def up_set(d, node, values):
    await d.js("([id, v]) => window.__hv.set(id, v)", [node, values])


async def upscale_telemetry(d: Director):
    """The second tutorial: the first video's workflow rebuilt quickly, Suite Telemetry, RealESRGAN x2 and a MOSS pass at 0.25."""
    P = UP_POS
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction, the fisherman of the first video
    d.title("HaDeS MOSS Suite", "Bigger images: Upscale x2 + Telemetry", 7.0)
    await d.say(0)
    await d.say(1)
    first = "data:image/png;base64," + base64.b64encode(UP_VIDEO1.read_bytes()).decode()
    await d.js("(h) => window.__hv.show(h)", f'<div style="text-align:center"><img src="{first}" style="height:760px;border-radius:6px">'
               '<div style="color:#ffcc33;font:600 28px Segoe UI,sans-serif;margin-top:14px">Your first MOSS-Image workflow · '
               'Base + Turbo 8-steps v1.0 · 8 steps · 1024 x 1024</div></div>')
    await d._after_voice(0.2)
    await d.say(2)
    await d.hold(2.0)
    await d.js("() => window.__hv.hide()")
    await d.say(3)
    await d.move(W * 0.55, H * 0.6, 2500)
    await d.say(4)
    await d.move(W * 0.6, H * 0.55, 2500)

    # 2. quick rebuild of the first video
    await d.say(5)
    loader = await basic_add(d, "HaDeSMossLoader", "MOSS Loader", P["loader"])
    await up_set(d, loader, BASIC_LOADER)
    await d.camera([loader], 1000, pad=200, top=60)
    await d.say(6)
    await basic_point(d, loader, "unet_name")
    await d.hold(0.5)
    await basic_point(d, loader, "clip_name")
    await d.hold(0.5)
    await basic_point(d, loader, "vae_name")
    await d.camera({"box": BASIC_BOX}, 1000, pad=60, top=60)
    await d.say(7)
    enhancer = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enhancer"])
    await d.camera([loader, enhancer], 1000, pad=50, top=60)
    await basic_point(d, enhancer, "mode", {"mode": "as written"})
    box = await d.js("""(id) => { const w = window.__hv.widget(id, 'idea'); w.value = ''; if (w.element) w.element.value = '';
        const r = w.element.getBoundingClientRect(); return [r.left, r.top]; }""", enhancer)
    await d.click(box[0] + 40, box[1] + 30)
    if not await d.js("() => document.activeElement?.tagName === 'TEXTAREA'"):
        raise SystemExit("the prompt box is not focused; nothing typed")
    async with d.fast(2.0):
        await d.page.keyboard.type(BASIC_PROMPT, delay=12)
        await basic_link(d, loader, "CLIP", enhancer, "clip", 800)
    await d.camera({"box": BASIC_BOX}, 1000, pad=60, top=60)
    await d.say(8)
    latent = await basic_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
    await basic_point(d, latent, "seed")
    await d.hold(0.4)
    await d.say(9)
    sampler = await basic_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["sampler"])
    async with d.fast(6.0):
        await d.camera([loader, enhancer, latent, sampler], 1000, pad=40, top=60)
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae")):
            await basic_link(d, a, out, sampler, inp, 700)
        await d.camera({"box": BASIC_BOX}, 1000, pad=60, top=60)
        turbo = await basic_add(d, "HaDeSMossTurbo", "MOSS Turbo LoRA", P["turbo"])
        await up_set(d, turbo, {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                "scheduler": "simple"})
    await basic_link(d, turbo, "turbo", sampler, "turbo", 800)
    await d.say(10)
    saver = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["saver"])
    await basic_link(d, sampler, "image", saver, "images", 700)
    await d._after_voice(0.1)

    # 3. Suite Telemetry
    await d.camera({"box": UP_BOX}, 1200, pad=60, top=60)
    await d.say(11)
    telemetry = await basic_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["telemetry"])
    await d.say(12)
    await d.camera([loader, enhancer, latent, sampler], 1200, pad=40, top=60)
    for n, out in ((loader, "info"), (enhancer, "info"), (latent, "info"), (sampler, "info")):
        x, y = await d.js("([id, n]) => window.__hv.slot(id, n, true)", [n, out])
        await d.move(x, y, 700)
        await d.hold(0.5)
    await d._after_voice(0.1)
    await d.camera({"box": [-60, -60, 1900, 1000]}, 1200, pad=40, top=60)
    await d.say(13)
    async with d.fast(5.0):
        for k, (n, out) in enumerate(((loader, "info"), (enhancer, "info"), (latent, "info"), (sampler, "info"), (turbo, "info"))):
            await basic_link(d, n, out, telemetry, f"info_{k + 1}", 800)
    await d._after_voice(0.1)
    await d.camera([sampler, saver, telemetry], 1200, pad=60, top=60)
    async with d.fast(3.0):
        preview = await basic_add(d, "PreviewAny", "Preview as Text", P["preview"])
        await d.js("(id) => { const n = window.__hv.node(id); n.setSize([440, 560]); n.setDirtyCanvas(true, true); }", preview)
        await basic_link(d, telemetry, "report", preview, "source", 700)
    await d.camera([telemetry, preview], 1200, pad=60, top=60)
    await d.say(14)
    await d._after_voice(0.1)
    await d.say(15)
    await d._after_voice(0.1)
    await d.camera([sampler, saver, telemetry], 1200, pad=60, top=60)
    await d.say(16)
    await basic_link(d, telemetry, "json", saver, "metadata", 800)
    await d._after_voice(0.1)
    await d.say(17)
    await d._after_voice(0.1)

    # 4. first run
    await d.camera({"box": UP_BOX}, 1200, pad=60, top=60)
    await d.say(18)
    await d._after_voice(0.1)
    url1, t1 = await basic_run(d, saver)
    await d.move(W * 0.5, H * 0.92)
    await d.say(19)
    await d.js("(h) => window.__hv.show(h)", f'<div style="text-align:center"><img src="{url1}" style="height:760px;border-radius:6px">'
               f'<div style="color:#ffcc33;font:600 28px Segoe UI,sans-serif;margin-top:14px">Base + Turbo 8-steps v1.0 · 8 steps · '
               f'1024 x 1024 · {t1:.1f} s</div></div>')
    await d.hold(3.0)
    await d.js("() => window.__hv.hide()")
    await d.camera([preview], 1200, pad=40, top=60)
    await d._after_voice(0.6)

    # 5. RealESRGAN x2
    await d.camera({"box": UP_BOX}, 1200, pad=60, top=60)
    await d.say(20)
    await d._after_voice(0.1)
    await d.say(21)
    upmodel = await basic_add(d, "UpscaleModelLoader", "Load Upscale Model", P["upmodel"])
    await up_set(d, upmodel, {"model_name": UP_UPSCALER})
    await d.camera([saver, upmodel], 1000, pad=80, top=60)
    await basic_point(d, upmodel, "model_name")
    await d._after_voice(0.1)
    await d.camera({"box": [1340, -60, 3060, 700]}, 1000, pad=40, top=60)
    await d.say(22)
    upscale = await basic_add(d, "ImageUpscaleWithModel", "Upscale Image using Model", P["upscale"])
    async with d.fast(4.0):
        await basic_link(d, upmodel, "UPSCALE_MODEL", upscale, "upscale_model", 700)
        await basic_link(d, sampler, "image", upscale, "image", 900)
        saver2 = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["saver2"])
        await basic_link(d, upscale, "IMAGE", saver2, "images", 700)
        await basic_link(d, telemetry, "json", saver2, "metadata", 900)
    await d._after_voice(0.1)
    url2, t2 = await basic_run(d, saver2)
    await d.move(W * 0.5, H * 0.92)
    await d.camera([saver2], 1000, pad=120, top=60)
    await d.say(23)
    await d._after_voice(0.1)
    await d.say(24)
    await d.js("(h) => window.__hv.show(h)", up_crops_html(
        [(url1, "1024, pixels doubled", True), (url2, f"RealESRGAN x2 · 2048 · {t2:.1f} s", False)], [UP_CROPS["face"], UP_CROPS["hands"]]))
    await d.move(W - 60, H - 90)
    await d._after_voice(1.0)
    await d.js("() => window.__hv.hide()")
    await d.hold(0.6)

    # 6. the MOSS pass at denoise 0.25
    await d.camera({"box": UP_BOX}, 1200, pad=60, top=60)
    await d.say(25)
    await d._after_voice(0.1)
    await d.say(26)
    encode = await basic_add(d, "VAEEncode", "VAE Encode", P["encode"])
    await d.camera({"box": [-60, -60, 3360, 1240]}, 1000, pad=40, top=60)
    await basic_link(d, upscale, "IMAGE", encode, "pixels", 800)
    await basic_link(d, loader, "VAE", encode, "vae", 900)
    async with d.fast(7.0):
        sampler2 = await basic_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["sampler2"])
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (encode, "LATENT", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae"), (turbo, "turbo", "turbo")):
            await basic_link(d, a, out, sampler2, inp, 700)
        saver3 = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["saver3"])
        await basic_link(d, sampler2, "image", saver3, "images", 700)
        await basic_link(d, telemetry, "json", saver3, "metadata", 900)
    await d._after_voice(0.1)
    await d.camera([sampler2], 1100, pad=160, top=60)
    await d.say(27)
    await basic_point(d, sampler2, "denoise", {"denoise": 0.25})
    await d._after_voice(0.1)
    await d.camera({"box": UP_BOX}, 1200, pad=60, top=60)
    await d.say(28)
    await d._after_voice(0.1)
    url3, t3 = await basic_run(d, saver3)
    await d.move(W * 0.5, H * 0.92)
    await d.camera([saver3], 1000, pad=120, top=60)
    await d.hold(0.6)
    await d.say(29)
    await d.js("(h) => window.__hv.show(h)", up_crops_html(
        [(url2, "RealESRGAN x2", False), (url3, f"x2 + MOSS pass, denoise 0.25 · {t3:.1f} s", False)],
        [UP_CROPS["face"], UP_CROPS["hands"]]))
    await d.move(W - 60, H - 90)
    await d._after_voice(1.0)
    await d.js("() => window.__hv.hide()")
    await d.hold(0.6)

    # 7. Telemetry notices the pass
    await d.camera([sampler2, telemetry], 1100, pad=60, top=60)
    await d.say(30)
    await basic_link(d, sampler2, "info", telemetry, "info_6", 1000)
    _, t4 = await basic_run(d, saver3)
    await d.move(W * 0.5, H * 0.92)
    await d.camera([preview], 1100, pad=40, top=60)
    await d._after_voice(0.6)

    # 8. which one, tips
    await d.camera({"box": UP_BOX}, 1200, pad=60, top=60)
    await d.say(31)
    await d._after_voice(0.1)
    await d.say(32)
    await d.js("(h) => window.__hv.show(h)", up_crops_html(
        [(url3, "denoise 0.25", False), (UP_D40, "denoise 0.40", False)], [UP_CROPS["face"]]))
    await d.move(W - 60, H - 90)
    await d._after_voice(0.8)
    await d.js("() => window.__hv.hide()")
    await d.camera([latent], 1100, pad=200, top=60)
    await d.say(33)
    await basic_point(d, latent, "control_after_generate")
    await d._after_voice(0.1)
    # the saver's own note, read from the last run (the node does not draw it)
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    note = (run["outputs"].get(str(saver3), {}).get("text") or [""])[0]
    await d.camera([saver3], 1100, pad=140, top=60)
    await d.say(34)
    if note:
        await d.js("(h) => window.__hv.show(h)", '<div style="margin-top:300px;max-width:1500px;padding:36px 44px;border-radius:14px;'
                   'background:#24242b;border:2px solid #ffcc33;color:#fff;font:500 32px Segoe UI,sans-serif;line-height:1.4">'
                   '<div style="color:#ffcc33;font:600 26px Segoe UI,sans-serif;margin-bottom:14px">MOSS Image Saver</div>'
                   + note.replace("&", "&amp;").replace("<", "&lt;") + '</div>')
    await d._after_voice(0.1)
    await d.say(35)
    await d._after_voice(0.3)
    await d.js("() => window.__hv.hide()")

    # 9. goodbye
    ids = [loader, enhancer, latent, sampler, saver, turbo, telemetry, preview, upmodel, upscale, saver2, encode, sampler2, saver3]
    await d.camera(ids, 1400, pad=50, top=60)
    await d.say(36)
    await d.move(W * 0.5, H * 0.9)
    await d._after_voice(0.1)
    await d.say(37)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Turbo 8 steps + Upscale x2.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:60px"><img src="{url3}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Bigger images: Upscale x2 + Telemetry</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(38)
    await d.say(39)
    await d.end(1.5)


upscale_telemetry.prepare = up_prepare


# ---- the Prompt Enhancer, every mode (the third tutorial) ----

PE_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("Today it's all about one node: the MOSS Prompt Enhancer. It can take your text as it is, or write the whole prompt for you.",
     "Hoy todo va de un solo nodo: el MOSS Prompt Enhancer. Puede usar tu texto tal cual, o escribirte el prompt entero."),
    ("We'll go through every mode, with the same fisherman as in the last two videos, and see what each one writes and draws.",
     "Vamos a ver todos sus modos, con el mismo pescador de los dos vídeos anteriores, y lo que escribe y dibuja cada uno."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 4: quick rebuild
    ("Quick rebuild of the first video's workflow: loader, prompt enhancer, latent, sampler with my Turbo LoRA, and the image saver.",
     "Montaje rápido del workflow del primer vídeo: loader, prompt enhancer, latent, sampler con mi LoRA Turbo, y el image saver.",
     "Quick rebuild of the first video's workflow: loader, prompt enhancer, latent, sampler with my Turbo Lora, and the image saver."),
    ("One addition: a Preview as Text on the enhancer's prompt output, so we can read what it writes.",
     "Una novedad: un Preview as Text en la salida prompt del enhancer, para poder leer lo que escribe."),
    # 6: the node
    ("Here's the trick: MOSS-Image's text encoder is Qwen 3.5, a full language model. The enhancer uses that same file to write "
     "the prompt, and then to encode it.",
     "El truco es este: el text encoder de MOSS-Image es Qwen 3.5, un modelo de lenguaje completo. El enhancer usa ese mismo "
     "archivo para escribir el prompt y luego para codificarlo.",
     "Here's the trick: MOSS-Image's text encoder is Qwen three point five, a full language model. The enhancer uses that same "
     "file to write the prompt, and then to encode it."),
    ("So there's no extra model to download. On my RTX 5090, an answer takes about two seconds.",
     "Así que no hay que descargar ningún modelo más. En mi RTX 5090, cada respuesta tarda unos dos segundos.",
     "So there's no extra model to download. On my RTX fifty ninety, an answer takes about two seconds."),
    ("You write a short idea, pick a mode, and the seed and temperature decide how it's written.",
     "Escribes una idea corta, eliges un modo, y la semilla y la temperatura deciden cómo se escribe."),
    # 9: as written
    ("Let's start with a short idea: an old fisherman on a dock.",
     "Empezamos con una idea corta: un pescador viejo en un muelle."),
    ("In \"as written\" mode nothing is generated: the text is encoded exactly as you type it. That's what we used in the first two videos.",
     "En el modo \"as written\" no se genera nada: el texto se codifica tal cual lo escribes. Es lo que usamos en los dos primeros vídeos.",
     "In as written mode nothing is generated: the text is encoded exactly as you type it. That's what we used in the first two videos."),
    ("With so few words, MOSS has to guess the rest: a small figure from behind, fishing on a stony shore. Not even a dock.",
     "Con tan pocas palabras, MOSS tiene que adivinar el resto: una figura pequeña de espaldas, pescando en una orilla de piedras. "
     "Ni siquiera hay muelle."),
    # 12: caption
    ("Now caption mode. Same idea, but now the enhancer writes a paragraph that keeps to it.",
     "Ahora el modo caption. La misma idea, pero ahora el enhancer escribe un párrafo que se ciñe a ella."),
    ("Here's what it wrote: a face lined by age and salt, a faded blue cap and a wool coat, a fishing rod, calm grey water and a few seagulls.",
     "Esto es lo que ha escrito: una cara marcada por la edad y la sal, una gorra azul desteñida y un abrigo de lana, una caña de "
     "pescar, agua gris en calma y unas gaviotas."),
    ("About ninety words. In six tries, caption wrote between about sixty and a hundred and twenty. And here's the image.",
     "Unas noventa palabras. En seis pruebas, caption escribió entre unas sesenta y ciento veinte. Y aquí está la imagen."),
    # 15: detailed
    ("Detailed mode goes further: a brief of about two hundred words, written like a photographer or an illustrator would.",
     "El modo detailed va más allá: un encargo de unas doscientas palabras, escrito como lo haría un fotógrafo o un ilustrador."),
    ("Where the light comes from, the lens and the shot for a photo, the technique and materials for other media.",
     "De dónde viene la luz, el objetivo y el plano en una foto, la técnica y los materiales en otros medios."),
    ("Look at the first words: oil painting on textured canvas. Our idea didn't name a medium, so detailed picked one.",
     "Fíjate en las primeras palabras: óleo sobre lienzo con textura. Nuestra idea no decía el medio, así que detailed eligió uno."),
    ("With this fisherman it chose an oil painting in eight of nine tries. And here it is.",
     "Con este pescador eligió un óleo en ocho de nueve pruebas. Y aquí está."),
    ("If you want a photo, say so. I add \"a photograph of\" to the idea.",
     "Si quieres una foto, dilo. Añado \"a photograph of\" a la idea.",
     "If you want a photo, say so. I add, a photograph of, to the idea."),
    ("Now it writes a documentary photograph: the lens, the aperture, the light. Six of six tries were photos.",
     "Ahora escribe una fotografía documental: el objetivo, la apertura, la luz. Seis de seis pruebas fueron fotos."),
    ("Side by side: as written, caption and detailed. Same idea, same image seed.",
     "Lado a lado: as written, caption y detailed. La misma idea, la misma semilla de imagen."),
    ("My impression is that detailed prompts give much better images. But that's my eye, not a measurement, so try them yourself.",
     "Mi impresión es que los prompts detailed dan imágenes mucho mejores. Pero es mi ojo, no una medición, así que pruébalos tú."),
    ("And read what it writes. Detailed may add a setting and props. Once in nine tries it even made our old fisherman a young man.",
     "Y lee lo que escribe. Detailed puede añadir escenario y atrezo. Una vez de nueve hasta convirtió a nuestro pescador viejo en "
     "un hombre joven."),
    # 24: seed and temperature
    ("The enhancer has its own seed. Same seed, same text. Change it, and you get another version of the prompt.",
     "El enhancer tiene su propia semilla. Misma semilla, mismo texto. Si la cambias, sale otra versión del prompt."),
    ("Temperature sets how free the writing is. At zero you always get the same answer.",
     "La temperatura decide cuánta libertad tiene al escribir. A cero, siempre da la misma respuesta."),
    # 26: random prompts
    ("Now a fun one: leave the idea empty in detailed mode, and the enhancer invents a prompt for you.",
     "Ahora una divertida: deja la idea vacía en modo detailed y el enhancer se inventa un prompt por ti."),
    ("The seed draws a medium from twenty-six, a subject from twenty, and a place or time from twenty. Same seed, same prompt.",
     "La semilla elige un medio entre veintiséis, un tema entre veinte y un lugar o época entre veinte. Misma semilla, mismo prompt."),
    ("Seed zero: a Japanese woodblock print of a Paris street. Seed four: a film still of a storm on a Nordic island.",
     "Semilla cero: un grabado japonés de una calle de París. Semilla cuatro: un fotograma de cine de una tormenta en una isla nórdica."),
    ("It's a great way to explore what MOSS can do when you're out of ideas.",
     "Es una buena forma de explorar lo que sabe hacer MOSS cuando te quedas sin ideas."),
    # 30: image input
    ("The enhancer can also see. Qwen 3.5 has a vision part, so you can connect an image.",
     "El enhancer también ve. Qwen 3.5 tiene una parte de visión, así que puedes conectarle una imagen.",
     "The enhancer can also see. Qwen three point five has a vision part, so you can connect an image."),
    ("With no idea, the image is the whole brief: it describes our fisherman, the white beard, the wool sweater, the net and the "
     "gulls, and MOSS draws a new picture from that.",
     "Sin idea, la imagen es todo el encargo: describe a nuestro pescador, la barba blanca, el jersey de lana, la red y las gaviotas, "
     "y MOSS dibuja una imagen nueva a partir de eso."),
    ("With an idea too, the image is the starting point. The same man in winter: the same beard and net, now with snow on the dock.",
     "Con una idea además, la imagen es el punto de partida. El mismo hombre en invierno: la misma barba y la misma red, ahora con "
     "nieve en el muelle."),
    # 33: negative
    ("The negative box. Leave it empty and you get MOSS-Image's official negative prompt.",
     "La caja del negativo. Si la dejas vacía, se usa el negativo oficial de MOSS-Image."),
    ("What you write there is also told to the enhancer. Our caption kept adding a hat, so I write \"hat\" as the negative. "
     "Now the text says \"no hat\".",
     "Lo que escribes ahí también se le dice al enhancer. Nuestro caption no paraba de añadir un sombrero, así que escribo \"hat\" "
     "en el negativo. Ahora el texto dice \"no hat\".",
     "What you write there is also told to the enhancer. Our caption kept adding a hat, so I write, hat, as the negative. "
     "Now the text says, no hat."),
    ("In six tries without the negative, five men wore a hat. With it, none did. That's the written prompt at work: at CFG one, "
     "the sampler doesn't use the negative at all.",
     "En seis pruebas sin el negativo, cinco hombres llevaban sombrero. Con él, ninguno. Eso es obra del prompt escrito: a CFG uno, "
     "el sampler no usa el negativo para nada."),
    # 36: tags
    ("Coming from other models with tag prompts? Paste your tags as the idea.",
     "¿Vienes de otros modelos con prompts de etiquetas? Pega tus tags como idea."),
    ("MOSS reads natural language, so the enhancer turns the tags into prose.",
     "MOSS lee lenguaje natural, así que el enhancer convierte las etiquetas en prosa."),
    ("And count tags like one boy or one girl give no age, so they're written as adults unless your idea says otherwise.",
     "Y los tags de cantidad como 1boy o 1girl no dicen la edad, así que se escriben como adultos salvo que tu idea diga otra cosa."),
    # 39: layout
    ("There's one more mode, layout, which places every element in a box. It deserves its own video, and it's coming.",
     "Queda un modo más, layout, que coloca cada elemento en una caja. Se merece su propio vídeo, y llegará."),
    # 40: goodbye
    ("That's the Prompt Enhancer: as written, caption, detailed, random prompts, images, negatives and tags, all with the text "
     "encoder you already have.",
     "Ese es el Prompt Enhancer: as written, caption, detailed, prompts aleatorios, imágenes, negativos y tags, todo con el text "
     "encoder que ya tienes."),
    ("This workflow is free to download from the description.",
     "Este workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
    # 44: corrections (said after the detailed section; measured in tools/RESULTS.md, "Corrections")
    ("One more trick, my favourite. Once you've seen the image, paste the detailed prompt back as the idea, and add your "
     "corrections at the end.",
     "Un truco más, mi favorito. Cuando ya has visto la imagen, pega el prompt detailed como idea y añade tus correcciones al final."),
    ("Write the label \"Corrections\", then what to change. Pasted back with no corrections, the picture comes out almost the "
     "same: five of six tries.",
     "Escribe la etiqueta \"Corrections\" y luego lo que quieras cambiar. Si lo pegas sin correcciones, la imagen sale casi igual: "
     "cinco de seis pruebas.",
     "Write the label, Corrections, then what to change. Pasted back with no corrections, the picture comes out almost the "
     "same: five of six tries."),
    ("Here I ask for a hat and a cat. Hat and cat in six of six tries, and the rest of the painting stays.",
     "Aquí pido un sombrero y un gato. Sombrero y gato en seis de seis pruebas, y el resto del cuadro se mantiene."),
    ("A tip: correct everything that goes against your change. \"A young man\" alone kept the white beard in the text, and only "
     "about half the men looked younger.",
     "Un consejo: corrige todo lo que vaya en contra de tu cambio. Solo con \"a young man\", el texto conservaba la barba blanca y "
     "solo la mitad de los hombres parecían más jóvenes.",
     "A tip: correct everything that goes against your change. A young man, alone, kept the white beard in the text, and only "
     "about half the men looked younger."),
    ("\"A young man with a short dark beard\" worked six of six.",
     "\"A young man with a short dark beard\" funcionó seis de seis.",
     "A young man with a short dark beard, worked six of six."),
    ("And you can write the corrections in your own language. In Spanish, \"Correcciones\", the enhancer applied them in six of "
     "six texts, and wrote the prompt in English.",
     "Y puedes escribir las correcciones en tu idioma. En español, \"Correcciones\", el enhancer las aplicó en seis de seis "
     "textos, y escribió el prompt en inglés.",
     "And you can write the corrections in your own language. In Spanish, correcciones, the enhancer applied them in six of "
     "six texts, and wrote the prompt in English."),
]

PE_POS = {"loader": (0, 0), "enhancer": (440, 0), "latent": (940, 0), "sampler": (1380, 0), "saver": (1820, 0), "turbo": (940, 450),
          "preview": (440, 720), "image": (0, 420)}
PE_BOX = [-260, -60, 2220, 1200]
PE_IDEA = "an old fisherman on a dock"
PE_TAGS = "1boy, old man, white beard, fishing net, pier, sunrise"
PE_WINTER = "the same man in winter, snow on the dock"
PE_IMAGE = "fisherman_video1.png"  # the first video's image, uploaded by pe_prepare into the recording instance's own input folder
# the measured texts (tools/RESULTS.md, "Every mode on one short idea"): a run whose text lacks its words is reported
PE_EXPECT = {"caption": "faded blue cap", "detailed": "oil painting", "photo": "documentary photograph", "random0": "ukiyo-e",
             "random4": "Nordic", "image": "white beard", "winter": "snow", "tags": "man", "negative": "hat", "hatcat": "cat"}
# the 6 + 6 measured images of the negative check (same clean instance, seeds 0-5)
PE_YOUNG_SHEET = [(f"/view?filename=corr_young_s{s}_00001_.png&subfolder=pe_measure&type=output",
                   f"/view?filename=corr_young_beard_s{s}_00001_.png&subfolder=pe_measure&type=output") for s in range(6)]
PE_NEG_SHEET = [(f"/view?filename=caption_s{s}_00001_.png&subfolder=pe_measure&type=output",
                 f"/view?filename=negative_hat_s{s}_00001_.png&subfolder=pe_measure&type=output") for s in range(6)]


async def pe_prepare(page):
    """Off camera: the first video's fisherman into the recording instance's input folder, then one run loads the models, the LoRA and the
    enhancer's language head."""
    import urllib.request
    import uuid
    url = page.url.split("/", 3)
    base = f"{url[0]}//{url[2]}"
    b = "----" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"fisherman_video1.png\"\r\nContent-Type: image/png\r\n\r\n").encode() \
        + UP_VIDEO1.read_bytes() + f"\r\n--{b}--\r\n".encode()
    urllib.request.urlopen(urllib.request.Request(base + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))
    # the page listed the input folder before the upload: refresh, or the Load Image node is framed red as a missing file
    await page.evaluate("() => window.__hv.app().extensionManager.command.execute('Comfy.RefreshNodeDefinitions')")
    await page.wait_for_timeout(1500)
    g = {k: v for k, v in up_api_graph(7, full=False).items() if k in ("1", "2", "3", "4", "5")}
    g["2"]["inputs"].update({"mode": "caption", "idea": "a red bicycle", "seed": 99})
    g["6"] = {"class_type": "PreviewImage", "inputs": {"images": ["5", 1]}}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


async def pe_type(d, node, name, text, paste=False, append=False):
    """Click the node's text box, select its text and type over it (an empty text just clears it). The box is the visible
    textarea with that placeholder nearest the node (the widget's own element can be a stale, unlaid-out copy)."""
    find = """([id, n, focus]) => { const nd = window.__hv.node(id), c = window.__hv.screen(id, nd.size[0] / 2, nd.size[1] / 2);
        const v = [...document.querySelectorAll('textarea')].filter(e => e.placeholder === n && e.getBoundingClientRect().width > 0);
        if (!v.length) return null;
        const dist = (e) => { const r = e.getBoundingClientRect(); return Math.hypot(r.x + r.width / 2 - c[0], r.y + r.height / 2 - c[1]); };
        const e = v.sort((a, b) => dist(a) - dist(b))[0], r = e.getBoundingClientRect();
        if (focus) { e.focus(); return document.activeElement === e; }
        return [r.left, r.top, r.width, r.height]; }"""
    box = await d.js(find, [node, name, False])
    if not box:
        print(f"{name}: no visible box on screen; text set by code", flush=True)
        await d.js("([id, n, t]) => window.__hv.set(id, {[n]: t})", [node, name, text])
        return
    await d.click(box[0] + min(60, box[2] / 3), box[1] + min(24, box[3] / 2))
    if not await d.js(find.replace("if (focus) { e.focus(); return", "if (focus) { return"), [node, name, True]):
        if not await d.js(find, [node, name, True]):  # focus the box itself, never the canvas
            raise SystemExit(f"the {name} box is not focused; nothing typed")
    before = await d.js("([id, n]) => window.__hv.widget(id, n).value", [node, name])
    if append:
        await d.page.keyboard.press("Control+End")
    else:
        await d.page.keyboard.press("Control+A")
        await d.page.keyboard.press("Backspace")
    if text and paste:
        await d.page.keyboard.insert_text(text)  # at once, as a paste
    elif text:
        await d.page.keyboard.type(text, delay=22)
    value = (before or "") + text if append else text
    await d.js("([id, n, t]) => window.__hv.set(id, {[n]: t})", [node, name, value])  # the widget value, as typed


async def pe_run(d, saver, enhancer, expect=None):
    """Run; returns (image url, seconds, the enhancer's text). A text without its measured words is reported, not hidden."""
    url, t = await basic_run(d, saver)
    await d.move(W * 0.5, H * 0.92)
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    text = (run["outputs"].get(str(enhancer), {}).get("text") or [""])[0]
    if expect and PE_EXPECT[expect].lower() not in text.lower():
        print(f"WARNING {expect}: the text lacks {PE_EXPECT[expect]!r}: {text[:200]!r}", flush=True)
    return url, t, text


async def pe_show(d, url, label, h=760):
    await d.js("(h) => window.__hv.show(h)", f'<div style="text-align:center"><img src="{url}" style="height:{h}px;border-radius:6px">'
               f'<div style="color:#ffcc33;font:600 28px Segoe UI,sans-serif;margin-top:14px">{label}</div></div>')


async def pe_hide(d):
    await d.js("() => window.__hv.hide()")
    await d.hold(0.6)


async def prompt_enhancer(d: Director):
    """The third tutorial: the first video's workflow rebuilt, then every Prompt Enhancer mode on the fisherman."""
    P = PE_POS
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "The Prompt Enhancer: every mode", 6.0)
    for i in range(4):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. quick rebuild
    await d.say(4)
    loader = await basic_add(d, "HaDeSMossLoader", "MOSS Loader", P["loader"])
    await up_set(d, loader, BASIC_LOADER)
    async with d.fast(4.0):
        enhancer = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enhancer"])
        await up_set(d, enhancer, {"mode": "as written", "idea": "", "seed": 0, "temperature": 0.7})
        await basic_link(d, loader, "CLIP", enhancer, "clip", 700)
        latent = await basic_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
        await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
    async with d.fast(6.0):
        sampler = await basic_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["sampler"])
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae")):
            await basic_link(d, a, out, sampler, inp, 700)
        turbo = await basic_add(d, "HaDeSMossTurbo", "MOSS Turbo LoRA", P["turbo"])
        await up_set(d, turbo, {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                "scheduler": "simple"})
        await basic_link(d, turbo, "turbo", sampler, "turbo", 700)
        saver = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["saver"])
        await up_set(d, saver, {"filename_prefix": "MOSS/HaDeS_MOSS_PE"})  # no metadata here: %model% and %seed% would be empty
        await basic_link(d, sampler, "image", saver, "images", 700)
    await d._after_voice(0.1)
    await d.camera({"box": PE_BOX}, 1000, pad=60, top=60)
    await d.say(5)
    preview = await basic_add(d, "PreviewAny", "Preview as Text", P["preview"])
    await d.js("(id) => { const n = window.__hv.node(id); n.setSize([440, 460]); n.setDirtyCanvas(true, true); }", preview)
    await basic_link(d, enhancer, "prompt", preview, "source", 800)
    await d._after_voice(0.1)

    # 3. what the node is
    await d.camera([enhancer], 1100, pad=60, top=60)
    await d.say(6)
    await d.move(*(await d.js("([id]) => window.__hv.slot(id, 'clip', false)", [enhancer])))
    await d._after_voice(0.1)
    await d.say(7)
    await d._after_voice(0.1)
    await d.say(8)
    for w in ("idea", "mode", "seed", "temperature"):
        await basic_point(d, enhancer, w)
        await d.hold(0.7)
    await d._after_voice(0.1)

    # 4. as written
    await d.say(9)
    await pe_type(d, enhancer, "idea", PE_IDEA)
    await d._after_voice(0.1)
    await d.say(10)
    await basic_point(d, enhancer, "mode")
    await d._after_voice(0.1)
    url_aw, t_aw, _ = await pe_run(d, saver, enhancer)
    await d.say(11)
    await pe_show(d, url_aw, f"as written · \"{PE_IDEA}\"")
    await d._after_voice(0.6)
    await pe_hide(d)

    # 5. caption
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(12)
    await basic_point(d, enhancer, "mode", {"mode": "caption", "seed": 1})
    await d.hold(0.4)
    await basic_point(d, enhancer, "seed")
    await d._after_voice(0.1)
    url_cap, t_cap, text_cap = await pe_run(d, saver, enhancer, "caption")
    await d.camera([preview], 1100, pad=40, top=60)
    await d.say(13)
    await d._after_voice(0.2)
    await d.say(14)
    await d.hold(1.8)
    await pe_show(d, url_cap, f"caption · {len(text_cap.split())} words")
    await d._after_voice(0.6)
    await pe_hide(d)

    # 6. detailed: an oil painting, then a photograph
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(15)
    await basic_point(d, enhancer, "mode", {"mode": "detailed", "seed": 2})
    await d._after_voice(0.1)
    url_det, t_det, text_det = await pe_run(d, saver, enhancer, "detailed")
    await d.camera([preview], 1100, pad=40, top=60)
    await d.say(16)
    await d._after_voice(0.1)
    await d.say(17)
    await d._after_voice(0.1)
    await d.say(18)
    await d.hold(1.5)
    await pe_show(d, url_det, f"detailed · {len(text_det.split())} words · no medium in the idea")
    await d._after_voice(0.6)
    await pe_hide(d)
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(19)
    await pe_type(d, enhancer, "idea", "a photograph of " + PE_IDEA)
    await up_set(d, enhancer, {"seed": 1})
    await d._after_voice(0.1)
    url_ph, t_ph, text_ph = await pe_run(d, saver, enhancer, "photo")
    await d.camera([preview], 1100, pad=40, top=60)
    await d.say(20)
    await d.hold(2.5)
    await pe_show(d, url_ph, f"detailed · {len(text_ph.split())} words · \"a photograph of ...\"")
    await d._after_voice(0.4)
    await d.say(21)
    side = 440
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               '<div style="display:flex;gap:24px;margin-top:150px">' + "".join(
                   f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px">'
                   f'<div style="color:#ffcc33;font:600 24px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>'
                   for u, t in ((url_aw, "as written"), (url_cap, "caption"), (url_det, "detailed"),
                                (url_ph, "detailed, \"a photograph of\""))) + "</div>")
    await d._after_voice(0.4)
    await d.say(22)
    await d._after_voice(0.3)
    await pe_hide(d)
    await d.camera([preview], 1100, pad=40, top=60)
    await d.say(23)
    await d._after_voice(0.2)

    # 6b. corrections: the oil-painting prompt pasted back with a note
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(44)
    await pe_type(d, enhancer, "idea", text_det.strip(), paste=True)
    await up_set(d, enhancer, {"mode": "detailed", "seed": 2})
    await d._after_voice(0.1)
    await d.say(45)
    await pe_type(d, enhancer, "idea", "\n\nCorrections: the fisherman wears a hat and there is a cat in the image.", append=True)
    await d._after_voice(0.1)
    url_hc, _, _ = await pe_run(d, saver, enhancer, "hatcat")
    await d.say(46)
    side = 640
    await d.js("(h) => window.__hv.show(h)", '<div style="display:flex;gap:40px;margin-top:120px">' + "".join(
        f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px">'
        f'<div style="color:#ffcc33;font:600 26px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>'
        for u, t in ((url_det, "detailed"), (url_hc, "+ Corrections: a hat and a cat"))) + "</div>")
    await d._after_voice(0.4)
    await d.say(47)
    side = 300
    cells = "".join(f'<img src="{a}" style="width:{side}px;height:{side}px;border-radius:4px">' for a, _ in PE_YOUNG_SHEET) + \
        "".join(f'<img src="{b}" style="width:{side}px;height:{side}px;border-radius:4px">' for _, b in PE_YOUNG_SHEET)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               '<div style="margin-top:110px;text-align:center;color:#fff;font:600 26px Segoe UI,sans-serif">'
               f'<div style="display:grid;grid-template-columns:repeat(6,{side}px);gap:10px">{cells}</div>'
               '<div style="display:flex;justify-content:space-between;margin-top:14px;color:#ffcc33">'
               '<span>top: "a young man"</span><span>bottom: "a young man with a short dark beard"</span></div></div>')
    await d._after_voice(0.3)
    await d.say(48)
    await d._after_voice(0.4)
    await pe_hide(d)
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(49)
    await d._after_voice(0.2)

    # 7. seed and temperature
    await d.camera([enhancer], 1100, pad=60, top=60)
    await d.say(24)
    await basic_point(d, enhancer, "seed")
    await d._after_voice(0.1)
    await d.say(25)
    await basic_point(d, enhancer, "temperature")
    await d._after_voice(0.1)

    # 8. random prompts
    await d.say(26)
    await pe_type(d, enhancer, "idea", "")
    await up_set(d, enhancer, {"mode": "detailed", "seed": 0})
    await basic_point(d, enhancer, "seed")
    await d._after_voice(0.1)
    url_r0, _, text_r0 = await pe_run(d, saver, enhancer, "random0")
    await d.say(27)
    await d.camera([preview], 1100, pad=40, top=60)
    await d._after_voice(0.1)
    await up_set(d, enhancer, {"seed": 4})
    url_r4, _, text_r4 = await pe_run(d, saver, enhancer, "random4")
    await d.say(28)
    side = 640
    await d.js("(h) => window.__hv.show(h)", '<div style="display:flex;gap:40px;margin-top:120px">' + "".join(
        f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px">'
        f'<div style="color:#ffcc33;font:600 26px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>'
        for u, t in ((url_r0, "random · seed 0"), (url_r4, "random · seed 4"))) + "</div>")
    await d._after_voice(0.3)
    await d.say(29)
    await d._after_voice(0.3)
    await pe_hide(d)

    # 10. negative
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(33)
    await pe_type(d, enhancer, "idea", PE_IDEA)
    await up_set(d, enhancer, {"mode": "caption", "seed": 0})
    await d._after_voice(0.1)
    await d.say(34)
    await pe_type(d, enhancer, "negative", "hat")
    url_neg, _, _ = await pe_run(d, saver, enhancer, "negative")
    await d.camera([preview], 1100, pad=40, top=60)
    await d._after_voice(0.3)
    await d.say(35)
    side = 300
    cells = "".join(f'<img src="{a}" style="width:{side}px;height:{side}px;border-radius:4px">' for a, _ in PE_NEG_SHEET) + \
        "".join(f'<img src="{b}" style="width:{side}px;height:{side}px;border-radius:4px">' for _, b in PE_NEG_SHEET)
    await d.js("(h) => window.__hv.show(h)", '<div style="margin-top:110px;text-align:center;color:#fff;font:600 26px Segoe UI,sans-serif">'
               f'<div style="display:grid;grid-template-columns:repeat(6,{side}px);gap:10px">{cells}</div>'
               '<div style="display:flex;justify-content:space-between;margin-top:14px;color:#ffcc33">'
               '<span>top: caption, seeds 0-5 · hat in 5 of 6</span><span>bottom: negative "hat" · hat in 0 of 6</span></div></div>')
    await d._after_voice(0.6)
    await pe_hide(d)
    await pe_type(d, enhancer, "negative", "")

    # 11. tags
    await d.camera([enhancer, preview], 1100, pad=40, top=60)
    await d.say(36)
    await pe_type(d, enhancer, "idea", PE_TAGS)
    await up_set(d, enhancer, {"seed": 3})
    await d._after_voice(0.1)
    url_tag, _, _ = await pe_run(d, saver, enhancer, "tags")
    await d.camera([preview], 1100, pad=40, top=60)
    await d.say(37)
    await d._after_voice(0.2)
    await pe_show(d, url_tag, "tags → prose")
    await d.say(38)
    await d._after_voice(0.4)
    await pe_hide(d)

    # 9. image input: the Load Image node is made by code (its file list would show the user's own images)
    await d.camera({"box": PE_BOX}, 1100, pad=60, top=60)
    await d.say(30)
    image = await d.js("""async ([pos, file]) => { const a = window.__hv.app(), n = window.LiteGraph.createNode('LoadImage');
        const w = n.widgets.find(w => w.name === 'image'); w.value = file; n.pos = [-20000, -20000]; a.graph.add(n); w.callback?.(file);
        for (let i = 0; i < 60; i++) { await new Promise(f => setTimeout(f, 100));
            const src = n.imgs?.[0]?.src || ''; if (src.includes('fisherman_video1')) break; }
        w.value = file; w.callback?.(file); await new Promise(f => setTimeout(f, 600));
        if (!(n.imgs?.[0]?.src || '').includes('fisherman_video1')) return null;
        n.pos = [...pos]; a.graph.setDirtyCanvas(true, true); return n.id; }""", [P["image"], PE_IMAGE])
    if not image:
        raise SystemExit("the Load Image node did not show the fisherman; stopped before it reached the screen")
    await d.camera([image, enhancer], 1100, pad=40, top=60)
    await basic_link(d, image, "IMAGE", enhancer, "image", 900)
    await d._after_voice(0.1)
    await d.say(31)
    await pe_type(d, enhancer, "idea", "")
    await up_set(d, enhancer, {"mode": "caption", "seed": 0})
    await basic_point(d, enhancer, "mode")
    url_im, _, _ = await pe_run(d, saver, enhancer, "image")
    first = "data:image/png;base64," + base64.b64encode(UP_VIDEO1.read_bytes()).decode()
    side = 640
    await d.js("(h) => window.__hv.show(h)", '<div style="display:flex;gap:40px;margin-top:120px">' + "".join(
        f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px">'
        f'<div style="color:#ffcc33;font:600 26px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>'
        for u, t in ((first, "the image in"), (url_im, "drawn from its description"))) + "</div>")
    await d._after_voice(0.4)
    await pe_hide(d)
    await d.camera([image, enhancer], 1100, pad=40, top=60)
    await d.say(32)
    await pe_type(d, enhancer, "idea", PE_WINTER)
    await up_set(d, enhancer, {"seed": 1})
    url_win, _, _ = await pe_run(d, saver, enhancer, "winter")
    await pe_show(d, url_win, f"image + \"{PE_WINTER}\"")
    await d._after_voice(0.5)
    await pe_hide(d)
    await d.js("([img, enh]) => { const n = window.__hv.node(enh), i = n.inputs.findIndex(s => s.name === 'image'); n.disconnect"
               "Input(i); window.__hv.app().graph.setDirtyCanvas(true, true); }", [image, enhancer])

    # 12. layout, mentioned
    await d.camera([enhancer], 1100, pad=60, top=60)
    await d.say(39)
    await basic_point(d, enhancer, "mode", {"mode": "layout"})
    await d._after_voice(0.2)
    await up_set(d, enhancer, {"mode": "detailed", "idea": "a photograph of " + PE_IDEA, "seed": 1})

    # 13. goodbye
    await d.camera([loader, enhancer, latent, sampler, saver, turbo, preview, image], 1300, pad=50, top=60)
    await d.say(40)
    await d._after_voice(0.1)
    await d.say(41)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Prompt Enhancer.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:60px"><img src="{url_ph}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">The Prompt Enhancer: every mode</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(42)
    await d.say(43)
    await d.end(1.5)


prompt_enhancer.prepare = pe_prepare


# ---- the LoRA toolkit (the fourth tutorial) ----

LT_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("Today: LoRAs. Every LoRA node in the Suite, with one practical example from start to finish.",
     "Hoy: LoRAs. Todos los nodos de LoRA de la Suite, con un ejemplo práctico de principio a fin.",
     "Today: Loras. Every Lora node in the Suite, with one practical example from start to finish."),
    ("My model Bulle d'Encre has two LoRAs baked inside. We're going to take them out, use them, see which layers carry the style, "
     "merge them into one file, make it smaller, and read its whole recipe.",
     "Mi modelo Bulle d'Encre lleva dos LoRAs dentro. Vamos a sacarlas, usarlas, ver qué capas llevan el estilo, fusionarlas en un "
     "archivo, hacerlo más pequeño y leer toda su receta.",
     "My model Bulle d'Encre has two Loras baked inside. We're going to take them out, use them, see which layers carry the style, "
     "merge them into one file, make it smaller, and read its whole recipe."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 4: what Bulle d'Encre holds
    ("First, a node that reads files: the MOSS Metadata Reader. It only reads the file's header, so it's instant, even on a ten "
     "gigabyte model.",
     "Primero, un nodo que lee archivos: el MOSS Metadata Reader. Solo lee la cabecera del archivo, así que es instantáneo, incluso "
     "con un modelo de diez gigas."),
    ("Bulle d'Encre's recipe: SFT with thirty percent of Base in layers twenty-four to twenty-nine, plus my Turbo LoRA at zero point "
     "five and my style LoRA at zero point eight.",
     "La receta de Bulle d'Encre: SFT con un treinta por ciento de Base en las capas veinticuatro a veintinueve, más mi LoRA Turbo a "
     "cero coma cinco y mi LoRA de estilo a cero coma ocho.",
     "Bulle d'Encre's recipe: S F T with thirty percent of Base in layers twenty-four to twenty-nine, plus my Turbo Lora at zero point "
     "five and my style Lora at zero point eight."),
    # 6: the base
    ("To take the LoRAs out, we need what's under them: the base. I rebuild it with the Block Merger: SFT and Base, thirty percent "
     "of Base in layers twenty-four to twenty-nine only.",
     "Para sacar las LoRAs necesitamos lo que hay debajo: la base. La rehago con el Block Merger: SFT y Base, un treinta por ciento de "
     "Base solo en las capas veinticuatro a veintinueve.",
     "To take the Loras out, we need what's under them: the base. I rebuild it with the Block Merger: S F T and Base, thirty percent "
     "of Base in layers twenty-four to twenty-nine only."),
    # 7: extractor
    ("Now the LoRA Extractor. It subtracts the base from the finetuned model, and saves the difference as a LoRA.",
     "Ahora el LoRA Extractor. Resta la base del modelo afinado y guarda la diferencia como una LoRA.",
     "Now the Lora Extractor. It subtracts the base from the finetuned model, and saves the difference as a Lora."),
    ("Rank sets the size. I tried thirty-two, sixty-four and a hundred and twenty-eight: sixty-four was the sweet spot.",
     "El rango marca el tamaño. Probé treinta y dos, sesenta y cuatro y ciento veintiocho: sesenta y cuatro fue el punto justo."),
    ("It keeps ninety-seven percent of the difference, in a two hundred and fifty megabyte file. It takes about a minute and a half.",
     "Conserva el noventa y siete por ciento de la diferencia, en un archivo de doscientos cincuenta megas. Tarda alrededor de minuto "
     "y medio."),
    # 10: multi-lora
    ("Quick rebuild of the generation part, with the MOSS Multi-LoRA between the model and the sampler. Each row is one LoRA, with "
     "its own switch and strength.",
     "Montaje rápido de la parte de generación, con el MOSS Multi-LoRA entre el modelo y el sampler. Cada fila es una LoRA, con su "
     "interruptor y su fuerza.",
     "Quick rebuild of the generation part, with the MOSS Multi-Lora between the model and the sampler. Each row is one Lora, with "
     "its own switch and strength."),
    ("The base plus the extracted LoRA. Left, the base alone. Middle, with the LoRA. Right, Bulle d'Encre itself. The LoRA brings "
     "the whole look back.",
     "La base más la LoRA extraída. A la izquierda, la base sola. En el centro, con la LoRA. A la derecha, el propio Bulle d'Encre. "
     "La LoRA devuelve todo el estilo.",
     "The base plus the extracted Lora. Left, the base alone. Middle, with the Lora. Right, Bulle d'Encre itself. The Lora brings "
     "the whole look back."),
    # 12: the turbo row
    ("A second row: my Turbo 8-steps v1.0 at zero point five. The row shows a STEPS tag: the file tells the Suite how it was trained.",
     "Una segunda fila: mi Turbo 8-steps v1.0 a cero coma cinco. La fila enseña una etiqueta STEPS: el archivo le dice a la Suite "
     "cómo se entrenó.",
     "A second row: my Turbo eight steps, version one, at zero point five. The row shows a steps tag: the file tells the Suite how it "
     "was trained."),
    ("Eight steps at CFG one: the same scene and the same look, not the same image. Once everything is loaded, about two seconds "
     "instead of eight.",
     "Ocho pasos a CFG uno: la misma escena y el mismo estilo, no la misma imagen. Con todo ya cargado, unos dos segundos en lugar "
     "de ocho."),
    # 14: groups
    ("Each row also folds out a weight for every group of six layers. Which layers carry this style?",
     "Cada fila despliega además un peso por cada grupo de seis capas. ¿Qué capas llevan este estilo?"),
    ("I switched off one group at a time. Layers twenty-four to thirty-five carry most of it: turning off either group undoes about "
     "three quarters of what the LoRA changes.",
     "Apagué un grupo cada vez. Las capas veinticuatro a treinta y cinco llevan la mayor parte: apagar cualquiera de los dos grupos "
     "deshace unas tres cuartas partes de lo que cambia la LoRA.",
     "I switched off one group at a time. Layers twenty-four to thirty-five carry most of it: turning off either group undoes about "
     "three quarters of what the Lora changes."),
    ("Layers six to eleven, less than a third. That's this LoRA: every LoRA is different, so the Suite has no presets. You measure yours.",
     "Las capas seis a once, menos de un tercio. Eso es esta LoRA: cada LoRA es distinta, así que la Suite no trae presets. La tuya, "
     "la mides tú.",
     "Layers six to eleven, less than a third. That's this Lora: every Lora is different, so the Suite has no presets. You measure yours."),
    ("The same control, for a single LoRA, is the MOSS LoRA Block Weight node.",
     "El mismo control, para una sola LoRA, es el nodo MOSS LoRA Block Weight.",
     "The same control, for a single Lora, is the MOSS Lora Block Weight node."),
    # 18: merger
    ("Two rows work, but one file is simpler to share. The LoRA Merger fuses them: the extracted LoRA at one, the Turbo at zero point five.",
     "Dos filas funcionan, pero un archivo es más fácil de compartir. El LoRA Merger las fusiona: la LoRA extraída a uno y el Turbo a "
     "cero coma cinco.",
     "Two rows work, but one file is simpler to share. The Lora Merger fuses them: the extracted Lora at one, the Turbo at zero point five."),
    ("Rank sixty-four plus two is exact. It took one second. I checked every layer: the file is the same maths within a quarter of a percent.",
     "Rango sesenta y cuatro más dos es exacto. Tardó un segundo. Comprobé cada capa: el archivo es la misma matemática con menos de un "
     "cuarto por ciento de diferencia."),
    ("At eight steps, though, even rounding moves small details. Same scene, same look, a few details shifted.",
     "Eso sí, a ocho pasos hasta el redondeo mueve detalles pequeños. Misma escena, mismo estilo, algunos detalles movidos."),
    # 21: resize
    ("Two hundred and fifty megabytes is big. The LoRA Resize lowers the rank. At thirty-two it's half the size, and the images stay "
     "within the noise.",
     "Doscientos cincuenta megas es mucho. El LoRA Resize baja el rango. A treinta y dos ocupa la mitad, y las imágenes se quedan "
     "dentro del ruido.",
     "Two hundred and fifty megabytes is big. The Lora Resize lowers the rank. At thirty-two it's half the size, and the images stay "
     "within the noise."),
    ("With dynamic energy at zero point nine, it's only forty megabytes, but it drifts. Look at the fisherman.",
     "Con dynamic energy a cero coma nueve, solo ocupa cuarenta megas, pero se aleja. Mira al pescador."),
    ("So: rank thirty-two, half the size, no visible cost.",
     "Así que: rango treinta y dos, la mitad de tamaño, sin coste visible."),
    # 24: recipe
    ("And now the Metadata Reader on the small file. It follows the whole chain: resized, from a merge, of a LoRA extracted from "
     "Bulle d'Encre and the Turbo at zero point five.",
     "Y ahora el Metadata Reader sobre el archivo pequeño. Sigue toda la cadena: reducido, de una fusión, de una LoRA extraída de "
     "Bulle d'Encre y del Turbo a cero coma cinco.",
     "And now the Metadata Reader on the small file. It follows the whole chain: resized, from a merge, of a Lora extracted from "
     "Bulle d'Encre and the Turbo at zero point five."),
    ("Every LoRA made with the Suite remembers where it came from.",
     "Toda LoRA hecha con la Suite recuerda de dónde viene.",
     "Every Lora made with the Suite remembers where it came from."),
    # 26: goodbye
    ("Extract, stack, choose the layers, merge, shrink and read: that's the LoRA Toolkit.",
     "Extraer, apilar, elegir las capas, fusionar, reducir y leer: eso es el LoRA Toolkit.",
     "Extract, stack, choose the layers, merge, shrink and read: that's the Lora Toolkit."),
    ("This workflow is free to download from the description.",
     "Este workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

LT_BULLE = "HaDeS_MOSS_preview_beta.safetensors"
LT_SFT, LT_BASE = "mossImage_sft_bf16.safetensors", "mossImage_base_bf16.safetensors"
LT_EXTRACTED, LT_MERGED, LT_SMALL = "bulle_extracted", "bulle_turbo_merged", "bulle_turbo_r32"
LT_POS = {"meta": (0, 0), "meta_text": (420, 0), "sft": (0, 520), "base": (0, 660), "bm": (320, 520), "bulle": (760, 820),
          "ext": (760, 520), "ext_text": (1180, 520), "loader": (0, 1160), "enhancer": (440, 1160), "latent": (940, 1160),
          "ml": (1380, 1160), "sampler": (1820, 1160), "saver": (2260, 1160), "merger": (0, 1900), "merge_text": (420, 1900),
          "resize": (920, 1900), "resize_text": (1340, 1900)}
LT_BOX = [-260, -60, 2700, 2320]
LT_BM = {"layers_00_05": 0.0, "layers_06_11": 0.0, "layers_12_17": 0.0, "layers_18_23": 0.0, "layers_24_29": 0.3, "layers_30_35": 0.0,
         "layers_36_41": 0.0, "text_embedder": 0.0, "time_modulation": 0.0, "io": 0.0}
# measured images (tools/RESULTS.md, "The LoRA toolkit on Bulle d'Encre"): the same settings on the same clean instance
LT_IMG = lambda name: f"/view?filename={name}&subfolder=lt_measure&type=output"  # noqa: E731
LT_FILE = {}  # filled by lt_prepare: measured image prefix -> file name in comfy_output/lt_measure


async def lt_prepare(page):
    """Off camera: the measured images' file names, and one render that loads the text encoder, the VAE and the models."""
    import glob, os
    for prefix in ("base_only_fisherman_1013", "bulle_fisherman_1013", "bulle_turbo_merged_fisherman_1013",
                   "bulle_turbo_r32_fisherman_1013", "bulle_turbo_dyn90_fisherman_1013", "turbo05_8_fisherman_1013"):
        f = sorted(glob.glob(str(Path("I:/HaDeS_MOSS/videos/comfy_output/lt_measure") / (prefix + "_*.png"))), key=os.path.getmtime)
        if not f:
            raise SystemExit(f"measured image {prefix} missing")
        LT_FILE[prefix] = Path(f[-1]).name
    g = {k: v for k, v in up_api_graph(7, full=False).items() if k in ("1", "2", "3", "5")}
    g["5"]["inputs"].pop("turbo")
    g["5"]["inputs"].update({"steps": 2, "cfg": 2.5})
    g["6"] = {"class_type": "PreviewImage", "inputs": {"images": ["5", 1]}}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


async def lt_refresh(d):
    """A file was written: refresh the lists, or a widget naming it is framed red as missing."""
    await d.js("() => window.__hv.app().extensionManager.command.execute('Comfy.RefreshNodeDefinitions')")
    await d.hold(1.2)


async def lt_add(d, type_, query, key, values=None):
    """basic_add, with the node's values set the moment it is created (no default file name is ever drawn)."""
    if values:
        await d.js("""([type, v]) => { const g = window.__hv.app().graph, prev = g.onNodeAdded;
            g.onNodeAdded = function (n) { if (n.type === type) { for (const [k, x] of Object.entries(v)) { const w = n.widgets?.find(w => w.name === k);
                if (w) w.value = x; } g.onNodeAdded = prev; } return prev?.apply(this, arguments); }; }""", [type_, values])
    n = await basic_add(d, type_, query, LT_POS[key])
    if values:
        await up_set(d, n, values)
    return n


async def lt_text(d, key, size=(440, 320)):
    n = await lt_add(d, "PreviewAny", "Preview as Text", key)
    await d.js("([id, s]) => { const n = window.__hv.node(id); n.setSize(s); n.setDirtyCanvas(true, true); }", [n, list(size)])
    return n


async def lt_run(d, saver=None):
    """Run (fast-forwarded); with a saver, returns its image url."""
    if saver:
        url, t = await basic_run(d, saver)
        await d.move(W - 6, H * 0.5)  # the right edge: a pointer resting on a node or a canvas button shows its tooltip
        return url, t
    run = d.page.locator("button:has-text('Run')").first
    b = await run.bounding_box()
    await d.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    async with d.fast(2.0):
        await d.hold(1.0)
        for _ in range(2400):
            if await d.js("() => fetch('/queue').then(r => r.json()).then(q => q.queue_running.length + q.queue_pending.length)") == 0:
                break
            await asyncio.sleep(0.25)
        else:
            raise SystemExit("the run did not finish in 10 minutes")
        await d.hold(0.5)
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    if run["status"].get("status_str") != "success":
        raise SystemExit(f"the run failed: {run['status']}")
    await d.move(W - 6, H * 0.5)  # the right edge: a pointer resting on a node or a canvas button shows its tooltip
    return None, None


def lt_cols(items, side=600, top=150):
    return (f'<div style="display:flex;gap:30px;margin-top:{top}px">' + "".join(
        f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px">'
        f'<div style="color:#ffcc33;font:600 24px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>' for u, t in items) + "</div>")


async def lora_toolkit(d: Director):
    """The fourth tutorial: Bulle d'Encre's LoRAs extracted, stacked, split by layers, merged, resized and read."""
    P = LT_POS
    await d.camera({"box": [-260, -60, 1700, 900]}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "The LoRA Toolkit", 6.0)
    for i in range(4):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. what Bulle d'Encre holds
    await d.say(4)
    meta = await lt_add(d, "HaDeSMossMetadataReader", "MOSS Metadata Reader", "meta", {"lora_name": "(none)", "model_name": LT_BULLE})
    meta_text = await lt_text(d, "meta_text", (620, 420))
    await basic_link(d, meta, "info", meta_text, "source", 800)
    await basic_point(d, meta, "model_name")
    await d._after_voice(0.1)
    await lt_run(d)
    await d.camera([meta_text], 1100, pad=40, top=60)
    await d.say(5)
    await d._after_voice(0.6)

    # 3. the base, rebuilt with the Block Merger
    await d.camera({"box": [-260, 400, 1700, 1100]}, 1100, pad=40, top=60)
    await d.say(6)
    async with d.fast(4.0):
        sft = await lt_add(d, "UNETLoader", "Load Diffusion Model", "sft", {"unet_name": LT_SFT, "weight_dtype": "default"})
        base = await lt_add(d, "UNETLoader", "Load Diffusion Model", "base", {"unet_name": LT_BASE, "weight_dtype": "default"})
    bm = await lt_add(d, "HaDeSMossModelBlockMerger", "MOSS Model Block Merger", "bm", LT_BM)
    async with d.fast(2.0):
        await basic_link(d, sft, "MODEL", bm, "model1", 700)
        await basic_link(d, base, "MODEL", bm, "model2", 700)
    await d.camera([bm], 1000, pad=120, top=60)
    await basic_point(d, bm, "layers_24_29")
    await d._after_voice(0.1)

    # 4. the Extractor
    await d.camera({"box": [-260, 400, 1700, 1100]}, 1100, pad=40, top=60)
    await d.say(7)
    ext = await lt_add(d, "HaDeSMossLoraExtractor", "MOSS LoRA Extractor", "ext",
                       {"rank": 64, "export_filename": LT_EXTRACTED, "overwrite_existing": True, "tuned_name": "Bulle d'Encre",
                        "base_name": "SFT + 30% Base in layers 24-29 (Block Merger)"})
    bulle = await lt_add(d, "UNETLoader", "Load Diffusion Model", "bulle", {"unet_name": LT_BULLE, "weight_dtype": "default"})
    async with d.fast(3.0):
        await basic_link(d, bulle, "MODEL", ext, "model_tuned", 700)
        await basic_link(d, bm, "MODEL", ext, "model_base", 700)
        ext_text = await lt_text(d, "ext_text", (440, 300))
        await basic_link(d, ext, "info", ext_text, "source", 700)
    await d._after_voice(0.1)
    await d.camera([ext], 1000, pad=120, top=60)
    await d.say(8)
    await basic_point(d, ext, "rank")
    await d._after_voice(0.1)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([ext, ext_text], 1100, pad=40, top=60)
    await d.say(9)
    await d._after_voice(0.6)

    # 5. the Multi-LoRA: the LoRA back on the base
    await d.camera({"box": [-260, 1100, 2700, 1900]}, 1200, pad=40, top=60)
    await d.say(10)
    async with d.fast(6.0):
        loader = await lt_add(d, "HaDeSMossLoader", "MOSS Loader", "loader", dict(BASIC_LOADER))
        enhancer = await lt_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", "enhancer",
                                {"mode": "as written", "idea": BASIC_PROMPT, "seed": 0})
        await basic_link(d, loader, "CLIP", enhancer, "clip", 600)
        latent = await lt_add(d, "HaDeSMossLatent", "MOSS Latent", "latent",
                              {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
        sampler = await lt_add(d, "HaDeSMossKSampler", "MOSS KSampler", "sampler",
                               {"steps": 15, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple"})
        for a, out, inp in ((enhancer, "positive", "positive"), (enhancer, "negative", "negative"), (latent, "latent", "latent_image"),
                            (latent, "seed", "seed"), (loader, "VAE", "vae")):
            await basic_link(d, a, out, sampler, inp, 600)
        saver = await lt_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", "saver", {"filename_prefix": "MOSS/HaDeS_MOSS_LT"})
        await basic_link(d, sampler, "image", saver, "images", 600)
    rows1 = [{"name": LT_EXTRACTED + ".safetensors", "on": True, "strength": 1.0}]
    ml = await lt_add(d, "HaDeSMossMultiLora", "MOSS Multi-LoRA", "ml", {"loras": json.dumps(rows1)})
    async with d.fast(2.0):
        await d.camera({"box": LT_BOX}, 800, pad=40, top=60)
        await basic_link(d, bm, "MODEL", ml, "model", 900)
        await basic_link(d, ml, "MODEL", sampler, "model", 700)
    await d.camera([ml], 1000, pad=140, top=60)
    await d._after_voice(0.1)
    url_lora, _ = await lt_run(d, saver)
    await d.say(11)
    await d.js("(h) => window.__hv.show(h)", lt_cols([
        (LT_IMG(LT_FILE["base_only_fisherman_1013"]), "the base alone"),
        (url_lora, "base + extracted LoRA"),
        (LT_IMG(LT_FILE["bulle_fisherman_1013"]), "Bulle d'Encre")]))
    await d._after_voice(0.5)
    await pe_hide(d)

    # 6. the Turbo as a second row
    await d.camera([ml, sampler], 1000, pad=80, top=60)
    await d.say(12)
    rows2 = rows1 + [{"name": BASIC_LORA, "on": True, "strength": 0.5}]
    await up_set(d, ml, {"loras": json.dumps(rows2)})
    await d.hold(0.6)
    await basic_point(d, ml, "hades_multilora_list")
    await d._after_voice(0.1)
    await basic_point(d, sampler, "steps", {"steps": 8, "cfg": 1.0})
    url_turbo, t_turbo = await lt_run(d, saver)
    await d.say(13)
    await d.js("(h) => window.__hv.show(h)", lt_cols([
        (LT_IMG(LT_FILE["bulle_fisherman_1013"]), "Bulle d'Encre · 15 steps · CFG 2.5"),
        (url_turbo, "+ Turbo 0.5 row · 8 steps · CFG 1")], side=700))
    await d._after_voice(0.5)
    await pe_hide(d)

    # 7. groups of layers
    await d.camera([ml], 1000, pad=140, top=60)
    await d.say(14)
    rows3 = [{"name": LT_EXTRACTED + ".safetensors", "on": True, "strength": 1.0, "groups": {"layers_30_35": 0.0}},
             {"name": BASIC_LORA, "on": False, "strength": 0.5}]
    await up_set(d, ml, {"loras": json.dumps(rows3)})
    await basic_point(d, ml, "hades_multilora_list")
    await up_set(d, sampler, {"steps": 15, "cfg": 2.5})
    await d._after_voice(0.1)
    url_off, _ = await lt_run(d, saver)
    await d.say(15)
    await d.js("(h) => window.__hv.show(h)", lt_cols([
        (LT_IMG(LT_FILE["base_only_fisherman_1013"]), "the base alone"),
        (url_off, "LoRA with layers 30-35 off"),
        (url_lora, "the whole LoRA")]))
    await d._after_voice(0.3)
    await d.say(16)
    await d._after_voice(0.3)
    await pe_hide(d)
    await d.say(17)
    await d._after_voice(0.2)

    # 8. the Merger
    await d.camera({"box": [-260, 1800, 1800, 2320]}, 1100, pad=40, top=60)
    await d.say(18)
    merger = await lt_add(d, "HaDeSMossLoraMerger", "MOSS LoRA Merger", "merger",
                          {"lora_a": LT_EXTRACTED + ".safetensors", "weight_a": 1.0, "lora_b": BASIC_LORA, "weight_b": 0.5, "rank": 72,
                           "export_filename": LT_MERGED, "dynamic_energy": 0.0, "overwrite_existing": True})
    async with d.fast(2.0):
        await d.camera({"box": LT_BOX}, 800, pad=40, top=60)
        await basic_link(d, bm, "MODEL", merger, "model", 900)
        await d.camera({"box": [-260, 1800, 1800, 2320]}, 800, pad=40, top=60)
        merge_text = await lt_text(d, "merge_text", (440, 260))
        await basic_link(d, merger, "info", merge_text, "source", 600)
    await d._after_voice(0.1)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([merger, merge_text], 1000, pad=60, top=60)
    await d.say(19)
    await d._after_voice(0.2)
    await d.say(20)
    await d.js("(h) => window.__hv.show(h)", lt_cols([
        (LT_IMG(LT_FILE["turbo05_8_fisherman_1013"]), "two rows live · 8 steps"),
        (LT_IMG(LT_FILE["bulle_turbo_merged_fisherman_1013"]), "the merged file · 8 steps")], side=700))
    await d._after_voice(0.5)
    await pe_hide(d)

    # 9. the Resize
    await d.camera({"box": [-260, 1800, 1800, 2320]}, 1000, pad=40, top=60)
    await d.say(21)
    resize = await lt_add(d, "HaDeSMossLoraResize", "MOSS LoRA Resize", "resize",
                          {"lora_name": LT_MERGED + ".safetensors", "target_rank": 32, "export_filename": LT_SMALL,
                           "dynamic_energy": 0.0, "overwrite_existing": True})
    async with d.fast(2.0):
        await d.camera({"box": LT_BOX}, 800, pad=40, top=60)
        await basic_link(d, bm, "MODEL", resize, "model", 900)
        await d.camera({"box": [-260, 1800, 1800, 2320]}, 800, pad=40, top=60)
        resize_text = await lt_text(d, "resize_text", (440, 260))
        await basic_link(d, resize, "info", resize_text, "source", 600)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([resize, resize_text], 1000, pad=60, top=60)
    await d._after_voice(0.2)
    await d.say(22)
    await d.js("(h) => window.__hv.show(h)", lt_cols([
        (LT_IMG(LT_FILE["bulle_turbo_merged_fisherman_1013"]), "merged · rank 72 · 255 MB"),
        (LT_IMG(LT_FILE["bulle_turbo_r32_fisherman_1013"]), "resized · rank 32 · 126 MB"),
        (LT_IMG(LT_FILE["bulle_turbo_dyn90_fisherman_1013"]), "dynamic energy 0.9 · 40 MB")]))
    await d._after_voice(0.4)
    await d.say(23)
    await d._after_voice(0.3)
    await pe_hide(d)

    # 10. the whole recipe
    await d.camera([meta, meta_text], 1200, pad=40, top=60)
    await d.say(24)
    await basic_point(d, meta, "lora_name", {"lora_name": LT_SMALL + ".safetensors", "model_name": "(none)"})
    await lt_run(d)
    await d.camera([meta_text], 1000, pad=30, top=60)
    await d._after_voice(0.4)
    await d.say(25)
    await d._after_voice(0.4)

    # 11. goodbye
    await d.camera({"box": LT_BOX}, 1400, pad=50, top=60)
    await d.move(W - 6, H * 0.5)
    await d.say(26)
    await d._after_voice(0.1)
    await d.say(27)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - LoRA Toolkit.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:60px"><img src="{url_lora}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">The LoRA Toolkit</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(28)
    await d.say(29)
    await d.end(1.5)


lora_toolkit.prepare = lt_prepare


# ---- metadata (the fifth tutorial) ----

MD_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("In the second video I promised a video about metadata. Here it is: everything your PNG remembers, and how to control it.",
     "En el segundo vídeo prometí un vídeo sobre metadatos. Aquí está: todo lo que recuerda tu PNG y cómo controlarlo.",
     "In the second video I promised a video about metadata. Here it is: everything your P N G remembers, and how to control it."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 3
    ("Quick rebuild of the first video's workflow, ending in the MOSS Image Saver.",
     "Montaje rápido del workflow del primer vídeo, terminando en el MOSS Image Saver."),
    # 4: no metadata
    ("First, the Saver on its own. It saves the image, but it tells you something's missing: it doesn't know your settings.",
     "Primero, el Saver solo. Guarda la imagen, pero te avisa de que falta algo: no conoce tus ajustes."),
    ("The PNG still carries the workflow, but no steps, no seed, no model.",
     "El PNG sigue llevando el workflow, pero ni pasos, ni semilla, ni modelo.",
     "The P N G still carries the workflow, but no steps, no seed, no model."),
    # 6: telemetry
    ("Every Suite node has an info output. Suite Telemetry gathers them, and its json output goes into the Saver's metadata input.",
     "Cada nodo de la Suite tiene una salida info. Suite Telemetry las reúne, y su salida json va a la entrada metadata del Saver.",
     "Every Suite node has an info output. Suite Telemetry gathers them, and its jason output goes into the Saver's metadata input."),
    ("Run it again.",
     "Lo lanzamos otra vez."),
    # 8: inside the PNG
    ("Here's what's written inside, in the format Civitai and most image browsers read. First, your prompt and the negative prompt.",
     "Esto es lo que va escrito dentro, en el formato que leen Civitai y la mayoría de visores de imágenes. Primero, tu prompt y el negativo."),
    ("Then the settings: steps, sampler, schedule, CFG, seed and size. Everything you need to make the same image again.",
     "Después los ajustes: pasos, sampler, programación, CFG, semilla y tamaño. Todo lo que hace falta para repetir la imagen."),
    ("The model, with its hash. The hash is a fingerprint of the file: ten characters of its SHA256.",
     "El modelo, con su hash. El hash es una huella del archivo: diez caracteres de su SHA256.",
     "The model, with its hash. The hash is a fingerprint of the file: ten characters of its S H A two fifty-six."),
    ("The shift, the VAE, the text encoder, and every LoRA with its own hash. Here, my Turbo.",
     "El shift, el VAE, el text encoder y cada LoRA con su propio hash. Aquí, mi Turbo.",
     "The shift, the V A E, the text encoder, and every Lora with its own hash. Here, my Turbo."),
    ("The first save reads the whole model once to compute its hash: about six seconds here, for ten gigabytes. The next save takes no time.",
     "El primer guardado lee el modelo entero una vez para calcular su hash: unos seis segundos aquí, para diez gigas. El siguiente "
     "guardado ya no tarda nada."),
    # 13: the workflow inside
    ("The PNG also keeps the whole workflow. Drop it into ComfyUI and everything comes back: nodes, wires and values.",
     "El PNG también guarda el workflow entero. Suéltalo en ComfyUI y vuelve todo: nodos, cables y valores.",
     "The P N G also keeps the whole workflow. Drop it into ComfyUI and everything comes back: nodes, wires and values."),
    ("And the Suite keeps its own copy of every node's settings, in a block called hades moss.",
     "Y la Suite guarda su propia copia de los ajustes de cada nodo, en un bloque llamado hades moss.",
     "And the Suite keeps its own copy of every node's settings, in a block called HaDeS moss."),
    # 15: file names
    ("The file name can use your settings too: percent model, percent seed, steps, CFG and more.",
     "El nombre del archivo también puede usar tus ajustes: porcentaje model, porcentaje seed, pasos, CFG y más."),
    # 16: two stages
    ("Now the case I promised. The second video's workflow has two samplers: the image, and the MOSS pass on the upscale.",
     "Ahora el caso que prometí. El workflow del segundo vídeo tiene dos samplers: la imagen y el pase MOSS sobre el escalado."),
    ("With both in Telemetry, the Saver writes the first sampler's settings, and tells you so.",
     "Con los dos en Telemetry, el Saver escribe los ajustes del primer sampler, y te lo avisa."),
    ("To choose, add a second Telemetry connected only to the stage this image comes from, and plug its json into primary stage metadata.",
     "Para elegir, añade un segundo Telemetry conectado solo a la etapa de la que sale esta imagen, y conecta su json a primary stage metadata.",
     "To choose, add a second Telemetry connected only to the stage this image comes from, and plug its jason into primary stage metadata."),
    ("Now the refined PNG says what really made it: twenty forty-eight, denoising strength zero point two five. And the full pipeline is "
     "still stored inside.",
     "Ahora el PNG refinado dice lo que lo hizo de verdad: 2048, denoising strength cero coma veinticinco. Y la cadena completa sigue "
     "guardada dentro.",
     "Now the refined P N G says what really made it: twenty forty-eight, denoising strength zero point two five. And the full pipeline is "
     "still stored inside."),
    # 20: merges
    ("One more case, fixed in version zero nine three. The Saver names the model your sampler really used, even when you merge models "
     "inside the workflow.",
     "Un caso más, arreglado en la versión 0.9.3. El Saver nombra el modelo que de verdad usó tu sampler, aunque mezcles modelos dentro "
     "del workflow.",
     "One more case, fixed in version zero point nine point three. The Saver names the model your sampler really used, even when you merge "
     "models inside the workflow."),
    ("A merge has no single file, so it's written as \"merge\", with the names of both models, and without a model hash.",
     "Una mezcla no tiene un archivo único, así que se escribe como \"merge\", con los nombres de los dos modelos, y sin hash de modelo.",
     "A merge has no single file, so it's written as merge, with the names of both models, and without a model hash."),
    # 22: privacy
    ("And when you'd rather not share: hashes off removes the fingerprints, and embed workflow off removes the workflow. The settings stay.",
     "Y cuando prefieras no compartir: hashes apagado quita las huellas, y embed workflow apagado quita el workflow. Los ajustes se quedan."),
    ("ComfyUI's disable metadata option removes everything from the PNG.",
     "La opción disable metadata de ComfyUI quita todo del PNG.",
     "ComfyUI's disable metadata option removes everything from the P N G."),
    # 24: goodbye
    ("That's metadata: your recipe, written inside every image.",
     "Eso son los metadatos: tu receta, escrita dentro de cada imagen."),
    ("This workflow is free to download from the description.",
     "Este workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

MD_POS = dict(UP_POS, tel2=(2500, 1200))
MD_BOX = [-260, -60, 3380, 1560]
# a PNG of the LoRA Toolkit example with the Multi-LoRA on (0.9.3: the Model field of a merge made in the workflow)
MD_MERGE_PNG = "/view?filename=HaDeS_MOSS_lora_1013_00003_.png&subfolder=MOSS&type=output"

MD_PNG_TEXT_JS = r"""
async (url) => {
  const b = new Uint8Array(await (await fetch(url)).arrayBuffer()), dv = new DataView(b.buffer), out = {};
  const dec = (a, enc) => new TextDecoder(enc).decode(a);
  for (let i = 8; i + 8 <= b.length;) {
    const len = dv.getUint32(i), type = dec(b.slice(i + 4, i + 8), 'latin1'), data = b.slice(i + 8, i + 8 + len);
    if (type === 'tEXt') { const z = data.indexOf(0); out[dec(data.slice(0, z), 'latin1')] = dec(data.slice(z + 1), 'latin1'); }
    if (type === 'iTXt') { const z = data.indexOf(0); let j = z + 3; j = data.indexOf(0, j) + 1; j = data.indexOf(0, j) + 1;
                           out[dec(data.slice(0, z), 'latin1')] = dec(data.slice(j), 'utf-8'); }
    if (type === 'IEND') break;
    i += 12 + len;
  }
  return out;
}
"""

MD_GROUPS = {"settings": ("Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size"),
             "model": ("Model hash", "Model"),
             "rest": ("Denoising strength", "Shift", "VAE", "Text encoder", "Lora hashes", "Hashes", "Version")}


def md_esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_params_html(chunks, title, highlight=None, mark=()):
    """The PNG's text chunks as a card: the chunk names, then `parameters` with one group lit (prompt / settings / model / rest)
    and any field named in `mark` lit too."""
    params = chunks.get("parameters", "")
    lines = params.split("\n")
    settings = lines[-1] if lines and re.match(r"^[A-Z][A-Za-z ]+: ", lines[-1]) else ""
    text_lines = lines[:-1] if settings else lines
    on = "background:#ffcc33;color:#111;border-radius:4px;padding:0 4px"
    body = "".join(f'<div style="{on if highlight == "prompt" else ""}">{md_esc(l)}</div>' for l in text_lines if l.strip())
    fields = re.split(r", (?=[A-Z][A-Za-z ]+: )", settings) if settings else []
    spans = []
    for f in fields:
        key = f.split(":")[0]
        lit = (highlight and key in MD_GROUPS.get(highlight, ())) or key in mark
        spans.append(f'<span style="{on if lit else ""}">{md_esc(f)}</span>')
    body += f'<div style="margin-top:12px">{", ".join(spans)}</div>' if spans else ""
    pills = "".join(f'<span style="background:#24242b;border:1px solid #ffcc33;color:#ffcc33;border-radius:8px;padding:4px 12px;'
                    f'margin-right:10px;font:600 22px Segoe UI,sans-serif">{md_esc(k)}</span>' for k in sorted(chunks))
    return (f'<div style="margin-top:120px;width:1640px;color:#e6e6ea">'
            f'<div style="font:700 34px Segoe UI,sans-serif;color:#fff;margin-bottom:16px">{md_esc(title)}</div>'
            f'<div style="margin-bottom:18px">{pills or "<i>no text in the PNG</i>"}</div>'
            f'<div style="background:#141418;border:2px solid #333;border-radius:12px;padding:26px 30px;font:500 25px Consolas,monospace;'
            f'line-height:1.45;white-space:normal;word-break:break-word">{body or "<i>(empty)</i>"}</div></div>')


async def md_chunks(d, url):
    return await d.js(MD_PNG_TEXT_JS, url)


async def md_add(d, type_, query, key, values=None):
    """basic_add at MD_POS[key], with the node's values set the moment it is created."""
    if values:
        await d.js("""([type, v]) => { const g = window.__hv.app().graph, prev = g.onNodeAdded;
            g.onNodeAdded = function (n) { if (n.type === type) { for (const [k, x] of Object.entries(v)) { const w = n.widgets?.find(w => w.name === k);
                if (w) w.value = x; } g.onNodeAdded = prev; } return prev?.apply(this, arguments); }; }""", [type_, values])
    n = await basic_add(d, type_, query, MD_POS[key])
    if values:
        await up_set(d, n, values)
    return n


async def metadata_video(d: Director):
    """The fifth tutorial: what the MOSS Image Saver writes into a PNG, Suite Telemetry, two stages, merges and privacy."""
    P = MD_POS
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "Metadata: what your PNG remembers", 6.0)
    for i in range(3):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. quick rebuild, the Saver with nothing on metadata
    await d.say(3)
    async with d.fast(6.0):
        loader = await md_add(d, "HaDeSMossLoader", "MOSS Loader", "loader", dict(BASIC_LOADER))
        enhancer = await md_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", "enhancer",
                                {"mode": "as written", "idea": BASIC_PROMPT, "seed": 0})
        await basic_link(d, loader, "CLIP", enhancer, "clip", 600)
        latent = await md_add(d, "HaDeSMossLatent", "MOSS Latent", "latent",
                              {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
        sampler = await md_add(d, "HaDeSMossKSampler", "MOSS KSampler", "sampler", {"sampler_name": "euler", "scheduler": "simple"})
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae")):
            await basic_link(d, a, out, sampler, inp, 600)
        turbo = await md_add(d, "HaDeSMossTurbo", "MOSS Turbo LoRA", "turbo",
                             {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
        await basic_link(d, turbo, "turbo", sampler, "turbo", 600)
    saver = await md_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", "saver")
    await basic_link(d, sampler, "image", saver, "images", 700)
    await d._after_voice(0.1)

    # 3. without metadata
    await d.camera([sampler, saver], 1000, pad=80, top=60)
    await d.say(4)
    url0, _ = await lt_run(d, saver)
    note0 = (next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))["outputs"]
             .get(str(saver), {}).get("text") or [""])[0]
    chunks = await md_chunks(d, url0)
    html = md_params_html(chunks, "Image Saver, metadata not connected")
    if note0:
        html = html.replace('<div style="margin-bottom:18px">', f'<div style="color:#ffcc33;font:600 26px Segoe UI,sans-serif;margin-bottom:14px">{md_esc(note0)}</div><div style="margin-bottom:18px">', 1)
    await d.js("(h) => window.__hv.show(h)", html)
    await d._after_voice(0.2)
    await d.say(5)
    await d._after_voice(0.4)
    await pe_hide(d)

    # 4. Suite Telemetry
    await d.camera({"box": UP_BOX}, 1100, pad=60, top=60)
    await d.say(6)
    telemetry = await md_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", "telemetry")
    async with d.fast(5.0):
        await d.camera({"box": [-60, -60, 2300, 1100]}, 800, pad=40, top=60)
        for k, (n, out) in enumerate(((loader, "info"), (enhancer, "info"), (latent, "info"), (sampler, "info"), (turbo, "info"))):
            await basic_link(d, n, out, telemetry, f"info_{k + 1}", 700)
    await basic_link(d, telemetry, "json", saver, "metadata", 900)
    await d._after_voice(0.1)
    await d.say(7)
    url1, t1 = await lt_run(d, saver)
    chunks = await md_chunks(d, url1)
    print("PNG chunks after Telemetry:", sorted(chunks), flush=True)

    # 5. what is inside
    title = "Image Saver + Suite Telemetry: the PNG's parameters"
    await d.js("(h) => window.__hv.show(h)", md_params_html(chunks, title))
    await d.say(8)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }", md_params_html(chunks, title, "prompt"))
    await d._after_voice(0.1)
    for i, group in ((9, "settings"), (10, "model"), (11, "rest")):
        await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }", md_params_html(chunks, title, group))
        await d.say(i)
        await d._after_voice(0.1)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               md_params_html(chunks, title, mark=("Model hash", "Lora hashes")))
    await d.say(12)
    await d._after_voice(0.3)
    await pe_hide(d)

    # 6. the workflow inside: the canvas cleared, the PNG dropped in
    await d.say(13)
    await d.js("() => { const a = window.__hv.app(); a.graph.clear(); a.graph.setDirtyCanvas(true, true); }")
    await d.hold(1.2)
    loaded = await d.js("""async (url) => { const blob = await (await fetch(url)).blob();
        const file = new File([blob], url.split('filename=')[1].split('&')[0], {type: 'image/png'});
        await window.__hv.app().handleFile(file); await new Promise(f => setTimeout(f, 1500));
        return window.__hv.app().graph._nodes.length; }""", url1)
    print("nodes after loading the PNG:", loaded, flush=True)
    if not loaded:
        raise SystemExit("the PNG did not load its workflow back")
    await d.camera({"box": UP_BOX}, 1100, pad=60, top=60)
    await d._after_voice(0.3)
    await d.say(14)
    hm = json.loads(chunks.get("hades_moss", "[]"))
    await d.js("(h) => window.__hv.show(h)", '<div style="margin-top:160px;width:1500px;color:#e6e6ea">'
               '<div style="font:700 34px Segoe UI,sans-serif;color:#fff;margin-bottom:16px">hades_moss: every node\'s settings</div>'
               '<div style="background:#141418;border:2px solid #333;border-radius:12px;padding:26px 30px;font:500 24px Consolas,monospace;line-height:1.5">'
               + "".join(f'<div><span style="color:#ffcc33">{md_esc(m.get("node"))}</span>: '
                         + md_esc(", ".join(f"{k}={v}" for k, v in m.items() if k not in ("node", "prompt", "negative", "model_files", "sources"))[:150])
                         + "…</div>" for m in hm) + "</div></div>")
    await d._after_voice(0.4)
    await pe_hide(d)

    # 7. file names
    await d.camera([saver], 1000, pad=140, top=60)
    await d.say(15)
    await basic_point(d, saver, "filename_prefix")
    name1 = url1.split("filename=")[1].split("&")[0]
    await d.js("(h) => window.__hv.show(h)", '<div style="margin-top:420px;text-align:center;color:#fff;font:600 34px Segoe UI,sans-serif">'
               '<div style="color:#c9c9d1;font-size:28px">MOSS/%model%_%seed%</div>'
               f'<div style="margin-top:14px;font:700 40px Consolas,monospace;color:#ffcc33">MOSS/{md_esc(name1)}</div></div>')
    await d._after_voice(0.4)
    await pe_hide(d)

    # 8. two stages: the second video's upscale and MOSS pass, then a second Telemetry for the refined image
    await d.camera({"box": MD_BOX}, 1200, pad=60, top=60)
    await d.say(16)
    async with d.fast(8.0):
        upmodel = await md_add(d, "UpscaleModelLoader", "Load Upscale Model", "upmodel", {"model_name": UP_UPSCALER})
        upscale = await md_add(d, "ImageUpscaleWithModel", "Upscale Image using Model", "upscale")
        await basic_link(d, upmodel, "UPSCALE_MODEL", upscale, "upscale_model", 600)
        await basic_link(d, sampler, "image", upscale, "image", 700)
        encode = await md_add(d, "VAEEncode", "VAE Encode", "encode")
        await basic_link(d, upscale, "IMAGE", encode, "pixels", 600)
        await basic_link(d, loader, "VAE", encode, "vae", 700)
        sampler2 = await md_add(d, "HaDeSMossKSampler", "MOSS KSampler", "sampler2",
                                {"sampler_name": "euler", "scheduler": "simple", "denoise": 0.25})
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (encode, "LATENT", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae"), (turbo, "turbo", "turbo")):
            await basic_link(d, a, out, sampler2, inp, 600)
        saver3 = await md_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", "saver3")
        await basic_link(d, sampler2, "image", saver3, "images", 600)
        await basic_link(d, telemetry, "json", saver3, "metadata", 700)
        await basic_link(d, sampler2, "info", telemetry, "info_6", 700)
    await d._after_voice(0.1)
    url2, _ = await lt_run(d, saver3)
    note2 = (next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))["outputs"]
             .get(str(saver3), {}).get("text") or [""])[0]
    chunks2 = await md_chunks(d, url2)
    await d.say(17)
    html = md_params_html(chunks2, "The refined 2048 image, Telemetry on both stages", mark=("Size",))
    if note2:
        html = html.replace('<div style="margin-bottom:18px">', f'<div style="color:#ffcc33;font:600 24px Segoe UI,sans-serif;margin-bottom:14px">{md_esc(note2)}</div><div style="margin-bottom:18px">', 1)
    await d.js("(h) => window.__hv.show(h)", html)
    await d._after_voice(0.4)
    await pe_hide(d)
    await d.camera([sampler2, saver3], 1000, pad=80, top=60)
    await d.say(18)
    async with d.fast(4.0):
        await d.camera({"box": MD_BOX}, 800, pad=50, top=60)
        tel2 = await md_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", "tel2", {"report_title": "Refine stage"})
        for k, (n, out) in enumerate(((loader, "info"), (enhancer, "info"), (sampler2, "info"), (turbo, "info"))):
            await basic_link(d, n, out, tel2, f"info_{k + 1}", 700)
        await basic_link(d, tel2, "json", saver3, "primary_stage_metadata", 700)
    await d._after_voice(0.1)
    url3, _ = await lt_run(d, saver3)
    chunks3 = await md_chunks(d, url3)
    await d.say(19)
    await d.js("(h) => window.__hv.show(h)", md_params_html(chunks3, "The refined image, primary_stage_metadata connected",
                                                            mark=("Size", "Denoising strength")))
    await d._after_voice(0.4)
    await pe_hide(d)

    # 9. a merge made in the workflow (the LoRA Toolkit example's PNG)
    await d.say(20)
    chunks4 = await md_chunks(d, MD_MERGE_PNG)
    await d.js("(h) => window.__hv.show(h)", md_params_html(chunks4, "LoRA Toolkit example: a model merged in the workflow", mark=("Model",)))
    await d._after_voice(0.2)
    await d.say(21)
    await d._after_voice(0.4)
    await pe_hide(d)

    # 10. privacy
    await d.camera([saver], 1000, pad=140, top=60)
    await d.say(22)
    await basic_point(d, saver, "hashes")
    await d.hold(1.2)
    await basic_point(d, saver, "embed_workflow")
    await d._after_voice(0.1)
    await d.say(23)
    await d.js("(h) => window.__hv.show(h)", '<div style="margin-top:380px;text-align:center;color:#fff">'
               '<div style="font:700 44px Consolas,monospace;color:#ffcc33">--disable-metadata</div>'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1;margin-top:18px">ComfyUI launch option: no text in the PNG at all</div></div>')
    await d._after_voice(0.4)
    await pe_hide(d)

    # 11. goodbye
    await d.camera({"box": MD_BOX}, 1300, pad=50, top=60)
    await d.move(W - 6, H * 0.5)
    await d.say(24)
    await d._after_voice(0.1)
    await d.say(25)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Metadata.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:60px"><img src="{url1}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Metadata: what your PNG remembers</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(26)
    await d.say(27)
    await d.end(1.5)


async def md_prepare(page):
    """Off camera: one 1024 run without the Saver loads the model, the text encoder, the VAE and the LoRA (no hash is computed, so
    the first save on camera hashes the model as a viewer's first save would)."""
    g = {k: v for k, v in up_api_graph(7, full=False).items() if k in ("1", "2", "3", "4", "5")}
    g["6"] = {"class_type": "PreviewImage", "inputs": {"images": ["5", 1]}}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


metadata_video.prepare = md_prepare


# ---- merge, save and telemetry (the sixth tutorial) ----

MS_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("Today: merging models. How to mix two models layer by layer, save the result as your own file, and keep its recipe.",
     "Hoy: mezclar modelos. Cómo combinar dos modelos capa a capa, guardar el resultado como tu propio archivo y conservar su receta."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 3
    ("Quick rebuild of the basic workflow, this time on MOSS-Image SFT, with Telemetry and the Image Saver from the last video.",
     "Montaje rápido del workflow básico, esta vez sobre MOSS-Image SFT, con Telemetry y el Image Saver del vídeo anterior.",
     "Quick rebuild of the basic workflow, this time on MOSS-Image S F T, with Telemetry and the Image Saver from the last video."),
    # 4: why merge
    ("Why merge? SFT gives polished, stock-photo pictures, but they all look alike. Base varies much more: film, black and white, "
     "different people.",
     "¿Por qué mezclar? SFT da fotos pulidas, de banco de imágenes, pero todas se parecen. Base varía mucho más: película, blanco y "
     "negro, gente distinta.",
     "Why merge? S F T gives polished, stock-photo pictures, but they all look alike. Base varies much more: film, black and white, "
     "different people."),
    ("And when the prompt doesn't say who, SFT leans one way. \"A woman sitting in a café\": SFT drew an East Asian woman in ten of "
     "sixteen seeds; Base in one or two.",
     "Y cuando el prompt no dice quién, SFT tira hacia un lado. \"Una mujer sentada en un café\": SFT dibujó una mujer asiática en "
     "diez de dieciséis semillas; Base, en una o dos.",
     "And when the prompt doesn't say who, S F T leans one way. A woman sitting in a café: S F T drew an East Asian woman in ten of "
     "sixteen seeds; Base in one or two."),
    ("So let's mix some of Base into SFT.",
     "Así que vamos a mezclar algo de Base en SFT.",
     "So let's mix some of Base into S F T."),
    # 7: the Block Merger
    ("Load Base with the core's Load Diffusion Model, and add the MOSS Model Block Merger: SFT as model one, Base as model two.",
     "Cargamos Base con el Load Diffusion Model del core y añadimos el MOSS Model Block Merger: SFT como modelo uno, Base como modelo "
     "dos.",
     "Load Base with the core's Load Diffusion Model, and add the MOSS Model Block Merger: S F T as model one, Base as model two."),
    ("Each slider is a group of six layers. Zero keeps model one, one takes model two, and half is half of each.",
     "Cada deslizador es un grupo de seis capas. Cero se queda con el modelo uno, uno toma el modelo dos, y la mitad es mitad de cada "
     "uno."),
    ("The last three are the text embedder, the time modulation, and the input and output layers.",
     "Los tres últimos son el text embedder, la modulación de tiempo y las capas de entrada y salida."),
    ("I measured which groups carry the change from SFT to Base: layers twenty-four to thirty-five carry the most. The tooltips give "
     "the numbers.",
     "Medí qué grupos llevan el cambio de SFT a Base: las capas 24 a 35 llevan la mayor parte. Los tooltips dan las cifras.",
     "I measured which groups carry the change from S F T to Base: layers twenty-four to thirty-five carry the most. The tooltips give "
     "the numbers."),
    ("For this video I'll use a test recipe: half of Base in layers eighteen to thirty-five, the rest SFT. It's a demonstration, not a "
     "recommendation.",
     "Para este vídeo usaré una receta de prueba: la mitad de Base en las capas 18 a 35, el resto SFT. Es una demostración, no una "
     "recomendación.",
     "For this video I'll use a test recipe: half of Base in layers eighteen to thirty-five, the rest S F T. It's a demonstration, not "
     "a recommendation."),
    # 12: the result
    ("Here it is, between the two: SFT's finish, with some of Base's variety. And it generated in the same time as SFT alone: about "
     "thirteen seconds here.",
     "Aquí está, entre los dos: el acabado de SFT con algo de la variedad de Base. Y generó en el mismo tiempo que SFT solo: unos "
     "trece segundos aquí.",
     "Here it is, between the two: S F T's finish, with some of Base's variety. And it generated in the same time as S F T alone: "
     "about thirteen seconds here."),
    ("And the bias? On the café prompt, about three of sixteen. On a fisherman portrait, it barely moved: thirteen or fourteen of "
     "sixteen, against sixteen on SFT.",
     "¿Y el sesgo? Con el prompt del café, unas tres de dieciséis. Con un retrato de pescador apenas se movió: trece o catorce de "
     "dieciséis, frente a dieciséis en SFT.",
     "And the bias? On the café prompt, about three of sixteen. On a fisherman portrait, it barely moved: thirteen or fourteen of "
     "sixteen, against sixteen on S F T."),
    ("The bias is spread over the whole model. Merging reduces it in part; naming who you want in the prompt works every time.",
     "El sesgo está repartido por todo el modelo. Mezclar lo reduce en parte; decir en el prompt a quién quieres funciona siempre."),
    ("For full control there's recipe override: one value per layer, forty-five numbers, instead of the sliders.",
     "Para control total está recipe override: un valor por capa, cuarenta y cinco números, en lugar de los deslizadores."),
    # 16: telemetry
    ("Plug the merger's info into Telemetry. The report lists the ratio of every layer, and the sampler knows it ran on SFT plus Base.",
     "Conectamos el info del merger a Telemetry. El informe da la proporción de cada capa, y el sampler sabe que corrió sobre SFT más "
     "Base.",
     "Plug the merger's info into Telemetry. The report lists the ratio of every layer, and the sampler knows it ran on S F T plus "
     "Base."),
    ("And as we saw in the last video, the PNG says \"merge\", with both models, and no model hash.",
     "Y como vimos en el vídeo anterior, el PNG dice \"merge\", con los dos modelos, y sin hash de modelo.",
     "And as we saw in the last video, the P N G says merge, with both models, and no model hash."),
    # 18: the Model Saver
    ("Happy with it? Save it with the MOSS Model Saver. Plug the merger's info into recipe, and the recipe goes inside the file.",
     "¿Te gusta? Guárdalo con el MOSS Model Saver. Conecta el info del merger a recipe, y la receta va dentro del archivo."),
    ("bfloat16 keeps the original precision: ten gigabytes. Stochastic rounding keeps even the smallest changes.",
     "bfloat16 mantiene la precisión original: diez gigas. El redondeo estocástico conserva hasta los cambios más pequeños.",
     "B float sixteen keeps the original precision: ten gigabytes. Stochastic rounding keeps even the smallest changes."),
    ("int8 ConvRot is Comfy-Org's smaller format: five and a half gigabytes.",
     "int8 ConvRot es el formato pequeño de Comfy-Org: cinco gigas y medio.",
     "Int eight Conv Rot is Comfy Org's smaller format: five and a half gigabytes."),
    ("The file goes to ComfyUI's output folder. Move it to models, diffusion models, to load it.",
     "El archivo va a la carpeta output de ComfyUI. Muévelo a models, diffusion_models, para cargarlo."),
    # 22: loading it back
    ("Load it like any other model. The bfloat16 file gives the same image as the live merge.",
     "Lo cargas como cualquier otro modelo. El archivo bfloat16 da la misma imagen que la mezcla en vivo.",
     "Load it like any other model. The B float sixteen file gives the same image as the live merge."),
    ("The int8 file rendered in six seconds instead of fifteen here, with small changes in the details.",
     "El archivo int8 generó en seis segundos en vez de quince aquí, con cambios pequeños en los detalles.",
     "The int eight file rendered in six seconds instead of fifteen here, with small changes in the details."),
    # 24: the Metadata Reader
    ("And months from now, the Metadata Reader tells you what this file is: its size and precision, and the recipe, with both models "
     "and the ratio of every layer.",
     "Y dentro de unos meses, el Metadata Reader te dice qué es este archivo: su tamaño y su precisión, y la receta, con los dos "
     "modelos y la proporción de cada capa."),
    # 25: the Multi-Model Merger
    ("One more node: the Multi-Model Merger, for two to four models with TIES, DARE, Task Arithmetic, Model Stock or SLERP.",
     "Un nodo más: el Multi-Model Merger, para dos a cuatro modelos con TIES, DARE, Task Arithmetic, Model Stock o SLERP.",
     "One more node: the Multi-Model Merger, for two to four models with ties, dare, task arithmetic, model stock or slerp."),
    ("On SFT and Base, SLERP at half gave about the same image as the Block Merger at half. It's meant for finetunes: give it the "
     "model they come from as base model.",
     "Con SFT y Base, SLERP a la mitad dio casi la misma imagen que el Block Merger a la mitad. Está pensado para finetunes: dale como "
     "base model el modelo del que vienen.",
     "On S F T and Base, slerp at half gave about the same image as the Block Merger at half. It's meant for finetunes: give it the "
     "model they come from as base model."),
    ("Which method is best depends on your models. I haven't measured a winner, so try them.",
     "Qué método es mejor depende de tus modelos. No he medido un ganador, así que pruébalos."),
    # 28: goodbye
    ("That's merging: mix, check with Telemetry, save with its recipe. And remember, the recipe here was only a test. Measure yours.",
     "Eso es mezclar: combinar, comprobar con Telemetry y guardar con su receta. Y recuerda, la receta de aquí era solo una prueba. "
     "Mide la tuya."),
    ("This workflow is free to download from the description.",
     "Este workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

MS_SFT, MS_BASE = "mossImage_sft_bf16.safetensors", "mossImage_base_bf16.safetensors"
MS_RECIPE = {"layers_00_05": 0.0, "layers_06_11": 0.0, "layers_12_17": 0.0, "layers_18_23": 0.5, "layers_24_29": 0.5, "layers_30_35": 0.5,
             "layers_36_41": 0.0, "text_embedder": 0.0, "time_modulation": 0.0, "io": 0.0}
MS_POS = {"loader": (0, 0), "enhancer": (440, 0), "latent": (940, 0), "sampler": (1380, 0), "saver": (1820, 0),
          "telemetry": (1380, 640), "report": (1820, 640), "base": (0, 560), "bm": (420, 560), "msaver": (900, 640),
          "file": (0, 1360), "meta": (440, 1360), "meta_text": (880, 1360), "multi": (1700, 1360)}
MS_BOX = [-260, -60, 2700, 1960]
MS_TOP = [-260, -60, 2300, 1200]
MS_OUT = Path("I:/HaDeS_MOSS/videos/comfy_output")
MS_MODELS = Path(r"H:/Comfy_UI_V61/ComfyUI/models/diffusion_models/HaDeS_video")
# measured images and contact sheets (tools/RESULTS.md, "Merge, save and telemetry: a test recipe"), copied by ms_prepare
MS_MEASURE = Path("I:/HaDeS_MOSS/videos/merge_save/measure")
MS_IMG = lambda name: f"/view?filename={name}&subfolder=ms_measure&type=output"  # noqa: E731
MS_FILE = {}


async def ms_add(d, type_, query, key, values=None):
    """basic_add at MS_POS[key], with the node's values set the moment it is created."""
    if values:
        await d.js("""([type, v]) => { const g = window.__hv.app().graph, prev = g.onNodeAdded;
            g.onNodeAdded = function (n) { if (n.type === type) { for (const [k, x] of Object.entries(v)) { const w = n.widgets?.find(w => w.name === k);
                if (w) w.value = x; } g.onNodeAdded = prev; } return prev?.apply(this, arguments); }; }""", [type_, values])
    n = await basic_add(d, type_, query, MS_POS[key])
    if values:
        await up_set(d, n, values)
    return n


async def ms_text(d, key, size):
    n = await ms_add(d, "PreviewAny", "Preview as Text", key)
    await d.js("([id, s]) => { const n = window.__hv.node(id); n.setSize(s); n.setDirtyCanvas(true, true); }", [n, list(size)])
    return n


async def ms_last_secs(d):
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    st = {m[0]: m[1]["timestamp"] for m in run["status"]["messages"]}
    return (st["execution_success"] - st["execution_start"]) / 1000


def ms_card(html, top=300):
    return f'<div style="color:#fff;text-align:center;margin-top:{top}px">{html}</div>'


def ms_bias_html(recipe=True):
    cell = "padding:14px 30px;border-bottom:1px solid #333"
    rows = [("", "SFT", "Test recipe", "Base"), ("\u201ca woman sitting in a caf\u00e9\u201d", "10 / 16", "~3 / 16", "~1-2 / 16"),
            ("studio fisherman portrait", "16 / 16", "~13-14 / 16", "~6 / 16")]
    rows = rows if recipe else [(r[0], r[1], r[3]) for r in rows[:2]]
    body = "".join("<tr>" + "".join(f'<td style="{cell};{"color:#ffcc33;font-weight:700" if i == 0 or j == 0 else ""}">{c}</td>'
                                     for j, c in enumerate(r)) + "</tr>" for i, r in enumerate(rows))
    return ('<div style="margin-top:250px;color:#e6e6ea;display:flex;flex-direction:column;align-items:center">'
            '<div style="font:700 34px Segoe UI,sans-serif;color:#fff;margin-bottom:20px">East Asian faces, 16 seeds, counted by eye</div>'
            f'<table style="border-collapse:collapse;font:500 32px Segoe UI,sans-serif;background:#141418;border:2px solid #333;'
            f'border-radius:12px">{body}</table></div>')


async def merge_save(d: Director):
    """The sixth tutorial: SFT and Base mixed with the Block Merger, checked with Telemetry, saved with the Model Saver, read back."""
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "Merge, Save and Telemetry", 6.0)
    for i in range(3):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. quick rebuild on SFT, Telemetry and the Image Saver
    await d.camera({"box": MS_TOP}, 900, pad=60, top=60)
    await d.say(3)
    async with d.fast(6.0):
        loader = await ms_add(d, "HaDeSMossLoader", "MOSS Loader", "loader", dict(BASIC_LOADER, unet_name=MS_SFT))
        enhancer = await ms_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", "enhancer",
                                {"mode": "as written", "idea": BASIC_PROMPT, "seed": 1})
        await basic_link(d, loader, "CLIP", enhancer, "clip", 600)
        latent = await ms_add(d, "HaDeSMossLatent", "MOSS Latent", "latent",
                              {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
        sampler = await ms_add(d, "HaDeSMossKSampler", "MOSS KSampler", "sampler",
                               {"steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple"})
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae")):
            await basic_link(d, a, out, sampler, inp, 600)
        saver = await ms_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", "saver")
        await basic_link(d, sampler, "image", saver, "images", 600)
        telemetry = await ms_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", "telemetry")
        for k, n in enumerate((loader, enhancer, latent, sampler)):
            await basic_link(d, n, "info", telemetry, f"info_{k + 1}", 600)
        await basic_link(d, telemetry, "json", saver, "metadata", 600)
        url_sft, t_sft = await lt_run(d, saver)
    await d._after_voice(0.1)

    # 3. why merge: the café prompt on SFT and on Base
    await d.say(4)
    await d.js("(h) => window.__hv.show(h)", lt_cols([(MS_IMG(MS_FILE["sheet_cafe_sft"]), "SFT · 16 seeds"),
                                                      (MS_IMG(MS_FILE["sheet_cafe_base"]), "Base · 16 seeds")], side=760, top=110))
    await d._after_voice(0.2)
    await d.say(5)
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               ms_bias_html(recipe=False))
    await d.say(6)
    await d._after_voice(0.2)
    await pe_hide(d)

    # 4. the Block Merger
    await d.camera({"box": MS_TOP}, 1000, pad=60, top=60)
    await d.say(7)
    base = await ms_add(d, "UNETLoader", "Load Diffusion Model", "base", {"unet_name": MS_BASE, "weight_dtype": "default"})
    bm = await ms_add(d, "HaDeSMossModelBlockMerger", "MOSS Model Block Merger", "bm", MS_RECIPE)
    async with d.fast(2.0):
        await basic_link(d, loader, "MODEL", bm, "model1", 700)
        await basic_link(d, base, "MODEL", bm, "model2", 700)
        await basic_link(d, bm, "MODEL", sampler, "model", 700)
    await d._after_voice(0.1)
    await d.camera([bm], 1000, pad=120, top=60)
    await d.say(8)
    await basic_point(d, bm, "layers_00_05")
    await d._after_voice(0.1)
    await d.say(9)
    await basic_point(d, bm, "text_embedder")
    await d.hold(0.8)
    await basic_point(d, bm, "io")
    await d._after_voice(0.1)
    await d.say(10)
    await basic_point(d, bm, "layers_24_29")
    await d.hold(1.0)
    await basic_point(d, bm, "layers_30_35")
    await d._after_voice(0.1)
    await d.say(11)
    await basic_point(d, bm, "layers_18_23")
    await d._after_voice(0.2)

    # 5. the result
    url_merge, t_merge = await lt_run(d, saver)
    print(f"SFT {t_sft:.1f} s, recipe {t_merge:.1f} s", flush=True)
    await d.say(12)
    await d.js("(h) => window.__hv.show(h)", lt_cols([(url_sft, "SFT"), (url_merge, "test recipe"),
                                                      (MS_IMG(MS_FILE["base_fisherman_1013"]), "Base")]))
    await d._after_voice(0.4)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }", ms_bias_html())
    await d.say(13)
    await d._after_voice(0.3)
    await d.say(14)
    await d._after_voice(0.4)
    await pe_hide(d)
    await d.camera([bm], 1000, pad=120, top=60)
    await d.say(15)
    await basic_point(d, bm, "recipe_override")
    await d._after_voice(0.3)

    # 6. Telemetry
    await d.camera({"box": MS_TOP}, 1000, pad=60, top=60)
    await d.say(16)
    await basic_link(d, bm, "info", telemetry, "info_5", 900)
    report = await ms_text(d, "report", (600, 520))
    await basic_link(d, telemetry, "report", report, "source", 700)
    url_tel, _ = await lt_run(d, saver)
    await d.camera([report], 1000, pad=40, top=60)
    await d._after_voice(0.4)
    await d.say(17)
    chunks = await md_chunks(d, url_tel)
    await d.js("(h) => window.__hv.show(h)", md_params_html(chunks, "The PNG of the merge", mark=("Model",)))
    await d._after_voice(0.4)
    await pe_hide(d)

    # 7. the Model Saver
    await d.camera({"box": MS_TOP}, 1000, pad=60, top=60)
    await d.say(18)
    msaver = await ms_add(d, "HaDeSMossModelSaver", "MOSS Model Saver", "msaver",
                          {"filename_prefix": "diffusion_models/MOSS_video_recipe", "precision": "bfloat16", "include_metadata": True,
                           "custom_author": "HaDeS", "custom_model_title": "", "rounding": "stochastic"})
    async with d.fast(2.0):
        await basic_link(d, bm, "MODEL", msaver, "model", 700)
        await basic_link(d, bm, "info", msaver, "recipe", 700)
    await d.camera([msaver], 1000, pad=140, top=60)
    await d._after_voice(0.1)
    import os
    files = {}
    for prec, prefix, line, label in (("bfloat16", "MOSS_video_recipe", 19, "bfloat16 · stochastic rounding"),
                                      ("int8 ConvRot (Comfy-Org layout)", "MOSS_video_recipe_int8", 20, "int8 ConvRot")):
        if prefix != "MOSS_video_recipe":
            await basic_point(d, msaver, "precision", {"precision": prec, "filename_prefix": f"diffusion_models/{prefix}"})
        await lt_run(d)
        secs = await ms_last_secs(d)
        f = sorted((MS_OUT / "diffusion_models").glob(prefix + "_0*.safetensors"), key=os.path.getmtime)[-1]
        files[prefix] = f
        size = os.path.getsize(f) / 1e9
        print(f"saved {f.name}: {size:.2f} GB in {secs:.1f} s", flush=True)
        await d.say(line)
        await d.js("(h) => window.__hv.show(h)", ms_card(
            f'<div style="font:700 44px Segoe UI,sans-serif">{label}</div>'
            f'<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:18px">{size:.2f} GB on disk · saved in {secs:.0f} s</div>'
            f'<div style="font:500 28px Consolas,monospace;color:#c9c9d1;margin-top:18px">output/diffusion_models/{f.name}</div>', 340))
        await d._after_voice(0.3)
        await pe_hide(d)
    # the files are moved where Load Diffusion Model reads them
    MS_MODELS.mkdir(parents=True, exist_ok=True)
    import shutil
    moving = asyncio.ensure_future(asyncio.gather(*(asyncio.to_thread(shutil.move, str(f), str(MS_MODELS / f.name)) for f in files.values())))
    await d.say(21)
    await d.js("(h) => window.__hv.show(h)", ms_card(
        '<div style="font:500 32px Consolas,monospace;color:#c9c9d1">ComfyUI/output/diffusion_models/</div>'
        '<div style="font:700 50px Segoe UI,sans-serif;color:#ffcc33;margin:16px 0">\u2193 move \u2193</div>'
        '<div style="font:600 34px Consolas,monospace;color:#fff">ComfyUI/models/diffusion_models/</div>', 360))
    await d._after_voice(0.4)
    async with d.fast(4.0):
        while not moving.done():
            await d.hold(0.5)
    moving.result()
    await lt_refresh(d)
    await pe_hide(d)

    # 8. loading it back
    bf = "HaDeS_video\\" + files["MOSS_video_recipe"].name
    i8 = "HaDeS_video\\" + files["MOSS_video_recipe_int8"].name
    await d.camera({"box": MS_BOX}, 1000, pad=50, top=60)
    await d.say(22)
    file = await ms_add(d, "UNETLoader", "Load Diffusion Model", "file", {"unet_name": bf, "weight_dtype": "default"})
    await basic_link(d, file, "MODEL", sampler, "model", 900)
    url_file, t_file = await lt_run(d, saver)
    await d.js("(h) => window.__hv.show(h)", lt_cols([(url_merge, "live merge"), (url_file, "the saved bfloat16 file")], side=720))
    await d._after_voice(0.4)
    await pe_hide(d)
    await d.camera([file], 1000, pad=140, top=60)
    await basic_point(d, file, "unet_name", {"unet_name": i8})
    url_i8, t_i8 = await lt_run(d, saver)
    print(f"bf16 file {t_file:.1f} s, int8 file {t_i8:.1f} s", flush=True)
    await d.say(23)
    await d.js("(h) => window.__hv.show(h)", lt_cols([(url_file, "the bfloat16 file"), (url_i8, "the int8 ConvRot file")], side=720))
    await d._after_voice(0.4)
    await pe_hide(d)

    # 9. the Metadata Reader
    await d.camera({"box": [-260, 1200, 2300, 1960]}, 1000, pad=50, top=60)
    meta = await ms_add(d, "HaDeSMossMetadataReader", "MOSS Metadata Reader", "meta", {"lora_name": "(none)", "model_name": bf})
    meta_text = await ms_text(d, "meta_text", (780, 560))
    await basic_link(d, meta, "info", meta_text, "source", 700)
    await lt_run(d)
    await d.camera([meta_text], 1000, pad=30, top=60)
    await d.say(24)
    await d._after_voice(0.6)

    # 10. the Multi-Model Merger
    await d.camera({"box": [-260, 400, 2700, 1960]}, 1000, pad=50, top=60)
    await d.say(25)
    multi = await ms_add(d, "HaDeSMossMultiModelMerger", "MOSS Multi-Model Merger", "multi",
                         {"method": "SLERP", "density": 0.5, "scale": 1.0, "weight_1": 1.0, "weight_2": 1.0})
    async with d.fast(2.0):
        await d.camera({"box": MS_BOX}, 600, pad=50, top=60)
        await basic_link(d, loader, "MODEL", multi, "model_1", 700)
        await basic_link(d, base, "MODEL", multi, "model_2", 700)
    await d.camera([multi], 1000, pad=140, top=60)
    await basic_point(d, multi, "method")
    await d._after_voice(0.1)
    await d.say(26)
    await d.js("(h) => window.__hv.show(h)", lt_cols([(MS_IMG(MS_FILE["half_fisherman_1013"]), "Block Merger, every group 0.5"),
                                                      (MS_IMG(MS_FILE["slerp_fisherman_1013"]), "SLERP 0.5")], side=720))
    await d._after_voice(0.3)
    await pe_hide(d)
    await d.say(27)
    await d._after_voice(0.4)

    # 11. goodbye
    await d.camera({"box": MS_BOX}, 1300, pad=50, top=60)
    await d.move(W - 6, H * 0.5)
    await d.say(28)
    await d._after_voice(0.1)
    await d.say(29)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Merge, Save and Telemetry.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:60px"><img src="{url_merge}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Merge, Save and Telemetry</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(30)
    await d.say(31)
    await d.end(1.5)


async def ms_prepare(page):
    """Off camera: the measured images and sheets served from comfy_output/ms_measure, last take's files out of the way, and one
    short merge render that loads the text encoder, the VAE, SFT and Base."""
    import glob, os, shutil
    dst = MS_OUT / "ms_measure"
    for name in ("sheet_cafe_sft", "sheet_cafe_base"):
        shutil.copy(MS_MEASURE / (name + ".png"), dst / (name + ".png"))
        MS_FILE[name] = name + ".png"
    for prefix in ("base_fisherman_1013", "half_fisherman_1013", "slerp_fisherman_1013"):
        f = sorted(glob.glob(str(dst / (prefix + "_*.png"))), key=os.path.getmtime)
        if not f:
            raise SystemExit(f"measured image {prefix} missing")
        MS_FILE[prefix] = Path(f[-1]).name
    for f in (MS_OUT / "diffusion_models").glob("MOSS_video_recipe*.safetensors"):
        raise SystemExit(f"{f} is in the output folder: the counter would not start at 00001")
    g = {"1": {"class_type": "HaDeSMossLoader", "inputs": dict(BASIC_LOADER, unet_name=MS_SFT, weight_dtype="default", clip_device="default")},
         "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MS_BASE, "weight_dtype": "default"}},
         "3": {"class_type": "HaDeSMossModelBlockMerger", "inputs": {"model1": ["1", 0], "model2": ["2", 0], **MS_RECIPE, "recipe_override": ""}},
         "4": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {"clip": ["1", 1], "idea": "a lighthouse", "mode": "as written", "width": 1024,
               "height": 1024, "temperature": 0.7, "seed": 0, "max_tokens": 1024, "negative": ""}},
         "5": {"class_type": "HaDeSMossLatent", "inputs": {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024,
               "batch_size": 1, "seed": 7}},
         "6": {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["3", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
               "seed": 7, "steps": 2, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0, "flow_shift": 0.0, "slg": "off",
               "vae": ["1", 2]}},
         "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 1]}}}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


merge_save.prepare = ms_prepare


# ---- compare models and LoRAs (the seventh tutorial) ----

CM_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("Last time I said: measure your merge. Today, how: comparing models and LoRAs side by side, fairly, and even blind.",
     "La última vez dije: mide tu mezcla. Hoy, cómo: comparar modelos y LoRAs lado a lado, de forma justa, e incluso a ciegas.",
     "Last time I said: measure your merge. Today, how: comparing models and Loras side by side, fairly, and even blind."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 3
    ("Quick setup: SFT in the MOSS Loader, Base and Bulle d'Encre with Load Diffusion Model, and last video's test recipe in the Block "
     "Merger.",
     "Preparación rápida: SFT en el MOSS Loader, Base y Bulle d'Encre con Load Diffusion Model, y la receta de prueba del vídeo anterior "
     "en el Block Merger.",
     "Quick setup: S F T in the MOSS Loader, Base and Bulle d'Encre with Load Diffusion Model, and last video's test recipe in the Block "
     "Merger."),
    # 4: the Compare Sampler
    ("Instead of a KSampler, the MOSS Compare Sampler. Up to four models, A to D, and the same prompt, latent, seed and settings for all "
     "of them.",
     "En lugar de un KSampler, el MOSS Compare Sampler. Hasta cuatro modelos, de la A a la D, y el mismo prompt, latente, semilla y "
     "ajustes para todos.",
     "Instead of a K Sampler, the MOSS Compare Sampler. Up to four models, A to D, and the same prompt, latent, seed and settings for "
     "all of them."),
    ("Twenty-five steps, CFG two point five. Run it.",
     "Veinticinco pasos, CFG dos coma cinco. Lo lanzamos."),
    # 6: the sheet
    ("One sheet, four models. Each label comes from what the model carries: its file, or \"merge\" with both models. You don't type them.",
     "Una hoja, cuatro modelos. Cada etiqueta sale de lo que lleva el modelo: su archivo, o \"merge\" con los dos modelos. No las "
     "escribes tú.",
     "One sheet, four models. Each label comes from what the model carries: its file, or merge with both models. You don't type them."),
    ("Underneath, what they all share: seed, steps, CFG, sampler and the prompt.",
     "Debajo, lo que comparten todos: semilla, pasos, CFG, sampler y el prompt."),
    ("Each column is exactly the image a MOSS KSampler gives with that model. It isn't faster than four KSamplers, and it needs no more "
     "memory: it just puts them side by side for you.",
     "Cada columna es exactamente la imagen que da un MOSS KSampler con ese modelo. No es más rápido que cuatro KSamplers, ni necesita "
     "más memoria: simplemente te los pone lado a lado.",
     "Each column is exactly the image a MOSS K Sampler gives with that model. It isn't faster than four K Samplers, and it needs no "
     "more memory: it just puts them side by side for you."),
    # 9: LoRAs
    ("Now LoRAs. Remember the LoRA we extracted from Bulle d'Encre in the fourth video? Here it is, on Bulle's base, rebuilt with the "
     "Block Merger.",
     "Ahora LoRAs. ¿Recuerdas la LoRA que sacamos de Bulle d'Encre en el cuarto vídeo? Aquí está, sobre la base de Bulle, reconstruida "
     "con el Block Merger.",
     "Now Loras. Remember the Lora we extracted from Bulle d'Encre in the fourth video? Here it is, on Bulle's base, rebuilt with the "
     "Block Merger."),
    ("Four Multi-LoRAs: one switched off, then strength zero point five, zero point eight and one.",
     "Cuatro Multi-LoRA: uno apagado, y luego fuerza cero coma cinco, cero coma ocho y uno.",
     "Four Multi-Loras: one switched off, then strength zero point five, zero point eight and one."),
    ("Fifteen steps and euler, Bulle d'Encre's settings. Run.",
     "Quince pasos y euler, los ajustes de Bulle d'Encre. Lanzamos.",
     "Fifteen steps and oiler, Bulle d'Encre's settings. Run."),
    # 12: the result
    ("The labels write themselves: the bare base when the Multi-LoRA is off, then the LoRA with its strength. Switching a row off "
     "compares without rewiring.",
     "Las etiquetas se escriben solas: la base sola cuando el Multi-LoRA está apagado, y luego la LoRA con su fuerza. Apagar una fila "
     "compara sin recablear.",
     "The labels write themselves: the bare base when the Multi-Lora is off, then the Lora with its strength. Switching a row off "
     "compares without rewiring."),
    ("And strength isn't linear. Against Bulle d'Encre itself, zero point five barely moved, zero point eight got part of the way, and "
     "most of the look arrived between zero point eight and one.",
     "Y la fuerza no es lineal. Comparado con la propia Bulle d'Encre, cero coma cinco apenas se movió, cero coma ocho hizo parte del "
     "camino, y casi todo el estilo llegó entre cero coma ocho y uno."),
    ("A Turbo LoRA on the turbo input applies to every column, so you can compare at eight steps too.",
     "Una LoRA Turbo en la entrada turbo se aplica a todas las columnas, así que también puedes comparar a ocho pasos.",
     "A Turbo Lora on the turbo input applies to every column, so you can compare at eight steps too."),
    # 15: blind
    ("Now the fair way: blind. Switch blind on, and the panels are shuffled and numbered, with no names.",
     "Ahora la forma justa: a ciegas. Activa blind, y los paneles se barajan y se numeran, sin nombres."),
    ("Pick your favourite first. Then look at the key: it's in the info output and inside the PNG, never on the sheet.",
     "Elige primero tu favorita. Después mira la clave: está en la salida info y dentro del PNG, nunca en la hoja.",
     "Pick your favourite first. Then look at the key: it's in the info output and inside the P N G, never on the sheet."),
    ("Blind seed gives a new order of the same images, so you can vote again without remembering positions.",
     "Blind seed da un orden nuevo de las mismas imágenes, para votar otra vez sin recordar posiciones."),
    # 18: the Compare Sheet
    ("And for anything else, the MOSS Compare Sheet: any images side by side, here two seeds, labelled from each KSampler's info.",
     "Y para todo lo demás, el MOSS Compare Sheet: cualquier imagen lado a lado, aquí dos semillas, con la etiqueta sacada del info de "
     "cada KSampler.",
     "And for anything else, the MOSS Compare Sheet: any images side by side, here two seeds, labelled from each K Sampler's info."),
    # 19: goodbye
    ("That's comparing: same seed, same settings, labels you can trust, and blind when it matters.",
     "Eso es comparar: misma semilla, mismos ajustes, etiquetas fiables y a ciegas cuando importa."),
    ("This workflow is free to download from the description.",
     "Este workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

CM_BULLE = "HaDeS_MOSS_preview_beta.safetensors"
CM_LORA = "bulle_extracted.safetensors"
CM_BULLE_BASE = {**dict.fromkeys(MS_RECIPE, 0.0), "layers_24_29": 0.3}
CM_POS = {"loader": (0, 0), "enhancer": (440, 0), "latent": (940, 0), "base": (0, 520), "bulle": (0, 660), "rec": (440, 780),
          "cmp": (1400, 0), "bb": (0, 1300), "ml0": (440, 1300), "ml1": (880, 1300), "ml2": (440, 1620), "ml3": (880, 1620),
          "cmp2": (1400, 1300), "info2": (1900, 1300), "ks_a": (0, 2500), "ks_b": (440, 2500), "sheet": (880, 2500)}
CM_TOP = [-260, -60, 2300, 1280]
CM_MID = [-260, 1200, 2700, 2400]
CM_LOW = [-260, 2400, 1800, 3300]
CM_BOX = [-260, -60, 2700, 3300]
CM_IMG = lambda name: f"/view?filename={name}&subfolder=cm_measure&type=output"  # noqa: E731
CM_FILE = {}


async def cm_add(d, type_, query, key, values=None):
    if values:
        await d.js("""([type, v]) => { const g = window.__hv.app().graph, prev = g.onNodeAdded;
            g.onNodeAdded = function (n) { if (n.type === type) { for (const [k, x] of Object.entries(v)) { const w = n.widgets?.find(w => w.name === k);
                if (w) w.value = x; } g.onNodeAdded = prev; } return prev?.apply(this, arguments); }; }""", [type_, values])
    n = await basic_add(d, type_, query, CM_POS[key])
    if values:
        await up_set(d, n, values)
    return n


def cm_sheet_html(url, top=260, width=1800):
    return f'<div style="margin-top:{top}px;display:flex;justify-content:center"><img src="{url}" style="width:{width}px;border-radius:6px"></div>'


async def cm_compare(d, key, values, models, enhancer, latent, loader):
    cmp = await cm_add(d, "HaDeSMossCompareSampler", "MOSS Compare Sampler", key, values)
    for s, m in zip("ABCD", models):
        await basic_link(d, m, "MODEL", cmp, f"model_{s}", 600)
    for a, out, inp in ((enhancer, "positive", "positive"), (enhancer, "negative", "negative"), (latent, "latent", "latent_image"),
                        (latent, "seed", "seed"), (loader, "VAE", "vae"), (enhancer, "info", "prompt_info")):
        await basic_link(d, a, out, cmp, inp, 600)
    return cmp


async def compare_video(d: Director):
    """The seventh tutorial: four models on one sheet, a LoRA at four strengths, blind mode, the Compare Sheet."""
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "Compare models and LoRAs", 6.0)
    for i in range(3):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. quick setup
    await d.camera({"box": CM_TOP}, 900, pad=60, top=60)
    await d.say(3)
    async with d.fast(6.0):
        loader = await cm_add(d, "HaDeSMossLoader", "MOSS Loader", "loader", dict(BASIC_LOADER, unet_name=MS_SFT))
        enhancer = await cm_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", "enhancer",
                                {"mode": "as written", "idea": BASIC_PROMPT, "seed": 1})
        await basic_link(d, loader, "CLIP", enhancer, "clip", 600)
        latent = await cm_add(d, "HaDeSMossLatent", "MOSS Latent", "latent",
                              {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
        base = await cm_add(d, "UNETLoader", "Load Diffusion Model", "base", {"unet_name": MS_BASE, "weight_dtype": "default"})
        bulle = await cm_add(d, "UNETLoader", "Load Diffusion Model", "bulle", {"unet_name": CM_BULLE, "weight_dtype": "default"})
        rec = await cm_add(d, "HaDeSMossModelBlockMerger", "MOSS Model Block Merger", "rec", MS_RECIPE)
        await basic_link(d, loader, "MODEL", rec, "model1", 600)
        await basic_link(d, base, "MODEL", rec, "model2", 600)
    await d._after_voice(0.1)

    # 3. the Compare Sampler
    await d.say(4)
    async with d.fast(3.0):
        cmp = await cm_compare(d, "cmp", {"steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple",
                                          "filename_prefix": "MOSS/compare/HaDeS_MOSS_models"},
                               (loader, base, rec, bulle), enhancer, latent, loader)
    await d._after_voice(0.1)
    await d.camera([cmp], 1000, pad=120, top=60)
    await d.say(5)
    await basic_point(d, cmp, "steps")
    await d.hold(0.6)
    await basic_point(d, cmp, "cfg")
    await d._after_voice(0.1)
    url_models, t_models = await lt_run(d, cmp)
    print(f"models sheet {t_models:.1f} s", flush=True)

    # 4. the sheet
    await d.js("(h) => window.__hv.show(h)", cm_sheet_html(url_models))
    await d.say(6)
    await d._after_voice(0.2)
    await d.say(7)
    await d._after_voice(0.2)
    await d.say(8)
    await d._after_voice(0.4)
    await pe_hide(d)

    # 5. LoRAs
    await d.camera({"box": CM_MID}, 1000, pad=60, top=60)
    await d.say(9)
    bb = await cm_add(d, "HaDeSMossModelBlockMerger", "MOSS Model Block Merger", "bb", CM_BULLE_BASE)
    async with d.fast(2.0):
        await d.camera({"box": CM_BOX}, 600, pad=50, top=60)
        await basic_link(d, loader, "MODEL", bb, "model1", 600)
        await basic_link(d, base, "MODEL", bb, "model2", 600)
        await d.camera({"box": CM_MID}, 600, pad=60, top=60)
    await d._after_voice(0.1)
    await d.say(10)
    mls = []
    for i, st in enumerate((None, 0.5, 0.8, 1.0)):
        rows = [{"name": CM_LORA, "on": True, "strength": st if st is not None else 1.0}]
        ml = await cm_add(d, "HaDeSMossMultiLora", "MOSS Multi-LoRA", f"ml{i}", {"loras": json.dumps(rows), "enabled": st is not None})
        await basic_link(d, bb, "MODEL", ml, "model", 500)
        mls.append(ml)
    await d._after_voice(0.1)
    await d.say(11)
    async with d.fast(3.0):
        await d.camera({"box": CM_BOX}, 600, pad=50, top=60)
        cmp2 = await cm_compare(d, "cmp2", {"steps": 15, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple",
                                            "filename_prefix": "MOSS/compare/HaDeS_MOSS_loras"}, mls, enhancer, latent, loader)
        info2 = await cm_add(d, "PreviewAny", "Preview as Text", "info2")
        await d.js("([id, s]) => { const n = window.__hv.node(id); n.setSize(s); n.setDirtyCanvas(true, true); }", [info2, [700, 420]])
        await basic_link(d, cmp2, "info", info2, "source", 600)
        await d.camera({"box": CM_MID}, 600, pad=60, top=60)
    await d._after_voice(0.1)
    url_loras, t_loras = await lt_run(d, cmp2)
    print(f"LoRA sheet {t_loras:.1f} s", flush=True)

    # 6. the result
    await d.js("(h) => window.__hv.show(h)", cm_sheet_html(url_loras))
    await d.say(12)
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }", '<div style="display:flex;flex-direction:column;align-items:center">' + lt_cols(
        [(CM_IMG(CM_FILE[k]), t) for k, t in (("off", "LoRA off · 0.86"), ("x0.5", "x0.5 · 0.74"), ("x0.8", "x0.8 · 0.59"),
                                                ("x1.0", "x1.0 · 0.26"), ("bulle", "Bulle d'Encre"))], side=330, top=330)
        + '<div style="text-align:center;color:#c9c9d1;font:600 26px Segoe UI,sans-serif;margin-top:26px">'
          'distance to Bulle d\'Encre itself (1 = another seed)</div></div>')
    await d.say(13)
    await d._after_voice(0.4)
    await pe_hide(d)
    await d.camera([cmp2], 1000, pad=140, top=60)
    await d.say(14)
    await d._after_voice(0.3)

    # 7. blind
    await d.say(15)
    await basic_point(d, cmp2, "blind", {"blind": True})
    await d._after_voice(0.1)
    url_blind, _ = await lt_run(d, cmp2)
    await d.js("(h) => window.__hv.show(h)", cm_sheet_html(url_blind))
    await d.hold(2.0)
    await d.say(16)
    await d._after_voice(0.3)
    await pe_hide(d)
    await d.camera([info2], 1000, pad=40, top=60)
    await d.hold(2.5)
    await d.camera([cmp2], 1000, pad=140, top=60)
    await d.say(17)
    await basic_point(d, cmp2, "blind_seed", {"blind_seed": 1})
    await d._after_voice(0.1)
    url_blind1, _ = await lt_run(d, cmp2)
    await d.js("(h) => window.__hv.show(h)", cm_sheet_html(url_blind1))
    await d.hold(2.5)
    await pe_hide(d)

    # 8. the Compare Sheet
    await d.camera({"box": CM_LOW}, 1000, pad=60, top=60)
    await d.say(18)
    async with d.fast(4.0):
        kss = []
        for key, seed in (("ks_a", 1013), ("ks_b", 1014)):
            k = await cm_add(d, "HaDeSMossKSampler", "MOSS KSampler", key,
                             {"seed": seed, "steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple"})
            kss.append(k)
        sheet = await cm_add(d, "HaDeSMossCompareSheet", "MOSS Compare Sheet", "sheet", {"filename_prefix": "MOSS/compare/HaDeS_MOSS_sheet"})
        await d.camera({"box": CM_BOX}, 600, pad=50, top=60)
        for k in kss:
            for a, out, inp in ((base, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                                (latent, "latent", "latent_image"), (loader, "VAE", "vae")):
                await basic_link(d, a, out, k, inp, 500)
        for s, k in zip("AB", kss):
            await basic_link(d, k, "image", sheet, f"image_{s}", 500)
            await basic_link(d, k, "info", sheet, f"info_{s}", 500)
        await d.camera({"box": CM_LOW}, 600, pad=60, top=60)
    url_sheet, _ = await lt_run(d, sheet)
    await d.js("(h) => window.__hv.show(h)", cm_sheet_html(url_sheet, top=220, width=1400))
    await d._after_voice(0.4)
    await pe_hide(d)

    # 9. goodbye
    await d.camera({"box": CM_BOX}, 1300, pad=50, top=60)
    await d.move(W - 6, H * 0.5)
    await d.say(19)
    await d._after_voice(0.1)
    await d.say(20)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Compare Models.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="margin-top:160px;display:flex;flex-direction:column;align-items:center;gap:50px">'
               f'<img src="{url_models}" style="width:1700px;border-radius:6px">'
               '<div style="color:#fff;text-align:center"><div style="font:900 64px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:14px">Compare models and LoRAs</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:24px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(21)
    await d.say(22)
    await d.end(1.5)


async def cm_prepare(page):
    """Off camera: the measured LoRA columns and Bulle d'Encre's image, the LoRA in models/loras, and one short render that loads the
    text encoder, the VAE, SFT, Base and Bulle d'Encre."""
    import glob, os
    dst = MS_OUT / "cm_measure"
    cols = sorted(glob.glob(str(dst / "loras_l1_col_*.png")), key=os.path.getmtime)[-4:]
    if len(cols) != 4:
        raise SystemExit("measured LoRA columns missing")
    for k, f in zip(("off", "x0.5", "x0.8", "x1.0"), cols):
        CM_FILE[k] = Path(f).name
    CM_FILE["bulle"] = Path(sorted(glob.glob(str(dst / "bulle_15_1013_*.png")), key=os.path.getmtime)[-1]).name
    if not (COMFY / "models" / "loras" / CM_LORA).exists():
        raise SystemExit(f"{CM_LORA} missing from models/loras (extract it as in the fourth video)")
    g = {"1": {"class_type": "HaDeSMossLoader", "inputs": dict(BASIC_LOADER, unet_name=MS_SFT, weight_dtype="default", clip_device="default")},
         "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MS_BASE, "weight_dtype": "default"}},
         "3": {"class_type": "UNETLoader", "inputs": {"unet_name": CM_BULLE, "weight_dtype": "default"}},
         "4": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {"clip": ["1", 1], "idea": "a lighthouse", "mode": "as written", "width": 1024,
               "height": 1024, "temperature": 0.7, "seed": 0, "max_tokens": 1024, "negative": ""}},
         "5": {"class_type": "HaDeSMossLatent", "inputs": {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024,
               "batch_size": 1, "seed": 7}}}
    for i, m in enumerate((["1", 0], ["2", 0], ["3", 0])):
        g[f"k{i}"] = {"class_type": "HaDeSMossKSampler", "inputs": {"model": m, "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
                      "seed": 7, "steps": 2, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0, "flow_shift": 0.0,
                      "slg": "off", "vae": ["1", 2]}}
        g[f"p{i}"] = {"class_type": "PreviewImage", "inputs": {"images": [f"k{i}", 1]}}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


compare_video.prepare = cm_prepare


# ---- layout + text (the eighth tutorial) ----

L8_LINES = [
    ("Hi, I'm HaDeS, and welcome back.",
     "Hola, soy HaDeS, y bienvenido de nuevo."),
    ("We've generated, merged and compared. Today: deciding where each thing goes in the picture, and checking that the words came out "
     "right.",
     "Ya hemos generado, mezclado y comparado. Hoy: decidir dónde va cada cosa en la imagen, y comprobar que las palabras salen bien."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    # 3
    ("For this one I'm using HaDeS MOSS v1.0 Turbo, my model with the Turbo baked in: eight steps, CFG one.",
     "Para este uso HaDeS MOSS v1.0 Turbo, mi modelo con el Turbo integrado: ocho pasos, CFG uno.",
     "For this one I'm using HaDeS MOSS version one Turbo, my model with the Turbo baked in: eight steps, CFG one."),
    ("The Prompt Enhancer in layout mode. Write the scene in one sentence, saying where each thing goes: the fisherman on the left, the "
     "sun in the upper right, the title at the top, the signature in the lower right.",
     "El Prompt Enhancer en modo layout. Escribe la escena en una frase, diciendo dónde va cada cosa: el pescador a la izquierda, el sol "
     "arriba a la derecha, el título arriba y la firma abajo a la derecha."),
    ("Give the exact words of any text in quotes. If you don't, it may invent them.",
     "Pon entre comillas las palabras exactas de cualquier texto. Si no, puede inventárselas."),
    # 6: DRAFT LAYOUT
    ("Connect the Layout Editor and press Draft Layout. Only the enhancer runs, no image: about three seconds, and the boxes appear on "
     "the canvas.",
     "Conecta el Layout Editor y pulsa Draft Layout. Solo se ejecuta el enhancer, sin imagen: unos tres segundos, y las cajas aparecen "
     "en el lienzo."),
    ("Orange boxes are characters, blue ones objects, purple ones text, each with a short description of what goes inside.",
     "Las cajas naranjas son personajes, las azules objetos y las moradas texto, cada una con una breve descripción de lo que va dentro."),
    ("Check them before generating. Over six enhancer seeds, the title, the fisherman and the signature were always in place, but the "
     "sun got its own box in only three.",
     "Revísalas antes de generar. Con seis semillas del enhancer, el título, el pescador y la firma siempre quedaron en su sitio, pero "
     "el sol solo tuvo caja propia en tres."),
    # 9: GENERATE
    ("The Layout Compiler turns the boxes into the prompt and a latent of the canvas size. Then the KSampler, eight steps, and Generate.",
     "El Layout Compiler convierte las cajas en el prompt y en un latente del tamaño del lienzo. Después el KSampler, ocho pasos, y "
     "Generate.",
     "The Layout Compiler turns the boxes into the prompt and a latent of the canvas size. Then the K Sampler, eight steps, and Generate."),
    ("Each thing lands in its box. With eight different seeds, the title, the sun and the signature were in their boxes every time.",
     "Cada cosa cae en su caja. Con ocho semillas distintas, el título, el sol y la firma quedaron en su caja todas las veces."),
    ("The fisherman was always on the left, but his net and legs went past his narrow box. A box is a strong hint, not a cage.",
     "El pescador siempre quedó a la izquierda, pero su red y sus piernas salieron de su caja estrecha. Una caja es una indicación "
     "fuerte, no una jaula."),
    # 12: editing the boxes
    ("Now move a box. I'll drag the sun down to the horizon, and generate again.",
     "Ahora mueve una caja. Arrastro el sol hasta el horizonte y genero otra vez."),
    ("The sun follows. With the same eight seeds, it was in its new box every time. Your edits are never replaced by a normal run.",
     "El sol la sigue. Con las mismas ocho semillas, quedó en su nueva caja todas las veces. Una ejecución normal nunca sustituye tus "
     "cambios."),
    # 14: Text Check
    ("And the words? MOSS Text Check reads each text box in the image, with MOSS's own text encoder. No extra model.",
     "¿Y las palabras? MOSS Text Check lee cada caja de texto en la imagen, con el propio text encoder de MOSS. Sin modelos extra."),
    ("For each text: exact, close, missing, or outside its box. Here both are exact, in all eight seeds, in about one second.",
     "Para cada texto: exact, close, missing u outside its box. Aquí los dos son exact, en las ocho semillas, en un segundo más o menos."),
    ("It's strict: one wrong letter is close, not exact. And failed mask marks the boxes that didn't pass, ready for a retry of just "
     "those areas.",
     "Es estricto: una letra mal es close, no exact. Y failed mask marca las cajas que no pasaron, listas para repetir solo esas zonas."),
    # 17: goodbye
    ("That's layout and text: say where, check the boxes, generate, and let Text Check read it back.",
     "Eso es layout y texto: di dónde, revisa las cajas, genera y deja que Text Check lo lea."),
    ("This workflow is free to download from the description.",
     "Este workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. "
     "It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y "
     "suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

L8_MODEL = "HaDeS_MOSS_v1.0_Turbo.safetensors"
L8_SENTENCE = ('A poster: an old fisherman with a grey beard mending a net on the left side, the sun rising over the sea in the upper right '
               'with seagulls, the title "NORTH SEA" at the top, and the signature "HaDeS" in the lower right corner.')
L8_DRAFT = Path("I:/HaDeS_MOSS/videos/layout_text/measure/drafts/draft_s2.json")  # the measured draft (enhancer seed 2)
L8_SUN_TO = [700, 516, 980, 766]  # the measured move: the sun down to the horizon (the editor moves boxes by 16 px)
L8_COLORS = {"object": "#6d8fd6", "character": "#c49a5a", "text": "#a786c4"}  # the Layout Editor's own colours
L8_POS = {"loader": (0, 0), "draft": (440, 0), "editor": (940, 0), "layout": (1760, 0), "enh2": (1760, 640), "sampler": (2400, 0),
          "saver": (2860, 0), "telemetry": (2400, 760), "check": (2860, 760), "check_text": (3300, 760)}
L8_BOX = [-260, -60, 3900, 1500]


async def l8_add(d, type_, query, key, values=None):
    if values:
        await d.js("""([type, v]) => { const g = window.__hv.app().graph, prev = g.onNodeAdded;
            g.onNodeAdded = function (n) { if (n.type === type) { for (const [k, x] of Object.entries(v)) { const w = n.widgets?.find(w => w.name === k);
                if (w) w.value = x; } g.onNodeAdded = prev; } return prev?.apply(this, arguments); }; }""", [type_, values])
    n = await basic_add(d, type_, query, L8_POS[key])
    # a large node (the Layout Compiler) can land away from its spot: put it there
    await d.js("([id, p]) => { const n = window.__hv.node(id); n.pos = [p[0], p[1]]; n.setDirtyCanvas(true, true); }", [n, list(L8_POS[key])])
    if values:
        await up_set(d, n, values)
    return n


async def l8_editor_button(d, editor, zone):
    z = await d.js("([id, k]) => window.__hv.zone(id, k)", [editor, zone])
    x, y = await d.screen(editor, z["x"] + z["w"] / 2, z["y"] + z["h"] / 2)
    await d.click(x, y)


async def l8_wait_queue(d):
    async with d.fast(2.0):
        await d.hold(1.0)
        for _ in range(2400):
            if await d.js("() => fetch('/queue').then(r => r.json()).then(q => q.queue_running.length + q.queue_pending.length)") == 0:
                break
            await asyncio.sleep(0.25)
        else:
            raise SystemExit("the run did not finish in 10 minutes")
        await d.hold(0.5)
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    if run["status"].get("status_str") != "success":
        raise SystemExit(f"the run failed: {run['status']}")
    return run


def l8_boxes_html(url, draft, ih=760):
    iw, s = ih * draft["width"] / draft["height"], ih / draft["height"]
    boxes = "".join(
        f'<div class="__hv_box" style="position:absolute;left:{it["bbox"][0] * s}px;top:{it["bbox"][1] * s}px;'
        f'width:{(it["bbox"][2] - it["bbox"][0]) * s}px;height:{(it["bbox"][3] - it["bbox"][1]) * s}px;'
        f'border:3px solid {L8_COLORS.get(it["label"], "#fff")};border-radius:4px;opacity:0;transition:opacity .5s ease;'
        f'box-shadow:0 0 0 1px rgba(0,0,0,.5)"></div>' for it in draft["items"])
    return (f'<div style="margin-top:40px;display:flex;justify-content:center"><div style="position:relative;width:{iw}px;height:{ih}px">'
            f'<img src="{url}" style="width:100%;height:100%;border-radius:6px;box-shadow:0 10px 50px rgba(0,0,0,.6)">{boxes}</div></div>')


async def layout_text(d: Director):
    """The eighth tutorial: a poster laid out from a sentence, a box moved, the text read back by MOSS Text Check."""
    await d.camera({"box": BASIC_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "Layout + text", 6.0)
    for i in range(3):
        await d.say(i)
        await d.move(W * (0.55 + 0.05 * (i % 2)), H * (0.5 + 0.04 * i), 2500)

    # 2. the model and the sentence
    await d.camera({"box": [-260, -60, 1700, 1000]}, 900, pad=60, top=60)
    await d.say(3)
    loader = await l8_add(d, "HaDeSMossLoader", "MOSS Loader", "loader", dict(BASIC_LOADER, unet_name=L8_MODEL))
    await d._after_voice(0.1)
    await d.say(4)
    draft = await l8_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", "draft",
                         {"mode": "layout", "width": 1024, "height": 1536, "seed": 2, "control_after_generate": "fixed"})
    await basic_link(d, loader, "CLIP", draft, "clip", 600)
    await d.camera([draft], 900, pad=100, top=60)
    await pe_type(d, draft, "idea", L8_SENTENCE)
    await d._after_voice(0.1)
    await d.say(5)
    await d._after_voice(0.3)

    # 3. DRAFT LAYOUT
    await d.camera({"box": [-260, -60, 1700, 1000]}, 900, pad=60, top=60)
    await d.say(6)
    editor = await l8_add(d, "HaDeSMossLayoutEditor", "MOSS Layout Editor", "editor")
    await d.js("""(id) => { const w = window.__hv.widget(id, 'draft');
        w.value = JSON.stringify({width: 1024, height: 1536, bbox_coord_space: 'abs', bbox_format: 'xyxy', summary: {canvas: 'single', caption: ''}, items: []}, null, 2);
        window.__hv.node(id).setDirtyCanvas(true, true); }""", editor)
    await basic_link(d, draft, "draft", editor, "draft_in", 700)
    await d.camera([draft, editor], 900, pad=40, top=60)
    before = await d.js("(id) => window.__hv.widget(id, 'draft').value", editor)
    await l8_editor_button(d, editor, "draft_run")
    async with d.fast():
        for _ in range(1500):
            if await d.js("""([id, before]) => { const n = window.__hv.node(id), v = window.__hv.widget(id, 'draft').value;
                    return v !== before && !n._hadesDraftRun?.pendingPrompt(); }""", [editor, before]):
                break
            await asyncio.sleep(0.2)
        else:
            raise SystemExit("the draft did not arrive in 5 minutes")
        await d.hold(0.4)
    got = json.loads(await d.js("(id) => window.__hv.widget(id, 'draft').value", editor))
    want = json.loads(L8_DRAFT.read_text(encoding="utf-8"))
    if [(i["label"], i["bbox"], i.get("text")) for i in got["items"]] != [(i["label"], i["bbox"], i.get("text")) for i in want["items"]]:
        print("draft on camera:", json.dumps(got["items"])[:1500], flush=True)
        raise SystemExit("the draft differs from the measured one (enhancer seed 2): the measurements would not hold")
    await d.camera([editor], 1100, pad=30, top=60)
    await d._after_voice(0.2)
    f = (await d.js("(id) => window.__hv.zone(id, 'canvas')", editor))["f"]
    at = lambda bx, by: d.screen(editor, f["x"] + bx * f["s"], f["y"] + by * f["s"])  # noqa: E731
    await d.say(7)
    for it in got["items"][:3]:
        b = it["bbox"]
        await d.move(*(await at((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
        await d.hold(1.0)
    await d._after_voice(0.1)
    await d.say(8)
    await d._after_voice(0.3)

    # 4. GENERATE
    await d.camera({"box": L8_BOX}, 1000, pad=50, top=60)
    await d.say(9)
    async with d.fast(5.0):
        layout = await l8_add(d, "HaDeSMossLayoutCompiler", "MOSS Layout Compiler", "layout")
        await basic_link(d, editor, "draft", layout, "draft", 600)
        enh2 = await l8_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", "enh2", {"mode": "as written", "seed": 1})
        await basic_link(d, loader, "CLIP", enh2, "clip", 600)
        await basic_link(d, layout, "prompt", enh2, "idea", 600)
        sampler = await l8_add(d, "HaDeSMossKSampler", "MOSS KSampler", "sampler",
                               {"seed": 1013, "control_after_generate": "fixed", "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
        for a, out, inp in ((loader, "MODEL", "model"), (enh2, "positive", "positive"), (enh2, "negative", "negative"),
                            (layout, "latent", "latent_image"), (loader, "VAE", "vae")):
            await basic_link(d, a, out, sampler, inp, 600)
        saver = await l8_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", "saver", {"filename_prefix": "MOSS/HaDeS_MOSS_layout_%seed%"})
        await basic_link(d, sampler, "image", saver, "images", 600)
        telemetry = await l8_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", "telemetry")
        for k, (n, out) in enumerate(((loader, "info"), (editor, "info"), (layout, "info"), (enh2, "info"), (sampler, "info"))):
            await basic_link(d, n, out, telemetry, f"info_{k + 1}", 600)
        await basic_link(d, telemetry, "json", saver, "metadata", 600)
    await d.camera([editor], 1000, pad=30, top=60)
    await d._after_voice(0.1)
    await l8_editor_button(d, editor, "generate")
    run = await l8_wait_queue(d)
    img = run["outputs"][str(saver)]["images"][0]
    url1 = f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
    print("render 1:", img["filename"], flush=True)
    await d.move(W - 6, H * 0.5)
    await d.js("(h) => window.__hv.show(h)", l8_boxes_html(url1, got))
    await d.say(10)
    await d.hold(1.2)
    await d.js("() => window.__hv.boxes(true)")
    await d._after_voice(0.2)
    await d.say(11)
    await d._after_voice(0.4)
    await d.js("() => window.__hv.boxes(false)")
    await pe_hide(d)

    # 5. move the sun
    await d.camera([editor], 1000, pad=30, top=60)
    await d.say(12)
    sun = next(i for i in got["items"] if i["caption"].lower().startswith("a large, bright sun"))
    b = sun["bbox"]
    # grab the sun below the seagulls' box (a click takes the smallest box under the pointer), move it by the measured offset
    gx, gy = (b[0] + b[2]) / 2, 300
    await d.click(*(await at(gx, gy)))
    await d.hold(0.4)
    await d.drag(*(await at(gx, gy)), *(await at(gx + L8_SUN_TO[0] - b[0], gy + L8_SUN_TO[1] - b[1])), 1400)
    await d.hold(0.6)
    moved = json.loads(await d.js("(id) => window.__hv.widget(id, 'draft').value", editor))
    want_moved = [dict(i, bbox=L8_SUN_TO) if i is sun else i for i in got["items"]]
    if [(i["label"], [round(v) for v in i["bbox"]]) for i in moved["items"]] != [(i["label"], i["bbox"]) for i in want_moved]:
        print("boxes after the drag:", [i["bbox"] for i in moved["items"]], flush=True)
        raise SystemExit("the drag did not give the measured sun box")
    await d._after_voice(0.1)
    await l8_editor_button(d, editor, "generate")
    run = await l8_wait_queue(d)
    img = run["outputs"][str(saver)]["images"][0]
    url2 = f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
    print("render 2:", img["filename"], flush=True)
    await d.move(W - 6, H * 0.5)
    await d.js("(h) => window.__hv.show(h)", l8_boxes_html(url2, moved))
    await d.say(13)
    await d.hold(1.0)
    await d.js("() => window.__hv.boxes(true)")
    await d._after_voice(0.4)
    await d.js("() => window.__hv.boxes(false)")
    await pe_hide(d)

    # 6. MOSS Text Check
    await d.camera({"box": [1900, 400, 3900, 1500]}, 1000, pad=50, top=60)
    await d.say(14)
    check = await l8_add(d, "HaDeSMossTextCheck", "MOSS Text Check", "check")
    async with d.fast(3.0):
        await d.camera({"box": L8_BOX}, 600, pad=50, top=60)
        await basic_link(d, sampler, "image", check, "image", 600)
        await basic_link(d, loader, "CLIP", check, "clip", 600)
        await basic_link(d, editor, "draft", check, "layout", 600)
        text = await l8_add(d, "PreviewAny", "Preview as Text", "check_text")
        await d.js("([id, s]) => { const n = window.__hv.node(id); n.setSize(s); n.setDirtyCanvas(true, true); }", [text, [520, 300]])
        await basic_link(d, check, "report", text, "source", 600)
        await d.camera({"box": [1900, 400, 3900, 1500]}, 600, pad=50, top=60)
    await d._after_voice(0.1)
    await lt_run(d)
    rep = (await ms_last_secs(d), next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))["outputs"])
    print("Text Check run:", f"{rep[0]:.1f} s", json.dumps(rep[1].get(str(text)))[:400], flush=True)
    await d.camera([check, text], 1000, pad=40, top=60)
    await d.say(15)
    await d._after_voice(0.4)
    await d.say(16)
    await d.js("(h) => window.__hv.show(h)", ms_card(
        '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">expected_text: NORTH SEE</div>'
        '<div style="font:700 44px Consolas,monospace;color:#ffcc33;margin-top:18px">[close] \'NORTH SEE\' \u2192 read \'NORTH SEA\' (0.89)</div>'
        '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1;margin-top:40px">failed_mask: the boxes that did not pass</div>', 380))
    await d._after_voice(0.4)
    await pe_hide(d)

    # 7. goodbye
    await d.camera({"box": L8_BOX}, 1300, pad=50, top=60)
    await d.move(W - 6, H * 0.5)
    await d.say(17)
    await d._after_voice(0.1)
    await d.say(18)
    await d.js("(h) => window.__hv.show(h)", '<div style="color:#fff;text-align:center;margin-top:300px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Layout from a Sentence.json</div></div>')
    await d._after_voice(0.3)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:40px"><img src="{url1}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Layout + text</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(19)
    await d.say(20)
    await d.end(1.5)


async def l8_prepare(page):
    """Off camera: one short render loads HaDeS MOSS v1.0 Turbo, the text encoder and the VAE (no Saver: the model is hashed on camera)."""
    g = {"1": {"class_type": "HaDeSMossLoader", "inputs": dict(BASIC_LOADER, unet_name=L8_MODEL, weight_dtype="default", clip_device="default")},
         "2": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {"clip": ["1", 1], "idea": "a lighthouse", "mode": "as written", "width": 1024,
               "height": 1024, "temperature": 0.7, "seed": 0, "max_tokens": 1024, "negative": ""}},
         "3": {"class_type": "HaDeSMossLatent", "inputs": {"aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024,
               "batch_size": 1, "seed": 7}},
         "4": {"class_type": "HaDeSMossKSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["2", 1], "latent_image": ["3", 0],
               "seed": 7, "steps": 2, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "flow_shift": 0.0, "slg": "off",
               "vae": ["1", 2]}},
         "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 1]}}}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


layout_text.prepare = l8_prepare


# ---- controlnet (ControlNet: Depth & Color from scratch) ----

CN_LINES = [
    ("Hi, I'm HaDeS, and welcome back to the channel.",
     "Hola, soy HaDeS, y bienvenido de nuevo al canal."),
    ("Today we're building a MOSS-Image ControlNet workflow from an empty canvas, one node at a time.",
     "Hoy vamos a montar un workflow de ControlNet para MOSS-Image desde un lienzo vacío, nodo a nodo."),
    ("We'll take an existing image, extract its depth map, and turn our fisherman into a medieval knight while keeping the exact same pose.",
     "Tomaremos una imagen existente, extraeremos su mapa de profundidad y convertiremos a nuestro pescador en un caballero medieval manteniendo exactamente la misma pose."),
    ("I'll also show you how the color control modes work, and a mistake that's very easy to make if you plug a photo straight in.",
     "También te enseñaré cómo funcionan los modos de control de color, y un error muy fácil de cometer si conectas una foto directamente."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    ("Let's start from zero with MOSS Loader.",
     "Empezamos desde cero con MOSS Loader."),
    ("I select MOSS-Image Base, the int8 text encoder and the Flux 2 VAE. The color and depth adapter was trained on Base, though I've also tested it on SFT and it works cleanly on both.",
     "Elijo MOSS-Image Base, el text encoder int8 y el VAE de Flux 2. El adaptador de color y profundidad se entrenó sobre Base, aunque también lo he medido en SFT y funciona limpio en ambos.",
     "I select MOSS-Image Base, the int eight text encoder and the Flux two VAE. The color and depth adapter was trained on Base, though I've also tested it on SFT and it works cleanly on both."),
    ("Next, I add MOSS Multi-LoRA and load the moss-color-depth-control adapter.",
     "Después añado MOSS Multi-LoRA y cargo el adaptador moss-color-depth-control.",
     "Next, I add MOSS Multi-LoRA and load the moss color depth control adapter."),
    ("Important detail: keep its strength at one here. ComfyUI replaces the model's input layer at full strength anyway, so the real strength slider is on the MOSS Control node we'll add in a moment.",
     "Detalle importante: deja su fuerza en uno aquí. ComfyUI sustituye la capa de entrada del modelo a fuerza completa de todas formas, así que el control real de fuerza está en el nodo MOSS Control que añadiremos ahora."),
    ("Now we need a reference image. I'll use our fisherman from the first tutorial in Load Image.",
     "Ahora necesitamos una imagen de referencia. Usaré nuestro pescador del primer tutorial en Load Image."),
    ("To get the depth map, I add Depth Anything V2 from controlnet aux, and connect a Preview Image so we can see it.",
     "Para obtener el mapa de profundidad, añado Depth Anything V2 de controlnet aux y conecto un Preview Image para poder verlo.",
     "To get the depth map, I add Depth Anything V two from controlnet ox, and connect a Preview Image so we can see it."),
    ("In MOSS Control, near is white and far is black. Depth Anything gives us that directly.",
     "En MOSS Control, lo cercano es blanco y lo lejano es negro. Depth Anything nos lo da directamente así."),
    ("Next, the MOSS Prompt Enhancer, connected to CLIP from the loader.",
     "Lo siguiente, el MOSS Prompt Enhancer, conectado al CLIP del loader."),
    ("I'll leave it on \"as written\" and describe a medieval knight in weathered steel armor mending a chainmail cloak on a castle parapet at sunset.",
     "Lo dejo en \"as written\" y describo un caballero medieval con armadura de acero desgastada remendando una capa de cota de malla en la almena de un castillo al atardecer."),
    ("Tip: if you want to recreate a photo without writing the prompt yourself, you can switch this node to \"caption\" mode and plug your image into its image input.",
     "Consejo: si quieres recrear una foto sin escribir tú el prompt, puedes cambiar este nodo al modo \"caption\" y conectar tu imagen a su entrada image."),
    ("Now I add MOSS Latent, set to 1024 by 1024, with a fixed seed so we can compare every setting cleanly.",
     "Ahora añado MOSS Latent, fijado en 1024 por 1024, con una semilla fija para poder comparar cada ajuste con limpieza.",
     "Now I add MOSS Latent, set to ten twenty-four by ten twenty-four, with a fixed seed so we can compare every setting cleanly."),
    ("Here is the star of today's workflow: MOSS Control.",
     "Aquí está el protagonista del workflow de hoy: MOSS Control."),
    ("It takes the positive and negative conditioning from the Prompt Enhancer, the VAE from the loader, and the latent from MOSS Latent.",
     "Recibe los condicionamientos positivo y negativo del Prompt Enhancer, el VAE del loader y el latente de MOSS Latent."),
    ("It resizes your maps in pixels to match the latent before encoding them with the VAE.",
     "Redimensiona tus mapas en píxeles para que encajen con el latente antes de codificarlos con el VAE."),
    ("I plug our depth map into depth map, and I also connect its maps output to a Preview Image. That way we always see the exact maps going into the model.",
     "Conecto nuestro mapa de profundidad a depth map, y también conecto su salida maps a un Preview Image. Así vemos siempre los mapas exactos que entran al modelo."),
    ("Let's finish the pipeline. I add MOSS KSampler and plug in my Turbo 8-steps v1.0 node.",
     "Vamos a terminar el flujo. Añado MOSS KSampler y le conecto mi nodo Turbo 8-steps v1.0.",
     "Let's finish the pipeline. I add MOSS KSampler and plug in my Turbo eight steps version one node."),
    ("On Base, twenty-five steps take about fourteen seconds. With Turbo v1.0, eight steps at CFG one take just over three seconds, over four times faster.",
     "En Base, veinticinco pasos tardan unos catorce segundos. Con Turbo v1.0, ocho pasos con CFG uno tardan poco más de tres segundos, más de cuatro veces más rápido.",
     "On Base, twenty-five steps take about fourteen seconds. With Turbo version one, eight steps at CFG one take just over three seconds, over four times faster."),
    ("I also add Suite Telemetry, connect the info outputs from every node in the Suite, add a Preview Any to check the report, and wire its json into MOSS Image Saver.",
     "Añado también Suite Telemetry, conecto las salidas info de todos los nodos de la Suite, pongo un Preview Any para ver el informe y llevo su json al MOSS Image Saver."),
    ("First run, with color mode set to \"off\", depth only. Let's launch it.",
     "Primera ejecución, con el modo de color en \"off\", solo profundidad. Vamos a lanzarlo."),
    ("Three point three seconds, and look at that: the hands, the tilt of the head, the net turned into chainmail, and the sea in the background all match the fisherman's depth map.",
     "Tres coma tres segundos, y mira esto: las manos, la inclinación de la cabeza, la red convertida en cota de malla y el mar al fondo siguen todos el mapa de profundidad del pescador."),
    ("Because color mode is off, all the colors, lighting and textures come from our prompt.",
     "Como el modo de color está desactivado, todos los colores, la iluminación y las texturas vienen de nuestro prompt."),
    ("Now let's plug the original photo into color map and see what happens with the other two modes.",
     "Ahora vamos a conectar la foto original a color map y ver qué pasa con los otros dos modos."),
    ("If you leave color mode on \"as given\" with a photo, watch closely: the adapter copies the photo's actual texture.",
     "Si dejas el modo de color en \"as given\" con una foto, fíjate bien: el adaptador copia la textura real de la foto."),
    ("It literally opened the helmet visor and put the fisherman's face and beard inside! That's because \"as given\" expects a flat segmentation map, not a detailed photograph.",
     "¡Literalmente ha abierto el visor del casco y ha metido la cara y la barba del pescador dentro! Eso pasa porque \"as given\" espera un mapa de segmentación plano, no una fotografía con detalle."),
    ("That's why I added \"flat regions\" to MOSS Control. It blurs the photo and reduces it to a few solid color blocks before the VAE.",
     "Por eso añadí \"flat regions\" a MOSS Control. Desenfoca la foto y la reduce a unos pocos bloques de color sólido antes del VAE."),
    ("Now it guides the warm light on the left and the darker tones on the right, without copying the fisherman's face.",
     "Ahora guía la luz cálida a la izquierda y los tonos más oscuros a la derecha, sin copiar la cara del pescador."),
    ("Two quick controls when you want more freedom: lowering strength below one softens how strictly it follows the map, and lowering end percent, say to zero point seven five, lets the final steps refine details more freely.",
     "Dos controles rápidos cuando quieres más libertad: bajar strength por debajo de uno suaviza el seguimiento del mapa, y bajar end percent, por ejemplo a cero coma setenta y cinco, deja que los últimos pasos afinen los detalles con más libertad."),
    ("And if we check Suite Telemetry, it says \"No findings\". Everything in our workflow matches what's been measured.",
     "Y si miramos Suite Telemetry, dice \"No findings\". Todo en nuestro workflow cuadra con lo que está medido."),
    ("That's it: depth and color ControlNet in MOSS-Image, built from scratch in just over three seconds per image.",
     "Y ya está: ControlNet de profundidad y color en MOSS-Image, montado desde cero en poco más de tres segundos por imagen."),
    ("This exact workflow is free to download from the description.",
     "Este mismo workflow se puede descargar gratis desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

CN_CTRL_LORA = "MOSS\\moss-color-depth-control.safetensors"
CN_PROMPT = ("Oil painting of a medieval knight in weathered steel armor mending a chainmail cloak "
             "on a stone castle parapet at sunset, warm torches, sea in the background.")
CN_POS = {
    "load_img": (0, 0),
    "depth": (440, 0),
    "prev_depth": (440, 200),
    "enhancer": (900, 0),
    "ctrl": (1400, 0),
    "prev_maps": (1400, 440),
    "sampler": (1860, 0),
    "saver": (2320, 0),
    "loader": (0, 560),
    "loras": (440, 560),
    "latent": (900, 680),
    "turbo": (1860, 580),
    "telemetry": (2320, 580),
    "report": (2760, 580),
}
CN_BOX = [-60, -60, 3260, 1220]


def cn_cols_html(items, side=560):
    head = "".join(f'<div style="color:#fff;font:600 24px Segoe UI,sans-serif;text-align:center">{t}</div>' for _, t in items)
    cells = "".join(f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:8px;border:3px solid #333">'
                    f'<img src="{u}" style="width:{side}px;height:{side}px;object-fit:cover"></div>' for u, _ in items)
    return f'<div style="display:grid;grid-template-columns:repeat({len(items)},{side}px);gap:14px 24px">{head}{cells}</div>'


async def cn_prepare(page):
    """Off camera: upload fisherman_video1.png into the recording input folder, refresh node definitions, and run one full warmup
    (including DepthAnythingV2, Control LoRA, Turbo and Image Saver hashes) so the first on-camera run is fast and honest."""
    import urllib.request
    import uuid
    url = page.url.split("/", 3)
    base = f"{url[0]}//{url[2]}"
    b = "----" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"fisherman_video1.png\"\r\nContent-Type: image/png\r\n\r\n").encode() \
        + UP_VIDEO1.read_bytes() + f"\r\n--{b}--\r\n".encode()
    urllib.request.urlopen(urllib.request.Request(base + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))
    await page.evaluate("() => window.__hv.app().extensionManager.command.execute('Comfy.RefreshNodeDefinitions')")
    await asyncio.sleep(0.5)
    g = {
        "1": {"class_type": "HaDeSMossLoader", "inputs": dict(BASIC_LOADER, weight_dtype="default", clip_device="default")},
        "2": {"class_type": "LoadImage", "inputs": {"image": "fisherman_video1.png"}},
        "3": {"class_type": "HaDeSMossMultiLora", "inputs": {
            "model": ["1", 0], "enabled": True, "loras": json.dumps([{"name": CN_CTRL_LORA, "on": True, "strength": 1.0}])}},
        "4": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {
            "clip": ["1", 1], "idea": "a knight", "mode": "as written", "width": 1024, "height": 1024,
            "temperature": 0.7, "seed": 1, "max_tokens": 1024, "negative": "worst quality, low quality"}},
        "5": {"class_type": "HaDeSMossLatent", "inputs": {
            "aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1, "seed": 7}},
        "6": {"class_type": "DepthAnythingV2Preprocessor", "inputs": {
            "image": ["2", 0], "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 1024}},
        "7": {"class_type": "HaDeSMossControl", "inputs": {
            "positive": ["4", 0], "negative": ["4", 1], "vae": ["1", 2], "latent": ["5", 0], "depth_map": ["6", 0],
            "color_mode": "off (depth only)", "colors": 8, "blur": 4.0, "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
        "8": {"class_type": "HaDeSMossTurbo", "inputs": {
            "lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}},
        "9": {"class_type": "HaDeSMossKSampler", "inputs": {
            "model": ["3", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["5", 0], "vae": ["1", 2],
            "turbo": ["8", 0], "seed": 7, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0, "flow_shift": 0.0, "slg": "off"}},
        "10": {"class_type": "HaDeSMossSuiteTelemetry", "inputs": {
            "report_title": "Warmup", "info_1": ["1", 3], "info_2": ["3", 1], "info_3": ["4", 6], "info_4": ["5", 3],
            "info_5": ["7", 3], "info_6": ["9", 2]}},
        "11": {"class_type": "HaDeSMossImageSaver", "inputs": {
            "images": ["9", 1], "metadata": ["10", 1], "filename_prefix": "MOSS/warmup_cn", "hashes": True, "embed_workflow": False}},
    }
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


async def controlnet_video(d: Director):
    """ControlNet: Depth & Color from scratch, built node by node from an empty canvas."""
    P = CN_POS
    await d.camera({"box": CN_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "ControlNet: Depth & Color from scratch", 7.0)
    await d.say(0)
    await d.say(1)
    first_url = "data:image/png;base64," + base64.b64encode(UP_VIDEO1.read_bytes()).decode()
    knight_preview = Path(r"I:\HaDeS_MOSS\videos\comfy_output\ctrl_measure\depth_only_end1.0_1013_00001_.png")
    if knight_preview.exists():
        k_url = "data:image/png;base64," + base64.b64encode(knight_preview.read_bytes()).decode()
        await d.js("(h) => window.__hv.show(h)", cn_cols_html([(first_url, "Input image"), (k_url, "Depth ControlNet · 8 steps · 3.3 s")], side=560))
    await d.say(2)
    await d.js("() => window.__hv.hide()")
    await d.say(3)
    await d.say(4)

    # 2. loader and Multi-LoRA with control adapter
    await d.say(5)
    loader = await basic_add(d, "HaDeSMossLoader", "MOSS Loader", P["loader"])
    await up_set(d, loader, BASIC_LOADER)
    await d.camera([loader], 900, pad=180, top=60)
    await d.say(6)
    await basic_point(d, loader, "unet_name")
    await d.hold(0.4)
    await basic_point(d, loader, "clip_name")
    await d.hold(0.4)
    await basic_point(d, loader, "vae_name")

    await d.camera({"box": [-40, 420, 920, 960]}, 900, pad=80, top=60)
    await d.say(7)
    loras = await basic_add(d, "HaDeSMossMultiLora", "MOSS Multi-LoRA", P["loras"])
    await up_set(d, loras, {"enabled": True, "loras": json.dumps([{"name": CN_CTRL_LORA, "on": True, "strength": 1.0}])})
    await basic_link(d, loader, "MODEL", loras, "model", 800)
    await d.camera([loras], 900, pad=220, top=60)
    await d.say(8)

    # 3. input image + Depth Anything V2 + PreviewImage
    await d.camera({"box": [-60, -60, 1000, 900]}, 900, pad=60, top=60)
    await d.say(9)
    load_img = await basic_add(d, "LoadImage", "Load Image", P["load_img"])
    await up_set(d, load_img, {"image": "fisherman_video1.png"})
    await d.camera([load_img], 900, pad=160, top=60)
    await d.hold(0.4)

    await d.camera({"box": [-40, -40, 920, 560]}, 900, pad=80, top=60)
    await d.say(10)
    depth = await basic_add(d, "DepthAnythingV2Preprocessor", "Depth Anything V2", P["depth"])
    await up_set(d, depth, {"ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 1024})
    await basic_link(d, load_img, "IMAGE", depth, "image", 800)
    prev_depth = await basic_add(d, "PreviewImage", "Preview Image", P["prev_depth"])
    await d.js("(id) => { const n = window.__hv.node(id); n.setSize([340, 260]); n.setDirtyCanvas(true, true); }", prev_depth)
    await basic_link(d, depth, "IMAGE", prev_depth, "images", 700)
    await d.camera([depth], 900, pad=200, top=60)
    await d.say(11)

    # 4. Prompt Enhancer + MOSS Latent
    await d.camera({"box": [-40, -40, 1400, 900]}, 900, pad=80, top=60)
    await d.say(12)
    enhancer = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enhancer"])
    await basic_link(d, loader, "CLIP", enhancer, "clip", 800)
    await basic_point(d, enhancer, "mode", {"mode": "as written"})
    await d.camera([enhancer], 900, pad=160, top=60)
    box = await d.js("""(id) => { const w = window.__hv.widget(id, 'idea'); w.value = ''; if (w.element) w.element.value = '';
        const r = w.element.getBoundingClientRect(); return [r.left, r.top]; }""", enhancer)
    await d.click(box[0] + 40, box[1] + 30)
    if not await d.js("() => document.activeElement?.tagName === 'TEXTAREA'"):
        raise SystemExit("the prompt box is not focused; nothing typed")
    async with d.fast(2.0):
        await d.page.keyboard.type(CN_PROMPT, delay=10)
    await d.say(13)
    await d.say(14)
    await basic_point(d, enhancer, "mode")

    await d.camera({"box": [840, -40, 1380, 1080]}, 900, pad=80, top=60)
    await d.say(15)
    latent = await basic_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})
    await basic_point(d, latent, "seed")

    # 5. MOSS Control
    await d.camera({"box": [-60, -60, 1860, 1060]}, 900, pad=60, top=60)
    await d.say(16)
    ctrl = await basic_add(d, "HaDeSMossControl", "MOSS Control", P["ctrl"])
    await up_set(d, ctrl, {"color_mode": "off (depth only)", "colors": 8, "blur": 4.0, "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0})
    await d.say(17)
    async with d.fast(4.0):
        await basic_link(d, enhancer, "positive", ctrl, "positive", 700)
        await basic_link(d, enhancer, "negative", ctrl, "negative", 700)
        await basic_link(d, loader, "VAE", ctrl, "vae", 700)
        await basic_link(d, latent, "latent", ctrl, "latent", 700)
    await d.say(18)
    await d.say(19)
    await basic_link(d, depth, "IMAGE", ctrl, "depth_map", 800)
    prev_maps = await basic_add(d, "PreviewImage", "Preview Image", P["prev_maps"])
    await d.js("(id) => { const n = window.__hv.node(id); n.setSize([380, 320]); n.setDirtyCanvas(true, true); }", prev_maps)
    await basic_link(d, ctrl, "maps", prev_maps, "images", 700)

    # 6. KSampler, Turbo, Telemetry, Image Saver
    await d.camera({"box": CN_BOX}, 900, pad=60, top=60)
    await d.say(20)
    sampler = await basic_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["sampler"])
    turbo = await basic_add(d, "HaDeSMossTurbo", "MOSS Turbo LoRA", P["turbo"])
    await up_set(d, turbo, {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
    async with d.fast(5.0):
        for a, out, inp in ((loras, "MODEL", "model"), (ctrl, "positive", "positive"), (ctrl, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (loader, "VAE", "vae"), (turbo, "turbo", "turbo")):
            await basic_link(d, a, out, sampler, inp, 650)
    await d.say(21)
    await d.say(22)
    async with d.fast(5.0):
        saver = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["saver"])
        await basic_link(d, sampler, "image", saver, "images", 700)
        telemetry = await basic_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["telemetry"])
        for k, (n, out) in enumerate(((loader, "info"), (loras, "info"), (enhancer, "info"), (latent, "info"), (ctrl, "info"), (sampler, "info"))):
            await basic_link(d, n, out, telemetry, f"info_{k + 1}", 600)
        report = await basic_add(d, "PreviewAny", "Preview as Text", P["report"])
        await d.js("(id) => { const n = window.__hv.node(id); n.setSize([440, 520]); n.setDirtyCanvas(true, true); }", report)
        await basic_link(d, telemetry, "report", report, "source", 700)
        await basic_link(d, telemetry, "json", saver, "metadata", 700)

    # 7. First run: off (depth only)
    await d.camera({"box": CN_BOX}, 1000, pad=60, top=60)
    await d.say(23)
    url_off, t_off = await basic_run(d, saver)
    await d.camera([saver, prev_depth], 900, pad=120, top=60)
    depth_url = await d.js("""(id) => { const n = window.__hv.node(id); const im = n.imgs?.[0]; return im ? im.src : ''; }""", prev_depth)
    await d.say(24)
    if depth_url:
        await d.js("(h) => window.__hv.show(h)", cn_cols_html([(first_url, "Input"), (depth_url, "Depth map"), (url_off, "off (depth only)")], side=480))
    await d.say(25)
    await d.js("() => window.__hv.hide()")

    # 8. color_mode: as given vs flat regions
    await d.camera([load_img, ctrl, prev_maps], 900, pad=80, top=60)
    await d.say(26)
    await basic_link(d, load_img, "IMAGE", ctrl, "color_map", 800)
    await d.camera([ctrl], 900, pad=160, top=60)
    await d.say(27)
    await basic_point(d, ctrl, "color_mode", {"color_mode": "as given"})
    url_as, t_as = await basic_run(d, saver)
    await d.camera([saver], 900, pad=120, top=60)
    await d.js("(h) => window.__hv.show(h)", cn_cols_html([(first_url, "Input photo"), (url_as, "as given (copies texture)")], side=560))
    await d.say(28)
    await d.js("() => window.__hv.hide()")

    await d.camera([ctrl, prev_maps], 900, pad=120, top=60)
    await d.say(29)
    await basic_point(d, ctrl, "color_mode", {"color_mode": "flat regions"})
    url_flat, t_flat = await basic_run(d, saver)
    await d.camera([prev_maps, saver], 900, pad=100, top=60)
    await d.say(30)
    await d.js("(h) => window.__hv.show(h)", cn_cols_html([(url_off, "off (depth only)"), (url_flat, "flat regions"), (url_as, "as given")], side=480))
    await d.hold(1.5)
    await d.js("() => window.__hv.hide()")

    # 9. strength, end_percent and Telemetry
    await d.camera([ctrl], 900, pad=180, top=60)
    await d.say(31)
    await basic_point(d, ctrl, "strength")
    await d.hold(0.5)
    await basic_point(d, ctrl, "end_percent")
    await d.camera([telemetry, report], 900, pad=80, top=60)
    await d.say(32)
    await d.hold(1.0)

    # 10. closing
    await d.camera({"box": CN_BOX}, 1200, pad=60, top=60)
    await d.say(33)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1200px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Control (color, depth).json</div></div>')
    await d.say(34)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:40px"><img src="{url_off}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">ControlNet: Depth &amp; Color from scratch</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(35)
    await d.say(36)
    await d.end(1.5)


controlnet_video.prepare = cn_prepare


# ---- img2img (Real Img2Img: exact vs core denoise) ----

I2I_LINES = [
    ("Hi, I'm HaDeS, and welcome to tutorial number ten.",
     "Hola, soy HaDeS, y bienvenido al tutorial número diez."),
    ("Today we're building a brand-new Img2Img workflow from an empty canvas, and solving a problem that tricks almost everyone with FLOW models in ComfyUI.",
     "Hoy vamos a montar un workflow nuevo de Img2Img desde un lienzo vacío, y a resolver un problema que engaña a casi todo el mundo con los modelos FLOW en ComfyUI."),
    ("If you've ever run Img2Img at eight steps with denoise zero point nine and watched ComfyUI ignore your input image completely, this is why, and how MOSS KSampler fixes it.",
     "Si alguna vez lanzaste Img2Img a ocho pasos con denoise cero coma nueve y viste que ComfyUI ignoraba por completo tu imagen de entrada, aquí tienes por qué ocurre y cómo lo arregla MOSS KSampler."),
    ("As always, everything is free, and the new workflow is in the description.",
     "Como siempre, todo es gratis y tienes el nuevo workflow en la descripción."),
    ("Let's build the workflow from scratch. First, MOSS Loader with MOSS-Image Base, the int8 text encoder and the Flux 2 VAE.",
     "Montemos el workflow desde cero. Primero, MOSS Loader con MOSS-Image Base, el text encoder int8 y el VAE de Flux 2.",
     "Let's build the workflow from scratch. First, MOSS Loader with MOSS-Image Base, the int eight text encoder and the Flux two VAE."),
    ("Next, Load Image with our fisherman from the first tutorial, and a VAE Encode node to turn that photo into a latent.",
     "Después, Load Image con nuestro pescador del primer tutorial, y un nodo VAE Encode para convertir esa foto en un latente."),
    ("Notice we don't need MOSS Latent here: the latent comes straight from our input image, at 1024 by 1024.",
     "Fíjate en que aquí no necesitamos MOSS Latent: el latente sale directamente de nuestra imagen de entrada, a 1024 por 1024.",
     "Notice we don't need MOSS Latent here: the latent comes straight from our input image, at ten twenty-four by ten twenty-four."),
    ("Now the MOSS Prompt Enhancer on \"as written\", connected to CLIP. I'll ask for an impressionist oil painting of our fisherman at stormy night with a glowing lantern.",
     "Ahora el MOSS Prompt Enhancer en \"as written\", conectado a CLIP. Le pediré un óleo impresionista de nuestro pescador en una noche de tormenta con un farol encendido."),
    ("Next, MOSS KSampler with my Turbo 8-steps v1.0 node plugged in: eight steps, CFG one, euler and simple.",
     "Lo siguiente, MOSS KSampler con mi nodo Turbo 8-steps v1.0 conectado: ocho pasos, CFG uno, euler y simple.",
     "Next, MOSS KSampler with my Turbo eight steps version one node plugged in: eight steps, CFG one, euler and simple."),
    ("And at the end, Suite Telemetry with a Preview Any for the report, feeding the MOSS Image Saver.",
     "Y al final, Suite Telemetry con un Preview Any para el informe, alimentando al MOSS Image Saver."),
    ("Now let's test denoise on the stock setting: \"steps / denoise (core)\".",
     "Ahora probemos el denoise con el ajuste estándar: \"steps / denoise (core)\"."),
    ("If you start at denoise zero point five, look at what happens in about three seconds.",
     "Si empiezas con denoise cero coma cinco, mira lo que pasa en unos tres segundos."),
    ("Almost nothing changed! In a FLOW model like MOSS, noise is concentrated near the top of the schedule. Denoise zero point five starts at sigma zero point eight three, which only tweaks fine textures.",
     "¡Casi no ha cambiado nada! En un modelo FLOW como MOSS, el ruido se concentra arriba de la escala. Denoise cero coma cinco empieza en sigma cero coma ocho tres, que solo retoca texturas finas."),
    ("So naturally you push denoise up to zero point eight. Let's run that.",
     "Así que, lógicamente, subes el denoise a cero coma ocho. Vamos a lanzarlo."),
    ("At zero point eight, start sigma reaches zero point nine five. Now we get the glowing lantern and storm clouds behind him while keeping our fisherman.",
     "En cero coma ocho, el sigma inicial llega a cero coma nueve cinco. Ahora ya tenemos el farol encendido y las nubes de tormenta detrás manteniendo a nuestro pescador."),
    ("Now watch what happens when you want just a bit more change and set denoise to zero point nine.",
     "Ahora mira qué pasa cuando quieres un poco más de cambio y pones el denoise en cero coma nueve."),
    ("Wait a second — it completely replaced the scene! The fisherman is tiny and sitting sideways on a different boat.",
     "¡Un segundo — ha reemplazado la escena por completo! El pescador es diminuto y está sentado de lado en otra barca."),
    ("Why? Look at Suite Telemetry: start sigma is one point zero zero zero!",
     "¿Por qué? Mira Suite Telemetry: ¡el sigma inicial es uno coma cero cero cero!"),
    ("ComfyUI's stock schedule divides steps by denoise and rounds to an integer: eight divided by zero point nine rounds to eight steps, which means pure noise. Your input image was thrown away.",
     "El planificador estándar de ComfyUI divide los pasos entre el denoise y redondea a entero: ocho entre cero coma nueve redondea a ocho pasos, lo que significa ruido puro. Tu imagen de entrada se ha tirado a la basura."),
    ("At four steps it's even worse: zero point seven and zero point eight produce the exact same image, and zero point nine is pure noise too.",
     "A cuatro pasos es aún peor: cero coma siete y cero coma ocho dan exactamente la misma imagen, y cero coma nueve también es ruido puro."),
    ("That's why I added \"denoise mode: exact\" to MOSS KSampler.",
     "Por eso añadí \"denoise mode: exact\" al MOSS KSampler."),
    ("Instead of rounding a step count, \"exact\" treats denoise as a real point on the model's noise-time axis and runs all eight steps from there.",
     "En lugar de redondear un número de pasos, \"exact\" trata el denoise como un punto real en el eje de ruido del modelo y ejecuta los ocho pasos desde ahí."),
    ("Let's run the exact same denoise zero point nine with \"exact\" selected.",
     "Lancemos exactamente el mismo denoise cero coma nueve con \"exact\" seleccionado."),
    ("Look at Telemetry now: start sigma is zero point nine seven eight, not one.",
     "Mira Telemetry ahora: el sigma inicial es cero coma nueve siete ocho, no uno."),
    ("And look at the image: in three point two seconds, it reimagined the scene with dramatic night lighting while keeping our fisherman centered on the dock.",
     "Y mira la imagen: en tres coma dos segundos, ha reimaginado la escena con iluminación nocturna dramática manteniendo a nuestro pescador centrado en el muelle."),
    ("Up to denoise zero point five, both modes are identical. Above zero point five, \"exact\" gives you smooth, predictable control without ever throwing away your input image.",
     "Hasta denoise cero coma cinco, ambos modos son idénticos. Por encima de cero coma cinco, \"exact\" te da un control suave y predecible sin tirar nunca tu imagen de entrada."),
    ("You can download this brand-new Img2Img workflow for free from the description.",
     "Puedes descargar gratis este nuevo workflow de Img2Img desde la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

I2I_PROMPT = ("Impressionist oil painting of an old fisherman mending a net on a wooden dock at stormy night, "
              "glowing lantern light, thick impasto brushstrokes, dramatic dark clouds.")
I2I_POS = {
    "load_img": (0, 0),
    "vae_enc": (440, 0),
    "enhancer": (900, 0),
    "sampler": (1400, 0),
    "saver": (1860, 0),
    "loader": (0, 500),
    "turbo": (1400, 540),
    "telemetry": (1860, 520),
    "report": (2300, 520),
}
I2I_BOX = [-60, -60, 2800, 1140]


async def i2i_prepare(page):
    """Off camera: upload fisherman_video1.png into the recording input folder, refresh node definitions, and run one full warmup
    through VAEEncode, KSampler (Turbo v1.0) and Image Saver (hashes=True)."""
    import urllib.request
    import uuid
    url = page.url.split("/", 3)
    base = f"{url[0]}//{url[2]}"
    b = "----" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"fisherman_video1.png\"\r\nContent-Type: image/png\r\n\r\n").encode() \
        + UP_VIDEO1.read_bytes() + f"\r\n--{b}--\r\n".encode()
    urllib.request.urlopen(urllib.request.Request(base + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))
    await page.evaluate("() => window.__hv.app().extensionManager.command.execute('Comfy.RefreshNodeDefinitions')")
    await asyncio.sleep(0.5)
    g = {
        "1": {"class_type": "HaDeSMossLoader", "inputs": dict(BASIC_LOADER, weight_dtype="default", clip_device="default")},
        "2": {"class_type": "LoadImage", "inputs": {"image": "fisherman_video1.png"}},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {
            "clip": ["1", 1], "idea": "a fisherman", "mode": "as written", "width": 1024, "height": 1024,
            "temperature": 0.7, "seed": 1, "max_tokens": 1024, "negative": "worst quality, low quality"}},
        "5": {"class_type": "HaDeSMossTurbo", "inputs": {
            "lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}},
        "6": {"class_type": "HaDeSMossKSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["3", 0], "vae": ["1", 2],
            "turbo": ["5", 0], "seed": 7, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 0.5, "flow_shift": 0.0, "slg": "off", "denoise_mode": "steps / denoise (core)"}},
        "7": {"class_type": "HaDeSMossSuiteTelemetry", "inputs": {
            "report_title": "Warmup", "info_1": ["1", 3], "info_2": ["4", 6], "info_3": ["6", 2]}},
        "8": {"class_type": "HaDeSMossImageSaver", "inputs": {
            "images": ["6", 1], "metadata": ["7", 1], "filename_prefix": "MOSS/warmup_i2i", "hashes": True, "embed_workflow": False}},
    }
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


async def img2img_video(d: Director):
    """Tutorial #10: Real Img2Img in MOSS-Image, built from scratch; core vs exact denoise at 8 steps."""
    P = I2I_POS
    await d.camera({"box": I2I_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "Real Img2Img: exact vs core denoise", 7.0)
    await d.say(0)
    await d.say(1)
    in_url = "data:image/png;base64," + base64.b64encode(UP_VIDEO1.read_bytes()).decode()
    c_prev = Path(r"I:\HaDeS_MOSS\videos\comfy_output\img2img_measure\core_d0.9_s1013_00001_.png")
    e_prev = Path(r"I:\HaDeS_MOSS\videos\comfy_output\img2img_measure\exact_d0.9_s1013_00001_.png")
    if c_prev.exists() and e_prev.exists():
        c_u = "data:image/png;base64," + base64.b64encode(c_prev.read_bytes()).decode()
        e_u = "data:image/png;base64," + base64.b64encode(e_prev.read_bytes()).decode()
        await d.js("(h) => window.__hv.show(h)", cn_cols_html(
            [(in_url, "Input"), (c_u, "core 0.9 (sigma 1.000)"), (e_u, "exact 0.9 (sigma 0.978)")], side=480))
    await d.say(2)
    await d.js("() => window.__hv.hide()")
    await d.say(3)

    # 2. build from scratch
    await d.camera({"box": [-60, -60, 1000, 960]}, 900, pad=60, top=60)
    await d.say(4)
    loader = await basic_add(d, "HaDeSMossLoader", "MOSS Loader", P["loader"])
    await up_set(d, loader, BASIC_LOADER)
    await d.camera([loader], 900, pad=180, top=60)
    await basic_point(d, loader, "unet_name")

    await d.camera({"box": [-60, -60, 960, 960]}, 900, pad=60, top=60)
    await d.say(5)
    load_img = await basic_add(d, "LoadImage", "Load Image", P["load_img"])
    await up_set(d, load_img, {"image": "fisherman_video1.png"})
    vae_enc = await basic_add(d, "VAEEncode", "VAE Encode", P["vae_enc"])
    await basic_link(d, load_img, "IMAGE", vae_enc, "pixels", 700)
    await basic_link(d, loader, "VAE", vae_enc, "vae", 700)
    await d.say(6)

    await d.camera({"box": [-60, -60, 1400, 960]}, 900, pad=60, top=60)
    await d.say(7)
    enhancer = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enhancer"])
    await basic_link(d, loader, "CLIP", enhancer, "clip", 700)
    await basic_point(d, enhancer, "mode", {"mode": "as written"})
    await d.camera([enhancer], 900, pad=160, top=60)
    box = await d.js("""(id) => { const w = window.__hv.widget(id, 'idea'); w.value = ''; if (w.element) w.element.value = '';
        const r = w.element.getBoundingClientRect(); return [r.left, r.top]; }""", enhancer)
    await d.click(box[0] + 40, box[1] + 30)
    if not await d.js("() => document.activeElement?.tagName === 'TEXTAREA'"):
        raise SystemExit("the prompt box is not focused; nothing typed")
    async with d.fast(2.0):
        await d.page.keyboard.type(I2I_PROMPT, delay=10)

    await d.camera({"box": I2I_BOX}, 900, pad=60, top=60)
    await d.say(8)
    sampler = await basic_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["sampler"])
    await up_set(d, sampler, {"seed": BASIC_SEED, "control_after_generate": "fixed", "denoise": 0.5, "denoise_mode": "steps / denoise (core)"})
    turbo = await basic_add(d, "HaDeSMossTurbo", "MOSS Turbo LoRA", P["turbo"])
    await up_set(d, turbo, {"lora_name": BASIC_LORA, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
    async with d.fast(5.0):
        for a, out, inp in ((loader, "MODEL", "model"), (enhancer, "positive", "positive"), (enhancer, "negative", "negative"),
                            (vae_enc, "LATENT", "latent_image"), (loader, "VAE", "vae"), (turbo, "turbo", "turbo")):
            await basic_link(d, a, out, sampler, inp, 650)

    await d.say(9)
    async with d.fast(5.0):
        saver = await basic_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["saver"])
        await basic_link(d, sampler, "image", saver, "images", 700)
        telemetry = await basic_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["telemetry"])
        for k, (n, out) in enumerate(((loader, "info"), (enhancer, "info"), (sampler, "info"))):
            await basic_link(d, n, out, telemetry, f"info_{k + 1}", 650)
        report = await basic_add(d, "PreviewAny", "Preview as Text", P["report"])
        await d.js("(id) => { const n = window.__hv.node(id); n.setSize([440, 520]); n.setDirtyCanvas(true, true); }", report)
        await basic_link(d, telemetry, "report", report, "source", 700)
        await basic_link(d, telemetry, "json", saver, "metadata", 700)

    # 3. denoise 0.5 and 0.8 on core
    await d.camera([sampler], 900, pad=160, top=60)
    await d.say(10)
    await basic_point(d, sampler, "denoise_mode")
    await d.say(11)
    url_5, _ = await basic_run(d, saver)
    await d.camera([telemetry, report], 900, pad=80, top=60)
    await d.say(12)

    await d.camera([sampler], 900, pad=160, top=60)
    await d.say(13)
    await basic_point(d, sampler, "denoise", {"denoise": 0.8})
    url_8, _ = await basic_run(d, saver)
    await d.camera([saver], 900, pad=140, top=60)
    await d.say(14)

    # 4. core fails at 0.9
    await d.camera([sampler], 900, pad=160, top=60)
    await d.say(15)
    await basic_point(d, sampler, "denoise", {"denoise": 0.9})
    url_c9, _ = await basic_run(d, saver)
    await d.camera([saver], 900, pad=140, top=60)
    await d.say(16)
    await d.camera([telemetry, report], 900, pad=80, top=60)
    await d.say(17)
    await d.say(18)
    await d.say(19)

    # 5. exact at 0.9
    await d.camera([sampler], 900, pad=160, top=60)
    await d.say(20)
    await basic_point(d, sampler, "denoise_mode", {"denoise_mode": "exact"})
    await d.say(21)
    await d.say(22)
    url_e9, _ = await basic_run(d, saver)
    await d.camera([telemetry, report], 900, pad=80, top=60)
    await d.say(23)
    await d.camera([saver], 900, pad=140, top=60)
    await d.say(24)
    await d.js("(h) => window.__hv.show(h)", cn_cols_html(
        [(in_url, "Input"), (url_c9, "core 0.9 (sigma 1.000)"), (url_e9, "exact 0.9 (sigma 0.978)")], side=480))
    await d.hold(1.5)
    await d.js("() => window.__hv.hide()")

    # 6. closing
    await d.camera({"box": I2I_BOX}, 1200, pad=60, top=60)
    await d.say(25)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1200px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - Img2Img.json</div></div>')
    await d.say(26)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:40px"><img src="{url_e9}" style="height:780px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Real Img2Img: exact vs core denoise</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(27)
    await d.say(28)
    await d.end(1.5)


img2img_video.prepare = i2i_prepare


# ---- hms_v1 (HMS v1: 10 GB to 4 GB in ComfyUI) ----

HMS_LINES = [
    ("Hi, I'm HaDeS, and welcome to tutorial number eleven.",
     "Hola, soy HaDeS, y bienvenido al tutorial número once."),
    ("Today we're talking about HMS v1, the Suite's own quantization format that shrinks any ten-gigabyte MOSS-Image model down to under four gigabytes.",
     "Hoy hablamos de HMS v1, el formato de cuantización propio de la Suite que reduce cualquier modelo de MOSS-Image de diez gigas a menos de cuatro.",
     "Today we're talking about HMS version one, the Suite's own quantization format that shrinks any ten-gigabyte MOSS-Image model down to under four gigabytes."),
    ("It cuts VRAM from nearly eighteen gigabytes to eleven point six, runs more than twice as fast, and loads with ComfyUI's stock diffusion model loader.",
     "Baja la VRAM de casi dieciocho gigas a once coma seis, corre más del doble de rápido y carga con el cargador estándar de ComfyUI."),
    ("I'll explain how it works, build the converter workflow from scratch, compare BF16 against HMS v1 live, and give you the honest pros and cons.",
     "Te explicaré cómo funciona, montaremos el workflow conversor desde cero, compararemos BF16 frente a HMS v1 en directo y veremos sus pros y contras con total honestidad.",
     "I'll explain how it works, build the converter workflow from scratch, compare BF sixteen against HMS version one live, and give you the honest pros and cons."),
    ("First, why isn't HMS v1 just naive four-bit quantization? Because naive quantization breaks text and fine details.",
     "Primero, ¿por qué HMS v1 no es una simple cuantización a cuatro bits? Porque cuantizar a ciegas rompe el texto y el detalle fino.",
     "First, why isn't HMS version one just naive four-bit quantization? Because naive quantization breaks text and fine details."),
    ("HMS v1 combines two ideas: mixed precision and calibration.",
     "HMS v1 combina dos ideas: precisión mixta y calibración.",
     "HMS version one combines two ideas: mixed precision and calibration."),
    ("For mixed precision, it puts the heavy feed-forward layers of blocks zero to twenty-nine in four-bit W4A8, while protecting all attention layers and the last twelve blocks in six-bit W6A8.",
     "En precisión mixta, pone las capas feed-forward de los bloques cero a veintinueve en W4A8 de cuatro bits, mientras protege toda la atención y los últimos doce bloques en W6A8 de seis bits.",
     "For mixed precision, it puts the heavy feed-forward layers of blocks zero to twenty-nine in four-bit W four A eight, while protecting all attention layers and the last twelve blocks in six-bit W six A eight."),
    ("And for calibration, it doesn't just round weights blindly: it runs seven fixed prompts — scenes and quoted text — through your exact model using GPTQ in the ConvRot basis, adapting the quantization to how that model actually behaves.",
     "Y en calibración, no redondea pesos a ciegas: pasa siete prompts fijos — escenas y texto entrecomillado — por tu modelo exacto usando GPTQ en la base ConvRot, adaptando la cuantización a cómo se comporta realmente ese modelo."),
    ("Everything uses ComfyUI's native kernels, so you don't need any custom loader to run the resulting file.",
     "Todo usa los kernels nativos de ComfyUI, así que no necesitas ningún cargador especial para ejecutar el archivo resultante."),
    ("Let's build the converter workflow from zero. It only needs three nodes.",
     "Montemos el workflow conversor desde cero. Solo necesita tres nodos."),
    ("First, MOSS Loader with the BF16 model you want to compress — or you can plug in a merge straight from MOSS Model Block Merger.",
     "Primero, MOSS Loader con el modelo BF16 que quieres comprimir — o puedes conectar un merge directamente desde el MOSS Model Block Merger.",
     "First, MOSS Loader with the BF sixteen model you want to compress — or you can plug in a merge straight from MOSS Model Block Merger."),
    ("Next, MOSS Model Saver. Set precision to \"HMS v1 (W4A8 / W6A8, calibrated)\".",
     "Después, MOSS Model Saver. Pon la precisión en \"HMS v1 (W4A8 / W6A8, calibrated)\".",
     "Next, MOSS Model Saver. Set precision to HMS version one calibrated."),
    ("Because HMS v1 calibrates on the model itself, you must connect CLIP from the loader to the saver's clip input.",
     "Como HMS v1 calibra sobre el propio modelo, es obligatorio conectar CLIP del loader a la entrada clip del saver.",
     "Because HMS version one calibrates on the model itself, you must connect CLIP from the loader to the saver's clip input."),
    ("You can also plug the loader's info into recipe so the saved file remembers how it was made.",
     "También puedes conectar el info del loader a recipe para que el archivo guardado recuerde cómo se fabricó."),
    ("Down here, match the calibration settings to your model: twenty-five steps at CFG two point five with er_sde for Base or SFT, eight steps at CFG one with euler for a Turbo model, or fifteen steps for Bulle d'Encre.",
     "Aquí abajo, ajusta la calibración a tu modelo: veinticinco pasos a CFG dos coma cinco con er_sde para Base o SFT, ocho pasos a CFG uno con euler para un modelo Turbo, o quince pasos para Bulle d'Encre."),
    ("Connect a Preview Any to info, press Run, and it saves a three point nine six gigabyte safetensors file in your output diffusion models folder.",
     "Conecta un Preview Any a info, pulsa Run, y guardará un safetensors de tres coma noventa y seis gigas en tu carpeta de salida diffusion_models."),
    ("Converting takes eight to ten minutes, so instead of watching a progress bar, let's compare my converted Base HMS v1 file live against the original BF16.",
     "Convertir tarda entre ocho y diez minutos, así que en lugar de mirar una barra de progreso, vamos a comparar en directo mi Base HMS v1 convertido frente al BF16 original.",
     "Converting takes eight to ten minutes, so instead of watching a progress bar, let's compare my converted Base HMS version one file live against the original BF sixteen."),
    ("I load both files with ComfyUI's stock Load Diffusion Model node: ten point two gigabytes on the left, three point nine six gigabytes on the right.",
     "Cargo ambos archivos con el nodo estándar Load Diffusion Model de ComfyUI: diez coma dos gigas a la izquierda, tres coma noventa y seis gigas a la derecha."),
    ("I wire both into MOSS Compare Sampler with a harbor sign prompt containing quoted text: \"HARBOR COFFEE\" and \"OPEN DAILY\".",
     "Conecto ambos al MOSS Compare Sampler con un prompt de cartel con texto entrecomillado: \"HARBOR COFFEE\" y \"OPEN DAILY\"."),
    ("Let's run it at twenty-five steps.",
     "Vamos a lanzarlo a veinticinco pasos."),
    ("Look at that sheet: every single letter is spelled right in both, and the lighting, brick texture and wood grain match.",
     "Mira esa hoja: todas las letras están bien escritas en ambos, y la iluminación, la textura del ladrillo y la veta de la madera encajan."),
    ("In single-model runs here on my RTX 5090, BF16 takes thirteen point eight seconds, while HMS v1 finishes in six point two seconds — over twice as fast.",
     "En ejecuciones individuales aquí en mi RTX 5090, BF16 tarda trece coma ocho segundos, mientras que HMS v1 termina en seis coma dos segundos — más del doble de rápido.",
     "In single-model runs here on my RTX fifty ninety, BF sixteen takes thirteen point eight seconds, while HMS version one finishes in six point two seconds — over twice as fast."),
    ("And on our fisherman, both versions look practically identical, with a mean pixel difference of only eight to nine out of two hundred fifty-five.",
     "Y en nuestro pescador, ambas versiones se ven prácticamente idénticas, con una diferencia media de solo ocho o nueve sobre doscientos cincuenta y cinco."),
    ("So what are the pros and cons of HMS v1?",
     "Entonces, ¿cuáles son los pros y los contras de HMS v1?",
     "So what are the pros and cons of HMS version one?"),
    ("Pros: sixty-one percent smaller on disk, eleven point six gigabytes of peak VRAM instead of seventeen point eight, more than twice as fast, native ComfyUI support, and text accuracy that matches BF16.",
     "Pros: un sesenta y uno por ciento menos en disco, once coma seis gigas de pico de VRAM en lugar de diecisiete coma ocho, más del doble de rápido, soporte nativo en ComfyUI y precisión de texto a la altura de BF16.",
     "Pros: sixty-one percent smaller on disk, eleven point six gigabytes of peak VRAM instead of seventeen point eight, more than twice as fast, native ComfyUI support, and text accuracy that matches BF sixteen."),
    ("Cons: converting takes eight to ten minutes and peaks around twenty-five gigabytes of VRAM during calibration, it only accepts BF16 or FP16 inputs, and on complex merges you should always compare the HMS file against its BF16 original for slight brightness shifts.",
     "Contras: convertir tarda de ocho a diez minutos y alcanza unos veinticinco gigas de pico de VRAM durante la calibración, solo acepta entradas en BF16 o FP16, y en merges complejos conviene comparar siempre el archivo HMS frente a su original en BF16 por si hay ligeros cambios de brillo.",
     "Cons: converting takes eight to ten minutes and peaks around twenty-five gigabytes of VRAM during calibration, it only accepts BF sixteen or FP sixteen inputs, and on complex merges you should always compare the HMS file against its BF sixteen original for slight brightness shifts."),
    ("Think of HMS v1 as a practical alternative if you want to save VRAM or just try MOSS-Image on smaller hardware — it's never going to beat the full BF16 models if you have the memory for them.",
     "Piensa en HMS v1 como una alternativa práctica si quieres ahorrar VRAM o simplemente probar MOSS-Image en equipos más modestos — nunca va a rivalizar con los modelos completos en BF16 si tienes memoria para ellos.",
     "Think of HMS version one as a practical alternative if you want to save VRAM or just try MOSS-Image on smaller hardware — it's never going to beat the full BF sixteen models if you have the memory for them."),
    ("And if you don't want to convert them yourself, Base and SFT are already published in HMS v1 on my Hugging Face and on Civitai.",
     "Y si no quieres convertirlos tú mismo, Base y SFT ya están publicados en HMS v1 en mi Hugging Face y en Civitai.",
     "And if you don't want to convert them yourself, Base and SFT are already published in HMS version one on my Hugging Face and on Civitai."),
    ("You can download this HMS v1 converter workflow for free from the description.",
     "Puedes descargar gratis este workflow conversor a HMS v1 desde la descripción.",
     "You can download this HMS version one converter workflow for free from the description."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

HMS_BF16 = "mossImage_base_bf16.safetensors"
HMS_FILE = "MOSS-Image_Base_HMS_v1.safetensors"
HMS_SIGN_PROMPT = (
    "Photograph of a rustic wooden harbor café sign reading \"HARBOR COFFEE\" and \"OPEN DAILY\", "
    "hanging beside a weathered brick wall in warm morning sunlight."
)
HMS_POS = {
    "loader": (0, 0),
    "saver": (460, 0),
    "info_prev": (940, 0),
    "unet_a": (0, 620),
    "unet_b": (0, 800),
    "clip": (0, 980),
    "vae": (0, 1140),
    "enhancer": (460, 620),
    "latent": (940, 620),
    "cmp": (1400, 620),
}
HMS_BOX = [-60, -60, 1980, 1300]


def hms_arch_card():
    return (
        '<div style="background:#14151a;border:2px solid #2b2d38;border-radius:16px;padding:38px 46px;max-width:1160px;color:#fff;font-family:Segoe UI,sans-serif">'
        '<div style="font:900 44px Segoe UI Black,sans-serif;margin-bottom:22px">How <span style="color:#ffcc33">HMS v1</span> works (3.96 GB)</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:28px">'
        '<div style="background:#1c1e26;border-radius:12px;padding:24px;border-left:6px solid #ffcc33">'
        '<div style="font:800 26px Segoe UI,sans-serif;color:#ffcc33;margin-bottom:10px">1. Mixed Precision (ffn30)</div>'
        '<div style="font:400 21px Segoe UI,sans-serif;line-height:1.45;color:#d8dae2">'
        '• Blocks 0–29 FFN → <b>W4A8</b> (4-bit)<br>'
        '• Blocks 30–41 FFN → <b>W6A8</b> (6-bit, detail &amp; text)<br>'
        '• All 42 Attention layers → <b>W6A8</b> (6-bit)<br>'
        '• Native ComfyUI / comfy-kitchen kernels</div></div>'
        '<div style="background:#1c1e26;border-radius:12px;padding:24px;border-left:6px solid #7ee787">'
        '<div style="font:800 26px Segoe UI,sans-serif;color:#7ee787;margin-bottom:10px">2. Calibrated (C1 + GPTQ)</div>'
        '<div style="font:400 21px Segoe UI,sans-serif;line-height:1.45;color:#d8dae2">'
        '• Runs <b>7 fixed prompts</b> (4 scenes + 3 quoted texts)<br>'
        '• Adapts quantization in the <b>ConvRot</b> basis<br>'
        '• Uses the model\'s own activations<br>'
        '• Stock <b>Load Diffusion Model</b> (no custom loader)</div></div>'
        '</div></div>'
    )


def hms_pros_cons_card():
    return (
        '<div style="background:#14151a;border:2px solid #2b2d38;border-radius:16px;padding:36px 44px;max-width:1200px;color:#fff;font-family:Segoe UI,sans-serif">'
        '<div style="font:900 44px Segoe UI Black,sans-serif;margin-bottom:22px"><span style="color:#ffcc33">HMS v1</span>: Honest Pros &amp; Cons</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:28px">'
        '<div style="background:#17221b;border-radius:12px;padding:24px;border-left:6px solid #7ee787">'
        '<div style="font:800 26px Segoe UI,sans-serif;color:#7ee787;margin-bottom:10px">Pros</div>'
        '<div style="font:400 21px Segoe UI,sans-serif;line-height:1.5;color:#d8dae2">'
        '• <b>3.96 GB</b> on disk (vs 10.2 GB BF16, −61%)<br>'
        '• <b>11.6 GB</b> peak VRAM (vs 17.8 GB BF16)<br>'
        '• <b>2.2× faster</b> (6.2 s vs 13.8 s at 25 steps)<br>'
        '• Text &amp; quality match BF16 closely<br>'
        '• Native ComfyUI loader</div></div>'
        '<div style="background:#24191b;border-radius:12px;padding:24px;border-left:6px solid #ff7b72">'
        '<div style="font:800 26px Segoe UI,sans-serif;color:#ff7b72;margin-bottom:10px">Cons &amp; Limits</div>'
        '<div style="font:400 21px Segoe UI,sans-serif;line-height:1.5;color:#d8dae2">'
        '• Conversion takes <b>8–10 min</b> (RTX 5090)<br>'
        '• Needs <b>~25 GB VRAM</b> during calibration<br>'
        '• BF16 / FP16 source models only<br>'
        '• Check complex merges vs BF16 for luma drift<br>'
        '• Alternative for VRAM savings — never replaces BF16</div></div>'
        '</div></div>'
    )


async def hms_prepare(page):
    """Off camera: load BF16, HMS v1, CLIP and VAE so the live Compare Sampler run is warm."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": HMS_BF16, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": HMS_FILE, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": BASIC_LOADER["clip_name"], "type": "moss_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": BASIC_LOADER["vae_name"]}},
        "5": {"class_type": "HaDeSMossPromptEnhancer", "inputs": {
            "clip": ["3", 0], "idea": "a harbor", "mode": "as written", "width": 1024, "height": 1024,
            "temperature": 0.7, "seed": 1, "max_tokens": 1024, "negative": "worst quality, low quality"}},
        "6": {"class_type": "HaDeSMossLatent", "inputs": {
            "aspect_ratio": "1:1", "megapixels": 1.0, "custom_width": 1024, "custom_height": 1024, "batch_size": 1, "seed": 7}},
        "7": {"class_type": "HaDeSMossCompareSampler", "inputs": {
            "model_A": ["1", 0], "model_B": ["2", 0], "positive": ["5", 0], "negative": ["5", 1],
            "latent_image": ["6", 0], "vae": ["4", 0], "seed": 7, "steps": 4, "cfg": 2.5,
            "sampler_name": "er_sde", "scheduler": "simple", "flow_shift": 0.0, "slg": "off",
            "label_A": "BF16", "label_B": "HMS", "blind": False, "blind_seed": 0, "footer": False,
            "save": False, "embed_workflow": False, "filename_prefix": "MOSS/warmup_hms"}},
    }
    print("warm-up run:", await page.evaluate(UP_WARM_JS, g), flush=True)


async def hms_v1_video(d: Director):
    """Tutorial #11: HMS v1 conversion from scratch, live BF16 vs HMS v1 comparison, and honest pros & cons."""
    P = HMS_POS
    await d.camera({"box": HMS_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.62, H * 0.55)

    # 1. introduction
    d.title("HaDeS MOSS Suite", "HMS v1: 10 GB to 4 GB in ComfyUI", 7.0)
    for i in range(4):
        await d.say(i)

    # 2. how HMS v1 works
    await d.js("(h) => window.__hv.show(h)", hms_arch_card())
    for i in range(4, 9):
        await d.say(i)
    await d.js("() => window.__hv.hide()")

    # 3. converter workflow from scratch
    await d.camera({"box": [-60, -60, 1460, 560]}, 900, pad=60, top=60)
    await d.say(9)
    loader = await basic_add(d, "HaDeSMossLoader", "MOSS Loader", P["loader"])
    await up_set(d, loader, BASIC_LOADER)
    await d.say(10)

    saver_m = await basic_add(d, "HaDeSMossModelSaver", "MOSS Model Saver", P["saver"])
    await up_set(d, saver_m, {"precision": "HMS v1 (W4A8 / W6A8, calibrated)", "filename_prefix": "diffusion_models/MOSS_Base_HMS_v1"})
    await basic_link(d, loader, "MODEL", saver_m, "model", 700)
    await d.camera([saver_m], 900, pad=180, top=60)
    await d.say(11)

    await d.camera([loader, saver_m], 900, pad=80, top=60)
    await d.say(12)
    await basic_link(d, loader, "CLIP", saver_m, "clip", 700)
    await d.say(13)
    await basic_link(d, loader, "info", saver_m, "recipe", 700)

    await d.camera([saver_m], 900, pad=160, top=60)
    await d.say(14)
    await basic_point(d, saver_m, "hms_steps")
    await d.hold(0.4)
    await basic_point(d, saver_m, "hms_cfg")

    await d.camera({"box": [-60, -60, 1460, 560]}, 900, pad=60, top=60)
    await d.say(15)
    info_prev = await basic_add(d, "PreviewAny", "Preview as Text", P["info_prev"])
    await d.js("(id) => { const n = window.__hv.node(id); n.setSize([380, 320]); n.setDirtyCanvas(true, true); }", info_prev)
    await basic_link(d, saver_m, "info", info_prev, "source", 700)

    # 4. live comparison: BF16 vs HMS v1
    await d.camera({"box": [-60, 560, 1980, 1300]}, 900, pad=60, top=60)
    await d.say(16)
    async with d.fast(5.0):
        unet_a = await basic_add(d, "UNETLoader", "Load Diffusion Model", P["unet_a"])
        await up_set(d, unet_a, {"unet_name": HMS_BF16, "weight_dtype": "default"})
        unet_b = await basic_add(d, "UNETLoader", "Load Diffusion Model", P["unet_b"])
        await up_set(d, unet_b, {"unet_name": HMS_FILE, "weight_dtype": "default"})
        clip_n = await basic_add(d, "CLIPLoader", "Load CLIP", P["clip"])
        await up_set(d, clip_n, {"clip_name": BASIC_LOADER["clip_name"], "type": "moss_image", "device": "default"})
        vae_n = await basic_add(d, "VAELoader", "Load VAE", P["vae"])
        await up_set(d, vae_n, {"vae_name": BASIC_LOADER["vae_name"]})
    await d.say(17)

    enh = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enhancer"])
    await basic_link(d, clip_n, "CLIP", enh, "clip", 650)
    await basic_point(d, enh, "mode", {"mode": "as written"})
    box = await d.js("""(id) => { const w = window.__hv.widget(id, 'idea'); w.value = ''; if (w.element) w.element.value = '';
        const r = w.element.getBoundingClientRect(); return [r.left, r.top]; }""", enh)
    await d.click(box[0] + 40, box[1] + 30)
    if not await d.js("() => document.activeElement?.tagName === 'TEXTAREA'"):
        raise SystemExit("the prompt box is not focused; nothing typed")
    async with d.fast(2.0):
        await d.page.keyboard.type(HMS_SIGN_PROMPT, delay=10)

    latent = await basic_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": BASIC_SEED, "control_after_generate": "fixed"})

    cmp = await basic_add(d, "HaDeSMossCompareSampler", "MOSS Compare Sampler", P["cmp"])
    await up_set(d, cmp, {"steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple",
                           "label_A": "BF16 (10.2 GB)", "label_B": "HMS v1 (3.96 GB)", "save": True, "embed_workflow": False})
    await d.say(18)
    async with d.fast(5.0):
        for a, out, inp in ((unet_a, "MODEL", "model_A"), (unet_b, "MODEL", "model_B"), (enh, "positive", "positive"),
                            (enh, "negative", "negative"), (latent, "latent", "latent_image"), (latent, "seed", "seed"), (vae_n, "VAE", "vae")):
            await basic_link(d, a, out, cmp, inp, 600)

    await d.say(19)
    await d.js("(id) => { const n = window.__hv.node(id); n.mode = 2; window.__hv.app().graph.setDirtyCanvas(true, true); }", saver_m)
    url_cmp, _ = await basic_run(d, cmp)
    await d.camera([cmp], 900, pad=120, top=60)
    await d.say(20)
    await d.js("(h) => window.__hv.show(h)", f'<div style="text-align:center"><img src="{url_cmp}" style="height:780px;border-radius:8px"></div>')
    await d.say(21)
    fish_cmp = Path(r"I:\HaDeS_MOSS\videos\comfy_output\hms_measure\cmp_fish_00001_.png")
    if fish_cmp.exists():
        fu = "data:image/png;base64," + base64.b64encode(fish_cmp.read_bytes()).decode()
        await d.js("(h) => window.__hv.show(h)", f'<div style="text-align:center"><img src="{fu}" style="height:780px;border-radius:8px"></div>')
    await d.say(22)
    await d.js("() => window.__hv.hide()")

    # 5. pros & cons
    await d.js("(h) => window.__hv.show(h)", hms_pros_cons_card())
    for i in range(23, 28):
        await d.say(i)
    await d.js("() => window.__hv.hide()")

    # 6. closing
    await d.camera({"box": HMS_BOX}, 1200, pad=60, top=60)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1200px">'
               '<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">Free download (link in the description)</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">HaDeS MOSS Suite - HMS v1 Converter.json</div></div>')
    await d.say(28)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:40px"><img src="{url_cmp}" style="height:720px;border-radius:6px">'
               '<div style="color:#fff;max-width:600px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">HMS v1: 10 GB to 4 GB in ComfyUI</div>'
               '<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">Workflow, Suite and models: links in the description</div></div></div>')
    await d.say(29)
    await d.say(30)
    await d.end(1.5)


hms_v1_video.prepare = hms_prepare


# ---- advanced workflow 1: Production Poster (examples/advanced) ----

AP_TITLE = "HaDeS MOSS Suite - Production Poster"
AP_FILE = None  # built on camera (guion v2, a lesson); the finished example is the download
AP_SEED = 1013  # the example's seed; measured warm 42.7 s, images 2 and 3 exact, picks 2 (videos/advanced/adv_poster/MEDIDAS.md)
AP_SHOTS = "adv_poster"  # the measured run of the same seed in comfy_output, for the overlays (the failing candidate, the crops)
AP_IDS = {"loader": 1, "draft": 2, "editor": 3, "compiler": 4, "enh": 5, "ks": 6, "candidates": 7, "check": 8, "check_report": 9,
          "upmodel": 10, "up": 11, "enc": 12, "ks2": 13, "check2": 14, "check2_report": 15, "save": 16, "tel": 17, "report": 18}

AP_LINES = [
    ("Hi, I'm HaDeS, and welcome to advanced workflow number one: Production Poster.",
     "Hola, soy HaDeS, y bienvenido al workflow avanzado número uno: Production Poster."),
    ("Today we'll make a festival poster with three texts, and the workflow itself will check that every text is right and in its place.",
     "Hoy vamos a hacer un póster de festival con tres textos, y el propio workflow comprobará que cada texto está bien escrito y en su sitio."),
    ("You'll learn how to plan an image with boxes, how to make several versions at once, how to let ComfyUI read the text for you, and how to make the winner twice as big.",
     "Vas a aprender a planificar una imagen con cajas, a hacer varias versiones a la vez, a dejar que ComfyUI lea el texto por ti y a hacer el ganador el doble de grande."),
    ("We'll build it together, from an empty canvas, one node at a time. As always, everything is free, and all the links are in the description.",
     "Lo montamos juntos, desde el lienzo vacío, nodo a nodo. Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    ("First, why do we need this? MOSS-Image is good at spelling. But it doesn't always put the text where you asked.",
     "Primero, ¿por qué hace falta esto? MOSS-Image escribe bien. Pero no siempre pone el texto donde se lo pides.",
     "First, why do we need this? MOSS Image is good at spelling. But it doesn't always put the text where you asked."),
    ("Look at this one. The yellow frame is where I asked for the title. The model broke it into two lines, and the second line fell out of the box.",
     "Mira este. El marco amarillo es donde pedí el título. El modelo lo partió en dos líneas, y la segunda se salió de la caja."),
    ("You could generate again and again and check every image by eye. Or you can let the workflow do it. That's what we'll build.",
     "Podrías generar una y otra vez y revisar cada imagen a ojo. O puedes dejar que lo haga el workflow. Eso es lo que vamos a montar."),
    ("Let's start with the MOSS Loader. The model is HaDeS MOSS v1.0 Turbo: MOSS-Image Base with my eight-step turbo already inside, so it needs only eight steps at CFG one.",
     "Empezamos con el MOSS Loader. El modelo es HaDeS MOSS v1.0 Turbo: MOSS-Image Base con mi turbo de ocho pasos ya dentro, así que solo necesita ocho pasos a CFG uno.",
     "Let's start with the MOSS Loader. The model is HaDeS MOSS version one Turbo: MOSS Image Base with my eight-step turbo already inside, so it needs only eight steps at C F G one."),
    ("The text encoder reads the prompt, and, as we'll see later, it can also read text in an image. The VAE turns the result into pixels.",
     "El text encoder lee el prompt y, como veremos luego, también sabe leer el texto de una imagen. El VAE convierte el resultado en píxeles.",
     "The text encoder reads the prompt, and, as we'll see later, it can also read text in an image. The V A E turns the result into pixels."),
    ("Now the plan of the poster. The MOSS Layout Editor holds a canvas with boxes. Each box says what goes there.",
     "Ahora el plano del póster. El MOSS Layout Editor tiene un lienzo con cajas. Cada caja dice qué va ahí."),
    ("There are two kinds. Text boxes hold the exact words to write: the title at the top, the town's name in a small badge, and the ticket line at the bottom.",
     "Hay dos tipos. Las cajas de texto llevan las palabras exactas que hay que escribir: el título arriba, el nombre del pueblo en una pequeña insignia y la línea de las entradas abajo."),
    ("The other boxes describe things: here, the fiddler on the right and the lighthouse on the left.",
     "Las otras cajas describen cosas: aquí, la violinista a la derecha y el faro a la izquierda."),
    ("You can move a box, or resize it, right here on the canvas. Remember the title box: its size will matter later.",
     "Puedes mover una caja, o cambiar su tamaño, aquí mismo en el lienzo. Acuérdate de la caja del título: su tamaño importará luego."),
    ("Where do the boxes come from? You can draw them, or let the Prompt Enhancer draft them. In layout mode, you write your idea in one sentence, connect it to the editor, and press Draft layout.",
     "¿De dónde salen las cajas? Puedes dibujarlas, o dejar que el Prompt Enhancer las proponga. En modo layout, escribes tu idea en una frase, lo conectas al editor y pulsas Draft layout."),
    ("I drafted this poster that way and then adjusted the boxes. For today, I mute the Prompt Enhancer, so these boxes stay as they are.",
     "Este póster lo empecé así y luego ajusté las cajas. Para hoy, desactivo el Prompt Enhancer, así estas cajas se quedan como están."),
    ("Next, the MOSS Layout Compiler. It turns the boxes into the prompt MOSS-Image understands, and it makes the empty image, the latent, with the poster's size.",
     "Después, el MOSS Layout Compiler. Convierte las cajas en el prompt que entiende MOSS-Image, y crea la imagen vacía, el latent, con el tamaño del póster.",
     "Next, the MOSS Layout Compiler. It turns the boxes into the prompt MOSS Image understands, and it makes the empty image, the latent, with the poster's size."),
    ("And here's the important setting: batch size, four. What is it for? Instead of one poster, we get four, each from different noise, in one run.",
     "Y aquí está el ajuste importante: batch size, cuatro. ¿Para qué sirve? En lugar de un póster, tenemos cuatro, cada uno con un ruido distinto, en una sola ejecución."),
    ("Why four? Because, as we saw, the model sometimes misplaces the title. With four tries, the chance that at least one is right is much better.",
     "¿Por qué cuatro? Porque, como vimos, el modelo a veces coloca mal el título. Con cuatro intentos, es mucho más probable que al menos uno salga bien."),
    ("A second Prompt Enhancer, in as written mode, encodes that prompt exactly as the compiler wrote it.",
     "Un segundo Prompt Enhancer, en modo as written, codifica ese prompt exactamente como lo escribió el compiler."),
    ("The MOSS KSampler makes the four posters: eight steps, CFG one, euler and simple, as the turbo model needs, and a fixed seed so you can repeat my result.",
     "El MOSS KSampler hace los cuatro pósters: ocho pasos, CFG uno, euler y simple, como necesita el modelo turbo, y una semilla fija para que puedas repetir mi resultado.",
     "The MOSS KSampler makes the four posters: eight steps, C F G one, euler and simple, as the turbo model needs, and a fixed seed so you can repeat my result."),
    ("A Preview Image will show all four.",
     "Un Preview Image enseñará los cuatro."),
    ("Now the key node: MOSS Text Check. What does it do? It reads the text in an image and compares it with the text you asked for.",
     "Ahora el nodo clave: MOSS Text Check. ¿Qué hace? Lee el texto de una imagen y lo compara con el que pediste."),
    ("It reads with MOSS-Image's own text encoder, the same one that read our prompt. No other model is loaded.",
     "Lee con el propio text encoder de MOSS-Image, el mismo que leyó nuestro prompt. No se carga ningún otro modelo.",
     "It reads with MOSS Image's own text encoder, the same one that read our prompt. No other model is loaded."),
    ("We give it the four posters, the text encoder, and the layout from the editor, so it knows which text should be in which box.",
     "Le damos los cuatro pósters, el text encoder y el layout del editor, para que sepa qué texto debe ir en cada caja."),
    ("For each text it gives a verdict: exact, close if some letters differ, missing, or outside its box if it's written, but in the wrong place.",
     "Para cada texto da un veredicto: exact si es exacto, close si cambian algunas letras, missing si falta, o outside its box si está escrito, pero en el sitio equivocado."),
    ("And its best image output passes on only the poster with the most texts exact. If none is perfect, its passed output says false, so you know to try another seed.",
     "Y su salida best image deja pasar solo el póster con más textos exactos. Si ninguno es perfecto, su salida passed dice false, y así sabes que toca probar otra semilla."),
    ("A Preview as Text shows its report.",
     "Un Preview as Text enseña su informe."),
    ("Now we make the winner twice as big. First, RealESRGAN x2, an upscaling model, doubles the pixels.",
     "Ahora hacemos el ganador el doble de grande. Primero, RealESRGAN x2, un modelo de escalado, duplica los píxeles.",
     "Now we make the winner twice as big. First, Real E S R GAN times two, an upscaling model, doubles the pixels."),
    ("But an upscaler can only guess the fine detail, so it looks soft. So VAE Encode turns the big image back into a latent, and a second MOSS KSampler redraws it lightly.",
     "Pero un escalador solo puede adivinar el detalle fino, así que queda blando. Por eso VAE Encode vuelve a convertir la imagen grande en latent, y un segundo MOSS KSampler la redibuja suavemente.",
     "But an upscaler can only guess the fine detail, so it looks soft. So V A E Encode turns the big image back into a latent, and a second MOSS KSampler redraws it lightly."),
    ("How lightly? That's denoise: zero point two five. It keeps the poster, and redraws only the fine detail. Same model, same prompt, same seed.",
     "¿Cómo de suave? Eso es el denoise: cero coma dos cinco. Mantiene el póster y redibuja solo el detalle fino. El mismo modelo, el mismo prompt, la misma semilla."),
    ("And a second Text Check reads the big poster, to make sure the redraw didn't break a letter.",
     "Y un segundo Text Check lee el póster grande, para asegurarse de que el redibujado no ha roto ninguna letra."),
    ("Last, Suite Telemetry gathers the settings of every node, and the MOSS Image Saver writes them into the PNG with the whole workflow. Drop the PNG on ComfyUI, and you get it all back.",
     "Por último, Suite Telemetry reúne los ajustes de cada nodo, y el MOSS Image Saver los escribe en el PNG con el workflow entero. Arrastra el PNG a ComfyUI y lo recuperas todo.",
     "Last, Suite Telemetry gathers the settings of every node, and the MOSS Image Saver writes them into the P N G with the whole workflow. Drop the P N G on ComfyUI, and you get it all back."),
    ("Let's run it.",
     "Vamos a ejecutarlo."),
    ("Here are the four posters. Look at the titles before we read the report. Can you see which ones broke?",
     "Aquí están los cuatro pósters. Mira los títulos antes de leer el informe. ¿Ves cuáles se han partido?"),
    ("Now Text Check's report: one line per poster. The first and the last have the title outside its box. The second and the third have all three texts exact.",
     "Ahora el informe de Text Check: una línea por póster. El primero y el último tienen el título fuera de su caja. El segundo y el tercero tienen los tres textos exactos."),
    ("When two tie, it looks at which one it read most closely, and then takes the first. Here, the second poster goes on.",
     "Cuando dos empatan, mira cuál ha leído con más precisión, y después se queda con el primero. Aquí sigue adelante el segundo póster."),
    ("And here it is at 2048 by 3072. The final check: all three texts still exact.",
     "Y aquí está a 2048 por 3072. La comprobación final: los tres textos siguen exactos.",
     "And here it is at twenty forty-eight by thirty seventy-two. The final check: all three texts still exact."),
    ("At one hundred percent, compare simply enlarging the picture with our MOSS pass: the lighthouse, the rocks and her face stay sharp.",
     "Al cien por cien, compara ampliar la imagen sin más con nuestra pasada de MOSS: el faro, las rocas y su cara se mantienen nítidos."),
    ("The whole run took about forty seconds on my RTX 5090.",
     "Toda la ejecución tardó unos cuarenta segundos en mi RTX 5090.",
     "The whole run took about forty seconds on my R T X fifty ninety."),
    ("I ran this on three seeds: twelve posters. Every text was spelled right on all twelve.",
     "Lo ejecuté con tres semillas: doce pósters. Todos los textos estaban bien escritos en los doce."),
    ("But only three of the twelve kept the title on one line, inside its box. That's why we make four and let Text Check choose.",
     "Pero solo tres de los doce mantuvieron el título en una línea, dentro de su caja. Por eso hacemos cuatro y dejamos que Text Check elija."),
    ("And making it twice as big never changed a verdict.",
     "Y hacerlo el doble de grande no cambió ningún veredicto."),
    ("Remember the title box? If you want the title on two lines, make that box taller. Text Check holds the image to your boxes, so the box decides.",
     "¿Te acuerdas de la caja del título? Si quieres el título en dos líneas, haz esa caja más alta. Text Check obliga a la imagen a respetar tus cajas, así que la caja decide."),
    ("Let's recap. One: plan the image with boxes in the Layout Editor.",
     "Repasemos. Uno: planifica la imagen con cajas en el Layout Editor."),
    ("Two: make several versions at once with batch size, and let Text Check pick the one with every text right.",
     "Dos: haz varias versiones a la vez con batch size, y deja que Text Check elija la que tiene todos los textos bien."),
    ("Three: double it with an upscaler and a light MOSS pass, and check it again.",
     "Tres: duplícala con un escalador y una pasada suave de MOSS, y vuelve a comprobarla."),
    ("You can download this workflow for free from GitHub. The link is in the description.",
     "Puedes descargar este workflow gratis desde GitHub. El enlace está en la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]

# graph-space positions of the lesson's nodes (two rows, the x2 stage to the right)
AP_POS = {"load": (0, 0), "draft": (450, 0), "editor": (950, 0), "compiler": (1550, 0), "enh": (1550, 600), "ks": (2050, 0),
          "candidates": (2050, 560), "check": (2500, 0), "check_report": (2500, 420), "upmodel": (2950, 0), "up": (2950, 150),
          "enc": (2950, 300), "ks2": (3350, 0), "check2": (3800, 0), "check2_report": (3800, 420), "save": (4250, 0), "tel": (4250, 560)}


def ap_view(name):
    return f"/view?filename={name}&subfolder={AP_SHOTS}&type=output"


def ap_card(lines, top=230):
    """Measured numbers, one per row, big."""
    rows = "".join(f'<div style="font:700 50px Segoe UI,sans-serif;margin:26px 0">{t}</div>' for t in lines)
    return f'<div style="color:#fff;text-align:center;margin-top:{top}px">{rows}</div>'


def ap_fail_html(es):
    """The failing candidate (seed 1013, image 1), its top part at 1:1 with the title's box drawn in yellow."""
    z = 1920 / W  # overlay px per video px: the layer is zoomed to the page width
    x0, y0, w, h = 100, 0, 824, 420
    box = (200, 60, 824, 140)
    label = "La caja del título" if es else "The title's box"
    return (f'<div style="text-align:center;color:#fff;font:600 30px Segoe UI,sans-serif">'
            f'<div style="margin:120px 0 24px;color:#ffcc33">{label}</div>'
            f'<div style="position:relative;display:inline-block;width:{w * 1.5 * z}px;height:{h * 1.5 * z}px;overflow:hidden;border-radius:8px">'
            f'<img src="{ap_view("s1013_all_00001_.png")}" style="position:absolute;max-width:none;width:{1024 * 1.5 * z}px;'
            f'left:{-x0 * 1.5 * z}px;top:{-y0 * 1.5 * z}px">'
            f'<div style="position:absolute;border:{6 * z}px solid #ffd200;box-sizing:border-box;left:{(box[0] - x0) * 1.5 * z}px;'
            f'top:{(box[1] - y0) * 1.5 * z}px;width:{(box[2] - box[0]) * 1.5 * z}px;height:{(box[3] - box[1]) * 1.5 * z}px"></div>'
            f'</div></div>')


def ap_crops_html(es):
    """100 % crops: the picked 1024 poster enlarged x2 by the browser against the final 2048 x 3072 (512 image px = 512 video px)."""
    side = 512 * 1920 / W
    crops = ((1250, 480), (220, 480), (100, 1800))  # face, lighthouse, rocks (2048 x 3072 space)
    labels = ("1024 ampliado ×2", "×2 + pasada MOSS") if es else ("1024 enlarged ×2", "×2 + MOSS pass")
    cells = []
    for x, y in crops:
        for url, scale in ((ap_view("s1013_best1024_00001_.png"), 2), (ap_view("s1013_final_00001_.png"), 1)):
            cells.append(f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:6px;position:relative">'
                         f'<img src="{url}" style="position:absolute;max-width:none;width:{2048 * side / 512}px;height:{3072 * side / 512}px;'
                         f'left:{-x * side / 512}px;top:{-y * side / 512}px"></div>')
    head = "".join(f'<div style="color:#fff;font:600 24px Segoe UI,sans-serif;text-align:center">{t}</div>' for t in labels)
    # three rows of two would not fit: three columns of pairs, face / lighthouse / rocks
    pairs = "".join(f'<div style="display:grid;grid-template-columns:{side}px;gap:12px">{cells[2 * i]}{cells[2 * i + 1]}</div>'
                    for i in range(3))
    left = f'<div style="display:grid;grid-template-rows:{side}px {side}px;gap:12px;align-items:center;color:#fff;font:600 24px Segoe UI,sans-serif;text-align:right">' \
           f'<div>{labels[0]}</div><div>{labels[1]}</div></div>'
    return (f'<div style="display:flex;gap:22px;align-items:flex-start;margin-top:10px">{left}{pairs}</div>')


def ap_candidates_html(urls, report, es):
    """The take's four candidates side by side, each with its line of Text Check's report (exact count, picked)."""
    import re as _re
    lines = {int(m.group(1)): m.group(0) for m in _re.finditer(r"image (\d+): [^\n]*", report)}
    cells = []
    for i, url in enumerate(urls, 1):
        line = lines.get(i, "")
        ok = "3/3" in line
        picked = "<- best" in line
        verdict = ("3 / 3 exactos" if es else "3 / 3 exact") if ok else ("título fuera de su caja" if es else "title out of its box")
        colour = "#5fd17a" if ok else "#ff6b5f"
        badge = (f'<div style="margin-top:10px;font:800 26px Segoe UI,sans-serif;color:#ffcc33">{"ELEGIDO" if es else "PICKED"}</div>'
                 if picked else '<div style="margin-top:10px;height:35px"></div>')
        border = "4px solid #ffcc33" if picked else "4px solid transparent"
        cells.append(f'<div style="text-align:center"><img src="{url}" style="height:600px;border-radius:6px;border:{border}">'
                     f'<div style="margin-top:12px;font:700 26px Segoe UI,sans-serif;color:{colour}">{verdict}</div>{badge}</div>')
    return f'<div style="display:flex;gap:30px;margin-top:40px">{"".join(cells)}</div>'


def ap_final_html(url, report, es):
    """The final 2048 x 3072 poster and its three texts as the final Text Check read them."""
    import re as _re
    names = ("Título", "Nombre del pueblo", "Entradas") if es else ("Title", "Town's name", "Ticket line")
    verdicts = _re.findall(r"^\[([a-z ]+)\]", report, flags=_re.M)
    rows = "".join(f'<div style="font:700 40px Segoe UI,sans-serif;margin:22px 0;color:{"#5fd17a" if v == "exact" else "#ff6b5f"}">'
                   f'{"✓" if v == "exact" else "✗"} {n}: {v}</div>' for n, v in zip(names, verdicts))
    head = "Comprobación final · 2048 × 3072" if es else "Final check · 2048 × 3072"
    return (f'<div style="display:flex;align-items:center;gap:80px;margin-top:30px"><img src="{url}" style="height:820px;border-radius:6px">'
            f'<div style="color:#fff"><div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-bottom:20px">{head}</div>{rows}</div></div>')


async def ap_take(d, ids):
    """What the on-camera run produced: the candidates' urls, both Text Check reports, the saved poster's url."""
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    out = run["outputs"]
    view = lambda im: f"/view?filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}"
    return ([view(im) for im in out[str(ids["candidates"])]["images"]], out[str(ids["check_report"])]["text"][0],
            out[str(ids["check2_report"])]["text"][0], view(out[str(ids["save"])]["images"][0]))


async def ap_prepare(page):
    """Off camera: the whole example once with another seed (1014), so the models, the upscaler and the prompt are loaded and the
    on-camera run is the measured warm one."""
    import copy
    import urllib.parse
    import make_examples as mx
    base = urllib.parse.urlsplit(page.url)
    info = mx.fetch(f"{base.scheme}://{base.netloc}", "/object_info")
    g = {k: v for k, v in copy.deepcopy(mx.EXAMPLES[AP_TITLE]).items() if k != "note"}
    g["ks"][1]["seed"] = g["ks2"][1]["seed"] = AP_SEED + 1
    g["save"][1]["filename_prefix"] = "MOSS/warmup_poster"
    print("warm-up run:", await page.evaluate(UP_WARM_JS, mx.to_api(g, info, {})), flush=True)


async def ap_add(d, *a):
    """basic_add, then the pointer leaves the new node's title (resting there shows the node's description tooltip)."""
    node = await basic_add(d, *a)
    await d.move(W * 0.97, H * 0.55, 400)
    return node


async def adv_poster(d: Director):
    """Advanced workflow #1 as a lesson (guion v2): the problem, the workflow built node by node with zooms, a run, the report read
    line by line, the final check and crops, four measured numbers, the title-box tip, a recap."""
    import make_examples as mx
    P = AP_POS
    es = d.language == "es"
    final_url = ap_view("s1013_final_00001_.png")
    await d.camera({"box": [-60, -80, 1600, 1100]}, 10, pad=60, top=60)
    await d.move(W * 0.97, H * 0.55, 10)

    # 1. what we'll learn
    await d.js("(h) => window.__hv.show(h, 10)",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:30px"><img src="{final_url}" style="height:820px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               f'<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">{"Workflow avanzado n.º 1" if es else "Advanced workflow #1"}</div>'
               '<div style="font:700 46px Segoe UI,sans-serif;margin-top:16px">Production Poster</div></div></div>')
    await d.hold(0.8)
    for k in range(4):
        await d.say(k)

    # 2. the problem
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_fail_html(es))
    await d.say(4)
    await d.say(5)
    await d.say(6)
    await d._after_voice(0.5)
    await d.js("() => window.__hv.hide()")

    # 3. the model
    load = await ap_add(d, "HaDeSMossLoader", "MOSS Loader", P["load"])
    await up_set(d, load, dict(BASIC_LOADER, unet_name="HaDeS_MOSS_v1.0_Turbo.safetensors"))
    await d.camera([load], 900, pad=140, top=60)
    await d.say(7)
    await basic_point(d, load, "unet_name")
    await d._after_voice(0.25)
    await basic_point(d, load, "clip_name")
    await d.say(8)
    await d.hold(2.0)
    await basic_point(d, load, "vae_name")

    # 4. planning with boxes
    await d._after_voice(0.25)
    await d.camera({"box": [880, -80, 1560, 1100]}, 1000, pad=60, top=60)
    editor = await ap_add(d, "HaDeSMossLayoutEditor", "MOSS Layout Editor", P["editor"])
    await d.js("""([id, v]) => { const w = window.__hv.widget(id, 'draft'); w.value = v; w.callback?.(v);
        window.__hv.node(id).setDirtyCanvas(true, true); }""", [editor, json.dumps(mx.FOLK_DRAFT, indent=2, ensure_ascii=False)])
    await d.hold(0.5)
    await d.camera([editor], 900, pad=60, top=60)
    await d.say(9)
    await d._after_voice(0.25)
    await d.say(10)
    await d._after_voice(0.25)
    await d.say(11)
    await d._after_voice(0.25)
    await d.say(12)
    await d._after_voice(1.0)
    await d.camera({"box": [380, -80, 1560, 1100]}, 900, pad=60, top=60)
    draft = await ap_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["draft"])
    await up_set(d, draft, {"mode": "layout", "width": 1024, "height": 1536, "temperature": 0.7, "seed": 2, "max_tokens": 1024,
                            "negative": "", "idea": mx.FOLK_IDEA})
    await d.say(13)
    async with d.fast(1.6):
        await basic_link(d, load, "CLIP", draft, "clip", 700)
        await basic_link(d, draft, "draft", editor, "draft_in", 700)
    await ml_zoom(d, draft, ["mode"])
    await basic_point(d, draft, "mode")
    await d.hold(1.5)
    z = await d.js("(id) => window.__hv.zone(id, 'draft_run')", editor)
    await d.camera([editor], 900, pad=60, top=60)
    if z:
        x, y = await d.screen(editor, z["x"] + z["w"] / 2, z["y"] + z["h"] / 2)
        await d.move(x, y)
    await d._after_voice(0.25)
    await d.js("(id) => { const n = window.__hv.node(id); n.mode = 2; window.__hv.app().graph.setDirtyCanvas(true, true); }", draft)
    await d.camera({"box": [380, -80, 1560, 1100]}, 900, pad=60, top=60)
    await d.say(14)

    # 5. four versions at once
    await d._after_voice(0.25)
    await d.camera({"box": [1480, -80, 2050, 1100]}, 1000, pad=60, top=60)
    comp = await ap_add(d, "HaDeSMossLayoutCompiler", "MOSS Layout Compiler", P["compiler"])
    await up_set(d, comp, {"batch_size": 1})
    async with d.fast(1.6):
        await d.camera({"box": [880, -80, 2050, 1100]}, 500, pad=60, top=60)
        await basic_link(d, editor, "draft", comp, "draft", 700)
    await d.camera([comp], 900, pad=120, top=60)
    await d.say(15)
    await d._after_voice(0.25)
    await ml_zoom(d, comp, ["batch_size"])
    await basic_point(d, comp, "batch_size")
    await d.hold(0.4)
    await up_set(d, comp, {"batch_size": 4})
    await d.say(16)
    await d.say(17)
    await d._after_voice(0.25)
    await d.camera({"box": [1480, -80, 2050, 1500]}, 900, pad=60, top=60)
    enh = await ap_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enh"])
    await up_set(d, enh, {"mode": "as written", "width": 1024, "height": 1024, "seed": 1, "negative": "", "max_tokens": 1024})
    async with d.fast(1.6):
        await d.camera({"box": [-60, -80, 2050, 1500]}, 500, pad=60, top=60)
        await basic_link(d, load, "CLIP", enh, "clip", 700)
        await d.camera({"box": [1480, -80, 2050, 1500]}, 500, pad=60, top=60)
        await basic_link(d, comp, "prompt", enh, "idea", 700)
    await ml_zoom(d, enh, ["mode"])
    await basic_point(d, enh, "mode")
    await d.say(18)

    # 6. generate
    await d._after_voice(0.25)
    await d.camera({"box": [1980, -80, 2460, 1100]}, 1000, pad=60, top=60)
    ks = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks"])
    await up_set(d, ks, {"seed": AP_SEED, "control_after_generate": "fixed", "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0})
    async with d.fast(3.0):
        await d.camera({"box": [-60, -80, 2460, 1500]}, 500, pad=60, top=60)
        await basic_link(d, load, "MODEL", ks, "model", 600)
        await basic_link(d, load, "VAE", ks, "vae", 600)
        await d.camera({"box": [1480, -80, 2460, 1500]}, 500, pad=60, top=60)
        for a, out, inp in ((enh, "positive", "positive"), (enh, "negative", "negative"), (comp, "latent", "latent_image")):
            await basic_link(d, a, out, ks, inp, 600)
    await ml_zoom(d, ks, ["steps", "cfg", "sampler_name", "scheduler", "seed"])
    await basic_point(d, ks, "steps")
    await d.say(19)
    await d.hold(2.5)
    await basic_point(d, ks, "seed")
    await d._after_voice(0.25)
    await d.camera({"box": [1980, -80, 2460, 1100]}, 700, pad=60, top=60)
    cand = await ap_add(d, "PreviewImage", "Preview Image", P["candidates"])
    await basic_link(d, ks, "image", cand, "images", 600)
    await d.camera([ks, cand], 900, pad=60, top=60)
    await d.say(20)

    # 7. let the machine read
    await d._after_voice(0.25)
    await d.camera({"box": [2430, -80, 2900, 900]}, 1000, pad=60, top=60)
    check = await ap_add(d, "HaDeSMossTextCheck", "MOSS Text Check", P["check"])
    await d.camera([check], 900, pad=120, top=60)
    await d.say(21)
    await d.say(22)
    await d._after_voice(0.25)
    await d.camera({"box": [-60, -80, 2900, 1500]}, 900, pad=60, top=60)
    await d.say(23)
    async with d.fast(2.0):
        await basic_link(d, load, "CLIP", check, "clip", 700)
        await basic_link(d, editor, "draft", check, "layout", 700)
        await d.camera({"box": [1980, -80, 2900, 1100]}, 500, pad=60, top=60)
        await basic_link(d, ks, "image", check, "image", 600)
    await d._after_voice(0.25)
    await d.camera([check], 900, pad=120, top=60)
    await d.say(24)
    await d._after_voice(0.25)
    await d.move(*await d.js("([id, n]) => window.__hv.slot(id, n, true)", [check, "best_image"]))
    await d.say(25)
    await d.hold(3.0)
    await d.move(*await d.js("([id, n]) => window.__hv.slot(id, n, true)", [check, "passed"]))
    await d._after_voice(0.25)
    await d.camera({"box": [2430, -80, 2900, 900]}, 700, pad=60, top=60)
    rep1 = await ap_add(d, "PreviewAny", "Preview as Text", P["check_report"])
    await basic_link(d, check, "report", rep1, "source", 600)
    await d.camera([check, rep1], 900, pad=60, top=60)
    await d.say(26)

    # 8. twice as big
    await d._after_voice(0.25)
    await d.camera({"box": [2880, -80, 3330, 700]}, 1000, pad=60, top=60)
    upm = await ap_add(d, "UpscaleModelLoader", "Load Upscale Model", P["upmodel"])
    await up_set(d, upm, {"model_name": "RealESRGAN_x2plus.pth"})
    up = await ap_add(d, "ImageUpscaleWithModel", "Upscale Image (using Model)", P["up"])
    async with d.fast(1.6):
        await basic_link(d, upm, "UPSCALE_MODEL", up, "upscale_model", 600)
        await d.camera({"box": [2430, -80, 3330, 900]}, 500, pad=60, top=60)
        await basic_link(d, check, "best_image", up, "image", 600)
    await d.camera([upm, up], 900, pad=80, top=60)
    await d.say(27)
    await d._after_voice(0.25)
    enc = await ap_add(d, "VAEEncode", "VAE Encode", P["enc"])
    ks2 = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks2"])
    await up_set(d, ks2, {"seed": AP_SEED, "control_after_generate": "fixed", "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "denoise": 0.25})
    await d.camera({"box": [2880, -80, 3800, 900]}, 900, pad=60, top=60)
    await d.say(28)
    async with d.fast(3.0):
        await basic_link(d, up, "IMAGE", enc, "pixels", 600)
        await basic_link(d, enc, "LATENT", ks2, "latent_image", 600)
        await d.camera({"box": [-60, -80, 3800, 1500]}, 500, pad=60, top=60)
        await basic_link(d, load, "MODEL", ks2, "model", 600)
        await basic_link(d, load, "VAE", ks2, "vae", 600)
        await basic_link(d, load, "VAE", enc, "vae", 600)
        await d.camera({"box": [1480, -80, 3800, 1500]}, 500, pad=60, top=60)
        await basic_link(d, enh, "positive", ks2, "positive", 600)
        await basic_link(d, enh, "negative", ks2, "negative", 600)
    await d._after_voice(0.25)
    await ml_zoom(d, ks2, ["denoise"])
    await basic_point(d, ks2, "denoise")
    await d.say(29)
    await d._after_voice(0.25)
    await d.camera({"box": [3730, -80, 4200, 900]}, 1000, pad=60, top=60)
    check2 = await ap_add(d, "HaDeSMossTextCheck", "MOSS Text Check", P["check2"])
    rep2 = await ap_add(d, "PreviewAny", "Preview as Text", P["check2_report"])
    await d.say(30)
    async with d.fast(2.0):
        await basic_link(d, check2, "report", rep2, "source", 600)
        await d.camera({"box": [3280, -80, 4200, 900]}, 500, pad=60, top=60)
        await basic_link(d, ks2, "image", check2, "image", 600)
        await d.camera({"box": [-60, -80, 4200, 1500]}, 500, pad=60, top=60)
        await basic_link(d, load, "CLIP", check2, "clip", 700)
        await basic_link(d, editor, "draft", check2, "layout", 700)

    # 9. saved with everything inside
    await d._after_voice(0.25)
    await d.camera({"box": [4180, -80, 4650, 1100]}, 1000, pad=60, top=60)
    tel = await ap_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["tel"])
    save = await ap_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["save"])
    await up_set(d, save, {"filename_prefix": "MOSS/HaDeS_MOSS_poster_%seed%", "hashes": True, "embed_workflow": True})
    await d.say(31)
    async with d.fast(3.0):
        await basic_link(d, tel, "json", save, "metadata", 600)
        await d.camera({"box": [3280, -80, 4650, 1100]}, 500, pad=60, top=60)
        await basic_link(d, ks2, "image", save, "images", 600)
        await d.camera({"box": [-60, -80, 4650, 1500]}, 500, pad=60, top=60)
        for k, (a, ) in enumerate(((load,), (editor,), (comp,), (enh,), (ks,), (check,), (ks2,), (check2,)), 1):
            await basic_link(d, a, "info", tel, f"info_{k}", 600)

    # 10. run and read the result
    await d._after_voice(0.25)
    await d.camera({"box": [-60, -80, 4650, 1500]}, 900, pad=60, top=60)
    await d.say(32)
    _, secs = await basic_run(d, save)
    ids = {"candidates": cand, "check_report": rep1, "check2_report": rep2, "save": save}
    cands, report, report2, final_take = await ap_take(d, ids)
    if report.count("3/3 exact") != 2 or "image 2: 3/3 exact <- best" not in report or report2.count("[exact]") != 3:
        raise SystemExit(f"the take's Text Check is not the measured one:\n{report}\n{report2}")
    await d.move(W * 0.97, H * 0.55)
    plain = [f'<div style="text-align:center"><img src="{u}" style="height:600px;border-radius:6px"></div>' for u in cands]
    await d.js("(h) => window.__hv.show(h)", f'<div style="display:flex;gap:30px;margin-top:40px">{"".join(plain)}</div>')
    await d.say(33)
    await d._after_voice(2.0)
    await d.js("(h) => window.__hv.show(h)", ap_candidates_html(cands, report, es))
    await d.say(34)
    await d.say(35)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_final_html(final_take, report2, es))
    await d.say(36)
    await d._after_voice(0.2)
    await d.js("(h) => window.__hv.show(h)", ap_crops_html(es))
    await d.say(37)
    await d._after_voice(0.3)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs:.1f} s · RTX 5090'], top=380))
    await d.say(38)

    # 11. measured, and a tip
    nums = (["12 / 12 pósters: todos los textos bien escritos", "3 / 12: el título en su caja", "×2: ningún veredicto cambió"] if es else
            ["12 / 12 posters: every text spelled right", "3 / 12: the title inside its box", "×2: no verdict changed"])
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(nums[:1]))
    await d.say(39)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(nums[:2]))
    await d.say(40)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(nums))
    await d.say(41)
    await d._after_voice(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera([editor], 900, pad=40, top=60)
    await d.say(42)

    # 12. recap and closing
    recap = (["1 · planifica con cajas", "2 · varias versiones, Text Check elige", "3 · ×2 y vuelve a comprobar"] if es else
             ["1 · plan with boxes", "2 · several versions, Text Check picks", "3 · ×2, and check again"])
    for k in range(3):
        await d._after_voice(0.25)
        await d.js("(h) => window.__hv.show(h)", ap_card(recap[:k + 1]))
        await d.say(43 + k)
    await d._after_voice(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": [-60, -80, 4650, 1500]}, 1200, pad=60, top=60)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1300px;margin-top:300px">'
               f'<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">{"Descarga gratis (enlace en la descripción)" if es else "Free download (link in the description)"}</div>'
               f'<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">examples/advanced/{AP_TITLE}.json</div></div>')
    await d.say(46)
    await d._after_voice(0.25)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               f'<div style="display:flex;align-items:center;gap:80px;margin-top:30px"><img src="{final_take}" style="height:820px;border-radius:6px">'
               '<div style="color:#fff;max-width:640px"><div style="font:900 72px Segoe UI Black,sans-serif;line-height:1.05">HaDeS MOSS Suite</div>'
               '<div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-top:22px">Production Poster</div>'
               f'<div style="font:400 26px Segoe UI,sans-serif;color:#c9c9d1;margin-top:44px">{"Workflow, Suite y modelos: enlaces en la descripción" if es else "Workflow, Suite and models: links in the description"}</div></div></div>')
    await d.say(47)
    await d.say(48)
    await d.end(1.5)


adv_poster.prepare = ap_prepare


# ---- advanced workflow 2: Merge Lab, as a lesson: built from an empty canvas (guion v2) ----

ML_FILE = None  # built on camera; the finished example is the download
ML_SEEDS = (1013, 1014)  # measured: key 1 = Block Merger, 2 = SFT, 3 = Base, 4 = SLERP on both (blind_seed 0); sheet ~61 s warm
ML_KEY = {"1": "C: merge (blocks 18-35 x0.5)", "2": "A: mossImage_sft_bf16", "3": "B: mossImage_base_bf16", "4": "D: merge (SLERP)"}
ML_NAMES = {"1": "Block Merger (18-35 ×0.5)", "2": "SFT", "3": "Base", "4": "SLERP"}
ML_PROMPT = ("Photograph of an old fisherman with a grey beard mending a net on a wooden dock at sunrise, his face and weathered hands "
             "in view, thick wool sweater, seagulls in the background.")
ML_POS = {"load": (0, 0), "base": (0, 420), "merge": (480, 0), "slerp": (480, 700), "enh": (960, 0), "latent": (1460, 0),
          "cmp": (1960, 0), "tel": (2460, 0), "saver": (1960, 1150), "meta": (2460, 1150)}
ML_BOX = [-60, -80, 2900, 1500]

ML_LINES = [
    ("Hi, I'm HaDeS, and welcome to advanced workflow number two: Merge Lab.",
     "Hola, soy HaDeS, y bienvenido al workflow avanzado número dos: Merge Lab."),
    ("Today you'll learn three things: how to mix two models in two different ways, how to judge the result without fooling yourself, and how to keep the winner as a small file.",
     "Hoy vas a aprender tres cosas: cómo mezclar dos modelos de dos formas distintas, cómo juzgar el resultado sin engañarte, y cómo quedarte con el ganador en un archivo pequeño."),
    ("We'll build the whole workflow together, from an empty canvas, one node at a time.",
     "Vamos a montar el workflow entero juntos, desde el lienzo vacío, nodo a nodo."),
    ("As always, everything is free, and all the links are in the description.",
     "Como siempre, todo es gratis y tienes todos los enlaces en la descripción."),
    ("First, what is a merge? A model is millions of numbers, its weights. A merge takes the weights of two models and mixes them into a new one.",
     "Primero, ¿qué es un merge? Un modelo son millones de números, sus pesos. Un merge toma los pesos de dos modelos y los mezcla en uno nuevo."),
    ("That only works when both models have the same shape. Here we use the two official MOSS-Image models: Base, and SFT, which is Base trained further. Same shape, so we can mix them.",
     "Eso solo funciona si los dos modelos tienen la misma forma. Aquí usamos los dos modelos oficiales de MOSS-Image: Base, y SFT, que es Base entrenado un poco más. La misma forma, así que se pueden mezclar.",
     "That only works when both models have the same shape. Here we use the two official MOSS Image models: Base, and S F T, which is Base trained further. Same shape, so we can mix them."),
    ("And one more idea: MOSS-Image is built as a stack of forty-two layers. We can mix every layer the same way, or decide layer by layer. Today we'll do both.",
     "Y una idea más: MOSS-Image está hecho como una pila de cuarenta y dos capas. Podemos mezclar todas las capas igual, o decidir capa a capa. Hoy hacemos las dos cosas.",
     "And one more idea: MOSS Image is built as a stack of forty-two layers. We can mix every layer the same way, or decide layer by layer. Today we'll do both."),
    ("Let's start. The MOSS Loader loads SFT, together with the text encoder, which reads the prompt, and the VAE, which turns the result into pixels.",
     "Empezamos. El MOSS Loader carga SFT, junto con el text encoder, que lee el prompt, y el VAE, que convierte el resultado en píxeles.",
     "Let's start. The MOSS Loader loads S F T, together with the text encoder, which reads the prompt, and the V A E, which turns the result into pixels."),
    ("For Base we only need the model itself, because the text encoder and the VAE are shared. So a plain Load Diffusion Model is enough.",
     "Para Base solo necesitamos el modelo, porque el text encoder y el VAE son compartidos. Así que basta con un Load Diffusion Model normal.",
     "For Base we only need the model itself, because the text encoder and the V A E are shared. So a plain Load Diffusion Model is enough."),
    ("The first way to mix them is the MOSS Model Block Merger. Model one is SFT, model two is Base.",
     "La primera forma de mezclarlos es el MOSS Model Block Merger. El modelo uno es SFT, el modelo dos es Base.",
     "The first way to mix them is the MOSS Model Block Merger. Model one is S F T, model two is Base."),
    ("Each slider is a group of six layers. What does the number mean? Zero keeps model one, SFT. One takes model two, Base. And zero point five is half and half.",
     "Cada deslizador es un grupo de seis capas. ¿Qué significa el número? Cero se queda con el modelo uno, SFT. Uno toma el modelo dos, Base. Y cero coma cinco es mitad y mitad.",
     "Each slider is a group of six layers. What does the number mean? Zero keeps model one, S F T. One takes model two, Base. And zero point five is half and half."),
    ("Here I set the middle groups, layers eighteen to thirty-five, to zero point five, and leave everything else at zero.",
     "Aquí pongo los grupos del medio, las capas dieciocho a treinta y cinco, a cero coma cinco, y dejo todo lo demás a cero."),
    ("So this merge is SFT, with half of Base only in the middle of the stack. It's the test recipe from my sixth video. Not a recommendation: just a starting point to experiment with.",
     "Así que este merge es SFT, con la mitad de Base solo en el medio de la pila. Es la receta de prueba de mi sexto vídeo. No es una recomendación: solo un punto de partida para experimentar.",
     "So this merge is S F T, with half of Base only in the middle of the stack. It's the test recipe from my sixth video. Not a recommendation: just a starting point to experiment with."),
    ("The second way is the MOSS Multi-Model Merger. It mixes every layer at once, with a method you choose.",
     "La segunda forma es el MOSS Multi-Model Merger. Mezcla todas las capas a la vez, con el método que elijas."),
    ("This time we use SLERP. What is it? It blends the two models along a curve instead of a straight line, a common way to mix exactly two models.",
     "Esta vez usamos SLERP. ¿Qué es? Mezcla los dos modelos siguiendo una curva en lugar de una línea recta, una forma habitual de mezclar exactamente dos modelos.",
     "This time we use S lerp. What is it? It blends the two models along a curve instead of a straight line, a common way to mix exactly two models."),
    ("With both weights at one, it goes halfway between SFT and Base.",
     "Con los dos pesos a uno, se queda a medio camino entre SFT y Base.",
     "With both weights at one, it goes halfway between S F T and Base."),
    ("Now the prompt: our fisherman from the earlier videos, in the Prompt Enhancer, written as is.",
     "Ahora el prompt: nuestro pescador de los vídeos anteriores, en el Prompt Enhancer, tal cual está escrito."),
    ("And the MOSS Latent: a square image, and a fixed seed, one zero one three, so every model starts from exactly the same noise.",
     "Y el MOSS Latent: una imagen cuadrada y una semilla fija, mil trece, para que todos los modelos partan exactamente del mismo ruido.",
     "And the MOSS Latent: a square image, and a fixed seed, ten thirteen, so every model starts from exactly the same noise."),
    ("Now the heart of this workflow: the MOSS Compare Sampler. It runs up to four models with the same prompt, the same seed and the same settings, and puts the images side by side on one sheet.",
     "Ahora el corazón del workflow: el MOSS Compare Sampler. Ejecuta hasta cuatro modelos con el mismo prompt, la misma semilla y los mismos ajustes, y pone las imágenes una al lado de otra en una hoja."),
    ("A is SFT, B is Base, C is our Block Merger, and D is SLERP. The only difference between the four images will be the model.",
     "La A es SFT, la B es Base, la C nuestro Block Merger y la D el SLERP. Lo único que cambiará entre las cuatro imágenes es el modelo.",
     "A is S F T, B is Base, C is our Block Merger, and D is S lerp. The only difference between the four images will be the model."),
    ("And this time we turn on blind mode. What is it for?",
     "Y esta vez activamos el modo blind, a ciegas. ¿Para qué sirve?"),
    ("When you've built a merge yourself, you want it to win. If you see its name, you'll find reasons to like it. That's not a judgement, that's hope.",
     "Cuando has hecho un merge tú mismo, quieres que gane. Si ves su nombre, encontrarás motivos para que te guste. Eso no es juzgar, es tener esperanza."),
    ("In blind mode, the Compare Sampler shuffles the four images and numbers them from one to four. No names on the sheet, and none hidden inside the image file either.",
     "En modo blind, el Compare Sampler baraja las cuatro imágenes y las numera del uno al cuatro. Sin nombres en la hoja, y tampoco escondidos dentro del archivo de imagen."),
    ("The answer, which number is which model, goes to a separate file next to the sheet: the key. You open it only after you've chosen.",
     "La respuesta, qué número es cada modelo, va a un archivo aparte, al lado de la hoja: la clave. Solo la abres después de haber elegido."),
    ("Blind seed decides the order of the shuffle. Keep it, and the order stays the same; change it, and you get a new order.",
     "Blind seed decide el orden de la mezcla. Si lo dejas igual, el orden se mantiene; si lo cambias, tienes un orden nuevo."),
    ("And the file name says where it's saved: in ComfyUI's output folder, under MOSS, merge lab.",
     "Y el nombre de archivo dice dónde se guarda: en la carpeta output de ComfyUI, dentro de MOSS, merge lab."),
    ("One more detail. Suite Telemetry reads the loader, the prompt and both mergers, but not the Compare Sampler. That way, it can't give the key away.",
     "Un detalle más. Suite Telemetry lee el loader, el prompt y los dos mergers, pero no el Compare Sampler. Así no puede delatar la clave."),
    ("Let's run it.",
     "Vamos a ejecutarlo."),
    ("Here are the four fishermen, numbered one to four. Take your time. Pause the video, and pick your favourite.",
     "Aquí están los cuatro pescadores, numerados del uno al cuatro. Tómate tu tiempo. Pausa el vídeo y elige tu favorito."),
    ("Got one? Now let's open the key. It's this file, next to the sheet, with the same name and key at the end.",
     "¿Ya lo tienes? Ahora abrimos la clave. Es este archivo, al lado de la hoja, con el mismo nombre y key al final."),
    ("Inside, blind key says: one is C, two is A, three is B, four is D. The letters are the Compare Sampler's inputs.",
     "Dentro, blind key dice: el uno es la C, el dos la A, el tres la B, el cuatro la D. Las letras son las entradas del Compare Sampler."),
    ("So one is the Block Merger, two is SFT, three is Base, and four is SLERP. Did you pick a merge, or one of the parents?",
     "Así que el uno es el Block Merger, el dos es SFT, el tres es Base y el cuatro es SLERP. ¿Elegiste un merge, o a uno de los padres?",
     "So one is the Block Merger, two is S F T, three is Base, and four is S lerp. Did you pick a merge, or one of the parents?"),
    ("By the way, the sheet took about a minute on my RTX 5090: four models, twenty-five steps each.",
     "Por cierto, la hoja tardó como un minuto en mi RTX 5090: cuatro modelos, veinticinco pasos cada uno.",
     "By the way, the sheet took about a minute on my R T X fifty ninety: four models, twenty-five steps each."),
    ("Look at one and four: with this seed, the two merges are nearly the same picture. Does that mean the two recipes are equal? Let's change the seed and see.",
     "Mira el uno y el cuatro: con esta semilla, los dos merges son casi la misma imagen. ¿Quiere decir que las dos recetas son iguales? Cambiemos la semilla y veamos."),
    ("With seed one zero one four, they're clearly different: the SLERP fisherman even has glasses. The order is the same as before, because blind seed didn't change.",
     "Con la semilla mil catorce son claramente distintos: el pescador de SLERP hasta lleva gafas. El orden es el mismo que antes, porque blind seed no ha cambiado.",
     "With seed ten fourteen, they're clearly different: the S lerp fisherman even has glasses. The order is the same as before, because blind seed didn't change."),
    ("So here's the lesson: never judge a merge on one seed. Try a few before you choose.",
     "Así que la lección es: nunca juzgues un merge con una sola semilla. Prueba unas cuantas antes de elegir."),
    ("Once you have a winner, you can keep it. Add a MOSS Model Saver, and connect the winning merger to model and to recipe.",
     "Cuando tengas un ganador, puedes quedártelo. Añade un MOSS Model Saver y conecta el merger ganador a model y a recipe."),
    ("As precision, choose HMS v1: the Suite's own compressed format. It calibrates on the merge itself, and writes the recipe into the file.",
     "Como precisión, elige HMS v1: el formato comprimido de la propia Suite. Se calibra con el propio merge y escribe la receta en el archivo.",
     "As precision, choose H M S version one: the Suite's own compressed format. It calibrates on the merge itself, and writes the recipe into the file."),
    ("For our test recipe it took under eight minutes, and gave a 3.96 gigabyte file that makes an image in 6.1 seconds, instead of 13.3 for the live merge.",
     "Con nuestra receta de prueba tardó menos de ocho minutos, y dio un archivo de 3,96 gigas que hace una imagen en 6,1 segundos, en vez de 13,3 con el merge en vivo.",
     "For our test recipe it took under eight minutes, and gave a three point nine six gigabyte file that makes an image in six point one seconds, instead of thirteen point three for the live merge."),
    ("But check it before you trust it: against the live merge it kept the scene, and on one seed the fisherman himself changed. Compare them with a Compare Sampler, just like today.",
     "Pero compruébalo antes de fiarte: frente al merge en vivo mantuvo la escena, y en una semilla cambió el propio pescador. Compáralos con un Compare Sampler, igual que hoy."),
    ("I leave the saver muted in the workflow, so it only runs when you decide. And the Metadata Reader can read the recipe back from the file.",
     "En el workflow dejo el saver desactivado, para que solo se ejecute cuando tú lo decidas. Y el Metadata Reader puede volver a leer la receta desde el archivo."),
    ("Let's recap. One: two ways to mix, layer by layer with the Block Merger, or all at once with SLERP.",
     "Repasemos. Uno: dos formas de mezclar, capa a capa con el Block Merger, o todo a la vez con SLERP.",
     "Let's recap. One: two ways to mix, layer by layer with the Block Merger, or all at once with S lerp."),
    ("Two: judge them blind, against their parents, on more than one seed.",
     "Dos: júzgalos a ciegas, frente a sus padres, con más de una semilla."),
    ("Three: keep the winner as an HMS v1 file, and compare it with the live merge.",
     "Tres: guarda al ganador en un archivo HMS v1, y compáralo con el merge en vivo.",
     "Three: keep the winner as an H M S version one file, and compare it with the live merge."),
    ("You can download this workflow for free from GitHub. The link is in the description.",
     "Puedes descargar este workflow gratis desde GitHub. El enlace está en la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]


def ml_sheet_html(url, labels=None, frames=(), width=1800, caption=None, top=60):
    """A Compare Sampler sheet (4144 x 1115: four 1024 panels 1040 apart under a 44 px number row); labels under each panel;
    yellow frames on the panels named in frames."""
    sx = width / 4144
    h = 1115 * sx
    boxes = "".join(f'<div style="position:absolute;left:{(int(i) - 1) * 1040 * sx - 4}px;top:{44 * sx - 4}px;width:{1024 * sx + 8}px;'
                    f'height:{1024 * sx + 8}px;border:5px solid #ffcc33;border-radius:4px;box-sizing:border-box"></div>' for i in frames)
    under = ""
    if labels:
        under = '<div style="display:flex;margin-top:14px">' + "".join(
            f'<div style="width:{1040 * sx}px;text-align:center;font:700 30px Segoe UI,sans-serif;color:{c}">{t}</div>' for t, c in labels) + "</div>"
    cap = f'<div style="font:600 30px Segoe UI,sans-serif;color:#ffcc33;margin-bottom:12px">{caption}</div>' if caption else ""
    return (f'<div style="margin-top:{top}px;color:#fff;text-align:center">{cap}<div style="position:relative;width:{width}px;height:{h}px;margin:auto">'
            f'<img src="{url}" style="width:{width}px;height:{h}px;border-radius:6px">{boxes}</div>{under}</div>')


def ml_merge_card(stage, es):
    """Stage 0: two models mixed into one; stage 1: the 42 layers in 7 groups of 6, the middle three (18-35) marked."""
    if stage == 0:
        box = lambda t, c: (f'<div style="width:300px;height:170px;border-radius:16px;background:{c};display:flex;align-items:center;'
                            f'justify-content:center;font:800 52px Segoe UI,sans-serif">{t}</div>')
        arrow = '<div style="font:800 70px Segoe UI,sans-serif;color:#c9c9d1">+</div>'
        return (f'<div style="color:#fff;text-align:center;margin-top:230px"><div style="display:flex;gap:40px;align-items:center;justify-content:center">'
                f'{box("SFT", "#2e4a7a")}{arrow}{box("Base", "#5a3a6e")}<div style="font:800 70px Segoe UI,sans-serif;color:#ffcc33">→</div>'
                f'{box("merge", "#6b5a1e")}</div><div style="font:600 34px Segoe UI,sans-serif;color:#c9c9d1;margin-top:50px">'
                f'{"millones de pesos, mezclados" if es else "millions of weights, mixed"}</div></div>')
    rows = "".join(f'<div style="width:520px;height:62px;margin:8px auto;border-radius:10px;display:flex;align-items:center;justify-content:center;'
                   f'font:700 28px Segoe UI,sans-serif;background:{"#6b5a1e" if 3 <= g <= 5 else "#2a2a33"}">'
                   f'{"capas" if es else "layers"} {g * 6}-{g * 6 + 5}</div>' for g in range(6, -1, -1))
    return (f'<div style="color:#fff;text-align:center;margin-top:60px"><div style="font:600 34px Segoe UI,sans-serif;color:#ffcc33;margin-bottom:20px">'
            f'MOSS-Image: {"42 capas" if es else "42 layers"}</div>{rows}</div>')


def ml_key_html(key_json, sheet_name, es, step):
    """The output folder (step 0) and the key file's content with blind_key marked (step 1)."""
    if step == 0:
        files = (sheet_name, sheet_name.replace(".png", "_key.json"))
        rows = "".join(f'<div style="font:600 34px Consolas,monospace;padding:14px 22px;border-radius:8px;margin:8px 0;'
                       f'background:{"#3a3214" if f.endswith(".json") else "#1d1d24"};color:{"#ffcc33" if f.endswith(".json") else "#e6e6ea"}">{f}</div>' for f in files)
        return (f'<div style="color:#fff;margin-top:220px;width:1100px"><div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1;margin-bottom:16px">'
                f'ComfyUI / output / MOSS / merge_lab</div>{rows}</div>')
    k = key_json.get("blind_key", {})
    lines = ['{', '  "labels": [ ... ],', '  "footer": "seed ... | 25 steps | CFG 2.5 ...",', '  "blind_key": {']
    lines += [f'    "{n}": "{v}"{"," if i < len(k) - 1 else ""}' for i, (n, v) in enumerate(k.items())]
    lines += ['  },', '  ...', '}']
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = "".join(f'<div style="white-space:pre;color:{"#ffcc33" if (2 < i < 5 + len(k)) else "#c9c9d1"}">{esc(l)}</div>'
                   for i, l in enumerate(lines))
    return (f'<div style="margin-top:160px;width:1500px;text-align:left;background:#15151b;border-radius:12px;padding:30px 40px;'
            f'font:500 30px Consolas,monospace;line-height:1.5">{body}</div>')


async def ml_zoom(d, node, names, ms=900, pad=40):
    """Camera on some widgets of a node (their rows, the node's width)."""
    box = await d.js("""([id, names]) => { const n = window.__hv.node(id);
        const ws = n.widgets.filter(w => names.some(m => w.name === m || w.name.endsWith('_' + m)));
        if (!ws.length) return null;
        const ys = ws.map(w => w.last_y ?? w.y ?? 0), y1 = Math.min(...ys) - 12, y2 = Math.max(...ys) + 40;
        return [n.pos[0] - 10, n.pos[1] + y1, n.pos[0] + n.size[0] + 10, n.pos[1] + y2]; }""", [node, names])
    if box:
        await d.camera({"box": box}, ms, pad=pad, top=60)


async def ml_prepare(page):
    """Off camera: the finished example once with another seed, so the models are loaded and the on-camera sheets are warm."""
    import copy
    import urllib.parse
    import make_examples as mx
    base = urllib.parse.urlsplit(page.url)
    info = mx.fetch(f"{base.scheme}://{base.netloc}", "/object_info")
    g = {k: v for k, v in copy.deepcopy(mx.EXAMPLES["HaDeS MOSS Suite - Merge Lab"]).items() if k != "note"}
    g["latent"][1]["seed"] = 1015
    g["cmp"][1]["filename_prefix"] = "MOSS/warmup_merge_lab"
    local = {mx.SFT: "mossImage_sft_bf16.safetensors", mx.BASE: "mossImage_base_bf16.safetensors"}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, mx.to_api(g, info, local)), flush=True)


async def merge_lab(d: Director):
    """Advanced workflow #2 as a lesson (guion v2): concepts, the workflow built node by node with zooms, a blind sheet with a pause,
    the key opened from its file, a second seed on camera, the winner kept, a recap."""
    P = ML_POS
    es = d.language == "es"
    await d.camera({"box": ML_BOX}, 10, pad=60, top=60)
    await d.move(W * 0.97, H * 0.55, 10)
    measured = "/view?filename=blind_s1013_00001_.png&subfolder=adv_merge_lab&type=output"

    # 1. what we'll learn
    await d.js("(h) => window.__hv.show(h, 10)",
               ml_sheet_html(measured, width=1700, top=80) +
               '<div style="position:fixed;left:0;right:0;bottom:120px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:600 36px Segoe UI,sans-serif;color:#ffcc33;margin-top:14px">{"Workflow avanzado n.º 2" if es else "Advanced workflow #2"}'
               ' · Merge Lab</div></div>')
    await d.hold(0.8)
    for i in range(4):
        await d.say(i)

    # 2. what a merge is
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ml_merge_card(0, es))
    await d.say(4)
    await d.say(5)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ml_merge_card(1, es))
    await d.say(6)
    await d._after_voice(1.0)
    await d.js("() => window.__hv.hide()")

    # 3. the two models
    await d.camera({"box": [-60, -80, 900, 900]}, 1000, pad=60, top=60)
    load = await basic_add(d, "HaDeSMossLoader", "MOSS Loader", P["load"])
    await up_set(d, load, dict(BASIC_LOADER, unet_name="mossImage_sft_bf16.safetensors"))
    await d.camera([load], 900, pad=140, top=60)
    await d.say(7)
    await basic_point(d, load, "unet_name")
    await d.hold(1.2)
    await basic_point(d, load, "clip_name")
    await d.hold(1.2)
    await basic_point(d, load, "vae_name")
    base = await basic_add(d, "UNETLoader", "Load Diffusion Model", P["base"])
    await up_set(d, base, {"unet_name": "mossImage_base_bf16.safetensors", "weight_dtype": "default"})
    await d.camera([load, base], 900, pad=100, top=60)
    await d.say(8)
    await basic_point(d, base, "unet_name")

    # 4. the Block Merger
    await d._after_voice(0.25)
    await d.camera({"box": [-60, -80, 1400, 1000]}, 1000, pad=60, top=60)
    merge = await basic_add(d, "HaDeSMossModelBlockMerger", "MOSS Model Block Merger", P["merge"])
    zero = {g: 0.0 for g in ("layers_00_05", "layers_06_11", "layers_12_17", "layers_18_23", "layers_24_29", "layers_30_35", "layers_36_41",
                             "text_embedder", "time_modulation", "io")}
    await up_set(d, merge, zero)
    await d.say(9)
    async with d.fast(1.6):
        await basic_link(d, load, "MODEL", merge, "model1", 700)
        await basic_link(d, base, "MODEL", merge, "model2", 700)
    await d._after_voice(0.25)
    await ml_zoom(d, merge, ["layers_00_05", "layers_36_41"])
    await d.say(10)
    await d._after_voice(0.3)
    await d.say(11)
    for g in ("layers_18_23", "layers_24_29", "layers_30_35"):
        await basic_point(d, merge, g)
        await d.hold(0.5)
        await up_set(d, merge, {g: 0.5})
        await d.hold(0.6)
    await d._after_voice(0.25)
    await d.camera([merge], 900, pad=60, top=60)
    await d.say(12)
    await d._after_voice(1.0)

    # 5. the Multi-Model Merger
    await d.camera({"box": [380, 500, 1400, 1500]}, 1000, pad=60, top=60)
    slerp = await basic_add(d, "HaDeSMossMultiModelMerger", "MOSS Multi-Model Merger", P["slerp"])
    await up_set(d, slerp, {"method": "SLERP", "weight_1": 1.0, "weight_2": 1.0})
    async with d.fast(1.6):
        await d.camera({"box": [-60, -80, 1400, 1500]}, 600, pad=60, top=60)
        await basic_link(d, load, "MODEL", slerp, "model_1", 700)
        await basic_link(d, base, "MODEL", slerp, "model_2", 700)
    await d.camera([slerp], 900, pad=60, top=60)
    await d.say(13)
    await d._after_voice(0.25)
    await ml_zoom(d, slerp, ["method"])
    await basic_point(d, slerp, "method")
    await d.say(14)
    await d._after_voice(0.25)
    await ml_zoom(d, slerp, ["weight_1", "weight_2"])
    await basic_point(d, slerp, "weight_1")
    await d.say(15)

    # 6. the prompt and the seed
    await d._after_voice(0.25)
    await d.camera({"box": [880, -80, 1960, 1000]}, 1000, pad=60, top=60)
    enh = await basic_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enh"])
    await up_set(d, enh, {"mode": "as written", "width": 1024, "height": 1024, "seed": 1, "negative": ""})
    await d.say(16)
    async with d.fast(2.0):
        await basic_link(d, load, "CLIP", enh, "clip", 700)
        await pe_type(d, enh, "idea", ML_PROMPT, paste=True)
    latent = await basic_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": ML_SEEDS[0], "control_after_generate": "fixed"})
    await d._after_voice(0.25)
    await ml_zoom(d, latent, ["seed"])
    await basic_point(d, latent, "seed")
    await d.say(17)

    # 7. the blind comparison
    await d._after_voice(0.25)
    await d.camera({"box": [1880, -80, 2500, 1050]}, 1000, pad=60, top=60)
    cmp = await basic_add(d, "HaDeSMossCompareSampler", "MOSS Compare Sampler", P["cmp"])
    await up_set(d, cmp, {"steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "save": True, "embed_workflow": True,
                          "filename_prefix": "MOSS/merge_lab/HaDeS_MOSS_blind", "blind": False, "blind_seed": 0})
    await d.say(18)
    await d._after_voice(0.25)
    await d.camera({"box": ML_BOX}, 900, pad=60, top=60)
    await d.say(19)
    async with d.fast(4.0):
        for a, out, inp in ((load, "MODEL", "model_A"), (base, "MODEL", "model_B"), (merge, "MODEL", "model_C"), (slerp, "MODEL", "model_D"),
                            (load, "VAE", "vae")):
            await basic_link(d, a, out, cmp, inp, 600)
        await d.camera({"box": [880, -80, 2500, 1100]}, 500, pad=60, top=60)  # closer: the small slots need it
        for a, out, inp in ((enh, "positive", "positive"), (enh, "negative", "negative"), (latent, "latent", "latent_image"),
                            (latent, "seed", "seed"), (enh, "info", "prompt_info")):
            await basic_link(d, a, out, cmp, inp, 600)
    await d._after_voice(0.25)
    await ml_zoom(d, cmp, ["blind"])
    await basic_point(d, cmp, "blind")
    await d.say(20)
    await d.hold(0.6)
    await up_set(d, cmp, {"blind": True})
    await d.say(21)
    await d.say(22)
    await d.say(23)
    await d._after_voice(0.25)
    await ml_zoom(d, cmp, ["blind_seed"])
    await basic_point(d, cmp, "blind_seed")
    await d.say(24)
    await d._after_voice(0.25)
    await ml_zoom(d, cmp, ["filename_prefix"])
    await basic_point(d, cmp, "filename_prefix")
    await d.say(25)
    await d._after_voice(0.25)
    await d.camera({"box": ML_BOX}, 900, pad=60, top=60)
    tel = await basic_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["tel"])
    await d.say(26)
    async with d.fast(2.0):
        for a, i in ((load, 1), (enh, 2), (merge, 3), (slerp, 4), (latent, 5)):
            await basic_link(d, a, "info", tel, f"info_{i}", 600)

    # 8. blind
    await d._after_voice(0.25)
    await d.camera({"box": ML_BOX}, 900, pad=60, top=60)
    await d.say(27)
    sheet, secs = await basic_run(d, cmp)
    key_url = sheet.replace(".png&", "_key.json&", 1)
    key = await d.js("(u) => fetch(u).then(r => r.json())", key_url)
    if any(not key.get("blind_key", {}).get(k, "").startswith(v) for k, v in ML_KEY.items()):
        raise SystemExit(f"the take's key is not the measured one: {key.get('blind_key')}")
    sheet_name = sheet.split("filename=")[1].split("&")[0]
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", ml_sheet_html(sheet, top=180))
    await d.say(28)
    await d._after_voice(0.1)
    await d.js("(h) => window.__hv.show(h)", ml_sheet_html(sheet, top=180) +
               f'<div style="position:fixed;left:0;right:0;bottom:150px;text-align:center;font:700 44px Segoe UI,sans-serif;color:#ffcc33">'
               f'⏸ {"Pausa y elige" if es else "Pause and pick"}</div>')
    await d.hold(4.0)
    await d.js("(h) => window.__hv.show(h)", ml_key_html(key, sheet_name, es, 0))
    await d.say(29)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ml_key_html(key, sheet_name, es, 1))
    await d.say(30)
    await d._after_voice(0.25)
    names = [(f'{k} · {ML_NAMES[k]}', "#ffcc33" if k in ("1", "4") else "#ffffff") for k in ("1", "2", "3", "4")]
    await d.js("(h) => window.__hv.show(h)", ml_sheet_html(sheet, labels=names, top=180))
    await d.say(31)
    await d._after_voice(1.5)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs:.1f} s · RTX 5090', "4 × 25 steps"], top=360))
    await d.say(32)

    # 9. one seed is not enough
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ml_sheet_html(sheet, frames=("1", "4"), top=180))
    await d.say(33)
    await d._after_voice(0.25)
    await d.js("() => window.__hv.hide()")
    await ml_zoom(d, latent, ["seed"])
    await basic_point(d, latent, "seed")
    await d.hold(0.5)
    await up_set(d, latent, {"seed": ML_SEEDS[1]})
    await d.hold(0.8)
    await d.camera({"box": ML_BOX}, 900, pad=60, top=60)
    sheet2, secs2 = await basic_run(d, cmp)
    key2 = await d.js("(u) => fetch(u).then(r => r.json())", sheet2.replace(".png&", "_key.json&", 1))
    if key2.get("blind_key") != key.get("blind_key"):
        raise SystemExit(f"the second sheet's order changed: {key2.get('blind_key')}")
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", '<div style="display:flex;flex-direction:column;gap:18px">' +
               ml_sheet_html(sheet, frames=("1", "4"), width=1500, top=10, caption=("Semilla 1013" if es else "Seed 1013")) +
               ml_sheet_html(sheet2, frames=("1", "4"), width=1500, top=10, caption=("Semilla 1014" if es else "Seed 1014")) + "</div>")
    await d.say(34)
    await d.say(35)
    await d._after_voice(1.0)

    # 10. keep the winner
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": [1880, 1050, 2900, 1900]}, 1000, pad=60, top=60)
    saver = await basic_add(d, "HaDeSMossModelSaver", "MOSS Model Saver", P["saver"])
    await up_set(d, saver, {"precision": "HMS v1 (W4A8 / W6A8, calibrated)", "filename_prefix": "diffusion_models/MOSS_merge_HMS_v1",
                            "include_metadata": True, "custom_author": "HaDeS"})
    await d.say(36)
    async with d.fast(2.0):
        await d.camera({"box": [-60, -80, 2900, 1900]}, 600, pad=60, top=60)
        await basic_link(d, merge, "MODEL", saver, "model", 700)
        await basic_link(d, merge, "info", saver, "recipe", 700)
        await basic_link(d, load, "CLIP", saver, "clip", 700)
    await d._after_voice(0.25)
    await ml_zoom(d, saver, ["precision"])
    await basic_point(d, saver, "precision")
    await d.say(37)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(["HMS v1 · 7.6 min · 3.96 GB", "6.1 s / image (13.3 s live)"]))
    await d.say(38)
    await d.say(39)
    await d._after_voice(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": [1880, 1050, 2900, 1900]}, 900, pad=60, top=60)
    meta = await basic_add(d, "HaDeSMossMetadataReader", "MOSS Metadata Reader", P["meta"])
    await d.js("(ids) => { for (const id of ids) window.__hv.node(id).mode = 2; window.__hv.app().graph.setDirtyCanvas(true, true); }", [saver, meta])
    await d.say(40)

    # 11. recap and closing
    await d._after_voice(0.25)
    recap = (["1 · capa a capa, o todo a la vez", "2 · a ciegas, con varias semillas", "3 · HMS v1, y compáralo"] if es else
             ["1 · layer by layer, or all at once", "2 · blind, on several seeds", "3 · HMS v1, and compare it"])
    await d.js("(h) => window.__hv.show(h)", ap_card(recap[:1]))
    await d.say(41)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(recap[:2]))
    await d.say(42)
    await d._after_voice(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(recap))
    await d.say(43)
    await d._after_voice(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": [-60, -80, 2900, 1900]}, 1200, pad=60, top=60)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1300px;margin-top:300px">'
               f'<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">{"Descarga gratis (enlace en la descripción)" if es else "Free download (link in the description)"}</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">examples/advanced/HaDeS MOSS Suite - Merge Lab.json</div></div>')
    await d.say(44)
    await d._after_voice(0.25)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               ml_sheet_html(sheet, labels=names, width=1700, top=80) +
               '<div style="position:fixed;left:0;right:0;bottom:120px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:400 28px Segoe UI,sans-serif;color:#c9c9d1;margin-top:18px">{"Workflow, Suite y modelos: enlaces en la descripción" if es else "Workflow, Suite and models: links in the description"}</div></div>')
    await d.say(45)
    await d.say(46)
    await d.end(1.5)


merge_lab.prepare = ml_prepare


TS_FILE = None  # built on camera (a lesson); the finished example is the download
TS_SEED = 1000  # the example's seed; measured: 0.3 ArcFace 0.878 vs stage 1, 0.7 0.682, the 0.7 rerun 3.0 s (videos/advanced/adv_two_stage/MEDIDAS.md)
TS_SHOTS = "adv_two_stage"  # the measured run of the same seed in comfy_output, for the opening and the crops
TS_CROP = (307, 212)  # 512 x 512 around the face (detector box on stage 1)
TS_PROMPT = ("Studio photograph of an elderly fisherman with a weathered face and a grey knitted cap, soft window light from the left, "
             "shallow depth of field, 85mm lens, muted colors.")
TS_LORA_BASE = "MOSS\\turbo_v10\\HaDeS-MOSS-Turbo-8steps-v1.0.safetensors"
TS_LORA_SFT = "MOSS\\HaDeS-MOSS-Turbo-8steps-v0.7.safetensors"
TS_POS = {"load": (0, 0), "sft": (0, 380), "enh": (450, 0), "latent": (950, 0), "t1": (950, 450), "ks": (1400, 0), "stage1": (1400, 560),
          "t2": (1850, 450), "ks2": (2300, 0), "save": (2750, 0), "tel": (2750, 480), "report": (3200, 480)}
TS_BOX = [-60, -120, 3640, 1050]

TS_LINES = [
    ("Hi, I'm HaDeS, and welcome to advanced workflow number three: two stages, two models.",
     "Hola, soy HaDeS, y bienvenido al workflow avanzado número tres: dos etapas, dos modelos."),
    ("Today you'll learn how to use two different models on the same image: one to compose the picture, and another one to finish it.",
     "Hoy vas a aprender a usar dos modelos distintos en la misma imagen: uno para componerla y otro para terminarla."),
    ("You'll also learn what denoise really does, and how to choose it with numbers instead of guessing.",
     "También vas a aprender qué hace de verdad el denoise, y cómo elegirlo con números en vez de a ojo.",
     "You'll also learn what denoise really does, and how to choose it with numbers instead of guessing."),
    ("As always, we'll build it together from an empty canvas, everything is free, and all the links are in the description.",
     "Como siempre, lo montamos juntos desde el lienzo vacío, todo es gratis y tienes todos los enlaces en la descripción."),
    ("MOSS-Image comes in two versions: Base and SFT. SFT is Base trained further, to make finished, polished images.",
     "MOSS-Image viene en dos versiones: Base y SFT. SFT es Base entrenado un poco más, para dar imágenes acabadas y pulidas.",
     "MOSS Image comes in two versions: Base and S F T. S F T is Base trained further, to make finished, polished images."),
    ("So why not always use SFT? Because each one is good at something different.",
     "Entonces, ¿por qué no usar siempre SFT? Porque cada uno es bueno en una cosa distinta.",
     "So why not always use S F T? Because each one is good at something different."),
    ("SFT gives skin texture: pores, wrinkles, a little more warmth. But on a neutral prompt it tends to draw the same kind of face again and again.",
     "SFT da textura de piel: poros, arrugas, algo más de calidez. Pero con un prompt neutro tiende a dibujar el mismo tipo de cara una y otra vez.",
     "S F T gives skin texture: pores, wrinkles, a little more warmth. But on a neutral prompt it tends to draw the same kind of face again and again."),
    ("I counted it: with \"a woman sitting in a café\" and sixteen seeds, SFT drew an East Asian woman about fourteen times. Base, about four.",
     "Lo conté: con \"una mujer sentada en una cafetería\" y dieciséis semillas, SFT dibujó una mujer asiática unas catorce veces. Base, unas cuatro.",
     "I counted it: with a woman sitting in a café, and sixteen seeds, S F T drew an East Asian woman about fourteen times. Base, about four."),
    ("Base gives you more variety in the people, but its faces are softer, almost smooth.",
     "Base te da más variedad de personas, pero sus caras son más suaves, casi lisas."),
    ("So the idea is simple: let Base decide the picture and the person, and let SFT add the finish on top.",
     "Así que la idea es sencilla: que Base decida la imagen y la persona, y que SFT ponga el acabado encima.",
     "So the idea is simple: let Base decide the picture and the person, and let S F T add the finish on top."),
    ("Let's start. The first node is the MOSS Loader. It loads three things at once: the image model, the text encoder and the VAE.",
     "Empecemos. El primer nodo es el MOSS Loader. Carga tres cosas a la vez: el modelo de imagen, el text encoder y el VAE.",
     "Let's start. The first node is the MOSS Loader. It loads three things at once: the image model, the text encoder and the V A E."),
    ("As the model, we choose Base. The text encoder reads your prompt, and the VAE turns the result into pixels.",
     "Como modelo, elegimos Base. El text encoder lee tu prompt, y el VAE convierte el resultado en píxeles.",
     "As the model, we choose Base. The text encoder reads your prompt, and the V A E turns the result into pixels."),
    ("Next, the Prompt Enhancer. Here we write the prompt: a studio photograph of an elderly fisherman, with a grey knitted cap and soft window light.",
     "Después, el Prompt Enhancer. Aquí escribimos el prompt: una foto de estudio de un pescador mayor, con gorro gris de punto y luz suave de ventana."),
    ("Its mode is \"as written\": it sends the prompt exactly as I typed it. We want both models to read the very same words.",
     "Su modo es \"as written\": manda el prompt tal cual lo escribí. Queremos que los dos modelos lean exactamente las mismas palabras.",
     "Its mode is as written: it sends the prompt exactly as I typed it. We want both models to read the very same words."),
    ("Then the MOSS Latent: the empty canvas the model will paint on. One to one, one megapixel, and seed one thousand.",
     "Después el MOSS Latent: el lienzo vacío sobre el que pintará el modelo. Uno a uno, un megapíxel, y semilla mil."),
    ("Now the sampler, the node that actually draws. We connect the model, the prompt, the latent and the VAE.",
     "Ahora el sampler, el nodo que de verdad dibuja. Conectamos el modelo, el prompt, el latent y el VAE.",
     "Now the sampler, the node that actually draws. We connect the model, the prompt, the latent and the V A E."),
    ("To make it fast, we add a MOSS Turbo node. Which one? Turbo 8-steps v1.0, because it was trained on Base, the model of this stage.",
     "Para que vaya rápido, añadimos un nodo MOSS Turbo. ¿Cuál? Turbo 8-steps v1.0, porque se entrenó sobre Base, el modelo de esta etapa.",
     "To make it fast, we add a MOSS Turbo node. Which one? Turbo eight steps, version one, because it was trained on Base, the model of this stage."),
    ("Eight steps and CFG one, the settings it was trained for. When the Turbo is connected, the sampler takes these settings from it.",
     "Ocho pasos y CFG uno, los ajustes con los que se entrenó. Cuando el Turbo está conectado, el sampler toma de él estos ajustes.",
     "Eight steps and C F G one, the settings it was trained for. When the Turbo is connected, the sampler takes these settings from it."),
    ("And a Preview Image after it, so we can see what Base did on its own before SFT touches it.",
     "Y un Preview Image detrás, para ver lo que hizo Base por sí solo antes de que SFT lo toque.",
     "And a Preview Image after it, so we can see what Base did on its own before S F T touches it."),
    ("That's stage one. I'll put it in a group, and call it \"Base composes\".",
     "Esa es la etapa uno. La meto en un grupo y lo llamo \"Base composes\".",
     "That's stage one. I'll put it in a group, and call it Base composes."),
    ("Stage two. Now we need SFT. But do we need another MOSS Loader? No: the text encoder and the VAE are the same for both models.",
     "Etapa dos. Ahora necesitamos SFT. ¿Pero hace falta otro MOSS Loader? No: el text encoder y el VAE son los mismos para los dos modelos.",
     "Stage two. Now we need S F T. But do we need another MOSS Loader? No: the text encoder and the V A E are the same for both models."),
    ("So we only load the image model, with ComfyUI's Load Diffusion Model node, and choose SFT. This way the text encoder is loaded only once.",
     "Así que solo cargamos el modelo de imagen, con el nodo Load Diffusion Model de ComfyUI, y elegimos SFT. Así el text encoder se carga una sola vez.",
     "So we only load the image model, with ComfyUI's Load Diffusion Model node, and choose S F T. This way the text encoder is loaded only once."),
    ("A second MOSS KSampler. Its model is SFT, and its prompt is the same Prompt Enhancer.",
     "Un segundo MOSS KSampler. Su modelo es SFT, y su prompt es el mismo Prompt Enhancer.",
     "A second MOSS K Sampler. Its model is S F T, and its prompt is the same Prompt Enhancer."),
    ("And here's the key wire: its latent doesn't come from an empty canvas. It comes from the first sampler: Base's picture.",
     "Y aquí está el cable clave: su latent no viene de un lienzo vacío. Viene del primer sampler: la imagen de Base."),
    ("Its turbo is Turbo 8-steps v0.7, the one trained on SFT. Each model with its own turbo.",
     "Su turbo es Turbo 8-steps v0.7, el que se entrenó sobre SFT. Cada modelo con su propio turbo.",
     "Its turbo is Turbo eight steps, version zero point seven, the one trained on S F T. Each model with its own turbo."),
    ("Now, the most important setting of the whole video: denoise. What is it for?",
     "Y ahora, el ajuste más importante de todo el vídeo: el denoise. ¿Para qué sirve?"),
    ("Denoise says how much of the image the second model is allowed to redraw. At one, it redraws everything, and Base's picture is lost. At zero, it changes nothing.",
     "El denoise dice cuánto de la imagen puede volver a dibujar el segundo modelo. A uno, lo redibuja todo y la imagen de Base se pierde. A cero, no cambia nada."),
    ("In between, it adds noise to Base's picture, and SFT cleans that noise its own way. Low values keep the shapes, and only touch the fine detail.",
     "Entre medias, añade ruido a la imagen de Base, y SFT limpia ese ruido a su manera. Los valores bajos mantienen las formas y solo tocan el detalle fino.",
     "In between, it adds noise to Base's picture, and S F T cleans that noise its own way. Low values keep the shapes, and only touch the fine detail."),
    ("And denoise mode? Set it to exact. With few steps, the usual mode can give the same image for two different denoise values. Exact follows every value you set.",
     "¿Y denoise mode? Ponlo en exact. Con pocos pasos, el modo normal puede dar la misma imagen con dos denoise distintos. Exact sigue cada valor que pongas."),
    ("So which value? I didn't guess it. I measured how much each face stayed the same person, with a face-recognition model, on thirty-two images.",
     "¿Entonces qué valor? No lo elegí a ojo. Medí cuánto seguía siendo la misma persona cada cara, con un modelo de reconocimiento facial, en treinta y dos imágenes."),
    ("At zero point three, all thirty-two were still the same person. At zero point seven, the faces start turning into SFT's faces.",
     "A cero coma tres, las treinta y dos seguían siendo la misma persona. A cero coma siete, las caras empiezan a convertirse en las de SFT.",
     "At zero point three, all thirty-two were still the same person. At zero point seven, the faces start turning into S F T's faces."),
    ("So we use zero point three: SFT's finish, Base's person.",
     "Así que usamos cero coma tres: el acabado de SFT, la persona de Base.",
     "So we use zero point three: S F T's finish, Base's person."),
    ("To finish, a MOSS Image Saver: it keeps the prompt, the settings and the workflow inside the PNG.",
     "Para terminar, un MOSS Image Saver: guarda el prompt, los ajustes y el workflow dentro del PNG.",
     "To finish, a MOSS Image Saver: it keeps the prompt, the settings and the workflow inside the P N G."),
    ("And Suite Telemetry, which reads every node and writes a report. We'll read it after the run.",
     "Y Suite Telemetry, que lee todos los nodos y escribe un informe. Lo leeremos después del Run."),
    ("Second group: \"SFT redraws at 0.3\". The workflow is ready. Let's run it.",
     "Segundo grupo: \"SFT redraws at 0.3\". El workflow está listo. Vamos a ejecutarlo.",
     "Second group: S F T redraws at zero point three. The workflow is ready. Let's run it."),
    ("The first run loads both models, so it takes a little longer. After that, both stages together take about six seconds an image.",
     "El primer Run carga los dos modelos, así que tarda algo más. Después, las dos etapas juntas tardan unos seis segundos por imagen."),
    ("On the left, Base alone. On the right, after SFT. The same man, the same cap, the same light.",
     "A la izquierda, Base solo. A la derecha, después de SFT. El mismo hombre, el mismo gorro, la misma luz.",
     "On the left, Base alone. On the right, after S F T. The same man, the same cap, the same light."),
    ("Let's look closer, at one hundred percent. Base's skin is smooth. After SFT, you get the pores, the wrinkles around the eyes, the loose hairs in the moustache, the fabric of the cap.",
     "Acerquémonos, al cien por cien. La piel de Base es lisa. Después de SFT tienes los poros, las arrugas alrededor de los ojos, los pelos sueltos del bigote, la tela de la gorra.",
     "Let's look closer, at one hundred percent. Base's skin is smooth. After S F T, you get the pores, the wrinkles around the eyes, the loose hairs in the moustache, the fabric of the cap."),
    ("And the Telemetry report? It lists both stages. Its note on the turbo at denoise zero point three only says the turbo was first measured at other values; this one is measured now, in the results file.",
     "¿Y el informe de Telemetry? Lista las dos etapas. Su nota sobre el turbo a denoise cero coma tres solo dice que el turbo se midió primero con otros valores; este ya está medido, en el archivo de resultados."),
    ("Now a question you might have: if zero point three is good, isn't more even better? Let's try zero point seven.",
     "Ahora una pregunta que quizá tengas: si cero coma tres está bien, ¿más no será mejor? Probemos cero coma siete."),
    ("Notice it only took about three seconds: ComfyUI remembers stage one, so only stage two runs again.",
     "Fíjate en que solo ha tardado unos tres segundos: ComfyUI recuerda la etapa uno, así que solo se vuelve a ejecutar la etapa dos."),
    ("Same pose, same clothes, same light. But look at the face: it's wider, the nose is different, the skin is darker. It's turning into another man.",
     "Misma pose, misma ropa, misma luz. Pero mira la cara: es más ancha, la nariz es distinta, la piel más oscura. Se está convirtiendo en otro hombre."),
    ("The face-recognition score against Base's man drops from zero point eight eight at zero point three, to zero point six eight here.",
     "La puntuación de reconocimiento facial frente al hombre de Base baja de cero coma ochenta y ocho a 0,3, a cero coma sesenta y ocho aquí."),
    ("That's the trade: the higher the denoise, the more SFT you get, and the less of Base's person you keep. Raise it for more of SFT, lower it to keep more of Base.",
     "Ese es el intercambio: cuanto más alto el denoise, más SFT tienes y menos conservas de la persona de Base. Súbelo para más SFT, bájalo para conservar más de Base.",
     "That's the trade: the higher the denoise, the more S F T you get, and the less of Base's person you keep. Raise it for more of S F T, lower it to keep more of Base."),
    ("Let's recap. One: Base composes the picture and chooses the person; SFT redraws it and adds the finish.",
     "Repasemos. Uno: Base compone la imagen y elige la persona; SFT la redibuja y pone el acabado.",
     "Let's recap. One: Base composes the picture and chooses the person; S F T redraws it and adds the finish."),
    ("Two: each model with its own turbo, and the text encoder loaded only once.",
     "Dos: cada modelo con su propio turbo, y el text encoder cargado una sola vez."),
    ("Three: denoise zero point three in exact mode keeps the person; measure before you raise it.",
     "Tres: denoise cero coma tres en modo exact conserva a la persona; mide antes de subirlo."),
    ("And one tip: if you want a specific kind of person, name it in the prompt. That's still the most reliable way.",
     "Y un consejo: si quieres un tipo de persona concreto, nómbralo en el prompt. Sigue siendo la forma más fiable."),
    ("You can download this workflow for free from GitHub. The link is in the description.",
     "Puedes descargar este workflow gratis desde GitHub. El enlace está en la descripción."),
    ("If you've made it to the end and enjoyed the video, and you'd like to see more content like this, leave a like and subscribe. It's free!",
     "Si has llegado hasta el final del vídeo, te ha gustado y quieres ver más contenido de este tipo, dale un buen like y suscríbete. ¡Es gratis!"),
    ("Thanks for watching, and see you in the next one.",
     "Gracias por verlo, y nos vemos en el próximo."),
]


def ts_view(name):
    return f"/view?filename={name}&subfolder={TS_SHOTS}&type=output"


def ts_crops_html(urls, labels, top=40):
    """512 x 512 face crops at 100 % (512 image px = 512 video px), side by side, a label over each."""
    side = 512 * 1920 / W
    x, y = TS_CROP
    cells = "".join(f'<div style="text-align:center"><div style="color:#fff;font:600 26px Segoe UI,sans-serif;margin-bottom:12px">{t}</div>'
                    f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:6px;position:relative">'
                    f'<img src="{u}" style="position:absolute;max-width:none;width:{1024 * side / 512}px;height:{1024 * side / 512}px;'
                    f'left:{-x * side / 512}px;top:{-y * side / 512}px"></div></div>' for u, t in zip(urls, labels))
    return f'<div style="display:flex;gap:30px;margin-top:{top}px">{cells}</div>'


def ts_pair_html(urls, labels, height=760):
    """Two full images side by side, a label under each."""
    cells = "".join(f'<div style="text-align:center"><img src="{u}" style="height:{height}px;border-radius:6px">'
                    f'<div style="color:#fff;font:700 30px Segoe UI,sans-serif;margin-top:14px">{t}</div></div>' for u, t in zip(urls, labels))
    return f'<div style="display:flex;gap:40px;margin-top:40px">{cells}</div>'


def ts_models_card(es, step):
    """Step 0: Base and SFT; 1: what each gives, with the café counts; 2: the plan, Base then SFT."""
    col = lambda name, colour, rows: (f'<div style="width:560px;border-radius:18px;background:#1d1d24;padding:30px 34px;text-align:left">'
                                      f'<div style="font:800 56px Segoe UI,sans-serif;color:{colour};margin-bottom:18px">{name}</div>'
                                      + "".join(f'<div style="font:600 32px Segoe UI,sans-serif;color:#e6e6ea;margin:14px 0">{r}</div>' for r in rows) + '</div>')
    if step < 2:
        base = ([("más variedad de personas" if es else "more variety in the people"), ("caras más lisas" if es else "softer faces"),
                 ("café: ~4 / 16 asiáticas" if es else "café: ~4 / 16 East Asian")] if step else [])
        sft = ([("textura de piel, poros, arrugas" if es else "skin texture, pores, wrinkles"), ("repite el mismo tipo de cara" if es else "repeats the same kind of face"),
                ("café: ~14 / 16 asiáticas" if es else "café: ~14 / 16 East Asian")] if step else [("Base, entrenado un poco más" if es else "Base, trained further")])
        return (f'<div style="display:flex;gap:60px;justify-content:center;margin-top:200px;color:#fff">'
                f'{col("Base", "#7fbf7f", base)}{col("SFT", "#7fa6e6", sft)}</div>')
    box = lambda t, sub, c: (f'<div style="width:420px;border-radius:16px;background:{c};padding:30px;text-align:center">'
                             f'<div style="font:800 50px Segoe UI,sans-serif">{t}</div><div style="font:600 30px Segoe UI,sans-serif;margin-top:12px;color:#e6e6ea">{sub}</div></div>')
    return (f'<div style="display:flex;gap:40px;align-items:center;justify-content:center;margin-top:260px;color:#fff">'
            f'{box("1 · Base", "la imagen y la persona" if es else "the picture and the person", "#2f4a2f")}'
            f'<div style="font:800 70px Segoe UI,sans-serif;color:#ffcc33">→</div>'
            f'{box("2 · SFT", "el acabado" if es else "the finish", "#2e4a7a")}</div>')


def ts_identity_html(es, mark=None):
    """The measured identity table (ArcFace against Base's face, 16 seeds x 2 prompts); mark: the row to frame."""
    rows = [("0.3", "0.80", "0.90", "32 / 32"), ("0.5", "0.66", "0.85", "29 / 32"), ("0.7", "0.51", "0.73", "19 / 32")]
    head = (("denoise", "café", "retrato" if es else "portrait", "≥ 0.6"))
    cell = lambda t, w, c="#e6e6ea", b=600: f'<div style="width:{w}px;font:{b} 36px Segoe UI,sans-serif;color:{c};text-align:center">{t}</div>'
    body = "".join(f'<div style="display:flex;padding:16px 0;border-radius:10px;{"background:#3a3214;" if r[0] == mark else ""}">'
                   + cell(r[0], 220, "#ffcc33", 800) + cell(r[1], 260) + cell(r[2], 260) + cell(r[3], 260) + "</div>" for r in rows)
    title = ("Misma persona que en Base (reconocimiento facial, 2 prompts × 16 semillas)" if es else
             "Same person as Base's (face recognition, 2 prompts × 16 seeds)")
    return (f'<div style="margin-top:200px;color:#fff;text-align:center"><div style="font:600 32px Segoe UI,sans-serif;color:#c9c9d1;margin-bottom:26px">{title}</div>'
            f'<div style="display:inline-block;background:#15151b;border-radius:14px;padding:20px 30px">'
            f'<div style="display:flex;padding-bottom:10px;border-bottom:2px solid #333">{"".join(cell(h, w, "#c9c9d1", 700) for h, w in zip(head, (220, 260, 260, 260)))}</div>'
            f'{body}</div></div>')


def ts_report_html(report, es):
    """The take's Telemetry report: the nodes it read and its one note."""
    import re as _re
    nodes = _re.search(r"Nodes read: ([^\n]*)", report)
    note = _re.search(r"\*\*NOTE\*\* ([^\n]*)", report)
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<div style="margin-top:220px;width:1600px;text-align:left;background:#15151b;border-radius:12px;padding:34px 44px;color:#e6e6ea">'
            f'<div style="font:700 34px Segoe UI,sans-serif;color:#ffcc33;margin-bottom:18px">HaDeS MOSS Suite Telemetry</div>'
            f'<div style="font:500 30px Consolas,monospace;line-height:1.5">Nodes read: {esc(nodes.group(1) if nodes else "")}</div>'
            f'<div style="font:500 30px Consolas,monospace;line-height:1.5;margin-top:18px;color:#ffcc33">NOTE {esc(note.group(1) if note else "")}</div></div>')


async def ts_group(d, title, bounding, color):
    """A group drawn around the stage, like the example's."""
    await d.js("""([t, b, c]) => { const g = new window.LGraphGroup(t); g.pos = [b[0], b[1]]; g.size = [b[2], b[3]]; g.color = c; g.font_size = 24;
        const a = window.__hv.app(); a.graph.add(g); a.graph.setDirtyCanvas(true, true); }""", [title, bounding, color])


async def ts_outputs(d, ids):
    """The last run: stage 1's preview url, the saved image's url, Telemetry's report."""
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    out = run["outputs"]
    view = lambda im: f"/view?filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}"
    return view(out[str(ids["stage1"])]["images"][0]), view(out[str(ids["save"])]["images"][0]), out[str(ids["report"])]["text"][0]


def ts_check(page_url, url, measured):
    """The take's image against the measured one (MAE at 512, 0-255): the take stops if it is another picture."""
    import io
    import urllib.parse
    import urllib.request
    from PIL import Image
    import numpy as np
    base = urllib.parse.urlsplit(page_url)
    got = Image.open(io.BytesIO(urllib.request.urlopen(f"{base.scheme}://{base.netloc}{url}").read())).convert("RGB").resize((512, 512))
    ref = Image.open(Path(r"I:/HaDeS_MOSS/videos/advanced/adv_two_stage/measure") / measured).convert("RGB").resize((512, 512))
    mae = float(np.abs(np.asarray(got, dtype=np.float32) - np.asarray(ref, dtype=np.float32)).mean())
    print(f"take vs {measured}: MAE {mae:.2f}", flush=True)
    if mae > 2.0:
        raise SystemExit(f"the take's image is not the measured {measured} (MAE {mae:.2f})")


async def ts_prepare(page):
    """Off camera: the finished example once with another seed (1001), so both models and both LoRAs are loaded."""
    import copy
    import urllib.parse
    import make_examples as mx
    base = urllib.parse.urlsplit(page.url)
    info = mx.fetch(f"{base.scheme}://{base.netloc}", "/object_info")
    g = {k: v for k, v in copy.deepcopy(mx.EXAMPLES["HaDeS MOSS Suite - Two Stages, Two Models"]).items() if k != "note"}
    g["latent"][1]["seed"] = TS_SEED + 1
    g["save"][1]["filename_prefix"] = "MOSS/warmup_two_stage"
    local = {mx.BASE: "mossImage_base_bf16.safetensors", mx.SFT: "mossImage_sft_bf16.safetensors",
             mx.TURBO_V10: TS_LORA_BASE, mx.TURBO_8STEPS: TS_LORA_SFT}
    print("warm-up run:", await page.evaluate(UP_WARM_JS, mx.to_api(g, info, local)), flush=True)


async def two_stage(d: Director):
    """Advanced workflow #3 as a lesson: why two models, stage 1 (Base) and stage 2 (SFT over Base's latent) built node by node,
    denoise explained with the measured identity table, a run with crops, denoise 0.7 on camera, a recap."""
    P = TS_POS
    es = d.language == "es"
    say_after = d._after_voice
    await d.camera({"box": [-60, -80, 1400, 1000]}, 10, pad=60, top=60)
    await d.move(W * 0.97, H * 0.55, 10)
    lab = (("Base sola", "+ SFT a 0,3", "+ SFT a 0,7") if es else ("Base alone", "+ SFT at 0.3", "+ SFT at 0.7"))

    # 1. what we'll learn
    await d.js("(h) => window.__hv.show(h, 10)",
               ts_crops_html([ts_view("d03_stage1_00001_.png"), ts_view("d03_00001_.png")], lab[:2], top=60) +
               '<div style="position:fixed;left:0;right:0;bottom:90px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:600 36px Segoe UI,sans-serif;color:#ffcc33;margin-top:14px">{"Workflow avanzado n.º 3" if es else "Advanced workflow #3"}'
               ' · Two Stages, Two Models</div></div>')
    await d.hold(0.8)
    for i in range(4):
        await d.say(i)

    # 2. why two models
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_models_card(es, 0))
    await d.say(4)
    await d.say(5)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_models_card(es, 1))
    await d.say(6)
    await d.say(7)
    await d.say(8)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_models_card(es, 2))
    await d.say(9)
    await say_after(1.0)
    await d.js("() => window.__hv.hide()")

    # 3. stage 1: Base, the prompt, the latent
    load = await ap_add(d, "HaDeSMossLoader", "MOSS Loader", P["load"])
    await up_set(d, load, BASIC_LOADER)
    await d.camera([load], 900, pad=140, top=60)
    await d.say(10)
    await say_after(0.25)
    await basic_point(d, load, "unet_name")
    await d.say(11)
    await d.hold(1.6)
    await basic_point(d, load, "clip_name")
    await d.hold(1.4)
    await basic_point(d, load, "vae_name")
    await say_after(0.25)
    await d.camera({"box": [-60, -80, 1000, 900]}, 1000, pad=60, top=60)
    enh = await ap_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enh"])
    await up_set(d, enh, {"mode": "as written", "width": 1024, "height": 1024, "seed": 1, "negative": ""})
    await d.say(12)
    async with d.fast(2.0):
        await basic_link(d, load, "CLIP", enh, "clip", 700)
        await pe_type(d, enh, "idea", TS_PROMPT, paste=True)
    await say_after(0.25)
    await ml_zoom(d, enh, ["mode"])
    await basic_point(d, enh, "mode")
    await d.say(13)
    await say_after(0.25)
    await d.camera({"box": [880, -80, 1400, 900]}, 1000, pad=60, top=60)
    latent = await ap_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": TS_SEED, "control_after_generate": "fixed"})
    await ml_zoom(d, latent, ["seed"])
    await basic_point(d, latent, "seed")
    await d.say(14)

    # 4. stage 1: the sampler and its turbo
    await say_after(0.25)
    await d.camera({"box": [-60, -80, 1850, 1000]}, 1000, pad=60, top=60)
    ks = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks"])
    await up_set(d, ks, {"denoise": 1.0, "denoise_mode": "steps / denoise (core)"})
    await d.say(15)
    async with d.fast(3.0):
        for a, out, inp in ((load, "MODEL", "model"), (enh, "positive", "positive"), (enh, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (load, "VAE", "vae")):
            await basic_link(d, a, out, ks, inp, 600)
    await say_after(0.25)
    t1 = await ap_add(d, "HaDeSMossTurbo", "MOSS Turbo", P["t1"])
    await up_set(d, t1, {"lora_name": TS_LORA_BASE, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
    async with d.fast(1.6):
        await basic_link(d, t1, "turbo", ks, "turbo", 700)
    await ml_zoom(d, t1, ["lora_name"])
    await basic_point(d, t1, "lora_name")
    await d.say(16)
    await say_after(0.25)
    await ml_zoom(d, t1, ["steps", "cfg"])
    await basic_point(d, t1, "steps")
    await d.say(17)
    await d.hold(1.2)
    await basic_point(d, t1, "cfg")
    await say_after(0.25)
    await d.camera({"box": [-60, -80, 1850, 1000]}, 900, pad=60, top=60)
    prev = await ap_add(d, "PreviewImage", "Preview Image", P["stage1"])
    async with d.fast(1.6):
        await basic_link(d, ks, "image", prev, "images", 700)
    await d.say(18)
    await say_after(0.25)
    await ts_group(d, "1. Base composes", [-30, -80, 1840, 960], "#335533")
    await d.say(19)

    # 5. stage 2: SFT
    await say_after(0.5)
    await d.camera({"box": [-60, -80, 1000, 900]}, 900, pad=60, top=60)
    await d.say(20)
    sft = await ap_add(d, "UNETLoader", "Load Diffusion Model", P["sft"])
    await up_set(d, sft, {"unet_name": "mossImage_sft_bf16.safetensors", "weight_dtype": "default"})
    await say_after(0.25)
    await d.camera([sft], 900, pad=140, top=60)
    await basic_point(d, sft, "unet_name")
    await d.say(21)

    # 6. stage 2: the second sampler and denoise
    await say_after(0.25)
    await d.camera({"box": [-60, -120, 2750, 1050]}, 1000, pad=60, top=60)
    ks2 = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks2"])
    await up_set(d, ks2, {"denoise": 1.0, "denoise_mode": "steps / denoise (core)"})
    await d.say(22)
    async with d.fast(3.0):
        for a, out, inp in ((sft, "MODEL", "model"), (enh, "positive", "positive"), (enh, "negative", "negative"),
                            (latent, "seed", "seed"), (load, "VAE", "vae")):
            await basic_link(d, a, out, ks2, inp, 600)
    await say_after(0.25)
    await d.camera({"box": [1350, -80, 2750, 700]}, 900, pad=60, top=60)
    await d.say(23)
    await basic_link(d, ks, "latent", ks2, "latent_image", 1800)
    await say_after(0.25)
    await d.camera({"box": [1350, -80, 2750, 1050]}, 900, pad=60, top=60)
    t2 = await ap_add(d, "HaDeSMossTurbo", "MOSS Turbo", P["t2"])
    await up_set(d, t2, {"lora_name": TS_LORA_SFT, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
    async with d.fast(1.6):
        await basic_link(d, t2, "turbo", ks2, "turbo", 700)
    await ml_zoom(d, t2, ["lora_name"])
    await basic_point(d, t2, "lora_name")
    await d.say(24)
    await say_after(0.25)
    await ml_zoom(d, ks2, ["denoise"])
    await basic_point(d, ks2, "denoise")
    await d.say(25)
    await d.say(26)
    await d.say(27)
    await say_after(1.0)
    await ml_zoom(d, ks2, ["denoise_mode"])
    await basic_point(d, ks2, "denoise_mode")
    await d.hold(0.4)
    await up_set(d, ks2, {"denoise_mode": "exact"})
    await d.say(28)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_identity_html(es))
    await d.say(29)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_identity_html(es, mark="0.3"))
    await d.say(30)
    await say_after(0.25)
    await d.js("() => window.__hv.hide()")
    await ml_zoom(d, ks2, ["denoise"])
    await basic_point(d, ks2, "denoise")
    await d.hold(0.4)
    await up_set(d, ks2, {"denoise": 0.3})
    await d.say(31)
    await say_after(1.0)

    # 7. save and check
    await d.camera({"box": [2200, -80, 3640, 1050]}, 1000, pad=60, top=60)
    save = await ap_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["save"])
    await up_set(d, save, {"filename_prefix": "MOSS/HaDeS_MOSS_two_stage_%seed%"})
    async with d.fast(1.6):
        await basic_link(d, ks2, "image", save, "images", 700)
    await d.say(32)
    await say_after(0.25)
    tel = await ap_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["tel"])
    await d.say(33)
    async with d.fast(4.0):
        await d.camera({"box": TS_BOX}, 600, pad=60, top=60)
        for i, a in enumerate((load, enh, latent, t1, ks, t2, ks2), 1):
            await basic_link(d, a, "info", tel, f"info_{i}", 600)
        await d.camera({"box": [2200, -80, 3640, 1050]}, 600, pad=60, top=60)
        await basic_link(d, tel, "json", save, "metadata", 600)
        rep = await ap_add(d, "PreviewAny", "Preview Any", P["report"])
        await basic_link(d, tel, "report", rep, "source", 600)
    await say_after(0.25)
    await d.camera({"box": TS_BOX}, 1000, pad=60, top=60)
    await ts_group(d, "2. SFT redraws at 0.3", [-30, -120, 3640, 1090], "#334466")
    await d.say(34)
    await say_after(0.25)

    # 8. the run
    ids = {"stage1": prev, "save": save, "report": rep}
    final, secs = await basic_run(d, save)
    stage1, final, report = await ts_outputs(d, ids)
    ts_check(d.page.url, final, "full_d03.png")
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs:.1f} s · RTX 5090', ("2 etapas · 8 + 8 pasos" if es else "2 stages · 8 + 8 steps")], top=360))
    await d.say(35)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_pair_html([stage1, final], lab[:2]))
    await d.say(36)
    await say_after(1.5)
    await d.js("(h) => window.__hv.show(h)", ts_crops_html([stage1, final], lab[:2], top=120))
    await d.say(37)
    await say_after(1.0)
    await d.js("(h) => window.__hv.show(h)", ts_report_html(report, es))
    await d.say(38)
    await say_after(0.5)
    await d.js("() => window.__hv.hide()")

    # 9. what if I raise denoise?
    await ml_zoom(d, ks2, ["denoise"])
    await basic_point(d, ks2, "denoise")
    await d.say(39)
    await d.hold(1.0)
    await up_set(d, ks2, {"denoise": 0.7})
    await say_after(0.25)
    await d.camera({"box": TS_BOX}, 900, pad=60, top=60)
    _, secs7 = await basic_run(d, save)
    _, final7, _ = await ts_outputs(d, ids)
    ts_check(d.page.url, final7, "full_d07.png")
    if secs7 > 5.0:
        raise SystemExit(f"the 0.7 run took {secs7:.1f} s: stage 1 was not cached, line 41 would be wrong")
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs7:.1f} s', ("solo la etapa 2" if es else "stage 2 only")], top=360))
    await d.say(40)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_crops_html([stage1, final, final7], lab, top=120))
    await d.say(41)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_crops_html([stage1, final, final7], lab, top=120) +
               '<div style="position:fixed;left:0;right:0;bottom:120px;text-align:center;font:700 40px Segoe UI,sans-serif;color:#ffcc33">'
               f'{"Reconocimiento facial frente a Base" if es else "Face recognition against Base"}: 0.88 → 0.68</div>')
    await d.say(42)
    await say_after(0.25)
    await d.js("() => window.__hv.hide()")
    await ml_zoom(d, ks2, ["denoise"])
    await basic_point(d, ks2, "denoise")
    await d.say(43)
    await d.hold(1.5)
    await up_set(d, ks2, {"denoise": 0.3})
    await say_after(1.0)

    # 10. recap and closing
    recap = (["1 · Base compone, SFT termina", "2 · cada modelo con su turbo", "3 · denoise 0,3 en exact"] if es else
             ["1 · Base composes, SFT finishes", "2 · each model with its own turbo", "3 · denoise 0.3, exact"])
    await d.js("(h) => window.__hv.show(h)", ap_card(recap[:1]))
    await d.say(44)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(recap[:2]))
    await d.say(45)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(recap))
    await d.say(46)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ap_card(recap + [("Consejo: nombra en el prompt a quien quieres" if es else "Tip: name who you want in the prompt")]))
    await d.say(47)
    await say_after(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": TS_BOX}, 1200, pad=60, top=60)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1300px;margin-top:300px">'
               f'<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">{"Descarga gratis (enlace en la descripción)" if es else "Free download (link in the description)"}</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">examples/advanced/HaDeS MOSS Suite - Two Stages, Two Models.json</div></div>')
    await d.say(48)
    await say_after(0.25)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               ts_crops_html([stage1, final], lab[:2], top=60) +
               '<div style="position:fixed;left:0;right:0;bottom:90px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:400 28px Segoe UI,sans-serif;color:#c9c9d1;margin-top:18px">{"Workflow, Suite y modelos: enlaces en la descripción" if es else "Workflow, Suite and models: links in the description"}</div></div>')
    await d.say(49)
    await d.say(50)
    await d.end(1.5)


two_stage.prepare = ts_prepare


# ---- advanced workflow #4: LoRA Surgery (a lesson) ----
LS_FILE = None  # built on camera (a lesson); the finished example is the download
LS_SEED = 1013  # the example's seed; measured: the 8-step fisherman equals the measured render (videos/advanced/adv_lora_surgery/MEDIDAS.md)
LS_OLD = "lora_surgery"  # the workflow's measured renders in comfy_output (advanced_workflows/lora_surgery/files.json)
LS_CROP = (300, 150)  # 512 x 512 on the fisherman's face
LS_PROMPT = ("Photograph of an old fisherman with a grey beard mending a net on a wooden dock at sunrise, his face and weathered hands in view, "
             "thick wool sweater, seagulls in the background.")
LS_TURBO = "MOSS\\turbo_v10\\HaDeS-MOSS-Turbo-8steps-v1.0.safetensors"
LS_EXT, LS_STYLE, LS_ONE, LS_R32 = "bulle_extracted", "bulle_style", "bulle_style_turbo", "bulle_style_turbo_r32"
LS_POS = {"load": (0, 0), "sft": (0, 420), "base_note": (0, 0), "bm": (460, 0), "bulle": (460, 620), "ext": (920, 0), "ext_text": (920, 470),
          "merger": (1400, 0), "merger_text": (1400, 520), "ml": (460, 1180), "enh": (920, 1180), "latent": (1400, 1180), "ks": (1850, 1180),
          "save": (2300, 1180), "tel": (2300, 1700), "rep": (2750, 1700), "merger2": (1880, 0), "m2_text": (1880, 520),
          "resize": (2360, 0), "rs_text": (2360, 470)}
LS_BOX = [-60, -120, 3250, 2250]
LS_TOP = [-60, -120, 2850, 1000]

LS_LINES = [
    ("Hi, I'm HaDeS, and welcome to advanced workflow number four: LoRA surgery.",
     "Hola, soy HaDeS, y bienvenido al workflow avanzado número cuatro: cirugía de LoRAs."),
    ("Today you'll learn how to pull a LoRA out of a finished model, how to cut one LoRA out of another, and how to put them back together in one file.",
     "Hoy vas a aprender a sacar una LoRA de un modelo terminado, a quitar una LoRA de dentro de otra, y a volver a juntarlas en un solo archivo.",
     "Today you'll learn how to pull a Lora out of a finished model, how to cut one Lora out of another, and how to put them back together in one file."),
    ("And you'll see why the obvious number is not always the right one, and how to find the right one by measuring.",
     "Y vas a ver por qué el número obvio no siempre es el bueno, y cómo encontrar el bueno midiendo."),
    ("As always, we build it together from an empty canvas, everything is free, and all the links are in the description.",
     "Como siempre, lo montamos juntos desde el lienzo vacío, todo es gratis y tienes todos los enlaces en la descripción."),
    ("Bulle d'Encre is my comic-style model. It isn't trained from zero: it's a recipe.",
     "Bulle d'Encre es mi modelo de estilo cómic. No está entrenado desde cero: es una receta.",
     "Bull Dahnkr is my comic style model. It isn't trained from zero: it's a recipe."),
    ("SFT, with thirty percent of Base in six layers. On top, half of my Turbo v1.0. And on top of that, a style LoRA at zero point eight.",
     "SFT, con un treinta por ciento de Base en seis capas. Encima, la mitad de mi Turbo v1.0. Y encima de eso, una LoRA de estilo a cero coma ocho.",
     "S F T, with thirty percent of Base in six layers. On top, half of my Turbo version one. And on top of that, a style Lora at zero point eight."),
    ("That style LoRA I keep private, so you won't see it in this video. But the model is public, and that's all we need.",
     "Esa LoRA de estilo la guardo en privado, así que no la vas a ver en este vídeo. Pero el modelo es público, y es todo lo que necesitamos.",
     "That style Lora I keep private, so you won't see it in this video. But the model is public, and that's all we need."),
    ("Here's the question. What if you like the style, but you want the full turbo, or no turbo at all?",
     "Aquí va la pregunta. ¿Y si te gusta el estilo, pero quieres la turbo entera, o ninguna turbo?"),
    ("The style and the half turbo are baked into ten gigabytes of weights. Today we take them out, separate them, and put them back the way we want.",
     "El estilo y la media turbo están cocinados dentro de diez gigas de pesos. Hoy los sacamos, los separamos y los volvemos a poner como queramos."),
    ("To pull a LoRA out of a model, you need what the model was before. A LoRA is just the difference between two models.",
     "Para sacar una LoRA de un modelo, necesitas lo que era el modelo antes. Una LoRA es solo la diferencia entre dos modelos.",
     "To pull a Lora out of a model, you need what the model was before. A Lora is just the difference between two models."),
    ("So first we rebuild Bulle's starting point. The MOSS Loader loads Base, with the text encoder and the VAE.",
     "Así que primero reconstruimos el punto de partida de Bulle. El MOSS Loader carga Base, con el text encoder y el VAE.",
     "So first we rebuild Bull's starting point. The MOSS Loader loads Base, with the text encoder and the V A E."),
    ("Then a Load Diffusion Model node with SFT.",
     "Después, un nodo Load Diffusion Model con SFT.",
     "Then a Load Diffusion Model node with S F T."),
    ("And the MOSS Model Block Merger mixes them layer by layer. Every slider is a group of layers.",
     "Y el MOSS Model Block Merger los mezcla capa a capa. Cada deslizador es un grupo de capas."),
    ("This slider is layers twenty-four to twenty-nine. At zero point three, it takes thirty percent of Base there and leaves SFT everywhere else. That's Bulle's recipe.",
     "Este deslizador son las capas veinticuatro a veintinueve. A cero coma tres, toma un treinta por ciento de Base ahí y deja SFT en todo lo demás. Es la receta de Bulle.",
     "This slider is layers twenty-four to twenty-nine. At zero point three, it takes thirty percent of Base there and leaves S F T everywhere else. That's Bull's recipe."),
    ("Nothing is saved to disk: the merge lives in memory, just for this workflow.",
     "No se guarda nada en disco: la mezcla vive en memoria, solo para este workflow."),
    ("Now the finished model: a second Load Diffusion Model with Bulle d'Encre.",
     "Ahora el modelo terminado: un segundo Load Diffusion Model con Bulle d'Encre.",
     "Now the finished model: a second Load Diffusion Model with Bull Dahnkr."),
    ("And the MOSS LoRA Extractor. It takes the finished model, subtracts the base, and squeezes the difference into a LoRA.",
     "Y el MOSS LoRA Extractor. Toma el modelo terminado, le resta la base y comprime la diferencia en una LoRA.",
     "And the MOSS Lora Extractor. It takes the finished model, subtracts the base, and squeezes the difference into a Lora."),
    ("Rank is how much detail the LoRA keeps. I use sixty-four. Higher keeps more, but the file grows.",
     "El rango es cuánto detalle guarda la LoRA. Uso sesenta y cuatro. Más alto guarda más, pero el archivo crece.",
     "Rank is how much detail the Lora keeps. I use sixty-four. Higher keeps more, but the file grows."),
    ("We call it bulle_extracted, and run. This one is heavy: three models at once, almost two minutes. I'll speed it up.",
     "La llamamos bulle_extracted y lanzamos. Esta es pesada: tres modelos a la vez, casi dos minutos. La acelero.",
     "We call it Bull extracted, and run. This one is heavy: three models at once, almost two minutes. I'll speed it up."),
    ("The report says: rank sixty-four, two hundred fifty-two megabytes, and ninety-seven percent of the energy kept.",
     "El informe dice: rango sesenta y cuatro, doscientos cincuenta y dos megas, y un noventa y siete por ciento de la energía conservada."),
    ("Energy is a simple way to say: how much of the difference fits in the LoRA. Ninety-seven percent is very good.",
     "La energía es una forma sencilla de decir cuánto de la diferencia cabe en la LoRA. Noventa y siete por ciento es muy bueno.",
     "Energy is a simple way to say: how much of the difference fits in the Lora. Ninety-seven percent is very good."),
    ("This LoRA has two things inside: the style, and half a turbo. Now we cut the turbo out.",
     "Esta LoRA lleva dos cosas dentro: el estilo y media turbo. Ahora le quitamos la turbo.",
     "This Lora has two things inside: the style, and half a turbo. Now we cut the turbo out."),
    ("The MOSS LoRA Merger adds LoRAs with a weight each. And a weight can be negative. A negative weight subtracts.",
     "El MOSS LoRA Merger suma LoRAs, cada una con su peso. Y un peso puede ser negativo. Un peso negativo resta.",
     "The MOSS Lora Merger adds Loras with a weight each. And a weight can be negative. A negative weight subtracts."),
    ("First LoRA: bulle_extracted, at one. Second: Turbo v1.0. And the weight?",
     "Primera LoRA: bulle_extracted, a uno. Segunda: Turbo v1.0. ¿Y el peso?",
     "First Lora: Bull extracted, at one. Second: Turbo version one. And the weight?"),
    ("The recipe had half a turbo, so minus zero point five, right? That's what I tried first.",
     "La receta tenía media turbo, así que menos cero coma cinco, ¿verdad? Es lo que probé primero."),
    ("And it went wrong. At minus zero point five every image came out dark and muddy. Look at the café: the light is gone.",
     "Y salió mal. A menos cero coma cinco todas las imágenes salían oscuras y embarradas. Mira la cafetería: la luz ha desaparecido."),
    ("So I measured the weights. The extracted LoRA doesn't carry half a turbo. It carries about two thirds of that half.",
     "Así que medí los pesos. La LoRA extraída no lleva media turbo. Lleva unos dos tercios de esa mitad.",
     "So I measured the weights. The extracted Lora doesn't carry half a turbo. It carries about two thirds of that half."),
    ("The rank-sixty-four squeeze and the way the model was saved lost some of it. So minus zero point five removed too much, and left a negative turbo behind.",
     "La compresión a rango sesenta y cuatro y la forma en que se guardó el modelo perdieron parte. Así que menos cero coma cinco quitaba de más y dejaba una turbo negativa."),
    ("Two thirds of zero point five is about zero point three three. I use minus zero point three.",
     "Dos tercios de cero coma cinco son unos cero coma tres tres. Uso menos cero coma tres."),
    ("That's the lesson of today: the recipe said zero point five, the measurement said zero point three. Trust the measurement.",
     "Esa es la lección de hoy: la receta decía cero coma cinco, la medida decía cero coma tres. Fíate de la medida."),
    ("Rank sixty-six, so nothing is lost: sixty-four plus the turbo's two. We call it bulle_style, and run. It takes about two seconds.",
     "Rango sesenta y seis, para no perder nada: sesenta y cuatro más los dos de la turbo. La llamamos bulle_style y lanzamos. Tarda unos dos segundos.",
     "Rank sixty-six, so nothing is lost: sixty-four plus the turbo's two. We call it Bull style, and run. It takes about two seconds."),
    ("How do we know it worked? I compared it with the original style LoRA, the private one, on four prompts.",
     "¿Cómo sabemos que ha funcionado? La comparé con la LoRA de estilo original, la privada, en cuatro prompts.",
     "How do we know it worked? I compared it with the original style Lora, the private one, on four prompts."),
    ("I measure the difference in seeds: one point zero means as different as changing the seed.",
     "Mido la diferencia en semillas: uno coma cero significa tan distinto como cambiar de semilla."),
    ("At minus zero point three, the distance is zero point one seven. At minus zero point five, zero point four six. Without surgery, zero point five five.",
     "A menos cero coma tres, la distancia es cero coma diecisiete. A menos cero coma cinco, cero coma cuarenta y seis. Sin cirugía, cero coma cincuenta y cinco."),
    ("So the surgery gives back the style, very close to the original, without the turbo.",
     "Así que la cirugía devuelve el estilo, muy cerca del original, sin la turbo."),
    ("Now let's use it. The MOSS Multi-LoRA loads several LoRAs in one node, each on its own row.",
     "Ahora vamos a usarla. El MOSS Multi-LoRA carga varias LoRAs en un solo nodo, cada una en su fila.",
     "Now let's use it. The MOSS Multi Lora loads several Loras in one node, each on its own row."),
    ("First row: bulle_style at one. Second row: the full Turbo v1.0, also at one.",
     "Primera fila: bulle_style a uno. Segunda fila: la Turbo v1.0 entera, también a uno.",
     "First row: Bull style at one. Second row: the full Turbo version one, also at one."),
    ("It goes on our rebuilt base, so the style sits on the same model it came from.",
     "Va sobre nuestra base reconstruida, así que el estilo se apoya en el mismo modelo del que salió."),
    ("The Prompt Enhancer, as written: an old fisherman mending a net on a dock at sunrise. The MOSS Latent, seed one thousand thirteen.",
     "El Prompt Enhancer, en modo \"as written\": un pescador mayor remendando una red en un muelle al amanecer. El MOSS Latent, semilla mil trece."),
    ("The MOSS KSampler with the turbo settings: eight steps, CFG one, euler, simple. And the Image Saver with Telemetry, so the PNG keeps everything.",
     "El MOSS KSampler con los ajustes de la turbo: ocho pasos, CFG uno, euler, simple. Y el Image Saver con Telemetry, para que el PNG lo guarde todo.",
     "The MOSS K Sampler with the turbo settings: eight steps, C F G one, oiler, simple. And the Image Saver with Telemetry, so the P N G keeps everything."),
    ("Run. The first time it also loads the text encoder and the LoRAs, so it takes a little longer. After that, under four seconds an image.",
     "Lanzamos. La primera vez también carga el text encoder y las LoRAs, así que tarda algo más. Después, menos de cuatro segundos por imagen.",
     "Run. The first time it also loads the text encoder and the Loras, so it takes a little longer. After that, under four seconds an image."),
    ("The style, with the full turbo, at eight steps. Measured against the original style with the same turbo: zero point one seven again.",
     "El estilo, con la turbo entera, a ocho pasos. Medido contra el estilo original con la misma turbo: cero coma diecisiete otra vez."),
    ("And for the style without any turbo, switch the turbo row off and use fifteen steps at CFG two point five.",
     "Y para el estilo sin turbo, apaga la fila de la turbo y usa quince pasos a CFG dos coma cinco.",
     "And for the style without any turbo, switch the turbo row off and use fifteen steps at C F G two point five."),
    ("Two rows work, but maybe you want one file to share. A second LoRA Merger: bulle_style at one, plus the turbo at one.",
     "Dos filas funcionan, pero a lo mejor quieres un solo archivo para compartir. Un segundo LoRA Merger: bulle_style a uno, más la turbo a uno.",
     "Two rows work, but maybe you want one file to share. A second Lora Merger: Bull style at one, plus the turbo at one."),
    ("Rank sixty-eight, two hundred fifty-nine megabytes. About one second.",
     "Rango sesenta y ocho, doscientos cincuenta y nueve megas. Alrededor de un segundo."),
    ("That's big. The MOSS LoRA Resize brings it down to rank thirty-two: one hundred twenty-six megabytes, and ninety-nine point eight percent of the energy.",
     "Es grande. El MOSS LoRA Resize lo baja a rango treinta y dos: ciento veintiséis megas, y un noventa y nueve coma ocho por ciento de la energía.",
     "That's big. The MOSS Lora Resize brings it down to rank thirty-two: one hundred twenty-six megabytes, and ninety-nine point eight percent of the energy."),
    ("One honest note: the single file is not identical to the two rows. It's zero point two eight seeds away. Small details move at eight steps.",
     "Una nota honesta: el archivo único no es idéntico a las dos filas. Está a cero coma veintiocho semillas. A ocho pasos se mueven detalles pequeños."),
    ("Let's recap. A LoRA is the difference between two models. To extract it, rebuild the base and let the Extractor subtract.",
     "Repasemos. Una LoRA es la diferencia entre dos modelos. Para extraerla, reconstruye la base y deja que el Extractor reste.",
     "Let's recap. A Lora is the difference between two models. To extract it, rebuild the base and let the Extractor subtract."),
    ("The LoRA Merger adds and subtracts LoRAs. Negative weights cut one LoRA out of another.",
     "El LoRA Merger suma y resta LoRAs. Los pesos negativos quitan una LoRA de dentro de otra.",
     "The Lora Merger adds and subtracts Loras. Negative weights cut one Lora out of another."),
    ("Don't trust the recipe's number blindly: measure. Here it was minus zero point three, not minus zero point five.",
     "No te fíes a ciegas del número de la receta: mide. Aquí era menos cero coma tres, no menos cero coma cinco."),
    ("And the Resize makes the final file smaller, keeping almost all of it.",
     "Y el Resize hace más pequeño el archivo final, conservando casi todo."),
    ("The workflow is free to download, with the extraction already muted so it doesn't run by accident. Link in the description.",
     "El workflow se descarga gratis, con la extracción ya silenciada para que no se lance sin querer. Enlace en la descripción."),
    ("If this helped you, leave a like and subscribe: it's free and it helps me keep making these. See you in the next one!",
     "Si te ha servido, deja un like y suscríbete: es gratis y me ayuda a seguir haciendo estos vídeos. ¡Nos vemos en el siguiente!"),
]


def ls_old(key, prompt="fisherman"):
    return f"/view?filename={prompt}_{key}_00001_.png&subfolder={LS_OLD}&type=output"


def ls_crops_html(urls, labels, top=120):
    """512 x 512 crops at 100 % around the fisherman's face, side by side, a label over each."""
    side = 512 * 1920 / W * 0.78
    x, y = LS_CROP
    cells = "".join(f'<div style="text-align:center"><div style="color:#fff;font:600 24px Segoe UI,sans-serif;margin-bottom:12px">{t}</div>'
                    f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:6px;position:relative">'
                    f'<img src="{u}" style="position:absolute;max-width:none;width:{1024 * side / 512}px;height:{1024 * side / 512}px;'
                    f'left:{-x * side / 512}px;top:{-y * side / 512}px"></div></div>' for u, t in zip(urls, labels))
    return f'<div style="display:flex;gap:24px;margin-top:{top}px">{cells}</div>'


def ls_recipe_card(es, step):
    """Bulle d'Encre's recipe as three stacked layers; step 1 frames the style LoRA as private."""
    rows = [("SFT + 30 % Base (capas 24-29)" if es else "SFT + 30 % Base (layers 24-29)", "#2e4a7a"),
            ("Turbo 8-steps v1.0 × 0.5", "#4a3a1a"),
            ("LoRA de estilo × 0.8" + (" (privada)" if step else "") if es else "style LoRA × 0.8" + (" (private)" if step else ""), "#4a2a4a")]
    blocks = "".join(f'<div style="width:900px;border-radius:14px;background:{c};padding:26px;font:700 40px Segoe UI,sans-serif;margin:14px auto">{t}</div>'
                     for t, c in reversed(rows))
    return (f'<div style="margin-top:180px;color:#fff;text-align:center"><div style="font:800 54px Segoe UI,sans-serif;color:#ffcc33;margin-bottom:24px">'
            f'Bulle d\'Encre = {"una receta" if es else "a recipe"}</div>{blocks}</div>')


def ls_minus05_html(es):
    """Bulle d'Encre against the extracted LoRA cut at -0.5, at 15 steps: the café and the fisherman (measured renders)."""
    lab = (("Bulle d'Encre", "cirugía −0,5") if es else ("Bulle d'Encre", "surgery −0.5"))
    cell = lambda u, t: (f'<div style="text-align:center"><img src="{u}" style="height:400px;border-radius:6px">'
                         f'<div style="color:#fff;font:700 28px Segoe UI,sans-serif;margin-top:10px">{t}</div></div>')
    return ('<div style="display:grid;grid-template-columns:auto auto auto auto;gap:24px;margin-top:150px">'
            + cell(ls_old("bulle_15", "cafe"), lab[0]) + cell(ls_old("style_15", "cafe"), lab[1])
            + cell(ls_old("bulle_15"), lab[0]) + cell(ls_old("style_15"), lab[1]) + '</div>')


def ls_distance_html(es, mark=None):
    """The measured distances to the original style LoRA (in seeds, 4 prompts, 15 steps, CFG 2.5)."""
    rows = [("−0.30", "0.17"), ("−0.5", "0.46"), ("sin cirugía" if es else "no surgery", "0.55")]
    cell = lambda t, w, c="#e6e6ea", b=600: f'<div style="width:{w}px;font:{b} 40px Segoe UI,sans-serif;color:{c};text-align:center">{t}</div>'
    body = "".join(f'<div style="display:flex;padding:16px 0;border-radius:10px;{"background:#3a3214;" if r[0] == mark else ""}">'
                   + cell(r[0], 360, "#ffcc33", 800) + cell(r[1], 300) + "</div>" for r in rows)
    title = ("Distancia a la LoRA de estilo original (en semillas, 4 prompts)" if es else
             "Distance to the original style LoRA (in seeds, 4 prompts)")
    return (f'<div style="margin-top:220px;color:#fff;text-align:center"><div style="font:600 32px Segoe UI,sans-serif;color:#c9c9d1;margin-bottom:26px">{title}</div>'
            f'<div style="display:inline-block;background:#15151b;border-radius:14px;padding:20px 40px">{body}</div>'
            f'<div style="font:600 28px Segoe UI,sans-serif;color:#c9c9d1;margin-top:22px">{"1,0 = tan distinto como otra semilla" if es else "1.0 = as different as another seed"}</div></div>')


def ls_check(page_url, url):
    """The take's fisherman against the measured one (MAE at 512): the take stops if it is another picture."""
    import io
    import urllib.parse
    import urllib.request
    from PIL import Image
    import numpy as np
    base = urllib.parse.urlsplit(page_url)
    got = Image.open(io.BytesIO(urllib.request.urlopen(f"{base.scheme}://{base.netloc}{url}").read())).convert("RGB").resize((512, 512))
    ref = Image.open(r"I:/HaDeS_MOSS/videos/advanced/adv_lora_surgery/measure/full_fisherman_8.png").convert("RGB").resize((512, 512))
    mae = float(np.abs(np.asarray(got, dtype=np.float32) - np.asarray(ref, dtype=np.float32)).mean())
    print(f"take vs measured fisherman: MAE {mae:.2f}", flush=True)
    if mae > 2.0:
        raise SystemExit(f"the take's fisherman is not the measured one (MAE {mae:.2f})")


async def ls_text(d, key, size=(440, 330)):
    """A Preview as Text node for a tool's report, sized so the report reads."""
    n = await ap_add(d, "PreviewAny", "Preview as Text", LS_POS[key])
    await d.js("([id, s]) => { const n = window.__hv.node(id); n.setSize(s); n.setDirtyCanvas(true, true); }", [n, list(size)])
    return n


async def ls_prepare(page):
    """Off camera: the measured renders must be in comfy_output/lora_surgery."""
    for key in ("bulle_15|fisherman", "extracted_15|fisherman", "style_15|fisherman", "c30_15|fisherman", "bulle_15|cafe", "style_15|cafe"):
        k, p = key.split("|")
        if not (Path("I:/HaDeS_MOSS/videos/comfy_output") / LS_OLD / f"{p}_{k}_00001_.png").exists():
            raise SystemExit(f"measured render {key} missing")


async def lora_surgery(d: Director):
    """Advanced workflow #4 as a lesson: Bulle d'Encre's recipe, the base rebuilt, the LoRA extracted, the turbo cut out at -0.30
    (with why not -0.5), the style + the full turbo at 8 steps, one file and a rank-32 copy, a recap."""
    P = LS_POS
    es = d.language == "es"
    say_after = d._after_voice
    await d.camera({"box": [-60, -80, 1400, 1000]}, 10, pad=60, top=60)
    await d.move(W * 0.97, H * 0.55, 10)
    clab = (("Bulle d'Encre", "extraída", "cirugía −0,5", "cirugía −0,30") if es else
            ("Bulle d'Encre", "extracted", "surgery −0.5", "surgery −0.30"))
    curls = [ls_old("bulle_15"), ls_old("extracted_15"), ls_old("style_15"), ls_old("c30_15")]

    # 1. what we'll learn
    await d.js("(h) => window.__hv.show(h, 10)",
               ls_crops_html([curls[0], curls[2], curls[3]], [clab[0], clab[2], clab[3]], top=60) +
               '<div style="position:fixed;left:0;right:0;bottom:90px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:600 36px Segoe UI,sans-serif;color:#ffcc33;margin-top:14px">{"Workflow avanzado n.º 4" if es else "Advanced workflow #4"}'
               ' · LoRA Surgery</div></div>')
    await d.hold(0.8)
    for i in range(4):
        await d.say(i)

    # 2. a model is a recipe
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ls_recipe_card(es, 0))
    await d.say(4)
    await d.say(5)
    await say_after(0.2)
    await d.js("(h) => window.__hv.show(h)", ls_recipe_card(es, 1))
    await d.say(6)
    await d.say(7)
    await d.say(8)
    await say_after(1.0)
    await d.js("() => window.__hv.hide()")

    # 3. the base, rebuilt
    await d.say(9)
    load = await ap_add(d, "HaDeSMossLoader", "MOSS Loader", P["load"])
    await up_set(d, load, BASIC_LOADER)
    await d.camera([load], 900, pad=140, top=60)
    await basic_point(d, load, "unet_name")
    await d.say(10)
    await say_after(0.25)
    sft = await ap_add(d, "UNETLoader", "Load Diffusion Model", P["sft"])
    await up_set(d, sft, {"unet_name": LT_SFT, "weight_dtype": "default"})
    await d.camera({"box": [-60, -80, 900, 900]}, 900, pad=60, top=60)
    await basic_point(d, sft, "unet_name")
    await d.say(11)
    await say_after(0.25)
    bm = await ap_add(d, "HaDeSMossModelBlockMerger", "MOSS Model Block Merger", P["bm"])
    await up_set(d, bm, LT_BM)
    async with d.fast(2.0):
        await d.camera({"box": [-60, -80, 1000, 1000]}, 600, pad=60, top=60)
        await basic_link(d, sft, "MODEL", bm, "model1", 700)
        await basic_link(d, load, "MODEL", bm, "model2", 700)
    await d.say(12)
    await say_after(0.25)
    await ml_zoom(d, bm, ["layers_24_29"])
    await basic_point(d, bm, "layers_24_29")
    await d.say(13)
    await say_after(0.25)
    await d.say(14)
    await say_after(0.5)

    # 4. the Extractor
    await d.camera({"box": [-60, -80, 1450, 1100]}, 1000, pad=60, top=60)
    bulle = await ap_add(d, "UNETLoader", "Load Diffusion Model", P["bulle"])
    await up_set(d, bulle, {"unet_name": LT_BULLE, "weight_dtype": "default"})
    await basic_point(d, bulle, "unet_name")
    await d.say(15)
    await say_after(0.25)
    ext = await ap_add(d, "HaDeSMossLoraExtractor", "MOSS LoRA Extractor", P["ext"])
    await up_set(d, ext, {"rank": 64, "export_filename": LS_EXT, "overwrite_existing": True, "tuned_name": "Bulle d'Encre",
                          "base_name": "SFT + 30% Base in layers 24-29 (Block Merger)"})
    async with d.fast(3.0):
        await basic_link(d, bulle, "MODEL", ext, "model_tuned", 700)
        await basic_link(d, bm, "MODEL", ext, "model_base", 700)
        ext_text = await ls_text(d, "ext_text")
        await basic_link(d, ext, "info", ext_text, "source", 700)
    await d.say(16)
    await say_after(0.25)
    await ml_zoom(d, ext, ["rank"])
    await basic_point(d, ext, "rank")
    await d.say(17)
    await say_after(0.25)
    await ml_zoom(d, ext, ["export_filename"])
    await basic_point(d, ext, "export_filename")
    await d.say(18)
    await say_after(0.1)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([ext_text], 1000, pad=60, top=60)
    await d.say(19)
    await say_after(0.25)
    await d.say(20)
    await say_after(1.0)
    await ts_group(d, "1. Extract", [-30, -80, 1400, 1180], "#335533")

    # 5. the surgery
    await d.camera({"box": [880, -80, 1950, 1000]}, 1000, pad=60, top=60)
    await d.say(21)
    merger = await ap_add(d, "HaDeSMossLoraMerger", "MOSS LoRA Merger", P["merger"])
    await up_set(d, merger, {"lora_a": LS_EXT + ".safetensors", "weight_a": 1.0, "lora_b": LS_TURBO, "weight_b": 0.0, "rank": 66,
                             "export_filename": LS_STYLE, "dynamic_energy": 0.0, "overwrite_existing": True})
    async with d.fast(2.0):
        await d.camera({"box": [-60, -80, 1950, 1000]}, 600, pad=60, top=60)
        await basic_link(d, bm, "MODEL", merger, "model", 900)
        m_text = await ls_text(d, "merger_text")
        await basic_link(d, merger, "info", m_text, "source", 600)
    await say_after(0.1)
    await ml_zoom(d, merger, ["weight_a", "weight_b"])
    await basic_point(d, merger, "weight_b")
    await d.say(22)
    await say_after(0.25)
    await ml_zoom(d, merger, ["lora_a", "weight_a", "lora_b"])
    await basic_point(d, merger, "lora_a")
    await d.say(23)
    await d.hold(0.4)
    await basic_point(d, merger, "lora_b")
    await say_after(0.25)
    await d.say(24)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ls_minus05_html(es))
    await d.say(25)
    await say_after(1.0)
    await d.js("() => window.__hv.hide()")
    await d.say(26)
    await d.say(27)
    await say_after(0.25)
    await ml_zoom(d, merger, ["weight_b"])
    await basic_point(d, merger, "weight_b")
    await d.say(28)
    await d.hold(0.3)
    await up_set(d, merger, {"weight_b": -0.30})
    await say_after(0.25)
    await d.say(29)
    await say_after(1.5)
    await ml_zoom(d, merger, ["rank", "export_filename"])
    await basic_point(d, merger, "rank")
    await d.say(30)
    await say_after(0.1)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([merger, m_text], 1000, pad=60, top=60)
    await say_after(0.8)

    # 6. did it work?
    await d.js("(h) => window.__hv.show(h)", ls_crops_html(curls, clab))
    await d.say(31)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ls_distance_html(es))
    await d.say(32)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ls_distance_html(es, mark="−0.30"))
    await d.say(33)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ls_crops_html(curls, clab))
    await d.say(34)
    await say_after(1.0)
    await d.js("() => window.__hv.hide()")
    await ts_group(d, "2. Cut the turbo out (-0.30)", [1370, -80, 480, 1180], "#664433")

    # 7. the style + the full turbo at 8 steps
    await d.camera({"box": [-60, 1100, 2750, 2250]}, 1100, pad=60, top=60)
    ml = await ap_add(d, "HaDeSMossMultiLora", "MOSS Multi-LoRA", P["ml"])
    await up_set(d, ml, {"loras": json.dumps([{"name": LS_STYLE + ".safetensors", "on": True, "strength": 1.0},
                                              {"name": LS_TURBO, "on": True, "strength": 1.0}])})
    await d.say(35)
    await say_after(0.25)
    await d.camera([ml], 900, pad=140, top=60)
    await basic_point(d, ml, "hades_multilora_list")
    await d.say(36)
    await say_after(0.25)
    async with d.fast(2.0):
        await d.camera({"box": [-60, -80, 1400, 2250]}, 600, pad=60, top=60)
        await basic_link(d, bm, "MODEL", ml, "model", 900)
    await d.say(37)
    await say_after(0.25)
    await d.camera({"box": [380, 1100, 1900, 2000]}, 1000, pad=60, top=60)
    enh = await ap_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enh"])
    await up_set(d, enh, {"mode": "as written", "width": 1024, "height": 1024, "seed": 1, "negative": ""})
    async with d.fast(2.0):
        await d.camera({"box": [-60, -80, 1900, 2250]}, 600, pad=60, top=60)
        await basic_link(d, load, "CLIP", enh, "clip", 700)
        await d.camera({"box": [380, 1100, 1900, 2000]}, 600, pad=60, top=60)
        await pe_type(d, enh, "idea", LS_PROMPT, paste=True)
    latent = await ap_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": LS_SEED, "control_after_generate": "fixed"})
    await ml_zoom(d, latent, ["seed"])
    await basic_point(d, latent, "seed")
    await d.say(38)
    await say_after(0.25)
    await d.camera({"box": [380, 1100, 2750, 2250]}, 1000, pad=60, top=60)
    ks = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks"])
    await up_set(d, ks, {"steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                         "denoise_mode": "steps / denoise (core)"})
    save = await ap_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["save"])
    await up_set(d, save, {"filename_prefix": "MOSS/HaDeS_MOSS_surgery_%seed%"})
    tel = await ap_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["tel"])
    async with d.fast(4.0):
        await d.camera({"box": LS_BOX}, 600, pad=60, top=60)
        for a, out, b, inp in ((ml, "MODEL", ks, "model"), (enh, "positive", ks, "positive"), (enh, "negative", ks, "negative"),
                               (latent, "latent", ks, "latent_image"), (latent, "seed", ks, "seed"), (load, "VAE", ks, "vae"),
                               (ks, "image", save, "images")):
            await basic_link(d, a, out, b, inp, 600)
        await d.camera({"box": LS_BOX}, 600, pad=60, top=60)
        for i, a in enumerate((load, bm, ml, enh, latent, ks), 1):
            await basic_link(d, a, "info", tel, f"info_{i}", 600)
        await basic_link(d, tel, "json", save, "metadata", 600)
        rep = await ap_add(d, "PreviewAny", "Preview Any", P["rep"])
        await basic_link(d, tel, "report", rep, "source", 600)
    await d.camera([ks], 900, pad=140, top=60)
    await ml_zoom(d, ks, ["steps", "cfg"])
    await basic_point(d, ks, "steps")
    await d.say(39)
    await say_after(0.25)
    await d.camera({"box": [380, 1100, 3250, 2250]}, 900, pad=60, top=60)
    final, secs = await basic_run(d, save)
    ls_check(d.page.url, final)
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs:.1f} s · RTX 5090', ("8 pasos · CFG 1" if es else "8 steps · CFG 1")], top=360))
    await d.say(40)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", ts_pair_html([ls_old("bulle_15"), final],
               (("Bulle d'Encre · 15 pasos", "estilo + turbo entera · 8 pasos") if es else ("Bulle d'Encre · 15 steps", "style + full turbo · 8 steps"))))
    await d.say(41)
    await say_after(0.5)
    await d.js("() => window.__hv.hide()")
    await d.camera([ml], 900, pad=140, top=60)
    await basic_point(d, ml, "hades_multilora_list")
    await d.say(42)
    await say_after(0.5)
    await ts_group(d, "3. The style + the full turbo, 8 steps", [420, 1120, 2780, 1100], "#334466")

    # 8. one file, and smaller
    await d.camera({"box": [1820, -80, 2850, 1000]}, 1000, pad=60, top=60)
    merger2 = await ap_add(d, "HaDeSMossLoraMerger", "MOSS LoRA Merger", P["merger2"])
    await up_set(d, merger2, {"lora_a": LS_STYLE + ".safetensors", "weight_a": 1.0, "lora_b": LS_TURBO, "weight_b": 1.0, "rank": 68,
                              "export_filename": LS_ONE, "dynamic_energy": 0.0, "overwrite_existing": True})
    async with d.fast(2.0):
        await d.camera({"box": LS_TOP}, 600, pad=60, top=60)
        await basic_link(d, bm, "MODEL", merger2, "model", 900)
        m2_text = await ls_text(d, "m2_text")
        await basic_link(d, merger2, "info", m2_text, "source", 600)
    await d.camera([merger2], 900, pad=120, top=60)
    await basic_point(d, merger2, "weight_b")
    await d.say(43)
    await say_after(0.1)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([merger2, m2_text], 1000, pad=60, top=60)
    await d.say(44)
    await say_after(0.25)
    resize = await ap_add(d, "HaDeSMossLoraResize", "MOSS LoRA Resize", P["resize"])
    await up_set(d, resize, {"lora_name": LS_ONE + ".safetensors", "target_rank": 32, "export_filename": LS_R32,
                             "dynamic_energy": 0.0, "overwrite_existing": True})
    async with d.fast(2.0):
        await d.camera({"box": LS_TOP}, 600, pad=60, top=60)
        await basic_link(d, bm, "MODEL", resize, "model", 900)
        rs_text = await ls_text(d, "rs_text")
        await basic_link(d, resize, "info", rs_text, "source", 600)
    await d.camera([resize], 900, pad=120, top=60)
    await basic_point(d, resize, "target_rank")
    await say_after(0.1)
    await lt_run(d)
    await lt_refresh(d)
    await d.camera([resize, rs_text], 1000, pad=60, top=60)
    await d.say(45)
    await say_after(0.25)
    await d.say(46)
    await say_after(0.5)
    await ts_group(d, "4. One file", [1850, -80, 980, 1180], "#553355")

    # 9. recap and closing
    recap = (["1 · LoRA = diferencia entre dos modelos", "2 · pesos negativos quitan una LoRA", "3 · mide: −0,30, no −0,5", "4 · Resize: más pequeña"] if es else
             ["1 · LoRA = the difference between two models", "2 · negative weights cut a LoRA out", "3 · measure: −0.30, not −0.5", "4 · Resize: smaller file"])
    for i in range(4):
        await d.js("(h) => window.__hv.show(h)", ap_card(recap[:i + 1]))
        await d.say(47 + i)
        await say_after(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": LS_BOX}, 1200, pad=60, top=60)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1300px;margin-top:300px">'
               f'<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">{"Descarga gratis (enlace en la descripción)" if es else "Free download (link in the description)"}</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">examples/advanced/HaDeS MOSS Suite - LoRA Surgery.json</div></div>')
    await d.say(51)
    await say_after(0.25)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               ls_crops_html([curls[0], curls[2], curls[3]], [clab[0], clab[2], clab[3]], top=60) +
               '<div style="position:fixed;left:0;right:0;bottom:90px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:400 28px Segoe UI,sans-serif;color:#c9c9d1;margin-top:18px">{"Workflow, Suite y modelos: enlaces en la descripción" if es else "Workflow, Suite and models: links in the description"}</div></div>')
    await d.say(52)
    await d.end(1.5)


lora_surgery.prepare = ls_prepare


# ---- advanced workflow #5: Fixed Pose (a lesson) ----
FP_FILE = None  # built on camera (a lesson); the finished example is the download
FP_SEED = 1013  # the example's seed; measured: the knight equals the example's render (videos/advanced/adv_fixed_pose/MEDIDAS.md)
FP_MAP = "depth_map.png"  # the fisherman's depth map (plain attention), in the recording input folder
FP_KNIGHT = ("Oil painting of a medieval knight in weathered steel armor mending a chainmail cloak on a stone castle parapet at sunset, "
             "warm torches, sea in the background.")
FP_POTTER = ("Photograph of a woman in her thirties, a potter in a linen apron, shaping clay with both hands at a workbench in a sunlit "
             "studio, shelves of vases behind her.")
FP_CTRL = "MOSS\\moss-color-depth-control.safetensors"
FP_TURBO = "MOSS\\turbo_v10\\HaDeS-MOSS-Turbo-8steps-v1.0.safetensors"
FP_CROP = (300, 40)  # 512 x 512 on the knight's visor and beard (1024 image)
FP_POS = {"load": (0, 0), "loras": (0, 380), "depth": (950, 420), "enh": (450, 0), "latent": (950, 0), "ctrl": (1400, 0),
          "turbo": (1400, 620), "ks": (1850, 0), "stage1": (1850, 600), "ks2": (2300, 0), "stage2": (2300, 600), "upmodel": (2750, 0),
          "up": (2750, 150), "enc": (2750, 300), "ks3": (3150, 0), "save": (3600, 0), "tel": (3600, 480), "report": (4050, 480)}
FP_BOX = [-60, -120, 4500, 1100]
FP_S1 = [-60, -120, 2300, 1100]
FP_MEASURE = Path(r"I:/HaDeS_MOSS/videos/advanced/adv_fixed_pose/measure")

FP_LINES = [
    ("Hi, I'm HaDeS, and welcome to advanced workflow number five: fixed pose.",
     "Hola, soy HaDeS, y bienvenido al workflow avanzado número cinco: pose fija."),
    ("Today you'll learn how to take the pose from one photo, and give it to any character you want, just by changing the prompt.",
     "Hoy vas a aprender a sacar la pose de una foto y dársela a cualquier personaje que quieras, solo cambiando el prompt."),
    ("You'll also see why the image the control gives you is soft, and how a second pass brings the detail back.",
     "También verás por qué la imagen que da el control sale suave, y cómo una segunda pasada devuelve el detalle."),
    ("As always, we build it together from an empty canvas, everything is free, and all the links are in the description.",
     "Como siempre, lo montamos juntos desde el lienzo vacío, todo es gratis y tienes todos los enlaces en la descripción."),
    ("It all starts with a depth map. This is my fisherman, and this is his depth map: near things are white, far things are dark.",
     "Todo empieza con un mapa de profundidad. Este es mi pescador, y este es su mapa: lo cercano es blanco y lo lejano, oscuro."),
    ("The map keeps the shape of the body, the head and the hands. It doesn't keep the colours or the clothes. That's what we'll change.",
     "El mapa guarda la forma del cuerpo, la cabeza y las manos. No guarda los colores ni la ropa. Eso es lo que vamos a cambiar."),
    ("One warning before we start. On my install, the Depth Anything preprocessor gave this: a blur, with no outline.",
     "Un aviso antes de empezar. En mi instalación, el preprocesador Depth Anything me dio esto: un borrón, sin contorno."),
    ("With a map like that, the control only follows the rough layout. So always look at your map first. It must show the silhouette.",
     "Con un mapa así, el control solo sigue la composición a grandes rasgos. Así que mira siempre tu mapa primero. Tiene que verse la silueta."),
    ("Let's build it. The MOSS Loader with Base, the text encoder and the VAE.",
     "Vamos a montarlo. El MOSS Loader con Base, el text encoder y el VAE.",
     "Let's build it. The MOSS Loader with Base, the text encoder and the V A E."),
    ("Then a MOSS Multi-LoRA with the control adapter. It's a LoRA that teaches the model to read a depth map. Keep it at one.",
     "Después, un MOSS Multi-LoRA con el adaptador de control. Es una LoRA que enseña al modelo a leer un mapa de profundidad. Déjala a uno.",
     "Then a MOSS Multi Lora with the control adapter. It's a Lora that teaches the model to read a depth map. Keep it at one."),
    ("A Load Image node with our depth map.",
     "Un nodo Load Image con nuestro mapa de profundidad."),
    ("The Prompt Enhancer, as written: an oil painting of a medieval knight mending a chainmail cloak on a castle parapet at sunset.",
     "El Prompt Enhancer, en modo \"as written\": un óleo de un caballero medieval remendando una capa de cota de malla en una muralla al atardecer."),
    ("The MOSS Latent: one megapixel, seed one thousand thirteen.",
     "El MOSS Latent: un megapíxel, semilla mil trece."),
    ("Now the key node: MOSS Control. It takes the prompt and the map, and adds the map to the conditioning.",
     "Ahora el nodo clave: MOSS Control. Toma el prompt y el mapa, y añade el mapa al condicionamiento."),
    ("Color mode off, so it uses the depth only. Strength one, from the first step to the last.",
     "Color mode en off, así que solo usa la profundidad. Fuerza uno, desde el primer paso hasta el último."),
    ("The MOSS Turbo with Turbo 8-steps v1.0: eight steps, CFG one.",
     "El MOSS Turbo con la Turbo 8-steps v1.0: ocho pasos, CFG uno.",
     "The MOSS Turbo with Turbo eight steps, version one: eight steps, C F G one."),
    ("And the MOSS KSampler, with the model from the Multi-LoRA and the prompt from the Control. A Preview Image to see stage one.",
     "Y el MOSS KSampler, con el modelo del Multi-LoRA y el prompt del Control. Un Preview Image para ver la etapa uno.",
     "And the MOSS K Sampler, with the model from the Multi Lora and the prompt from the Control. A Preview Image to see stage one."),
    ("That's stage one: the pose from the depth map.",
     "Esa es la etapa uno: la pose sacada del mapa."),
    ("Stage two fixes a problem. The adapter's image is soft: smooth skin, blurred metal. So we redraw it without the adapter.",
     "La etapa dos arregla un problema. La imagen del adaptador sale suave: piel lisa, metal borroso. Así que la redibujamos sin el adaptador."),
    ("A second MOSS KSampler. Its model comes straight from the Loader, without the adapter. Its prompt comes straight from the Prompt Enhancer, without the control.",
     "Un segundo MOSS KSampler. Su modelo viene directo del Loader, sin el adaptador. Su prompt viene directo del Prompt Enhancer, sin el control.",
     "A second MOSS K Sampler. Its model comes straight from the Loader, without the adapter. Its prompt comes straight from the Prompt Enhancer, without the control."),
    ("And its latent is stage one's picture. Denoise zero point three, in exact mode.",
     "Y su latent es la imagen de la etapa uno. Denoise cero coma tres, en modo exact."),
    ("Why zero point three? Zero point four looked the same, and zero point five changed the picture more. Zero point three keeps the most.",
     "¿Por qué cero coma tres? Cero coma cuatro se veía igual, y cero coma cinco cambiaba más la imagen. Cero coma tres es la que más conserva."),
    ("Stage three: twice the size. An Upscale Model Loader with RealESRGAN x2, and Upscale Image using Model.",
     "Etapa tres: el doble de tamaño. Un Upscale Model Loader con RealESRGAN x2, y Upscale Image using Model.",
     "Stage three: twice the size. An Upscale Model Loader with Real E S R GAN times two, and Upscale Image using Model."),
    ("A VAE Encode turns it back into a latent, and a third MOSS KSampler cleans it at denoise zero point two five.",
     "Un VAE Encode lo vuelve a convertir en latent, y un tercer MOSS KSampler lo limpia a denoise cero coma dos cinco.",
     "A V A E Encode turns it back into a latent, and a third MOSS K Sampler cleans it at denoise zero point two five."),
    ("To finish, the MOSS Image Saver and Suite Telemetry, so the PNG keeps the prompt, the settings and the workflow.",
     "Para terminar, el MOSS Image Saver y Suite Telemetry, para que el PNG guarde el prompt, los ajustes y el workflow.",
     "To finish, the MOSS Image Saver and Suite Telemetry, so the P N G keeps the prompt, the settings and the workflow."),
    ("Run. The first time it loads the models, so it takes longer.",
     "Lanzamos. La primera vez carga los modelos, así que tarda más."),
    ("Here's the fisherman's map, and here's the knight. Same build, same head, same hands mending the net. But a knight, in oil paint.",
     "Aquí está el mapa del pescador, y aquí el caballero. La misma complexión, la misma cabeza, las mismas manos remendando la red. Pero un caballero, al óleo."),
    ("How close is the pose? I measured it: I took the depth of each result and compared it with the map. One means the same depth.",
     "¿Cuánto se parece la pose? Lo medí: saqué la profundidad de cada resultado y la comparé con el mapa. Uno significa la misma profundidad."),
    ("Without control, zero point six two. With control, zero point nine seven seven. After stage two and the x2, it stays at zero point nine seven six.",
     "Sin control, cero coma sesenta y dos. Con control, cero coma novecientos setenta y siete. Tras la etapa dos y el x2, se queda en cero coma novecientos setenta y seis."),
    ("Now stage one against stage two, at one hundred percent. Stage one is soft. Stage two has the chainmail, the beard strands, the rust on the helmet. Same face, same pose.",
     "Ahora la etapa uno contra la etapa dos, al cien por cien. La uno es suave. La dos tiene la cota de malla, los pelos de la barba, el óxido del casco. La misma cara, la misma pose."),
    ("Now the best part. The pose stays; only the prompt changes.",
     "Y ahora lo mejor. La pose se queda; solo cambia el prompt."),
    ("A potter, a woman in her thirties, shaping clay at her workbench. Run again: the models are loaded now, so it's quicker.",
     "Una alfarera, una mujer de unos treinta años, dando forma al barro en su mesa de trabajo. Lanzamos otra vez: los modelos ya están cargados, así que va más rápido."),
    ("The same pose: the head, the shoulders, both hands on the clay.",
     "La misma pose: la cabeza, los hombros, las dos manos sobre el barro."),
    ("But look at her face. I asked for a woman in her thirties, and she looks older. The map carries more than the pose: the build, the head, even the age of the fisherman.",
     "Pero mira su cara. Pedí una mujer de unos treinta, y parece mayor. El mapa lleva más que la pose: la complexión, la cabeza, incluso la edad del pescador."),
    ("It happened on all four seeds I tested. And the knight got the fisherman's beard under his visor. Choose a map close to the character you want.",
     "Pasó en las cuatro semillas que probé. Y el caballero sacó la barba del pescador bajo el casco. Elige un mapa parecido al personaje que quieres."),
    ("Let's recap. One: the depth map holds the pose. Look at it first: it must show the silhouette.",
     "Repasemos. Uno: el mapa de profundidad guarda la pose. Míralo primero: tiene que verse la silueta."),
    ("Two: the control adapter and MOSS Control give the pose, at eight steps with the turbo.",
     "Dos: el adaptador de control y MOSS Control dan la pose, a ocho pasos con la turbo."),
    ("Three: redraw without the adapter at zero point three, to bring the detail back.",
     "Tres: redibuja sin el adaptador a cero coma tres, para recuperar el detalle."),
    ("Four: change only the prompt for a new character, and choose a map that looks like them.",
     "Cuatro: cambia solo el prompt para un personaje nuevo, y elige un mapa que se le parezca."),
    ("You can download this workflow for free from GitHub, with the depth map I used. The link is in the description.",
     "Puedes descargar este workflow gratis desde GitHub, con el mapa de profundidad que usé. El enlace está en la descripción."),
    ("If this helped you, leave a like and subscribe: it's free and it helps me keep making these. See you in the next one!",
     "Si te ha servido, deja un like y suscríbete: es gratis y me ayuda a seguir haciendo estos vídeos. ¡Nos vemos en el siguiente!"),
]


def fp_data(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fp_cols(items, side=520, top=150):
    cells = "".join(f'<div style="text-align:center"><img src="{u}" style="width:{side}px;height:{side}px;border-radius:6px;object-fit:cover">'
                    f'<div style="color:#fff;font:700 26px Segoe UI,sans-serif;margin-top:12px">{t}</div></div>' for u, t in items)
    return f'<div style="display:flex;gap:30px;margin-top:{top}px">{cells}</div>'


def fp_crops(urls, labels, top=140):
    """512 x 512 crops at 100 % on the knight's visor (1024 images)."""
    side = 512 * 1920 / W
    x, y = FP_CROP
    cells = "".join(f'<div style="text-align:center"><div style="color:#fff;font:600 26px Segoe UI,sans-serif;margin-bottom:12px">{t}</div>'
                    f'<div style="width:{side}px;height:{side}px;overflow:hidden;border-radius:6px;position:relative">'
                    f'<img src="{u}" style="position:absolute;max-width:none;width:{1024 * side / 512}px;height:{1024 * side / 512}px;'
                    f'left:{-x * side / 512}px;top:{-y * side / 512}px"></div></div>' for u, t in zip(urls, labels))
    return f'<div style="display:flex;gap:30px;margin-top:{top}px">{cells}</div>'


def fp_pose_html(es, mark=None):
    rows = [("sin control" if es else "no control", "0.62"), ("con control" if es else "with control", "0.977"),
            ("+ etapa 2 + x2" if es else "+ stage 2 + x2", "0.976")]
    cell = lambda t, w, c="#e6e6ea", b=600: f'<div style="width:{w}px;font:{b} 40px Segoe UI,sans-serif;color:{c};text-align:center">{t}</div>'
    body = "".join(f'<div style="display:flex;padding:16px 0;border-radius:10px;{"background:#3a3214;" if r[1] == mark else ""}">'
                   + cell(r[0], 440, "#ffcc33", 800) + cell(r[1], 260) + "</div>" for r in rows)
    title = ("Pose: profundidad del resultado frente al mapa (3 personajes × 4 semillas)" if es else
             "Pose: the result's depth against the map (3 characters × 4 seeds)")
    return (f'<div style="margin-top:220px;color:#fff;text-align:center"><div style="font:600 32px Segoe UI,sans-serif;color:#c9c9d1;margin-bottom:26px">{title}</div>'
            f'<div style="display:inline-block;background:#15151b;border-radius:14px;padding:20px 40px">{body}</div>'
            f'<div style="font:600 28px Segoe UI,sans-serif;color:#c9c9d1;margin-top:22px">{"1 = la misma profundidad" if es else "1 = the same depth"}</div></div>')


async def fp_outputs(d, ids):
    run = next(iter((await d.js("() => fetch('/history?max_items=1').then(r => r.json())")).values()))
    out = run["outputs"]
    view = lambda im: f"/view?filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}"
    return {k: view(out[str(v)]["images"][0]) for k, v in ids.items()}


def fp_check(page_url, url, measured):
    import io
    import urllib.parse
    import urllib.request
    from PIL import Image
    import numpy as np
    base = urllib.parse.urlsplit(page_url)
    got = Image.open(io.BytesIO(urllib.request.urlopen(f"{base.scheme}://{base.netloc}{url}").read())).convert("RGB").resize((512, 512))
    ref = Image.open(FP_MEASURE / measured).convert("RGB").resize((512, 512))
    mae = float(np.abs(np.asarray(got, dtype=np.float32) - np.asarray(ref, dtype=np.float32)).mean())
    print(f"take vs {measured}: MAE {mae:.2f}", flush=True)
    if mae > 2.0:
        raise SystemExit(f"the take's image is not the measured {measured} (MAE {mae:.2f})")


async def fp_prepare(page):
    for f in ("knight_save.png", "potter_save.png", "knight_stage1.png", "knight_stage2.png"):
        if not (FP_MEASURE / f).exists():
            raise SystemExit(f"measured image {f} missing")
    if not (Path(r"I:/HaDeS_MOSS/videos/comfy_input") / FP_MAP).exists():
        raise SystemExit("the depth map is missing from the recording input folder")


async def fixed_pose(d: Director):
    """Advanced workflow #5 as a lesson: the depth map (and a faulty one), stage 1 with the control adapter and MOSS Control, stage 2
    without the adapter at 0.3, stage 3 x2, a run with the measured pose and 100 % crops, the prompt changed to a potter, a recap."""
    P = FP_POS
    es = d.language == "es"
    say_after = d._after_voice
    fisher = fp_data(r"I:/HaDeS_MOSS/videos/comfy_input/fisherman_video1.png")
    dmap = fp_data(r"I:/HaDeS_MOSS/videos/comfy_input/depth_map.png")
    blur = fp_data(r"I:/HaDeS_MOSS/advanced_workflows/fixed_pose/depth_installed.png")
    knight = fp_data(FP_MEASURE / "knight_stage1.png")
    potter = fp_data(FP_MEASURE / "potter_save.png")
    await d.camera({"box": [-60, -80, 1400, 1000]}, 10, pad=60, top=60)
    await d.move(W * 0.97, H * 0.55, 10)

    # 1. what we'll learn
    await d.js("(h) => window.__hv.show(h, 10)",
               fp_cols([(dmap, "depth map" if not es else "mapa de profundidad"), (knight, "knight" if not es else "caballero"),
                        (potter, "potter" if not es else "alfarera")], side=460, top=60) +
               '<div style="position:fixed;left:0;right:0;bottom:90px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:600 36px Segoe UI,sans-serif;color:#ffcc33;margin-top:14px">{"Workflow avanzado n.º 5" if es else "Advanced workflow #5"}'
               ' · Fixed Pose</div></div>')
    await d.hold(0.8)
    for i in range(4):
        await d.say(i)

    # 2. the depth map
    await say_after(0.25)
    lab = (("la foto", "su mapa", "mapa defectuoso") if es else ("the photo", "its depth map", "a faulty map"))
    await d.js("(h) => window.__hv.show(h)", fp_cols([(fisher, lab[0]), (dmap, lab[1])], side=620))
    await d.say(4)
    await d.say(5)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", fp_cols([(dmap, lab[1]), (blur, lab[2])], side=620))
    await d.say(6)
    await d.say(7)
    await say_after(1.0)
    await d.js("() => window.__hv.hide()")

    # 3. stage 1
    load = await ap_add(d, "HaDeSMossLoader", "MOSS Loader", P["load"])
    await up_set(d, load, BASIC_LOADER)
    await d.camera([load], 900, pad=140, top=60)
    await basic_point(d, load, "unet_name")
    await d.say(8)
    await say_after(0.25)
    await d.camera({"box": [-60, -80, 900, 900]}, 900, pad=60, top=60)
    loras = await ap_add(d, "HaDeSMossMultiLora", "MOSS Multi-LoRA", P["loras"])
    await up_set(d, loras, {"enabled": True, "loras": json.dumps([{"name": FP_CTRL, "on": True, "strength": 1.0}])})
    async with d.fast(1.6):
        await basic_link(d, load, "MODEL", loras, "model", 700)
    await d.camera([loras], 900, pad=140, top=60)
    await basic_point(d, loras, "hades_multilora_list")
    await d.say(9)
    await say_after(0.25)
    await d.camera({"box": [380, -80, 1400, 1000]}, 900, pad=60, top=60)
    depth = await ap_add(d, "LoadImage", "Load Image", P["depth"])
    await up_set(d, depth, {"image": FP_MAP})
    await d.camera([depth], 900, pad=140, top=60)
    await d.say(10)
    await say_after(0.25)
    await d.camera({"box": [-60, -80, 1400, 1000]}, 900, pad=60, top=60)
    enh = await ap_add(d, "HaDeSMossPromptEnhancer", "MOSS Prompt Enhancer", P["enh"])
    await up_set(d, enh, {"mode": "as written", "width": 1024, "height": 1024, "seed": 1, "negative": "worst quality, low quality"})
    async with d.fast(2.0):
        await basic_link(d, load, "CLIP", enh, "clip", 700)
        await pe_type(d, enh, "idea", FP_KNIGHT, paste=True)
    await ml_zoom(d, enh, ["mode"])
    await basic_point(d, enh, "mode")
    await d.say(11)
    await say_after(0.25)
    await d.camera({"box": [880, -80, 1400, 900]}, 900, pad=60, top=60)
    latent = await ap_add(d, "HaDeSMossLatent", "MOSS Latent", P["latent"])
    await up_set(d, latent, {"aspect_ratio": "1:1", "megapixels": 1.0, "seed": FP_SEED, "control_after_generate": "fixed"})
    await ml_zoom(d, latent, ["seed"])
    await basic_point(d, latent, "seed")
    await d.say(12)
    await say_after(0.25)
    await d.camera({"box": [-60, -80, 1850, 1100]}, 900, pad=60, top=60)
    ctrl = await ap_add(d, "HaDeSMossControl", "MOSS Control", P["ctrl"])
    await up_set(d, ctrl, {"color_mode": "off (depth only)", "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0})
    async with d.fast(3.0):
        for a, out, inp in ((enh, "positive", "positive"), (enh, "negative", "negative"), (load, "VAE", "vae"),
                            (latent, "latent", "latent"), (depth, "IMAGE", "depth_map")):
            await basic_link(d, a, out, ctrl, inp, 600)
    await d.camera([ctrl], 900, pad=140, top=60)
    await d.say(13)
    await say_after(0.25)
    await ml_zoom(d, ctrl, ["color_mode", "strength"])
    await basic_point(d, ctrl, "color_mode")
    await d.say(14)
    await say_after(0.25)
    await d.camera({"box": [1350, 500, 1850, 1100]}, 900, pad=60, top=60)
    turbo = await ap_add(d, "HaDeSMossTurbo", "MOSS Turbo", P["turbo"])
    await up_set(d, turbo, {"lora_name": FP_TURBO, "strength": 1.0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"})
    await ml_zoom(d, turbo, ["lora_name", "steps", "cfg"])
    await basic_point(d, turbo, "lora_name")
    await d.say(15)
    await say_after(0.25)
    await d.camera({"box": FP_S1}, 900, pad=60, top=60)
    ks = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks"])
    await up_set(d, ks, {"denoise": 1.0, "denoise_mode": "steps / denoise (core)"})
    s1 = await ap_add(d, "PreviewImage", "Preview Image", P["stage1"])
    async with d.fast(3.0):
        for a, out, inp in ((loras, "MODEL", "model"), (ctrl, "positive", "positive"), (ctrl, "negative", "negative"),
                            (latent, "latent", "latent_image"), (latent, "seed", "seed"), (load, "VAE", "vae"), (turbo, "turbo", "turbo")):
            await basic_link(d, a, out, ks, inp, 600)
        await basic_link(d, ks, "image", s1, "images", 600)
    await d.say(16)
    await say_after(0.25)
    await ts_group(d, "1. The pose from the depth map (control, 8 steps)", [-30, -80, 2290, 1130], "#335533")
    await d.say(17)
    await say_after(0.5)

    # 4. stage 2
    await d.camera({"box": [1800, -80, 2750, 1000]}, 900, pad=60, top=60)
    await d.say(18)
    ks2 = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks2"])
    await up_set(d, ks2, {"denoise": 0.3, "denoise_mode": "exact"})
    s2 = await ap_add(d, "PreviewImage", "Preview Image", P["stage2"])
    async with d.fast(3.0):
        await d.camera({"box": [-60, -120, 2750, 1100]}, 600, pad=60, top=60)
        for a, out, inp in ((load, "MODEL", "model"), (enh, "positive", "positive"), (enh, "negative", "negative"),
                            (latent, "seed", "seed"), (load, "VAE", "vae"), (turbo, "turbo", "turbo"), (ks, "latent", "latent_image")):
            await basic_link(d, a, out, ks2, inp, 600)
        await basic_link(d, ks2, "image", s2, "images", 600)
    await d.camera([ks2], 900, pad=140, top=60)
    await d.say(19)
    await say_after(0.25)
    await ml_zoom(d, ks2, ["denoise", "denoise_mode"])
    await basic_point(d, ks2, "denoise")
    await d.say(20)
    await say_after(0.25)
    await d.say(21)
    await say_after(0.25)
    await ts_group(d, "2. Redraw without control at 0.3", [2270, -80, 440, 930], "#334466")

    # 5. stage 3
    await d.camera({"box": [2700, -80, 4500, 1100]}, 1000, pad=60, top=60)
    upm = await ap_add(d, "UpscaleModelLoader", "Load Upscale Model", P["upmodel"])
    await up_set(d, upm, {"model_name": "RealESRGAN_x2plus.pth"})
    up = await ap_add(d, "ImageUpscaleWithModel", "Upscale Image (using Model)", P["up"])
    async with d.fast(2.0):
        await d.camera({"box": [1800, -80, 4500, 1100]}, 600, pad=60, top=60)
        await basic_link(d, upm, "UPSCALE_MODEL", up, "upscale_model", 600)
        await basic_link(d, ks2, "image", up, "image", 600)
    await d.say(22)
    await say_after(0.25)
    enc = await ap_add(d, "VAEEncode", "VAE Encode", P["enc"])
    ks3 = await ap_add(d, "HaDeSMossKSampler", "MOSS KSampler", P["ks3"])
    await up_set(d, ks3, {"denoise": 0.25, "denoise_mode": "steps / denoise (core)"})
    async with d.fast(3.0):
        await d.camera({"box": FP_BOX}, 600, pad=60, top=60)
        await basic_link(d, up, "IMAGE", enc, "pixels", 600)
        await basic_link(d, load, "VAE", enc, "vae", 600)
        for a, out, inp in ((load, "MODEL", "model"), (enh, "positive", "positive"), (enh, "negative", "negative"),
                            (enc, "LATENT", "latent_image"), (latent, "seed", "seed"), (load, "VAE", "vae"), (turbo, "turbo", "turbo")):
            await basic_link(d, a, out, ks3, inp, 600)
    await d.camera([ks3], 900, pad=140, top=60)
    await ml_zoom(d, ks3, ["denoise"])
    await basic_point(d, ks3, "denoise")
    await d.say(23)
    await say_after(0.25)
    await d.camera({"box": [2700, -80, 4500, 1100]}, 900, pad=60, top=60)
    save = await ap_add(d, "HaDeSMossImageSaver", "MOSS Image Saver", P["save"])
    await up_set(d, save, {"filename_prefix": "MOSS/HaDeS_MOSS_fixed_pose_%seed%"})
    tel = await ap_add(d, "HaDeSMossSuiteTelemetry", "MOSS Suite Telemetry", P["tel"])
    async with d.fast(4.0):
        await d.camera({"box": FP_BOX}, 600, pad=60, top=60)
        await basic_link(d, ks3, "image", save, "images", 600)
        for i, a in enumerate((load, loras, enh, latent, ctrl, ks, ks2, ks3), 1):
            await basic_link(d, a, "info", tel, f"info_{i}", 600)
        await basic_link(d, tel, "json", save, "metadata", 600)
        rep = await ap_add(d, "PreviewAny", "Preview Any", P["report"])
        await basic_link(d, tel, "report", rep, "source", 600)
    await d.say(24)
    await say_after(0.25)
    await ts_group(d, "3. x2", [2720, -80, 1740, 1050], "#553355")

    # 6. the run
    await d.camera({"box": FP_BOX}, 900, pad=60, top=60)
    ids = {"stage1": s1, "stage2": s2, "save": save}
    await d.say(25)
    _, secs = await basic_run(d, save)
    outs = await fp_outputs(d, ids)
    fp_check(d.page.url, outs["save"], "knight_save.png")
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs:.1f} s · RTX 5090', ("3 etapas · 2048 × 2048" if es else "3 stages · 2048 × 2048")], top=360))
    await say_after(0.8)
    await d.js("(h) => window.__hv.show(h)", fp_cols([(dmap, lab[1]), (outs["save"], "caballero · 2048" if es else "knight · 2048")], side=640))
    await d.say(26)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", fp_pose_html(es))
    await d.say(27)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", fp_pose_html(es, mark="0.977"))
    await d.say(28)
    await say_after(0.5)
    await d.js("(h) => window.__hv.show(h)", fp_crops([outs["stage1"], outs["stage2"]],
               (("etapa 1 · con el adaptador", "etapa 2 · sin él, a 0,3") if es else ("stage 1 · with the adapter", "stage 2 · without it, at 0.3"))))
    await d.say(29)
    await say_after(1.0)
    await d.js("() => window.__hv.hide()")

    # 7. another character
    await d.camera([enh], 900, pad=140, top=60)
    await d.say(30)
    await say_after(0.25)
    async with d.fast(2.0):
        await pe_type(d, enh, "idea", FP_POTTER, paste=True)
    await d.say(31)
    await d.camera({"box": FP_BOX}, 900, pad=60, top=60)
    _, secs2 = await basic_run(d, save)
    outs2 = await fp_outputs(d, ids)
    fp_check(d.page.url, outs2["save"], "potter_save.png")
    await d.move(W * 0.97, H * 0.55)
    await d.js("(h) => window.__hv.show(h)", ap_card([f'{secs2:.1f} s', ("mismo mapa, otro prompt" if es else "same map, new prompt")], top=360))
    await say_after(0.8)
    await d.js("(h) => window.__hv.show(h)", fp_cols([(dmap, lab[1]), (outs["save"], "caballero" if es else "knight"),
                                                       (outs2["save"], "alfarera" if es else "potter")], side=560))
    await d.say(32)
    await say_after(0.25)
    await d.js("(h) => window.__hv.show(h)", fp_cols([(fisher, "el pescador" if es else "the fisherman"),
                                                       (outs2["save"], "«en sus treinta»" if es else "\u201cin her thirties\u201d")], side=640))
    await d.say(33)
    await d.say(34)
    await say_after(0.8)
    await d.js("() => window.__hv.hide()")

    # 8. recap and closing
    recap = (["1 · el mapa guarda la pose: míralo primero", "2 · adaptador + MOSS Control, 8 pasos", "3 · redibuja sin adaptador a 0,3",
              "4 · cambia solo el prompt; elige un mapa parecido"] if es else
             ["1 · the map holds the pose: look at it first", "2 · adapter + MOSS Control, 8 steps", "3 · redraw without the adapter at 0.3",
              "4 · change only the prompt; pick a similar map"])
    for i in range(4):
        await d.js("(h) => window.__hv.show(h)", ap_card(recap[:i + 1]))
        await d.say(35 + i)
        await say_after(0.25)
    await d.js("() => window.__hv.hide()")
    await d.camera({"box": FP_BOX}, 1200, pad=60, top=60)
    await d.js("(h) => window.__hv.show(h)",
               '<div style="color:#fff;text-align:center;max-width:1300px;margin-top:300px">'
               f'<div style="font:600 30px Segoe UI,sans-serif;color:#c9c9d1">{"Descarga gratis (enlace en la descripción)" if es else "Free download (link in the description)"}</div>'
               '<div style="font:700 44px Segoe UI,sans-serif;margin-top:22px">examples/advanced/HaDeS MOSS Suite - Fixed Pose.json</div></div>')
    await d.say(39)
    await say_after(0.25)
    await d.js("(h) => { const el = document.getElementById('__hv_show'); el.innerHTML = h; }",
               fp_cols([(dmap, lab[1]), (outs["save"], "caballero" if es else "knight"), (outs2["save"], "alfarera" if es else "potter")], side=460, top=60) +
               '<div style="position:fixed;left:0;right:0;bottom:90px;text-align:center;color:#fff">'
               '<div style="font:900 72px Segoe UI Black,sans-serif">HaDeS MOSS Suite</div>'
               f'<div style="font:400 28px Segoe UI,sans-serif;color:#c9c9d1;margin-top:18px">{"Workflow, Suite y modelos: enlaces en la descripción" if es else "Workflow, Suite and models: links in the description"}</div></div>')
    await d.say(40)
    await d.end(1.5)


fixed_pose.prepare = fp_prepare


SCENES = {"basic_generation": (None, basic_generation, BASIC_LINES),
          "layout_test": ("HaDeS MOSS Suite - Layout from a Sentence.json", layout_test, LAYOUT_TEST_LINES),
          "layout_sentence": ("HaDeS MOSS Suite - Layout from a Sentence.json", layout_sentence, LAYOUT_SENTENCE_LINES),
          "turbo_8_steps_upscale_x2": ("HaDeS MOSS Suite - Turbo 8 steps + Upscale x2.json", turbo_upscale_x2,
                                       TURBO_UPSCALE_LINES),
          "upscale_telemetry": (None, upscale_telemetry, UP_LINES),
          "prompt_enhancer": (None, prompt_enhancer, PE_LINES),
          "lora_toolkit": (None, lora_toolkit, LT_LINES),
          "metadata": (None, metadata_video, MD_LINES),
          "merge_save": (None, merge_save, MS_LINES),
          "compare": (None, compare_video, CM_LINES),
          "layout_text": (None, layout_text, L8_LINES),
          "controlnet": (None, controlnet_video, CN_LINES),
          "img2img": (None, img2img_video, I2I_LINES),
          "hms_v1": (None, hms_v1_video, HMS_LINES),
          "adv_poster": (AP_FILE, adv_poster, AP_LINES),
          "adv_merge_lab": (ML_FILE, merge_lab, ML_LINES),
          "adv_two_stage": (TS_FILE, two_stage, TS_LINES),
          "adv_lora_surgery": (LS_FILE, lora_surgery, LS_LINES),
          "adv_fixed_pose": (FP_FILE, fixed_pose, FP_LINES)}

# spoken forms where the written caption reads badly aloud
SPOKEN = {"DRAFT LAYOUT": "Draft layout"}


# ---- voice ----

def voice_job(ref, lang, text):
    h = hashlib.sha256(f"{Path(ref).resolve()}|{Path(ref).stat().st_mtime_ns}|{lang}|{text}".encode()).hexdigest()[:16]
    return f"{lang}_{h}.wav"


def synth(lines, lang, ref, cache, python):
    """Each line's voice (cached), made in ComfyUI's Python; returns {line index: (wav path, seconds)}."""
    cache.mkdir(parents=True, exist_ok=True)
    k = 0 if lang == "en" else 1
    # a line may carry a third item: how the English caption is said aloud ("v1.0" -> "version one")
    texts = {i: (l[2] if lang == "en" and len(l) > 2 else l[k]) for i, l in enumerate(lines)}
    for a, b in SPOKEN.items():
        texts = {i: t.replace(a, b) for i, t in texts.items()}
    jobs = {i: cache / voice_job(ref, lang, t) for i, t in texts.items()}
    todo = [{"text": texts[i], "out": str(p)} for i, p in jobs.items() if not p.exists()]
    if todo:
        (cache / "jobs.json").write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
        subprocess.run([python, "-X", "utf8", __file__, "_voice", str(ref), lang, str(cache / "jobs.json")], check=True)
    def seconds(wav):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                             capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    return {i: (p, seconds(p)) for i, p in jobs.items()}


def voice_worker(ref, lang, jobs_path):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained(str(COMFY / "models" / "TTS" / "Qwen" / "Qwen3-TTS-12Hz-1.7B-Base"),
                                        device_map="cuda:0", dtype=torch.bfloat16)
    prompt = tts.create_voice_clone_prompt(ref_audio=ref, ref_text=None, x_vector_only_mode=True)
    for job in json.loads(Path(jobs_path).read_text(encoding="utf-8")):
        torch.manual_seed(0)
        wavs, sr = tts.generate_voice_clone(text=job["text"], language={"en": "english", "es": "spanish"}[lang],
                                            voice_clone_prompt=prompt)
        sf.write(job["out"], wavs[0], sr)
        print(f"voice {len(wavs[0]) / sr:.1f} s: {job['text']}", flush=True)


# ---- edit ----

def remap(fast_parts, t0):
    """Recording time -> video time: each fast part lasts its target length."""
    parts = sorted(fast_parts)

    def f(t):
        out = t - t0
        for a, b, target in parts:
            if t <= a:
                break
            span = b - a
            done = min(t, b) - a
            out -= done - done * (target / span if span > target else 1)
        return max(0.0, out)
    return f


def ts(t, sep=","):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep) if sep == "," else f"{h}:{m:02d}:{s:05.2f}"


def edit(rec_frames, d: Director, folder: Path, name: str, fps: int, cap_lang: str, voices: dict):
    t0 = rec_frames[0][0]
    f = remap(d.fast_parts, t0)
    end_rec = max(rec_frames[-1][0], d.t_stop)
    tail = 1.5
    lines = []
    for i, (t, fn) in enumerate(rec_frames):
        nxt = rec_frames[i + 1][0] if i + 1 < len(rec_frames) else end_rec
        dur = f(nxt) - f(t) if i + 1 < len(rec_frames) else f(end_rec) - f(t) + tail
        lines.append(f"file '{fn}'\nduration {max(dur, 0.0):.6f}")
    lines.append(f"file '{rec_frames[-1][1]}'")
    (folder / "frames.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = f(end_rec) + tail

    caps = []
    for i, (t, en, es) in enumerate(d.captions):
        t_end = d.captions[i + 1][0] if i + 1 < len(d.captions) else end_rec
        if en:
            caps.append((f(t), f(t_end) if i + 1 < len(d.captions) else total, en, es))
    for lang, k in (("en", 2), ("es", 3)):
        srt = [f"{i + 1}\n{ts(a)} --> {ts(b)}\n{c[k]}\n" for i, c in enumerate(caps) for a, b in [(c[0], c[1])]]
        (folder / f"{name}.{lang}.srt").write_text("\n".join(srt), encoding="utf-8")

    ass = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
           "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           "Style: Cap,Segoe UI Semibold,46,&H00FFFFFF,&H00FFFFFF,&H28101012,&H28101012,0,0,0,0,100,100,0,0,3,14,0,2,160,160,64,1",
           "Style: Fast,Segoe UI Semibold,34,&H0033CCFF,&H0033CCFF,&H28101012,&H28101012,0,0,0,0,100,100,0,0,3,10,0,9,40,40,40,1",
           "Style: Title,Segoe UI Black,64,&H00FFFFFF,&H00FFFFFF,&H50101012,&H50101012,0,0,0,0,100,100,1,0,3,22,0,7,90,90,120,1",
           "Style: Sub,Segoe UI Semibold,36,&H0033CCFF,&H0033CCFF,&H50101012,&H50101012,0,0,0,0,100,100,0,0,3,14,0,7,90,90,222,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for a, b, en, es in caps if cap_lang != "none" else []:  # none: the .srt files only
        ass.append(f"Dialogue: 0,{ts(a, '.')},{ts(b, '.')},Cap,,0,0,0,,{{\\fad(220,180)}}{en if cap_lang == 'en' else es}")
    for t, secs, text, subtitle in d.titles:
        a, b = f(t), f(t) + secs
        ass.append(f"Dialogue: 2,{ts(a, '.')},{ts(b, '.')},Title,,0,0,0,,{{\\fad(400,400)}}{text}")
        ass.append(f"Dialogue: 2,{ts(a + 0.25, '.')},{ts(b, '.')},Sub,,0,0,0,,{{\\fad(400,400)}}{subtitle}")
    for a, b, target in d.fast_parts:
        speed = (b - a) / target
        if speed > 1.2:
            ass.append(f"Dialogue: 1,{ts(f(a), '.')},{ts(f(b), '.')},Fast,,0,0,0,,{{\\fad(120,120)}}» x{speed:.0f}")
    (folder / "subs.ass").write_text("\n".join(ass) + "\n", encoding="utf-8-sig")

    vf = (f"scale={W}:{H}:flags=lanczos,fps={fps},subtitles=subs.ass,"
          f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0.0, total - 0.6):.2f}:d=0.6")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "frames.txt"]
    spoken = [(f(t), voices[i][0]) for t, i in d.spoken if i in voices]
    if spoken:
        parts = []
        for k, (at, wav) in enumerate(spoken):
            cmd += ["-i", str(wav)]
            parts.append(f"[{k + 1}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={int(round(at * 1000))}:all=1[v{k}]")
        mix = "".join(f"[v{k}]" for k in range(len(spoken)))
        mix += f"amix=inputs={len(spoken)}:normalize=0:dropout_transition=0," if len(spoken) > 1 else "anull,"
        graph = ";".join([f"[0:v]{vf}[vid]"] + parts
                         + [f"{mix}apad,atrim=0:{total:.3f},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aud]"])
        cmd += ["-filter_complex", graph, "-map", "[vid]", "-map", "[aud]", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-vf", vf]
    cmd += ["-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{name}.mp4"]
    subprocess.run(cmd, cwd=folder, check=True)
    return folder / f"{name}.mp4", total, caps


async def record(scene, url, folder, unet, voice_lens, lora=None, language="en"):
    unet = unet or ("HaDeS_MOSS_v1.0_Turbo.safetensors" if scene == "layout_sentence" else None)
    from playwright.async_api import async_playwright
    wf_name, play, lines = SCENES[scene]
    # no example workflow: the scene starts from an empty canvas
    workflow = json.loads((ROOT / "examples" / wf_name).read_text(encoding="utf-8")) if wf_name else {"nodes": []}
    for n in workflow["nodes"]:
        if unet and n["type"] == "HaDeSMossLoader":
            n["widgets_values"][0] = unet
        if lora and n["type"] == "HaDeSMossTurbo":
            n["widgets_values"][0] = lora
        local = getattr(play, "local_names", {})
        if local and isinstance(n.get("widgets_values"), list):
            n["widgets_values"] = [local.get(v, v) if isinstance(v, str) else v for v in n["widgets_values"]]
    frames = folder / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, locale="en-US")
        await ctx.add_init_script(CURSOR_JS)
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_function("() => (window.comfyAPI?.app?.app || window.app)?.graph", timeout=60000)
        await page.wait_for_timeout(2500)
        await page.keyboard.press("Escape")
        await page.evaluate(HELPERS_JS)
        if hasattr(play, "prepare"):  # off camera, before recording: e.g. load the models so on-screen times are the runs'
            await play.prepare(page)
        if wf_name:
            await page.evaluate("async (wf) => { await window.__hv.app().loadGraphData(wf); }", workflow)
        else:
            await page.evaluate("() => { const a = window.__hv.app(); a.graph.clear(); a.canvas.ds.scale = 1; a.canvas.ds.offset[0] = 0; a.canvas.ds.offset[1] = 0; a.graph.setDirtyCanvas(true, true); }")
        await page.wait_for_timeout(1500)
        await page.mouse.move(W / 2, H / 2)
        await page.screenshot(path=str(folder / "start.png"))
        rec = Recorder(page, frames)
        await rec.start()
        d = Director(page, lines, voice_lens, language)
        await asyncio.sleep(0.6)
        try:
            await play(d)
        except StopScene:
            d.captions.append((time.time(), "", ""))
            await asyncio.sleep(1.0)
        await rec.stop()
        d.t_stop = rec.t_stop
        await browser.close()
    return rec.frames, d


def serve(url, out, python, extra_args, extra_whitelist=()):
    """A clean ComfyUI for recording: only this pack and ComfyUI-Manager, its own user folder (English UI, no minimap, no
    node badges), the registry list copied from the main install's Manager cache so the Manager finds the Suite."""
    port = int(url.rsplit(":", 1)[1].split("/")[0])
    user = Path(out) / "comfy_user"
    (user / "default").mkdir(parents=True, exist_ok=True)
    settings = user / "default" / "comfy.settings.json"
    cur = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    cur.update({"Comfy.TutorialCompleted": True, "Comfy.Locale": "en", "Comfy.Minimap.Visible": False, "Comfy.Graph.CanvasInfo": False,
                "Comfy.NodeBadge.NodeIdBadgeMode": "None", "Comfy.NodeBadge.NodeSourceBadgeMode": "None",
                "Comfy.NodeSearchBoxImpl.NodePreview": False})
    settings.write_text(json.dumps(cur, indent=4), encoding="utf-8")
    cache = user / "__manager" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for f in (COMFY / "user" / "__manager" / "cache").glob("*_nodes.json"):
        shutil.copy2(f, cache / f.name)
    # Keep Manager on its copied local registry cache so a recording never waits on network refreshes.
    manager_config = cache.parent / "config.ini"
    config_text = manager_config.read_text(encoding="utf-8") if manager_config.exists() else "[default]\n"
    if re.search(r"(?m)^network_mode\s*=.*$", config_text):
        config_text = re.sub(r"(?m)^network_mode\s*=.*$", "network_mode = offline", config_text)
    else:
        config_text = config_text.rstrip() + "\nnetwork_mode = offline\n"
    manager_config.write_text(config_text, encoding="utf-8")
    log = open(Path(out) / f"comfy_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen([python, "main.py", "--port", str(port), "--disable-auto-launch", "--disable-all-custom-nodes",
                             "--whitelist-custom-nodes", ROOT.name, "ComfyUI-Manager", *extra_whitelist, "--user-directory", str(user.resolve()),
                             "--output-directory", str((Path(out) / "comfy_output").resolve()),
                             "--temp-directory", str((Path(out) / "comfy_temp").resolve()),
                             "--input-directory", str((Path(out) / "comfy_input").resolve())]  # never the user's own input folder
                            + extra_args,
                            cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(180):
        try:
            urllib.request.urlopen(url + "/system_stats", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"the recording ComfyUI stopped, see {log.name}")
            time.sleep(1)
    proc.terminate()
    raise SystemExit("the recording ComfyUI did not answer in 3 minutes")


def main():
    global W, H, UNTIL
    if sys.argv[1:2] == ["_voice"]:
        return voice_worker(*sys.argv[2:5])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--resolution", choices=("1080p", "1440p"), default="1080p", help="1440p: a 2560 x 1440 page; the canvas camera, the cursor, overlays and captions scale with it")
    ap.add_argument("--unet", help="model file for the MOSS Loader, when the example's name is not the one installed")
    ap.add_argument("--lora", help="LoRA path relative to the ComfyUI LoRA directory, as listed by MOSS Turbo")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--until", type=int, help="test take: stop the scene before line N + 1")
    ap.add_argument("--voice", choices=["en", "es"], help="speak the captions in this language with the cloned voice")
    ap.add_argument("--ref", help="reference recording of the voice to clone (wav, 10-30 s, one speaker)")
    ap.add_argument("--captions", choices=["en", "es", "none"],
                    help="language of the burned captions (default: the voice's, else en); none: no burned captions, the .srt files only")
    ap.add_argument("--tts-python", default=str(COMFY / "venv" / "Scripts" / "python.exe"), help="a Python with qwen-tts")
    ap.add_argument("--serve", action="store_true", help="start a clean ComfyUI at --url for the recording and stop it after")
    ap.add_argument("--comfy-args", default="--use-sage-attention",
                    help="extra arguments for the clean ComfyUI, as your own install is started (drafts and images can differ otherwise)")
    a = ap.parse_args()
    if a.resolution == "1440p":
        W, H = 2560, 1440
    UNTIL = a.until
    language = a.captions or a.voice or "en"
    if a.voice and not a.ref:
        ap.error("--voice needs --ref")
    folder = Path(a.out) / a.scene
    folder.mkdir(parents=True, exist_ok=True)
    voices = synth(SCENES[a.scene][2], a.voice, a.ref, Path(a.out) / "voice_cache", a.tts_python) if a.voice else {}
    extra_wl = ("comfyui_controlnet_aux",) if a.scene == "controlnet" else ()
    server = serve(a.url, a.out, a.tts_python, a.comfy_args.split(), extra_wl) if a.serve else None
    try:
        frames, d = asyncio.run(record(a.scene, a.url, folder, a.unet, {i: v[1] for i, v in voices.items()}, a.lora, language))
    finally:
        if server:
            server.terminate()
            server.wait(30)
    video, total, caps = edit(frames, d, folder / "frames", a.scene, a.fps, language, voices)
    final = folder / video.name
    shutil.move(str(video), final)
    for lang in ("en", "es"):
        shutil.move(str(folder / "frames" / f"{a.scene}.{lang}.srt"), folder / f"{a.scene}.{lang}.srt")
    if not a.keep_frames:
        shutil.rmtree(folder / "frames", ignore_errors=True)
    print(f"{final}  {total:.1f} s, {len(frames)} frames recorded, {len(caps)} captions, fast parts: "
          + ", ".join(f"{b - a_:.1f} s -> {t:.1f} s" for a_, b, t in d.fast_parts))


if __name__ == "__main__":
    main()
