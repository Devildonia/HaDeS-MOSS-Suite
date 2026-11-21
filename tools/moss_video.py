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
    el.innerHTML = html; requestAnimationFrame(() => { el.style.opacity = '1'; });
  },
  hide(ms = 600) { const el = document.getElementById('__hv_show'); if (el) { el.style.transition = `opacity ${ms}ms ease`; el.style.opacity = '0'; setTimeout(() => el.remove(), ms + 50); } },
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


SCENES = {"basic_generation": (None, basic_generation, BASIC_LINES),
          "layout_test": ("HaDeS MOSS Suite - Layout from a Sentence.json", layout_test, LAYOUT_TEST_LINES),
          "layout_sentence": ("HaDeS MOSS Suite - Layout from a Sentence.json", layout_sentence, LAYOUT_SENTENCE_LINES),
          "turbo_8_steps_upscale_x2": ("HaDeS MOSS Suite - Turbo 8 steps + Upscale x2.json", turbo_upscale_x2,
                                       TURBO_UPSCALE_LINES),
          "upscale_telemetry": (None, upscale_telemetry, UP_LINES),
          "prompt_enhancer": (None, prompt_enhancer, PE_LINES),
          "lora_toolkit": (None, lora_toolkit, LT_LINES)}

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


def serve(url, out, python, extra_args):
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
                             "--whitelist-custom-nodes", ROOT.name, "ComfyUI-Manager", "--user-directory", str(user.resolve()),
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
    server = serve(a.url, a.out, a.tts_python, a.comfy_args.split()) if a.serve else None
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
