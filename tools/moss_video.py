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
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
W, H = 1920, 1080

# a drawn cursor (headless Chromium shows none) with a ring on each press
CURSOR_JS = r"""
(() => {
  const add = () => {
    if (document.getElementById('__hv_cursor')) return;
    const c = document.createElement('div');
    c.id = '__hv_cursor';
    c.innerHTML = '<svg width="28" height="28" viewBox="0 0 28 28"><path d="M3 2 L3 22 L8.5 17 L12.5 26 L16 24.5 L12 15.8 L19.5 15.8 Z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, {position: 'fixed', left: '0', top: '0', zIndex: 2147483647, pointerEvents: 'none',
      transform: 'translate(960px, 540px)', filter: 'drop-shadow(0 2px 3px rgba(0,0,0,.45))'});
    document.body.appendChild(c);
    const ring = (x, y) => {
      const r = document.createElement('div');
      Object.assign(r.style, {position: 'fixed', left: (x - 22) + 'px', top: (y - 22) + 'px', width: '44px', height: '44px',
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
  target(ids, pad, top, bottom = 150) {
    const a = this.app(), r = a.canvas.canvas.getBoundingClientRect();
    let [x1, y1, x2, y2] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const id of ids) { const b = this.rect(id); x1 = Math.min(x1, b[0]); y1 = Math.min(y1, b[1]); x2 = Math.max(x2, b[2]); y2 = Math.max(y2, b[3]); }
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
      Object.assign(el.style, {position: 'fixed', inset: '0', zIndex: 2147483600, background: 'rgba(10,10,12,0.96)', display: 'flex',
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


class Director:
    """The scene's tools: captions, fast parts, a cursor that moves like a hand, clicks, typing, the camera."""

    def __init__(self, page, lines, voice_lens):
        self.page, self.captions, self.fast_parts, self.pos = page, [], [], (W / 2, H / 2)
        self.lines, self.voice_lens = lines, voice_lens
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
        en, es = self.lines[i]
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
        t0 = time.time()
        yield
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


SCENES = {"layout_test": ("HaDeS MOSS Suite - Layout from a Sentence.json", layout_test, LAYOUT_TEST_LINES),
          "layout_sentence": ("HaDeS MOSS Suite - Layout from a Sentence.json", layout_sentence, LAYOUT_SENTENCE_LINES)}

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
    texts = {i: l[k] for i, l in enumerate(lines)}
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

    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
           "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           "Style: Cap,Segoe UI Semibold,46,&H00FFFFFF,&H00FFFFFF,&H28101012,&H28101012,0,0,0,0,100,100,0,0,3,14,0,2,160,160,64,1",
           "Style: Fast,Segoe UI Semibold,34,&H0033CCFF,&H0033CCFF,&H28101012,&H28101012,0,0,0,0,100,100,0,0,3,10,0,9,40,40,40,1",
           "Style: Title,Segoe UI Black,64,&H00FFFFFF,&H00FFFFFF,&H50101012,&H50101012,0,0,0,0,100,100,1,0,3,22,0,7,90,90,120,1",
           "Style: Sub,Segoe UI Semibold,36,&H0033CCFF,&H0033CCFF,&H50101012,&H50101012,0,0,0,0,100,100,0,0,3,14,0,7,90,90,222,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for a, b, en, es in caps:
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


async def record(scene, url, folder, unet, voice_lens):
    unet = unet or ("HaDeS_MOSS_v1.0_Turbo.safetensors" if scene == "layout_sentence" else None)
    from playwright.async_api import async_playwright
    wf_name, play, lines = SCENES[scene]
    workflow = json.loads((ROOT / "examples" / wf_name).read_text(encoding="utf-8"))
    for n in workflow["nodes"]:
        if unet and n["type"] == "HaDeSMossLoader":
            n["widgets_values"][0] = unet
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
        await page.evaluate("async (wf) => { await window.__hv.app().loadGraphData(wf); }", workflow)
        await page.wait_for_timeout(1500)
        await page.mouse.move(W / 2, H / 2)
        await page.screenshot(path=str(folder / "start.png"))
        rec = Recorder(page, frames)
        await rec.start()
        d = Director(page, lines, voice_lens)
        await asyncio.sleep(0.6)
        await play(d)
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
                "Comfy.NodeBadge.NodeIdBadgeMode": "None", "Comfy.NodeBadge.NodeSourceBadgeMode": "None"})
    settings.write_text(json.dumps(cur, indent=4), encoding="utf-8")
    cache = user / "__manager" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for f in (COMFY / "user" / "__manager" / "cache").glob("*_nodes.json"):
        shutil.copy2(f, cache / f.name)
    log = open(Path(out) / f"comfy_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen([python, "main.py", "--port", str(port), "--disable-auto-launch", "--disable-all-custom-nodes",
                             "--whitelist-custom-nodes", ROOT.name, "ComfyUI-Manager", "--user-directory", str(user.resolve())]
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
    if sys.argv[1:2] == ["_voice"]:
        return voice_worker(*sys.argv[2:5])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--unet", help="model file for the MOSS Loader, when the example's name is not the one installed")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--voice", choices=["en", "es"], help="speak the captions in this language with the cloned voice")
    ap.add_argument("--ref", help="reference recording of the voice to clone (wav, 10-30 s, one speaker)")
    ap.add_argument("--captions", choices=["en", "es"], help="language of the burned captions (default: the voice's, else en)")
    ap.add_argument("--tts-python", default=str(COMFY / "venv" / "Scripts" / "python.exe"), help="a Python with qwen-tts")
    ap.add_argument("--serve", action="store_true", help="start a clean ComfyUI at --url for the recording and stop it after")
    ap.add_argument("--comfy-args", default="--use-sage-attention",
                    help="extra arguments for the clean ComfyUI, as your own install is started (drafts and images can differ otherwise)")
    a = ap.parse_args()
    if a.voice and not a.ref:
        ap.error("--voice needs --ref")
    folder = Path(a.out) / a.scene
    folder.mkdir(parents=True, exist_ok=True)
    voices = synth(SCENES[a.scene][2], a.voice, a.ref, Path(a.out) / "voice_cache", a.tts_python) if a.voice else {}
    server = serve(a.url, a.out, a.tts_python, a.comfy_args.split()) if a.serve else None
    try:
        frames, d = asyncio.run(record(a.scene, a.url, folder, a.unet, {i: v[1] for i, v in voices.items()}))
    finally:
        if server:
            server.terminate()
            server.wait(30)
    video, total, caps = edit(frames, d, folder / "frames", a.scene, a.fps, a.captions or a.voice or "en", voices)
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
