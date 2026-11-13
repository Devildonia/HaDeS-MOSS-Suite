"""Removal pairs for an own MOSS-Image edit LoRA, by reversed inpainting (dev tool, not a node).

Removing an object is what the published edit LoRAs fail at (tools/RESULTS.md, edit benchmark 2026-11-12); adding one is easy. So
the pairs are made the other way round, through a running ComfyUI (stock nodes only):
  1. B: a clean scene from MOSS-Image SFT, without the object;
  2. the object is painted into a box of B (SetLatentNoiseMask on B's latent, denoise 1, the scene prompt plus the object);
  3. A = B with that box pasted in through a feathered mask, so A equals B pixel for pixel outside the box.
A training pair is reference A + "Remove the <object>." -> target B: the target is a real image with the untouched background.

The boxes are generous on purpose (steam, smoke, shadows and whole sign boards belong to the object), and the objects avoid the
benchmark's scene / object combinations (no mug, car on a street, lamp post, floor lamp, dock, classroom desks or door sign).
The pixel difference inside the box does not tell whether the object appeared (the whole box is repainted either way; in the
smoke test a failed wall clock scored 45.8, a good chalkboard sign 23.9). So the MOSS text encoder (Qwen3.5-9B) is asked, with the
stock TextGenerate node and no sampling, whether the object is in a crop around the box of A, then of B: `vlm` = [A, B] answers,
and a usable pair reads ["yes", "no"]. In the smoke test this matched the eye on 5 of 6 pairs; the sixth was a scooter painted
behind a shop window, which it rightly saw. The sheet mode puts the pairs it flags first.

  python tools/moss_edit_pairs.py run --scenes 150 --out I:/HaDeS_MOSS/moss_edit_pairs/pilot
  python tools/moss_edit_pairs.py sheet --out I:/HaDeS_MOSS/moss_edit_pairs/pilot

Mode `add` makes A with a published edit LoRA instead of inpainting, on the same B images and objects as a `run` folder: the LoRA
gets B and "Add <object>." (SFT, Euler / normal, 30 steps, CFG 4, strength 1, the edit benchmark's settings), and only the area
where its output differs from B is pasted onto B (per-pixel difference blurred, thresholded, small specks dropped, dilated and
feathered), so A still equals B outside that area. The LoRA only draws training inputs; nothing of its weights is trained on.
  python tools/moss_edit_pairs.py add --from I:/HaDeS_MOSS/moss_edit_pairs/pilot --lora MOSS/armV5-moss-edit-lora.safetensors --scenes 50 --out I:/HaDeS_MOSS/moss_edit_pairs/add_armV5
`--lora qwen2509` uses Qwen-Image-Edit 2509 instead (the adder of the dataset, chosen on the pilot). `add` only draws; mode `check`
then asks the text encoder whether the object is in A and not in B (`vlm`) and how many there are in A (`count`). A usable pair
reads vlm ["yes", "no"] and count 1.
  python tools/moss_edit_pairs.py run --b-only --scenes 1000 --seed 2 --out I:/HaDeS_MOSS/moss_edit_pairs/full
  python tools/moss_edit_pairs.py add --from I:/HaDeS_MOSS/moss_edit_pairs/full --lora qwen2509 --scenes 1000 --out I:/HaDeS_MOSS/moss_edit_pairs/full_qwen2509
  python tools/moss_edit_pairs.py check --out I:/HaDeS_MOSS/moss_edit_pairs/full_qwen2509
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

URL = "http://127.0.0.1:8188"
SIZE = 1024
NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
FEATHER = 12  # px of the paste blur, inside the painted box
PAD = 24  # px the painted box extends beyond the object box, so the feather never reaches unpainted pixels

# regions: (x0, y0, x1, y1) where the object's box centre may fall; sizes: (w_min, w_max, h_min, h_max), fractions of the image
TABLE, FLOOR, WALL, SKY, GROUND, WATER, SHELF = (.25, .62, .75, .85), (.2, .7, .8, .88), (.2, .2, .8, .55), (.15, .1, .85, .3), (.15, .62, .85, .88), (.15, .55, .85, .8), (.25, .3, .75, .6)
SMALL, MID, TALL, WIDE, BIG = (.12, .2, .12, .22), (.18, .3, .18, .32), (.12, .22, .3, .5), (.3, .45, .15, .25), (.28, .42, .3, .45)

OBJECTS = {
    "teapot": ("a steaming white teapot with steam rising from its spout", TABLE, MID),
    "candle": ("a lit candle in a glass jar with a thin trail of smoke", TABLE, SMALL),
    "incense": ("a burning incense stick on a small holder with curling smoke", TABLE, TALL),
    "wine glass": ("a glass of red wine", TABLE, SMALL),
    "vase": ("a blue ceramic vase with yellow tulips", TABLE, TALL),
    "smartphone": ("a black smartphone lying flat", TABLE, SMALL),
    "apple": ("a green apple", TABLE, SMALL),
    "laptop": ("an open silver laptop", TABLE, MID),
    "stack of books": ("a stack of three old books", TABLE, MID),
    "houseplant": ("a potted monstera houseplant", FLOOR, TALL),
    "backpack": ("a yellow backpack", FLOOR, MID),
    "dog": ("a sleeping golden retriever", FLOOR, WIDE),
    "suitcase": ("a red hard-shell suitcase", FLOOR, TALL),
    "guitar": ("an acoustic guitar leaning against the wall", FLOOR, TALL),
    "wall clock": ("a round wall clock", WALL, SMALL),
    "framed poster": ("a framed poster of a mountain", WALL, MID),
    "neon sign": ("a pink neon sign on the wall that reads \"OPEN LATE\"", WALL, WIDE),
    "poster": ("a paper poster on the wall that reads \"JAZZ NIGHT\"", WALL, MID),
    "sticky note": ("a yellow sticky note that reads \"CALL MOM\"", WALL, SMALL),
    "chalkboard sign": ("a chalkboard sandwich-board sign that reads \"FRESH COFFEE\"", GROUND, TALL),
    "trash bin": ("a green metal trash bin", GROUND, TALL),
    "bicycle": ("a parked blue bicycle", GROUND, WIDE),
    "scooter": ("a parked white scooter", GROUND, MID),
    "bench": ("a wooden park bench", GROUND, WIDE),
    "traffic cone": ("an orange traffic cone", GROUND, SMALL),
    "person": ("a man in a grey coat walking", GROUND, TALL),
    "road sign": ("a road sign on a pole that reads \"SLOW\"", GROUND, TALL),
    "hot air balloon": ("a colorful hot air balloon", SKY, MID),
    "airplane": ("a small white airplane", SKY, SMALL),
    "kite": ("a red kite", SKY, SMALL),
    "boat": ("a small white sailboat", WATER, MID),
    "swan": ("a white swan", WATER, SMALL),
    "buoy": ("a red and white buoy", WATER, SMALL),
    "potted cactus": ("a small potted cactus", SHELF, SMALL),
    "teddy bear": ("a brown teddy bear", SHELF, SMALL),
    "price tag": ("a white price tag that reads \"SALE 50%\"", SHELF, SMALL),
    # added for the full dataset (2026-11-12): more objects and scenes, so 2000 pairs do not repeat the pilot's few combinations
    "cat": ("a grey tabby cat sitting", FLOOR, MID),
    "umbrella": ("a closed black umbrella leaning against the wall", FLOOR, TALL),
    "basketball": ("an orange basketball", FLOOR, SMALL),
    "sneakers": ("a pair of white sneakers", FLOOR, SMALL),
    "pumpkin": ("an orange pumpkin", FLOOR, SMALL),
    "bowl of fruit": ("a ceramic bowl full of oranges", TABLE, MID),
    "camera": ("a vintage film camera", TABLE, SMALL),
    "headphones": ("a pair of black over-ear headphones", TABLE, SMALL),
    "toolbox": ("a red metal toolbox", TABLE, MID),
    "radio": ("a vintage portable radio", TABLE, SMALL),
    "mirror": ("a round mirror with a gold frame", WALL, MID),
    "flag": ("a small red flag on a pole", GROUND, TALL),
    "shopping cart": ("a metal shopping cart", GROUND, MID),
    "motorcycle": ("a parked black motorcycle", GROUND, WIDE),
    "fire hydrant": ("a red fire hydrant", GROUND, SMALL),
    "pigeon": ("a grey pigeon", GROUND, SMALL),
    "woman": ("a woman in a red dress walking", GROUND, TALL),
    "horse": ("a brown horse standing", GROUND, BIG),
    "tent": ("a small orange camping tent", GROUND, WIDE),
    "deer": ("a young deer", GROUND, MID),
    "seagull": ("a flying seagull", SKY, SMALL),
    "drone": ("a small white quadcopter drone", SKY, SMALL),
    "kayak": ("a yellow kayak", WATER, WIDE),
    "duck": ("a mallard duck", WATER, SMALL),
    "globe": ("a small desk globe", SHELF, SMALL),
    "jar of honey": ("a glass jar of honey", SHELF, SMALL),
}
# for the count question; the rest add -s / -es
PLURAL = {"person": "people", "woman": "women", "stack of books": "stacks of books", "incense": "incense sticks", "sneakers": "pairs of sneakers",
          "headphones": "pairs of headphones", "bowl of fruit": "bowls of fruit", "jar of honey": "jars of honey", "deer": "deer",
          "potted cactus": "potted cacti"}

SCENES = {
    "kitchen": ("Photograph of a bright kitchen with a white marble counter, wooden cabinets and a window, morning light.", ["teapot", "candle", "wine glass", "vase", "apple", "smartphone", "sticky note", "wall clock"]),
    "study": ("Photograph of a cozy study with a large oak desk, a leather chair and bookshelves, warm lamp light.", ["laptop", "stack of books", "candle", "incense", "smartphone", "houseplant", "framed poster", "wall clock", "guitar"]),
    "bedroom": ("Photograph of a minimalist bedroom with a white bed, a bedside table and a light grey wall, soft daylight.", ["vase", "candle", "smartphone", "backpack", "suitcase", "dog", "houseplant", "framed poster", "wall clock"]),
    "cafe": ("Photograph of the inside of a small café with round marble tables, wooden chairs and a brick wall, afternoon light.", ["teapot", "wine glass", "laptop", "candle", "neon sign", "poster", "houseplant", "backpack"]),
    "loft": ("Photograph of an industrial loft with concrete floor, large steel windows and a dark leather sofa.", ["guitar", "dog", "houseplant", "suitcase", "neon sign", "framed poster", "backpack"]),
    "bar": ("Photograph of a dim cocktail bar with a long wooden counter and bottles on shelves behind it, moody lighting.", ["wine glass", "candle", "smartphone", "neon sign", "poster", "potted cactus"]),
    "boutique": ("Photograph of a small boutique with wooden shelves of folded sweaters and a white wall, bright lighting.", ["teddy bear", "potted cactus", "price tag", "vase", "houseplant", "backpack", "poster"]),
    "alley": ("Photograph of a narrow alley with brick walls and wet pavement in the evening, warm window light.", ["trash bin", "bicycle", "scooter", "traffic cone", "chalkboard sign", "person", "poster", "neon sign"]),
    "plaza": ("Photograph of a sunny stone plaza in an old town with a fountain in the background.", ["bench", "bicycle", "scooter", "trash bin", "chalkboard sign", "person", "traffic cone"]),
    "park": ("Photograph of a green city park with a gravel path and tall trees, summer afternoon.", ["bench", "bicycle", "trash bin", "dog", "person", "kite", "road sign"]),
    "country_road": ("Photograph of an empty country road through golden wheat fields under a wide sky.", ["road sign", "traffic cone", "bicycle", "person", "hot air balloon", "airplane", "kite"]),
    "beach": ("Photograph of a quiet sandy beach with gentle waves and a clear sky, late afternoon.", ["boat", "buoy", "kite", "person", "dog", "airplane", "hot air balloon"]),
    "harbor": ("Photograph of a calm harbor with stone quay walls and still water, overcast morning.", ["boat", "buoy", "swan", "bicycle", "person", "trash bin", "airplane"]),
    "pond": ("Photograph of a pond in a botanical garden surrounded by reeds and willow trees.", ["swan", "boat", "buoy", "bench", "person", "hot air balloon"]),
    "desert": ("Photograph of a wide desert landscape with red rock formations under a blue sky.", ["hot air balloon", "airplane", "road sign", "person", "traffic cone"]),
    "snow_field": ("Photograph of a snowy meadow with pine trees and mountains in the distance, bright winter day.", ["person", "dog", "bench", "hot air balloon", "airplane", "kite"]),
    "product_table": ("Studio product photograph of a light wooden table against a plain beige wall, soft shadows.", ["vase", "candle", "potted cactus", "apple", "smartphone", "wine glass", "teapot", "price tag"]),
    "anime_room": ("Anime illustration of a cozy bedroom with a desk by the window and posters, soft evening light, clean line art.", ["laptop", "teddy bear", "houseplant", "guitar", "backpack", "poster", "wall clock", "candle"]),
    "anime_street": ("Anime illustration of a quiet residential street with power lines and small houses, summer day, clean line art.", ["bicycle", "trash bin", "traffic cone", "road sign", "person", "airplane", "kite"]),
    "anime_beach": ("Anime illustration of a bright beach with turquoise water and white clouds, clean line art.", ["boat", "buoy", "kite", "person", "dog", "airplane"]),
    "watercolor_garden": ("Watercolor painting of a cottage garden with flower beds and a stone path.", ["bench", "dog", "bicycle", "person", "kite", "houseplant"]),
    "oil_still_life": ("Oil painting of a still life on a draped table in front of a dark background, classical style.", ["vase", "apple", "candle", "wine glass", "teapot", "stack of books"]),
    "living_room": ("Photograph of a living room with a grey sofa, a low wooden coffee table and a patterned rug, afternoon light.", ["cat", "dog", "houseplant", "vase", "laptop", "headphones", "bowl of fruit", "framed poster", "mirror", "basketball"]),
    "office": ("Photograph of an open-plan office with white desks, monitors and large windows, daylight.", ["laptop", "smartphone", "houseplant", "backpack", "headphones", "sticky note", "wall clock", "stack of books"]),
    "library": ("Photograph of a quiet library reading room with long wooden tables and tall bookshelves.", ["stack of books", "globe", "laptop", "backpack", "houseplant", "camera", "person"]),
    "hallway": ("Photograph of an entrance hallway with a wooden floor, a coat rack and a white door, soft light.", ["umbrella", "sneakers", "backpack", "suitcase", "mirror", "cat", "dog", "houseplant"]),
    "workshop": ("Photograph of a woodworking workshop with a heavy workbench and tools hanging on a pegboard.", ["toolbox", "radio", "cat", "wall clock", "backpack", "traffic cone", "camera"]),
    "patio": ("Photograph of a sunny garden patio with a wooden table, folding chairs and potted flowers.", ["bowl of fruit", "wine glass", "candle", "cat", "umbrella", "pumpkin", "basketball", "radio"]),
    "market": ("Photograph of an outdoor street market with fruit stalls and striped awnings, morning.", ["shopping cart", "bicycle", "person", "woman", "pigeon", "chalkboard sign", "trash bin"]),
    "train_station": ("Photograph of an old train station platform with iron columns and a glass roof.", ["suitcase", "backpack", "person", "woman", "bench", "trash bin", "pigeon"]),
    "mountain_trail": ("Photograph of a mountain hiking trail through alpine meadows with peaks in the background.", ["tent", "person", "dog", "horse", "drone", "flag", "backpack"]),
    "lake": ("Photograph of a calm mountain lake with a pebble shore and pine forest, clear morning.", ["kayak", "boat", "duck", "swan", "tent", "drone", "seagull"]),
    "farm": ("Photograph of a farmyard with a red barn, a wooden fence and green hills.", ["horse", "dog", "cat", "bicycle", "pumpkin", "flag", "traffic cone"]),
    "rooftop": ("Photograph of a rooftop terrace with a city skyline at sunset.", ["drone", "airplane", "seagull", "houseplant", "bench", "person", "wine glass", "flag"]),
    "forest": ("Photograph of a sunlit forest clearing with ferns and tall oak trees.", ["tent", "deer", "person", "dog", "backpack", "drone"]),
    "night_street": ("Photograph of a pedestrian city street at night with shop windows and wet pavement.", ["fire hydrant", "motorcycle", "person", "woman", "neon sign", "trash bin", "scooter"]),
    "museum": ("Photograph of a quiet art gallery with white walls, framed paintings and a wooden floor.", ["bench", "person", "woman", "houseplant", "framed poster", "globe"]),
    "render_room": ("3D render of a pastel colored playroom with soft toys and a round window, soft lighting.", ["teddy bear", "basketball", "houseplant", "globe", "headphones", "cat", "pumpkin"]),
    "anime_cafe": ("Anime illustration of a small café interior with wooden tables and plants, warm light, clean line art.", ["teapot", "cat", "laptop", "candle", "poster", "neon sign", "houseplant"]),
    "sketch_street": ("Pencil sketch of an old European street with cobblestones and balconies.", ["bicycle", "person", "bench", "trash bin", "fire hydrant", "pigeon"]),
    "watercolor_lake": ("Watercolor painting of a lake with wooden boats and hills in the distance.", ["boat", "swan", "duck", "kayak", "seagull", "hot air balloon"]),
}


def req(path, data=None, raw=False):
    for attempt in range(6):  # a busy ComfyUI sometimes misses a request (WinError 10060 after ~1000 calls); retry instead of stopping a night run
        try:
            b = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(data).encode() if data is not None else None, {"Content-Type": "application/json"}), timeout=120).read()
            return b if raw else json.loads(b)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 5:
                raise
            print("retry", path, e, flush=True)
            time.sleep(10)


def upload(name, png_bytes):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ntemp\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += png_bytes + f"\r\n--{b}--\r\n".encode()
    up = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"})).read())
    return (f"{up['subfolder']}/" if up["subfolder"] else "") + f"{up['name']} [temp]"


def pick_box(rng, region, size):
    w, h = round(rng.uniform(*size[:2]) * SIZE), round(rng.uniform(*size[2:]) * SIZE)
    cx, cy = rng.uniform(region[0], region[2]) * SIZE, rng.uniform(region[1], region[3]) * SIZE
    x0 = int(min(max(cx - w / 2, PAD), SIZE - PAD - w))
    y0 = int(min(max(cy - h / 2, PAD), SIZE - PAD - h))
    return x0, y0, x0 + w, y0 + h


def masks(box):
    """Painted area (box + PAD, hard) and paste mask (box + PAD - FEATHER, blurred: it stays inside the painted area)."""
    paint, paste = Image.new("L", (SIZE, SIZE), 0), Image.new("L", (SIZE, SIZE), 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(paint).rectangle((x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD), fill=255)
    m = PAD - FEATHER
    ImageDraw.Draw(paste).rectangle((x0 - m, y0 - m, x1 + m, y1 + m), fill=255)
    # the blur's tail is cut at the painted area, so A equals B exactly outside it
    paste = np.asarray(paste.filter(ImageFilter.GaussianBlur(FEATHER / 2))) * (np.asarray(paint) > 0)
    return paint, Image.fromarray(paste.astype(np.uint8))


def png(img):
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def graph(scene_prompt, seed_b, items):
    """B from the scene prompt; per item, the object painted into its box of B's latent."""
    g = {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "mossImage_sft_bf16.safetensors", "weight_dtype": "default"}},
         "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
         "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
         "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["clip", 0]}},
         "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": scene_prompt, "clip": ["clip", 0]}},
         "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
         "ks": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
                "seed": seed_b, "steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
         "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
         "out_b": {"class_type": "PreviewImage", "inputs": {"images": ["dec", 0]}}}
    for i, it in enumerate(items):
        g[f"m{i}"] = {"class_type": "LoadImageMask", "inputs": {"image": it["paint_mask"], "channel": "red"}}
        g[f"nm{i}"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["ks", 0], "mask": [f"m{i}", 0]}}
        g[f"p{i}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": it["paint_prompt"], "clip": ["clip", 0]}}
        g[f"ks{i}"] = {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": [f"p{i}", 0], "negative": ["neg", 0], "latent_image": [f"nm{i}", 0],
                       "seed": it["seed_a"], "steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}}
        g[f"d{i}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"ks{i}", 0], "vae": ["vae", 0]}}
        g[f"out{i}"] = {"class_type": "PreviewImage", "inputs": {"images": [f"d{i}", 0]}}
    return g


def ask(img, obj, tag, question=None):
    """Yes / no from the MOSS text encoder: is the object in this crop? (or another short question)"""
    question = question or f"Is there a {obj} in this picture? Answer only yes or no."
    g = {"clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
         "img": {"class_type": "LoadImage", "inputs": {"image": upload(f"hades_vlm_{tag}.png", png(img))}},
         "gen": {"class_type": "TextGenerate", "inputs": {"clip": ["clip", 0], "prompt": question,
                 "max_length": 8, "sampling_mode": "off", "image": ["img", 0], "thinking": False}},
         "show": {"class_type": "PreviewAny", "inputs": {"source": ["gen", 0]}}}
    pid = req("/prompt", {"prompt": g, "client_id": "hades_edit_pairs"})["prompt_id"]
    while pid not in (h := req(f"/history/{pid}")):
        time.sleep(0.3)
    assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
    return h[pid]["outputs"]["show"]["text"][0].strip().strip(".").lower()


def generate(text, max_length=160):
    """Text from the MOSS text encoder, no image, no sampling."""
    g = {"clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
         "gen": {"class_type": "TextGenerate", "inputs": {"clip": ["clip", 0], "prompt": text, "max_length": max_length, "sampling_mode": "off", "thinking": False}},
         "show": {"class_type": "PreviewAny", "inputs": {"source": ["gen", 0]}}}
    pid = req("/prompt", {"prompt": g, "client_id": "hades_edit_pairs"})["prompt_id"]
    while pid not in (h := req(f"/history/{pid}")):
        time.sleep(0.3)
    assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
    return h[pid]["outputs"]["show"]["text"][0].strip()


# v0.2 data: scenes from a prompt list (the turbo v1.0 prompts), objects proposed per scene by the text encoder. Three kinds are
# offered per scene (it picks those that fit); the removal kinds of ConceptEdit-12M's taxonomy are among them. The benchmark's
# objects stay out, so the benchmark keeps measuring objects the LoRA never saw.
KINDS = ["animal", "bird", "vehicle", "bicycle or scooter", "piece of furniture", "food or drink", "household item", "electronic device",
         "tool", "toy", "plant in a pot", "bag or luggage", "sports equipment", "musical instrument", "box or container", "sign or poster with text",
         "street object such as a bin, cone or barrier", "garden object", "kitchen item", "decoration", "standing person", "light source",
         "book or stationery", "boat", "aircraft or balloon", "trash or clutter on the ground", "fence or barrier", "statue or sculpture"]
BENCH = ("mug", "cup", "car", "lamp post", "street lamp", "floor lamp", "dock", "pier", "desk", "chair", "hanging sign")
POSITIONS = ("on the left side of the picture", "on the right side of the picture", "in the middle of the picture",
             "in the lower left part of the picture", "in the lower right part of the picture", "in the upper part of the picture")
PROPOSE_Q = ("Image description: \"{}\"\n\nSuggest 6 objects to add to this image, from these kinds: {}. Each must be one concrete physical "
             "object that is not already in the image and stands on its own: not worn or held by anyone, not part of a building, not a "
             "liquid, a light effect or scattered pieces. Prefer medium-sized objects that are easy to see. Avoid the words vintage, wooden "
             "and rustic. Write exactly 6 lines, no numbering, each as: object name | the object in a few words\n"
             "Example:\ntraffic cone | an orange traffic cone\nbicycle | a red bicycle with a basket")
VERBS = ("add ", "place ", "put ", "set ", "lay ", "hang ", "float ", "rest ", "stand ", "position ", "drape ", "include ")
CUT = re.compile(r"\s(?:on|near|beside|in|inside|against|next to|across|at|by|under|above|behind|over|sits|stands|is|are|placed|leaning|resting|parked|perched|hanging|lying|floating|sitting|standing|tipped|curled|filled|with|that|which|walking|flying|surrounding|blocking|along|holding)(?:\s.*)?$")
GENERIC = {"street object", "garden object", "light source", "decoration", "ornament", "toy", "household item", "container", "clutter pile", "object", "item"}


def parse_proposals(text, prompt):
    """(name, phrase) per answer line, whatever separator the model used; drops kind names echoed back and objects of the prompt."""
    kinds = {k for k in KINDS} | {w for k in KINDS for w in k.split(" or ")}
    out = []
    for line in text.splitlines():
        line = line.strip().strip("-*").strip()
        if not line or line.lower().startswith("example"):
            continue
        sep = "|" if "|" in line else ("," if "," in line else (":" if ":" in line else None))
        if not sep:
            continue
        name, phrase = (x.strip() for x in line.split(sep, 1))
        name = re.sub(r"^[0-9.)\s]+", "", name).strip("<>\"' ").lower()
        phrase = phrase.strip("<>\"' ").rstrip(".")
        if "|" in phrase:
            phrase = phrase.split("|", 1)[1].strip()
        low = phrase.lower()
        for v in VERBS:
            if low.startswith(v):
                phrase, low = phrase[len(v):], low[len(v):]
        phrase = CUT.sub("", phrase).strip(" ,")
        if not re.match(r"(?i)^(a|an|the|some|two|one)\s", phrase):
            phrase = ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase
        if name in kinds or name in GENERIC:  # the kind echoed back: name the object from its phrase instead
            words = re.sub(r"(?i)^(a|an|the|some|one)\s", "", phrase).split()
            name = " ".join(words[-2:] if len(words) >= 3 else words).lower()
        name = re.sub(r"^(a|an|the)\s", "", name)
        if not name or name in kinds or name in GENERIC or " or " in name or len(name.split()) > 3 or len(phrase.split()) > 12:
            continue
        if any(b in name for b in BENCH) or name in prompt.lower():
            continue
        out.append((name, phrase[0].lower() + phrase[1:]))
    return out


def propose(args):
    """Plan: scenes from --prompts, --objects-per-scene objects each, with a position each so their changed areas rarely touch."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else []
    done = {sc["scene_id"] for sc in plan}
    prompts = json.load(open(args.prompts, encoding="utf-8"))
    rng = random.Random(args.seed)
    order = rng.sample(range(len(prompts)), len(prompts))
    for s_i in range(args.scenes):
        scene_id = f"s{s_i:05d}"
        prompt = prompts[order[s_i % len(order)]]
        kinds, positions = rng.sample(KINDS, 3), rng.sample(POSITIONS, args.objects_per_scene)
        seed_b, seeds_a = args.seed * 100000 + s_i, [rng.randrange(2 ** 31) for _ in range(args.objects_per_scene)]
        if scene_id in done:
            continue
        text = generate(PROPOSE_Q.format(prompt, ", ".join(kinds)))
        objs, seen = [], set()
        for name, phrase in parse_proposals(text, prompt):
            if name in seen:
                continue
            seen.add(name)
            objs.append({"object": name, "phrase": phrase})
            if len(objs) == args.objects_per_scene:
                break
        for o, pos, sa in zip(objs, positions, seeds_a):
            o.update(position=pos, seed_a=sa)
        plan.append({"scene_id": scene_id, "scene_prompt": prompt, "seed_b": seed_b, "kinds": kinds, "objects": objs, "raw": text})
        if s_i % 20 == 0:
            plan_path.write_text(json.dumps(plan, indent=1))
        print(scene_id, [o["object"] for o in objs], flush=True)
    plan_path.write_text(json.dumps(plan, indent=1))


def scenes(args):
    """B for every scene of a `propose` plan (MOSS-Image SFT, the run mode's settings), one manifest entry per planned object."""
    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    plan = json.loads((out / "plan.json").read_text())
    manifest_path = out / "pairs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {p["scene_id"] for p in manifest}
    for sc in plan:
        if sc["scene_id"] in done or not sc["objects"]:
            continue
        while (q := req("/queue"))["queue_running"] or q["queue_pending"]:
            time.sleep(2)
        pid = req("/prompt", {"prompt": graph(sc["scene_prompt"], sc["seed_b"], []), "client_id": "hades_edit_pairs"})["prompt_id"]
        while pid not in (h := req(f"/history/{pid}")):
            time.sleep(0.5)
        assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
        b_path = out / "pairs" / f"{sc['scene_id']}_B.png"
        fetch(h[pid]["outputs"], "out_b").save(b_path)
        for j, o in enumerate(sc["objects"]):
            manifest.append({"scene_id": sc["scene_id"], "pair_id": f"{sc['scene_id']}_{j}", "scene": "prompt", "scene_prompt": sc["scene_prompt"],
                             "seed_b": sc["seed_b"], "object": o["object"], "phrase": o["phrase"], "position": o["position"], "seed_a": o["seed_a"],
                             "instruction": f"Remove the {o['object']}.", "b": b_path.name, "sha256_b": hashlib.sha256(b_path.read_bytes()).hexdigest()})
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(sc["scene_id"], "B", [o["object"] for o in sc["objects"]], flush=True)


def combine(args):
    """Pairs from combinations: every set of 2+ usable objects of a scene whose changed areas keep GAP px apart is pasted together onto B
    (exact: B plus each object's A - B), and each object of the set is removed from it, the target keeping the others. Writes a new
    folder (--out) with the composites, hard links to the single-object images it uses, and its own pairs.json. CPU only."""
    import itertools
    import scipy.ndimage as nd
    src, out = Path(args.src), Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    gap = 8
    m = json.loads((src / "pairs.json").read_text())
    ok = lambda p: p.get("vlm") == ["yes", "no"] and p.get("count") == 1 and p.get("mask_area", 1) <= args.max_area  # noqa: E731
    by_scene = {}
    for p in m:
        if ok(p):
            by_scene.setdefault(p["scene_id"], []).append(p)
    pairs, n_sets = [], 0

    def link(name):
        dst = out / "pairs" / name
        if not dst.exists():
            try:
                os.link(src / "pairs" / name, dst)
            except OSError:
                shutil.copy2(src / "pairs" / name, dst)
        return name

    for sid, ps in by_scene.items():
        names = [p["object"] for p in ps]
        ps = [p for k, p in enumerate(ps) if names.count(p["object"]) == 1]  # "Remove the cat." is ambiguous with two cats
        if len(ps) < 2:
            continue
        b = np.asarray(Image.open(src / "pairs" / ps[0]["b"]).convert("RGB"), np.int16)
        d = [np.asarray(Image.open(src / "pairs" / p["a"]).convert("RGB"), np.int16) - b for p in ps]
        grown = [nd.binary_dilation(np.any(x != 0, -1), iterations=gap) for x in d]
        sup = [np.any(x != 0, -1) for x in d]
        apart = {(i, j) for i in range(len(ps)) for j in range(len(ps)) if i < j and not (grown[i] & sup[j]).any()}
        names_of = {}

        def image_of(subset):
            if len(subset) == 1:
                return link(ps[subset[0]]["a"])
            if subset not in names_of:
                name = f"{sid}_C{''.join(str(i) for i in subset)}.png"
                Image.fromarray(np.clip(b + sum(d[i] for i in subset), 0, 255).astype(np.uint8)).save(out / "pairs" / name)
                names_of[subset] = name
            return names_of[subset]
        link(ps[0]["b"])
        for size in range(2, len(ps) + 1):
            for subset in itertools.combinations(range(len(ps)), size):
                if not all(pair in apart for pair in itertools.combinations(subset, 2)):
                    continue
                n_sets += 1
                for i in subset:
                    rest = tuple(j for j in subset if j != i)
                    pairs.append({"scene_id": sid, "pair_id": f"{sid}_C{''.join(map(str, subset))}_rm{i}", "scene": ps[i].get("scene"),
                                  "scene_prompt": ps[i].get("scene_prompt"), "object": ps[i]["object"], "kept": [ps[j]["object"] for j in rest],
                                  "instruction": f"Remove the {ps[i]['object']}.", "a": image_of(subset), "b": image_of(rest),
                                  "vlm": ["yes", "no"], "count": 1, "mask_area": ps[i]["mask_area"], "box": ps[i]["box"]})
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    print("scenes", len(by_scene), "object sets", n_sets, "combination pairs", len(pairs), flush=True)


def plural(obj):
    return PLURAL.get(obj, obj + ("es" if obj.endswith(("s", "ch", "sh", "x")) else "s"))


WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "a": 1, "an": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def count(img, obj, tag):
    """How many of the object the MOSS text encoder sees in the crop (-1 if the answer has no number): Qwen-Image-Edit sometimes
    adds two of the object, and "Remove the X." on two of them is a different instruction."""
    t = ask(img, obj, tag, f"How many {plural(obj)} are in this picture? Answer with a number only.")
    for w in t.replace(".", " ").replace(",", " ").split():
        if w.isdigit():
            return int(w)
        if w in WORDS:
            return WORDS[w]
    return -1


def fetch(outputs, key):
    return Image.open(__import__("io").BytesIO(req("/view?" + urllib.parse.urlencode(outputs[key]["images"][0]), raw=True))).convert("RGB")


def run(args):
    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "pairs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {p["scene_id"] for p in manifest}
    rng = random.Random(args.seed)
    names = sorted(SCENES)
    plan = []
    for s in range(args.scenes):
        scene = names[s % len(names)] if s < len(names) else rng.choice(names)
        objs = rng.sample(SCENES[scene][1], args.objects_per_scene)
        plan.append((f"s{s:05d}", scene, args.seed * 100000 + s, objs, [rng.randrange(2 ** 31) for _ in objs], [rng.random() for _ in objs]))
    for scene_id, scene, seed_b, objs, seeds_a, box_draws in plan:
        if scene_id in done:
            continue
        prompt = SCENES[scene][0]
        if args.b_only:
            # B and the planned objects only, for `add`
            while (q := req("/queue"))["queue_running"] or q["queue_pending"]:
                time.sleep(2)
            pid = req("/prompt", {"prompt": graph(prompt, seed_b, []), "client_id": "hades_edit_pairs"})["prompt_id"]
            while pid not in (h := req(f"/history/{pid}")):
                time.sleep(0.5)
            assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
            b_path = out / "pairs" / f"{scene_id}_B.png"
            fetch(h[pid]["outputs"], "out_b").save(b_path)
            for j, (obj, seed_a) in enumerate(zip(objs, seeds_a)):
                manifest.append({"scene_id": scene_id, "pair_id": f"{scene_id}_{j}", "scene": scene, "scene_prompt": prompt, "seed_b": seed_b,
                                 "object": obj, "seed_a": seed_a, "instruction": f"Remove the {obj}.", "b": b_path.name,
                                 "sha256_b": hashlib.sha256(b_path.read_bytes()).hexdigest()})
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(scene_id, scene, "B only", objs, flush=True)
            continue
        items = []
        for j, (obj, seed_a, draw) in enumerate(zip(objs, seeds_a, box_draws)):
            phrase, region, size = OBJECTS[obj]
            box = pick_box(random.Random(draw), region, size)
            paint, paste = masks(box)
            items.append({"object": obj, "seed_a": seed_a, "box": box, "paste": paste,
                          "paint_prompt": f"{phrase[0].upper()}{phrase[1:]}. {prompt}",
                          "paint_mask": upload(f"hades_pair_{scene_id}_{j}.png", png(paint))})
        while (q := req("/queue"))["queue_running"] or q["queue_pending"]:
            time.sleep(2)
        pid = req("/prompt", {"prompt": graph(prompt, seed_b, items), "client_id": "hades_edit_pairs"})["prompt_id"]
        while pid not in (h := req(f"/history/{pid}")):
            time.sleep(0.5)
        assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
        o = h[pid]["outputs"]
        b_img = fetch(o, "out_b")
        b_path = out / "pairs" / f"{scene_id}_B.png"
        b_img.save(b_path)
        b = np.asarray(b_img, np.float32)
        for j, it in enumerate(items):
            painted = np.asarray(fetch(o, f"out{j}"), np.float32)
            m = np.asarray(it["paste"], np.float32)[..., None] / 255
            a = np.clip(np.rint(b * (1 - m) + painted * m), 0, 255).astype(np.uint8)
            a_path = out / "pairs" / f"{scene_id}_{j}_A.png"
            Image.fromarray(a).save(a_path)
            x0, y0, x1, y1 = it["box"]
            diff_in = float(np.abs(a[y0:y1, x0:x1].astype(np.float32) - b[y0:y1, x0:x1]).mean())
            outside = np.ones((SIZE, SIZE), bool)
            outside[max(y0 - PAD, 0):y1 + PAD + 1, max(x0 - PAD, 0):x1 + PAD + 1] = False  # PIL rectangles include their end pixel
            diff_out = float(np.abs(a[outside].astype(np.float32) - b[outside]).max())
            crop = (max(x0 - 96, 0), max(y0 - 96, 0), min(x1 + 96, SIZE), min(y1 + 96, SIZE))
            vlm = [ask(Image.fromarray(a).crop(crop), it["object"], f"{scene_id}_{j}_a"), ask(b_img.crop(crop), it["object"], f"{scene_id}_{j}_b")]
            manifest.append({"scene_id": scene_id, "pair_id": f"{scene_id}_{j}", "scene": scene, "scene_prompt": prompt, "seed_b": seed_b,
                             "object": it["object"], "paint_prompt": it["paint_prompt"], "seed_a": it["seed_a"], "box": it["box"], "pad": PAD,
                             "feather": FEATHER, "instruction": f"Remove the {it['object']}.", "a": a_path.name, "b": b_path.name,
                             "diff_in": round(diff_in, 2), "diff_out_max": diff_out, "vlm": vlm,
                             "sha256_a": hashlib.sha256(a_path.read_bytes()).hexdigest(), "sha256_b": hashlib.sha256(b_path.read_bytes()).hexdigest()})
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(scene_id, scene, [(p["object"], p["vlm"]) for p in manifest[-len(items):]], flush=True)


DIFF_SIGMA, DIFF_THRESHOLD, MIN_SPECK, GROW = 4, 40, 0.002, 16  # first try for the pilot, not measured


def change_mask(a, b):
    """Where the LoRA really drew something: blurred per-pixel difference over a threshold, specks under MIN_SPECK of the image
    dropped, grown by GROW px and feathered; returns the paste mask (0-1) and the area it covers."""
    import scipy.ndimage as nd
    d = nd.gaussian_filter(np.abs(a - b).mean(-1), DIFF_SIGMA) > DIFF_THRESHOLD
    lab, n = nd.label(d)
    if n:
        sizes = nd.sum(d, lab, range(1, n + 1))
        d = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_SPECK * d.size))
    d = nd.binary_dilation(d, iterations=GROW)
    m = nd.gaussian_filter(d.astype(np.float32), FEATHER / 2) * d  # the feather stays inside the grown area
    return m, float(d.mean())


def qwen_graph(image, text, seed):
    """Qwen-Image-Edit 2509 (Apache 2.0) as the adder: Q8 gguf (ComfyUI-GGUF loader), lightx2v's 2509 Lightning 8-step LoRA, the
    stock edit template otherwise (shift 3, CFGNorm, CFG 1, euler / simple, B's own latent at denoise 1)."""
    return {"unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q8_0.gguf"}},
            "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "qwen_edit_2509\\Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
            "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": 3.0}},
            "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "img": {"class_type": "LoadImage", "inputs": {"image": image}},
            "pos": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], "image1": ["img", 0]}},
            "neg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": "", "vae": ["vae", 0], "image1": ["img", 0]}},
            "lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0], "vae": ["vae", 0]}},
            "ks": {"class_type": "KSampler", "inputs": {"model": ["norm", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
                   "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
            "out": {"class_type": "PreviewImage", "inputs": {"images": ["dec", 0]}}}


def add(args):
    src, out = Path(args.src), Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "pairs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {p["pair_id"] for p in manifest}
    source = json.loads((src / "pairs.json").read_text())
    scenes = sorted({p["scene_id"] for p in source})[:args.scenes]
    for p in source:
        if p["scene_id"] not in scenes or p["pair_id"] in done:
            continue
        b_img = Image.open(src / "pairs" / p["b"]).convert("RGB")
        phrase = p.get("phrase") or OBJECTS[p["object"]][0]
        text = f"Add {phrase} {p['position']}." if p.get("position") else f"Add {phrase}."
        g = {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "mossImage_sft_bf16.safetensors", "weight_dtype": "default"}},
             "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": args.lora.replace("/", "\\"), "strength_model": 1.0}},
             "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
             "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
             "img": {"class_type": "LoadImage", "inputs": {"image": upload(f"hades_add_{p['scene_id']}_B.png", png(b_img))}},
             "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
             "ks": {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
                    "seed": p["seed_a"], "steps": 30, "cfg": 4.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
             "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
             "out": {"class_type": "PreviewImage", "inputs": {"images": ["dec", 0]}}}
        for k, t in (("pos", text), ("neg", NEG)):
            g[k] = {"class_type": "TextEncodeMOSSImageEdit", "inputs": {"clip": ["clip", 0], "prompt": t, "edit_format": "Moss-Image-Edit",
                    "images.image_0": ["img", 0], "image_strength": 1.0, "vae": ["vae", 0]}}
        if args.lora == "qwen2509":
            g = qwen_graph(g["img"]["inputs"]["image"], text, p["seed_a"])
        while (q := req("/queue"))["queue_running"] or q["queue_pending"]:
            time.sleep(2)
        pid = req("/prompt", {"prompt": g, "client_id": "hades_edit_pairs"})["prompt_id"]
        while pid not in (h := req(f"/history/{pid}")):
            time.sleep(0.5)
        assert h[pid]["status"]["status_str"] == "success", h[pid]["status"]
        raw = fetch(h[pid]["outputs"], "out")
        b = np.asarray(b_img, np.float32)
        r = np.asarray(raw, np.float32)
        m, area = change_mask(r, b)
        a = np.clip(np.rint(b * (1 - m[..., None]) + r * m[..., None]), 0, 255).astype(np.uint8)
        ys, xs = np.nonzero(m > 0)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, 0, 0]
        b_path, a_path = out / "pairs" / p["b"], out / "pairs" / f"{p['pair_id']}_A.png"
        if not b_path.exists():
            b_img.save(b_path)
        Image.fromarray(a).save(a_path)
        raw.save(out / "pairs" / f"{p['pair_id']}_raw.png")
        manifest.append({**{k: p[k] for k in ("scene_id", "pair_id", "scene", "scene_prompt", "seed_b", "object", "seed_a", "instruction", "b", "sha256_b")},
                         "adder": args.lora, "add_text": text, "box": box, "mask_area": round(area, 4), "a": a_path.name,
                         "raw_outside_mae": round(float(np.abs(r - b)[m == 0].mean()), 2) if (m == 0).any() else None,
                         "sha256_a": hashlib.sha256(a_path.read_bytes()).hexdigest()})
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(p["pair_id"], p["object"], "area", round(area, 3), flush=True)


def check(args):
    """Presence in A and B (`vlm`) and the count in A (`count`), for the pairs of an `add` folder that miss them. Run after `add`,
    so ComfyUI keeps one model set loaded per phase instead of swapping the editor and the text encoder at every pair."""
    out = Path(args.out)
    manifest_path = out / "pairs.json"
    manifest = json.loads(manifest_path.read_text())
    for i, p in enumerate(manifest):
        if "vlm" in p and "count" in p:
            continue
        box = p["box"]
        crop = (max(box[0] - 96, 0), max(box[1] - 96, 0), min(box[2] + 96, SIZE), min(box[3] + 96, SIZE)) if box[2] > box[0] else (0, 0, SIZE, SIZE)
        a_img = Image.open(out / "pairs" / p["a"]).convert("RGB").crop(crop)
        if "vlm" not in p:
            b_img = Image.open(out / "pairs" / p["b"]).convert("RGB").crop(crop)
            p["vlm"] = [ask(a_img, p["object"], f"{p['pair_id']}_a"), ask(b_img, p["object"], f"{p['pair_id']}_b")]
        p["count"] = count(a_img, p["object"], f"{p['pair_id']}_c")
        if i % 20 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=1))
        print(p["pair_id"], p["object"], p["vlm"], p["count"], flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))


def sheet(args):
    out = Path(args.out)
    ok = lambda p: p["vlm"] == ["yes", "no"] and p.get("count", 1) == 1  # noqa: E731
    manifest = sorted(json.loads((out / "pairs.json").read_text()), key=lambda p: (ok(p), p["pair_id"]))
    t, cols, rows = 256, 4, 8  # 4 pairs per row (A, B side by side), 32 pairs per page
    from PIL import ImageFont
    font = ImageFont.load_default(size=14)
    for page in range(0, len(manifest), cols * rows):
        chunk = manifest[page:page + cols * rows]
        s = Image.new("RGB", (cols * (2 * t + 12), -(-len(chunk) // cols) * (t + 22)), (18, 18, 20))
        d = ImageDraw.Draw(s)
        for k, p in enumerate(chunk):
            x, y = (k % cols) * (2 * t + 12), (k // cols) * (t + 22)
            for side, name in enumerate((p["a"], p["b"])):
                im = Image.open(out / "pairs" / name).convert("RGB")
                ImageDraw.Draw(im).rectangle(p["box"], outline=(255, 40, 40), width=6)
                s.paste(im.resize((t, t)), (x + side * t, y + 20))
            d.text((x + 2, y + 2), f"{page + k + 1}. {p['pair_id']}  {p['object']}  vlm {'/'.join(p['vlm'])}  n {p.get('count', '?')}", fill=(230, 230, 230) if ok(p) else (255, 120, 120), font=font)
        s.save(out / f"sheet_{page // (cols * rows) + 1:02d}.jpg", quality=88)
    print("pages", (len(manifest) + cols * rows - 1) // (cols * rows), "pairs", len(manifest),
          "vlm yes/no", sum(p["vlm"] == ["yes", "no"] for p in manifest), "and count 1", sum(ok(p) for p in manifest),
          "max diff outside", max((p.get("diff_out_max", 0) for p in manifest), default=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("run", "propose", "scenes", "add", "check", "combine", "sheet"))
    p.add_argument("--prompts", help="propose: json list of scene prompts (e.g. I:/HaDeS_MOSS/dmd_prompts/turbo_v10_prompts.json)")
    p.add_argument("--max-area", type=float, default=0.3, help="combine: largest changed area of a usable pair")
    p.add_argument("--from", dest="src", help="add: the run folder whose B images and objects to reuse")
    p.add_argument("--lora", help="add: the edit LoRA that draws the object (ComfyUI loras/ path), or qwen2509 for Qwen-Image-Edit 2509")
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", type=int, default=150)
    p.add_argument("--objects-per-scene", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--b-only", action="store_true", help="run: render B and plan the objects, no inpainting (the source for add)")
    a = p.parse_args()
    {"run": run, "propose": propose, "scenes": scenes, "add": add, "check": check, "combine": combine, "sheet": sheet}[a.mode](a)
