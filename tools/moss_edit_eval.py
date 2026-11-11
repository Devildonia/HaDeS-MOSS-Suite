"""Measure MOSS-edit layer ablations through an already running local ComfyUI.

All persistent images and records go to --out (default I:/HaDeS_MOSS/moss_edit_2026-11-10).
The original LoRA and the ComfyUI installation are not modified. Native PreviewImage
temporaries are downloaded, verified, and removed only from this checkout's temp directory.
Run prepare, inspect the references, run the full controls, then select ablations.
"""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[1]
GROUPS = [f"layers_{i:02d}_{i + 5:02d}" for i in range(0, 42, 6)]
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
STILL = ("Studio product photograph of one large plain red ceramic mug on the left half of a pale oak tabletop, "
         "one closed blue hardcover book lying on the right half, and a small green leafy plant in a white flowerpot "
         "behind the book at the far right. Plain warm beige wall, soft daylight from the left, realistic materials, "
         "straight-on three-quarter view, all three objects fully visible, clean separated silhouettes, no people, no writing.")
PORTRAIT = ("Natural waist-up photograph of an adult woman with short dark curly hair, brown eyes and a few freckles, "
            "wearing a plain cobalt blue zip-up jacket over a white shirt, standing in a leafy public park. "
            "She faces the camera with a relaxed neutral expression, hands outside the frame. Soft overcast daylight, "
            "green trees and a distant wooden bench in the softly blurred background, realistic skin, no text.")
REFERENCES = {"still_life": {"prompt": STILL, "seed": 32101}, "portrait": {"prompt": PORTRAIT, "seed": 32102}}
CASES = {
    "mug_blue": {"reference": "still_life", "instruction": "Change only the red mug to cobalt blue. Keep its shape and position, the book, plant, table, background and lighting unchanged.",
                 "caption": STILL.replace("red ceramic mug", "cobalt blue ceramic mug"), "edit_box": [0.05, 0.20, 0.65, 0.95]},
    "mug_remove": {"reference": "still_life", "instruction": "Remove only the red mug. Fill its former place with the same pale oak tabletop. Keep the book, plant, background, lighting and camera position unchanged.",
                   "caption": "Studio product photograph of a pale oak tabletop with empty space on the left half, one closed blue hardcover book lying on the right half, and a small green leafy plant in a white flowerpot behind the book at the far right. Plain warm beige wall, soft daylight from the left, realistic materials, straight-on three-quarter view, no people, no writing.",
                   "edit_box": [0.05, 0.20, 0.65, 0.95]},
    "jacket_red": {"reference": "portrait", "instruction": "Change only the blue jacket to bright red. Preserve the woman's face, hair, expression, pose, white shirt, background, camera framing and lighting.",
                   "caption": PORTRAIT.replace("cobalt blue zip-up jacket", "bright red zip-up jacket"), "edit_box": [0.10, 0.38, 0.90, 1.0], "identity_box": [0.25, 0.05, 0.75, 0.40]},
    "identity": {"reference": "still_life", "instruction": "Keep this image unchanged. Preserve every object, color, position, material, camera framing and lighting.",
                 "caption": STILL, "edit_box": None},
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path, previous=None):
    st = path.stat()
    identity = {"path": str(path.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if previous and all(previous.get(k) == v for k, v in identity.items()):
        return previous
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return {**identity, "sha256": h.hexdigest()}


def request(server, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(server + path, body, {"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')}") from e


def stage_reference(args, path):
    """Use ComfyUI's supported temporary upload, rather than bypassing its input-path boundary."""
    raw = path.read_bytes()
    name = f"hades_edit_ref_{digest(raw)[:20]}.png"
    boundary = "hades-moss-edit-eval-multipart"
    body = bytearray()
    for key, value in (("type", "temp"), ("subfolder", "hades_moss_edit_eval"), ("overwrite", "true")):
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode())
    body.extend(raw)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(args.server + "/upload/image", bytes(body), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        uploaded = json.load(r)
    return uploaded, f"{uploaded['subfolder']}/{uploaded['name']} [temp]"


def remove_temporary(args, name, subfolder):
    temp = (args.comfy_root / "temp").resolve()
    candidate = (temp / subfolder / name).resolve()
    if candidate.is_relative_to(temp) and candidate.is_file():
        candidate.unlink()


def base_graph(args):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": args.unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": "moss_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": args.size, "height": args.size, "batch_size": 1}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "out": {"class_type": "PreviewImage", "inputs": {"images": ["dec", 0]}},
    }


def reference_graph(args, reference):
    g = base_graph(args)
    for name, text in (("pos", reference["prompt"]), ("neg", NEGATIVE)):
        g[name] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": text}}
    g["ks"] = {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0],
               "latent_image": ["latent", 0], "seed": reference["seed"], "steps": 25, "cfg": 2.5,
               "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}}
    return g


def configurations():
    full = dict.fromkeys(GROUPS, 1.0)
    out = {"stock": None, "full": full, "none": dict.fromkeys(GROUPS, 0.0)}
    for group in GROUPS:
        suffix = group.removeprefix("layers_")
        out["in_" + suffix] = {g: float(g == group) for g in GROUPS}
        out["out_" + suffix] = {**full, group: 0.0}
    out["late_24_41"] = {g: float(i >= 4) for i, g in enumerate(GROUPS)}
    out["early_00_23"] = {g: float(i < 4) for i, g in enumerate(GROUPS)}
    return out


def edit_graph(args, case, config, seed):
    g = base_graph(args)
    g["ref"] = {"class_type": "LoadImage", "inputs": {"image": args.reference_inputs[case["reference"]]}}
    for name, text in (("pos", case[args.prompt_style]), ("neg", NEGATIVE)):
        g[name] = {"class_type": "TextEncodeMOSSImageEdit", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                   "prompt": text, "edit_format": "Moss-Image-Edit", "image_strength": 1.0, "images.image_0": ["ref", 0]}}
    model = "unet"
    if config == "stock":
        g["edit"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [model, 0], "lora_name": args.lora, "strength_model": 1.0}}
        model = "edit"
    elif config != "none":
        g["edit"] = {"class_type": "HaDeSMossLoraBlockWeight", "inputs": {"model": [model, 0], "lora_name": args.lora,
                     "strength_model": 1.0, "other_layers": 1.0, **configurations()[config]}}
        model = "edit"
    if args.turbo:
        g["turbo"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [model, 0], "lora_name": args.turbo, "strength_model": 1.0}}
        model = "turbo"
    g["ks"] = {"class_type": "KSampler", "inputs": {"model": [model, 0], "positive": ["pos", 0], "negative": ["neg", 0],
               "latent_image": ["latent", 0], "seed": seed, "steps": args.steps, "cfg": args.cfg,
               "sampler_name": args.sampler, "scheduler": args.scheduler, "denoise": 1.0}}
    return g


def run_graph(args, graph, path, identities):
    record_path = path.with_suffix(".json")
    specification = {"graph": graph, "models": identities}
    fingerprint = digest(json.dumps(specification, sort_keys=True).encode())
    if path.exists() and record_path.exists():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint or old["image_sha256"] != digest(path.read_bytes()):
            raise ValueError(f"Inputs or image changed for {path}; use a separate --out directory.")
        print("reuse", path, flush=True)
        return
    queue = json.loads(request(args.server, "/queue"))
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI has other queued work. Nothing was cancelled; rerun when the queue is empty.")
    try:
        submitted = json.loads(request(args.server, "/prompt", {"prompt": graph, "client_id": "hades_moss_edit_eval"}))
    except RuntimeError as e:
        write_json(record_path, {"fingerprint": fingerprint, **specification, "state": "validation_failed", "error": str(e)})
        raise
    pid = submitted["prompt_id"]
    write_json(record_path, {"fingerprint": fingerprint, **specification, "prompt_id": pid, "state": "submitted"})
    print("submitted", pid, path, flush=True)
    start = time.monotonic()
    while True:
        hist = json.loads(request(args.server, f"/history/{pid}"))
        if pid in hist:
            entry = hist[pid]
            break
        if time.monotonic() - start > args.timeout:
            raise TimeoutError(f"Timed out waiting for {pid}; its ComfyUI job was not cancelled.")
        time.sleep(0.5)
    if entry.get("status", {}).get("status_str") != "success":
        write_json(record_path, {"fingerprint": fingerprint, **specification, "prompt_id": pid, "history": entry, "state": "failed"})
        raise RuntimeError(f"ComfyUI failed on {pid}: {entry.get('status')}")
    image = entry["outputs"]["out"]["images"][0]
    raw = request(args.server, "/view?" + urllib.parse.urlencode(image))
    with Image.open(io.BytesIO(raw)) as check:
        check.load()
        if check.size != (args.size, args.size):
            raise ValueError(f"Unexpected output size: {check.size}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    messages = entry.get("status", {}).get("messages", [])
    stamps = {kind: data.get("timestamp") for kind, data in messages if "timestamp" in data}
    seconds = ((stamps["execution_success"] - stamps["execution_start"]) / 1000
               if "execution_success" in stamps and "execution_start" in stamps else None)
    write_json(record_path, {"fingerprint": fingerprint, **specification, "prompt_id": pid, "state": "success",
                            "image_sha256": digest(raw), "seconds": seconds, "status": entry.get("status"), "output": image})
    # Delete only the exact temporary produced by this graph, after its bytes and record are safely on --out.
    if image.get("type") == "temp" and image["filename"].startswith("ComfyUI_temp_"):
        remove_temporary(args, image["filename"], image.get("subfolder", ""))
    print("saved", path, f"({seconds}s)", flush=True)


def model_identities(args):
    manifest = args.out / "model_identities.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    files = {"unet": ("diffusion_models", args.unet), "clip": ("text_encoders", args.clip), "vae": ("vae", args.vae), "lora": ("loras", args.lora)}
    if args.turbo:
        files["turbo"] = ("loras", args.turbo)
    out = {}
    for kind, (folder, name) in files.items():
        print("fingerprint", kind, name, flush=True)
        out[kind] = file_identity(args.comfy_root / "models" / folder / name.replace("\\", "/"), previous.get(kind))
    write_json(manifest, out)
    return out


def spectra(args):
    torch.set_num_threads(4)
    path = args.comfy_root / "models" / "loras" / args.lora.replace("\\", "/")
    ranks = (2, 4, 8, 16, 32, 48, 64)
    rows = []
    with safe_open(path, framework="pt", device="cpu") as f:
        config = json.loads((f.metadata() or {}).get("adapter_config", "{}"))
        for key in f.keys():
            if not key.endswith(".lora_A"):
                continue
            down, up = f.get_tensor(key).float(), f.get_tensor(key[:-len("lora_A")] + "lora_B").float()
            ru, rd = torch.linalg.qr(up, mode="reduced").R, torch.linalg.qr(down.T, mode="reduced").R
            energy = torch.linalg.svdvals(ru @ rd.T).double().square()
            total = float(energy.sum())
            rows.append({"module": key[:-len(".lora_A")], "rank": down.shape[0], "energy": total,
                         "energy_kept": {str(r): float(energy[:r].sum()) / total if total else 1.0 for r in ranks}})
    report = {"source": file_identity(path), "adapter_metadata": config.get("metadata"), "modules": rows,
              "rank_mean_energy_pct": {str(r): 100 * sum(row["energy_kept"][str(r)] for row in rows) / len(rows) for r in ranks},
              "note": "Weight-space Frobenius energy, not edit quality or layer importance."}
    write_json(args.out / "spectra.json", report)
    print(json.dumps(report["rank_mean_energy_pct"], indent=2))


def region_mask(size, box):
    mask = np.zeros((size[1], size[0]), dtype=bool)
    if box is not None:
        x0, y0, x1, y1 = [round(v * size[i % 2]) for i, v in enumerate(box)]
        mask[y0:y1, x0:x1] = True
    return mask


def report(args):
    results = []
    root = args.out / "runs" / args.prompt_style
    font = ImageFont.truetype("DejaVuSans.ttf", 14) if pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").exists() else ImageFont.load_default()
    for case_name, case in CASES.items():
        ref_path = args.out / "references" / (case["reference"] + ".png")
        if not ref_path.exists():
            continue
        with Image.open(ref_path) as im:
            ref = np.asarray(im.convert("RGB"), dtype=np.float32)
        edit = region_mask((ref.shape[1], ref.shape[0]), case["edit_box"])
        for seed in args.seeds:
            paths = [(c, root / c / case_name / f"{seed}.png") for c in configurations()]
            paths = [(c, p) for c, p in paths if p.exists()]
            if not paths:
                continue
            full_path = root / "stock" / case_name / f"{seed}.png"
            full = np.asarray(Image.open(full_path).convert("RGB"), dtype=np.float32) if full_path.exists() else None
            tiles = [("reference", ref_path)] + paths
            columns, tile = 5, 256
            sheet = Image.new("RGB", (columns * tile, ((len(tiles) + columns - 1) // columns) * (tile + 30)), "#181818")
            draw = ImageDraw.Draw(sheet)
            for i, (name, path) in enumerate(tiles):
                x, y = (i % columns) * tile, (i // columns) * (tile + 30)
                draw.text((x + 4, y + 6), name, fill="white", font=font)
                with Image.open(path) as im:
                    sheet.paste(im.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), (x, y + 30))
                if name == "reference":
                    continue
                pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
                delta = np.abs(pixels - ref)
                row = {"case": case_name, "seed": seed, "config": name, "reference_mae_255": float(delta.mean()),
                       "preserved_region_mae_255": float(delta[~edit].mean()) if (~edit).any() else None,
                       "edited_region_mae_255": float(delta[edit].mean()) if edit.any() else None}
                if full is not None:
                    row.update(full_mae_255=float(np.abs(pixels - full).mean()), full_max_abs_255=float(np.abs(pixels - full).max()))
                if case.get("identity_box"):
                    identity = region_mask((ref.shape[1], ref.shape[0]), case["identity_box"])
                    row["identity_region_mae_255"] = float(delta[identity].mean())
                results.append(row)
            dest = args.out / "sheets" / args.prompt_style / f"{case_name}_{seed}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(dest, quality=94)
    write_json(args.out / f"metrics_{args.prompt_style}.json", {"results": results,
               "note": "Pixel differences measure drift, not instruction success. Regions are coarse boxes; inspect sheets and full-resolution files."})
    if results:
        fields = list(dict.fromkeys(k for row in results for k in row))
        with (args.out / f"metrics_{args.prompt_style}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    print("metrics", len(results), "runs", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("prepare", "spectra", "run", "report"))
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("I:/HaDeS_MOSS/moss_edit_2026-11-10"))
    p.add_argument("--comfy-root", type=pathlib.Path, default=ROOT.parents[1])
    p.add_argument("--unet", default="mossImage_sft_bf16.safetensors")
    p.add_argument("--clip", default="mossImage_base_txt_int4.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--lora", default="MOSS/MOSS-edit.safetensors")
    p.add_argument("--turbo", default="")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="normal")
    p.add_argument("--prompt-style", choices=("caption", "instruction"), default="instruction")
    p.add_argument("--configs", nargs="+", choices=list(configurations()), default=["stock", "full", "none"])
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    p.add_argument("--seeds", nargs="+", type=int, default=[1000])
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "spectra":
        spectra(args)
        return
    if args.mode == "report":
        report(args)
        return
    stats = json.loads(request(args.server, "/system_stats"))
    lora_info = json.loads(request(args.server, "/object_info/LoraLoaderModelOnly"))["LoraLoaderModelOnly"]
    names = lora_info["input"]["required"]["lora_name"][0]
    for field in ("lora", "turbo"):
        value = getattr(args, field)
        if value:
            match = next((name for name in names if name.replace("\\", "/") == value.replace("\\", "/")), None)
            if match is None:
                raise ValueError(f"LoRA not advertised by this ComfyUI: {value}")
            setattr(args, field, match)
    identities = model_identities(args)
    invocation = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arguments": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
                  "system": stats, "script_sha256": digest(pathlib.Path(__file__).read_bytes())}
    with (args.out / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(invocation) + "\n")
    write_json(args.out / "suite.json", {"references": REFERENCES, "cases": CASES, "groups": GROUPS})
    if args.mode == "prepare":
        for name, ref in REFERENCES.items():
            run_graph(args, reference_graph(args, ref), args.out / "references" / f"{name}.png", identities)
    else:
        args.reference_inputs = {}
        staged = []
        for name in {CASES[c]["reference"] for c in args.cases}:
            item, annotated = stage_reference(args, args.out / "references" / (name + ".png"))
            args.reference_inputs[name] = annotated
            staged.append(item)
        for config in args.configs:
            for case_name in args.cases:
                for seed in args.seeds:
                    case = CASES[case_name]
                    graph = edit_graph(args, case, config, seed)
                    ref = args.out / "references" / (case["reference"] + ".png")
                    run_graph(args, graph, args.out / "runs" / args.prompt_style / config / case_name / f"{seed}.png",
                              {**identities, "reference": file_identity(ref)})
            report(args)
        for item in staged:
            if item.get("type") == "temp" and item["name"].startswith("hades_edit_ref_"):
                remove_temporary(args, item["name"], item.get("subfolder", ""))


if __name__ == "__main__":
    main()
