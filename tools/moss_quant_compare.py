"""Score paired Base renders, run MOSS Text Check, and produce blinded pairs for human review."""
import argparse
import csv
import importlib
import json
import random
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from moss_v094_range_eval import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "candidate", "core", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--blind-zip", type=Path, help="Write original A/B PNGs under anonymous names")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    suite = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(suite.parent))
    check = importlib.import_module(f"{suite.name}.moss_textcheck_nodes").check
    indices = (0, 6, 10, 14, 20, 25, 27, 30, 34, 35)
    labels = {14: "OPEN", 34: "VISIT THE FJORDS"}
    key, texts, rows = {}, {}, []
    sheet = Image.new("RGB", (1024, 5*284), "white")
    pen = ImageDraw.Draw(sheet)
    rng = random.SystemRandom()
    with torch.inference_mode():
        clip, = nodes.CLIPLoader().load_clip("qwen3.5_9b_w4a8_hybrid.safetensors", "moss_image")
        for slot, idx in enumerate(indices):
            base_path, cand_path = [d / f"{idx:02}.png" for d in (args.reference, args.candidate)]
            graphs = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in (base_path,cand_path)]
            for g in graphs:
                del g["unet"]["inputs"]["unet_name"]
            if graphs[0] != graphs[1]:
                raise ValueError(f"Non-model workflow difference in pair {idx}")
            pair = [Image.open(p).convert("RGB") for p in (base_path, cand_path)]
            row = dict(index=idx)
            for size in (64,16,None):
                samples = [im.resize((size,size),Image.Resampling.BICUBIC) if size else im for im in pair]
                a,b = [np.array(im,dtype=np.int16) for im in samples]
                mae = float(np.abs(a-b).mean())
                if size:
                    row[f"l1_{size}_pct"] = mae / 255 * 100
                else:
                    row["mae_rgb_255"] = mae
            for tag,path,im in zip(("reference","candidate"),(base_path,cand_path),pair):
                row[f"lap_var_{tag}"] = metrics(path.read_bytes())["laplacian_std"]**2
                if idx in labels:
                    tensor = torch.from_numpy(np.array(im)).float().unsqueeze(0)/255
                    verdicts, _, lines = check(clip,tensor,expected_text=labels[idx])
                    texts[f"{idx}_{tag}"] = dict(verdicts=verdicts,lines=lines)
            rows.append(row)
            left = rng.randrange(2)
            key[idx] = "reference" if left == 0 else "candidate"
            x, y = slot%2*512, slot//2*284
            for j,im in enumerate((pair[left],pair[1-left])):
                sheet.paste(im.resize((256,256),Image.Resampling.LANCZOS),(x+256*j,y+25))
                pen.text((x+256*j+5,y+5), f"{idx:02} {'AB'[j]}",fill="black")
    with (args.out / "pairs.csv").open("x",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    sheet.save(args.out / "blind_pairs.jpg", quality=94)
    (args.out / "blind_key_A.json").write_text(json.dumps(key,indent=2))
    if args.blind_zip:
        write_blind_zip(args.blind_zip, key, args.reference, args.candidate)
    (args.out / "text_check.json").write_text(json.dumps(texts,indent=2))
    summary = {}
    for name in ("l1_64_pct", "l1_16_pct", "mae_rgb_255", "lap_var_reference", "lap_var_candidate"):
        values = [row[name] for row in rows]
        summary[name] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                             minimum=min(values), maximum=max(values), max_index=rows[int(np.argmax(values))]["index"])
    for label, folder in (("reference", args.reference), ("candidate", args.candidate)):
        runs = json.loads((folder / "runs.json").read_text())
        if [r["index"] for r in runs] != list(indices):
            raise ValueError(f"Incomplete run list in {folder}")
        summary[label] = dict(first_seconds=runs[0]["seconds"],
                              median_warm_seconds=float(np.median([r["seconds"] for r in runs[1:]])),
                              peak_allocated_bytes=max(r["peak_allocated"] for r in runs),
                              nvfp4_mm_completed=sum(r["nvfp4_mm_completed"] for r in runs))
        for name in ("convrot_mm_completed", "w4a8_mm_completed"):
            if name in runs[0]:
                summary[label][name] = sum(r[name] for r in runs)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


def write_blind_zip(path, key, reference, candidate):
    """Keep the exact PNG bytes and use the sheet's per-pair A/B assignments."""
    with ZipFile(path, "x") as archive:
        for idx, left in key.items():
            folders = (reference, candidate) if left == "reference" else (candidate, reference)
            for letter, folder in zip("AB", folders):
                name = f"{int(idx):02}"
                archive.writestr(f"{name}{letter}.png", (folder / f"{name}.png").read_bytes())


if __name__ == "__main__":
    main()
