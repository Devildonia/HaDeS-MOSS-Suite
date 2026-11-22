"""Score rendered text images with MOSS Text Check and build comparison sheets."""
import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EXPECTED_TEXTS = [
    "FRESH BREAD DAILY",
    "VISIT THE FJORDS",
    "OPEN 24 HOURS",
    "COFFEE 2.50\nTEA 1.80",
    "THE LAST NIGHT\nM. ROSSI",
    "MAIN STREET",
    "NORTH SEA\nSOON",
    "EGGS FOR SALE",
]
SEEDS = (1001, 1002, 1003)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes

    suite = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(suite.parent))
    check = importlib.import_module(f"{suite.name}.moss_textcheck_nodes").check

    families = {
        "base": [("bf16", args.root / "base_bf16"), ("v1", args.root / "base_v1"), ("ffn30", args.root / "base_ffn30")],
        "turbo": [("bf16", args.root / "turbo_bf16"), ("v1", args.root / "turbo_v1"), ("ffn30", args.root / "turbo_ffn30")],
    }

    details = []
    summary = {}
    with torch.inference_mode():
        clip, = nodes.CLIPLoader().load_clip("qwen3.5_9b_w4a8_hybrid.safetensors", "moss_image")
        for fam, models in families.items():
            summary[fam] = {}
            for mname, mdir in models:
                counts = {"exact": 0, "close": 0, "missing": 0, "outside its box": 0}
                failures = []
                for p_idx, exp in enumerate(EXPECTED_TEXTS):
                    for seed in SEEDS:
                        img_path = mdir / f"p{p_idx:02d}_s{seed}.png"
                        im = Image.open(img_path).convert("RGB")
                        tensor = torch.from_numpy(np.array(im)).float().unsqueeze(0) / 255.0
                        verdicts, _, lines = check(clip, tensor, expected_text=exp)
                        for v in verdicts:
                            vd = v["verdict"]
                            counts[vd] = counts.get(vd, 0) + 1
                            details.append(dict(
                                family=fam,
                                model=mname,
                                prompt_index=p_idx,
                                seed=seed,
                                expected=v["text"],
                                read=v["read"],
                                similarity=v["similarity"],
                                verdict=vd,
                                literal=v.get("literal", True),
                            ))
                            if vd != "exact":
                                failures.append(f"p{p_idx:02d}/s{seed}: {v['text']} -> {v['read']} [{vd}]")
                summary[fam][mname] = {"counts": counts, "failures": failures}
                print(fam, mname, counts, flush=True)

    sheets_dir = args.root / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    tile = 384
    pad_top = 28
    for fam, models in families.items():
        for p_idx in range(len(EXPECTED_TEXTS)):
            sheet = Image.new("RGB", (tile * 3, (tile + pad_top) * len(SEEDS)), "white")
            pen = ImageDraw.Draw(sheet)
            for r_idx, seed in enumerate(SEEDS):
                y0 = r_idx * (tile + pad_top)
                for c_idx, (mname, mdir) in enumerate(models):
                    x0 = c_idx * tile
                    im = Image.open(mdir / f"p{p_idx:02d}_s{seed}.png").convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                    sheet.paste(im, (x0, y0 + pad_top))
                    pen.text((x0 + 8, y0 + 6), f"{fam.upper()} | {mname} | p{p_idx:02d} | s{seed}", fill="black")
            sheet.save(sheets_dir / f"{fam}_p{p_idx:02d}.jpg", quality=92)

    (args.root / "text_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.root / "text_details.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)


if __name__ == "__main__":
    main()
