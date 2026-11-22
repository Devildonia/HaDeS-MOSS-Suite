"""Score 10-seed Base text renders (BF16, ffn30, C2) with MOSS Text Check and build failure sheets."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="2001,2002,2003,2004,2005,2006,2007,2008,2009,2010")
    parser.add_argument("--models", type=str, default=None, help="Optional comma-separated name=dir pairs")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.models:
        models = [(k.strip(), Path(v.strip())) for item in args.models.split(",") for k, v in [item.split("=")]]
    else:
        models = [("bf16", args.root / "base_bf16"), ("ffn30", args.root / "base_ffn30"), ("c2", args.root / "base_c2")]

    sys.path.insert(0, str(args.core.resolve()))
    sys.argv[1:] = ["--use-pytorch-cross-attention", "--offline", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes

    suite = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(suite.parent))
    check = importlib.import_module(f"{suite.name}.moss_textcheck_nodes").check

    details = []
    summary = {}
    fail_cases = {}  # (p_idx, seed) -> {mname: [fail_strs]}

    with torch.inference_mode():
        clip, = nodes.CLIPLoader().load_clip("qwen3.5_9b_w4a8_hybrid.safetensors", "moss_image")
        for mname, mdir in models:
            counts = {"exact": 0, "close": 0, "missing": 0, "outside its box": 0}
            failures = []
            for p_idx, exp in enumerate(EXPECTED_TEXTS):
                for seed in seeds:
                    img_path = mdir / f"p{p_idx:02d}_s{seed}.png"
                    im = Image.open(img_path).convert("RGB")
                    tensor = torch.from_numpy(np.array(im)).float().unsqueeze(0) / 255.0
                    verdicts, _, _ = check(clip, tensor, expected_text=exp)
                    for v in verdicts:
                        vd = v["verdict"]
                        counts[vd] = counts.get(vd, 0) + 1
                        details.append(dict(
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
                            msg = f"{v['text']} -> {v['read']} [{vd}, {v['similarity']:.2f}]"
                            failures.append(f"p{p_idx:02d}_s{seed}: {msg}")
                            fail_cases.setdefault((p_idx, seed), {}).setdefault(mname, []).append(msg)
            summary[mname] = {"counts": counts, "failures": failures}
            print(mname, counts, flush=True)

    fails_dir = args.root / "fails"
    fails_dir.mkdir(parents=True, exist_ok=True)
    tile = 512
    pad_top = 48
    for (p_idx, seed), per_model in sorted(fail_cases.items()):
        sheet = Image.new("RGB", (tile * len(models), tile + pad_top), "white")
        pen = ImageDraw.Draw(sheet)
        for c_idx, (mname, mdir) in enumerate(models):
            x0 = c_idx * tile
            im = Image.open(mdir / f"p{p_idx:02d}_s{seed}.png").convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(im, (x0, pad_top))
            note = "; ".join(per_model.get(mname, ["EXACT"]))
            pen.text((x0 + 8, 6), f"{mname.upper()} | p{p_idx:02d}_s{seed}", fill="black")
            pen.text((x0 + 8, 24), note[:68], fill="red" if mname in per_model else "green")
        sheet.save(fails_dir / f"p{p_idx:02d}_s{seed}.jpg", quality=92)

    (args.root / "text_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.root / "text_details.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)
    print(f"Saved {len(fail_cases)} failure sheets to {fails_dir}", flush=True)


if __name__ == "__main__":
    main()
