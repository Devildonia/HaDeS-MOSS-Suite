"""Score a completed HMS grid and text set against the frozen BF16 PNGs."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from moss_text_score_sheets import EXPECTED_TEXTS, SEEDS
from moss_v094_range_eval import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("core","reference","candidate","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--text",action="store_true")
    p.add_argument("--seeds",type=str,default=None)
    a=p.parse_args()
    seeds=[int(s.strip()) for s in a.seeds.split(",") if s.strip()] if a.seeds else SEEDS
    a.out.mkdir(exist_ok=False,parents=True)
    sys.path.insert(0,str(a.core.resolve()))
    sys.argv[1:]=["--use-pytorch-cross-attention","--offline","--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import torch
    suite=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(suite.parent))
    check=importlib.import_module(f"{suite.name}.moss_textcheck_nodes").check
    if a.text:
        cases={f"p{i:02}_s{s}":v for i,v in enumerate(EXPECTED_TEXTS) for s in seeds}
    else:
        cases={f"{i:02}":{14:"OPEN",34:"VISIT THE FJORDS"}.get(i) for i in (0,6,10,14,20,25,27,30,34,35)}
    runs=json.loads((a.candidate/"runs.json").read_text())
    ref_runs=json.loads((a.reference/"runs.json").read_text())
    def by_case(records):
        return {(Path(v["file"]).stem if a.text else f"{v['index']:02}"):v for v in records}
    run_map, ref_map = by_case(runs),by_case(ref_runs)
    if len(runs)!=len(cases) or set(run_map)!=set(cases) or set(ref_map)!=set(cases):
        raise ValueError("Incomplete run")
    rows,verdicts=[],{}
    with torch.inference_mode():
        clip,=nodes.CLIPLoader().load_clip("qwen3.5_9b_w4a8_hybrid.safetensors","moss_image")
        for case,expected in cases.items():
            base,cand=[d/(case+".png") for d in (a.reference,a.candidate)]
            pair=[Image.open(path).convert("RGB") for path in (base,cand)]
            for im,record in zip(pair,(ref_map[case],run_map[case])):
                if hashlib.sha256(im.tobytes()).hexdigest()!=record["rgb_sha256"]:
                    raise ValueError(f"RGB hash mismatch in {case}")
            if any(im.size!=(1024,1024) for im in pair):
                raise ValueError(case)
            row=dict(case=case)
            for size in (None,64,16):
                images=[im.resize((size,size),Image.Resampling.BICUBIC) if size else im for im in pair]
                x,y=[np.array(im,dtype=np.int16) for im in images]
                mae=float(np.abs(y-x).mean())
                if size:
                    row[f"l1_{size}_pct"]=mae*100/255
                else:
                    row["mae_rgb_255"]=mae
            for path,tag in ((base,"reference"),(cand,"candidate")):
                m=metrics(path.read_bytes())
                row["lap_"+tag]=m["laplacian_std"]**2
                row["luma_"+tag]=m["mean_luma_255"]
            row["delta_luma"]=row["luma_candidate"]-row["luma_reference"]
            rows.append(row)
            if expected:
                tensor=torch.from_numpy(np.array(pair[1])).float().unsqueeze(0)/255
                vv,_,lines=check(clip,tensor,expected_text=expected)
                verdicts[case]=dict(verdicts=vv,lines=lines)
            (a.out/"text_check.json").write_text(json.dumps(verdicts,indent=2))
    with (a.out/"cases.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary={k:float(np.mean([row[k] for row in rows])) for k in rows[0] if k!="case"}
    summary.update(luma_median=float(np.median([row["delta_luma"] for row in rows])),
                   brighter=sum(row["delta_luma"]>0 for row in rows),n=len(rows),
                   exact=sum(v["verdict"]=="exact" for case in verdicts.values() for v in case["verdicts"]),
                   text_total=sum(len(case["verdicts"]) for case in verdicts.values()),
                   warm_s=float(np.median([v["seconds"] for v in runs[1:]])),
                   peak_bytes=max(v["peak_allocated"] for v in runs),
                   native_calls=sum(v["w4a8_mm_completed"] for v in runs))
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
