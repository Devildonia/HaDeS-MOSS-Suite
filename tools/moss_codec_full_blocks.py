"""Extend the frozen pilot dictionaries to full MOSS matrices; rank corrections globally per matrix."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from moss_codec_pilot import nearest, top_mask
from moss_vector_codec import VERSION, encode, read_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source","pilot","captures","out"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    tables = load_file(str(args.pilot))
    cal = [load_file(str(p)) for p in sorted(args.captures.glob("cal_*.safetensors"))]
    if len(cal) != 2:
        raise ValueError("Expected the two original calibration scenes")
    packed = {"weights":{}, "activations":{}}
    widths = {}
    with safe_open(args.source, framework="pt") as src:
        for name in sorted(cal[0]):
            w = src.get_tensor(name+".weight").cuda().float()
            width = widths[name] = w.shape[1]
            x = torch.cat([d[name] for d in cal]).cuda().float().reshape(-1,width//4,4)
            cov = torch.einsum("ngi,ngj->gij",x,x)/len(x)
            base, residual = [tables[name+"."+tag].cuda().float() for tag in ("base","residual")]
            scales = w.square().mean(1).sqrt().half().float().clamp_min(1e-7)
            chunks = []
            for start in range(0,len(w),128):
                scale = scales[start:start+128,None,None]
                target = w[start:start+128].reshape(-1,width//4,4)
                normalized = target/scale
                codes = nearest(normalized.reshape(-1,4),base).reshape(target.shape[:2])
                rcodes = nearest((normalized-base[codes]).reshape(-1,4),residual).reshape_as(codes)
                e0 = target-(base[codes]*scale).bfloat16().float()
                e1 = target-((base[codes]+residual[rcodes])*scale).bfloat16().float()
                def impact(e):
                    return torch.einsum("rgi,gij,rgj->rg",e,cov,e)
                chunks.append((codes.byte(),rcodes.byte(),(e0.square()-e1.square()).sum(-1),impact(e0)-impact(e1)))
            codes,rcodes,weight_gain,act_gain = [torch.cat(parts) for parts in zip(*chunks)]
            for tag,gain in (("weights",weight_gain),("activations",act_gain)):
                fields = encode(scales,base,residual,codes,rcodes,top_mask(gain))
                packed[tag].update({name+"."+k:v.cpu().contiguous() for k,v in fields.items()})
            print(name, tuple(w.shape), flush=True)
    meta = dict(version=VERSION,columns=json.dumps(widths,sort_keys=True))
    sizes=[]
    for tag,fields in packed.items():
        path = args.out/(tag+".safetensors")
        tmp = path.with_suffix(".tmp")
        save_file(fields,str(tmp),metadata=meta)
        for name in widths:
            data = read_matrix(tmp,name,"cuda")
            if not torch.isfinite(data).all():
                raise ValueError(name)
        tmp.rename(path)
        sizes.append(path.stat().st_size)
    if sizes[0]!=sizes[1]:
        raise RuntimeError("Different byte budgets")


if __name__=="__main__":
    with torch.inference_mode():
        main()
