"""GPTQ in the ConvRot basis using fixed C1 grids, group 16, GPTQ block 128."""
import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from moss_hms_calibrate import rotate, core, _pack_codes


def gptq(weight, rows, scale, ch, lut):
    w = rotate(weight.float().cuda(), 256)
    x = rotate(rows.float().cuda(), 256)
    h = x.T @ x / len(rows)
    h.diagonal().add_(h.diagonal().mean() * 0.01)
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h, x
    n,k = w.shape
    bits = 6 if lut is None else 4
    table = torch.arange(-31,32,device="cuda").float() if lut is None else lut
    # Work in channel-scale units. Scaling each output row commutes with the GPTQ updates.
    w.div_(ch[:,None])
    codes = torch.empty((n,k), dtype=torch.uint8,device="cuda")
    recon = torch.empty_like(w)
    for start in range(0,k,128):
        stop = min(k,start+128)
        local = w[:,start:stop].contiguous()
        errors = torch.empty_like(local)
        for i in range(stop-start):
            j = start+i
            if j%16 == 0:
                levels = core._grid_levels(table,scale[:,j//16:j//16+1]).squeeze(1).contiguous()
            col = local[:,i:i+1].contiguous()
            pos = torch.searchsorted(levels,col)
            lo = (pos-1).clamp_(0,len(table)-1)
            hi = pos.clamp_(0,len(table)-1)
            a,b = levels.gather(1,lo),levels.gather(1,hi)
            pick = (col-b).abs() < (col-a).abs()
            q = torch.where(pick,b,a)
            codes[:,j:j+1] = torch.where(pick,hi,lo) + int(bits==6)
            recon[:,j:j+1] = q
            error = (col-q) / upper[j,j]
            errors[:,i:i+1] = error
            local[:,i:].sub_(error * upper[j,j:stop])
        if stop < k:
            w[:,stop:].addmm_(errors,upper[start:stop,stop:],alpha=-1)
    packed = _pack_codes(codes.int(),bits)
    for i in range(0,n,128):
        s=slice(i,i+128)
        expected=rotate((recon[s]*ch[s,None]).bfloat16(),256)
        actual=core.dequantize_w4a8_int8_weight(packed[s],scale[s],ch[s],lut)
        if not torch.equal(expected,actual):
            raise ValueError("GPTQ reconstruction mismatch")
    return packed.cpu()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","ffn30","rows","c1","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    p.add_argument("--blocks",type=int,nargs="+",required=True)
    p.add_argument("--expected-scenes",type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    scenes=sorted(a.rows.glob("cal_*.safetensors"))
    if len(scenes)!=a.expected_scenes:
        raise ValueError(f"Expected {a.expected_scenes} calibration scenes, got {len(scenes)}")
    with safe_open(a.source,framework="pt") as src,safe_open(a.ffn30,framework="pt") as ref:
        for block in a.blocks:
            out=a.out/f"block_{block:02}.safetensors"
            if out.exists():
                raise FileExistsError(out)
            tensors=load_file(str(a.c1/out.name))
            for fam in ("qkv","o","ffn_up_gate","ffn_down"):
                key=f"layers.{block}.proj.{fam}"
                rows=[]
                for scene in scenes:
                    with safe_open(scene,framework="pt") as data:
                        rows.append(data.get_tensor(key))
                lut=tensors.get(key+".weight_codebook")
                tensors[key+".weight"]=gptq(src.get_tensor(key+".weight"), torch.cat(rows),
                    tensors[key+".weight_s_rel"].cuda(),ref.get_tensor(key+".weight_s_channel").cuda(),
                    lut.cuda() if lut is not None else None)
                print("verified",key,flush=True)
            tmp=out.with_suffix(".partial")
            save_file(tensors,str(tmp))
            tmp.rename(out)


if __name__=="__main__":
    main()
