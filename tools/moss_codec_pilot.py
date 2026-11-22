"""Compare weight-selected and activation-selected residuals on identical byte budgets on real MOSS weights."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from moss_vector_codec import VERSION, encode, read_matrix


def nearest(points, table):
    norms = table.square().sum(1)
    return torch.cat([(norms - 2 * part @ table.T).argmin(1) for part in points.split(8192)])


def fit(points):
    # Fixed initialization, deterministic CPU centroid sums, stored FP16 centroids used for assignment.
    table = points[:256].clone()
    for _ in range(10):
        indices = nearest(points, table).cpu()
        sums = torch.zeros(256, 4).index_add_(0, indices, points.cpu())
        counts = torch.bincount(indices, minlength=256)
        active = counts > 0
        updated = table.cpu()
        updated[active] = sums[active] / counts[active, None]
        table = updated.cuda()
    return table.half().float()


def top_mask(score):
    indices = score.flatten().topk(score.numel()//4, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.flatten()[indices] = True
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "captures", "out"):
        p.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = torch.Generator().manual_seed(1121)
    cal = [load_file(str(f)) for f in sorted(args.captures.glob("cal_*.safetensors"))]
    if len(cal) != 2:
        raise ValueError("Expected two calibration scenes")
    variants = ("weights", "activations")
    archives = {k:{} for k in variants}
    manifest, expected, rows_by_name = {}, {}, {}
    with safe_open(args.source, framework="pt") as src:
        for name in sorted(cal[0]):
            tensor = src.get_tensor(name+".weight")
            indices = torch.randperm(tensor.shape[0],generator=rng)[:256]
            train = tensor[indices[:128]].cuda().float()
            w = tensor[indices[128:]].cuda().float()
            rows_by_name[name] = indices[128:].tolist()
            width = w.shape[1]
            manifest[name] = width
            def normalize(x):
                scale = x.square().mean(1).sqrt().half().float().clamp_min(1e-7)
                return (x/scale[:,None]).reshape(len(x),-1,4),scale
            training, _ = normalize(train)
            pts = training.reshape(-1,4)
            pts = pts[torch.randperm(len(pts),generator=rng)[:16384].cuda()]
            base = fit(pts)
            residual = fit(pts-base[nearest(pts,base)])
            target, scales = normalize(w)
            main_idx = nearest(target.reshape(-1,4),base).reshape(target.shape[:2])
            err = target-base[main_idx]
            res_idx = nearest(err.reshape(-1,4),residual).reshape_as(main_idx)
            # Gains reflect the actual BF16 rounding performed by the decoder.
            before = w.reshape_as(target) - (base[main_idx]*scales[:,None,None]).bfloat16().float()
            after = w.reshape_as(target) - ((base[main_idx]+residual[res_idx])*scales[:,None,None]).bfloat16().float()
            x = torch.cat([scene[name] for scene in cal]).cuda().float().reshape(-1,width//4,4)
            covariance = torch.einsum("ngi,ngj->gij",x,x)/len(x)
            def impact(error):
                return torch.einsum("rgi,gij,rgj->rg",error,covariance,error)
            gains = (before.square().sum(-1)-after.square().sum(-1), impact(before)-impact(after))
            for variant, gain in zip(variants,gains):
                sel = top_mask(gain)
                fields = encode(scales,base,residual,main_idx,res_idx,sel)
                archives[variant].update({name+"."+k:v.cpu().contiguous() for k,v in fields.items()})
                dense = (base[main_idx]+sel[:,:,None]*residual[res_idx])*scales[:,None,None]
                expected[name,variant] = dense.reshape_as(w).bfloat16().cpu()
    metadata = dict(version=VERSION, columns=json.dumps(manifest,sort_keys=True))
    for variant in variants:
        tmp = args.out/(variant+".tmp")
        save_file(archives[variant],str(tmp),metadata=metadata)
        for name in manifest:
            for device in ("cpu", "cuda"):
                dense = read_matrix(tmp, name, device).cpu()
                if not torch.equal(dense,expected[name,variant]):
                    raise ValueError("Round-trip mismatch on "+device)
        tmp.rename(args.out/(variant+".safetensors"))
    sizes = [(args.out/(v+".safetensors")).stat().st_size for v in variants]
    if sizes[0]!=sizes[1]:
        raise ValueError("Unequal byte budget")
    # Open validation activations only after both artifacts are frozen.
    records = []
    with safe_open(args.source,framework="pt") as src:
        for path in sorted(args.captures.glob("val_*.safetensors")):
            scene = load_file(str(path))
            for name,ids in rows_by_name.items():
                w = src.get_tensor(name+".weight")[ids].cuda().float()
                x = scene[name].cuda().float()
                reference = x @ w.T
                row = dict(scene=path.stem, matrix=name, tokens=len(x))
                for v in variants:
                    y = x @ expected[name,v].cuda().float().T
                    row[v] = float((y-reference).norm()/reference.norm())
                records.append(row)
    if len(records)!=24:
        raise ValueError("Incomplete held-out evaluation")
    report = dict(bytes_each=sizes[0],rows=rows_by_name,validation=records,seed=1121,
                  hashes={v:hashlib.sha256((args.out/(v+".safetensors")).read_bytes()).hexdigest() for v in variants})
    (args.out/"results.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    with torch.inference_mode():
        main()
