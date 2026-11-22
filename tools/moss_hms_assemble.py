"""Fill the original ffn30 container with all verified HMS shards, preserving file length."""
import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from safetensors import safe_open


def assemble(template, shards, captures, out, version, expected_scenes=4):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    if out.exists() or tmp.exists():
        raise FileExistsError(out)
    scenes = sorted(captures.glob("cal_*.json"))
    prompts = [json.loads(p.read_text(encoding="utf-8")) for p in scenes]
    if len(prompts) != expected_scenes:
        raise ValueError(f"{expected_scenes} calibration prompts required, got {len(prompts)}")
    recipe = dict(method="HMS", version=1, calibration=version, protected_ffn_blocks=list(range(30,42)),
                  prompt_file="prompts_40_variados.txt", prompt_indices_zero_based=[p["index"] for p in prompts])
    with template.open("rb") as source, tmp.open("xb") as dest:
        length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(length))
        header["__metadata__"]["hades/recipe"] = json.dumps(recipe,separators=(",",":"))
        raw = json.dumps(header,separators=(",",":"),ensure_ascii=False).encode()
        if len(raw)>length:
            raise ValueError("Recipe exceeds original header")
        dest.write(struct.pack("<Q", length))
        dest.write(raw.ljust(length,b" "))
        shutil.copyfileobj(source,dest,1<<23)
        for block in range(42):
            path = shards / f"block_{block:02}.safetensors"
            expected = {k for k in header if k.startswith(f"layers.{block}.proj.") and
                        k.rsplit(".",1)[-1] in ("weight","weight_s_rel","weight_codebook")}
            with safe_open(path,framework="pt") as f:
                if set(f.keys()) != expected:
                    raise ValueError(f"Incomplete {path}")
                for k in f.keys():
                    import torch
                    tensor = f.get_tensor(k)
                    spec = header[k]
                    sl = f.get_slice(k)
                    if sl.get_shape()!=spec["shape"] or sl.get_dtype()!=spec["dtype"]:
                        raise ValueError(k)
                    a,b = spec["data_offsets"]
                    data = tensor.view(torch.uint8).numpy()
                    if data.nbytes != b-a:
                        raise ValueError(k)
                    dest.seek(8+length+a)
                    dest.write(data)
    if tmp.stat().st_size != template.stat().st_size:
        raise ValueError("Size mismatch")
    with safe_open(tmp,framework="pt") as f:
        assert json.loads(f.metadata()["hades/recipe"]) == recipe
    tmp.rename(out)
    with out.open("rb") as f:
        print(out, out.stat().st_size, hashlib.file_digest(f,"sha256").hexdigest(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("template","shards","captures","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--calibration",default="C1 activation-weighted scale + codebook")
    p.add_argument("--expected-scenes",type=int,default=4)
    args=p.parse_args()
    assemble(args.template,args.shards,args.captures,args.out,args.calibration,args.expected_scenes)
