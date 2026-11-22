"""Reference HaDeS vector codec v0: four-weight codes, a bitmap and sparse residual codes."""
import json

import torch
from safetensors import safe_open

VERSION = "hades_vq4r_pilot_v0"


def encode(rows, base, residual, main_index, residual_index, selected):
    """Pack one row-normalized matrix. Bit 0 addresses the first group in each byte."""
    flat = selected.flatten()
    mask = torch.nn.functional.pad(flat, (0, (-flat.numel()) % 8)).reshape(-1, 8)
    bits = (mask.to(torch.int32) << torch.arange(8, device=rows.device)).sum(1).byte()
    return dict(base=base.half(), residual=residual.half(), codes=main_index.byte(),
                mask=bits, correction=residual_index.flatten()[flat].byte(), scales=rows.half())


def decode(fields, columns):
    codes = fields["codes"]
    count = codes.numel()
    device = codes.device
    if codes.ndim != 2 or columns < 1 or codes.shape[1] != (columns+3)//4:
        raise ValueError("Invalid original width or index shape")
    for name in ("base", "residual"):
        if fields[name].shape != (256, 4) or fields[name].dtype != torch.float16:
            raise ValueError(f"Invalid {name} dictionary")
    if fields["scales"].shape != (codes.shape[0],) or fields["scales"].dtype != torch.float16:
        raise ValueError("Invalid row scales")
    if any(fields[k].dtype != torch.uint8 for k in ("codes", "mask", "correction")):
        raise ValueError("Expected byte indices and bitmap")
    if fields["mask"].ndim != 1 or fields["correction"].ndim != 1:
        raise ValueError("Expected flat bitmap and correction indices")
    if fields["mask"].numel() != (count+7)//8:
        raise ValueError("Invalid correction bitmap length")
    marked = ((fields["mask"][:,None].long() >> torch.arange(8,device=device)) & 1).flatten().bool()
    if marked[count:].any() or marked.sum().item() != fields["correction"].numel():
        raise ValueError("Correction count does not match bitmap")
    v = fields["base"][codes.long()].float().reshape(-1,4)
    v[marked[:count]] += fields["residual"][fields["correction"].long()].float()
    return (v.reshape(codes.shape[0],-1)[:,:columns] * fields["scales"].float().reshape(-1,1)).to(torch.bfloat16)


def read_matrix(path, name, device="cpu"):
    """Read one matrix from a v0 archive using stock tensor operations."""
    with safe_open(path, framework="pt") as file:
        metadata = file.metadata() or {}
        if metadata.get("version") != VERSION:
            raise ValueError("Unsupported vector codec version")
        columns = json.loads(metadata["columns"])[name]
        fields = {k:file.get_tensor(name+"."+k).to(device)
                  for k in ("base", "residual", "codes", "mask", "correction", "scales")}
    return decode(fields, columns)
