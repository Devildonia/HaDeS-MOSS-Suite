"""Convert a complete MOSS-Image BF16 checkpoint to FWHT-64 packed format and optionally decode to dense BF16."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from moss_codec_v2_screen import decode_fwht_scalar, encode_fwht_scalar, hadamard_matrix

VERSION_FULL = "hades_fwht64_v2_full"
TARGET_SUFFIXES = (".proj.qkv.weight", ".proj.o.weight", ".proj.ffn_up_gate.weight", ".proj.ffn_down.weight")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=5, choices=(4, 5, 6))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dense-out", type=Path, default=None)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    for p in (args.out, args.audit_out, args.dense_out):
        if p is not None:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing path: {p}")
            p.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    h64 = hadamard_matrix(64, device="cuda")
    t0 = time.perf_counter()

    packed_tensors = {}
    columns = {}
    packed_bytes = 0
    scale_bytes = 0
    literal_bytes = 0
    quantized_count = 0
    literal_count = 0
    expected_hashes = {}

    with safe_open(args.source, framework="pt") as src:
        for key in sorted(src.keys()):
            tensor = src.get_tensor(key)
            if key.startswith("layers.") and key.endswith(TARGET_SUFFIXES):
                mat_name = key[: -len(".weight")]
                width = int(tensor.shape[1])
                columns[mat_name] = width
                fields = encode_fwht_scalar(tensor, h64, args.bits)
                recon = decode_fwht_scalar(fields, width, args.bits, h64).cpu().contiguous()
                if not torch.isfinite(recon).all():
                    raise ValueError(f"Non-finite reconstruction: {key}")
                expected_hashes[key] = hashlib.sha256(recon.view(torch.uint8).numpy().tobytes()).hexdigest()
                packed_tensors[f"{mat_name}.packed"] = fields["packed"]
                packed_tensors[f"{mat_name}.scales"] = fields["scales"]
                packed_bytes += fields["packed"].numel() * fields["packed"].element_size()
                scale_bytes += fields["scales"].numel() * fields["scales"].element_size()
                quantized_count += 1
            else:
                t_cpu = tensor.cpu().contiguous()
                packed_tensors[key] = t_cpu
                literal_bytes += t_cpu.numel() * t_cpu.element_size()
                literal_count += 1

    meta = {
        "version": VERSION_FULL,
        "group_size": "64",
        "bits": str(args.bits),
        "columns": json.dumps(columns, sort_keys=True),
    }
    tmp_out = args.out.with_suffix(".tmp")
    save_file(packed_tensors, str(tmp_out), metadata=meta)
    tmp_out.rename(args.out)
    del packed_tensors

    # Round-trip verify every matrix from the saved packed file and optionally materialize dense BF16.
    dense_tensors = {} if args.dense_out is not None else None
    with safe_open(args.out, framework="pt") as pf:
        for key in sorted(pf.keys()):
            if key.endswith(".packed"):
                mat_name = key[: -len(".packed")]
                width = columns[mat_name]
                fields = {"packed": pf.get_tensor(key), "scales": pf.get_tensor(f"{mat_name}.scales")}
                recon = decode_fwht_scalar(fields, width, args.bits, h64).cpu().contiguous()
                got_hash = hashlib.sha256(recon.view(torch.uint8).numpy().tobytes()).hexdigest()
                wkey = f"{mat_name}.weight"
                if got_hash != expected_hashes[wkey]:
                    raise RuntimeError(f"Round-trip mismatch for {wkey}")
                if dense_tensors is not None:
                    dense_tensors[wkey] = recon
            elif not key.endswith(".scales") and dense_tensors is not None:
                dense_tensors[key] = pf.get_tensor(key).cpu().contiguous()

    if dense_tensors is not None:
        tmp_dense = args.dense_out.with_suffix(".tmp")
        save_file(dense_tensors, str(tmp_dense))
        tmp_dense.rename(args.dense_out)
        del dense_tensors

    total_bytes = args.out.stat().st_size
    tensor_bytes = packed_bytes + scale_bytes + literal_bytes
    audit = {
        "source": str(args.source),
        "out": str(args.out),
        "out_sha256": sha256_file(args.out),
        "bits": args.bits,
        "group_size": 64,
        "quantized_matrices": quantized_count,
        "literal_tensors": literal_count,
        "total_bytes": total_bytes,
        "packed_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "literal_bytes": literal_bytes,
        "header_and_metadata_bytes": total_bytes - tensor_bytes,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    if args.dense_out is not None:
        audit["dense_out"] = str(args.dense_out)
        audit["dense_bytes"] = args.dense_out.stat().st_size
    args.audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
