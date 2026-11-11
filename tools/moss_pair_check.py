"""Checks that two images form a fair A / B pair before they are compared: same size and the same declared sampling settings.

    python tools/moss_pair_check.py reference.png candidate.png [--no-prompt]

Each side is a PNG saved by the MOSS Image Saver (its `hades_moss_primary_stage` chunk when present, else `hades_moss`), a Suite
Telemetry json file, or a text file with one MOSS info output. A different model or LoRA is what an A / B test changes, so it is
listed, not rejected. Exit code 0: a fair pair; 1: a setting differs (printed). It reads metadata only: it cannot tell whether the
metadata belongs to the pixels, nor judge the images.
"""
import argparse
import json
import pathlib
import sys

from PIL import Image

SAMPLING = ("seed", "steps", "cfg", "sampler", "scheduler", "flow_shift", "denoise", "width", "height", "slg")
PROMPT = ("prompt", "negative")
ALLOWED = (("loader", "model_file"), ("multi_lora", "loras"), ("ksampler", "turbo"))
PREFIX = "HADES_META "


def metas_from_text(text):
    """The Suite entries of a Telemetry json or of one info output."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return [m for m in data["nodes"] if isinstance(m, dict) and "node" in m]
    if isinstance(data, list):  # the Image Saver's PNG chunks hold the list itself
        return [m for m in data if isinstance(m, dict) and "node" in m]
    for line in reversed(str(text).splitlines()):
        if line.startswith(PREFIX):
            meta = json.loads(line[len(PREFIX):])
            if isinstance(meta, dict) and "node" in meta:
                return [meta]
    raise ValueError("no Suite metadata (Telemetry json, an info output, or a MOSS Image Saver PNG)")


def load_side(path):
    """(metas, (width, height) or None) of a PNG or a text / json file."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".png":
        with Image.open(path) as img:
            chunks, size = dict(img.text), img.size
        text = chunks.get("hades_moss_primary_stage") or chunks.get("hades_moss")
        if text is None:
            raise ValueError(f"{path.name}: no hades_moss chunk (not saved by the MOSS Image Saver, or saved with --disable-metadata)")
        return metas_from_text(text), size
    return metas_from_text(path.read_text(encoding="utf-8")), None


def check_pair(reference, candidate, compare_prompt=True):
    """{"mismatches", "matched", "unverified", "differences"} for two (metas, size) sides; a fair pair has no mismatches."""
    out = {"mismatches": [], "matched": [], "unverified": [], "differences": []}
    sides = []
    for name, (metas, size) in (("reference", reference), ("candidate", candidate)):
        ks = [m for m in metas if m["node"] == "ksampler"]
        if len(ks) != 1:
            raise ValueError(f"{name}: {len(ks)} KSampler entries; a stage needs exactly one (use a Telemetry of that stage only)")
        enh = [m for m in metas if m["node"] == "prompt_enhancer"]
        if size is not None and (ks[0].get("width"), ks[0].get("height")) != size:
            out["mismatches"].append(f"{name}: image {size[0]}x{size[1]}, its KSampler {ks[0].get('width')}x{ks[0].get('height')}")
        sides.append((metas, size, ks[0], enh[0] if len(enh) == 1 else None))
    (a_metas, a_size, a_ks, a_enh), (b_metas, b_size, b_ks, b_enh) = sides
    if a_size and b_size and a_size != b_size:
        out["mismatches"].append(f"size: {a_size[0]}x{a_size[1]} against {b_size[0]}x{b_size[1]}")
    fields = [(f, a_ks, b_ks) for f in SAMPLING]
    if compare_prompt:
        if a_enh is None or b_enh is None:
            out["unverified"].append("prompt (each side needs one Prompt Enhancer entry)")
        else:
            fields += [(f, a_enh, b_enh) for f in PROMPT]
    for field, a, b in fields:
        if field not in a and field not in b:
            out["unverified"].append(field)
        elif a.get(field) != b.get(field):
            out["mismatches"].append(f"{field}: {a.get(field)!r} against {b.get(field)!r}" if field not in PROMPT else f"{field} differs")
        else:
            out["matched"].append(field)
    for node, field in ALLOWED:
        a = [m.get(field) for m in a_metas if m["node"] == node]
        b = [m.get(field) for m in b_metas if m["node"] == node]
        if a != b:
            out["differences"].append(f"{node} {field}: {a!r} against {b!r}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--no-prompt", action="store_true", help="do not compare the prompt texts (e.g. a prompt A / B test)")
    args = p.parse_args(argv)
    result = check_pair(load_side(args.reference), load_side(args.candidate), compare_prompt=not args.no_prompt)
    print(("MISMATCH" if result["mismatches"] else "fair pair") + f" | matched: {', '.join(result['matched']) or 'none'}")
    for key, label in (("mismatches", "differs"), ("differences", "compared"), ("unverified", "unverified")):
        for item in result[key]:
            print(f"  {label}: {item}")
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
