"""Checks that two images form a fair A / B pair before they are compared: same size and the same declared sampling settings.

    python tools/moss_pair_check.py reference.png candidate.png [--no-prompt] [--strict]
    python tools/moss_pair_check.py sheet.png:A sheet.png:B        (two branches of a MOSS Compare Sheet / Compare Sampler PNG)

Each side is a PNG saved by the MOSS Image Saver (its `hades_moss_primary_stage` chunk when present, else `hades_moss`), a Suite
Telemetry json file, or a text file with one MOSS info output. A different model or LoRA is what an A / B test changes, so it is
listed, not rejected. Verdicts (audit 0.9.6 F16: "fair pair" was printed even with fields it could not check):
  fair pair: every field compared and equal (exit 0)
  fair on what was recorded: no difference, some optional fields unverified, listed (exit 0; exit 2 with --strict)
  NOT ENOUGH EVIDENCE: a field every pair needs (seed, steps, CFG, sampler, scheduler, size) could not be compared (exit 2)
  MISMATCH: a setting differs (exit 1)
It reads metadata only: it cannot tell whether the metadata belongs to the pixels, nor judge the images.
"""
import argparse
import json
import pathlib
import sys

from PIL import Image

# denoise_mode / start_sigma / schedule_steps / steps_run: the schedule actually run (the same denoise runs differently in core and exact)
SAMPLING = ("seed", "steps", "cfg", "sampler", "scheduler", "flow_shift", "denoise", "width", "height", "slg",
            "denoise_mode", "start_sigma", "schedule_steps", "steps_run")
PROMPT = ("prompt", "negative")
ALLOWED = (("loader", "model_file"), ("multi_lora", "loras"), ("ksampler", "turbo"))
PREFIX = "HADES_META "
REQUIRED = ("seed", "steps", "cfg", "sampler", "scheduler", "width", "height")


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
    """(metas, (width, height) or None) of a PNG or a text / json file; 'sheet.png:B' reads branch B of a Compare Sheet PNG."""
    branch = None
    text_path = str(path)
    if ":" in text_path[2:] and text_path.rsplit(":", 1)[1] in ("A", "B", "C", "D"):
        text_path, branch = text_path.rsplit(":", 1)
    path = pathlib.Path(text_path)
    if branch is not None:
        with Image.open(path) as img:
            chunk = dict(img.text).get("hades_moss_compare")
        if chunk is None:
            raise ValueError(f"{path.name}: no hades_moss_compare chunk (not a Compare Sheet PNG, a blind sheet, or saved without metadata)")
        metas = (json.loads(chunk).get("branches") or {}).get(branch)
        if not isinstance(metas, list):
            raise ValueError(f"{path.name}: no branch {branch} in the sheet")
        return [m for m in metas if isinstance(m, dict) and "node" in m], None  # the sheet's size is not the panel's
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
        elif field in ("denoise_mode", "start_sigma") and (field not in a or field not in b):
            out["unverified"].append(f"{field} (one side was saved before the Suite recorded it)")
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
    p.add_argument("--strict", action="store_true", help="exit 2 when any field could not be compared, not only the required ones")
    args = p.parse_args(argv)
    result = check_pair(load_side(args.reference), load_side(args.candidate), compare_prompt=not args.no_prompt)
    text, code = verdict(result, args.strict)
    print(text + f" | matched: {', '.join(result['matched']) or 'none'}")
    for key, label in (("mismatches", "differs"), ("differences", "compared"), ("unverified", "unverified")):
        for item in result[key]:
            print(f"  {label}: {item}")
    return code


def verdict(result, strict=False):
    """(text, exit code) of a check_pair result: equality of the fields available is told apart from enough evidence."""
    if result["mismatches"]:
        return "MISMATCH", 1
    missing = [f for f in REQUIRED if f not in result["matched"]]
    if missing:
        return f"NOT ENOUGH EVIDENCE (not compared: {', '.join(missing)})", 2
    if result["unverified"]:
        return f"fair on what was recorded ({len(result['unverified'])} field(s) unverified)", 2 if strict else 0
    return "fair pair", 0


if __name__ == "__main__":
    sys.exit(main())
