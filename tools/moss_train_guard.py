"""Run guard shared by the trainers (moss_dmd_train, moss_edit_train, moss_slider_train), audit 0.9.6 F01 / F09.

- run_lock(out): an exclusive lock (O_EXCL) on the output folder, so two launches cannot both pass the check and train into it.
- seal_run(out, args, files): the run's identity (every argument that changes the result, and the identity of the models, VAE,
  prompts and datasets it reads) is written to run_identity.json on the first launch; a later launch on the same folder with a
  different identity stops before any checkpoint or cache is loaded. Operational arguments (total updates, save / preview
  intervals, gradient checkpointing) may change between launches.
- save_torch_atomic(obj, path): caches and checkpoints written to a per-process temporary file, then renamed.
- prompt_tag(prompt): a short hash of a prompt, part of the keys of per-index caches (regression pairs), so a different prompt at
  the same index never reuses an old target.
"""
import contextlib
import hashlib
import json
import os

OPERATIONAL = {"out", "updates", "steps", "save_every", "sample_every", "checkpoint", "no_checkpoint", "grad_check",
               "preview_pairs", "preview_steps", "preview_cfg"}
SMALL = 256 * 1024 * 1024  # files below this are identified by their SHA-256; larger ones (models) by name and size


def file_identity(path):
    if path is None:
        return None
    if os.path.isdir(path):
        h = hashlib.sha256()
        for root, _, names in sorted(os.walk(path)):
            for n in sorted(names):
                p = os.path.join(root, n)
                h.update(f"{os.path.relpath(p, path)}|{os.path.getsize(p)}\n".encode("utf-8"))
        return {"folder": os.path.basename(os.path.normpath(path)), "listing_sha256": h.hexdigest()}
    st = os.stat(path)
    ident = {"name": os.path.basename(path), "size": st.st_size}
    if st.st_size < SMALL:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ident["sha256"] = h.hexdigest()
    return ident


@contextlib.contextmanager
def run_lock(out):
    lock = os.path.join(out, "TRAIN_RUNNING")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another run may be writing to this folder (delete the file if it is not)") from None
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.remove(lock)


def run_identity(args, files):
    structural = {k: v for k, v in sorted(vars(args).items()) if k not in OPERATIONAL}
    return {"args": json.loads(json.dumps(structural, default=str)),
            "files": {k: file_identity(p) for k, p in sorted(files.items()) if p}}


def seal_run(out, args, files):
    """Writes or checks <out>/run_identity.json. Returns the identity. Stops with the differences when they do not match."""
    ident = run_identity(args, files)
    path = os.path.join(out, "run_identity.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diffs = []
        for part in ("args", "files"):
            for k in sorted(set(old.get(part, {})) | set(ident[part])):
                if old.get(part, {}).get(k) != ident[part].get(k):
                    diffs.append(f"{part}.{k}: {old.get(part, {}).get(k)!r} -> {ident[part].get(k)!r}")
        if diffs:
            raise SystemExit(f"{out} holds a run with another recipe; its checkpoint and caches would be mixed with this one. "
                             "Use a new --out. Differences:\n  " + "\n  ".join(diffs))
        return ident
    if any(os.path.exists(os.path.join(out, n)) for n in ("ckpt.pt", "prompts.pt", "reg_pairs.pt", "pairs.pt")):
        print(f"note: {out} has a checkpoint or caches from before run_identity.json existed; they cannot be checked and are "
              "taken as matching this launch", flush=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ident, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return ident


def save_torch_atomic(obj, path):
    import torch
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def prompt_tag(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
