"""Checks the Comfy Registry package of this pack, as a user gets it: the zip `comfy node pack` builds, and a clean ComfyUI that
loaded it from that zip (no git checkout, no tests, no Dev Tools).

    python tests/check_package.py contents node.zip
    python tests/check_package.py server  [--server http://127.0.0.1:8188] [--log comfyui.log]

`contents`: the files that must ship (every node module, the frontend, the example workflows, README / CHANGELOG / LICENSE /
pyproject and tools/RESULTS.md, which the README and tooltips cite) and the ones that must not (tests, CI, docs, the other
tools, the Dev Tools and their examples).
`server`: the running ComfyUI registered exactly the pack's distributed nodes, serves its frontend files, did not log an import
failure for it, and every example workflow in the zip only uses nodes it knows, with the widget count each node declares.
Exit code 0: all checks passed; 1: a check failed (printed).
"""
import argparse
import ast
import json
import pathlib
import sys
import time
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_examples  # noqa: E402  (input_specs / is_widget: the same widget rules the examples were built with)

DEV_FILES = {"moss_dev_nodes.py", "web/moss_dev_tools.js"}
FORBIDDEN_DIRS = ("tests/", ".github/", "docs/", "examples/dev/")
REQUIRED = {"__init__.py", "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", "tools/RESULTS.md"}
# frontend-only node types the examples may use besides the server's nodes
FRONTEND_NODES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}


def distributed_nodes():
    """Node ids of the pack's modules except the Dev Tools, read from their NODE_CLASS_MAPPINGS without importing them."""
    ids = set()
    for path in sorted(ROOT.glob("moss_*.py")):
        if path.name in DEV_FILES:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NODE_CLASS_MAPPINGS" for t in node.targets):
                ids |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return ids


def zip_names(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n.replace("\\", "/") for n in z.namelist() if not n.endswith("/")]
    # comfy-cli zips the pack's files at the root; tolerate one top folder
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and not any(n == next(iter(tops)) for n in names) and "__init__.py" not in names:
        prefix = next(iter(tops)) + "/"
        names = [n[len(prefix):] for n in names]
    return set(names)


def check_contents(zip_path):
    names = zip_names(zip_path)
    errors = []
    expected_py = {p.name for p in ROOT.glob("moss_*.py")} - DEV_FILES
    expected_web = {f"web/{p.name}" for p in (ROOT / "web").glob("*") if p.is_file()} - DEV_FILES
    expected_examples = {f"examples/{p.name}" for p in (ROOT / "examples").glob("*.json")}
    for need in sorted(REQUIRED | expected_py | expected_web | expected_examples):
        if need not in names:
            errors.append(f"missing from the package: {need}")
    for name in sorted(names):
        if name in DEV_FILES or name.startswith(FORBIDDEN_DIRS) or (name.startswith("tools/") and name != "tools/RESULTS.md"):
            errors.append(f"must not ship: {name}")
    errors += broken_links(names)
    print(f"package: {len(names)} files, {len(expected_py)} node modules, {len(expected_web)} frontend files, {len(expected_examples)} examples")
    return errors


def broken_links(names, docs=("README.md", "CHANGELOG.md")):
    """Relative links in the shipped docs whose target is not in the package (audit 0.9.6 F13): a link to something the
    Registry package leaves out must be a full URL to the repository."""
    import re
    dirs = {n.rsplit("/", 1)[0] for n in names if "/" in n}
    errors = []
    for doc in docs:
        text = (ROOT / doc).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = target.split("#", 1)[0].rstrip("/")
            if path and path not in names and path not in dirs:
                errors.append(f"{doc}: relative link to {target}, which the package does not contain")
    return errors


def fetch(server, path, raw=False):
    with urllib.request.urlopen(server + path, timeout=60) as r:
        data = r.read()
    return data if raw else json.loads(data)


def wait_for(server, seconds):
    end = time.time() + seconds
    while time.time() < end:
        try:
            return fetch(server, "/object_info")
        except OSError:
            time.sleep(2)
    raise SystemExit(f"ComfyUI did not answer at {server} within {seconds} s")


def check_server(server, log, pack_dir):
    errors = []
    info = wait_for(server, 300)
    want = distributed_nodes()
    got = {k for k in info if k.startswith("HaDeSMoss")}
    errors += [f"node not registered: {k}" for k in sorted(want - got)]
    errors += [f"unexpected node (Dev Tools in the package?): {k}" for k in sorted(got - want)]
    # the frontend files the server announces for this pack, and that each one is served
    exts = fetch(server, "/extensions")
    ours = [e for e in exts if f"/{pack_dir}/" in e]
    if not ours:
        errors.append(f"no frontend file served for custom_nodes/{pack_dir}")
    for e in ours:
        try:
            fetch(server, e, raw=True)
        except OSError as exc:
            errors.append(f"frontend file not served: {e} ({exc})")
    if any(e.endswith("moss_dev_tools.js") for e in ours):
        errors.append("the Dev Tools frontend is served by the package")
    if log:
        text = pathlib.Path(log).read_text(encoding="utf-8", errors="replace")
        # ComfyUI's own import summary ("(IMPORT FAILED): path") and its "Cannot import ... module for custom nodes" line
        errors += [f"ComfyUI log: {line.strip()}" for line in text.splitlines()
                   if pack_dir in line and ("IMPORT FAILED" in line or "Cannot import" in line)]
    # every example the package ships uses only known nodes, with the widget count each declares
    examples = sorted((ROOT / "examples").glob("*.json"))
    for path in examples:
        wf = json.loads(path.read_text(encoding="utf-8"))
        for node in wf["nodes"]:
            cls = node["type"]
            if cls in FRONTEND_NODES:
                continue
            if cls not in info:
                errors.append(f"{path.name}: node type {cls} is unknown to the server")
                continue
            n_widgets = 0
            for _, kind, options, _ in make_examples.input_specs(info[cls]):
                if make_examples.is_widget(kind, options):
                    n_widgets += 1 + bool(options.get("control_after_generate"))
            if len(node.get("widgets_values", [])) != n_widgets:
                errors.append(f"{path.name}: {cls} (node {node['id']}) has {len(node.get('widgets_values', []))} widget values, the server declares {n_widgets}")
    print(f"server: {len(got)} pack nodes, {len(ours)} frontend files, {len(examples)} examples checked")
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["contents", "server"])
    p.add_argument("zip", nargs="?", help="contents: the zip built by comfy node pack")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--log", default="", help="server: ComfyUI's console log, scanned for an import failure of the pack")
    p.add_argument("--pack-dir", default="HaDeS-MOSS-Suite", help="server: the pack's folder name under custom_nodes")
    a = p.parse_args()
    errors = check_contents(a.zip) if a.mode == "contents" else check_server(a.server, a.log, a.pack_dir)
    for e in errors:
        print("FAIL", e)
    print("OK" if not errors else f"{len(errors)} check(s) failed")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
