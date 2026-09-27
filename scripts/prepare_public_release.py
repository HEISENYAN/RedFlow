"""Check and export the current source tree without its Git history.

Run with Python 3.11+ from a Git checkout; no training dependencies required.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Match recognizable credentials, not references to environment variable names.
SECRET_PATTERNS = (
    re.compile(rb"wandb_v1_[A-Za-z0-9_-]{20,}"),
    re.compile(rb"[\"']hf_[A-Za-z0-9]{20,}[\"']"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
)
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "checkpoints",
    "datasets",
    "dataset",
    "wandb",
    "qwenvl_runs",
    "policy_records",
}
EXCLUDED_SUFFIXES = {".log", ".out", ".err", ".pyc", ".safetensors", ".ckpt", ".hdf5", ".h5", ".parquet"}


def source_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        {ROOT / name.decode() for name in result.stdout.split(b"\0") if name and (ROOT / name.decode()).is_file()}
    )


def check(files: list[Path]) -> None:
    names = {p.relative_to(ROOT).as_posix() for p in files}
    for required in ("README.md", "LICENSE", "LICENSE_GEMMA.txt", "uv.lock", "run_scripts/jax_train_redflow.sh"):
        if required not in names:
            raise ValueError(f"Missing release file: {required}")
    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_symlink() or EXCLUDED_PARTS.intersection(relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
            raise ValueError(f"Local artifact or symlink included in source: {relative}")
        if path.name.startswith(".env.") and path.name != ".env.example":
            raise ValueError(f"Local environment file included in source: {relative}")
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Possible embedded credential in {relative}; value withheld")
        if path.suffix == ".py":
            ast.parse(data, filename=str(relative))
        elif path.suffix in {".sh", ".bash"}:
            subprocess.run(["bash", "-n", str(path)], check=True)
        elif path.suffix == ".ipynb":
            notebook = json.loads(data)
            if any(cell.get("outputs") for cell in notebook["cells"]):
                raise ValueError(f"Clear notebook outputs before release: {relative}")
        elif path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", data.decode()):
                if "://" in link or link.startswith(("#", "mailto:")):
                    continue
                target = (path.parent / link.split("#")[0]).resolve()
                try:
                    target_name = target.relative_to(ROOT).as_posix()
                except ValueError:
                    raise ValueError(f"Markdown link leaves repository: {relative}: {link}") from None
                if target_name not in names and not any(n.startswith(target_name + "/") for n in names):
                    raise ValueError(f"Broken source link: {relative}: {link}")
    print(f"Source checks passed: {len(files)} files (syntax, links, artifacts, recognizable credentials).")


def export(files: list[Path]) -> None:
    output = ROOT / "dist" / "redflow-source.zip"
    output.parent.mkdir(exist_ok=True)
    # Read gitlink revisions without copying .git or its potentially sensitive history.
    index = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT).decode()
    submodules = {}
    for entry in index.split("\0"):
        if entry.startswith("160000 "):
            metadata, name = entry.split("\t", 1)
            submodules[name] = metadata.split()[1]
    manifest = {
        "files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "optional_submodule_revisions": submodules,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "redflow/" + path.relative_to(ROOT).as_posix())
        archive.writestr("redflow/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
        for path in files:
            if archive.read("redflow/" + path.relative_to(ROOT).as_posix()) != path.read_bytes():
                raise ValueError(f"Archive differs from working tree: {path.relative_to(ROOT)}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Created and verified {output} (no Git history).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check source without creating an archive")
    args = parser.parse_args()
    sources = source_files()
    check(sources)
    if not args.check:
        export(sources)
