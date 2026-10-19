"""Export portable working source and clear notebook execution output."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("config", "notebooks", "examples", "mlops-models", "mlops-features", "release", "tests")
# Explicit handoff guides are public source; other planning documents stay excluded.
FILES = ("README.md", "docs/implementation/README.md", "docs/integration/README.md",
         "pyproject.toml", "requirements-infrastructure.txt", "requirements-infrastructure.lock.txt",
         "requirements-tracking.txt", ".gitignore", ".gitlab-ci.yml")
SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".ipynb", ".md", ".Dockerfile"}


def source_files():
    """Return distributable source paths, excluding generated files and non-README Markdown."""
    paths = [ROOT / name for name in FILES if (ROOT / name).is_file()]
    for folder in FOLDERS:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(ROOT)
            if relative.as_posix() == "notebooks/config.json":
                continue  # Local connection values and generated data paths are not source.
            if any(part.startswith(".") or part.endswith((".egg-info", ".dist-info"))
                   or part in {"node_modules", "__pycache__", "cdk.out", "build", "dist"}
                   for part in relative.parts):
                continue
            if path.suffix not in SUFFIXES:
                continue
            if path.suffix.lower() == ".md" and path.name.lower() != "readme.md":
                continue
            if path.name in {"definition.rendered.json", "definition.deployed.json", "deployment-config.json"}:
                continue
            paths.append(path)
    return sorted(set(paths))


def clean_notebook(content):
    """Return a notebook with execution results and transient metadata removed."""
    notebook = json.loads(content)
    notebook["metadata"] = {key: value for key, value in notebook.get("metadata", {}).items()
                            if key in {"kernelspec", "language_info"}}
    for cell in notebook["cells"]:
        cell["metadata"] = {}
        cell.pop("attachments", None)
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode()


def export_source(destination):
    """Write reviewed source files to a ZIP and return its checksum and file inventory."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            content = path.read_bytes()
            if path.suffix == ".ipynb":
                content = clean_notebook(content)
            name = "enterprise-mlops-ci-on-aws/" + path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if path.stat().st_mode & 0o111 else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
            inventory.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP integrity verification failed.")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n")
    return {"sha256": digest, "files": inventory}


def main():
    """Read an output path, export the current working source and print its checksum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_source(args.output)
    print(json.dumps({"file": str(args.output), "sha256": result["sha256"],
                      "source_files": len(result["files"])}, indent=2))


if __name__ == "__main__":
    main()
