"""Build the Lambda ZIP inside the approved Linux/Python 3.12 job image.

python operations/completion/build.py --output /tmp/completion.zip

Copies the installed MLflow client dependency closure, using the same versions
as the model jobs. No pip install, extra requirements file or second image.
Upload the ZIP to an immutable S3 key and supply completion_code_uri in config.json.
"""

import argparse
import importlib.metadata as metadata
from pathlib import Path
import platform
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def build(output):
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
        raise RuntimeError("Build in the approved Linux x86_64 Python 3.12 job image.")
    root = Path(__file__).resolve().parents[2]
    pins = {canonicalize_name(line.split("==")[0]): line.split("==")[1]
            for line in (root / "requirements.txt").read_text().splitlines() if "==" in line}
    pending, distributions = ["boto3", "mlflow-skinny", "sagemaker-mlflow"], {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in distributions:
            continue
        dist = metadata.distribution(name)
        if name in pins and dist.version != pins[name]:
            raise ValueError(f"{name}: installed {dist.version}, required {pins[name]}.")
        distributions[name] = dist
        for value in dist.requires or []:
            dep = Requirement(value)
            if dep.marker is None or dep.marker.evaluate({"extra": ""}):
                pending.append(dep.name)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = set()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for dist in distributions.values():
            for item in dist.files or []:
                path = Path(str(item))
                if path.is_absolute() or ".." in path.parts or "__pycache__" in path.parts:
                    continue
                source = Path(dist.locate_file(item))
                if source.is_file() and str(path) not in written:
                    archive.write(source, path)
                    written.add(str(path))
        archive.write(Path(__file__).with_name("handler.py"), "handler.py")
        package = root / "mlops-models/src/mlops_models"
        for name in ("__init__.py", "storage.py", "pipelines/__init__.py",
                     "pipelines/inspect.py", "pipelines/tracking.py"):
            archive.write(package / name, "mlops_models/" + name)
        archive.writestr("completion-build.json", __import__("json").dumps(
            {name: dist.version for name, dist in sorted(distributions.items())}, indent=2))
        size = sum(item.file_size for item in archive.infolist())
        if size > 250 * 1024 * 1024:
            raise ValueError(f"Lambda package exceeds the uncompressed size limit: {size}")
    print(f"{output}: {output.stat().st_size} compressed bytes, {size} uncompressed bytes")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    build(parser.parse_args().output)
