"""Exercise the real source loader and dispatcher in separate Python processes."""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest

from mlops_models.pipelines import source
from mlops_skeleton.pipelines import walk_steps
import test_source_deployment as deployment_tests


class InferenceCodeChoice(unittest.TestCase):
    def test_execution_choice_reaches_all_development_processing_stages(self):
        fixture = deployment_tests.SourceDeployment()
        fixture.setUp()
        for runtime in ("processing", "both"):
            bodies = fixture.definitions(runtime, model_code_source="artifact",
                                         monitoring={"enabled": True})
            body = bodies["inference"]
            parameters = {p["Name"]: p for p in body["Parameters"]}
            self.assertEqual(parameters["ModelCodeSource"], {
                "Name": "ModelCodeSource", "Type": "String", "DefaultValue": "artifact",
                "EnumValues": ["artifact", "pipeline"]})
            self.assertNotIn("ModelCodeSource", {p["Name"] for p in bodies["training"]["Parameters"]})
            for step in walk_steps(body["Steps"]):
                if step["Type"] != "Processing" or step["Name"] == "ResolveModel":
                    continue
                args = step["Arguments"]
                expected = ("artifact" if step["Name"].endswith("Transform")
                            else {"Get": "Parameters.ModelCodeSource"})
                self.assertEqual(args["Environment"]["MLOPS_MODEL_CODE_SOURCE"], expected)
                self.assertEqual(args["Environment"]["MLOPS_MODEL_METADATA_SHA256"]["Std:JsonGet"]["Path"],
                                 "model_metadata_sha256")
                self.assertEqual(args["Environment"]["MLOPS_MODEL_ARTIFACT_URI"]["Std:JsonGet"]["Path"],
                                 "checkpoint")

    def test_ci_does_not_expose_the_development_override(self):
        fixture = deployment_tests.SourceDeployment()
        fixture.setUp()
        fixture.identity["mode"] = "ci-deploy"
        for body in fixture.definitions("processing").values():
            self.assertNotIn("ModelCodeSource", {p["Name"] for p in body["Parameters"]})
        for step in walk_steps(fixture.definitions("processing")["inference"]["Steps"]):
            if step["Type"] == "Processing" and step["Name"] != "ResolveModel":
                self.assertEqual(step["Arguments"]["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "artifact")

    def test_real_loader_switches_imports_and_checks_metadata_before_model_code(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "source"
            metadata = json.dumps({"feature_mapping": {f"f{i}": f"f{i}" for i in range(39)}}).encode()
            archived = {}
            for label in ("current", "archived"):
                files = {}
                for folder in ("", "entrypoints", "runtime", "models",
                               "models/usecase1", "models/usecase1/inference"):
                    files[f"mlops_models/{folder + '/' if folder else ''}__init__.py"] = b""
                for name in ("entrypoints/__main__.py", "entrypoints/predict.py", "runtime/job_paths.py"):
                    files["mlops_models/" + name] = (source.PACKAGE / name).read_bytes()
                files["mlops_models/models/usecase1/parameters.json"] = b'{"primary_key":"ID"}'
                files["mlops_models/models/usecase1/model_metadata.json"] = b'{"feature_mapping":{"local":"local"}}'
                files["source.json"] = json.dumps({"commit": label}).encode()
                files["mlops_models/models/usecase1/inference/predict.py"] = (
                    "import json\n"
                    "def predict_with_model(input_path, model_dir, output, config, parameter_path):\n"
                    f"    print(json.dumps({{'probe': '{label}', 'module': __file__, "
                    "'features': len(config['feature_mapping'])}), flush=True)\n"
                ).encode()
                if label == "current":
                    for name, data in files.items():
                        path = current / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                else:
                    archived = {"code/" + name: data for name, data in files.items()}
            archived.update({"model_metadata.json": metadata, "parameters.json": b'{"primary_key":"ID"}'})
            archive_path = root / "model.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, data in archived.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            shutil.copyfile(source.PACKAGE / "runtime/load_source.py", current / "load_source.py")
            entrypoint = current / "mlops_models/entrypoints/predict.py"
            original = entrypoint.read_bytes()
            base_environment = {key: value for key, value in os.environ.items()
                                if not key.startswith(("MLOPS_", "MODEL_", "PIPELINE_", "SOURCE_"))}
            base_environment.update(
                MLOPS_MODEL_PACKAGE="package/17", MLOPS_MODEL_ARTIFACT_URI="s3://bucket/model.tar.gz",
                MLOPS_MODEL_METADATA_URI="s3://bucket/reports/model_metadata.json",
                MLOPS_PIPELINE_ARN="pipeline", PIPELINE_EXECUTION_ID="execution")
            for mode, broken, correct_metadata in (
                    ("artifact", True, True), ("pipeline", True, True),
                    ("pipeline", False, True), ("pipeline", True, False)):
                with self.subTest(mode=mode, broken=broken, correct_metadata=correct_metadata):
                    entrypoint.write_bytes(b'raise RuntimeError("BROKEN_CURRENT_PREDICT")\n' if broken else original)
                    env = {**base_environment, "MLOPS_MODEL_CODE_SOURCE": mode,
                           "MLOPS_MODEL_METADATA_SHA256": hashlib.sha256(metadata).hexdigest()
                           if correct_metadata else "0" * 64}
                    result = subprocess.run([
                        sys.executable, "-I", str(current / "load_source.py"), "score", "--model", "usecase1",
                        "--model-dir", str(archive_path), "--input", str(root / "prepared"),
                        "--output", str(root / "output"),
                    ], env=env, capture_output=True, text=True)
                    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                    loaded = next(record for record in records if record.get("event") == "inference_metadata_loaded")
                    self.assertEqual(loaded["feature_count"], 39)
                    self.assertEqual(loaded["verified"], correct_metadata)
                    if not correct_metadata:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("does not match the selected Registry/override metadata", result.stderr)
                        self.assertNotIn("BROKEN_CURRENT_PREDICT", result.stderr)
                        self.assertFalse(any(record.get("probe") for record in records))
                    elif broken and mode == "pipeline":
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("BROKEN_CURRENT_PREDICT", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        probe = next(record for record in records if record.get("probe"))
                        self.assertEqual(probe["probe"], "archived" if mode == "artifact" else "current")
                        self.assertEqual(probe["features"], 39)
                        selected = next(record for record in records if record.get("event") == "source_loaded")
                        self.assertEqual(selected["pipeline_execution_id"], "execution")
                        self.assertEqual(selected["model_package"], "package/17")
                        self.assertEqual(selected["model_metadata"]["sha256"], hashlib.sha256(metadata).hexdigest())


if __name__ == "__main__":
    unittest.main()
