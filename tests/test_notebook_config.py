"""Exercise the shared notebook file handoff without notebook edits or AWS writes."""

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def code_cells(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


PREPARE = code_cells("01_prepare_data.ipynb")
DEPLOY = code_cells("02_deploy_pipelines.ipynb")


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class NotebookConfiguration(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "renamed-foundation"
        (self.root / "mlops-models/src").mkdir(parents=True)
        (self.root / "notebooks").mkdir()
        self.path = self.root / "notebooks/config.json"
        self.template = self.root / "notebooks/config.example.json"
        shutil.copyfile(ROOT / "notebooks/config.example.json", self.template)

    def save(self, **values):
        self.path.write_text(json.dumps(values))

    def load(self, cells):
        namespace = {}
        with working_directory(self.root / "notebooks"):
            exec(cells[0], namespace)
            exec(cells[1], namespace)
        return namespace

    def test_missing_file_creates_placeholders_and_stops_without_overwriting(self):
        for cells in (PREPARE, DEPLOY):
            with self.subTest(notebook=cells):
                self.path.unlink(missing_ok=True)
                with self.assertRaisesRegex(ValueError, "Fill in your inputs"):
                    self.load(cells)
                self.assertEqual(self.path.read_bytes(), self.template.read_bytes())
                settings = json.loads(self.path.read_text())
                self.assertEqual(settings, dict.fromkeys((
                    "metadata_csv", "bucket", "role_arn", "image_uri",
                    "csv_input", "feature_store_config_uri", "registry_group_prefix",
                ), ""))

    def test_saved_inputs_are_used_but_optional_settings_come_from_notebooks(self):
        self.save(metadata_csv="table.csv", bucket="test-bucket", role_arn="pipeline-role",
                  feature_store_role_arn="writer-role", image_uri="existing-image",
                  csv_input="sample.csv", namespace="review", registry_group_prefix="long-team-prefix-",
                  sample_rows=7, cache_steps=False, custom_setting={"keep": "this"})
        before = self.path.read_bytes()
        first, second = self.load(PREPARE), self.load(DEPLOY)
        self.assertEqual(first["metadata_csv"], "table.csv")
        self.assertEqual(first["feature_store_role_arn"], "pipeline-role")
        self.assertEqual(second["role_arn"], "pipeline-role")
        self.assertEqual(second["job_role_arn"], "pipeline-role")
        self.assertEqual(second["registry_group_prefix"], "long-team-prefix-")
        self.assertEqual(first["namespace"], second["namespace"])
        self.assertEqual(first["namespace"], "wsf-dev")
        self.assertEqual(first["data_source"], second["data_source"])
        self.assertEqual(first["sample_rows"], 100)
        self.assertTrue(second["cache_steps"])
        self.assertFalse(second["include_hpo"])
        self.assertEqual(second["csv_input"], "sample.csv")
        self.assertEqual(self.path.read_bytes(), before)

    def test_csv_handoff_is_read_by_next_notebook_and_preserves_other_settings(self):
        self.save(bucket="test-bucket", role_arn="pipeline-role",
                  feature_store_config_uri="s3://old/binding.json", extra={"preserve": True})
        first = self.load(PREPARE)
        sample = self.root / ".runs/feature-store/review/sample.csv"
        sample.parent.mkdir(parents=True)
        sample.write_text("id,value\n0001,2\n")
        first["sample"] = {"sample": sample}
        # An unrelated edit made while preparation runs must survive the handoff.
        settings = json.loads(self.path.read_text())
        settings["future_option"] = "new value"
        self.path.write_text(json.dumps(settings))
        template_before = self.template.read_bytes()
        exec(PREPARE[6], first)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["csv_input"], ".runs/feature-store/review/sample.csv")
        self.assertEqual(saved["feature_store_config_uri"], "")
        self.assertEqual(saved["extra"], {"preserve": True})
        self.assertEqual(saved["future_option"], "new value")
        second = self.load(DEPLOY)
        self.assertEqual((second["ROOT"] / second["csv_input"]).read_bytes(), sample.read_bytes())
        self.assertEqual(second["role_arn"], "pipeline-role")
        self.assertEqual(self.template.read_bytes(), template_before)

    def test_feature_store_handoff_saves_uri_and_keeps_pipeline_role(self):
        self.save(bucket="test-bucket", role_arn="pipeline-role", csv_input="old-sample.csv")
        first = self.load(PREPARE)
        sample = self.root / "sample.csv"
        sample.write_text("id,value\n0001,2\n")
        first.update(sample={"sample": sample}, data_source="feature_store", feature_store_role_arn="writer-role",
                     settings={"prefix": "review/features/"}, receipt={"delivery_id": "ready"},
                     session=Mock(), uuid4=uuid4)
        with patch("mlops_features.data.write_once") as write:
            exec(PREPARE[6], first)
            write.assert_called_once()
        second = self.load(DEPLOY)
        self.assertEqual(second["csv_input"], "sample.csv")
        self.assertEqual(second["inference_data_source"], "csv_direct")
        self.assertTrue(second["feature_store_config_uri"].startswith(
            "s3://test-bucket/review/features/configurations/"))
        self.assertEqual(second["role_arn"], "pipeline-role")
        self.assertNotIn("feature_store_role_arn", json.loads(self.path.read_text()))

    def test_failed_feature_store_publication_does_not_replace_saved_inputs(self):
        self.save(bucket="test-bucket",
                  feature_store_config_uri="s3://previous/binding.json", extra="keep")
        first = self.load(PREPARE)
        first.update(data_source="feature_store", settings={"prefix": "review/features/"},
                     receipt={"delivery_id": "ready"},
                     session=Mock(), uuid4=uuid4)
        before = self.path.read_bytes()
        with patch("mlops_features.data.write_once", side_effect=RuntimeError("denied")):
            with self.assertRaisesRegex(RuntimeError, "denied"):
                exec(PREPARE[6], first)
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_json_is_not_replaced_with_defaults(self):
        self.path.write_text('{"bucket":')
        before = self.path.read_bytes()
        for cells in (PREPARE, DEPLOY):
            with self.assertRaises(json.JSONDecodeError):
                self.load(cells)
            self.assertEqual(self.path.read_bytes(), before)

    def test_export_includes_template_and_excludes_populated_local_config(self):
        self.save(bucket="private-local-value", csv_input="/private/sample.csv")
        spec = importlib.util.spec_from_file_location("notebook_export", ROOT / "release/export.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with patch.object(exporter, "ROOT", self.root):
            files = exporter.source_files()
        self.assertIn(self.template, files)
        self.assertNotIn(self.path, files)


if __name__ == "__main__":
    unittest.main()
