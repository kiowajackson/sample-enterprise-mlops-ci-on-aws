"""Exercise the shared notebook file handoff without notebook edits or AWS writes."""

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
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
        self.path.write_text(json.dumps({"user_alias": "notebook-user", "primary_key": "customer_id", **values}))

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
                self.assertEqual(settings.pop("s3_prefix"), "mlops")
                self.assertEqual(settings, dict.fromkeys((
                    "primary_key", "metadata_csv", "bucket", "role_arn", "image_uri",
                    "csv_input", "feature_store_config_uri", "registry_group_prefix", "tracking_uri", "namespace", "completion_code_uri", "user_alias",
                ), ""))

    def test_saved_inputs_are_used_but_optional_settings_come_from_notebooks(self):
        self.save(metadata_csv="table.csv", primary_key="CLIENT_ID", bucket="test-bucket", role_arn="pipeline-role",
                  feature_store_role_arn="writer-role", image_uri="existing-image",
                  csv_input="sample.csv", namespace="review", registry_group_prefix="long-team-prefix-",
                  sample_rows=7, cache_steps=False, batch_runtime="processing",
                  model_code_source="artifact", custom_setting={"keep": "this"},
                  tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/test")
        before = self.path.read_bytes()
        first, second = self.load(PREPARE), self.load(DEPLOY)
        self.assertEqual(first["metadata_csv"], "table.csv")
        self.assertEqual(first["feature_store_role_arn"], "pipeline-role")
        self.assertEqual(second["role_arn"], "pipeline-role")
        self.assertEqual(second["job_role_arn"], "pipeline-role")
        self.assertEqual(second["registry_group_prefix"], "long-team-prefix-")
        self.assertEqual(second["namespace"], "review")
        self.assertEqual(first["namespace"], "review")
        self.assertEqual(first["record_id"], "CLIENT_ID")
        self.assertEqual(first["data_source"], second["data_source"])
        self.assertEqual(first["sample_rows"], 100)
        self.assertTrue(second["cache_steps"])
        self.assertFalse(second["include_hpo"])
        self.assertEqual(second["batch_runtime"], "processing")
        self.assertEqual(second["model_code_source"], "artifact")
        self.assertEqual(second["csv_input"], "sample.csv")
        self.assertEqual(second["tracking_uri"],
                         "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/test")
        self.assertEqual(self.path.read_bytes(), before)

    def test_storage_folder_default_and_override_are_shared_by_both_notebooks(self):
        for options, expected in (({}, "mlops/review"),
                                  ({"s3_prefix": "/team/project/"}, "team/project/review")):
            for cells in (PREPARE, DEPLOY):
                with self.subTest(options=options, notebook=cells[0][:30]):
                    self.save(namespace="review", **options)
                    scope = self.load(cells)
                    self.assertEqual(scope["storage_prefix"], expected)
                    self.assertEqual(scope["namespace"], "review")

    def test_invalid_storage_folders_stop_before_publication(self):
        for value in ("", "/", ".", "../team", "team//project", "team/./project",
                      "s3://bucket/team", "team?query"):
            for cells in (PREPARE, DEPLOY):
                with self.subTest(value=value, notebook=cells[0][:30]):
                    self.save(namespace="review", s3_prefix=value)
                    with self.assertRaisesRegex(ValueError, "s3_prefix"):
                        self.load(cells)


    def test_preparation_requires_an_explicit_primary_key_without_rewriting_config(self):
        for value in ("", "   "):
            self.save(primary_key=value, target_key="CLIENT_ID")
            before = self.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Set primary_key"):
                self.load(PREPARE)
            self.assertEqual(self.path.read_bytes(), before)

    def test_generated_csv_uses_configured_id_and_shared_namespace_in_both_modes(self):
        shutil.copytree(ROOT / "mlops-features/mlops_features",
                        self.root / "mlops-features/mlops_features",
                        ignore=shutil.ignore_patterns("__pycache__"))
        metadata = self.root / "metadata.csv"
        metadata.write_text(
            "column_name,column_map,table_nm,ref_val1,ref_val2,ref_val3,ref_val4,ref_val5\n"
            "value,value,example,1,2,3,4,5\n"
            "outcome,outcome,example,10,20,30,40,50\n")
        parameters = self.root / "mlops-models/src/mlops_models/models/usecase1/parameters.json"
        parameters.parent.mkdir(parents=True)
        parameters.write_text(json.dumps({"primary_key": "CLIENT_ID", "timestamp": "feature_asof_dt",
                                          "target_key": "outcome", "task_type": "regression"}))
        previous_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), previous_path))
        with patch.dict(sys.modules):
            for name in list(sys.modules):
                if name == "mlops_features" or name.startswith("mlops_features."):
                    del sys.modules[name]
            for demo in (False, True):
                with self.subTest(demo=demo):
                    self.save(metadata_csv=str(metadata), primary_key="CLIENT_ID", extra="preserve")
                    scope = self.load(PREPARE)
                    scope.update(display=lambda value: None, sample_rows=3,
                                 demo_10000_features=demo, demo_model_id="usecase1")
                    exec(PREPARE[2], scope)
                    frame = scope["df"]
                    self.assertEqual(frame["CLIENT_ID"].tolist(),
                                     ["sample-000001", "sample-000002", "sample-000003"])
                    self.assertNotIn("customer_id", frame.columns)
                    self.assertIn("outcome", frame.columns)
                    if demo:
                        self.assertEqual(len(frame.columns), 10003)
                    exec(PREPARE[6], scope)
                    saved = json.loads(self.path.read_text())
                    self.assertEqual(saved["primary_key"], "CLIENT_ID")
                    self.assertEqual(saved["extra"], "preserve")
                    self.assertEqual((self.root / saved["csv_input"]).resolve(),
                                     scope["sample"]["sample"].resolve())
                    deployed = self.load(DEPLOY)
                    self.assertEqual(deployed["namespace"], scope["namespace"])
                    self.assertEqual(deployed["csv_input"], saved["csv_input"])
                    self.assertEqual(self.load(PREPARE)["namespace"], scope["namespace"])
                    if demo:
                        scope["record_id"] = "OTHER_ID"
                        with self.assertRaisesRegex(AssertionError, "Align the ID names"):
                            exec(PREPARE[2], scope)

    def test_user_deployment_namespace_is_stable_after_restart_and_config_recreation(self):
        self.save(bucket="test-bucket", role_arn="pipeline-role", extra={"keep": True})
        first = self.load(DEPLOY)
        second = self.load(DEPLOY)
        self.assertRegex(first["namespace"], r"^wsf-noteboo-[0-9a-f]{8}$")
        self.assertLessEqual(len(first["namespace"]), 20)
        self.assertEqual(first["namespace"], second["namespace"])
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["namespace"], first["namespace"])
        self.assertEqual(saved["extra"], {"keep": True})
        self.save(bucket="test-bucket", role_arn="pipeline-role")
        self.assertEqual(self.load(DEPLOY)["namespace"], first["namespace"])

    def test_studio_profile_identity_survives_app_restart_and_distinguishes_users_and_domains(self):
        metadata_path = Path("/opt/ml/metadata/resource-metadata.json")
        original_read, original_is_file = Path.read_text, Path.is_file
        names = []
        for user, domain, app in (("analyst.one", "d-a", "first-app"),
                                  ("analyst.one", "d-a", "restarted-app"),
                                  ("analyst.two", "d-a", "first-app"),
                                  ("analyst.one", "d-b", "first-app")):
            metadata = {"UserProfileName": user, "DomainId": domain, "ResourceName": app}
            self.save(bucket="test-bucket", role_arn="pipeline-role", user_alias="")
            with patch.object(Path, "is_file", lambda path: path == metadata_path or original_is_file(path)), \
                    patch.object(Path, "read_text", lambda path, *a, **kw: json.dumps(metadata)
                                 if path == metadata_path else original_read(path, *a, **kw)):
                names.append(self.load(DEPLOY)["namespace"])
        self.assertEqual(names[0], names[1])
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(len(name) <= 20 for name in names))

    def test_missing_user_identity_stops_without_using_shared_login_or_writing_config(self):
        self.save(bucket="test-bucket", role_arn="pipeline-role", user_alias="")
        before = self.path.read_bytes()
        original_is_file = Path.is_file
        with patch.object(Path, "is_file", lambda path: False
                          if path == Path("/opt/ml/metadata/resource-metadata.json") else original_is_file(path)), \
                patch.dict(os.environ, {"USER": "sagemaker-user", "USERNAME": "sagemaker-user"}):
            with self.assertRaisesRegex(ValueError, "Set user_alias once"):
                self.load(DEPLOY)
        self.assertEqual(self.path.read_bytes(), before)

    def test_user_labels_that_sanitize_or_truncate_alike_keep_distinct_names(self):
        names = []
        for alias in ("analyst.one", "analyst_one", "analyst.another", "測試"):
            self.save(bucket="test-bucket", role_arn="pipeline-role", user_alias=alias)
            names.append(self.load(DEPLOY)["namespace"])
        self.assertEqual(len(set(names)), 4)
        self.assertTrue(all(len(name) <= 20 for name in names))

    def test_deployment_has_no_execution_or_sync_cells(self):
        self.assertNotIn("sync_execution", "\n".join(DEPLOY))
        self.assertNotIn("start_pipeline_execution", "\n".join(DEPLOY))
        self.assertFalse((ROOT / "notebooks/03_inspect_results.ipynb").exists())
        self.assertFalse((ROOT / "notebooks/04_monitoring_example.ipynb").exists())

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
                     settings={"prefix": "mlops/review/features/"}, receipt={"delivery_id": "ready"},
                     session=Mock(), uuid4=uuid4)
        with patch("mlops_features.data.write_once") as write:
            exec(PREPARE[6], first)
            write.assert_called_once()
        second = self.load(DEPLOY)
        self.assertEqual(second["csv_input"], "sample.csv")
        self.assertEqual(second["tracking_uri"], "")
        self.assertEqual(second["inference_data_source"], "csv_direct")
        self.assertTrue(second["feature_store_config_uri"].startswith(
            "s3://test-bucket/mlops/review/features/configurations/"))
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
