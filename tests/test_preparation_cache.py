"""Verify reusable preparation signatures, invalidation and runtime provenance."""

import json
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND
from unittest.mock import Mock, patch

from mlops_models.entrypoints import get_data
from mlops_models.pipelines import source
from mlops_models.runtime import load_source
from mlops_skeleton.pipelines import walk_steps


class PreparationCache(unittest.TestCase):
    def definitions(self, revision="revision-1", commit="a" * 40, **kwargs):
        return source.build_definitions(
            "cache", "arn:aws:iam::123456789012:role/jobs", "image@sha256:abc",
            "s3://bucket/cache", "s3://bucket/cache/source/direct-deploy/review",
            {"mode": "direct-deploy", "name": "review", "commit": commit, "dirty": True},
            region="us-east-1", data_source="csv_direct",
            csv_input_uri="s3://bucket/input/data.csv",
            preparation_cache_revision=revision, **kwargs,
        )

    def test_only_preparation_is_cached_and_its_keys_are_stable_across_executions(self):
        first = self.definitions()
        second = self.definitions(commit="b" * 40)
        cached = {"GetData", "FeatureSelection", "Preprocess"}
        steps = {s["Name"]: s for s in walk_steps(first["training"]["Steps"])}
        newer = {s["Name"]: s for s in walk_steps(second["training"]["Steps"])}
        for name, step in steps.items():
            if name in cached:
                self.assertEqual(step["CacheConfig"], {"Enabled": True, "ExpireAfter": "PT24H"})
                for key in ("AppSpecification", "ProcessingInputs", "Environment"):
                    self.assertEqual(step["Arguments"].get(key), newer[name]["Arguments"].get(key))
                    self.assertNotIn("Execution.PipelineExecutionId",
                                     json.dumps(step["Arguments"].get(key)))
                self.assertIn("Execution.PipelineExecutionId",
                              json.dumps(step["Arguments"]["ProcessingOutputConfig"]))
            else:
                self.assertNotIn("CacheConfig", step)
        self.assertFalse(any("CacheConfig" in s for s in walk_steps(first["inference"]["Steps"])))
        self.assertEqual(
            steps["Train"]["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
            {"Get": "Steps.Preprocess.ProcessingOutputConfig.Outputs['prepared'].S3Output.S3Uri"},
        )
        self.assertEqual(
            steps["Preprocess"]["Arguments"]["ProcessingInputs"][1]["S3Input"]["S3Uri"],
            {"Get": "Steps.FeatureSelection.ProcessingOutputConfig.Outputs['selection'].S3Output.S3Uri"},
        )
        uncached = self.definitions(revision="")
        self.assertFalse(any("CacheConfig" in s for s in walk_steps(uncached["training"]["Steps"])))

    def test_stage_edits_only_invalidate_their_own_code_fingerprint(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            package = root / "mlops_models"
            files = {
                "model_base/training/train.py": "train-v1",
                "frameworks/autogluon.py": "gluon-v1",
                "models/usecase1/training/train.py": "wrapper-v1",
                "entrypoints/train.py": "entrypoint-v1",
                "model_base/training/preprocess.py": "prepare-v1",
                "models/usecase1/training/preprocess.py": "prepare-wrapper-v1",
                "entrypoints/preprocess_training.py": "prepare-entrypoint-v1",
                "model_base/training/feature_selection.py": "select-v1",
                "models/usecase1/training/feature_selection.py": "select-wrapper-v1",
                "entrypoints/feature_selection.py": "select-entrypoint-v1",
                "entrypoints/get_data.py": "get-v1",
                "models/usecase1/parameters.json": '{"target": "y"}',
                "models/usecase1/model_metadata.json": "{}",
                "models/usecase2/parameters.json": '{"target": "other"}',
                "runtime/load_source.py": "loader-v1",
                "runtime/job_paths.py": "paths-v1",
                "entrypoints/__main__.py": "dispatch-v1",
                "model_base/training/shared_helper.py": "helper-v1",
                "entrypoints/shared_helper.py": "entrypoint-helper-v1",
                "model_base/training/evaluate.py": "evaluate-v1",
                "entrypoints/evaluate.py": "evaluate-entrypoint-v1",
                "model_base/inference/preprocess.py": "inference-prepare-v1",
                "models/usecase1/inference/predict.py": "inference-v1",
                "entrypoints/predict.py": "predict-v1",
                "pipelines/source.py": "publisher-v1",
            }
            for name, content in files.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            csv = root / "data.csv"
            csv.write_text("x,y\n1,2\n")
            session = Mock()
            with patch.object(source, "PACKAGE", package):
                fingerprint = lambda: source.preparation_fingerprint(session, "s3://bucket/data.csv", csv)
                initial = fingerprint()
                expected = {
                    "model_base/training/train.py": set(),
                    "frameworks/autogluon.py": set(),
                    "models/usecase1/training/train.py": set(),
                    "entrypoints/train.py": set(),
                    "model_base/training/preprocess.py": {"Preprocess"},
                    "models/usecase1/training/preprocess.py": {"Preprocess"},
                    "entrypoints/preprocess_training.py": {"Preprocess"},
                    "model_base/training/feature_selection.py": {"FeatureSelection"},
                    "models/usecase1/training/feature_selection.py": {"FeatureSelection"},
                    "entrypoints/feature_selection.py": {"FeatureSelection"},
                    "entrypoints/get_data.py": {"GetData"},
                    "models/usecase1/parameters.json": {"FeatureSelection", "Preprocess"},
                    "models/usecase1/model_metadata.json": {"FeatureSelection", "Preprocess"},
                    "models/usecase2/parameters.json": {"FeatureSelection", "Preprocess"},
                    "runtime/load_source.py": set(initial),
                    "runtime/job_paths.py": set(initial),
                    "entrypoints/__main__.py": set(initial),
                    "model_base/training/shared_helper.py": set(initial),
                    "entrypoints/shared_helper.py": set(initial),
                    "model_base/training/evaluate.py": set(),
                    "entrypoints/evaluate.py": set(),
                    "model_base/inference/preprocess.py": set(),
                    "models/usecase1/inference/predict.py": set(),
                    "entrypoints/predict.py": set(),
                    "pipelines/source.py": set(),
                }
                for name, changed in expected.items():
                    with self.subTest(file=name):
                        path = package / name
                        path.write_text(files[name] + "\nchanged")
                        actual = fingerprint()
                        self.assertEqual({step for step in initial if initial[step] != actual[step]},
                                         changed)
                        path.write_text(files[name])
                csv.write_text("x,y\n1,3\n")
                actual = fingerprint()
                self.assertTrue(all(initial[step] != actual[step] for step in initial))
                session.client.assert_not_called()
                session.client.return_value.head_object.return_value = {"ETag": '"first"'}
                remote = source.preparation_fingerprint(session, "s3://bucket/data.csv")
                session.client.return_value.head_object.return_value = {"ETag": '"second"'}
                self.assertNotEqual(
                    source.preparation_fingerprint(session, "s3://bucket/data.csv"), remote)

    def test_native_cache_keys_keep_upstream_jobs_eligible_after_a_downstream_edit(self):
        revisions = {"GetData": "raw-v1", "FeatureSelection": "selection-v1", "Preprocess": "prep-v1"}
        first = self.definitions(revision=revisions)["training"]
        keys = ("AppSpecification", "Environment", "ProcessingInputs")
        initial = {s["Name"]: {key: s["Arguments"][key] for key in keys}
                   for s in first["Steps"] if s["Name"] in revisions}
        for name, step in initial.items():
            self.assertEqual(step["Environment"]["MLOPS_PREPARATION_REVISION"], revisions[name])

        for changed_step in revisions:
            changed = {**revisions, changed_step: "edited"}
            newer = self.definitions(revision=changed, commit="b" * 40)["training"]
            actual = {s["Name"]: {key: s["Arguments"][key] for key in keys}
                      for s in newer["Steps"] if s["Name"] in revisions}
            self.assertEqual({name for name in initial if initial[name] != actual[name]}, {changed_step})

        # Downstream native inputs reference the producing step's output, so an
        # upstream rerun supplies a new URI; a cache hit retains its old URI.
        steps = {s["Name"]: s for s in first["Steps"]}
        self.assertEqual(steps["FeatureSelection"]["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"],
                         {"Get": "Steps.GetData.ProcessingOutputConfig.Outputs['raw'].S3Output.S3Uri"})
        self.assertEqual(steps["Preprocess"]["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"],
                         {"Get": "Steps.GetData.ProcessingOutputConfig.Outputs['raw'].S3Output.S3Uri"})
        self.assertEqual(steps["Preprocess"]["Arguments"]["ProcessingInputs"][1]["S3Input"]["S3Uri"],
                         {"Get": "Steps.FeatureSelection.ProcessingOutputConfig.Outputs['selection'].S3Output.S3Uri"})

    def test_feature_store_requires_a_pinned_delivery_strategy_before_caching(self):
        with self.assertRaisesRegex(ValueError, "fixed csv_direct"):
            source.build_definitions(
                "cache", "role", "image", "s3://bucket/cache", "s3://bucket/source",
                {"mode": "direct-deploy", "name": "review", "commit": "", "dirty": True},
                region="us-east-1", feature_store_config_uri="s3://bucket/features.json",
                preparation_cache_revision="revision",
            )

    def test_pasteable_cache_cell_reads_all_pages_and_reports_the_changed_key(self):
        cell = Path(__file__).resolve().parents[1] / "mlops-models/scratch/cell.py"
        prefix = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/review/execution/"
        current, previous = prefix + "current", prefix + "previous"
        job_prefix = "arn:aws:sagemaker:us-east-1:123456789012:processing-job/"
        def step(name, job, cached=False):
            value = {"StepName": name, "StepStatus": "Succeeded",
                     "Metadata": {"ProcessingJob": {"Arn": job_prefix + job}}}
            if cached:
                value["CacheHitResult"] = {"SourcePipelineExecutionArn": previous}
            return value
        client = Mock()
        client.get_paginator.return_value.paginate.side_effect = lambda **request: [
            {"PipelineExecutionSteps": [step("GetData", "raw", cached=request["PipelineExecutionArn"] == current)]},
            {"PipelineExecutionSteps": [
                step("FeatureSelection", "new" if request["PipelineExecutionArn"] == current else "old"),
                {"StepName": "Preprocess", "StepStatus": "Failed", "FailureReason": "example failure"},
            ]},
        ]
        client.describe_processing_job.side_effect = lambda ProcessingJobName: {
            "AppSpecification": {"ImageUri": "image"},
            "Environment": {"MLOPS_PREPARATION_REVISION": ProcessingJobName},
            "ProcessingInputs": [],
        }
        code = cell.read_text().replace('\nexecution_arn = ""', f'\nexecution_arn = {current!r}', 1)
        code = code.replace('\nprevious_execution_arn = ""', f'\nprevious_execution_arn = {previous!r}', 1)
        with TemporaryDirectory() as work, patch("boto3.client", return_value=client), \
                patch.object(Path, "cwd", return_value=Path(work)), redirect_stdout(io.StringIO()) as output:
            exec(compile(code, str(cell), "exec"), {})
            report = json.loads((Path(work) / "cache-report-current.json").read_text())
        self.assertEqual(len(report["steps"]), 3)
        self.assertIn("GetData: Succeeded | CACHE HIT", output.getvalue())
        self.assertEqual(report["changed_job_fields"]["FeatureSelection"],
                         ["Environment.MLOPS_PREPARATION_REVISION"])
        self.assertIn("match", report["changed_job_fields"]["GetData"])
        self.assertIn("No compute job", report["changed_job_fields"]["Preprocess"])

    def test_uncached_resolver_passes_its_declared_output_to_tracking(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            (root / "mlops_models").mkdir()
            (root / "source.json").write_text('{"commit": "current", "dirty": false}')
            uri = "s3://bucket/models/usecase1/runs/resolve-one/reports/authorization"
            job = {"AppSpecification": {"ImageUri": "image"}, "ProcessingOutputConfig": {"Outputs": [{"OutputName": "authorization", "S3Output": {
                "S3Uri": uri, "LocalPath": "/opt/ml/processing/output"}}]}}
            original_read = Path.read_text
            def read(path, *args, **kwargs):
                if str(path) == "/opt/ml/config/processingjobconfig.json":
                    return json.dumps(job)
                return original_read(path, *args, **kwargs)
            def dispatch():
                output = Path(sys.argv[-1])  # The loader appends the writable output override.
                (output / "package.json").write_text('{"package": "approved"}')
            start = Mock(return_value=nullcontext(None))
            logger = {"job_tracking": start, "log_job_outputs": Mock()}
            env = {"MLOPS_PROCESSING_OUTPUT_UPLOAD": "application", "MLOPS_TRACKING_URI": "configured",
                   "MLOPS_STEP_REVISION": "", "MLOPS_PREPARATION_REVISION": "", "PIPELINE_EXECUTION_ID": ""}
            with patch.dict(os.environ, env), patch.object(load_source, "__file__", str(root / "load_source.py")), \
                    patch.object(sys, "argv", ["load_source.py", "resolve-model", "--model", "usecase1",
                                              "--output", "/opt/ml/processing/output"]), \
                    patch.object(Path, "read_text", read), patch("boto3.client", return_value=Mock()), \
                    patch.object(load_source.runpy, "run_path", return_value=logger), \
                    patch("mlops_models.entrypoints.__main__.main", side_effect=dispatch), redirect_stdout(io.StringIO()):
                load_source.main()
            start.assert_called_once_with("usecase1", uri)

    def test_loader_restores_provenance_and_uploads_the_execution_scoped_output(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            (root / "mlops_models").mkdir()
            (root / "source.json").write_text(json.dumps({"commit": "b" * 40, "dirty": True}))
            raw = root / "input"
            raw.mkdir()
            (raw / "data.csv").write_text("x,y\n1,2\n")
            job = {"AppSpecification": {"ImageUri": "image"}, "ProcessingOutputConfig": {"Outputs": [{"OutputName": "raw", "S3Output": {
                "S3Uri": "s3://bucket/models/usecase1/runs/execution-two/data/raw",
                "LocalPath": "/opt/ml/processing/output/raw",
            }}]}}
            original_read = Path.read_text
            def read(path, *args, **kwargs):
                if str(path) == "/opt/ml/config/processingjobconfig.json":
                    return json.dumps(job)
                return original_read(path, *args, **kwargs)
            def dispatch():
                self.assertEqual(os.environ["PIPELINE_EXECUTION_ID"], "execution-two")
                self.assertEqual(os.environ["SOURCE_COMMIT"], "b" * 40)
                self.assertEqual(os.environ["SOURCE_DIRTY"], "true")
                output = sys.argv[sys.argv.index("--output") + 1]
                get_data.execute(FIRST, raw, output)
            uploaded = {}
            client = Mock()
            client.upload_file.side_effect = lambda path, bucket, key, **kwargs: uploaded.update(
                {key: Path(path).read_bytes()})
            environment = {
                "MLOPS_PREPARATION_REVISION": "revision", "MLOPS_DATA_SOURCE": "csv_direct",
                "MLOPS_PROCESSING_OUTPUT_UPLOAD": "application",
                "PIPELINE_EXECUTION_ID": "original", "SOURCE_COMMIT": "original", "SOURCE_DIRTY": "false",
            }
            with patch.dict(os.environ, environment), patch.object(load_source, "__file__", str(root / "load_source.py")), \
                    patch.object(sys, "argv", ["load_source.py", "get-data"]), \
                    patch.object(Path, "read_text", read), patch("boto3.client", return_value=client), \
                    patch("mlops_models.entrypoints.__main__.main", side_effect=dispatch):
                load_source.main()
                self.assertEqual(os.environ["PIPELINE_EXECUTION_ID"], "original")
                self.assertEqual(os.environ["SOURCE_COMMIT"], "original")
            prefix = "models/usecase1/runs/execution-two/data/raw/"
            self.assertEqual(uploaded[prefix + "data.csv"], b"x,y\n1,2\n")
            self.assertEqual(json.loads(uploaded[prefix + "provenance.json"])["pipeline_execution_id"],
                             "execution-two")


if __name__ == "__main__":
    unittest.main()
