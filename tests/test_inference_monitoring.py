"""Verify joins, cached reference selection, pipeline placement and direct MLflow publication."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mlops_models.model_base.inference import drift
from mlops_models.model_base.inference.drift import read_batch, reference_outputs, compare_batches
from mlops_models.pipelines import inspect, source
import test_execution_tracking as fixtures
import test_pipeline_cache as cache_tests


class InferenceMonitoring(unittest.TestCase):
    def test_join_preserves_string_ids_and_matches_reordered_predictions(self):
        result = read_batch(
            io.StringIO("id,x,target\n001,10,0\n002,20,1\n"),
            io.StringIO("id,score\n002,0.9\n001,0.1\n"),
            primary_key="id", features=["x"], target="target", score_column="score")
        self.assertEqual(result.id.tolist(), ["001", "002"])
        self.assertEqual(result._monitor_score.tolist(), [0.1, 0.9])

    def test_join_rejects_missing_extra_and_duplicate_prediction_keys(self):
        for rows in ("001,0.1\n", "001,0.1\n002,0.9\n003,0.2\n", "001,0.1\n001,0.9\n"):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                read_batch(io.StringIO("id,x\n001,10\n002,20\n"), io.StringIO("id,score\n" + rows),
                           primary_key="id", features=["x"], target="target", score_column="score")

    def test_reference_reads_job_written_artifacts_without_step_names(self):
        sm, s3 = fixtures.SageMaker(), fixtures.S3()
        BASE, NOW = fixtures.BASE, fixtures.NOW
        session = SimpleNamespace(client=lambda name, **kwargs: sm if name == "sagemaker" else s3)
        original, reference = [BASE + ":pipeline/test-inference/execution/" + name for name in ("original", "reference")]
        sm.executions[reference] = {"PipelineExecutionArn": reference, "PipelineExecutionStatus": "Succeeded", "CreationTime": NOW}
        sm.parameters[reference] = {"ModelId": "usecase2", "OutputDataRoot": "s3://bucket/models"}
        artifacts = {"model_id": "usecase2", "environment": "dev", "model_package_arn": "package",
                     "data": {"uri": "s3://bucket/original/raw/data.csv", "producer_execution_arn": original},
                     "predictions": {"messaging_final.csv": {"uri": "s3://bucket/original/scores/messaging_final.csv",
                                                              "producer_execution_arn": original}}}
        # No native step/job inventory is configured: only the final report is read.
        s3.save("s3://bucket/models/usecase2/runs/reference/run.json", {
            "report_version": 1, "execution": {"PipelineExecutionArn": reference}, "inference": artifacts})
        result = reference_outputs(session, reference, "usecase2", "dev")
        self.assertEqual(result, artifacts)
        sm.executions[reference]["PipelineExecutionStatus"] = "Failed"
        with self.assertRaisesRegex(ValueError, "successful"):
            reference_outputs(session, reference, "usecase2", "dev")
        sm.executions[reference]["PipelineExecutionStatus"] = "Succeeded"
        with self.assertRaisesRegex(ValueError, "same ModelId"):
            reference_outputs(session, reference, "usecase1", "dev")
        s3.objects.clear()
        with self.assertRaisesRegex(ValueError, "updated pipeline"):
            reference_outputs(session, reference, "usecase2", "dev")

    def test_comparison_is_conditional_and_does_not_change_prediction_jobs(self):
        helper = cache_tests.PipelineCache()
        plain = helper.definitions()
        monitored = helper.definitions(monitoring={"enabled": True})
        self.assertEqual(plain["training"], monitored["training"])
        for before, after in zip(plain["inference"]["Steps"], monitored["inference"]["Steps"]):
            self.assertEqual(before, after)
        gate = monitored["inference"]["Steps"][-1]
        self.assertEqual(gate["Name"], "HasReference")
        self.assertEqual(gate["DependsOn"], ["Postprocess"])
        self.assertEqual(gate["Arguments"]["ElseSteps"], [])
        condition = gate["Arguments"]["Conditions"][0]
        self.assertEqual(condition, {"Type": "Not", "Condition": {
            "Type": "Equals", "LeftValue": {"Get": "Parameters.ReferenceExecutionArn"}, "RightValue": "none"}})
        parameters = {p["Name"]: p for p in monitored["inference"]["Parameters"]}
        self.assertEqual(parameters["ReferenceExecutionArn"]["DefaultValue"], "none")
        step, = gate["Arguments"]["IfSteps"]
        self.assertEqual(step["Name"], "DriftComparison")
        self.assertEqual(step["DependsOn"], ["Postprocess"])
        env = step["Arguments"]["Environment"]
        self.assertEqual(env["MLOPS_REFERENCE_EXECUTION_ARN"], {"Get": "Parameters.ReferenceExecutionArn"})
        self.assertNotIn("MLOPS_TRACKING_URI", env)
        self.assertNotIn("CacheConfig", step)
        self.assertEqual(env["MLOPS_MODEL_CODE_SOURCE"], {"Get": "Parameters.ModelCodeSource"})
        self.assertEqual(env["MLOPS_WRITE_RUN_REPORT"], "1")
        self.assertEqual(step["Arguments"]["AppSpecification"]["ContainerEntrypoint"][-1], "drift")
        post = plain["inference"]["Steps"][-1]["Arguments"]
        self.assertEqual(step["Arguments"]["AppSpecification"]["ImageUri"], post["AppSpecification"]["ImageUri"])
        self.assertEqual(step["Arguments"]["ProcessingInputs"], post["ProcessingInputs"])

    def test_drift_edits_refresh_the_saved_source_without_changing_prediction_revisions(self):
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "mlops_models"
            shutil.copytree(source.PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            with patch.object(source, "PACKAGE", package):
                for name in ("monitoring/report.py", "model_base/inference/drift.py",
                             "models/usecase2/inference/drift.py", "entrypoints/drift.py"):
                    with self.subTest(path=name):
                        before = source.pipeline_fingerprints(Mock())
                        path = package / name
                        path.write_text(path.read_text() + "\n# Drift-only edit\n")
                        after = source.pipeline_fingerprints(Mock())
                        self.assertEqual({key for key in before if before[key] != after[key]},
                                         {"Train", "DriftComparison"})

    def test_comparison_rejects_missing_self_and_foreign_reference_before_reading_data(self):
        pipeline = fixtures.BASE + ":pipeline/example"
        current = pipeline + "/execution/current"
        for reference in ("", " ", current, "not-an-arn", current.replace("123456789012", "999999999999")):
            with self.subTest(reference=reference), patch.dict(os.environ, {
                    "MLOPS_PIPELINE_ARN": pipeline, "PIPELINE_EXECUTION_ID": "current",
                    "MLOPS_REFERENCE_EXECUTION_ARN": reference}, clear=True):
                with self.assertRaisesRegex(ValueError, "different execution"):
                    compare_batches(None, None, None, None, None)

    def test_dispatch_uses_the_selected_models_custom_comparison(self):
        from importlib import import_module
        from mlops_models.entrypoints.__main__ import dispatch_pipeline_step
        for model in ("usecase1", "usecase2"):
            with self.subTest(model=model):
                module = import_module(f"mlops_models.models.{model}.inference.drift")
                args = SimpleNamespace(step="drift", model=model, input="scores", output="output",
                                       model_dir="model", data="raw", selection="selected.csv")
                with patch.object(module, "compare_drift", return_value=Path("custom-output")) as compare:
                    result = dispatch_pipeline_step(args)
                self.assertEqual(result, Path("custom-output"))
                compare.assert_called_once_with(Path("raw"), Path("selected.csv"), Path("scores"),
                                                Path("output"), Path("model"))

    def test_default_step_owns_input_resolution_comparison_report_and_output(self):
        from mlops_models.entrypoints.drift import execute
        for model in ("usecase1", "usecase2"):
            with self.subTest(model=model), TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "data.csv").write_text("id,x\n001,2\n")
                selection = root / "selected.csv"
                selection.write_text("features\nx\n")
                scores = root / "scores/nested"
                scores.mkdir(parents=True)
                (scores / "predictions.csv").write_text("id,score\n001,.5\n")
                saved_model = root / "model"
                saved_model.mkdir()
                (saved_model / "parameters.json").write_text('{"primary_key":"id"}')
                (saved_model / "model_metadata.json").write_text('{"saved":"configuration"}')
                calls = []
                def compare(data, features, predictions, output, config):
                    calls.append("compare")
                    self.assertEqual(data, root / "data.csv")
                    self.assertEqual(features, selection)
                    self.assertEqual(predictions, {"nested/predictions.csv": scores / "predictions.csv"})
                    self.assertEqual(config["model_metadata"], {"saved": "configuration"})
                    report = output / "custom/report.html"
                    report.parent.mkdir()
                    report.write_text("<html>comparison</html>")
                def report(output, **kwargs):
                    calls.append("report")
                    self.assertTrue((output / "custom/report.html").is_file())
                    self.assertEqual(kwargs["data_path"], root / "data.csv")
                with patch.object(drift, "compare_batches", side_effect=compare), \
                        patch.object(inspect, "write_inference_report", side_effect=report):
                    result = execute(model, scores.parent, root / "output",
                                     model_dir=saved_model, data_directory=root, selection_file=selection)
                self.assertEqual(calls, ["compare", "report"])
                self.assertEqual(result, root / "output")
                self.assertEqual((result / "custom/report.html").read_text(), "<html>comparison</html>")

    def test_combined_pipeline_only_monitors_the_processing_branch(self):
        definitions = source.build_definitions(
            "cache", "arn:aws:iam::123456789012:role/jobs", "image@sha256:abc",
            "s3://bucket/cache", "s3://bucket/cache/source",
            {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": False},
            region="us-east-1", data_source="csv_direct", csv_input_uri="s3://bucket/data.csv",
            batch_runtime="both", monitoring={"enabled": True})
        from mlops_skeleton.pipelines import walk_steps
        body = definitions["inference"]
        steps = {step["Name"]: step for step in walk_steps(body["Steps"])}
        self.assertIn("MLOPS_MONITORING_CONFIG", steps["DriftComparison"]["Arguments"]["Environment"])
        self.assertNotIn("MLOPS_MONITORING_CONFIG", steps["Postprocess"]["Arguments"]["Environment"])
        self.assertNotIn("MLOPS_MONITORING_CONFIG", steps["PostprocessTransform"]["Arguments"]["Environment"])
        self.assertEqual(sum(parameter["Name"] == "ReferenceExecutionArn" for parameter in body["Parameters"]), 1)
        self.assertEqual(steps["HasReference"]["DependsOn"], ["ResolveModel"])
        self.assertIn({"Type": "Equals", "LeftValue": {"Get": "Parameters.InferenceRuntime"},
                       "RightValue": "processing"}, steps["HasReference"]["Arguments"]["Conditions"])
        for step in body["Steps"]:
            if step["Type"] == "Condition":
                for branch in ("IfSteps", "ElseSteps"):
                    self.assertTrue(all(child["Type"] != "Condition" for child in step["Arguments"][branch]))


class MonitoringTracking(unittest.TestCase):
    setUp = fixtures.ExecutionTracking.setUp
    execution = fixtures.ExecutionTracking.execution
    processing = fixtures.ExecutionTracking.processing
    record = fixtures.ExecutionTracking.record
    direct = fixtures.ExecutionTracking.direct
    sync = fixtures.ExecutionTracking.sync

    def test_nested_monitoring_metrics_share_inference_run_and_survive_cached_sync(self):
        arn = self.execution("inference", name="original")
        summary = {"context": {"reference_execution_arn": "prior", "model_id": "usecase1"},
                   "drift_status": "compared", "quality_status": "unavailable"}
        files = {"summary.json": {"rows": 2}, "metrics.json": {"custom": 42},
                 "monitoring/summary.json": summary, "monitoring/metrics.json": {"drifted_fraction": 0.5},
                 "monitoring/evidently_data_drift/report.html": b"<html>drift report</html>"}
        receipt = self.direct(arn, "DriftComparison", files, kind="inference")
        run = self.mlflow.get_run(receipt["run_id"])
        self.assertEqual(run.info.status, "RUNNING")
        self.assertEqual(run.data.metrics["monitoring.drifted_fraction"], 0.5)
        self.assertEqual(run.data.tags["monitoring.reference_execution_arn"], "prior")
        output = "s3://bucket/original/predictions"
        self.processing(arn, "DriftComparison", output)
        for name, value in files.items():
            self.s3.save(output + "/" + name, value)
        self.record(output)
        cached = self.execution("inference", name="cached")
        self.sm.steps[cached] = [dict(deepcopy(self.sm.steps[arn][0]),
                                      CacheHitResult={"SourcePipelineExecutionArn": arn})]
        result = self.sync(cached, kind="inference")
        replay = self.mlflow.get_run(result["run_id"])
        self.assertEqual(replay.data.metrics["monitoring.drifted_fraction"], 0.5)
        self.assertEqual(replay.data.tags["monitoring.reference_execution_arn"], "prior")
        self.assertEqual(replay.data.tags["step.DriftComparison.producer_execution_arn"], arn)


if __name__ == "__main__":
    unittest.main()
