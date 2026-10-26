"""Producer-owned artifacts, explicit processing order and native final status."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlops_models.runtime import load_source
from mlops_models.pipelines import inspect
from mlops_models.model_base.inference.postprocess import postprocess_predictions
from mlops_models.model_base.inference import postprocess as analysis_module
import test_execution_tracking as fixtures


class ExecutionReports(unittest.TestCase):
    def setUp(self):
        self.sm, self.s3 = fixtures.SageMaker(), fixtures.S3()
        self.session = SimpleNamespace(client=lambda name, **kwargs: self.sm if name == "sagemaker" else self.s3)
        self.arn = fixtures.BASE + ":pipeline/example/execution/current"
        self.sm.executions[self.arn] = {"PipelineExecutionArn": self.arn, "PipelineExecutionStatus": "Executing"}
        self.sm.parameters[self.arn] = {"ModelId": "usecase2", "OutputDataRoot": "s3://bucket/models"}
        self.sm.steps[self.arn] = []

    def job(self, name, status="Succeeded"):
        uri = "s3://bucket/models/usecase2/runs/current/" + name
        self.sm.jobs[name] = {
            "AppSpecification": {"ImageUri": "image"},
            "ProcessingOutputConfig": {"Outputs": [{"OutputName": "result", "S3Output": {"S3Uri": uri}}]}}
        self.sm.steps[self.arn].append({
            "StepName": name, "StepStatus": status, "AttemptCount": 1,
            "Metadata": {"ProcessingJob": {"Arn": fixtures.BASE + ":processing-job/" + name}}})
        return uri

    def test_new_and_renamed_jobs_declare_arbitrary_output_folders(self):
        for name in ("CustomAnalysis", "RenamedOutput"):
            uri = self.job(name)
            with TemporaryDirectory() as directory:
                path = Path(directory) / "nested/custom.csv"
                path.parent.mkdir()
                path.write_text("id,value\n1,2\n")
                with patch.dict(os.environ, {"MLOPS_LOG_ARTIFACTS": "1"}):
                    record = load_source.describe_outputs({}, {"custom": (directory, uri)})
                self.s3.save(uri + "/_execution.json", record)
        report = inspect.collect_execution(self.session, self.arn)
        self.assertEqual({a["step"] for a in report["artifacts"]}, {"CustomAnalysis", "RenamedOutput"})
        self.assertTrue(all(a["relative_path"] == "nested/custom.csv" for a in report["artifacts"]))
        self.assertTrue(all(a["publish_to_mlflow"] for a in report["artifacts"]))
        with patch.dict(os.environ, {}, clear=True), TemporaryDirectory() as directory:
            (Path(directory) / "raw.csv").write_text("x\n1\n")
            record = load_source.describe_outputs({}, {"raw": (directory, "s3://bucket/raw")})
        self.assertFalse(record["artifacts"][0]["publish_to_mlflow"])

    def test_saved_report_refreshes_status_without_describing_jobs_again(self):
        self.job("FinalAnalysis", "Executing")
        report = inspect.collect_execution(self.session, self.arn)
        report["report_phase"] = "outputs_prepared"
        uri = inspect.write_run_index(self.session, report)
        self.assertEqual(inspect._read_json(self.s3, uri)["execution"]["PipelineExecutionStatus"], "Executing")
        # A failed upload after report creation must remain a failed execution.
        self.sm.executions[self.arn]["PipelineExecutionStatus"] = "Failed"
        self.sm.steps[self.arn][0].update(StepStatus="Failed", FailureReason="upload failed")
        with patch.object(self.sm, "describe_processing_job", side_effect=AssertionError("unexpected job lookup")):
            final = inspect.collect_execution(self.session, self.arn)
        self.assertEqual(final["steps"][0]["failure"], "upload failed")
        inspect.write_run_index(self.session, final)
        self.assertEqual(inspect._read_json(self.s3, uri)["execution"]["PipelineExecutionStatus"], "Failed")

    def test_partial_caller_snapshot_does_not_hide_later_jobs(self):
        report = inspect.collect_execution(self.session, self.arn)
        inspect.write_run_index(self.session, report)
        self.job("LaterJob")
        report = inspect.collect_execution(self.session, self.arn)
        self.assertEqual([s["name"] for s in report["steps"]], ["LaterJob"])

    def test_retry_replaces_failed_producer_record(self):
        uri = self.job("Analysis", "Failed")
        self.sm.executions[self.arn]["PipelineExecutionStatus"] = "Failed"
        inspect.write_run_index(self.session, inspect.collect_execution(self.session, self.arn))
        self.sm.steps[self.arn][0].update(StepStatus="Succeeded", AttemptCount=2)
        self.sm.executions[self.arn]["PipelineExecutionStatus"] = "Succeeded"
        self.s3.save(uri + "/_execution.json", {"record_version": 1, "reports": {"new.json": {"value": 2}}})
        report = inspect.collect_execution(self.session, self.arn)
        self.assertEqual(report["reports"]["Analysis/new.json"], {"value": 2})

    def test_postprocess_calls_analysis_and_report_without_monitoring(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.csv").write_text("id,x\n1,2\n")
            (root / "selected.csv").write_text("features\nx\n")
            predictions = root / "scores"
            predictions.mkdir()
            (predictions / "scores.csv").write_text("id,score\n1,.5\n")
            calls = []
            def analyze(data, selection, scores, output):
                calls.append("analysis")
                (output / "analysis.csv").write_text("result\n1\n")
            (root / "parameters.json").write_text('{"primary_key":"id"}')
            (root / "model_metadata.json").write_text('{"feature_mapping":{"x":"x"}}')
            with patch.object(analysis_module, "prediction_inference", analyze), \
                    patch("mlops_models.model_base.inference.drift.compare_batches",
                       side_effect=lambda *a, **kw: calls.append("monitor")), \
                    patch.object(inspect, "write_inference_report",
                                 side_effect=lambda *a, **kw: calls.append("report")):
                postprocess_predictions(root, root / "selected.csv", predictions, root / "output",
                                        root)
            self.assertEqual(calls, ["analysis", "report"])

    def test_postprocess_writes_reference_artifacts_then_caller_finalizes_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "data.csv").write_text("id,x\n001,2\n")
            (inputs / "selected.csv").write_text("features\nx\n")
            (inputs / "predictions.csv").write_text("id,score\n001,.5\n")
            (inputs / "_execution.json").write_text(json.dumps({"producer_execution_arn": "original"}))
            package = root / "source"
            package.mkdir()
            (package / "source.json").write_text('{"commit":"saved-source"}')
            output_uri = self.job("Postprocess", "Executing")
            job = self.sm.jobs["Postprocess"]
            job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["LocalPath"] = "/opt/ml/processing/output"
            job["ProcessingInputs"] = [{"InputName": "inputs", "S3Input": {
                "LocalPath": str(inputs), "S3Uri": "s3://bucket/original"}}]
            read_text = Path.read_text
            def read(path, *args, **kwargs):
                if str(path) == "/opt/ml/config/processingjobconfig.json":
                    return json.dumps(job)
                return read_text(path, *args, **kwargs)
            def analyze(data, selection, predictions, output):
                (output / "analysis.csv").write_text("id,result\n001,1\n")
            environment = {
                "MLOPS_WRITE_RUN_REPORT": "1", "MLOPS_STEP_NAME": "Postprocess",
                "MLOPS_PIPELINE_ARN": self.arn.rsplit("/execution/", 1)[0],
                "PIPELINE_EXECUTION_ID": "current", "MODEL_ID": "usecase2",
                "MLOPS_ENVIRONMENT": "dev", "MLOPS_MODEL_PACKAGE": "model-package",
            }
            (root / "parameters.json").write_text('{"primary_key":"id"}')
            (root / "model_metadata.json").write_text('{"feature_mapping":{"x":"x"}}')
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(analysis_module, "prediction_inference", analyze), \
                    patch.object(inspect, "__file__", str(package / "mlops_models/pipelines/inspect.py")), \
                    patch.object(Path, "read_text", read), patch("boto3.Session", return_value=self.session):
                postprocess_predictions(inputs, inputs / "selected.csv", inputs, root / "output",
                                        root)
            uri = inspect.run_report_uri(self.arn, self.sm.parameters[self.arn])
            report = inspect._read_json(self.s3, uri)
            self.assertEqual(report["report_phase"], "outputs_prepared")
            self.assertEqual(report["execution"]["PipelineExecutionStatus"], "Executing")
            self.assertEqual(report["inference"]["data"], {
                "uri": "s3://bucket/original/data.csv", "producer_execution_arn": "original"})
            self.assertEqual(report["inference"]["model_package_arn"], "model-package")
            self.assertEqual(report["artifacts"][0]["uri"], output_uri + "/analysis.csv")
            self.sm.executions[self.arn]["PipelineExecutionStatus"] = "Succeeded"
            self.sm.steps[self.arn][0]["StepStatus"] = "Succeeded"
            inspect.index_execution(self.session, self.arn)
            reference = inspect.read_run_report(self.session, self.arn)
            self.assertEqual(reference["report_phase"], "finalized")
            self.assertEqual(reference["inference"], report["inference"])

    def test_later_conditional_job_refreshes_artifacts_and_preserves_reference_inputs(self):
        self.job("Postprocess")
        self.sm.steps[self.arn].append({
            "StepName": "DriftComparison", "StepStatus": "Pending", "AttemptCount": 1, "Metadata": {}})
        report = inspect.collect_execution(self.session, self.arn)
        report["inference"] = {"data": {"uri": "s3://bucket/raw/data.csv"}}
        report["report_phase"] = "outputs_prepared"
        inspect.write_run_index(self.session, report)
        self.sm.steps[self.arn].pop()
        uri = self.job("DriftComparison")
        self.s3.save(uri + "/_execution.json", {
            "record_version": 1, "reports": {"monitoring/summary.json": {"drift_status": "compared"}}})
        self.sm.executions[self.arn]["PipelineExecutionStatus"] = "Succeeded"
        final = inspect.collect_execution(self.session, self.arn)
        self.assertEqual(final["inference"], report["inference"])
        self.assertEqual(final["reports"]["DriftComparison/monitoring/summary.json"]["drift_status"], "compared")

    def test_skipped_gate_keeps_postprocess_input_mapping_and_has_no_comparison_artifacts(self):
        self.job("Postprocess")
        report = inspect.collect_execution(self.session, self.arn)
        report["inference"] = {"model_id": "usecase2", "data": {"uri": "s3://bucket/raw/data.csv"}}
        report["report_phase"] = "outputs_prepared"
        inspect.write_run_index(self.session, report)
        self.sm.steps[self.arn].append({
            "StepName": "HasReference", "StepStatus": "Succeeded", "Metadata": {"Condition": {"Outcome": "False"}}})
        self.sm.executions[self.arn]["PipelineExecutionStatus"] = "Succeeded"
        final = inspect.collect_execution(self.session, self.arn)
        self.assertEqual(final["inference"], report["inference"])
        self.assertEqual({s["name"] for s in final["steps"]}, {"Postprocess", "HasReference"})
        self.assertFalse(final["artifacts"])

    def test_semantic_inputs_retain_cached_producer_and_nested_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "_execution.json").write_text(json.dumps({"producer_execution_arn": "original"}))
            job = {"ProcessingInputs": [{"InputName": "anything", "S3Input": {
                "LocalPath": directory, "S3Uri": "s3://bucket/original"}}]}
            with patch.dict(os.environ, {"MODEL_ID": "usecase2", "MLOPS_ENVIRONMENT": "dev",
                                         "MLOPS_MODEL_PACKAGE": "package"}):
                report = inspect.inference_artifacts(
                    job, data_path=root / "data.csv", selection_file=root / "selection.csv",
                    predictions={"nested/scores.csv": root / "nested/scores.csv"},
                    config={"primary_key": "id", "target_key": "label"})
            self.assertEqual(report["predictions"]["nested/scores.csv"], {
                "uri": "s3://bucket/original/nested/scores.csv", "producer_execution_arn": "original"})


if __name__ == "__main__":
    unittest.main()
