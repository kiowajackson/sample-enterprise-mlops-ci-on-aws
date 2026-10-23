"""Exercise real MLflow records against simulated SageMaker/S3 execution evidence."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError
try:
    from mlflow.tracking import MlflowClient
except ImportError:
    MlflowClient = None
from mlops_models.pipelines import source, tracking
from mlops_skeleton.pipelines import walk_steps

ACCOUNT = "123456789012"
BASE = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}"
TRACKING_URI = BASE + ":mlflow-tracking-server/test"
NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


def failure(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "S3")


class S3:
    exceptions = SimpleNamespace(ClientError=ClientError)

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **kwargs):
        existing = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and existing is not None:
            raise failure("PreconditionFailed")
        if IfMatch and (existing is None or self.etag(existing) != IfMatch):
            raise failure("PreconditionFailed")
        self.objects[Bucket, Key] = Body
        return {"ETag": self.etag(Body)}

    @staticmethod
    def etag(body):
        return '"' + hashlib.md5(body).hexdigest() + '"'

    def get_object(self, Bucket, Key, IfMatch=None):
        if (Bucket, Key) not in self.objects:
            raise failure("NoSuchKey")
        content = self.objects[Bucket, Key]
        if IfMatch and self.etag(content) != IfMatch:
            raise failure("PreconditionFailed")
        return {"Body": io.BytesIO(content), "ETag": self.etag(content), "ContentLength": len(content)}

    def delete_object(self, Bucket, Key, IfMatch=None):
        if (Bucket, Key) not in self.objects:
            return
        if IfMatch and self.etag(self.objects[Bucket, Key]) != IfMatch:
            raise failure("PreconditionFailed")
        del self.objects[Bucket, Key]

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return SimpleNamespace(paginate=lambda Bucket, Prefix: [{
            "Contents": [{"Key": key, "Size": len(value), "ETag": self.etag(value)}
                         for (bucket, key), value in sorted(self.objects.items())
                         if bucket == Bucket and key.startswith(Prefix)]
        }])

    def save(self, uri, value):
        bucket, key = tracking.s3_location(uri)
        self.put_object(Bucket=bucket, Key=key,
                        Body=value if isinstance(value, bytes) else json.dumps(value).encode())


class SageMaker:
    def __init__(self):
        self.executions, self.steps, self.parameters, self.jobs, self.packages = {}, {}, {}, {}, {}

    def describe_pipeline_execution(self, PipelineExecutionArn):
        return deepcopy(self.executions[PipelineExecutionArn])

    def list_pipeline_parameters_for_execution(self, PipelineExecutionArn):
        return {"PipelineParameters": [{"Name": key, "Value": value}
                                       for key, value in self.parameters[PipelineExecutionArn].items()]}

    def get_paginator(self, operation):
        assert operation == "list_pipeline_execution_steps"
        return SimpleNamespace(paginate=lambda PipelineExecutionArn: [
            {"PipelineExecutionSteps": deepcopy(self.steps[PipelineExecutionArn])}])

    def describe_processing_job(self, ProcessingJobName):
        return deepcopy(self.jobs[ProcessingJobName])

    def describe_training_job(self, TrainingJobName):
        return deepcopy(self.jobs[TrainingJobName])

    def describe_model_package(self, ModelPackageName):
        return deepcopy(self.packages[ModelPackageName])


@unittest.skipUnless(MlflowClient, "Install requirements-tracking.txt for real MLflow integration tests.")
class ExecutionTracking(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        store = Path(self.work.name) / "mlruns"
        store.mkdir()
        import mlflow
        previous = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, previous)
        mlflow.set_tracking_uri(store.as_uri())
        self.mlflow = MlflowClient(tracking_uri=store.as_uri())
        self.sm, self.s3 = SageMaker(), S3()
        self.session = SimpleNamespace(client=lambda name, **kwargs:
                                       self.sm if name == "sagemaker" else self.s3)
        self.settings = {"uri": TRACKING_URI, "s3_root": "s3://bucket/test",
                         "experiment_prefix": "models", "environment": "dev", "kind": "training"}

    def execution(self, kind, model="usecase1", name="one", status="Succeeded"):
        arn = BASE + f":pipeline/test-{kind}/execution/{name}"
        self.sm.executions[arn] = {"PipelineExecutionArn": arn, "PipelineExecutionStatus": status,
                                   "CreationTime": NOW, "LastModifiedTime": NOW}
        self.sm.parameters[arn] = {"ModelId": model, "Environment": "dev",
                                   "OutputDataRoot": "s3://bucket/test/models"}
        self.sm.steps[arn] = []
        return arn

    def processing(self, arn, step, output=None, status="Succeeded"):
        name = arn.rsplit("/", 1)[-1] + "-" + step
        job = {"ProcessingJobArn": BASE + ":processing-job/" + name,
               "AppSpecification": {"ImageUri": "image@sha256:abc"},
               "Environment": {"MLOPS_STEP_REVISION": "source-" + step},
               "ProcessingOutputConfig": {"Outputs": []}}
        if output:
            job["ProcessingOutputConfig"]["Outputs"] = [{"OutputName": "output",
                                                        "S3Output": {"S3Uri": output}}]
        self.sm.jobs[name] = job
        self.sm.steps[arn].append({"StepName": step, "StepStatus": status,
                                  "Metadata": {"ProcessingJob": {"Arn": job["ProcessingJobArn"]}}})
        return job

    def training(self, model="usecase1", name="train"):
        arn = self.execution("training", model, name)
        raw = f"s3://bucket/test/{name}/raw"
        self.processing(arn, "GetData", raw)
        self.s3.save(raw + "/provenance.json", {"sha256": "actual-data-digest", "records": 3})
        self.s3.save(raw + "/_execution.json", {"source": {"commit": "raw-commit"}})
        evaluation = f"s3://bucket/test/{name}/evaluation"
        self.processing(arn, "Evaluate", evaluation)
        task = "regression" if model == "usecase1" else "classification"
        self.s3.save(evaluation + "/evaluation.json", {
            "metrics": {"mae": 1.25} if task == "regression" else {"roc_auc": 0.8},
            "data_usage": "validation", "model_source_commit": "model-commit",
        })
        self.s3.save(evaluation + "/parameters.json", {"task_type": task, "primary_key": "id"})
        package = BASE + f":model-package/{model}/1"
        registration = f"s3://bucket/test/{name}/registry"
        job = self.processing(arn, "RegisterModel")
        job["Environment"]["REGISTRATION_OUTPUT_URI"] = registration
        self.s3.save(registration + "/package.json", {"package": package})
        self.sm.packages[package] = {
            "ModelPackageArn": package, "ModelApprovalStatus": "PendingManualApproval",
            "MetadataProperties": {"GeneratedBy": arn},
            "InferenceSpecification": {"Containers": [{"Image": "image@sha256:abc",
                                                     "ModelDataUrl": "s3://bucket/model.tar.gz"}]},
        }
        return arn, package

    def sync(self, arn, kind="training", phase="sync"):
        return tracking.sync_execution(self.session, arn, {**self.settings, "kind": kind},
                                       phase=phase, client=self.mlflow)

    def test_both_tasks_log_real_mlflow_params_metrics_and_datasets_without_duplicates(self):
        for model in ("usecase1", "usecase2"):
            arn, package = self.training(model, model)
            first = self.sync(arn)
            second = self.sync(arn)
            self.assertEqual(first["run_id"], second["run_id"])
            run = self.mlflow.get_run(first["run_id"])
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.tags["registry.package_arn"], package)
            self.assertEqual(run.data.tags["source.model_commit"], "model-commit")
            self.assertEqual(run.data.params["training.primary_key"], "id")
            self.assertEqual(len(run.inputs.dataset_inputs), 1)
            metric = "evaluation.mae" if model == "usecase1" else "evaluation.roc_auc"
            self.assertEqual(len(self.mlflow.get_metric_history(first["run_id"], metric)), 1)
            self.assertEqual(len(self.mlflow.search_runs([first["experiment_id"]])), 1)

    def test_inference_links_the_registered_training_run_and_publishes_analysis(self):
        trained, package = self.training()
        train_run = self.sync(trained)
        self.sm.packages[package]["ModelApprovalStatus"] = "Approved"
        arn = self.execution("inference", name="score")
        self.processing(arn, "ResolveModel", "s3://bucket/test/score/resolve")
        self.s3.save("s3://bucket/test/score/resolve/package.json", {"package": package})
        self.processing(arn, "Postprocess", "s3://bucket/test/score/analysis")
        self.s3.save("s3://bucket/test/score/analysis/summary.json", {"rows": 3})
        self.s3.save("s3://bucket/test/score/analysis/prediction_analysis.csv",
                     b"id,PREDICTION\n001,1.2\n002,1.3\n003,1.4\n")
        self.s3.save("s3://bucket/test/score/analysis/_execution.json", {
            "source_parameters": {"primary_key": "current_id"},
            "checkpoint_parameters": {"primary_key": "fitted_id", "threshold": 0.7},
        })
        receipt = self.sync(arn, "inference")
        run = self.mlflow.get_run(receipt["run_id"])
        self.assertEqual(run.data.params["Postprocess.source_parameters.primary_key"], "current_id")
        self.assertEqual(run.data.params["Postprocess.checkpoint_parameters.primary_key"], "fitted_id")
        self.assertEqual(run.data.params["Postprocess.checkpoint_parameters.threshold"], "0.7")
        self.assertEqual(run.data.tags["model.training_run_id"], train_run["run_id"])
        self.assertEqual(run.data.metrics["prediction.rows"], 3)
        artifacts = self.mlflow.list_artifacts(receipt["run_id"], "outputs/Postprocess")
        self.assertEqual([item.path for item in artifacts], ["outputs/Postprocess/prediction_analysis.csv"])

    def test_cached_execution_uses_original_job_outputs_and_code_identity(self):
        previous, _ = self.training()
        self.sync(previous)
        current, _ = self.training(name="repeat")
        self.sm.steps[current][0] = {
            "StepName": "GetData", "StepStatus": "Succeeded", "Metadata": {},
            "CacheHitResult": {"SourcePipelineExecutionArn": previous},
        }
        report = tracking.collect_execution(self.session, current)
        step = report["steps"][0]
        self.assertTrue(step["cached"])
        self.assertEqual(step["producer_execution_arn"], previous)
        self.assertEqual(step["source_identity"]["source"]["commit"], "raw-commit")
        receipt = self.sync(current)
        self.assertEqual(self.mlflow.get_run(receipt["run_id"]).data.metrics["steps.cached"], 1)
        index = tracking._read_json(self.s3, receipt["index_uri"])
        self.assertEqual(index["steps"][0]["producer_execution_arn"], previous)
        self.assertEqual(index["steps"][0]["outputs"]["output"], "s3://bucket/test/train/raw/")
        self.assertEqual(index["mlflow"]["run_id"], receipt["run_id"])

    def test_custom_analysis_metrics_and_html_are_visible_without_changing_analysis(self):
        arn = self.execution("inference", name="analysis")
        output = "s3://bucket/test/analysis"
        self.processing(arn, "Postprocess", output)
        self.s3.save(output + "/metrics.json", {"rmse": 2.5, "mae": 1.2, "r2": None})
        html = b"<html><body><h1>Prediction analysis</h1></body></html>"
        self.s3.save(output + "/plots/analysis.html", html)
        self.s3.save(output + "/details.json", {"segment": "example", "rows": 7})
        receipt = self.sync(arn, "inference")
        self.sync(arn, "inference")
        run = self.mlflow.get_run(receipt["run_id"])
        self.assertEqual(run.data.metrics["prediction.rmse"], 2.5)
        self.assertNotIn("prediction.r2", run.data.metrics)
        self.assertEqual(len(self.mlflow.get_metric_history(receipt["run_id"], "prediction.rmse")), 1)
        downloaded = self.mlflow.download_artifacts(receipt["run_id"], "outputs/Postprocess/plots/analysis.html")
        self.assertEqual(Path(downloaded).read_bytes(), html)
        self.assertEqual(tracking._read_json(self.s3, receipt["index_uri"])["status"], "Succeeded")

    def test_selection_reports_are_copied_and_index_works_without_mlflow(self):
        arn, _ = self.training()
        output = "s3://bucket/test/selection"
        self.processing(arn, "FeatureSelection", output)
        self.s3.save(output + "/selected_features.csv", b"feature\\nx\\ny\\n")
        index_uri = tracking.index_execution(self.session, arn)
        index = tracking._read_json(self.s3, index_uri)
        self.assertIsNone(index["mlflow"])
        self.assertTrue(any(item["step"] == "FeatureSelection" for item in index["artifacts"]))
        receipt = self.sync(arn)
        self.assertEqual([item.path for item in self.mlflow.list_artifacts(
            receipt["run_id"], "outputs/FeatureSelection")], ["outputs/FeatureSelection/selected_features.csv"])

    def test_failed_sync_keeps_s3_index_and_existing_mlflow_link(self):
        arn, _ = self.training()
        receipt = self.sync(arn)
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Failed"
        with patch.object(self.mlflow, "get_experiment_by_name", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.sync(arn)
        index = tracking._read_json(self.s3, receipt["index_uri"])
        self.assertEqual(index["status"], "Failed")
        self.assertEqual(index["mlflow"]["run_id"], receipt["run_id"])

    def test_failure_stop_and_retry_update_same_run(self):
        arn = self.execution("training", status="Executing")
        first = self.sync(arn, phase="start")
        self.assertEqual(first["status"], "RUNNING")
        self.processing(arn, "Train", status="Failed")
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Failed"
        failed = self.sync(arn)
        self.assertEqual(failed["status"], "FAILED")
        self.sm.steps[arn] = []
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Executing"
        retry = self.sync(arn, phase="start")
        self.assertEqual(retry["status"], "RUNNING")
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Stopped"
        stopped = self.sync(arn)
        self.assertEqual(stopped["status"], "KILLED")
        self.assertEqual({x["run_id"] for x in (first, failed, retry, stopped)}, {first["run_id"]})

    def test_lost_receipt_recovers_run_and_live_lock_rejects_concurrent_publisher(self):
        arn, _ = self.training()
        first = self.sync(arn)
        key = hashlib.sha256((TRACKING_URI + "/" + arn).encode()).hexdigest()
        uri = "s3://bucket/test/tracking/" + key + "/run.json"
        bucket, path = tracking.s3_location(uri)
        self.s3.delete_object(Bucket=bucket, Key=path)
        self.assertEqual(self.sync(arn)["run_id"], first["run_id"])
        with tracking._lease(self.s3, uri.replace("run.json", "lock.json"), ""):
            with self.assertRaisesRegex(RuntimeError, "already being synchronized"):
                self.sync(arn)

    def test_final_job_marks_business_completion_but_records_nonterminal_pipeline(self):
        arn, _ = self.training()
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Executing"
        receipt = self.sync(arn, phase="finish")
        self.assertEqual(receipt["status"], "FINISHED")
        run = self.mlflow.get_run(receipt["run_id"])
        self.assertEqual(run.data.tags["pipeline.status"], "Executing")
        self.assertEqual(run.data.tags["tracking.finalized_before_pipeline_terminal"], "true")
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Succeeded"
        receipt = self.sync(arn)
        self.assertEqual(self.mlflow.get_run(receipt["run_id"]).data.tags[
            "tracking.finalized_before_pipeline_terminal"], "false")


    def test_finish_waits_for_native_step_visibility_and_rejects_incomplete_snapshot(self):
        arn, _ = self.training()
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Executing"
        ready = tracking.collect_execution(self.session, arn)
        pending = deepcopy(ready)
        pending["steps"][-1]["status"] = "Executing"
        with patch.object(tracking, "collect_execution", side_effect=[pending, ready]), \
                patch.object(tracking.time, "sleep") as wait:
            result = self.sync(arn, phase="finish")
            self.assertEqual(result["status"], "FINISHED")
            wait.assert_called_once_with(2)
        with patch.object(tracking, "collect_execution", return_value=pending), \
                patch.object(tracking.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "cannot confirm"):
                self.sync(arn, phase="finish")


class TrackingDefinition(unittest.TestCase):
    def test_tracking_main_allows_omitted_optional_kms_key(self):
        settings = {"uri": TRACKING_URI, "experiment_prefix": "review",
                    "s3_root": "s3://bucket/review", "environment": "dev", "kind": "training"}
        env = {"MLOPS_TRACKING_" + key.upper(): value for key, value in settings.items()}
        env["PIPELINE_EXECUTION_ARN"] = "execution"
        with patch.dict("os.environ", env, clear=True), \
                patch("sys.argv", ["tracking", "--phase", "start"]), \
                patch("boto3.Session", return_value="session"), \
                patch.object(tracking, "sync_execution", return_value={}) as sync, \
                patch("builtins.print"):
            tracking.main()
        sync.assert_called_once_with("session", "execution", {**settings, "kms_key": ""}, phase="start")

    def test_tracking_observers_preserve_compute_arguments_and_both_branches(self):
        common = dict(
            region="us-east-1", data_source="csv_direct", csv_input_uri="s3://bucket/input.csv",
            batch_runtime="both", model_code_source="pipeline",
            step_cache_revisions={name: "fixed" for name in (
                "GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate",
                "BatchGetData", "BatchPreprocess", "Score", "Postprocess")},
        )
        args = ("test", f"arn:aws:iam::{ACCOUNT}:role/jobs", "image@sha256:abc",
                "s3://bucket/test", "s3://bucket/source",
                {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": False})
        before = source.build_definitions(*args, **common)
        after = source.build_definitions(*args, **common, tracking={"uri": TRACKING_URI})
        for kind in before:
            # SageMaker rejects nested objects in the pipeline Metadata map.
            self.assertTrue(all(isinstance(value, str) for value in after[kind]["Metadata"].values()))
            self.assertEqual(json.loads(after[kind]["Metadata"]["Tracking"])["uri"], TRACKING_URI)
            old = {step["Name"]: step for step in walk_steps(before[kind]["Steps"])}
            new = {step["Name"]: step for step in walk_steps(after[kind]["Steps"])}
            self.assertIn("StartTracking", new)
            for name in tracking.TRACKING_STEPS & new.keys():
                prefix = new[name]["Arguments"]["ProcessingJobName"]
                self.assertIsInstance(prefix, str)
                self.assertTrue(prefix.startswith("test-track-"))
                self.assertLessEqual(len(prefix), 38)
            env = new["StartTracking"]["Arguments"]["Environment"]
            self.assertTrue(all(len(value) <= 256 for value in env.values() if isinstance(value, str)))
            self.assertEqual(env["MLOPS_TRACKING_URI"], TRACKING_URI)
            expected_settings = {
                key: env["MLOPS_TRACKING_" + key.upper()]
                for key in ("uri", "experiment_prefix", "s3_root", "environment", "kind", "kms_key")
            }
            with patch.dict("os.environ", {key: value for key, value in env.items()
                                          if isinstance(value, str)} | {"PIPELINE_EXECUTION_ARN": "execution"}), \
                    patch("sys.argv", ["tracking", "--phase", "start"]), \
                    patch("boto3.Session", return_value="session"), \
                    patch.object(tracking, "sync_execution", return_value={}) as sync, \
                    patch("builtins.print"):
                tracking.main()
            sync.assert_called_once_with("session", "execution", expected_settings, phase="start")
            for name, step in old.items():
                if step["Type"] in {"Processing", "Training", "Transform"}:
                    self.assertEqual(step["Arguments"], new[name]["Arguments"])
                    self.assertEqual(step.get("CacheConfig"), new[name].get("CacheConfig"))
            for name in set(new) - set(old):
                self.assertNotIn("CacheConfig", new[name])
        inference = {step["Name"] for step in walk_steps(after["inference"]["Steps"])}
        self.assertTrue({"TrackInference", "TrackTransform"} <= inference)


if __name__ == "__main__":
    unittest.main()
