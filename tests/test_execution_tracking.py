"""Exercise real MLflow records against simulated SageMaker/S3 execution evidence."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
try:
    from mlflow.tracking import MlflowClient
except ImportError:
    MlflowClient = None
from mlops_models.pipelines import source, tracking, inspect
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

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])

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

    def test_connection_check_writes_and_reads_each_model_experiment(self):
        import mlflow
        names = ["review/dev/usecase1", "review/dev/usecase2"]
        session = Mock()
        with patch.object(tracking, "managed_client", return_value=nullcontext(self.mlflow)):
            result = tracking.check_tracking(
                session, {"uri": BASE + ":mlflow-app/test"}, names)
        session.client.assert_not_called()  # App management APIs are not needed.
        self.assertEqual(result["client_version"], mlflow.__version__)
        self.assertIsNone(result["server_version"])
        self.assertEqual([item["name"] for item in result["experiments"]], names)
        for item in result["experiments"]:
            run = self.mlflow.get_run(item["run_id"])
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.metrics["connection_check"], 1)
            self.assertEqual(run.data.params["purpose"], "connection-check")
            self.assertEqual(run.data.tags["mlops.purpose"], "connection-check")
            self.assertEqual(len(run.inputs.dataset_inputs), 1)
            path = self.mlflow.download_artifacts(item["run_id"], "connection-check.json")
            self.assertEqual(json.loads(Path(path).read_text())["run_id"], item["run_id"])
            self.assertEqual(item["artifact_location"],
                             self.mlflow.get_experiment(item["experiment_id"]).artifact_location)

    def test_tracking_server_reports_versions_without_requiring_a_match(self):
        uri = TRACKING_URI.replace("us-east-1", "us-west-2")
        for server_version, client_version in [
            ("2.16.1", "3.10.1"), ("3.0.0", "3.16.1"), ("99.0.0", "3.16.1"),
        ]:
            with self.subTest(server_version=server_version):
                session = Mock()
                session.client.return_value.describe_mlflow_tracking_server.return_value = {
                    "TrackingServerArn": uri, "MlflowVersion": server_version,
                    "IsActive": "Active", "TrackingServerStatus": "Created",
                }
                with patch.object(tracking, "managed_client",
                                  return_value=nullcontext(self.mlflow)) as managed, \
                        patch("mlflow.__version__", client_version):
                    result = tracking.check_tracking(
                        session, {"uri": uri}, [f"review/{server_version}/usecase1"])
                self.assertEqual(result["server_version"], server_version)
                self.assertEqual(result["client_version"], client_version)
                self.assertEqual(session.client.call_args.kwargs["region_name"], "us-west-2")
                session.client.return_value.describe_mlflow_tracking_server.assert_called_once_with(
                    TrackingServerName="test")
                managed.assert_called_once_with(uri)

    def test_failed_artifact_access_stops_check_and_marks_run_failed(self):
        with patch.object(tracking, "managed_client", return_value=nullcontext(self.mlflow)), \
                patch.object(self.mlflow, "download_artifacts", side_effect=PermissionError("S3 denied")):
            with self.assertRaisesRegex(RuntimeError, "read artifact.*S3 denied"):
                tracking.check_tracking(
                    Mock(), {"uri": BASE + ":mlflow-app/test"}, ["review/denied"])
        experiment = self.mlflow.get_experiment_by_name("review/denied")
        runs = self.mlflow.search_runs([experiment.experiment_id])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].info.status, "FAILED")

    def test_incorrect_artifact_contents_fail_verification(self):
        wrong = Path(self.work.name) / "wrong.json"
        wrong.write_text('{"purpose": "another-run"}')
        with patch.object(tracking, "managed_client", return_value=nullcontext(self.mlflow)), \
                patch.object(self.mlflow, "download_artifacts", return_value=str(wrong)):
            with self.assertRaisesRegex(RuntimeError, "Downloaded artifact differs"):
                tracking.check_tracking(
                    Mock(), {"uri": BASE + ":mlflow-app/test"}, ["review/corrupt"])
        experiment = self.mlflow.get_experiment_by_name("review/corrupt")
        self.assertEqual(self.mlflow.search_runs([experiment.experiment_id])[0].info.status, "FAILED")

    def test_existing_client_is_used_and_tracking_uri_is_restored(self):
        import mlflow
        previous = mlflow.get_tracking_uri()
        with patch("mlflow.__version__", "3.16.1"), patch("mlflow.MlflowClient") as client:
            with tracking.managed_client(TRACKING_URI) as actual:
                self.assertIs(actual, client.return_value)
                self.assertEqual(mlflow.get_tracking_uri(), TRACKING_URI)
        client.assert_called_once_with(tracking_uri=TRACKING_URI)
        self.assertEqual(mlflow.get_tracking_uri(), previous)

    def test_dataset_api_failure_is_reported_instead_of_silently_skipped(self):
        with patch.object(tracking, "managed_client", return_value=nullcontext(self.mlflow)), \
                patch.object(self.mlflow, "log_inputs", side_effect=RuntimeError("unsupported API")):
            with self.assertRaisesRegex(RuntimeError, "log dataset reference.*unsupported API"):
                tracking.check_tracking(
                    Mock(), {"uri": BASE + ":mlflow-app/test"}, ["review/dataset-api"])
        experiment = self.mlflow.get_experiment_by_name("review/dataset-api")
        self.assertEqual(self.mlflow.search_runs([experiment.experiment_id])[0].info.status, "FAILED")

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

    def record(self, output, source=None):
        """Simulate the producer saving its output description before upload."""
        from mlops_models.runtime.load_source import describe_outputs
        bucket, prefix = tracking.s3_location(output.rstrip("/") + "/")
        with TemporaryDirectory() as folder, patch.dict("os.environ", {"MLOPS_LOG_ARTIFACTS": "1"}):
            for (item_bucket, key), value in self.s3.objects.items():
                if item_bucket == bucket and key.startswith(prefix):
                    path = Path(folder) / key[len(prefix):]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value)
            record = describe_outputs(source or {}, {"output": (folder, output)})
        self.s3.save(output + "/_execution.json", record)

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
        job = self.processing(arn, "RegisterModel", registration)
        job["Environment"]["REGISTRATION_OUTPUT_URI"] = registration
        self.s3.save(registration + "/package.json", {"package": package})
        self.sm.packages[package] = {
            "ModelPackageArn": package, "ModelApprovalStatus": "PendingManualApproval",
            "MetadataProperties": {"GeneratedBy": arn},
            "InferenceSpecification": {"Containers": [{"Image": "image@sha256:abc",
                                                     "ModelDataUrl": "s3://bucket/model.tar.gz"}]},
        }
        self.record(raw, {"source": {"commit": "raw-commit"}})
        self.record(evaluation)
        self.record(registration)
        return arn, package

    def sync(self, arn, kind="training"):
        return tracking.sync_execution(self.session, arn, {**self.settings, "kind": kind}, client=self.mlflow)

    def direct(self, arn, step, files, kind="training"):
        settings = {**self.settings, "kind": kind}
        receipt = tracking.ensure_run(self.mlflow, self.session, arn, "usecase1", settings)
        output = Path(self.work.name) / step
        output.mkdir(exist_ok=True)
        for name, content in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
        with patch.dict("os.environ", {"MLOPS_LOG_ARTIFACTS": "1"}):
            tracking.log_job_outputs({"client": self.mlflow, "step": step, **receipt}, [output],
                                     {"source_parameters": {"primary_key": "id"}})
        return receipt

    def test_two_jobs_log_to_one_running_run_then_caller_finalizes(self):
        arn, package = self.training()
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Executing"
        first = self.direct(arn, "Train", {"evaluation.json": {"metrics": {"mae": 1.25}}})
        second = self.direct(arn, "Evaluate", {"evaluation.json": {"metrics": {"mae": 1.25}}})
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(self.mlflow.get_run(first["run_id"]).info.status, "RUNNING")
        self.assertEqual(self.mlflow.get_run(first["run_id"]).data.metrics["evaluation.mae"], 1.25)
        self.sm.executions[arn]["PipelineExecutionStatus"] = "Succeeded"
        final = self.sync(arn)
        self.sync(arn)
        run = self.mlflow.get_run(final["run_id"])
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.tags["model.package_arn"], package)
        self.assertEqual(len(run.inputs.dataset_inputs), 1)
        self.assertEqual(len(self.mlflow.get_metric_history(run.info.run_id, "evaluation.mae")), 1)
        self.assertEqual(len(self.mlflow.search_runs([final["experiment_id"]])), 1)

    def test_analysis_logs_nested_html_and_optional_metrics_directly(self):
        arn = self.execution("inference", name="analysis", status="Executing")
        html = b"<html><body>Customer analysis</body></html>"
        receipt = self.direct(arn, "Postprocess", {
            "plots/analysis.html": html, "metrics.json": {"rmse": 2.5, "unavailable": None},
            "summary.json": {"rows": 3}}, "inference")
        run = self.mlflow.get_run(receipt["run_id"])
        self.assertEqual(run.data.metrics["prediction.rmse"], 2.5)
        self.assertEqual(run.data.metrics["prediction.rows"], 3)
        self.assertNotIn("prediction.unavailable", run.data.metrics)
        downloaded = self.mlflow.download_artifacts(run.info.run_id, "outputs/Postprocess/plots/analysis.html")
        self.assertEqual(Path(downloaded).read_bytes(), html)
        self.assertEqual(run.info.status, "RUNNING")

    def test_cached_step_uses_original_outputs_and_never_claims_a_new_producer(self):
        previous, _ = self.training(name="original")
        self.processing(previous, "FeatureSelection", "s3://bucket/original/selection")
        self.s3.save("s3://bucket/original/selection/selected_features.csv", b"feature\nx\ny\n")
        self.record("s3://bucket/original/selection")
        original = self.sync(previous)
        current, _ = self.training(name="repeat")
        self.sm.steps[current].append({"StepName": "FeatureSelection", "StepStatus": "Succeeded", "Metadata": {},
                                       "CacheHitResult": {"SourcePipelineExecutionArn": previous}})
        receipt = self.sync(current)
        run = self.mlflow.get_run(receipt["run_id"])
        self.assertNotEqual(original["run_id"], receipt["run_id"])
        self.assertEqual(run.data.metrics["steps.cached"], 1)
        self.assertEqual(run.data.tags["step.FeatureSelection.producer_execution_arn"], previous)
        downloaded = self.mlflow.download_artifacts(run.info.run_id, "outputs/FeatureSelection/selected_features.csv")
        self.assertEqual(Path(downloaded).read_bytes(), b"feature\nx\ny\n")
        index = tracking._read_json(self.s3, receipt["index_uri"])
        self.assertTrue(next(s for s in index["steps"] if s["name"] == "FeatureSelection")["cached"])

    def test_native_failure_stop_and_retry_reuse_the_same_run(self):
        arn = self.execution("training", status="Executing")
        first = self.sync(arn)
        for native, expected in [("Failed", "FAILED"), ("Executing", "RUNNING"), ("Stopped", "KILLED")]:
            self.sm.executions[arn]["PipelineExecutionStatus"] = native
            result = self.sync(arn)
            self.assertEqual(result["run_id"], first["run_id"])
            self.assertEqual(self.mlflow.get_run(first["run_id"]).info.status, expected)

    def test_missing_receipt_recovers_existing_run_without_duplicate(self):
        arn, _ = self.training()
        first = self.sync(arn)
        bucket, key = tracking.s3_location(first["receipt_uri"])
        self.s3.delete_object(Bucket=bucket, Key=key)
        self.assertEqual(self.sync(arn)["run_id"], first["run_id"])

    def test_inference_links_original_training_run(self):
        trained, package = self.training()
        previous = self.sync(trained)
        arn = self.execution("inference", name="inference")
        self.processing(arn, "ResolveModel", "s3://bucket/inference/package")
        self.s3.save("s3://bucket/inference/package/package.json", {"package": package})
        self.record("s3://bucket/inference/package")
        result = self.sync(arn, "inference")
        self.assertEqual(self.mlflow.get_run(result["run_id"]).data.tags["model.training_run_id"], previous["run_id"])

    def test_runtime_context_sets_run_id_without_ending_it(self):
        from contextlib import nullcontext
        import os
        env = {"MLOPS_TRACKING_URI": TRACKING_URI, "MLOPS_TRACKING_EXPERIMENT_PREFIX": "models",
               "MLOPS_TRACKING_S3_ROOT": "s3://bucket/test", "MLOPS_TRACKING_ENVIRONMENT": "dev",
               "MLOPS_TRACKING_KIND": "training", "MLOPS_TRACKING_STEP": "Train",
               "MLOPS_TRACKING_PIPELINE_ARN": BASE + ":pipeline/test-training"}
        with patch.dict(os.environ, env, clear=True), patch("boto3.Session", return_value=self.session), \
                patch.object(tracking, "managed_client", return_value=nullcontext(self.mlflow)):
            with tracking.job_tracking("usecase1", "s3://bucket/models/usecase1/runs/job123/model") as active:
                self.assertEqual(os.environ["MLFLOW_RUN_ID"], active["run_id"])
                self.assertTrue(active["execution_arn"].endswith("/execution/job123"))
            self.assertNotIn("MLFLOW_RUN_ID", os.environ)
            self.assertEqual(self.mlflow.get_run(active["run_id"]).info.status, "RUNNING")


class TrackingDefinition(unittest.TestCase):
    def test_disabled_preflight_needs_no_clients_or_aws_access(self):
        session = Mock()
        with patch.dict("sys.modules", {"mlflow": None, "sagemaker_mlflow": None}):
            self.assertIsNone(tracking.check_tracking(session, None, []))
        session.client.assert_not_called()

    def test_inactive_or_wrong_server_stops_before_mlflow_writes(self):
        original = {"TrackingServerArn": TRACKING_URI, "MlflowVersion": "3.0.0",
                    "IsActive": "Active", "TrackingServerStatus": "Created"}
        for update, error in [
            ({"IsActive": "Inactive", "TrackingServerStatus": "Stopped"}, "not active"),
            ({"TrackingServerArn": TRACKING_URI + "-other"}, "does not match"),
        ]:
            with self.subTest(update=update):
                session = Mock()
                session.client.return_value.describe_mlflow_tracking_server.return_value = {
                    **original, **update}
                with patch.object(tracking, "managed_client") as client:
                    with self.assertRaisesRegex(ValueError, error):
                        tracking.check_tracking(session, {"uri": TRACKING_URI}, ["review/dev/usecase1"])
                client.assert_not_called()

    def test_direct_logging_adds_no_jobs_and_keeps_cache_configuration(self):
        args = ("test", f"arn:aws:iam::{ACCOUNT}:role/jobs", "image@sha256:abc",
                "s3://bucket/test", "s3://bucket/source",
                {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": False})
        options = dict(region="us-east-1", data_source="csv_direct", csv_input_uri="s3://bucket/input.csv",
                       batch_runtime="both", model_code_source="pipeline",
                       step_cache_revisions={name: "fixed" for name in ("GetData", "FeatureSelection",
                       "Preprocess", "Train", "Evaluate", "BatchGetData", "BatchPreprocess", "Score", "Postprocess")})
        before = source.build_definitions(*args, **options)
        after = source.build_definitions(
            *args, **options, tracking={"uri": TRACKING_URI, "client_version": "3.16.1"})
        for kind in before:
            self.assertEqual(
                json.loads(after[kind]["Metadata"]["Tracking"])["notebook_client_version"], "3.16.1")
            old = {s["Name"]: s for s in walk_steps(before[kind]["Steps"])}
            new = {s["Name"]: s for s in walk_steps(after[kind]["Steps"])}
            self.assertEqual(old.keys(), new.keys())
            for name, step in old.items():
                if step["Type"] in {"Processing", "Training"}:
                    actual = deepcopy(new[name]["Arguments"])
                    env = actual["Environment"]
                    tracking_values = {key: env.pop(key) for key in list(env) if key.startswith("MLOPS_TRACKING_")}
                    self.assertEqual(tracking_values["MLOPS_TRACKING_URI"], TRACKING_URI)
                    self.assertNotIn("MLOPS_TRACKING_CLIENT_VERSION", tracking_values)
                    self.assertTrue(all(isinstance(v, str) for v in tracking_values.values()))
                    self.assertEqual(actual, step["Arguments"])
                    self.assertEqual(step.get("CacheConfig"), new[name].get("CacheConfig"))
                    self.assertNotIn("MLFLOW_RUN_ID", env)
            self.assertNotIn("StartTracking", new)
        with self.assertRaisesRegex(ValueError, "same|separate"):
            source.build_definitions(*args, **options, tracking={"uri": TRACKING_URI, "image_uri": "another-image"})

    def test_disabled_tracking_requires_no_mlflow_client(self):
        with patch.dict("os.environ", {}, clear=True), patch.dict("sys.modules", {"mlflow": None}):
            with tracking.job_tracking("usecase1") as context:
                self.assertIsNone(context)


if __name__ == "__main__":
    unittest.main()
