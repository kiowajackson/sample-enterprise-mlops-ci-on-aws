"""Exercise event completion against real local MLflow and simulated native/S3 records."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import json
import io
from pathlib import Path
import sys
from unittest.mock import Mock, patch
import unittest
import zipfile

from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parents[1]))
from operations.completion import handler as completion
from operations.completion.deploy import deploy_completion
from mlops_models.pipelines import tracking
import test_execution_tracking as fixtures


class AutomaticCompletion(unittest.TestCase):
    setUp = fixtures.ExecutionTracking.setUp
    execution = fixtures.ExecutionTracking.execution
    processing = fixtures.ExecutionTracking.processing
    record = fixtures.ExecutionTracking.record
    training = fixtures.ExecutionTracking.training

    def finish(self, arn, tracking_enabled=True):
        self.sm.describe_pipeline_definition_for_execution = Mock(return_value={"PipelineDefinition": json.dumps({
            "Metadata": {"Tracking": json.dumps(self.settings)} if tracking_enabled else {},
        })})
        with patch.object(tracking, "managed_client", return_value=nullcontext(self.mlflow)):
            return completion.finalize(self.session, arn, {arn.split("/execution/")[0]: "training"},
                                       self.settings["s3_root"], fixtures.NOW - timedelta(days=1))

    def test_terminal_events_finalize_without_a_launcher_and_repeated_delivery_is_idempotent(self):
        for native, expected in (("Succeeded", "FINISHED"), ("Failed", "FAILED"), ("Stopped", "KILLED")):
            arn = self.execution("training", name=native, status=native)
            first = self.finish(arn)
            self.assertEqual(self.mlflow.get_run(first["run_id"]).info.status, expected)
            self.assertEqual(self.finish(arn)["run_id"], first["run_id"])
            self.sm.describe_pipeline_definition_for_execution.assert_not_called()
            index = json.loads(self.s3.objects[("bucket", f"test/models/usecase1/runs/{native}/run.json")])
            self.assertEqual(index["report_phase"], "finalized")
            self.assertEqual(index["execution"]["PipelineExecutionStatus"], native)

    def test_cache_only_execution_copies_outputs_and_retains_original_producer(self):
        original, _ = self.training()
        arn = self.execution("training", name="cached")
        self.sm.steps[arn] = [
            {**deepcopy(step), "CacheHitResult": {"SourcePipelineExecutionArn": original}}
            for step in self.sm.steps[original]]
        result = self.finish(arn)
        run = self.mlflow.get_run(result["run_id"])
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.metrics["steps.cached"], 3)
        self.assertEqual(run.data.tags["step.Evaluate.producer_execution_arn"], original)
        self.assertEqual(run.data.metrics["evaluation.mae"], 1.25)

    def test_tracking_disabled_still_finalizes_s3_report(self):
        arn = self.execution("training", name="no-tracking")
        with patch.object(completion, "sync_execution") as sync:
            result = self.finish(arn, tracking_enabled=False)
        sync.assert_not_called()
        self.assertEqual(result["status"], "Succeeded")
        self.assertTrue(result["index_uri"].endswith("/no-tracking/run.json"))

    def test_active_and_foreign_executions_do_not_write_completion(self):
        arn = self.execution("training", status="Executing")
        result = self.finish(arn)
        self.assertEqual(result["status"], "ignored")
        self.assertFalse(self.s3.objects)
        with self.assertRaisesRegex(ValueError, "outside"):
            completion.finalize(self.session, arn, {}, "s3://bucket/test", fixtures.NOW)

    def test_failed_publication_is_retried_and_resumed_execution_is_refinalized(self):
        arn = self.execution("training", name="retry", status="Failed")
        with patch.object(completion, "sync_execution", side_effect=RuntimeError("service unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                self.finish(arn)
        self.assertFalse(self.s3.objects)  # No false completed marker or remaining lock.
        first = self.finish(arn)
        self.sm.executions[arn].update(
            PipelineExecutionStatus="Succeeded", LastModifiedTime=fixtures.NOW + timedelta(minutes=1))
        second = self.finish(arn)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(self.mlflow.get_run(second["run_id"]).info.status, "FINISHED")

    def test_reconciliation_continues_after_one_failure_and_reports_it(self):
        sm = Mock()
        sm.get_paginator.return_value.paginate.return_value = [{
            "PipelineExecutionSummaries": [
                {"PipelineExecutionArn": "failed", "PipelineExecutionStatus": "Failed"},
                {"PipelineExecutionArn": "success", "PipelineExecutionStatus": "Succeeded"},
            ]}]
        session = Mock()
        session.client.return_value = sm
        context = Mock()
        context.get_remaining_time_in_millis.return_value = 900000
        with patch.dict("os.environ", {"PIPELINES": '{"pipeline":"training"}',
                                      "S3_ROOT": "s3://bucket/test", "WATCH_FROM": fixtures.NOW.isoformat()}), \
                patch.object(completion.boto3, "Session", return_value=session), \
                patch.object(completion, "finalize", side_effect=[RuntimeError("denied"), {}]) as finish:
            with self.assertRaisesRegex(RuntimeError, "1 execution"):
                completion.handler({"reconcile": True}, context)
        self.assertEqual(finish.call_count, 2)


class CompletionDeployment(unittest.TestCase):
    def test_nested_storage_root_scopes_writes_without_changing_pipeline_identity(self):
        root = Path(__file__).parents[1]
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.write(root / "operations/completion/handler.py", "handler.py")
            for name in ("__init__.py", "storage.py", "pipelines/__init__.py",
                         "pipelines/inspect.py", "pipelines/tracking.py"):
                archive.write(root / "mlops-models/src/mlops_models" / name,
                              "mlops_models/" + name)
        for prefix in ("mlops/review", "team/project/review", "review"):
            with self.subTest(prefix=prefix):
                s3, cf, events, session = Mock(), Mock(), Mock(), Mock()
                s3.get_object.return_value = {"Body": io.BytesIO(content.getvalue())}
                cf.exceptions.ClientError = ClientError
                cf.describe_stacks.side_effect = [
                    ClientError({"Error": {"Code": "ValidationError",
                                          "Message": "Stack does not exist"}}, "DescribeStacks"),
                    {"Stacks": [{"StackStatus": "CREATE_COMPLETE", "Outputs": [
                        {"OutputKey": key, "OutputValue": value} for key, value in {
                            "EventRule": "review-completed", "ReconciliationRule": "review-reconcile",
                            "FunctionArn": "review-function"}.items()]}]},
                ]
                events.describe_rule.return_value = {"State": "ENABLED"}
                session.client.side_effect = {"s3": s3, "cloudformation": cf, "events": events}.__getitem__
                result = deploy_completion(session, "review", f"s3://bucket/{prefix}/",
                                           "s3://bucket/mlops/code/completion.zip")
                self.assertEqual(result["stack"], "review-completion")
                request = cf.create_stack.call_args.kwargs
                values = {item["ParameterKey"]: item["ParameterValue"] for item in request["Parameters"]}
                self.assertEqual(values["Namespace"], "review")
                self.assertEqual(values["DataPrefix"], prefix)
                self.assertEqual(values["S3Root"], f"s3://bucket/{prefix}")
                template = json.loads(request["TemplateBody"])
                self.assertIn("DataPrefix", template["Parameters"])
                self.assertIn("arn:${AWS::Partition}:s3:::${DataBucket}/${DataPrefix}/*",
                              request["TemplateBody"])
                self.assertNotIn("arn:${AWS::Partition}:s3:::${DataBucket}/${Namespace}/*",
                                 request["TemplateBody"])

    def test_mismatched_or_invalid_storage_root_stops_before_aws_access(self):
        for uri in ("s3://bucket/mlops/other", "s3://bucket", "s3://bucket/../review",
                    "s3://bucket/mlops//review", "https://bucket/mlops/review",
                    "s3://bucket/mlops/review?query"):
            session = Mock()
            with self.subTest(uri=uri), self.assertRaisesRegex(ValueError, "output root"):
                deploy_completion(session, "review", uri, "s3://bucket/code.zip")
            session.client.assert_not_called()

    def test_stale_package_is_rejected_before_infrastructure_changes(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("handler.py", "# older implementation")
        session = Mock()
        session.client.return_value.get_object.return_value = {"Body": io.BytesIO(content.getvalue())}
        with self.assertRaisesRegex(ValueError, "another revision"):
            deploy_completion(session, "test", "s3://bucket/test", "s3://bucket/package.zip")
        self.assertEqual([call.args[0] for call in session.client.call_args_list], ["s3"])

    def test_empty_uri_skips_service_deployment_but_configured_errors_stop_publication(self):
        root = Path(__file__).parents[1]
        notebook = json.loads((root / "notebooks/02_deploy_pipelines.ipynb").read_text())
        cell = next("".join(cell["source"]) for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "deploy_completion(" in "".join(cell["source"]))
        sm = Mock()
        scope = dict(uses_csv=False, sys=sys, ROOT=root, session=Mock(), namespace="test",
                     s3_root="s3://bucket/test", completion_code_uri="", tracking_uri="",
                     registry_group_prefix="", tracking_artifact_uris=[], infrastructure={}, completion_permissions_boundary_arn="", sm=sm,
                     source_uri="s3://bucket/source", verified_source_uri="s3://bucket/source",
                     identity={}, source_location=Mock(return_value="s3://bucket/source"), definitions={},
                     published_cache_sources=[], cache_source_bindings=[])
        # Run the deployment prefix up to publication; no service calls for an empty URI.
        prefix = cell.rsplit("published = {}", 1)[0]
        with patch("operations.completion.deploy.deploy_completion") as deploy:
            exec(compile(prefix, "notebook02-service-optional", "exec"), scope)
            deploy.assert_not_called()
            self.assertEqual(scope["completion"], {"deployment": "skipped"})
            scope["session"].client.assert_not_called()
            scope["completion_code_uri"] = "s3://bucket/package.zip"
            deploy.side_effect = RuntimeError("deployment denied")
            with self.assertRaisesRegex(RuntimeError, "deployment denied"):
                exec(compile(cell, "notebook02-service-required", "exec"), scope)
            deploy.assert_called_once()
        self.assertFalse(sm.mock_calls)


if __name__ == "__main__":
    unittest.main()
