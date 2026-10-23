"""Exercise native completion events, duplicate delivery and a failed tracking job."""

from copy import deepcopy
from datetime import datetime, timezone, timedelta
import importlib.util
import json
from pathlib import Path
import unittest

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("completion", ROOT / "operations/completion/handler.py")
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)
spec = importlib.util.spec_from_file_location("completion_template", ROOT / "operations/completion/template.py")
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)
PIPELINE = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test-training"
ARN = PIPELINE + "/execution/one"
ROLE = "arn:aws:iam::123456789012:role/job"


class SageMaker:
    def __init__(self):
        self.status = "Succeeded"
        self.modified = datetime(2026, 10, 23, tzinfo=timezone.utc)
        self.jobs = {}
        self.definition = {"Steps": [{"Name": "StartTracking", "Arguments": {
            "ProcessingJobName": "tracking-start", "RoleArn": ROLE,
            "AppSpecification": {"ImageUri": "image@sha256:abc", "ContainerEntrypoint": [
                "python", "-m", "mlops_models.pipelines.tracking", "--phase", "start"]},
            "Environment": {"PIPELINE_EXECUTION_ARN": {"Get": "Execution.PipelineExecutionArn"}},
            "ProcessingInputs": [{"InputName": "source", "S3Input": {"S3Uri": "s3://bucket/original-source"}}],
        }}]}

    def describe_pipeline_execution(self, **kwargs):
        return {"PipelineExecutionStatus": self.status, "LastModifiedTime": self.modified}

    def describe_pipeline_definition_for_execution(self, **kwargs):
        return {"PipelineDefinition": json.dumps(self.definition)}

    def describe_processing_job(self, ProcessingJobName):
        if ProcessingJobName not in self.jobs:
            raise ClientError({"Error": {"Code": "ValidationException", "Message": "Could not find job"}},
                              "DescribeProcessingJob")
        return self.jobs[ProcessingJobName]

    def create_processing_job(self, **request):
        arn = PIPELINE.split(":pipeline")[0] + ":processing-job/" + request["ProcessingJobName"]
        self.jobs[request["ProcessingJobName"]] = {**request, "ProcessingJobArn": arn,
                                                  "ProcessingJobStatus": "InProgress"}
        return {"ProcessingJobArn": arn}


class Completion(unittest.TestCase):
    def setUp(self):
        self.sm = SageMaker()
        self.event = {"source": "aws.sagemaker", "detail-type": completion.PIPELINE_EVENT,
                      "detail": {"pipelineExecutionArn": ARN, "currentPipelineExecutionStatus": "Failed"}}

    def dispatch(self, event=None):
        return completion.dispatch(event or self.event, self.sm, pipeline_arns=[PIPELINE], job_roles=[ROLE])

    def test_uses_actual_execution_definition_and_reuses_duplicate(self):
        first = self.dispatch()
        second = self.dispatch()
        self.assertEqual(first["job_arn"], second["job_arn"])
        self.assertEqual(len(self.sm.jobs), 1)
        request = next(iter(self.sm.jobs.values()))
        self.assertEqual(request["Environment"]["PIPELINE_EXECUTION_ARN"], ARN)
        self.assertEqual(request["AppSpecification"]["ContainerEntrypoint"][-1], "sync")
        self.assertEqual(request["ProcessingInputs"][0]["S3Input"]["S3Uri"], "s3://bucket/original-source")
        self.assertEqual(self.sm.definition["Steps"][0]["Arguments"]["AppSpecification"][
            "ContainerEntrypoint"][-1], "start")

    def test_stale_event_during_retry_does_not_finalize_and_new_terminal_revision_runs(self):
        self.sm.status = "Executing"
        self.assertIn("ignored", self.dispatch())
        self.assertFalse(self.sm.jobs)
        self.sm.status = "Failed"
        first = self.dispatch()
        self.sm.status = "Succeeded"
        self.sm.modified += timedelta(minutes=1)
        self.assertNotEqual(self.dispatch()["job_arn"], first["job_arn"])

    def test_processing_failure_gets_one_retry_and_manual_replay_is_distinct(self):
        self.event["replay_id"] = "operator-1"
        self.dispatch()
        name = next(iter(self.sm.jobs))
        event = {"source": "aws.sagemaker", "detail-type": completion.PROCESSING_EVENT,
                 "detail": {"ProcessingJobName": name, "ProcessingJobStatus": "Failed"}}
        retry = self.dispatch(event)
        self.assertEqual(retry["attempt"], 1)
        self.assertTrue(self.dispatch(event)["reused"])
        retry_name = retry["job_arn"].rsplit("/", 1)[-1]
        self.assertEqual(self.sm.jobs[retry_name]["Environment"]["MLOPS_SYNC_REPLAY_ID"], "operator-1")
        event["detail"]["ProcessingJobName"] = retry_name
        with self.assertRaisesRegex(RuntimeError, "replay"):
            self.dispatch(event)
        self.event["replay_id"] = "operator-2"
        self.assertNotEqual(self.dispatch()["job_arn"], retry["job_arn"])

    def test_tracking_disabled_and_outside_scope(self):
        self.sm.definition["Steps"] = []
        self.assertEqual(self.dispatch()["ignored"], "tracking disabled for this execution")
        self.event["detail"]["pipelineExecutionArn"] = ARN.replace("test-training", "unrelated")
        with self.assertRaisesRegex(ValueError, "scope"):
            self.dispatch()

    def test_platform_template_has_scoped_rules_and_existing_failure_destination(self):
        result = render.template("demo-completion", [PIPELINE], [ROLE],
                                 failure_destination="arn:aws:sns:us-east-1:123456789012:existing")
        resources = result["Resources"]
        self.assertEqual(resources["PipelineRule"]["Properties"]["EventPattern"]["detail"]["pipelineArn"], [PIPELINE])
        self.assertEqual(resources["SyncFailureRule"]["Properties"]["EventPattern"]["detail"][
            "ProcessingJobName"], [{"prefix": "demo-completion-sync-"}])
        self.assertEqual(resources["AsyncFailures"]["Properties"]["DestinationConfig"]["OnFailure"]["Destination"],
                         "arn:aws:sns:us-east-1:123456789012:existing")
        self.assertNotIn("AWS::SQS::Queue", json.dumps(result))
        self.assertNotIn("AWS::SNS::Topic", json.dumps(result))
