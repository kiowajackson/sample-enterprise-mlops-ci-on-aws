"""Native graph wiring uses existing producers and remains reusable on a second call."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from botocore.session import Session
from botocore.validate import validate_parameters
from mlops_models.pipelines import lineage
from test_execution_tracking import S3

BASE = "arn:aws:sagemaker:us-east-1:123456789012"
EXECUTION = BASE + ":pipeline/demo-training/execution/new"
PRODUCER = BASE + ":pipeline/demo-training/execution/old"


class Native:
    def __init__(self):
        self.artifacts, self.components, self.edges = {}, {}, []
        self.api = Session().get_service_model("sagemaker")

    def list_artifacts(self, SourceUri):
        return {"ArtifactSummaries": [item for item in self.artifacts.values() if item["Source"]["SourceUri"] == SourceUri]}

    def create_artifact(self, **request):
        validate_parameters(request, self.api.operation_model("CreateArtifact").input_shape)
        arn = BASE + ":artifact/" + request["ArtifactName"]
        self.artifacts[arn] = {**request, "ArtifactArn": arn}
        return {"ArtifactArn": arn}

    def list_trial_components(self, **request):
        validate_parameters(request, self.api.operation_model("ListTrialComponents").input_shape)
        return {"TrialComponentSummaries": self.components.get(request["SourceArn"], [])}

    def list_associations(self, SourceArn, AssociationType):
        return {"AssociationSummaries": [item for item in self.edges if
                item["SourceArn"] == SourceArn and item["AssociationType"] == AssociationType]}

    def add_association(self, **request):
        validate_parameters(request, self.api.operation_model("AddAssociation").input_shape)
        self.edges.append(request)
        return request

    def query_lineage(self, **request):
        validate_parameters(request, self.api.operation_model("QueryLineage").input_shape)
        return {"Vertices": [], "Edges": deepcopy(self.edges)}


class NativeLineage(unittest.TestCase):
    def setUp(self):
        self.sm, self.s3 = Native(), S3()
        self.session = SimpleNamespace(client=lambda name, **kwargs: self.sm if name == "sagemaker" else self.s3)
        self.raw = "s3://bucket/models/a/runs/old/raw"
        self.model = "s3://bucket/model.tar.gz"
        self.package = BASE + ":model-package/a/1"
        self.get_job = BASE + ":processing-job/get-old"
        self.train_job = BASE + ":training-job/train-new"
        self.report = {
            "execution": {"PipelineExecutionArn": EXECUTION},
            "parameters": {"OutputDataRoot": "s3://bucket/models", "ModelId": "a"},
            "steps": [
                {"name": "GetData", "status": "Succeeded", "job_arn": self.get_job,
                 "job_type": "ProcessingJob", "producer_execution_arn": PRODUCER, "outputs": {"raw": self.raw},
                 "job": {"ProcessingInputs": [{"InputName": "input", "S3Input": {"S3Uri": "s3://bucket/input.csv"}}]}},
                {"name": "Train", "status": "Succeeded", "job_arn": self.train_job,
                 "job_type": "TrainingJob", "producer_execution_arn": EXECUTION, "model_artifact": self.model,
                 "job": {"InputDataConfig": [{"DataSource": {"S3DataSource": {"S3Uri": self.raw + "/data.csv"}}}]}},
                {"name": "RegisterModel", "status": "Succeeded", "job_arn": BASE + ":processing-job/register",
                 "job_type": "ProcessingJob", "producer_execution_arn": EXECUTION,
                 "outputs": {"registry": "s3://bucket/registry"}, "job": {"ProcessingInputs": []}},
                {"name": "FailedExtra", "status": "Failed", "job_arn": BASE + ":processing-job/failed",
                 "job_type": "ProcessingJob", "producer_execution_arn": EXECUTION,
                 "outputs": {"raw": "s3://bucket/not-produced"}, "job": {}},
            ],
            "datasets": [{"uri": self.raw + "/data.csv", "producer_execution_arn": PRODUCER,
                          "provenance": {"queries": [{"group": "features", "query_id": "query"}]}}],
            "registry": [{"package": {"ModelPackageArn": self.package,
                                     "InferenceSpecification": {"Containers": [{"ModelDataUrl": self.model}]}}}],
        }
        for step in self.report["steps"]:
            self.sm.components[step["job_arn"]] = [{
                "TrialComponentArn": BASE + ":experiment-trial-component/" + step["job_arn"].rsplit("/", 1)[-1]
            }]

    def publish(self):
        with patch.object(lineage, "collect_execution", return_value=deepcopy(self.report)):
            return lineage.publish_lineage(self.session, EXECUTION, prefix="test")

    def test_graph_links_actual_cached_producer_and_model_without_rereading_data(self):
        first = self.publish()
        count = len(self.sm.edges)
        second = self.publish()
        self.assertEqual(len(self.sm.edges), count)
        self.assertFalse(any(second["created"].values()))
        self.assertEqual(first["jobs"][self.get_job],
                         self.sm.components[self.get_job][0]["TrialComponentArn"])
        self.assertNotIn("s3://bucket/not-produced", first["artifacts"])
        self.assertIn({"SourceArn": first["artifacts"][self.raw],
                       "DestinationArn": first["jobs"][self.train_job],
                       "AssociationType": "ContributedTo"}, first["associations"])
        group = first["artifacts"][BASE + ":feature-group/features"]
        self.assertIn({"SourceArn": group, "DestinationArn": first["jobs"][self.get_job],
                       "AssociationType": "ContributedTo"}, first["associations"])
        self.assertIn({"SourceArn": first["jobs"][BASE + ":processing-job/register"],
                       "DestinationArn": first["artifacts"][self.package],
                       "AssociationType": "Produced"}, first["associations"])
        graph = lineage.query_lineage(self.session, first["package_roots"][0])
        self.assertEqual(len(graph["Edges"]), count)
        self.assertEqual(len(self.s3.objects), 1)  # Only its receipt, no dataset reads/writes.

    def test_missing_or_ambiguous_native_job_fails_before_metadata_writes(self):
        component = self.sm.components[self.train_job][0]
        for matches in ([], [component, component]):
            with self.subTest(matches=matches):
                self.sm.components[self.train_job] = matches
                with self.assertRaisesRegex(ValueError, "Expected one native lineage component"):
                    self.publish()
                self.assertFalse(self.sm.artifacts)
                self.assertFalse(self.sm.edges)
                self.assertFalse(self.s3.objects)
