"""Focused checks for the independent review's failure cases."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlops-models"))
sys.path.insert(0, str(ROOT / "tests"))

from checks import check_definitions
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND
from mlops_skeleton.names import discover_models, job_prefix, validate_model_id, validate_namespace
from mlops_models.entrypoints.evaluate import check_quality_thresholds
from mlops_models.model_base.inference.preprocess import process_inference
from mlops_models.model_base.inference.postprocess import read_predictions as read_batch_scores
from mlops_models.registry.model_package import validate_approved_model_package as validate_package
from mlops_models.pipelines.inspect import inspect_run


class ReviewHardening(unittest.TestCase):
    """Exercise naming, data isolation, bounded batch input and repeated collection."""

    def setUp(self):
        """Load the synthetic dataframe and the classification model's configuration."""
        self.config = json.loads((ROOT / f"mlops-models/src/mlops_models/models/{SECOND}/parameters.json").read_text())
        self.frame = pd.read_csv(ROOT / "examples/data/curated_fake.csv")

    def test_names_reject_late_service_failures(self):
        """Reject identifiers that Python accepts but ECR or SageMaker rejects."""
        for model in ("sample_model", "SampleModel", "échange", "a" * 25, "../usecase1"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_model_id(model)
        with self.assertRaises(ValueError):
            validate_namespace("a" * 21)

    def test_longest_job_prefix_leaves_service_suffix_budget(self):
        """Keep the longest supported namespace, model and step name below the service limit."""
        prefix = job_prefix("a" * 20, "b" * 24, "inference") + "-score"
        self.assertLessEqual(len(prefix) + 24, 63)
        self.assertNotEqual(job_prefix("example", "usecase1", "training"),
                            job_prefix("example-dev", "usecase1", "training"))



    def test_every_discovered_model_has_valid_effective_graphs(self):
        """Render both effective graphs for every model without requiring copied definitions."""
        graphs = check_definitions({"namespace": "review"})
        self.assertEqual(len([name for name in graphs if not name.endswith("/tuning")]), 2 * len(discover_models()))
        for name, steps in graphs.items():
            self.assertIn("ValidatePackage" if name.endswith("inference") else "QualityGate", steps)



    def test_quality_gate_records_missing_and_failed_metrics(self):
        """Fail configured missing or out-of-bound metrics without claiming unconfigured thresholds passed."""
        report = {"metrics": {"roc_auc": 0.6, "rmse": 10}}
        self.assertFalse(check_quality_thresholds(report, {"roc_auc": {"min": 0.8}})["passed"])
        self.assertFalse(check_quality_thresholds(report, {"missing": {"min": 0}})["passed"])
        self.assertTrue(check_quality_thresholds(report, {"rmse": {"max": 11}})["passed"])
        self.assertFalse(check_quality_thresholds(report, {})["configured"])

    def test_batch_shards_preserve_columns_for_model_owned_selection(self):
        """Write bounded shards; the predictor owns feature selection and transformations."""
        frame = self.frame.assign(unused_customer_text="not needed")
        with TemporaryDirectory() as directory:
            frame.to_csv(Path(directory) / "data.csv", index=False)
            config = {**self.config, "model_metadata": {"feature_mapping": {"value": "value"}}}
            config["model_metadata"]["feature_mapping"] = {c: c for c in frame.columns if c != config["primary_key"]}
            output, rows = process_inference(directory, directory, config, shard_rows=31)
            files = list((output / "shards").glob("*.jsonl"))
            self.assertEqual(rows, 160)
            self.assertEqual(len(files), 6)
            records = [json.loads(line) for path in files for line in path.read_text().splitlines()]
            self.assertTrue(all(set(row) == set(frame.columns) for row in records))
            self.assertEqual(len(records), 160)

    def test_batch_scores_are_read_in_bounded_chunks_with_ids(self):
        """Retain explicit identity while reading independently ordered output files."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl.out"
            path.write_text("".join(json.dumps({"row_id": str(i), "score": i / 100}) + "\n" for i in range(57)))
            chunks = list(read_batch_scores(directory, 11))
            self.assertEqual([len(chunk) for chunk in chunks], [11, 11, 11, 11, 11, 2])
            self.assertEqual([row for chunk in chunks for row in chunk["row_id"]], [str(i) for i in range(57)])

    def test_unapproved_or_substituted_package_is_rejected(self):
        """Reject a caller-selected image or checkpoint that differs from the approved registry entry."""
        client = Mock()
        package = {"ModelPackageGroupName": "review-usecase2", "ModelApprovalStatus": "Approved", "ModelPackageStatus": "Completed",
                   "InferenceSpecification": {"Containers": [{"Image": "image:1.0", "ModelDataUrl": "s3://model"}]}}
        client.describe_model_package.return_value = package
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                validate_package("arn", "review-usecase2", "image:evil", "s3://model", directory, client=client)
            package["ModelApprovalStatus"] = "PendingManualApproval"
            with self.assertRaises(ValueError):
                validate_package("arn", "review-usecase2", "image:1.0", "s3://model", directory, client=client)

    def test_approval_client_uses_the_package_region_without_environment_defaults(self):
        """Validate a package when the Processing container has no configured AWS region."""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/review-usecase2/1"
        with TemporaryDirectory() as directory, patch("boto3.client") as create_client:
            create_client.return_value.describe_model_package.return_value = {
                "ModelPackageGroupName": "review-usecase2", "ModelApprovalStatus": "Approved", "ModelPackageStatus": "Completed",
                "InferenceSpecification": {"Containers": [{"Image": "image:1.0", "ModelDataUrl": "s3://model"}]},
            }
            result = validate_package(arn, "review-usecase2", "image:1.0", "s3://model", directory)
            create_client.assert_called_once_with("sagemaker", region_name="us-east-1")
            self.assertEqual(result["approval"], "Approved")




if __name__ == "__main__":
    unittest.main()
