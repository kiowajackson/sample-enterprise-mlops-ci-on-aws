"""Exercise saved training scores through packaging, Evaluate and Registry handoff."""

import json
import io
from contextlib import redirect_stdout
import hashlib
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mlops_models.entrypoints import evaluate as entrypoint
from mlops_models.frameworks.autogluon import evaluation_report
from mlops_models.model_base.training.train import write_container_output
from mlops_models.model_base.training.evaluate import evaluate
from mlops_models.pipelines.tracking import report_metrics
from mlops_models.registry.model_package import register_existing_group
from model_fixtures import REGRESSION_MODEL, CLASSIFICATION_MODEL


class SavedTrainingMetrics(unittest.TestCase):
    def report(self, task="regression"):
        config = {"task_type": task, "model_id": (
            REGRESSION_MODEL if task == "regression" else CLASSIFICATION_MODEL)}
        performance = ({"root_mean_squared_error": np.float64(-2.5),
                        "mean_absolute_error": -1.5, "r2": -0.2}
                       if task == "regression" else
                       {"roc_auc": 0.85, "accuracy": 0.8, "log_loss": -0.4})
        return evaluation_report(performance, config, "validation")

    def test_regression_preserves_raw_scores_and_negative_r2(self):
        report = self.report()
        self.assertEqual(report["metrics"], {"rmse": 2.5, "mae": 1.5, "r2": -0.2})
        self.assertEqual(report["raw_metrics"]["root_mean_squared_error"], -2.5)
        self.assertEqual(report["data_usage"], "validation")
        json.dumps(report, allow_nan=False)

    def test_classification_keeps_auc_and_converts_log_loss(self):
        report = self.report("classification")
        self.assertEqual(report["metrics"], {"roc_auc": 0.85, "accuracy": 0.8, "log_loss": 0.4})
        self.assertEqual(report["raw_metrics"]["log_loss"], -0.4)

    def test_demo_report_rejects_bad_performance(self):
        for performance in ({}, None, {"r2": float("nan")}, {"r2": float("inf")}, {"r2": True}):
            with self.subTest(performance=performance), self.assertRaises(ValueError):
                evaluation_report(
                    performance, {"model_id": REGRESSION_MODEL, "task_type": "regression"}, "validation")

    def test_report_requires_explicit_data_usage_and_known_autogluon_metric(self):
        config = {"model_id": REGRESSION_MODEL, "task_type": "regression"}
        with self.assertRaisesRegex(ValueError, "scored data"):
            evaluation_report({"r2": 0.8}, config, "")
        with self.assertRaises(ValueError):
            evaluation_report({"invented_metric": 0.8}, config, "validation")

    def test_models_with_or_without_scores_reach_registry_without_rescoring(self):
        for task, model_id, has_scores in (("regression", REGRESSION_MODEL, True),
                                          ("classification", CLASSIFICATION_MODEL, True),
                                          ("regression", REGRESSION_MODEL, False)):
            with self.subTest(task=task, has_scores=has_scores), TemporaryDirectory() as directory:
                root = Path(directory)
                working = root / "working"
                (working / "code").mkdir(parents=True)
                (working / "opaque-model.bin").write_bytes(b"not a loadable predictor")
                native_report = b"Native evaluation report; no assumed parsing format.\n"
                (working / "valid_performance.txt").write_bytes(native_report)
                (working / "code/source.json").write_text(json.dumps({"commit": "a" * 40, "dirty": False,
                    "uri": "s3://model-bucket/source/training/", "sha256": "d" * 64}))
                raw = self.report(task)["raw_metrics"]
                if has_scores:
                    (working / "evaluation.json").write_text(json.dumps(self.report(task)))
                package = write_container_output(
                    working, {"selected_model": "chosen-model"}, root / "package", root / "outputs",
                    parameters={"model_id": model_id, "task_type": task},
                )
                saved_report = (package / "evaluation.json").read_bytes() if has_scores else None
                self.assertFalse(working.exists())
                self.assertFalse((package / "evaluation/data.csv").exists())
                # Evaluate must not depend on the old package parameter or dataset contract.
                (package / "parameters.json").unlink()
                archive_path = root / "model.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    for path in package.rglob("*"):
                        if path.is_file():
                            archive.add(path, arcname=str(path.relative_to(package)))
                checkpoint_uri = "s3://model-bucket/run/model/application/model.tar.gz"
                with patch.dict(os.environ, {"CHECKPOINT_URI": checkpoint_uri, "SOURCE_COMMIT": "b" * 40}), \
                     patch("autogluon.tabular.TabularPredictor.load",
                           side_effect=AssertionError("Evaluate must not load the predictor")):
                    report = entrypoint.execute(model_id, archive_path, root / "evaluation", input_path=None)
                self.assertEqual(report["metrics"], self.report(task)["metrics"] if has_scores else {})
                if has_scores:
                    self.assertEqual(report["raw_metrics"], raw)
                    self.assertEqual((package / "evaluation.json").read_bytes(), saved_report)
                else:
                    self.assertNotIn("raw_metrics", report)
                    self.assertNotIn("data_usage", report)
                    self.assertFalse((package / "evaluation.json").exists())
                    self.assertEqual(report_metrics("Evaluate", {"evaluation.json": report}), {})
                self.assertEqual((package / "valid_performance.txt").read_bytes(), native_report)
                self.assertEqual(report["checkpoint_uri"], checkpoint_uri)
                self.assertEqual(report["model_source_commit"], "a" * 40)
                self.assertEqual(report["model_source_dirty"], "false")
                self.assertEqual(report["evaluation_source_commit"], "b" * 40)
                self.assertTrue(report["quality"]["passed"])
                self.assertFalse(report["quality"]["configured"])
                self.assertFalse((root / "evaluation/unpacked").exists())
                self.assertEqual({p.name for p in (root / "evaluation").iterdir()}, {"evaluation.json", "model_metadata.json"})
                self.assertEqual((root / "evaluation/model_metadata.json").read_bytes(), (package / "model_metadata.json").read_bytes())

                group = f"platform-{model_id}-sm-dev"
                group_arn = f"arn:aws:sagemaker:us-east-1:123456789012:model-package-group/{group}"
                package_arn = f"arn:aws:sagemaker:us-east-1:123456789012:model-package/{group}/1"
                client, s3 = Mock(), Mock()
                client.describe_model_package_group.return_value = {"ModelPackageGroupArn": group_arn}
                client.create_model_package.return_value = {"ModelPackageArn": package_arn}
                client.describe_model_package.return_value = {"ModelPackageStatus": "Completed"}
                settings = {
                    "MODEL_PACKAGE_GROUP": group,
                    "IMAGE_URI": "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "c" * 64,
                    "PIPELINE_EXECUTION_ARN": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/demo/execution/run",
                    "DEPLOYMENT_NAMESPACE": "reviewer-a", "ENVIRONMENT": "dev",
                    "MODEL_ID": model_id, "EVALUATION_URI": "s3://model-bucket/run/reports/evaluation/evaluation.json",
                    "REGISTRATION_OUTPUT_URI": "s3://model-bucket/run/reports/registry",
                    "SOURCE_URI": "s3://model-bucket/source/source.tar.gz", "DEPLOY_MODE": "direct-deploy",
                }
                s3.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO((root / "evaluation/model_metadata.json").read_bytes())}
                receipt = register_existing_group(report, settings, client=client, s3=s3)
                request = client.create_model_package.call_args.kwargs
                self.assertEqual(request["ModelApprovalStatus"], "PendingManualApproval")
                self.assertEqual(request["InferenceSpecification"]["Containers"][0]["ModelDataUrl"], checkpoint_uri)
                self.assertEqual(request["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"],
                                 settings["EVALUATION_URI"])
                self.assertEqual(receipt["checkpoint"], checkpoint_uri)
                self.assertEqual(request["CustomerMetadataProperties"]["ModelMetadataUri"], "s3://model-bucket/run/reports/evaluation/model_metadata.json")
                self.assertEqual(request["CustomerMetadataProperties"]["ModelMetadataSha256"], report["model_metadata_sha256"])
                s3.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(b"changed")}
                client.create_model_package.reset_mock()
                with self.assertRaisesRegex(ValueError, "differs"):
                    register_existing_group(report, settings, client=client, s3=s3)
                client.create_model_package.assert_not_called()
                self.assertEqual(json.loads(s3.put_object.call_args.kwargs["Body"])["package"], package_arn)

    def test_current_use_case_thresholds_gate_saved_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            (package / "evaluation.json").write_text(json.dumps(self.report()))
            (package / "model_metadata.json").write_text("{}")
            source = root / "source"
            model_folder = source / "models" / REGRESSION_MODEL
            model_folder.mkdir(parents=True)
            parameters = model_folder / "parameters.json"
            for maximum, expected in ((3.0, True), (2.0, False)):
                parameters.write_text(json.dumps({"task_type": "regression",
                                                  "quality_gate": {"rmse": {"max": maximum}}}))
                with patch.object(entrypoint, "__file__", str(source / "entrypoints/evaluate.py")):
                    report = entrypoint.execute(
                        REGRESSION_MODEL, package, root / "output", input_path=None)
                self.assertEqual(report["quality"]["passed"], expected)
                self.assertEqual(report["quality"]["checks"][0]["value"], 2.5)

    def test_missing_or_empty_scores_pass_through_without_using_metadata_as_metrics(self):
        for saved in (None, {}, {"model_id": REGRESSION_MODEL, "task": "regression"},
                      {"model_id": REGRESSION_MODEL, "task": "regression", "metrics": {}}):
            with self.subTest(saved=saved), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "model_metadata.json").write_text('{"metrics": {"r2": 1}}')
                if saved is not None:
                    (root / "evaluation.json").write_text(json.dumps(saved))
                report = entrypoint.execute(REGRESSION_MODEL, root, root / "output", input_path=None)
                self.assertEqual(report["model_id"], REGRESSION_MODEL)
                self.assertEqual(report["task"], "regression")
                self.assertEqual(report["metrics"], {})
                self.assertEqual(report["quality"], {"configured": False, "passed": True, "checks": []})
                self.assertEqual(report_metrics("Evaluate", {"evaluation.json": report}), {})
                self.assertTrue((root / "output/model_metadata.json").is_file())
                self.assertEqual(json.loads((root / "output/evaluation.json").read_text()), report)

    def test_missing_scores_do_not_pass_a_configured_quality_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model_metadata.json").write_text("{}")
            source = root / "source"
            model_folder = source / "models" / REGRESSION_MODEL
            model_folder.mkdir(parents=True)
            (model_folder / "parameters.json").write_text(json.dumps({
                "task_type": "regression", "quality_gate": {"rmse": {"max": 3}},
            }))
            with patch.object(entrypoint, "__file__", str(source / "entrypoints/evaluate.py")):
                report = entrypoint.execute(REGRESSION_MODEL, model, root / "output", input_path=None)
            self.assertTrue(report["quality"]["configured"])
            self.assertFalse(report["quality"]["passed"])
            self.assertEqual(report["quality"]["checks"], [
                {"metric": "rmse", "value": None, "limits": {"max": 3}, "passed": False}])

    def test_effective_parameters_and_training_provenance_are_copied_from_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "evaluation.json").write_text(json.dumps(self.report()))
            (model / "model_metadata.json").write_text("{}")
            parameters = b'{"task_type":"regression","num_cpus":17,"fitted_setting":"retained"}\n'
            (model / "parameters.json").write_bytes(parameters)
            (model / "_execution.json").write_text('{"source":{"commit":"training-source"}}')
            report = entrypoint.execute(REGRESSION_MODEL, model, root / "output", input_path=None)
            self.assertTrue(report["effective_parameters_available"])
            self.assertEqual((root / "output/parameters.json").read_bytes(), parameters)
            self.assertEqual((root / "output/training_execution.json").read_bytes(),
                             (model / "_execution.json").read_bytes())

    def test_wrong_model_task_or_bad_metric_stops_publication(self):
        for change in ({"model_id": "wrong"}, {"task": "classification"},
                       {"metrics": []}, {"metrics": {"r2": float("nan")}},
                       {"metrics": {"r2": True}}):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "evaluation.json").write_text(json.dumps({**self.report(), **change}))
                with self.assertRaises(ValueError):
                    evaluate(None, root, root / "output",
                             {"model_id": REGRESSION_MODEL, "task_type": "regression"})
                self.assertFalse((root / "output").exists())

    def test_diagnostics_identify_missing_data_usage_or_metrics_in_exact_saved_archive(self):
        for change, expected in (({"data_usage": ""}, "data_usage is missing"),
                                 ({"metrics": []}, "metrics must be a dictionary")):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                model = root / "model"
                model.mkdir()
                saved = {**self.report(), **change}
                (model / "evaluation.json").write_text(json.dumps(saved))
                archive_path = root / "model.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    archive.add(model / "evaluation.json", arcname="evaluation.json")
                logs = io.StringIO()
                with redirect_stdout(logs), self.assertRaisesRegex(ValueError, expected):
                    entrypoint.execute(REGRESSION_MODEL, archive_path, root / "output", input_path=None)
                events = [json.loads(line) for line in logs.getvalue().splitlines()]
                received = next(event for event in events if event["event"] == "event_received")
                loaded = next(event for event in events if event["event"] == "saved_evaluation_loaded")
                self.assertEqual(received["model_artifact"], str(archive_path))
                self.assertEqual(loaded["model_id"], saved["model_id"])
                self.assertEqual(loaded["task"], saved["task"])
                self.assertEqual(loaded["data_usage"], saved["data_usage"])
                self.assertEqual(loaded["metric_names"], list(saved["metrics"]))
                self.assertEqual(Path(loaded["path"]).name, "evaluation.json")
                self.assertFalse(Path(loaded["path"]).exists(), "Unpacked input must be cleaned on failure.")
                self.assertTrue(archive_path.is_file(), "The original model archive must remain intact.")
                self.assertFalse((root / "output/evaluation.json").exists())

    def test_explicit_dataset_is_rejected_by_saved_report_mode(self):
        with self.assertRaisesRegex(ValueError, "do not pass"):
            evaluate("data.csv", "checkpoint", "output", {})


if __name__ == "__main__":
    unittest.main()
