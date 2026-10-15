"""Exercise copied/renamed models and the dataframe prediction contract."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from mlops_models.pipelines import source
from mlops_skeleton.pipelines import walk_steps
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND


class ModelHandoff(unittest.TestCase):
    def definitions(self, **settings):
        return source.build_definitions(
            "review", "arn:aws:iam::123456789012:role/jobs",
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/model:review",
            "s3://bucket/review", "s3://bucket/review/source",
            {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": True},
            region="us-east-1", batch_runtime="processing", data_source="csv_direct",
            csv_input_uri="s3://bucket/review/data.csv", **settings,
        )

    def test_rename_copy_and_different_settings_need_no_shared_code_changes(self):
        with TemporaryDirectory() as directory:
            package = Path(directory) / "mlops_models"
            shutil.copytree(source.PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
            models = package / "models"
            shutil.rmtree(models)
            shutil.copytree(source.PACKAGE / "models" / FIRST, models / "renamed")
            shutil.copytree(source.PACKAGE / "models" / SECOND, models / SECOND)
            shutil.copytree(models / "renamed", models / "usecase3")
            config_path = models / "usecase3/parameters.json"
            config = json.loads(config_path.read_text())
            config["autogluon"]["regression"]["max_depth"] = 3
            config["target_key"] = "third_target"
            config_path.write_text(json.dumps(config))
            with patch.object(source, "PACKAGE", package):
                definitions = self.definitions()
            self.assertEqual(set(definitions), {"training", "inference"})
            self.assertNotIn(FIRST, json.dumps(definitions))
            for body in definitions.values():
                parameter = next(p for p in body["Parameters"] if p["Name"] == "ModelId")
                self.assertEqual(parameter["EnumValues"], sorted(["renamed", SECOND, "usecase3"]))
            steps = {s["Name"]: s for s in walk_steps(definitions["training"]["Steps"])}
            self.assertEqual(steps["RegisterModel"]["Arguments"]["Environment"]["MODEL_PACKAGE_GROUP"],
                             source.join({"Get": "Parameters.ModelId"}, "sm", "dev", on="-"))
            # Job identity stays shared when folders change; only selection/settings change.
            before = self.definitions()
            self.assertEqual(before["training"]["Steps"][0]["Arguments"]["ProcessingJobName"],
                             definitions["training"]["Steps"][0]["Arguments"]["ProcessingJobName"])

    def test_inference_requires_registry_and_has_no_direct_artifact_input(self):
        body = self.definitions(models=[FIRST])["inference"]
        self.assertEqual([s["Name"] for s in body["Steps"]],
                         ["ResolveModel", "Preprocess", "Score", "Postprocess"])
        self.assertEqual(body["Metadata"]["RegistryMode"], "model_registry")
        parameters = {p["Name"] for p in body["Parameters"]}
        self.assertFalse(parameters & {"ModelDataUri", "InferenceImageUri", "ModelPackageArn"})
        for step in body["Steps"][1:]:
            checkpoint = next(p for p in step["Arguments"]["ProcessingInputs"]
                              if p["InputName"] == "checkpoint")
            self.assertEqual(checkpoint["S3Input"]["S3Uri"]["Std:JsonGet"]["Path"], "checkpoint")
            self.assertEqual(step["Arguments"]["AppSpecification"]["ImageUri"]["Std:JsonGet"]["Path"], "image")

    def test_training_cannot_skip_registration_after_passing_quality(self):
        for hpo in (False, True):
            body = self.definitions(include_hpo=hpo)["training"]
            self.assertNotIn("RegisterModel", {p["Name"] for p in body["Parameters"]})
            steps = {step["Name"]: step for step in walk_steps(body["Steps"])}
            for suffix in (("", "Tuned") if hpo else ("",)):
                gate = steps["QualityGate" + suffix]["Arguments"]
                self.assertEqual(len(gate["Conditions"]), 1)
                self.assertEqual(gate["Conditions"][0]["LeftValue"]["Std:JsonGet"]["Path"], "quality.passed")
                self.assertEqual(gate["IfSteps"][0]["Type"], "Processing")
                self.assertTrue(gate["IfSteps"][0]["Arguments"]["AppSpecification"]["ContainerEntrypoint"][-1]
                                .endswith("/registry/model_package.py"))
                self.assertNotIn("CacheConfig", gate["IfSteps"][0])
                self.assertEqual(gate["ElseSteps"][0]["Type"], "Fail")

    def test_removed_registry_bypass_arguments_are_rejected(self):
        for settings in ({"registry_mode": "offline_registry"},
                         {"inference_model_uri": "s3://bucket/model.tar.gz"}):
            with self.subTest(settings=settings), self.assertRaises(TypeError):
                self.definitions(**settings)

    def test_registry_environment_and_explicit_model_subset(self):
        for environment in ("dev", "prod"):
            bodies = self.definitions(environment=environment, models=[SECOND])
            steps = {s["Name"]: s for s in walk_steps(bodies["training"]["Steps"])}
            self.assertEqual(steps["RegisterModel"]["Arguments"]["Environment"]["MODEL_PACKAGE_GROUP"],
                             source.join({"Get": "Parameters.ModelId"}, "sm", environment, on="-"))
            for body in bodies.values():
                model = next(p for p in body["Parameters"] if p["Name"] == "ModelId")
                self.assertEqual(model["EnumValues"], [SECOND])
        with self.assertRaises(ValueError):
            self.definitions(models=[])

    def test_hpo_is_hidden_by_default_and_can_be_restored(self):
        from checks import find_references
        for include_hpo in (False, True):
            with self.subTest(include_hpo=include_hpo):
                bodies = self.definitions(include_hpo=include_hpo)
                body = bodies["training"]
                steps = {s["Name"]: s for s in walk_steps(body["Steps"])}
                parameters = {p["Name"] for p in body["Parameters"]}
                self.assertEqual("Tune" in steps, include_hpo)
                self.assertEqual("TrainingMode" in steps, include_hpo)
                self.assertEqual("UseHPO" in parameters, include_hpo)
                self.assertEqual("TuningTrials" in parameters, include_hpo)
                if not include_hpo:
                    self.assertEqual(list(steps)[:5], [
                        "GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate",
                    ])
                    self.assertEqual(steps["Train"]["DependsOn"], ["Preprocess"])
                    self.assertNotIn("EvaluateTuned", steps)
                    self.assertNotIn("RegisterTunedModel", steps)
                for step in steps.values():
                    self.assertFalse(set(step.get("DependsOn", [])) - steps.keys())
                for reference in find_references(body):
                    scope, name, *_ = reference.split(".")
                    if scope == "Parameters":
                        self.assertIn(name, parameters)
                    elif scope == "Steps":
                        self.assertIn(name, steps)

    def test_ordinary_publication_needs_no_demo_tuning_fields(self):
        with TemporaryDirectory() as directory:
            package = Path(directory) / "mlops_models"
            shutil.copytree(source.PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
            for model in (FIRST, SECOND):
                path = package / "models" / model / "parameters.json"
                path.write_text(json.dumps({
                    "target_key": "custom_target", "primary_key": "custom_id",
                    "task_type": "regression", "customer_owned_option": "keep",
                }))
            with patch.object(source, "PACKAGE", package):
                self.assertEqual(set(self.definitions()), {"training", "inference"})
                with self.assertRaisesRegex(ValueError, "tuning settings"):
                    self.definitions(include_hpo=True)

    def test_renaming_changes_source_prefix_but_training_edits_do_not(self):
        identity = {"mode": "direct-deploy", "name": "review"}
        with TemporaryDirectory() as directory:
            package = Path(directory)
            train = package / "models/first/train.py"
            train.parent.mkdir(parents=True)
            train.write_text("first")
            with patch.object(source, "PACKAGE", package):
                original = source.source_location("s3://bucket/review", identity)
                train.write_text("second")
                self.assertEqual(original, source.source_location("s3://bucket/review", identity))
                train.parent.rename(package / "models/renamed")
                self.assertNotEqual(original, source.source_location("s3://bucket/review", identity))

    def test_evaluation_matches_reordered_predictions_to_identifiers(self):
        from mlops_models.model_base.training.evaluate import evaluate
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["0001", "0002", "0003"], "target": [1., 2., 3.]}).to_csv(
                root / "data.csv", index=False)
            def predict(df, checkpoint_path, *, output_dir):
                Path(output_dir).mkdir(parents=True)
                return pd.DataFrame({"id": ["0003", "0001", "0002"], "score": [3., 1., 2.]})
            report = evaluate(root / "data.csv", root / "model", root / "output",
                              {"primary_key": "id", "target_key": "target", "task_type": "regression"},
                              predict=predict)
            self.assertEqual(report["metrics"]["rmse"], 0)
            saved = pd.read_csv(root / "output/predictions.csv", dtype={"id": str})
            self.assertEqual(saved["id"].tolist(), ["0001", "0002", "0003"])
            self.assertEqual(saved["prediction"].tolist(), saved["target"].tolist())

    def test_http_adapter_preserves_dataframe_identifiers(self):
        from mlops_models.entrypoints.serve import score_request
        class Predictor:
            def predict_with_best_model(self, df):
                return pd.DataFrame({"customer_id": df["customer_id"], "score": df["x"] * 2})
        response = score_request("custom", Predictor(), '{"customer_id":"0001","x":3}\n')
        self.assertEqual(json.loads(response.get_data(as_text=True)),
                         {"customer_id": "0001", "score": 6})

    def test_wrong_model_artifact_fails_before_reading_scoring_data(self):
        from mlops_models.entrypoints.preprocess_inference import execute
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parameters.json").write_text('{"model_id": "othermodel"}')
            with self.assertRaisesRegex(ValueError, "Artifact belongs to othermodel"):
                execute(FIRST, root / "unused-input", root / "unused-output", model_dir=root)
            self.assertFalse((root / "unused-output").exists())


if __name__ == "__main__":
    unittest.main()
