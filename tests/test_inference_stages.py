"""Verify independent acquisition, cached producer paths and customer file outputs."""

from importlib import import_module
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from mlops_models.entrypoints import preprocess_inference, postprocess
from mlops_models.model_base.inference import predict
from mlops_models.pipelines import source


class InferenceStages(unittest.TestCase):
    def definitions(self, **settings):
        return source.build_definitions(
            "review", "arn:aws:iam::123456789012:role/jobs", "image:tag",
            "s3://bucket/review", "s3://bucket/review/source",
            {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": True},
            region="us-east-1", batch_runtime="processing", model_code_source="pipeline",
            csv_input_uri="s3://bucket/scoring.csv",
            feature_store_config_uri="s3://bucket/features.json",
            step_cache_revisions={name: name for name in (
                "GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate",
                "BatchGetData", "BatchPreprocess", "Score", "Postprocess")},
            **settings,
        )

    def test_training_store_and_scoring_csv_have_independent_readers(self):
        definitions = self.definitions(data_source="feature_store", inference_data_source="csv_direct")
        train = definitions["training"]["Steps"][0]
        steps = {step["Name"]: step for step in definitions["inference"]["Steps"]}
        self.assertEqual(list(steps), ["ResolveModel", "GetData", "Preprocess", "Score", "Postprocess"])
        self.assertNotIn("CacheConfig", train)
        self.assertEqual(train["Arguments"]["Environment"]["MLOPS_DATA_SOURCE"], "feature_store")
        for name in ("GetData", "Preprocess", "Score", "Postprocess"):
            self.assertTrue(steps[name]["CacheConfig"]["Enabled"])
        reader = steps["GetData"]["Arguments"]
        self.assertEqual(reader["Environment"]["MLOPS_DATA_SOURCE"], "csv_direct")
        self.assertNotIn("MLOPS_FEATURE_STORE_CONFIG_URI", reader["Environment"])
        for name in ("Preprocess", "Score", "Postprocess"):
            self.assertNotIn("MLOPS_DATA_SOURCE", steps[name]["Arguments"]["Environment"])

    def test_downstream_mounts_reference_actual_producers_after_cache_hits(self):
        steps = {s["Name"]: s for s in self.definitions(data_source="csv_direct")["inference"]["Steps"]}
        inputs = lambda name: {i["S3Input"]["LocalPath"]: i["S3Input"]["S3Uri"]
                               for i in steps[name]["Arguments"]["ProcessingInputs"]}
        raw = {"Get": "Steps.GetData.ProcessingOutputConfig.Outputs['raw'].S3Output.S3Uri"}
        self.assertEqual(inputs("Preprocess")["/opt/ml/processing/raw"], raw)
        self.assertEqual(inputs("Postprocess")["/opt/ml/processing/raw"], raw)
        self.assertEqual(inputs("Postprocess")["/opt/ml/processing/selection"], {
            "Get": "Steps.Preprocess.ProcessingOutputConfig.Outputs['selection'].S3Output.S3Uri"})
        self.assertEqual(inputs("Postprocess")["/opt/ml/processing/input"], {
            "Get": "Steps.Score.ProcessingOutputConfig.Outputs['scores'].S3Output.S3Uri"})
        self.assertEqual(inputs("Score")["/opt/ml/processing/prepared"], {
            "Get": "Steps.Preprocess.ProcessingOutputConfig.Outputs['prepared'].S3Output.S3Uri"})

    def test_feature_store_is_only_queried_in_get_data(self):
        steps = {s["Name"]: s for s in self.definitions(data_source="feature_store")["inference"]["Steps"]}
        self.assertNotIn("CacheConfig", steps["GetData"])
        self.assertTrue(steps["Preprocess"]["CacheConfig"]["Enabled"])
        self.assertNotIn("MLOPS_FEATURE_STORE_CONFIG_URI", steps["Preprocess"]["Arguments"]["Environment"])

    def test_postprocess_edits_keep_acquisition_preparation_and_score_fingerprints(self):
        with TemporaryDirectory() as directory:
            package = Path(directory) / "mlops_models"
            shutil.copytree(source.PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
            csv = Path(directory) / "data.csv"
            csv.write_text("id,value\n0001,1\n")
            with patch.object(source, "PACKAGE", package):
                before = source.pipeline_fingerprints(Mock(), local_csv=csv)
                file = package / "model_base/inference/postprocess.py"
                file.write_text(file.read_text() + "\n# analysis change\n")
                after = source.pipeline_fingerprints(Mock(), local_csv=csv)
            for name in ("BatchGetData", "BatchPreprocess", "Score"):
                self.assertEqual(before[name], after[name])
            self.assertNotEqual(before["Postprocess"], after["Postprocess"])

    def test_entrypoint_calls_customer_preprocess_before_any_primary_key_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "parameters.json").write_text('{"primary_key":"UPPER_ID"}')
            (model / "model_metadata.json").write_text('{"feature_mapping":{"value":"VALUE"}}')
            (root / "data.csv").write_text("lower_id,value\n0001,2\n")
            module = import_module("mlops_models.models.usecase1.inference.preprocess")
            def process(data_path, output, config, *, shard_rows):
                self.assertEqual((data_path / "data.csv").read_text(), "lower_id,value\n0001,2\n")
                self.assertEqual(config["model_metadata"]["feature_mapping"], {"value": "VALUE"})
                return output, 1
            with patch.object(module, "process_inference", side_effect=process):
                preprocess_inference.execute("usecase1", root, root / "output", model_dir=model)

    def test_prediction_files_and_analysis_preserve_all_three_variable_pairs(self):
        for task, column in (("regression", "prediction"), ("classification", "score")):
            with self.subTest(task=task), TemporaryDirectory() as directory:
                root = Path(directory)
                raw, prepared, model = root / "raw", root / "prepared", root / "model"
                for path in (raw, prepared, model):
                    path.mkdir()
                (raw / "data.csv").write_text("id,value,extra\n0001,2,retain-for-analysis\n")
                (prepared / "data.csv").write_text("id,value\n0001,2\n")
                selection = root / "selected_features.csv"
                selection.write_text("features,description\nvalue,value\n")
                (model / "parameters.json").write_text(json.dumps({"primary_key": "id", "task_type": task}))
                (model / "model_metadata.json").write_text('{"feature_mapping":{"value":"value"}}')
                scores = pd.DataFrame({"id": ["0001"], column: [0.8]})
                messaging = pd.DataFrame({"id": ["0001"], **{
                    key: [value] for n in range(1, 4)
                    for key, value in ((f"variable_{n}", f"value{n}"), (f"variable_{n}_value", n))}})
                final = messaging.merge(scores, on="id")
                class CustomerPredictor:
                    def __init__(self, checkpoint_path):
                        self.checkpoint_path = checkpoint_path
                    def predict_file(self, path):
                        if path != prepared / "data.csv":
                            raise AssertionError(path)
                        return scores, messaging, final
                output = predict.predict_with_model(prepared, model, root / "scores", predictor_cls=CustomerPredictor)
                self.assertEqual(pd.read_csv(output / "messaging_final.csv").columns.tolist(), final.columns.tolist())
                result = postprocess.execute("usecase1", output, root / "analysis", model_dir=model,
                                             data_directory=raw, selection_file=selection)
                data = pd.read_csv(result, dtype={"id": str})
                self.assertEqual(data["id"].tolist(), ["0001"])
                for n in range(1, 4):
                    self.assertEqual(data[f"variable_{n}_value"].tolist(), [n])
                self.assertEqual(result.name, "prediction_analysis.csv")
                (output / "messaging.csv").unlink()
                with self.assertRaises(FileNotFoundError):
                    postprocess.execute("usecase1", output, root / "missing", model_dir=model,
                                        data_directory=raw, selection_file=selection)


if __name__ == "__main__":
    unittest.main()
