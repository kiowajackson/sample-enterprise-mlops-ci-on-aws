"""Exercise the registered metadata and file-based prediction handoff."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from mlops_models.entrypoints import get_data, predict, preprocess_inference, postprocess
from mlops_models.model_base.inference import predict as base
from mlops_models.runtime.load_source import prepare_inference_inputs
from model_fixtures import REGRESSION_MODEL


class InferenceMetadata(unittest.TestCase):
    def artifact(self, root, *, task="regression", predictor_path=None):
        model = root / "model"
        model.mkdir()
        (model / "parameters.json").write_text(json.dumps({
            "model_id": REGRESSION_MODEL, "primary_key": "id", "task_type": task,
        }))
        metadata = {"feature_mapping": {"raw_value": "VALUE"}}
        if predictor_path is not None:
            metadata["predictor_path"] = predictor_path
        (model / "model_metadata.json").write_text(json.dumps(metadata))
        return model

    def load_model(self, model):
        return base.ModelPredictor(
            model, json.loads((model / "model_metadata.json").read_text()),
            json.loads((model / "parameters.json").read_text()))

    def predictor(self):
        result = Mock()
        result.features.return_value = ["VALUE"]
        result.predict.side_effect = lambda df: df["VALUE"] * 2
        return result

    def test_raw_csv_to_final_predictions_uses_exact_registered_mapping(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MLOPS_DATA_SOURCE": "csv_direct",
        }):
            root = Path(directory)
            model = self.artifact(root)
            parameter_path = root / f"source/models/{REGRESSION_MODEL}/parameters.json"
            parameter_path.parent.mkdir(parents=True)
            parameter_path.write_bytes((model / "parameters.json").read_bytes())
            raw = root / "input.csv"
            raw.write_text("id,raw_value,extra\n0001,3,keep\n0002,4,NA\n0003,5,text\n")
            raw = get_data.execute(REGRESSION_MODEL, raw, root / "acquired", model_dir=model)
            selection = prepare_inference_inputs(model, root / "prepared", REGRESSION_MODEL)
            with patch.object(preprocess_inference, "__file__", str(root / "source/entrypoints/preprocess_inference.py")):
                result = preprocess_inference.execute(
                    REGRESSION_MODEL, raw.parent, selection, root / "prepared", model_dir=model,
                )
            self.assertIsNone(result)
            prepared = root / "prepared/prepared"
            frame = pd.read_csv(prepared / "data.csv")
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame.columns.tolist(), ["id", "raw_value"])
            self.assertEqual([p.name for p in prepared.iterdir()], ["data.csv"])
            fitted = self.predictor()
            with patch.object(base.TabularPredictor, "load", return_value=fitted) as load, \
                    patch.object(predict, "__file__", str(root / "source/entrypoints/predict.py")):
                scores = predict.execute(REGRESSION_MODEL, prepared, model, root / "scores")
            load.assert_called_once_with(str(model.resolve()))
            final = postprocess.execute(REGRESSION_MODEL, scores, root / "final", model_dir=model, data_directory=raw.parent, selection_file=root / "prepared/selection/selected_features.csv")
            df = pd.read_csv(final / "prediction_analysis.csv", dtype={"id": str})
            self.assertEqual(df.to_dict("list"),
                             {"id": ["0001", "0002", "0003"], "PREDICTION": [6, 8, 10]})
            for call in fitted.predict.call_args_list:
                self.assertEqual(call.args[0].columns.tolist(), ["VALUE"])

    def test_explicit_nested_predictor_path(self):
        with TemporaryDirectory() as directory:
            model = self.artifact(Path(directory), predictor_path="predictor")
            with patch.object(base.TabularPredictor, "load", return_value=self.predictor()) as load:
                self.load_model(model)
            load.assert_called_once_with(str(model.resolve() / "predictor"))

    def test_paths_outside_the_registered_archive_are_rejected(self):
        for value in ("../unrelated", "/opt/ml/old-training-model"):
            with self.subTest(value=value), TemporaryDirectory() as directory:
                model = self.artifact(Path(directory), predictor_path=value)
                with patch.object(base.TabularPredictor, "load") as load:
                    with self.assertRaisesRegex(ValueError, "inside the registered"):
                        self.load_model(model)
                    load.assert_not_called()

    def test_missing_mapping_is_not_replaced_with_parameter_features(self):
        with TemporaryDirectory() as directory:
            model = self.artifact(Path(directory))
            (model / "model_metadata.json").write_text("{}")
            with patch.object(base.TabularPredictor, "load") as load:
                with self.assertRaisesRegex(ValueError, "feature_mapping"):
                    self.load_model(model)
                load.assert_not_called()

    def test_mapping_must_match_the_actual_predictor(self):
        with TemporaryDirectory() as directory:
            model = self.artifact(Path(directory))
            fitted = self.predictor()
            fitted.features.return_value = ["ANOTHER_MODEL_COLUMN"]
            with patch.object(base.TabularPredictor, "load", return_value=fitted):
                with self.assertRaisesRegex(ValueError, "does not cover"):
                    self.load_model(model)

    def test_mapping_can_retain_an_input_discarded_by_the_fitted_predictor(self):
        with TemporaryDirectory() as directory:
            model = self.artifact(Path(directory))
            (model / "model_metadata.json").write_text(json.dumps({
                "feature_mapping": {"raw_value": "VALUE", "unused": "UNUSED"},
            }))
            with patch.object(base.TabularPredictor, "load", return_value=self.predictor()):
                predictor = self.load_model(model)
            result = predictor.predict_with_best_model(pd.DataFrame({
                "id": ["0001"], "raw_value": [3], "unused": [0],
            }))
            self.assertEqual(result.to_dict("list"), {"id": ["0001"], "score": [6]})

    def test_column_casing_failure_is_explicit(self):
        with TemporaryDirectory() as directory:
            model = self.artifact(Path(directory))
            with patch.object(base.TabularPredictor, "load", return_value=self.predictor()):
                predictor = self.load_model(model)
            with self.assertRaisesRegex(ValueError, "case-sensitive"):
                predictor.predict_with_best_model(pd.DataFrame({"id": ["0001"], "RAW_VALUE": [3]}))

    def test_binary_probability_uses_saved_positive_class_not_integer_one(self):
        with TemporaryDirectory() as directory:
            model = self.artifact(Path(directory), task="classification")
            fitted = self.predictor()
            fitted.problem_type, fitted.positive_class = "binary", "yes"
            fitted.predict_proba.return_value = pd.DataFrame({"no": [0.2], "yes": [0.8]})
            with patch.object(base.TabularPredictor, "load", return_value=fitted):
                predictor = self.load_model(model)
            result = predictor.predict_with_best_model(pd.DataFrame({"id": ["0001"], "raw_value": [3]}))
            self.assertEqual(result.to_dict("list"), {"id": ["0001"], "score": [0.8]})
            fitted.problem_type = "multiclass"
            with self.assertRaisesRegex(ValueError, "multiclass"):
                predictor.predict_with_best_model(pd.DataFrame({"id": ["0001"], "raw_value": [3]}))

    def test_file_hook_receives_prepared_csv_and_publishes_three_outputs(self):
        received = []
        class FilePredictor:
            def __init__(self, checkpoint, config, parameters):
                self.checkpoint = checkpoint
            def predict_file(self, user_data_path, feature_mapping, primary_key, *, output_directory):
                received.append((user_data_path, feature_mapping, primary_key))
                df = pd.read_csv(user_data_path, dtype={"id": str})
                predictions = df.assign(score=df["value"] * 2)[["id", "score"]]
                predictions.to_csv(output_directory / "predictions.csv", index=False)
                df[["id"]].to_csv(output_directory / "messaging.csv", index=False)
                predictions.to_csv(output_directory / "messaging_final.csv", index=False)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "data.csv").write_text("id,value\n0001,3\n")
            parameter_path = root / "parameters.json"
            parameter_path.write_text('{"primary_key":"id"}')
            with patch.object(base, "ModelPredictor", FilePredictor):
                output = base.predict_with_model(inputs, root / "model", root / "scores",
                                                 {"feature_mapping": {"value": "VALUE"}}, parameter_path)
            self.assertEqual(pd.read_csv(output / "predictions.csv", dtype={"id": str}).to_dict("records"),
                             [{"id": "0001", "score": 6}])
            self.assertEqual({p.name for p in output.iterdir()},
                             {"predictions.csv", "messaging.csv", "messaging_final.csv"})
            self.assertEqual(received, [(inputs / "data.csv", {"value": "VALUE"}, "id")])
            self.assertTrue(received[0][0].exists())



if __name__ == "__main__":
    unittest.main()
