"""Verify local CSV reads and path-only training adapters."""

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND


class StepIO(unittest.TestCase):
    def test_batch_prediction_loads_once_and_uses_the_use_case_override(self):
        import json
        from mlops_models.entrypoints import predict
        one = import_module(f"mlops_models.models.{FIRST}.inference.predict")
        two = import_module(f"mlops_models.models.{SECOND}.inference.predict")

        calls = []
        class CustomerPredictor:
            def __init__(self, checkpoint):
                calls.append(checkpoint)
            def predict_file(self, user_data_path):
                df = pd.read_csv(user_data_path, dtype={"id": str})
                result = pd.DataFrame({"id": df["id"], "score": 42.0})
                return result, df[["id"]], result.copy()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "data.csv").write_text("id,value\n0001,1\n0002,2\n")
            other_class = two.UseCaseModelPredictor
            with patch.object(one, "UseCaseModelPredictor", CustomerPredictor):
                scores = predict.execute(FIRST, inputs, root / "checkpoint", root / "scores")
            records = pd.read_csv(scores / "predictions.csv", dtype={"id": str}).to_dict("records")
            self.assertEqual(records, [{"id": "0001", "score": 42.0}, {"id": "0002", "score": 42.0}])
            self.assertEqual(calls, [root / "checkpoint"])
            self.assertIs(two.UseCaseModelPredictor, other_class)

    def test_batch_postprocessing_uses_the_use_case_formatter_and_saved_config(self):
        import json
        from mlops_models.entrypoints import postprocess
        one = import_module(f"mlops_models.models.{FIRST}.inference.postprocess")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, model = root / "scores", root / "model"
            inputs.mkdir()
            model.mkdir()
            config = {"primary_key": "id", "task_type": "regression", "customer_setting": "saved"}
            (model / "parameters.json").write_text(json.dumps(config))
            (model / "model_metadata.json").write_text('{"feature_mapping":{"value":"value"}}')
            raw = root / "raw"
            raw.mkdir()
            (raw / "data.csv").write_text("id,value\n0001,4\n0002,5\n")
            selection = root / "selected_features.csv"
            selection.write_text("features,description\nvalue,value\n")
            for name in ("predictions.csv", "messaging.csv", "messaging_final.csv"):
                (inputs / name).write_text("id,score,var1,var1_value\n0001,42,value,4\n0002,43,value,5\n")
            chunks = []
            def customer_format(data_path, selected, predictions, settings):
                self.assertEqual(data_path.read_bytes(), (raw / "data.csv").read_bytes())
                self.assertEqual(selected["features"].tolist(), ["value"])
                self.assertEqual(settings["model_metadata"]["feature_mapping"], {"value": "value"})
                result = predictions["messaging_final.csv"]
                self.assertEqual(settings["customer_setting"], "saved")
                chunks.append(len(result))
                return result.rename(columns={"score": "prediction"}).assign(prediction=result["score"] * 2)
            with patch.object(one, "use_case_postprocess", customer_format):
                output = postprocess.execute(FIRST, inputs, root / "output", model_dir=model, data_directory=raw, selection_file=selection, chunk_rows=1)
            df = pd.read_csv(output, dtype={"id": str})
            self.assertEqual(df["id"].tolist(), ["0001", "0002"])
            self.assertEqual(df["prediction"].tolist(), [84, 86])
            self.assertEqual(chunks, [2])
            self.assertEqual(df["var1_value"].tolist(), [4, 5])
            self.assertEqual(output.name, "prediction_analysis.csv")
            self.assertEqual(json.loads((output.parent / "summary.json").read_text())["rows"], 2)

    def test_local_csv_readers_preserve_identifiers_and_missing_values(self):
        for model_id in (FIRST, SECOND):
            with self.subTest(model=model_id), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "data.csv").write_text("id,value,category\n0001,1,NA\n0002,,text\n")
                selected = root / "selected.csv"
                selected.write_text("feature\nvalue\n")
                selection = import_module("mlops_models.model_base.training.feature_selection")
                preprocess = import_module("mlops_models.model_base.training.preprocess")
                for df in (selection.get_container_input(root, {"primary_key": "id"}),
                           preprocess.get_container_input(root, selected, {"primary_key": "id"})[0]):
                    self.assertEqual(df["id"].tolist(), ["0001", "0002"])
                    self.assertEqual(df["category"].tolist(), ["NA", "text"])
                    self.assertTrue(pd.isna(df.loc[1, "value"]))

    def test_feature_selection_collects_only_csv_and_pdf(self):
        for model_id in (FIRST, SECOND):
            with self.subTest(model=model_id), TemporaryDirectory() as directory:
                root = Path(directory)
                csv, pdf = root / "features.csv", root / "plot.pdf"
                csv.write_text("feature\nvalue\n")
                pdf.write_bytes(b"their-report")
                from mlops_models.model_base.training.feature_selection import write_container_output
                class Selector:
                    def export_features_to_csv(self, path):
                        Path(path).write_bytes(csv.read_bytes())
                    def export_to_pdf(self, path):
                        Path(path).write_bytes(pdf.read_bytes())
                result = write_container_output(Selector(), root / "output")
                self.assertEqual((result / "selected_features.csv").read_bytes(), csv.read_bytes())
                self.assertEqual((result / "random_forest_plots.pdf").read_bytes(), pdf.read_bytes())
                self.assertEqual({p.name for p in result.iterdir()},
                                 {"selected_features.csv", "random_forest_plots.pdf"})

    def test_training_passes_paths_without_reading_or_interpreting_the_files(self):
        for model_id in (FIRST, SECOND):
            module = import_module(f"mlops_models.models.{model_id}.training.train")
            with self.subTest(model=model_id), TemporaryDirectory() as directory:
                root = Path(directory)
                def model_script(trainer, checkpoint_path, output):
                    self.assertEqual(trainer.feature_path, Path("/selection.csv"))
                    self.assertEqual(trainer.data_path, Path("/prepared"))
                    self.assertEqual(trainer.config["their_parameter"], "value")
                    self.assertEqual((checkpoint_path, output), (root / "output/checkpoint", root / "output"))
                    checkpoint_path.mkdir(parents=True)
                    (checkpoint_path / "their-model.bin").write_bytes(b"opaque model")
                    return checkpoint_path, {"phase": "initial", "evaluation": {"usage": "validation"}}, {"r2": 0.8}
                with patch.object(module.UseCaseModelTrainer, "train_model", model_script):
                    config = {"their_parameter": "value", "parameters": {},
                              "model_id": model_id, "task_type": "regression"}
                    result = module.train_model("/prepared", "/selection.csv",
                                                root / "model", root / "output", config)
                self.assertEqual((result / "their-model.bin").read_bytes(), b"opaque model")
                self.assertTrue((result / "model_metadata.json").is_file())
                self.assertFalse((root / "output/checkpoint").exists())

    def test_use_case_override_changes_only_its_own_selection(self):
        from mlops_models.model_base.training.feature_selection import FeatureSelector
        one = import_module(f"mlops_models.models.{FIRST}.training.feature_selection")
        UseCaseFeatureSelector, run_feature_selection = one.UseCaseFeatureSelector, one.run_feature_selection
        OtherSelector = import_module(f"mlops_models.models.{SECOND}.training.feature_selection").UseCaseFeatureSelector

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.csv").write_text("id,first,second\n0001,1,2\n0002,3,4\n")
            config = {"primary_key": "id", "feature_selection": {"top_k": 1}}
            def fit(self, df):
                self.df_seen = df
                self.selected = pd.Series([0.9], index=["second"])
            with patch.object(UseCaseFeatureSelector, "fit", fit):
                output = run_feature_selection(root, root / "output", config)
                self.assertEqual(pd.read_csv(output / "selected_features.csv")["feature"].tolist(), ["second"])
                self.assertTrue((output / "random_forest_plots.pdf").is_file())
                self.assertIs(OtherSelector.fit, FeatureSelector.fit)

    def test_base_preprocessing_is_inherited_and_can_be_overridden_per_use_case(self):
        from mlops_models.model_base.training import preprocess as base
        one = import_module(f"mlops_models.models.{FIRST}.training.preprocess")
        two = import_module(f"mlops_models.models.{SECOND}.training.preprocess")
        from mlops_models.entrypoints import preprocess_training

        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "features.csv"
            selected.write_text("feature\nvalue\n")
            (root / "data.csv").write_text("id,date,value,target\n0001,2026-01-01,5,1\n")
            config = {"primary_key": "id", "timestamp": "date", "target_key": "target"}
            def custom_process(data, selection, output, config):
                df, _ = base.get_container_input(data, selection, {"primary_key": "id"})
                return base.write_container_output(df.assign(value=df["value"] * 2), output)
            with patch.object(one, "process_train", custom_process):
                output = preprocess_training.execute(FIRST, root, selected, root / "output")
                self.assertEqual(pd.read_csv(output / "data.csv")["value"].tolist(), [10])
                self.assertIs(two.process_train, base.process_train)


if __name__ == "__main__":
    unittest.main()
