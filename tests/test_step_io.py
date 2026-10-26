"""Verify local CSV reads and path-only training adapters."""

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND


class StepIO(unittest.TestCase):
    def test_analysis_mode_routes_through_dispatcher_with_saved_parameters_and_exact_paths(self):
        import io
        import json
        import os
        import sys
        import tarfile
        from contextlib import redirect_stdout
        from mlops_models.entrypoints import __main__ as dispatcher
        from mlops_models.pipelines import inspect

        for model_id in (FIRST, SECOND):
            module = import_module(f"mlops_models.models.{model_id}.inference.postprocess")
            for mode in ("prediction_inference", "prediction_test"):
                with self.subTest(model=model_id, mode=mode), TemporaryDirectory() as directory:
                    root = Path(directory)
                    raw, scores, model = root / "raw", root / "scores", root / "model"
                    for path in (raw, scores, model):
                        path.mkdir()
                    (raw / "data.csv").write_text("id,x\n001,2\n")
                    selected = root / "selected.csv"
                    selected.write_text("features\nx\n")
                    prediction = scores / "custom.csv"
                    prediction.write_text("id,score\n001,3\n")
                    settings = {"project": "customer-project", "test_ymid": "202610",
                                "primary_key": "id"}
                    (model / "parameters.json").write_text(json.dumps(settings))
                    (model / "model_metadata.json").write_text('{"feature_mapping":{"x":"x"}}')
                    if mode == "prediction_test":
                        archive_path = root / "model.tar.gz"
                        with tarfile.open(archive_path, "w:gz") as archive:
                            for path in model.iterdir():
                                archive.add(path, arcname=path.name)
                        model = archive_path
                    output = root / "output"
                    called = []
                    # Emulate helper functions reading the analysis module's parameters.
                    shared_parameters = module.parameters
                    shared_parameters["stale_from_previous_job"] = True
                    def analyze(data_path, selection_file, predictions, destination):
                        parameters = module.parameters
                        self.assertIs(parameters, shared_parameters)
                        self.assertNotIn("stale_from_previous_job", parameters)
                        self.assertEqual((data_path, selection_file, predictions, destination),
                                         (raw / "data.csv", selected, {"custom.csv": prediction}, output))
                        self.assertEqual({key: parameters[key] for key in settings}, settings)
                        self.assertEqual(parameters["model_metadata"]["feature_mapping"], {"x": "x"})
                        called.append(mode)
                        (destination / "plots").mkdir()
                        (destination / "plots/report.html").write_text(mode)
                    other = "prediction_test" if mode == "prediction_inference" else "prediction_inference"
                    command = ["step", "postprocess", "--model", model_id,
                               "--input", str(scores), "--output", str(output),
                               "--data", str(raw), "--selection", str(selected), "--model-dir", str(model)]
                    # The default and the pipeline's environment override both reach the real dispatcher.
                    environment = {} if mode == "prediction_inference" else {"MLOPS_ANALYSIS_MODE": mode}
                    logs = io.StringIO()
                    with patch.dict(os.environ, environment, clear=True), patch.object(sys, "argv", command), \
                            patch.object(module, mode, side_effect=analyze), patch.object(module, other) as unused, \
                            patch.object(inspect, "write_inference_report") as report, redirect_stdout(logs):
                        dispatcher.main()
                    unused.assert_not_called()
                    report.assert_called_once()
                    self.assertEqual(called, [mode])
                    self.assertEqual((output / "plots/report.html").read_text(), mode)
                    self.assertIn(f"Project: {settings['project']} | Test YMID: {settings['test_ymid']}", logs.getvalue())
                    self.assertIn(f"Analysis output directory: {output}", logs.getvalue())

    def test_invalid_mode_or_format_fails_before_analysis_and_analysis_failure_does_not_report(self):
        import json
        import os
        from mlops_models.entrypoints import postprocess
        module = import_module(f"mlops_models.models.{FIRST}.inference.postprocess")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.csv").write_text("id,x\n1,2\n")
            (root / "selected.csv").write_text("features\nx\n")
            (root / "parameters.json").write_text(json.dumps({"primary_key": "id"}))
            (root / "model_metadata.json").write_text('{"feature_mapping":{"x":"x"}}')
            arguments = dict(model_dir=root, data_directory=root, selection_file=root / "selected.csv")
            with patch.object(module, "prediction_inference", side_effect=RuntimeError("customer analysis failed")) as analysis, \
                    patch("mlops_models.pipelines.inspect.write_inference_report") as report:
                with patch.dict(os.environ, {"MLOPS_PREDICTION_FORMAT": "parquet"}):
                    with self.assertRaisesRegex(ValueError, "FORMAT"):
                        postprocess.execute(FIRST, root, root / "output", **arguments)
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, "analysis_mode"):
                        postprocess.execute(FIRST, root, root / "output", analysis_mode="typo", **arguments)
                    analysis.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "customer analysis failed"):
                        postprocess.execute(FIRST, root, root / "output", **arguments)
                report.assert_not_called()

    def test_batch_prediction_dispatches_to_selected_use_case_and_loads_once(self):
        import json
        from mlops_models.entrypoints import predict
        from mlops_models.model_base.inference import predict as base
        one = import_module(f"mlops_models.models.{FIRST}.inference.predict")
        two = import_module(f"mlops_models.models.{SECOND}.inference.predict")

        calls = []
        class CustomerPredictor:
            def __init__(self, checkpoint, config, parameters):
                calls.append(checkpoint)
            def predict_file(self, user_data_path, feature_mapping, primary_key, *, output_directory):
                df = pd.read_csv(user_data_path, dtype={"id": str})
                result = pd.DataFrame({"id": df["id"], "score": 42.0})
                result.to_csv(output_directory / "predictions.csv", index=False)
                df[["id"]].to_csv(output_directory / "messaging.csv", index=False)
                result.to_csv(output_directory / "messaging_final.csv", index=False)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "data.csv").write_text("id,value\n0001,1\n0002,2\n")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model_metadata.json").write_text('{"feature_mapping":{"value":"VALUE"}}')
            parameter_path = root / "source/models" / FIRST / "parameters.json"
            parameter_path.parent.mkdir(parents=True)
            parameter_path.write_text('{"primary_key":"id"}')
            with patch.object(base, "ModelPredictor", CustomerPredictor), \
                    patch.object(predict, "__file__", str(root / "source/entrypoints/predict.py")), \
                    patch.object(one, "predict_with_model", wraps=one.predict_with_model) as selected, \
                    patch.object(two, "predict_with_model") as other:
                scores = predict.execute(FIRST, inputs, root / "checkpoint", root / "scores")
                selected.assert_called_once()
                other.assert_not_called()
            records = pd.read_csv(scores / "predictions.csv", dtype={"id": str}).to_dict("records")
            self.assertEqual(records, [{"id": "0001", "score": 42.0}, {"id": "0002", "score": 42.0}])
            self.assertEqual(calls, [root / "checkpoint"])

    def test_batch_postprocessing_passes_paths_and_publishes_customer_files_unchanged(self):
        import json
        from mlops_models.entrypoints import postprocess
        from mlops_models.runtime.load_source import upload_processing_outputs
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
            (inputs / "batches").mkdir()
            (inputs / "batches/leads.CSV").write_text("id,score,var1,var1_value\n0001,42,value,4\n0002,43,value,5\n")
            calls = []
            def customer_analysis(data_path, selected, predictions, destination):
                settings = one.parameters
                self.assertEqual(data_path.read_bytes(), (raw / "data.csv").read_bytes())
                self.assertEqual(selected, selection)
                self.assertEqual(settings["model_metadata"]["feature_mapping"], {"value": "value"})
                self.assertEqual(list(predictions), ["batches/leads.CSV"])
                self.assertEqual(predictions["batches/leads.CSV"], inputs / "batches/leads.CSV")
                result = pd.read_csv(predictions["batches/leads.CSV"], dtype={"id": str})
                self.assertEqual(settings["customer_setting"], "saved")
                calls.append(len(result))
                reports = destination / "reports"
                plots = destination / "plots"
                reports.mkdir()
                plots.mkdir()
                result = result.rename(columns={"score": "prediction"}).assign(prediction=result["score"] * 2)
                result.to_csv(reports / "leads.csv", index=False)
                (reports / "regression_metrics.json").write_text('{"mae": 0.25}\n')
                (plots / "analysis.pdf").write_bytes(b"customer PDF bytes")
                (plots / "segment.png").write_bytes(b"customer PNG bytes")
                # File-producing customer analysis returns None and need not write summary.json.
            with patch.object(one, "prediction_inference", customer_analysis):
                output = postprocess.execute(FIRST, inputs, root / "output", model_dir=model, data_directory=raw, selection_file=selection, chunk_rows=1)
            df = pd.read_csv(output / "reports/leads.csv", dtype={"id": str})
            self.assertEqual(df["id"].tolist(), ["0001", "0002"])
            self.assertEqual(df["prediction"].tolist(), [84, 86])
            self.assertEqual(calls, [2])
            self.assertEqual(df["var1_value"].tolist(), [4, 5])
            self.assertEqual(output, root / "output")
            self.assertFalse((output / "summary.json").exists())
            expected = {p.relative_to(output).as_posix(): p.read_bytes()
                        for p in output.rglob("*") if p.is_file()}
            uploaded = {}
            def upload_file(path, bucket, key, **kwargs):
                self.assertEqual(bucket, "artifacts")
                uploaded[key.removeprefix("runs/example/predictions/")] = Path(path).read_bytes()
            with patch("boto3.client") as client:
                client.return_value.upload_file.side_effect = upload_file
                upload_processing_outputs({"Outputs": [{
                    "OutputName": "predictions", "S3Output": {
                        "LocalPath": "/opt/ml/processing/output",
                        "S3Uri": "s3://artifacts/runs/example/predictions",
                    },
                }]}, Path("/opt/ml/processing/output"), output)
            self.assertEqual(uploaded, expected)

    def test_postprocessing_fails_when_analysis_writes_nothing(self):
        from mlops_models.model_base.inference.postprocess import write_container_output
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "wrote no files"):
                write_container_output(directory, data_path=None, selection_file=None,
                                       predictions={}, parameters={})

    def test_transform_analysis_still_receives_file_paths_with_string_ids(self):
        import json
        import os
        from mlops_models.entrypoints import postprocess
        one = import_module(f"mlops_models.models.{FIRST}.inference.postprocess")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw, scores = root / "raw", root / "scores"
            raw.mkdir()
            scores.mkdir()
            (raw / "data.csv").write_text("id,value\n0001,4\n0002,5\n")
            selection = root / "selection.csv"
            selection.write_text("features\nvalue\n")
            (scores / "part.out").write_text(
                json.dumps({"id": "0001", "score": 0.8}) + "\n"
                + json.dumps({"id": "0002", "score": 0.2}) + "\n")
            observed = []
            def analysis(data_path, selection_path, predictions, destination):
                observed.append(predictions["messaging_final.csv"])
                frame = pd.read_csv(observed[0], dtype={"id": str})
                self.assertEqual(frame["id"].tolist(), ["0001", "0002"])
                self.assertEqual(frame["score"].tolist(), [0.8, 0.2])
                frame.to_csv(destination / "leads.csv", index=False)
            (root / "parameters.json").write_text('{"primary_key":"id"}')
            (root / "model_metadata.json").write_text('{"feature_mapping":{"value":"value"}}')
            with patch.object(one, "prediction_inference", analysis), \
                    patch.dict(os.environ, {"MLOPS_PREDICTION_FORMAT": "jsonlines"}):
                result = postprocess.execute(
                    FIRST, scores, root / "output", model_dir=root, data_directory=raw,
                    selection_file=selection, chunk_rows=1)
            self.assertTrue((result / "leads.csv").is_file())
            self.assertFalse(observed[0].exists())

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
        import json
        for model_id in (FIRST, SECOND):
            module = import_module(f"mlops_models.models.{model_id}.training.train")
            with self.subTest(model=model_id), TemporaryDirectory() as directory:
                root = Path(directory)
                def model_script(trainer, checkpoint_path, output):
                    self.assertEqual(trainer.feature_path, Path("/selection.csv"))
                    self.assertEqual(trainer.data_path, Path("/prepared"))
                    self.assertEqual(trainer.config["their_parameter"], "value")
                    self.assertEqual(trainer.parameters["task_type"], "regression")
                    self.assertNotIn("task_type", trainer.config)
                    self.assertEqual((checkpoint_path, output), (root / "output/checkpoint", root / "output"))
                    checkpoint_path.mkdir(parents=True)
                    (checkpoint_path / "their-model.bin").write_bytes(b"opaque model")
                    return checkpoint_path, {"phase": "initial"}
                with patch.object(module.UseCaseModelTraining, "train_model", model_script):
                    config = {"their_parameter": "value"}
                    parameter_path = root / model_id / "parameters.json"
                    parameter_path.parent.mkdir()
                    parameter_path.write_text(json.dumps({"task_type": "regression"}))
                    parameter_path.with_name("model_metadata.json").write_text(json.dumps(config))
                    result = module.train_model("/prepared", "/selection.csv",
                                                root / "model", root / "output", parameter_path)
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
            def custom_process(data, selection, output, config, parameter_path):
                df, _ = base.get_container_input(data, selection, {"primary_key": "id"})
                return base.write_container_output(df.assign(value=df["value"] * 2), output)
            with patch.object(one, "process_train", custom_process):
                self.assertIsNone(preprocess_training.execute(FIRST, root, selected, root / "output"))
                self.assertEqual(pd.read_csv(root / "output/prepared/data.csv")["value"].tolist(), [10])
                self.assertIs(two.process_train, base.process_train)


if __name__ == "__main__":
    unittest.main()
