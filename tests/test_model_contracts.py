"""Verify metadata/parameter separation and the loader-to-model file contract."""

from importlib import import_module
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from mlops_models.entrypoints import train, predict
from mlops_models.entrypoints import __main__ as dispatcher
from mlops_models.runtime import load_source
from mlops_models.model_base.inference import predict as base_predict


class ModelContracts(unittest.TestCase):
    def test_prediction_wrappers_import_when_customer_exports_only_file_prediction(self):
        base = ModuleType("mlops_models.model_base.inference.predict")
        base.predict_with_model = Mock(return_value=None)
        models = Path(__file__).parents[1] / "mlops-models/src/mlops_models/models"
        args = (Path("prepared"), Path("model"), Path("output"),
                {"feature_mapping": {"value": "VALUE"}}, Path("parameters.json"))
        for model_id in ("usecase1", "usecase2"):
            with self.subTest(model_id=model_id), patch.dict(sys.modules, {base.__name__: base}):
                wrapper = runpy.run_path(str(models / model_id / "inference/predict.py"))
                self.assertIsNone(wrapper["predict_with_model"](*args))
                base.predict_with_model.assert_called_once_with(*args)
                base.predict_with_model.reset_mock()

    def test_training_wrappers_accept_customer_model_training_class_name(self):
        base = ModuleType("mlops_models.model_base.training.train")
        base.ModelTraining = type("ModelTraining", (), {})
        base.train_model = Mock(return_value=Path("model"))
        models = Path(__file__).parents[1] / "mlops-models/src/mlops_models/models"
        args = (Path("prepared"), Path("selection.csv"), Path("model"),
                Path("output"), {}, Path("parameters.json"))
        for model_id in ("usecase1", "usecase2"):
            with self.subTest(model_id=model_id), patch.dict(sys.modules, {base.__name__: base}):
                wrapper = runpy.run_path(str(models / model_id / "training/train.py"))
                self.assertEqual(wrapper["train_model"](*args, hyperparameters={"depth": "3"}), Path("model"))
                training_class = base.train_model.call_args.kwargs["trainer_cls"]
                self.assertTrue(issubclass(training_class, base.ModelTraining))
                self.assertEqual(training_class.__name__, "UseCaseModelTraining")
                base.train_model.assert_called_once_with(
                    *args, trainer_cls=training_class, hyperparameters={"depth": "3"})
                base.train_model.reset_mock()

    def test_training_keeps_initial_metadata_parameters_and_learned_metadata_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "source/models/usecase1"
            folder.mkdir(parents=True)
            parameters = {"primary_key": "id", "task_type": "regression"}
            initial = {"task_type": "metadata-value", "best_config": {"depth": 4}}
            learned = {"feature_mapping": {"value": "VALUE"}, "evaluation": {"usage": "validation"}}
            (folder / "parameters.json").write_text(json.dumps(parameters))
            (folder / "model_metadata.json").write_text(json.dumps(initial))
            selection = root / "selected_features.csv"
            selection.write_text("features\nvalue\n")
            module = import_module("mlops_models.models.usecase1.training.train")

            def fit(trainer, checkpoint, output):
                self.assertEqual(trainer.config, initial)
                self.assertEqual(trainer.parameters, {**parameters, "model_id": "usecase1"})
                checkpoint.mkdir(parents=True)
                (checkpoint / "weights").write_bytes(b"fitted model")
                return checkpoint, learned, {"r2": 0.75}

            with patch.object(train, "__file__", str(root / "source/entrypoints/train.py")), \
                    patch.object(module.UseCaseModelTraining, "train_model", fit):
                model = train.execute("usecase1", root / "inputs", root / "model", root / "output",
                                      selection_path=selection)
            self.assertEqual(json.loads((model / "model_metadata.json").read_text()), learned)
            self.assertEqual(json.loads((model / "parameters.json").read_text()),
                             {**parameters, "model_id": "usecase1"})
            report = json.loads((model / "evaluation.json").read_text())
            self.assertEqual((report["task"], report["metrics"]["r2"]), ("regression", 0.75))
            self.assertEqual(json.loads((folder / "model_metadata.json").read_text()), initial)
            self.assertEqual(json.loads((folder / "parameters.json").read_text()), parameters)

    def test_prediction_gets_three_explicit_inputs_without_reading_pipeline_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "source/models/usecase1"
            folder.mkdir(parents=True)
            (folder / "parameters.json").write_text('{"primary_key":"id","task_type":"regression"}')
            (folder / "model_metadata.json").write_text('{"feature_mapping":{"wrong":"WRONG"}}')
            metadata = {"feature_mapping": {"value": "VALUE"}, "primary_key": "metadata-value"}
            model = root / "model"
            model.mkdir()
            (model / "model_metadata.json").write_text(json.dumps(metadata))
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "data.csv").write_text("id,value\n0001,2\n")
            calls = []

            class Predictor:
                def __init__(self, checkpoint, config, parameters):
                    calls.append((checkpoint, config, parameters))

                def predict_file(self, user_test_data_path, feature_mapping, primary_key, *, output_directory):
                    calls.append((user_test_data_path, feature_mapping, primary_key))
                    if output_directory != root / "output":
                        raise AssertionError(output_directory)
                    # Keep customer formatting and additional artifacts unchanged.
                    (output_directory / "predictions.csv").write_bytes(b'"id","prediction"\r\n"0001",5.0000\r\n')
                    (output_directory / "messaging.csv").write_text("id\n0001\n")
                    (output_directory / "messaging_final.csv").write_text("id,prediction\n0001,5\n")
                    (output_directory / "native_report.txt").write_bytes(b"native report\n")

            with patch.object(predict, "__file__", str(root / "source/entrypoints/predict.py")), \
                    patch.object(base_predict, "ModelPredictor", Predictor):
                output = predict.execute("usecase1", prepared, model, root / "output")
            self.assertEqual(calls, [
                (model, metadata, {"primary_key": "id", "task_type": "regression"}),
                (prepared / "data.csv", {"value": "VALUE"}, "id"),
            ])
            self.assertEqual(pd.read_csv(output / "predictions.csv", dtype={"id": str}).to_dict("records"),
                             [{"id": "0001", "prediction": 5}])
            self.assertEqual((output / "predictions.csv").read_bytes(), b'"id","prediction"\r\n"0001",5.0000\r\n')
            self.assertEqual((output / "native_report.txt").read_bytes(), b"native report\n")

    def test_prediction_collects_nested_csvs_and_upload_preserves_relative_paths_and_bytes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            files = {
                "period1/leads.csv": b'"id","score"\r\n"0001",0.5000\r\n',
                "period2/leads.csv": b"id,score\n0002,0.75\n",
                "period2/reports/details.CSV": b"id,explanation\n0002,example\n",
                "native_report.txt": b"keep this too\n",
            }
            for name, data in files.items():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (output / "directory.csv").mkdir()
            printed = io.StringIO()
            with redirect_stdout(printed):
                result = base_predict.write_container_output(output)
            self.assertEqual(result, output)
            events = [json.loads(line) for line in printed.getvalue().splitlines()]
            self.assertEqual([event["relative_path"] for event in events],
                             sorted(name for name in files if name.lower().endswith(".csv")))
            for event in events:
                self.assertEqual(event["event"], "prediction_output_csv")
                self.assertEqual(event["path"], str(output / event["relative_path"]))
            uploaded = {}
            s3 = Mock()
            s3.upload_file.side_effect = lambda filename, bucket, key, **kwargs: uploaded.update(
                {key: Path(filename).read_bytes()})
            config = {"Outputs": [{"OutputName": "predictions", "S3Output": {
                "LocalPath": "/opt/ml/processing/output",
                "S3Uri": "s3://bucket/run/predictions"}}]}
            with patch("boto3.client", return_value=s3):
                load_source.upload_processing_outputs(config, Path("/opt/ml/processing/output"), result)
            self.assertEqual(uploaded, {f"run/predictions/{name}": data for name, data in files.items()})
            for name, data in files.items():
                self.assertEqual((output / name).read_bytes(), data)

    def test_prediction_fails_if_no_csv_was_written(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.txt").write_text("no prediction data")
            for path in (output, output / "missing"):
                with self.subTest(path=path), self.assertRaisesRegex(FileNotFoundError, "any CSV files"):
                    base_predict.write_container_output(path)

    def test_loader_supplies_selection_and_registered_metadata_before_dispatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "mlops_models").mkdir(parents=True)
            archive_path = root / "model.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, value in {
                    "parameters.json": {"model_id": "usecase1"},
                    "model_metadata.json": {"feature_mapping": {"value": "VALUE"}},
                }.items():
                    data = json.dumps(value).encode()
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            before = archive_path.read_bytes()
            configured_output = root / "read-only-output"
            job = {"ProcessingOutputConfig": {"Outputs": [
                {"OutputName": name, "S3Output": {
                    "LocalPath": str(configured_output / name), "S3Uri": f"s3://bucket/run/{name}"}}
                for name in ("prepared", "selection")
            ]}}
            read_text = Path.read_text

            def read(path, *args, **kwargs):
                if str(path) == "/opt/ml/config/processingjobconfig.json":
                    return json.dumps(job)
                return read_text(path, *args, **kwargs)

            def dispatch():
                args = dispatcher.parse_job_arguments()
                self.assertNotEqual(Path(args.output), configured_output)
                self.assertEqual(pd.read_csv(args.selection).to_dict("list"),
                                 {"features": ["value"], "description": ["VALUE"]})
                self.assertEqual(json.loads((Path(args.model_dir) / "model_metadata.json").read_text()),
                                 {"feature_mapping": {"value": "VALUE"}})
                prepared = Path(args.output) / "prepared"
                prepared.mkdir()
                (prepared / "data.csv").write_text("id,value\n0001,2\n")

            uploaded = {}
            s3 = Mock()
            s3.upload_file.side_effect = lambda filename, bucket, key, **kwargs: uploaded.update(
                {key: Path(filename).read_bytes()})
            with patch.dict(os.environ, {"MLOPS_MODEL_CODE_SOURCE": "pipeline",
                                         "MLOPS_PROCESSING_OUTPUT_UPLOAD": "application"}), \
                    patch.object(load_source, "__file__", str(source / "load_source.py")), \
                    patch.object(sys, "argv", ["load_source.py", "preprocess", "--model", "usecase1",
                                              "--model-dir", str(archive_path),
                                              "--output", str(configured_output)]), \
                    patch.object(Path, "read_text", read), \
                    patch.object(dispatcher, "main", side_effect=dispatch), \
                    patch("boto3.client", return_value=s3):
                load_source.main()
            self.assertEqual(set(uploaded), {
                "run/selection/selected_features.csv", "run/prepared/data.csv",
                "run/selection/_execution.json", "run/prepared/_execution.json",
            })
            record = json.loads(uploaded["run/prepared/_execution.json"])
            self.assertEqual(record["step"], "preprocess")
            self.assertEqual(record["model_code_source"], "pipeline")
            self.assertEqual(uploaded["run/prepared/_execution.json"],
                             uploaded["run/selection/_execution.json"])
            self.assertEqual(uploaded["run/prepared/data.csv"], b"id,value\n0001,2\n")
            self.assertEqual(archive_path.read_bytes(), before)
            self.assertFalse(configured_output.exists())


if __name__ == "__main__":
    unittest.main()
