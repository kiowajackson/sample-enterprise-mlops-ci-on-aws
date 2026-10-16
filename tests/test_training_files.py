"""Check file handoffs without interpreting a model script's intermediate schema."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock

from mlops_models.entrypoints import feature_selection, preprocess_training, get_data, train
from mlops_models.pipelines.source import build_definitions
from mlops_skeleton.pipelines import walk_steps
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND


class TrainingFiles(unittest.TestCase):
    def test_model_hooks_receive_distinct_inputs_and_keep_opaque_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "data.csv").write_text("customer_field\nraw-value\n")
            selection_bytes = b"private_selection_schema\nopaque-value\n"
            prepared_bytes = b"private_training_schema\nprepared-value\n"
            report_bytes = b"%PDF-1.4\nopaque report bytes supplied by the model script"
            def select(source, output, config):
                self.assertEqual(source, raw)
                output = output / "selection"
                output.mkdir(parents=True)
                (output / "selected_features.csv").write_bytes(selection_bytes)
                (output / "random_forest_plots.pdf").write_bytes(report_bytes)
                return output
            def preprocess(source, selection, output, config):
                self.assertEqual(source, raw)
                self.assertEqual(selection.read_bytes(), selection_bytes)
                output = output / "prepared"
                output.mkdir(parents=True)
                (output / "period1.csv").write_bytes(prepared_bytes)
                (output / "period2.csv").write_bytes(prepared_bytes)
                return output
            with patch(f"mlops_models.models.{FIRST}.training.feature_selection.run_feature_selection", side_effect=select):
                selected = feature_selection.execute(FIRST, raw, root / "select")
            with patch(f"mlops_models.models.{FIRST}.training.preprocess.process_train", side_effect=preprocess):
                prepared = preprocess_training.execute(FIRST, raw, selected / "selected_features.csv", root / "preprocess")
            self.assertEqual((selected / "selected_features.csv").read_bytes(), selection_bytes)
            self.assertEqual((prepared / "period1.csv").read_bytes(), prepared_bytes)
            self.assertEqual((prepared / "period2.csv").read_bytes(), prepared_bytes)
            self.assertEqual((selected / "random_forest_plots.pdf").read_bytes(), report_bytes)
            from mlops_models.runtime.load_source import upload_processing_outputs
            uploads = {}
            def upload(filename, bucket, key, **kwargs):
                uploads[key] = Path(filename).read_bytes()
            for name, directory in (("selection", selected), ("prepared", prepared)):
                config = {"Outputs": [{"OutputName": name, "S3Output": {
                    "LocalPath": f"/opt/ml/processing/output/{name}", "S3Uri": f"s3://bucket/run/{name}"}}]}
                client = Mock()
                client.upload_file.side_effect = upload
                with patch("boto3.client", return_value=client):
                    upload_processing_outputs(config, Path("/opt/ml/processing/output"), directory.parent)
            self.assertEqual(uploads["run/selection/random_forest_plots.pdf"], report_bytes)
            self.assertEqual(uploads["run/prepared/period2.csv"], prepared_bytes)

    def test_missing_output_and_script_failure_stop_at_producing_step(self):
        from mlops_models.runtime.load_source import upload_processing_outputs
        with TemporaryDirectory() as directory:
            for output in ("selection", "prepared"):
                config = {"Outputs": [{"OutputName": output, "S3Output": {
                    "LocalPath": f"/opt/ml/processing/output/{output}",
                    "S3Uri": f"s3://bucket/run/{output}"}}]}
                with self.assertRaisesRegex(FileNotFoundError, "has no files"):
                    upload_processing_outputs(config, Path("/opt/ml/processing/output"), Path(directory))
            with patch(f"mlops_models.models.{FIRST}.training.preprocess.process_train",
                       side_effect=RuntimeError("customer-script-failed")):
                with self.assertRaisesRegex(RuntimeError, "customer-script-failed"):
                    preprocess_training.execute(FIRST, "raw.csv", "selection.csv", directory)

    def test_get_data_queries_feature_store_once_then_selection_reads_only_csv(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MLOPS_FEATURE_STORE_CONFIG_URI": "s3://bucket/store.json"}
        ):
            root = Path(directory)
            def query(store_config_uri, model_id, config, *, inference=False):
                self.assertEqual(store_config_uri, "s3://bucket/store.json")
                self.assertEqual(model_id, FIRST)
                self.assertFalse(inference)
                import pandas as pd
                return pd.DataFrame({"customer_id": ["0001"], "amount": [2]}), {"delivery_id": "sample"}
            def select(source, output, config):
                self.assertEqual((source / "data.csv").read_text(), "customer_id,amount\n0001,2\n")
                output.mkdir()
                (output / "selected_features.csv").write_text("feature\namount\n")
                return output
            with patch("mlops_models.entrypoints.get_data.get_container_input", side_effect=query) as query_store:
                raw_csv = get_data.execute(FIRST, None, root / "get")
                with patch(f"mlops_models.models.{FIRST}.training.feature_selection.run_feature_selection",
                           side_effect=select):
                    feature_selection.execute(FIRST, raw_csv.parent, root / "select")
                query_store.assert_called_once()


    def test_train_receives_both_paths_and_preserves_saved_model_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "input/prepared"
            selected = root / "input/selection/selected_features.csv"
            prepared.mkdir(parents=True)
            selected.parent.mkdir()
            selected.write_text("their_schema\nselected-name\n")
            for name in ("period1", "period2"):
                (prepared / (name + ".csv")).write_text("their_data\nvalue\n")
            (root / "model").mkdir()  # The managed container starts with an empty model directory.
            def model_script(trainer, checkpoint, output):
                self.assertEqual(trainer.data_path, prepared)
                self.assertEqual(trainer.feature_path, selected)
                self.assertEqual(len(list(trainer.data_path.glob("*.csv"))), 2)
                self.assertEqual(trainer.feature_path.read_text(), "their_schema\nselected-name\n")
                checkpoint.mkdir(parents=True)
                (checkpoint / "model.bin").write_bytes(b"saved-model")
                (checkpoint / "metadata.json").write_text('{"phase": "initial"}')
                return checkpoint, {"phase": "initial", "evaluation": {"usage": "validation"}}, {"r2": 0.8}
            with patch(f"mlops_models.models.{FIRST}.training.train.UseCaseModelTrainer.train_model", model_script):
                model = train.execute(FIRST, root / "input", root / "model", root / "output",
                                      selection_path=selected)
            self.assertEqual((model / "model.bin").read_bytes(), b"saved-model")
            self.assertEqual(json.loads((model / "metadata.json").read_text()), {"phase": "initial"})
            config = json.loads((model / "parameters.json").read_text())
            self.assertEqual(config["model_id"], FIRST)
            self.assertIn("autogluon", config)
            self.assertFalse((model / "model_config.json").exists())
            self.assertTrue((model / "code/mlops_models/entrypoints/get_data.py").is_file())
            self.assertFalse((model / "code/mlops_features").exists())

    def test_packaging_preserves_trainer_settings_and_validates_json_before_copying(self):
        from mlops_models.model_base.training.train import write_container_output
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.bin").write_bytes(b"saved-model")
            (checkpoint / "parameters.json").write_text('{"trees": 7}')
            model = write_container_output(checkpoint, {"path": Path("predictor")},
                root / "model", root / "output",
                config={"trees": 99, "model_id": FIRST, "task_type": "regression"},
                performance={"r2": 0.8}, data_usage="validation")
            self.assertEqual(json.loads((model / "parameters.json").read_text()), {"trees": 7})
            self.assertEqual(json.loads((model / "model_metadata.json").read_text()), {"path": "predictor"})

            checkpoint.mkdir()
            (checkpoint / "model.bin").write_bytes(b"saved-model")
            with self.assertRaises(TypeError):
                write_container_output(checkpoint, {}, root / "bad-model", root / "output",
                                       config={"model_id": FIRST, "task_type": "regression", "unsupported": object()},
                                       performance={"r2": 0.8}, data_usage="validation")
            self.assertFalse((root / "bad-model").exists())
            self.assertTrue((checkpoint / "model.bin").is_file())

    def test_dispatcher_resolves_selection_from_training_channels(self):
        from mlops_models.entrypoints.__main__ import parse_job_arguments
        from mlops_models.runtime.job_paths import job_paths
        with patch.object(sys, "argv", ["entrypoint", "train", "--model", FIRST, "--input", "/tmp/channels"]):
            self.assertEqual(parse_job_arguments().selection, "/tmp/channels/selection/selected_features.csv")
        self.assertEqual(job_paths("select-features")[0], "/opt/ml/processing/raw")
        self.assertEqual(job_paths("preprocess-training")[0], "/opt/ml/processing/raw")

    def test_training_loads_only_selected_use_case_parameters_and_applies_trial(self):
        for model_id, task, depth in (
            (FIRST, "regression", 8), (SECOND, "classification", 4),
        ):
            for trial in (None, {"max_depth": "6", "_tuning_objective_metric": "ignored"}):
                with self.subTest(model=model_id, trial=trial), TemporaryDirectory() as directory:
                    root = Path(directory)
                    selection = root / "selected_features.csv"
                    selection.write_text("feature\namount\n")
                    def model_script(trainer, checkpoint, output):
                        config = trainer.config
                        self.assertEqual(config["task_type"], task)
                        parameters = config["autogluon"]
                        self.assertEqual(set(parameters), {task})
                        self.assertEqual(parameters[task]["max_depth"], 6 if trial else depth)
                        self.assertEqual(parameters[task]["n_estimators"], 40)
                        self.assertNotIn("_tuning_objective_metric", parameters[task])
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "model.bin").write_bytes(b"model")
                        return checkpoint, {"evaluation": {"usage": "validation"}}, {"r2": 0.8}
                    with patch(
                        f"mlops_models.models.{model_id}.training.train.UseCaseModelTrainer.train_model",
                        model_script,
                    ):
                        saved = train.execute(model_id, root / "input", root / "model", root / "output",
                                              selection_path=selection, hyperparameters=trial)
                    source = saved / "code/mlops_models/models" / model_id
                    original = json.loads((source / "parameters.json").read_text())
                    self.assertEqual(original["autogluon"][task]["max_depth"], depth)
                    self.assertNotIn("parameters", json.loads((source / "parameters.json").read_text()))

    def test_entrypoints_pass_through_to_use_case_functions_without_creating_outputs(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            selection = Path(directory) / "selected_features.csv"
            selection.write_text("feature\namount\n")
            with patch(f"mlops_models.models.{FIRST}.training.feature_selection.run_feature_selection",
                       return_value="selection-result") as select:
                self.assertEqual(feature_selection.execute(FIRST, "raw", output), "selection-result")
                self.assertEqual(select.call_args.args[:2], (Path("raw"), output))
            with patch(f"mlops_models.models.{FIRST}.training.preprocess.process_train",
                       return_value="preprocess-result") as preprocess:
                self.assertEqual(preprocess_training.execute(FIRST, "raw", "features.csv", output),
                                 "preprocess-result")
                self.assertEqual(preprocess.call_args.args[:3], (Path("raw"), Path("features.csv"), output))
            with patch(f"mlops_models.models.{FIRST}.training.train.train_model",
                       return_value="training-result") as training:
                self.assertEqual(train.execute(FIRST, "input", "model", output,
                                               selection_path=selection), "training-result")
                self.assertEqual(training.call_args.args[:4],
                                 (Path("input/prepared"), selection,
                                  Path("model"), output))
            self.assertFalse(output.exists())

    def test_training_requires_the_supplied_file_and_never_uses_an_alternate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            alternate = root / "input/selection/selected_features.csv"
            alternate.parent.mkdir(parents=True)
            alternate.write_text("feature\nwrong-file\n")
            with patch(f"mlops_models.models.{FIRST}.training.train.train_model") as training:
                with self.assertRaisesRegex(TypeError, "selection_path"):
                    train.execute(FIRST, root / "input", root / "model", root / "output")
                for selection in (root / "missing.csv", ""):
                    with self.subTest(selection=selection):
                        with self.assertRaisesRegex(FileNotFoundError, "supplied path"):
                            train.execute(FIRST, root / "input", root / "model", root / "output",
                                          selection_path=selection)
                training.assert_not_called()
            self.assertFalse((root / "model").exists())
            self.assertFalse((root / "output").exists())

    def test_graph_passes_files_and_one_model_artifact_in_both_training_modes(self):
        body = build_definitions(
            "files", "arn:aws:iam::123456789012:role/jobs", "image:tag", "s3://bucket/files",
            "s3://bucket/source", {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": True},
            region="us-east-1", batch_runtime="processing", include_hpo=True,
            feature_store_config_uri="s3://data/features/config.json")["training"]
        self.assertEqual([s["Name"] for s in body["Steps"]][:4],
                         ["GetData", "FeatureSelection", "Preprocess", "TrainingMode"])
        steps = {s["Name"]: s for s in walk_steps(body["Steps"])}
        self.assertEqual(steps["TrainingMode"]["DependsOn"], ["Preprocess"])
        for name, sources in (("FeatureSelection", {"raw": "GetData"}),
                              ("Preprocess", {"raw": "GetData", "selection": "FeatureSelection"})):
            inputs = {i["InputName"]: i["S3Input"] for i in steps[name]["Arguments"]["ProcessingInputs"]}
            for artifact, producer in sources.items():
                self.assertEqual(inputs[artifact]["LocalPath"], f"/opt/ml/processing/{artifact}")
                self.assertIn("Steps." + producer + ".", inputs[artifact]["S3Uri"]["Get"])
        for name in ("Train", "Tune"):
            job = steps[name]["Arguments"]
            if name == "Tune":
                job = job["TrainingJobDefinition"]
            self.assertEqual([i["ChannelName"] for i in job["InputDataConfig"]], ["prepared", "selection", "source"])
            self.assertEqual(job["Environment"]["MLOPS_TRAINING_INPUT"], "prepared-directory")
            selection = job["InputDataConfig"][1]["DataSource"]["S3DataSource"]["S3Uri"]["Get"]
            self.assertEqual(selection, "Steps.FeatureSelection.ProcessingOutputConfig.Outputs['selection'].S3Output.S3Uri")
        for name in ("Evaluate", "EvaluateTuned"):
            job = steps[name]["Arguments"]
            self.assertEqual(job["Environment"]["MLOPS_EVALUATION_INPUT"], "model-artifact")
            self.assertNotIn("/opt/ml/processing/test", json.dumps(job["ProcessingInputs"]))
        self.assertNotIn("Steps.Prepare.", json.dumps(body))
        for step in walk_steps(body["Steps"]):
            if step["Type"] == "Processing":
                self.assertTrue(all(o["AppManaged"] for o in step["Arguments"]["ProcessingOutputConfig"]["Outputs"]))


if __name__ == "__main__":
    unittest.main()
