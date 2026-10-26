"""Verify independent acquisition, cached producer paths and customer file outputs."""

from importlib import import_module
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from mlops_models.entrypoints import __main__ as dispatcher
from mlops_models.entrypoints import preprocess_inference, postprocess
from mlops_models.model_base.inference import predict
from mlops_models.model_base.inference import preprocess as inference_preprocess
from mlops_models.model_base.inference import postprocess as base_postprocess
from mlops_models.model_base.training import preprocess as training_preprocess
from mlops_models.pipelines import source
from mlops_models.runtime.load_source import upload_processing_outputs, prepare_inference_inputs


class InferenceStages(unittest.TestCase):
    def example_analysis(self, frame, config, name="messaging_final.csv"):
        """Exercise the example through real CSV inputs and written output."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input" / name
            path.parent.mkdir(parents=True)
            frame.to_csv(path, index=False)
            key = config.get("primary_key", "id")
            with patch.dict(base_postprocess.parameters, {"primary_key": key, **config}, clear=True):
                base_postprocess.prediction_inference(None, None, {name: path}, root)
            return pd.read_csv(root / "prediction_analysis.csv", dtype={key: str})

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
        for name in ("GetData", "Preprocess", "Score"):
            self.assertTrue(steps[name]["CacheConfig"]["Enabled"])
        self.assertNotIn("CacheConfig", steps["Postprocess"])
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

    def test_customer_preprocess_writes_uploadable_files_without_returning_a_value(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "parameters.json").write_text('{"model_id":"usecase1","primary_key":"saved_id"}')
            metadata = {"feature_mapping": {"value": "VALUE"}, "primary_key": "metadata_id"}
            (model / "model_metadata.json").write_text(json.dumps(metadata))
            original_artifact = {p.name: p.read_bytes() for p in model.iterdir()}
            parameter_path = root / "source/models/usecase1/parameters.json"
            parameter_path.parent.mkdir(parents=True)
            parameter_path.write_text('{"primary_key":"UPPER_ID"}')
            (root / "data.csv").write_text("lower_id,value\n0001,2\n")
            module = import_module("mlops_models.models.usecase1.inference.preprocess")
            def process(data_path, selection_file, output, config, parameters):
                self.assertEqual((data_path / "data.csv").read_text(), "lower_id,value\n0001,2\n")
                self.assertEqual(config, metadata)  # No merged parameters or nested metadata.
                self.assertEqual(parameters, parameter_path)
                self.assertEqual(json.loads(parameters.read_text()), {"primary_key": "UPPER_ID"})
                self.assertEqual(pd.read_csv(selection_file).to_dict("list"),
                                 {"features": ["value"], "description": ["VALUE"]})
                # Customer code owns normalization and writes files without returning.
                (output / "prepared").mkdir()
                (output / "prepared/data.csv").write_text("UPPER_ID,value\n0001,2\n")
            selection = prepare_inference_inputs(model, root / "output", "usecase1")
            with patch.object(preprocess_inference, "__file__", str(root / "source/entrypoints/preprocess_inference.py")), \
                    patch.object(module, "process_inference", side_effect=process) as call:
                result = preprocess_inference.execute("usecase1", root, selection, root / "output", model_dir=model)
            self.assertIsNone(result)
            self.assertEqual(len(call.call_args.args), 5)
            self.assertEqual(call.call_args.kwargs, {})
            uploaded = {}
            with patch("boto3.client") as client:
                client.return_value.upload_file.side_effect = (
                    lambda filename, bucket, key, **kwargs: uploaded.update({key: Path(filename).read_bytes()}))
                upload_processing_outputs({"Outputs": [
                    {"OutputName": name, "S3Output": {
                        "LocalPath": f"/opt/ml/processing/output/{name}",
                        "S3Uri": f"s3://bucket/run/{name}"}}
                    for name in ("prepared", "selection")
                ]}, Path("/opt/ml/processing/output"), root / "output")
            self.assertEqual(set(uploaded),
                             {"run/prepared/data.csv", "run/selection/selected_features.csv"})
            self.assertEqual(uploaded["run/prepared/data.csv"], b"UPPER_ID,value\n0001,2\n")
            self.assertEqual({p.name: p.read_bytes() for p in model.iterdir()}, original_artifact)

    def test_processing_dispatcher_calls_preprocess_without_formatting_or_batching(self):
        args = SimpleNamespace(step="preprocess", model="usecase1", input="raw",
                               output="output", model_dir="model", selection="selected.csv", chunk_rows=31,
                               analysis_mode="prediction_inference")
        with patch.dict(os.environ, {"MLOPS_PREPARE_TRANSFORM_INPUT": ""}), \
                patch.object(preprocess_inference, "execute", return_value=None) as call, \
                patch("mlops_models.runtime.transform_input.prepare_transform_input") as formatter:
            self.assertIsNone(dispatcher.dispatch_pipeline_step(args))
            call.assert_called_once_with("usecase1", "raw", "selected.csv", "output", model_dir="model",
                                         analysis_mode="prediction_inference")
            formatter.assert_not_called()

    def test_execution_mode_selects_the_matching_use_case_preprocessor(self):
        import sys

        for model_id in ("usecase1", "usecase2"):
            module = import_module(f"mlops_models.models.{model_id}.inference.preprocess")
            for mode, method in (("prediction_inference", "process_inference"),
                                 ("prediction_test", "process_test")):
                with self.subTest(model=model_id, mode=mode), TemporaryDirectory() as directory:
                    root = Path(directory)
                    model = root / "model"
                    model.mkdir()
                    metadata = {"feature_mapping": {"value": "value"}}
                    (model / "model_metadata.json").write_text(json.dumps(metadata))
                    parameter_path = root / "source/models" / model_id / "parameters.json"
                    parameter_path.parent.mkdir(parents=True)
                    parameter_path.write_text('{"primary_key":"id"}')
                    (root / "data.csv").write_text("id,value,target\n0001,2,1\n")
                    selected = root / "selected.csv"
                    selected.write_text("feature\nvalue\n")
                    output = root / "output"

                    def process(data, selection, destination, config, parameters_path):
                        self.assertEqual(config, metadata)
                        self.assertEqual(parameters_path, parameter_path)
                        frame, _ = inference_preprocess.get_container_input(
                            data, selection, json.loads(parameters_path.read_text()))
                        inference_preprocess.write_container_output(
                            frame.assign(chosen_mode=mode), destination)

                    other = "process_test" if method == "process_inference" else "process_inference"
                    command = ["step", "preprocess", "--model", model_id, "--input", str(root),
                               "--output", str(output), "--selection", str(selected),
                               "--model-dir", str(model)]
                    environment = {} if mode == "prediction_inference" else {"MLOPS_ANALYSIS_MODE": mode}
                    with patch.dict(os.environ, environment, clear=True), \
                            patch.object(sys, "argv", command), \
                            patch.object(preprocess_inference, "__file__",
                                         str(root / "source/entrypoints/preprocess_inference.py")), \
                            patch.object(module, method, process), \
                            patch.object(module, other, side_effect=AssertionError("Wrong preprocessor")):
                        dispatcher.main()
                    result = pd.read_csv(output / "prepared/data.csv", dtype={"id": str})
                    self.assertEqual(result["id"].tolist(), ["0001"])
                    self.assertEqual(result["chosen_mode"].tolist(), [mode])

    def test_test_preprocessor_placeholder_writes_the_same_prepared_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.csv").write_text("id,value,target,unused\n0001,2,1,9\n")
            selection = root / "selected.csv"
            selection.write_text("feature\nvalue\n")
            parameter_path = root / "parameters.json"
            parameter_path.write_text('{"primary_key":"id","target_key":"target"}')
            inference_preprocess.process_test(root, selection, root / "output", {}, parameter_path)
            self.assertEqual((root / "output/prepared/data.csv").read_text(),
                             "id,value,target\n0001,2,1\n")

    def test_base_routers_keep_default_file_outputs_for_both_modes(self):
        for mode in ("prediction_inference", "prediction_test"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "data.csv").write_text("id,value,target,unused\n0001,2,1,9\n")
                selection = root / "selected.csv"
                selection.write_text("feature\nvalue\n")
                parameter_path = root / "parameters.json"
                parameter_path.write_text(
                    '{"primary_key":"id","target_key":"target","task_type":"regression"}')
                metadata = {"feature_mapping": {"value": "value"}}
                (root / "model_metadata.json").write_text(json.dumps(metadata))
                inference_preprocess.preprocess_data(
                    root, selection, root / "preprocessed", metadata, parameter_path,
                    analysis_mode=mode)
                self.assertEqual((root / "preprocessed/prepared/data.csv").read_text(),
                                 "id,value,target\n0001,2,1\n")
                scores = root / "scores"
                scores.mkdir()
                (scores / "messaging_final.csv").write_text("id,score\n0001,0.8\n")
                with patch.dict(os.environ, {}, clear=True):
                    output = base_postprocess.postprocess_predictions(
                        root, selection, scores, root / "analysis", root, analysis_mode=mode)
                result = pd.read_csv(output / "prediction_analysis.csv", dtype={"id": str})
                self.assertEqual(result.to_dict("records"), [{"id": "0001", "PREDICTION": 0.8}])
                self.assertEqual(json.loads((output / "summary.json").read_text()),
                                 {"rows": 1, "task": "regression", "file": "prediction_analysis.csv"})

    def test_invalid_preprocessing_mode_stops_before_running_model_functions(self):
        modules = [inference_preprocess] + [
            import_module(f"mlops_models.models.{model_id}.inference.preprocess")
            for model_id in ("usecase1", "usecase2")
        ]
        for module in modules:
            with self.subTest(module=module.__name__), \
                    patch.object(module, "process_inference") as inference, \
                    patch.object(module, "process_test") as test:
                with self.assertRaisesRegex(ValueError, "analysis_mode"):
                    module.preprocess_data("raw", "selected.csv", "output", {},
                                           "parameters.json", analysis_mode="typo")
                inference.assert_not_called()
                test.assert_not_called()

    def test_entrypoints_still_allow_complete_use_case_router_replacements(self):
        for model_id in ("usecase1", "usecase2"):
            with self.subTest(model=model_id), TemporaryDirectory() as directory:
                root = Path(directory)
                metadata = {"feature_mapping": {"value": "value"}}
                (root / "model_metadata.json").write_text(json.dumps(metadata))
                pre = import_module(f"mlops_models.models.{model_id}.inference.preprocess")
                post = import_module(f"mlops_models.models.{model_id}.inference.postprocess")
                parameter_path = root / "source/models" / model_id / "parameters.json"
                args = (root / "raw", root / "selected.csv", root / "prepared",
                        metadata, parameter_path)
                result = object()
                with patch.object(preprocess_inference, "__file__",
                                  str(root / "source/entrypoints/preprocess_inference.py")), \
                        patch.object(pre, "preprocess_data", return_value=result) as replaced:
                    self.assertIs(preprocess_inference.execute(
                        model_id, *args[:3], model_dir=root, analysis_mode="prediction_test"), result)
                    replaced.assert_called_once_with(*args, analysis_mode="prediction_test")
                with patch.object(post, "postprocess_predictions", return_value=result) as replaced:
                    self.assertIs(postprocess.execute(
                        model_id, root / "scores", root / "analysis", model_dir=root,
                        data_directory=root / "raw", selection_file=root / "selected.csv",
                        analysis_mode="prediction_test", chunk_rows=7), result)
                    replaced.assert_called_once_with(
                        root / "raw", root / "selected.csv", root / "scores", root / "analysis",
                        root, analysis_mode="prediction_test", chunk_rows=7)

    def test_training_and_inference_read_two_frames_and_publish_the_same_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            params = {"primary_key": "id", "timestamp": "date", "target_key": "target"}
            parameter_path = root / "parameters.json"
            parameter_path.write_text(json.dumps(params))
            selection_file = root / "selected_features.csv"
            selection_file.write_text("features,description\nvalue,VALUE\n")
            for labeled in (True, False):
                source = "id,value,unused" + (",target" if labeled else "") + "\n0001,2,99" + (",1" if labeled else "") + "\n"
                (root / "data.csv").write_text(source)
                written = []
                for name, module, process in (
                        ("train", training_preprocess, training_preprocess.process_train),
                        ("inference", inference_preprocess, inference_preprocess.process_inference)):
                    with self.subTest(labeled=labeled, mode=name):
                        data, selection = module.get_container_input(root, selection_file, params)
                        self.assertIsInstance(data, pd.DataFrame)
                        self.assertEqual(data["id"].tolist(), ["0001"])
                        self.assertEqual(selection["features"].tolist(), ["value"])
                        self.assertIsNone(process(root, selection_file, root / name, {}, parameter_path))
                        written.append((root / name / "prepared/data.csv").read_bytes())
                self.assertEqual(written[0], written[1])
                self.assertEqual(written[0], (
                    b"id,value,target\n0001,2,1\n" if labeled else b"id,value\n0001,2\n"))

    def test_transform_formatting_runs_only_after_customer_preprocessing_writes_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_path = root / "source/models/usecase1/parameters.json"
            parameter_path.parent.mkdir(parents=True)
            parameter_path.write_text('{"primary_key":"id"}')
            prepared = root / "output/prepared"
            def process(*args, **kwargs):
                prepared.mkdir(parents=True)
                (prepared / "data.csv").write_text("id,value,text\n0001,2,NA\n0002,,001\n")
            args = SimpleNamespace(step="preprocess", model="usecase1", input=str(root / "raw"),
                                   output=str(root / "output"), model_dir=str(root / "model"), selection="selected.csv",
                                   analysis_mode="prediction_inference")
            with patch.dict(os.environ, {"MLOPS_PREPARE_TRANSFORM_INPUT": "1"}), \
                    patch.object(dispatcher, "__file__", str(root / "source/entrypoints/__main__.py")), \
                    patch.object(preprocess_inference, "execute", side_effect=process):
                self.assertIsNone(dispatcher.dispatch_pipeline_step(args))
            formatted = prepared / "shards/part-000000.jsonl"
            self.assertEqual([json.loads(line) for line in formatted.read_text().splitlines()],
                             [{"id": "0001", "value": 2, "text": "NA"},
                              {"id": "0002", "value": None, "text": "001"}])
            steps = {s["Name"]: s for s in self.definitions()["inference"]["Steps"]}
            self.assertNotIn("MLOPS_PREPARE_TRANSFORM_INPUT", steps["Preprocess"]["Arguments"]["Environment"])
            self.assertNotIn("--chunk-rows", json.dumps(steps["Preprocess"]["Arguments"]["AppSpecification"]))
            definitions = source.build_definitions(
                "review", "arn:aws:iam::123456789012:role/jobs", "image:tag",
                "s3://bucket/review", "s3://bucket/review/source",
                {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": True},
                region="us-east-1", batch_runtime="transform", data_source="csv_direct",
                csv_input_uri="s3://bucket/scoring.csv")
            native = next(s for s in definitions["inference"]["Steps"] if s["Name"] == "Preprocess")
            self.assertEqual(native["Arguments"]["Environment"]["MLOPS_PREPARE_TRANSFORM_INPUT"], "1")

    def test_inference_metadata_must_come_from_matching_registered_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "parameters.json").write_text('{"model_id":"usecase2"}')
            with self.assertRaisesRegex(ValueError, "belongs to usecase2"):
                prepare_inference_inputs(model, root / "output", "usecase1")
            (model / "parameters.json").write_text('{"model_id":"usecase1"}')
            # Even if source metadata exists, it cannot replace missing learned metadata.
            source_metadata = root / "source/models/usecase1/model_metadata.json"
            source_metadata.parent.mkdir(parents=True)
            source_metadata.write_text('{"feature_mapping":{"value":"VALUE"}}')
            with patch.object(preprocess_inference, "__file__", str(root / "source/entrypoints/preprocess_inference.py")):
                with self.assertRaises(FileNotFoundError):
                    prepare_inference_inputs(model, root / "output", "usecase1")

    def test_prediction_files_and_analysis_preserve_all_three_variable_pairs(self):
        for task, column in (("regression", "prediction"), ("regression", "PREDICTION"),
                             ("classification", "score"), ("classification", "SCORE")):
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
                    def __init__(self, checkpoint_path, config, parameters):
                        self.checkpoint_path = checkpoint_path
                    def predict_file(self, path, feature_mapping, primary_key, *, output_directory):
                        if path != prepared / "data.csv":
                            raise AssertionError(path)
                        destination = output_directory / "native/output"
                        destination.mkdir(parents=True)
                        scores.to_csv(destination / "predictions.csv", index=False)
                        messaging.to_csv(destination / "messaging.csv", index=False)
                        final.to_csv(destination / "messaging_final.csv", index=False)
                with patch.object(predict, "ModelPredictor", CustomerPredictor):
                    output = predict.predict_with_model(prepared, model, root / "scores",
                                                        {"feature_mapping": {"value": "value"}},
                                                        model / "parameters.json")
                self.assertEqual(pd.read_csv(output / "native/output/messaging_final.csv").columns.tolist(), final.columns.tolist())
                result = postprocess.execute("usecase1", output, root / "analysis", model_dir=model,
                                             data_directory=raw, selection_file=selection)
                data = pd.read_csv(result / "prediction_analysis.csv", dtype={"id": str})
                self.assertEqual(data["id"].tolist(), ["0001"])
                self.assertEqual(data["PREDICTION"].tolist(), [0.8 if task == "regression" else 1])
                if task == "classification":
                    self.assertEqual(data["PROBABILITY"].tolist(), [0.8])
                for n in range(1, 4):
                    self.assertEqual(data[f"variable_{n}_value"].tolist(), [n])
                self.assertEqual(result, root / "analysis")
                (output / "native/output/messaging.csv").unlink()
                # This analysis only needs the combined result, not a fixed trio of files.
                postprocess.execute("usecase1", output, root / "without-messaging", model_dir=model,
                                    data_directory=raw, selection_file=selection)
                (output / "native/output/messaging_final.csv").unlink()
                with self.assertRaisesRegex(ValueError, "needs exactly one messaging_final.csv"):
                    postprocess.execute("usecase1", output, root / "missing", model_dir=model,
                                         data_directory=raw, selection_file=selection)

    def test_postprocess_score_casing_preserves_values_ids_and_explanations(self):
        for score_column in ("SCORE", "score", "Score"):
            with self.subTest(column=score_column):
                frame = pd.DataFrame({
                    "CAD_ECIF_ID": ["0001", "0002", "0003"],
                    "VARIABLE_1": ["feature_a"] * 3,
                    "VARIABLE_1_VALUE": [9.0, 8.0, 7.0],
                    score_column: [0.69, 0.7, 0.9],
                })
                before = frame.copy(deep=True)
                log = io.StringIO()
                with redirect_stdout(log):
                    result = self.example_analysis(frame, {
                        "primary_key": "CAD_ECIF_ID", "task_type": "classification", "threshold": 0.7,
                    }, name="output/messaging_final.csv")
                self.assertEqual(result["PREDICTION"].tolist(), [0, 1, 1])
                self.assertEqual(result["PROBABILITY"].tolist(), [0.69, 0.7, 0.9])
                pd.testing.assert_frame_equal(result.iloc[:, :3], before.iloc[:, :3])
                pd.testing.assert_frame_equal(frame, before)
                self.assertIn(f"Analysis input output/messaging_final.csv (classification): {list(before.columns)}", log.getvalue())
                self.assertIn("First scores: {'PROBABILITY': '0.69', 'PREDICTION': '0'}", log.getvalue())
                self.assertNotIn("0001", log.getvalue())

    def test_postprocess_keeps_regression_estimates_and_existing_class_labels(self):
        for column in ("PREDICTION", "prediction", "SCORE"):
            with self.subTest(column=column):
                frame = pd.DataFrame({"id": ["001", "002"], column: [-2.5, 200.25]})
                result = self.example_analysis(frame, {"task_type": "regression"})
                self.assertEqual(result["PREDICTION"].tolist(), [-2.5, 200.25])
                self.assertNotIn("PROBABILITY", result)
                self.assertEqual(result["id"].tolist(), ["001", "002"])
        frame = pd.DataFrame({"id": ["001"], "prediction": ["positive"], "probability": [0.8]})
        result = self.example_analysis(frame, {"task_type": "classification"})
        self.assertEqual(result["PREDICTION"].tolist(), ["positive"])
        self.assertEqual(result["PROBABILITY"].tolist(), [0.8])

    def test_postprocess_rejects_ambiguous_or_invalid_score_inputs(self):
        cases = [
            ({"score": [0.2], "SCORE": [0.8]}, "Ambiguous SCORE"),
            ({"id": ["001"]}, "needs PREDICTION or SCORE"),
            ({"SCORE": [float("nan")]}, "probabilities between"),
            ({"SCORE": [1.2]}, "probabilities between"),
        ]
        for columns, message in cases:
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, message):
                self.example_analysis(pd.DataFrame(columns), {"task_type": "classification"})

    def test_postprocess_keeps_duplicate_basenames_separate_and_rejects_ambiguous_example_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "raw/data.csv").write_text("id,value\n0001,2\n")
            selection = root / "selected.csv"
            selection.write_text("features\nvalue\n")
            for part, score in (("first", 0.2), ("second", 0.8)):
                folder = root / "scores" / part
                folder.mkdir(parents=True)
                (folder / "messaging_final.csv").write_text(f"id,score\n0001,{score}\n")
            (root / "parameters.json").write_text('{"project":"review","test_ymid":"202610"}')
            (root / "model_metadata.json").write_text('{"feature_mapping":{"value":"value"}}')
            with patch.object(base_postprocess.pd, "read_csv", side_effect=AssertionError("Paths only")):
                data_path, selected, predictions, output, parameters = base_postprocess.get_container_input(
                    root / "raw", selection, root / "scores", root / "output", root)
            self.assertEqual(parameters["project"], "review")
            self.assertEqual(parameters["test_ymid"], "202610")
            self.assertEqual(output, root / "output")
            self.assertEqual(list(predictions), ["first/messaging_final.csv", "second/messaging_final.csv"])
            self.assertEqual(selected, selection)
            self.assertEqual(predictions["first/messaging_final.csv"], root / "scores/first/messaging_final.csv")
            self.assertEqual(predictions["second/messaging_final.csv"], root / "scores/second/messaging_final.csv")
            with patch.dict(base_postprocess.parameters, {"primary_key": "id", "task_type": "regression"}, clear=True):
                with self.assertRaisesRegex(ValueError, "found 2"):
                    base_postprocess.prediction_inference(data_path, selected, predictions, root / "output")

    def test_postprocess_reports_when_no_prediction_csv_exists(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.csv").write_text("id,value\n0001,2\n")
            selection = root / "selection.csv"
            selection.write_text("features\nvalue\n")
            with self.assertRaisesRegex(FileNotFoundError, "no prediction CSV"):
                base_postprocess.get_container_input(root, selection, root / "missing", root / "output", root)


if __name__ == "__main__":
    unittest.main()
