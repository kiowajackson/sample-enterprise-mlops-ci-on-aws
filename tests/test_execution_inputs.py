"""Exercise execution-selected training modes and independent metadata inputs."""

from copy import deepcopy
from importlib import import_module
import io
import json
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from mlops_models.entrypoints import preprocess_training, train
from mlops_models.entrypoints import __main__ as dispatcher
from mlops_models.pipelines.source import build_definitions
from mlops_models.registry.model_package import resolve_latest_approved_model
from mlops_models.runtime.load_source import prepare_inference_inputs, upload_processing_outputs
from mlops_models.storage import read_model_metadata
from mlops_skeleton.pipelines import walk_steps
from test_development_registry import package


MODES = {"Preliminary models with validation": "1", "Final model with all data": "2"}


class ExecutionInputs(unittest.TestCase):
    def test_published_modes_metadata_and_hpo_are_independent(self):
        for runtime in ("processing", "transform", "both"):
            with self.subTest(runtime=runtime):
                definitions = build_definitions(
                    "review", "arn:aws:iam::123456789012:role/jobs", "image",
                    "s3://bucket/team", "s3://bucket/source",
                    {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": True},
                    region="us-east-1", batch_runtime=runtime, models=["usecase1"], include_hpo=True,
                    data_source="csv_direct", csv_input_uri="s3://bucket/input.csv",
                    step_cache_revisions={name: "v1" for name in ("Preprocess", "Train", "Score")},
                )
                training, inference = definitions["training"], definitions["inference"]
                params = {p["Name"]: p for p in training["Parameters"]}
                self.assertEqual(params["TrainingMode"]["EnumValues"], list(MODES))
                self.assertEqual(params["TrainingMode"]["DefaultValue"], next(iter(MODES)))
                self.assertEqual(params["ModelMetadataUri"]["DefaultValue"], "none")
                steps = {s["Name"]: s for s in walk_steps(training["Steps"])}
                for name in ("Train", "Tune", "Preprocess"):
                    job = steps[name]["Arguments"]
                    if name == "Tune":
                        job = job["TrainingJobDefinition"]
                    self.assertEqual(job["Environment"]["MLOPS_MODEL_METADATA_URI"],
                                     {"Get": "Parameters.ModelMetadataUri"})
                    if name != "Preprocess":
                        self.assertNotIn("MLOPS_TRAINING_MODE", job["Environment"])
                        self.assertEqual(job["Environment"]["MLOPS_TRAINING_SELECTION"],
                                         {"Get": "Parameters.TrainingMode"})
                metadata_parameters = [p for p in inference["Parameters"] if p["Name"] == "ModelMetadataUri"]
                self.assertEqual(len(metadata_parameters), 1)
                self.assertEqual(metadata_parameters[0]["DefaultValue"], "registry")
                resolver = next(s for s in inference["Steps"] if s["Name"] == "ResolveModel")
                self.assertNotIn("CacheConfig", resolver)
                self.assertEqual(resolver["Arguments"]["Environment"]["MLOPS_MODEL_METADATA_URI"],
                                 {"Get": "Parameters.ModelMetadataUri"})
                # Both Processing model inputs and native CreateModel read the
                # resolver's effective checkpoint, never a separate stale archive.
                for step in walk_steps(inference["Steps"]):
                    args = step["Arguments"]
                    if step["Type"] == "Model":
                        checkpoint = args["PrimaryContainer"]["ModelDataUrl"]
                        self.assertEqual(checkpoint["Std:JsonGet"]["Path"], "checkpoint")
                    for item in args.get("ProcessingInputs", []):
                        if item["InputName"] == "model":
                            self.assertEqual(item["S3Input"]["S3Uri"]["Std:JsonGet"]["Path"], "checkpoint")

    def test_all_four_training_combinations_reach_customer_class_and_preprocess(self):
        initial = {"best_config": {"depth": 3}}
        for model_id in ("usecase1", "usecase2"):
            for mode, flag in MODES.items():
                for uri in ("none", "", "s3://bucket/metadata.json"):
                    with self.subTest(model=model_id, mode=mode, metadata=uri), TemporaryDirectory() as temp:
                        root = Path(temp)
                        folder = root / "source/models" / model_id
                        folder.mkdir(parents=True)
                        parameters = {"primary_key": "id", "task_type": "regression"}
                        (folder / "parameters.json").write_text(json.dumps(parameters))
                        # Explicit blank must not accidentally pick up a bundled file.
                        (folder / "model_metadata.json").write_text('{"wrong": true}')
                        (root / "selection.csv").write_text("feature\nvalue\n")
                        expected = initial if uri.startswith("s3://") else {}
                        trained = import_module(f"mlops_models.models.{model_id}.training.train")
                        prepared = import_module(f"mlops_models.models.{model_id}.training.preprocess")
                        seen = []

                        def fit(trainer, checkpoint, output):
                            seen.append(deepcopy(trainer.config))
                            self.assertEqual(os.environ["MLOPS_TRAINING_MODE"], flag)
                            self.assertNotIn("user_final_model", trainer.parameters)
                            self.assertEqual(trainer.config, expected)
                            checkpoint.mkdir(parents=True)
                            (checkpoint / "weights").write_bytes(b"fitted model")
                            return checkpoint, {"newly_trained": True}

                        s3 = Mock()
                        s3.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(json.dumps(initial).encode())}
                        with patch.dict(os.environ, {"MLOPS_TRAINING_SELECTION": mode, "MLOPS_MODEL_METADATA_URI": uri}), \
                                patch("boto3.client", return_value=s3), \
                                patch.object(train, "__file__", str(root / "source/entrypoints/train.py")), \
                                patch.object(preprocess_training, "__file__", str(root / "source/entrypoints/preprocess_training.py")), \
                                patch.object(prepared, "process_train") as process, \
                                patch.object(trained.UseCaseModelTraining, "train_model", fit):
                            preprocess_training.execute(model_id, root / "raw", root / "selection.csv", root / "prepared")
                            self.assertEqual(process.call_args.args[3], expected)
                            model = dispatcher.dispatch_pipeline_step(SimpleNamespace(
                                step="train", model=model_id, input=root / "prepared",
                                model_dir=root / "model", output=root / "output",
                                selection=root / "selection.csv"))
                        self.assertEqual(seen, [expected])
                        self.assertEqual(json.loads((model / "parameters.json").read_text()),
                                         {**parameters, "model_id": model_id})
                        self.assertEqual(json.loads((model / "model_metadata.json").read_text()), {"newly_trained": True})
                        self.assertEqual(json.loads((folder / "parameters.json").read_text()), parameters)
                        if not uri.startswith("s3://"):
                            s3.get_object.assert_not_called()

    def test_bad_metadata_does_not_fall_back_to_absent(self):
        for content in (b"{broken", b"[]", b"null"):
            with self.subTest(content=content):
                s3 = Mock()
                s3.get_object.return_value = {"Body": io.BytesIO(content)}
                with self.assertRaises(ValueError):
                    read_model_metadata("s3://bucket/metadata.json", s3=s3)
        for uri in ("file:///tmp/metadata.json", "s3://bucket/", "s3://bucket/meta.json?versionId=1"):
            with self.subTest(uri=uri):
                s3 = Mock()
                with self.assertRaises(ValueError):
                    read_model_metadata(uri, s3=s3)
                s3.get_object.assert_not_called()
        s3 = Mock()
        failure = ClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}}, "GetObject")
        s3.get_object.side_effect = failure
        with self.assertRaises(ClientError) as caught:
            read_model_metadata("s3://bucket/metadata.json", s3=s3)
        self.assertIs(caught.exception, failure)

    def test_unknown_training_mode_stops_before_fitting(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            parameters = root / "source/models/usecase1/parameters.json"
            parameters.parent.mkdir(parents=True)
            parameters.write_text("{}")
            (root / "selection.csv").write_text("feature\nvalue\n")
            module = import_module("mlops_models.models.usecase1.training.train")
            with patch.dict(os.environ, {"MLOPS_TRAINING_MODE": "full", "MLOPS_MODEL_METADATA_URI": ""}), \
                    patch.object(train, "__file__", str(root / "source/entrypoints/train.py")), \
                    patch.object(module.UseCaseModelTraining, "train_model") as fit:
                with self.assertRaisesRegex(ValueError, "Unknown TrainingMode"):
                    train.execute("usecase1", root, root / "model", root / "out", selection_path=root / "selection.csv")
                fit.assert_not_called()

    def test_mode_code_is_available_before_model_code_and_independent_of_hpo(self):
        args = SimpleNamespace(step="train", model="usecase1", input="prepared",
                               model_dir="model", output="output", selection="selection.csv")
        for label, code in MODES.items():
            for tuning in (False, True):
                with self.subTest(label=label, tuning=tuning):
                    environment = {"MLOPS_TRAINING_SELECTION": label, "MLOPS_TRAINING_MODE": "stale"}
                    if tuning:
                        environment["TUNING_OBJECTIVE"] = "enabled"
                    hyperparameters = {"max_depth": "4", "mode": "model-owned"}

                    def execute(*args, **kwargs):
                        self.assertEqual(os.environ["MLOPS_TRAINING_MODE"], code)
                        self.assertEqual(kwargs["hyperparameters"], hyperparameters if tuning else None)
                        return "trained"

                    with patch.dict(os.environ, environment, clear=True), \
                            patch.object(Path, "read_text", return_value=json.dumps(hyperparameters)), \
                            patch.object(train, "execute", side_effect=execute):
                        self.assertEqual(dispatcher.dispatch_pipeline_step(args), "trained")

    def test_direct_codes_and_older_pipeline_labels_normalize_before_training(self):
        args = SimpleNamespace(step="train", model="usecase1", input="prepared",
                               model_dir="model", output="output", selection="selection.csv")
        cases = [({}, "1"), ({"MLOPS_TRAINING_MODE": "2"}, "2")]
        cases.extend(({"MLOPS_TRAINING_MODE": label}, code) for label, code in MODES.items())
        for environment, code in cases:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch.object(train, "execute") as execute:
                dispatcher.dispatch_pipeline_step(args)
                self.assertEqual(os.environ["MLOPS_TRAINING_MODE"], code)
                execute.assert_called_once()

    def test_inference_override_preserves_weights_source_parameters_and_registry(self):
        original = package(4)
        original["CustomerMetadataProperties"].update(
            ModelMetadataUri="s3://bucket/original.json", ModelMetadataSha256="c" * 64)
        files = {
            "parameters.json": b'{"model_id":"usecase2","threshold":0.7}',
            "./model_metadata.json": b'{"feature_mapping":{"old":"OLD"}}',
            "predictor/weights": b"fitted weights\x00\x01",
            "code/mlops_models/entrypoints/serve.py": b"# older unchanged serving code\n",
        }
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                if name.endswith("model_metadata.json"):
                    member.pax_headers = {"size": str(len(content))}
                tar.addfile(member, io.BytesIO(content))
        metadata = b'{ "feature_mapping": {"value": "VALUE"}, "predictor_path": "predictor" }\n'
        client, s3 = Mock(), Mock()
        client.describe_model_package.return_value = deepcopy(original)
        objects = {"models/4/model.tar.gz": archive.getvalue(), "override.json": metadata}
        s3.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(objects[kw["Key"]])}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            record = resolve_latest_approved_model(
                "shared", root, namespace="parul", model_id="usecase2", environment="dev",
                package_arn=original["ModelPackageArn"], client=client, s3=s3,
                metadata_uri="s3://bucket/override.json",
                metadata_checkpoint_uri="s3://bucket/runs/new/metadata-override/model.tar.gz")
            self.assertEqual(record["registered_checkpoint"], original["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])
            self.assertEqual(record["registered_model_metadata"], "s3://bucket/original.json")
            self.assertTrue(record["metadata_override"])
            self.assertEqual(record["checkpoint"], "s3://bucket/runs/new/metadata-override/model.tar.gz")
            with tarfile.open(root / "metadata-override/model.tar.gz") as tar:
                for member in tar:
                    self.assertEqual(tar.extractfile(member).read(),
                                     metadata if member.name == "./model_metadata.json" else files[member.name])
                tar.extractall(root / "effective", filter="data")
            selection = prepare_inference_inputs(root / "effective", root / "prepared", "usecase2")
            self.assertIn("value,VALUE", selection.read_text())
            self.assertEqual(client.describe_model_package.return_value, original)
            s3.put_object.assert_not_called()
            s3.upload_file.assert_not_called()  # The normal Processing uploader owns publication.
            uploaded = {}
            s3.upload_file.side_effect = lambda filename, bucket, key, **kw: uploaded.update({
                f"s3://{bucket}/{key}": Path(filename).read_bytes()})
            with patch("boto3.client", return_value=s3):
                upload_processing_outputs({"Outputs": [{
                    "OutputName": "authorization", "S3Output": {
                        "LocalPath": "/opt/ml/processing/output",
                        "S3Uri": "s3://bucket/runs/new"}}]},
                    Path("/opt/ml/processing/output"), root)
            self.assertEqual(uploaded[record["checkpoint"]], (root / "metadata-override/model.tar.gz").read_bytes())

    def test_blank_inference_metadata_preserves_registry_artifact_without_s3_calls(self):
        original = package(4)
        client, s3 = Mock(), Mock()
        client.describe_model_package.return_value = original
        with TemporaryDirectory() as temp:
            record = resolve_latest_approved_model(
                "shared", temp, namespace="parul", model_id="usecase2", environment="dev",
                package_arn=original["ModelPackageArn"], client=client, s3=s3, metadata_uri="registry")
            self.assertEqual(record["checkpoint"], original["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])
            self.assertNotIn("metadata_override", record)
            self.assertFalse((Path(temp) / "metadata-override").exists())
        self.assertEqual(s3.mock_calls, [])

    def test_metadata_override_does_not_bypass_package_approval(self):
        original = package(4)
        original["ModelApprovalStatus"] = "Rejected"
        client, s3 = Mock(), Mock()
        client.describe_model_package.return_value = original
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "No eligible"):
                resolve_latest_approved_model(
                    "shared", temp, namespace="parul", model_id="usecase2", environment="dev",
                    package_arn=original["ModelPackageArn"], client=client, s3=s3,
                    metadata_uri="s3://bucket/override.json",
                    metadata_checkpoint_uri="s3://bucket/runs/new/model.tar.gz")
            self.assertFalse((Path(temp) / "package.json").exists())
        self.assertEqual(s3.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
