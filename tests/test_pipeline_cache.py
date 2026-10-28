"""Cache training and downstream jobs without losing their original artifacts."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from mlops_models.pipelines import source
from mlops_models.runtime import load_source
from mlops_skeleton.pipelines import walk_steps


class PipelineCache(unittest.TestCase):
    def definitions(self, **options):
        return source.build_definitions(
            "cache", "arn:aws:iam::123456789012:role/jobs", "image@sha256:abc",
            "s3://bucket/cache", "s3://bucket/cache/source",
            {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": False},
            region="us-east-1", data_source="csv_direct", csv_input_uri="s3://bucket/data.csv",
            batch_runtime="processing",
            step_cache_revisions={name: name + "-revision" for name in (
                "GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate",
                "BatchGetData", "BatchPreprocess", "Score", "Postprocess")},
            **options,
        )

    def test_training_and_reports_reference_the_actual_producer(self):
        bodies = self.definitions()
        steps = {step["Name"]: step for step in walk_steps(bodies["training"]["Steps"])}
        model = source.join({"Get": "Steps.Train.OutputDataConfig.S3OutputPath"}, "application/model.tar.gz")
        report = source.join(
            {"Get": "Steps.Evaluate.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri"},
            "evaluation.json")
        for name in ("GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate"):
            step = steps[name]
            self.assertTrue(step["CacheConfig"]["Enabled"])
            self.assertNotIn("Execution.PipelineExecutionId", json.dumps(step["Arguments"]["Environment"]))
        train = steps["Train"]["Arguments"]
        self.assertEqual(train["Environment"]["MLOPS_TRAINING_OUTPUT_UPLOAD"], "application")
        self.assertNotIn("MLOPS_TRAINING_MODEL_URI", train["Environment"])
        self.assertIn("Execution.PipelineExecutionId", json.dumps(train["OutputDataConfig"]))
        evaluate = steps["Evaluate"]["Arguments"]
        self.assertEqual(evaluate["ProcessingInputs"][0]["S3Input"]["S3Uri"], model)
        self.assertEqual(evaluate["Environment"]["CHECKPOINT_URI"], model)
        self.assertEqual(evaluate["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "pipeline")
        self.assertEqual(steps["QualityGate"]["Arguments"]["Conditions"][0]["LeftValue"]["Std:JsonGet"]["S3Uri"], report)
        register = steps["RegisterModel"]["Arguments"]
        self.assertEqual(register["Environment"]["EVALUATION_URI"], report)
        self.assertEqual(register["ProcessingInputs"][1]["S3Input"]["S3Uri"], report)
        self.assertNotIn("MLOPS_STEP_REVISION", register["Environment"])
        self.assertEqual(register["RoleArn"], "arn:aws:iam::123456789012:role/jobs")
        self.assertEqual(register["Environment"]["PYTHONPATH"], "/opt/ml/processing/source")
        self.assertEqual(register["Environment"]["PYTHONSAFEPATH"], "1")
        for name in ("QualityGate", "RegisterModel"):
            self.assertNotIn("CacheConfig", steps[name])

    def test_batch_compute_is_cached_but_approval_lookup_is_live(self):
        for code in ("artifact", "pipeline"):
            body = self.definitions(model_code_source=code)["inference"]
            steps = {step["Name"]: step for step in body["Steps"]}
            for name in ("GetData", "Preprocess", "Score"):
                self.assertTrue(steps[name]["CacheConfig"]["Enabled"])
                environment = steps[name]["Arguments"]["Environment"]
                # Registry JsonGet expressions contain the current decision
                # file URI, but resolve to the selected image/checkpoint value.
                self.assertNotIn("PIPELINE_EXECUTION_ID", environment)
                self.assertEqual(environment["MLOPS_MODEL_CODE_SOURCE"], {"Get": "Parameters.ModelCodeSource"})
            self.assertNotIn("CacheConfig", steps["ResolveModel"])
            self.assertNotIn("CacheConfig", steps["Postprocess"])
            self.assertEqual(steps["Postprocess"]["Arguments"]["Environment"]["MLOPS_MODEL_CODE_SOURCE"], {"Get": "Parameters.ModelCodeSource"})

    def test_evaluation_edits_preserve_training_and_postprocess_edits_preserve_scoring(self):
        with TemporaryDirectory() as work:
            package = Path(work) / "mlops_models"
            shutil.copytree(source.PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            csv = Path(work) / "data.csv"
            csv.write_text("x,y\n1,2\n")
            with patch.object(source, "PACKAGE", package):
                fingerprint = lambda: source.pipeline_fingerprints(Mock(), "s3://bucket/data.csv", csv)
                initial = fingerprint()
                evaluate = package / "model_base/training/evaluate.py"
                evaluate.write_text(evaluate.read_text() + "\n# new metric\n")
                changed = fingerprint()
                self.assertEqual({key for key in initial if initial[key] != changed[key]}, {"Evaluate"})
                initial = changed
                postprocess = package / "model_base/inference/postprocess.py"
                postprocess.write_text(postprocess.read_text() + "\n# new formatting\n")
                changed = fingerprint()
                self.assertEqual({key for key in initial if initial[key] != changed[key]},
                                 {"Train", "Evaluate", "Postprocess"})

    def test_cached_training_uploads_using_its_own_recorded_output_path(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            (root / "mlops_models").mkdir()
            (root / "source.json").write_text(json.dumps({"commit": "model-commit", "dirty": False}))
            job = {"OutputDataConfig": {
                "S3OutputPath": "s3://bucket/models/example/runs/producer/model", "KmsKeyId": "key"}}
            sm, s3 = Mock(), Mock()
            sm.describe_training_job.return_value = job
            uploaded = {}
            def upload(path, bucket, key, **kwargs):
                if key.endswith(".json"):
                    uploaded[key] = Path(path).read_bytes()
                else:
                    with tarfile.open(path) as archive:
                        uploaded[key] = {name: archive.extractfile(name).read() for name in archive.getnames()
                                         if archive.getmember(name).isfile()}
                self.assertEqual(kwargs["ExtraArgs"]["SSEKMSKeyId"], "key")
            s3.upload_file.side_effect = upload
            def dispatch():
                self.assertEqual(os.environ["PIPELINE_EXECUTION_ID"], "producer")
                self.assertEqual(os.environ["SOURCE_COMMIT"], "model-commit")
                model = Path(os.environ["SM_MODEL_DIR"])
                self.assertTrue(model.is_relative_to(root))
                (model / "weights").write_text("saved model")
                (Path(os.environ["SM_OUTPUT_DATA_DIR"]) / "metrics.json").write_text("{}")
            env = {"MLOPS_STEP_REVISION": "train-v1", "MLOPS_TRAINING_OUTPUT_UPLOAD": "application",
                   "TRAINING_JOB_NAME": "own-job", "TMPDIR": str(root),
                   "TRAINING_JOB_ARN": "arn:aws:sagemaker:us-east-1:123456789012:training-job/own-job"}
            def scratch(**kwargs):
                self.assertEqual(kwargs["dir"], str(root))
                return TemporaryDirectory(prefix=kwargs["prefix"], dir=root)
            with patch.dict(os.environ, env), patch.object(load_source, "__file__", str(root / "load_source.py")), \
                    patch.object(sys, "argv", ["load_source.py", "train"]), \
                    patch.object(load_source.tempfile, "TemporaryDirectory", side_effect=scratch), \
                    patch("boto3.client", side_effect=lambda name, **kwargs: sm if name == "sagemaker" else s3), \
                    patch("mlops_models.entrypoints.__main__.main", side_effect=dispatch), redirect_stdout(io.StringIO()):
                load_source.main()
            sm.describe_training_job.assert_called_once_with(TrainingJobName="own-job")
            self.assertEqual(uploaded["models/example/runs/producer/model/application/model.tar.gz"]["weights"],
                             b"saved model")
            self.assertIn("models/example/runs/producer/reports/training/output.tar.gz", uploaded)
            self.assertEqual(uploaded["models/example/runs/producer/model/application/_execution.json"],
                             uploaded["models/example/runs/producer/model/application/model.tar.gz"]["_execution.json"])

    def test_report_edits_refresh_the_source_saved_by_training(self):
        with TemporaryDirectory() as work:
            package = Path(work) / "mlops_models"
            shutil.copytree(source.PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            csv = Path(work) / "data.csv"
            csv.write_text("x,y\n1,2\n")
            with patch.object(source, "PACKAGE", package):
                before = source.pipeline_fingerprints(Mock(), "s3://bucket/data.csv", csv)
                report = package / "pipelines/inspect.py"
                report.write_text(report.read_text() + "\n# changed report output\n")
                after = source.pipeline_fingerprints(Mock(), "s3://bucket/data.csv", csv)
            self.assertEqual({step for step in before if before[step] != after[step]},
                             {"Train", "Postprocess", "DriftComparison"})

    def test_model_code_source_selects_current_code_or_archived_code(self):
        for code in ("pipeline", "artifact"):
            with self.subTest(code=code), TemporaryDirectory() as work:
                root = Path(work)
                current = root / "source"
                (current / "mlops_models").mkdir(parents=True)
                (current / "source.json").write_text(json.dumps({"commit": "new-code"}))
                artifact = root / "artifact"
                (artifact / "code/mlops_models").mkdir(parents=True)
                (artifact / "code/source.json").write_text(json.dumps({"commit": "old-code"}))
                archive_path = root / "model.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    archive.add(artifact / "code", arcname="code")
                job = {"AppSpecification": {"ImageUri": "image"}, "ProcessingOutputConfig": {"Outputs": [{"OutputName": "evaluation", "S3Output": {
                    "S3Uri": "s3://bucket/models/example/runs/new-run/reports/evaluation",
                    "LocalPath": "/opt/ml/processing/evaluation",
                }}]}}
                original_read = Path.read_text
                def read(path, *args, **kwargs):
                    if str(path) == "/opt/ml/config/processingjobconfig.json":
                        return json.dumps(job)
                    return original_read(path, *args, **kwargs)
                def dispatch():
                    self.assertEqual(os.environ["SOURCE_COMMIT"], "new-code" if code == "pipeline" else "old-code")
                    self.assertEqual(os.environ["PIPELINE_EXECUTION_ID"], "new-run")
                    output = Path(sys.argv[sys.argv.index("--output") + 1])
                    (output / "evaluation.json").write_text("{}")
                env = {"MLOPS_STEP_REVISION": "eval-v1", "MLOPS_MODEL_CODE_SOURCE": code,
                       "MLOPS_PROCESSING_OUTPUT_UPLOAD": "application"}
                with patch.dict(os.environ, env), patch.object(load_source, "__file__", str(current / "load_source.py")), \
                        patch.object(sys, "argv", ["load_source.py", "evaluate", "--model-dir", str(archive_path)]), \
                        patch.object(Path, "read_text", read), patch("boto3.client", return_value=Mock()), \
                        patch("mlops_models.entrypoints.__main__.main", side_effect=dispatch), redirect_stdout(io.StringIO()):
                    load_source.main()


class CachedSourceBindings(unittest.TestCase):
    def setUp(self):
        from test_execution_tracking import S3
        self.workspace = TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.package = self.root / "mlops_models"
        shutil.copytree(source.PACKAGE, self.package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        patched = patch.object(source, "PACKAGE", self.package)
        patched.start()
        self.addCleanup(patched.stop)
        self.csv = self.root / "data.csv"
        self.csv.write_text("x,y\n1,2\n")
        self.s3 = S3()
        self.session = Mock()
        self.session.client.return_value = self.s3
        self.s3_root = "s3://bucket/cache"

    def publish(self, *, commit="a" * 40, model_code_source="pipeline"):
        from copy import deepcopy
        identity = {"mode": "direct-deploy", "name": "review", "commit": commit, "dirty": True}
        uri = source.source_location(self.s3_root, identity)
        revisions = source.pipeline_fingerprints(self.session, "s3://bucket/data.csv", self.csv)
        definitions = source.build_definitions(
            "cache", "arn:aws:iam::123456789012:role/jobs", "image@sha256:abc",
            self.s3_root, uri, identity, region="us-east-1", data_source="csv_direct",
            csv_input_uri="s3://bucket/data.csv", batch_runtime="processing",
            step_cache_revisions=revisions, model_code_source=model_code_source,
        )
        original = deepcopy(definitions)
        bindings = source.configure_cache_sources(definitions)
        source.upload_source(self.session, self.s3_root, identity, expected_uri=uri)
        published = source.publish_cache_sources(self.session, self.s3_root, uri, bindings)
        return {"uri": uri, "revisions": revisions, "definitions": definitions,
                "original": original, "bindings": bindings, "published": published}

    def resolve(self, value, body, outputs=None):
        parameters = {p["Name"]: p.get("DefaultValue") for p in body["Parameters"]}
        if isinstance(value, list):
            return [self.resolve(item, body, outputs) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"Get"}:
            name = value["Get"]
            if name.startswith("Parameters."):
                return parameters[name.removeprefix("Parameters.")]
            return (outputs or {}).get(name, "retained:" + name)
        if set(value) == {"Std:Join"}:
            join = value["Std:Join"]
            return join["On"].join(str(self.resolve(v, body, outputs)) for v in join["Values"])
        if set(value) == {"Std:JsonGet"}:
            item = value["Std:JsonGet"]
            uri = self.resolve(item["S3Uri"], body, outputs)
            if "/cache-sources-v1/" in uri:
                bucket, key = source.s3_location(uri)
                return json.loads(self.s3.objects[bucket, key])[item["Path"]]
            return "retained:" + item["Path"]
        return {key: self.resolve(item, body, outputs) for key, item in value.items()}

    def cache_keys(self, body, outputs=None):
        fields = {
            "Processing": ("AppSpecification", "Environment", "ProcessingInputs"),
            "Training": ("AlgorithmSpecification", "CheckpointConfig", "DebugHookConfig",
                         "DebugRuleConfigurations", "Environment", "HyperParameters", "InputDataConfig"),
        }
        return {
            step["Name"]: self.resolve(
                {name: step["Arguments"].get(name) for name in fields[step["Type"]]}, body, outputs)
            for step in walk_steps(body["Steps"])
            if step.get("CacheConfig", {}).get("Enabled") and step["Type"] in fields
        }

    def edit(self, relative):
        path = self.package / relative
        path.write_text(path.read_text() + "\n# cache regression edit\n")

    def test_registry_edit_keeps_resolved_compute_keys_and_uses_new_registration_source(self):
        first = self.publish()
        self.edit("registry/model_package.py")
        second = self.publish()
        self.assertNotEqual(first["uri"], second["uri"])
        self.assertEqual(first["revisions"], second["revisions"])
        # The old tests compared unresolved parameter expressions. Resolve the
        # actual URI values to demonstrate both the original bug and the fix.
        old = [self.cache_keys(p["original"]["training"]) for p in (first, second)]
        self.assertEqual({name for name in old[0] if old[0][name] != old[1][name]},
                         {"GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate"})
        self.assertEqual(self.cache_keys(first["definitions"]["training"]),
                         self.cache_keys(second["definitions"]["training"]))
        registration = next(step for step in walk_steps(second["definitions"]["training"]["Steps"])
                            if step["Name"] == "RegisterModel")
        source_input = next(item for item in registration["Arguments"]["ProcessingInputs"]
                            if item["InputName"] == "source")
        self.assertEqual(self.resolve(source_input["S3Input"]["S3Uri"],
                                      second["definitions"]["training"]), second["uri"])
        self.assertTrue(all(item["source_uri"] == first["uri"] for item in second["published"]))
        # Original source provenance is preserved with the retained complete snapshot.
        bucket, key = source.s3_location(first["uri"] + "/source.json")
        self.assertEqual(json.loads(self.s3.objects[bucket, key])["commit"], "a" * 40)

    def test_training_edit_preserves_preparation_and_new_model_invalidates_evaluation(self):
        first = self.publish()
        self.edit("model_base/training/train.py")
        second = self.publish()
        old = self.cache_keys(first["definitions"]["training"])
        new = self.cache_keys(second["definitions"]["training"])
        self.assertEqual({name for name in old if old[name] != new[name]}, {"Train"})
        updated = self.cache_keys(second["definitions"]["training"], {
            "Steps.Train.OutputDataConfig.S3OutputPath": "s3://bucket/new-producer/model"})
        self.assertNotEqual(new["Evaluate"], updated["Evaluate"])

    def test_evaluation_edit_preserves_all_upstream_compute(self):
        first = self.publish()
        self.edit("model_base/training/evaluate.py")
        second = self.publish()
        old = self.cache_keys(first["definitions"]["training"])
        new = self.cache_keys(second["definitions"]["training"])
        self.assertEqual({name for name in old if old[name] != new[name]}, {"Evaluate"})

    def test_framework_edit_invalidates_training_and_preserves_inference_code(self):
        first = self.publish()
        self.edit("frameworks/autogluon.py")
        second = self.publish()
        old = self.cache_keys(first["definitions"]["training"])
        new = self.cache_keys(second["definitions"]["training"])
        self.assertEqual({name for name in old if old[name] != new[name]}, {"Train"})
        self.assertEqual(self.cache_keys(first["definitions"]["inference"]),
                         self.cache_keys(second["definitions"]["inference"]))
        uri = second["uri"] + "/mlops_models/frameworks/autogluon.py"
        self.assertEqual(self.s3.objects[source.s3_location(uri)],
                         (self.package / "frameworks/autogluon.py").read_bytes())

    def test_logger_edit_refreshes_every_executing_step(self):
        first = self.publish()
        self.edit("pipelines/tracking.py")
        second = self.publish()
        old = self.cache_keys(first["definitions"]["training"])
        new = self.cache_keys(second["definitions"]["training"])
        self.assertTrue(all(old[name] != new[name] for name in old))

    def test_commit_only_change_preserves_source_and_repeated_publication_is_immutable(self):
        first = self.publish()
        second = self.publish(commit="b" * 40)
        self.assertNotEqual(first["uri"], second["uri"])
        self.assertEqual(self.cache_keys(first["definitions"]["training"]),
                         self.cache_keys(second["definitions"]["training"]))
        saved = self.s3.objects.copy()
        self.publish(commit="b" * 40)
        self.assertEqual(self.s3.objects, saved)

    def test_concurrent_first_writer_keeps_the_completed_source_binding(self):
        from botocore.exceptions import ClientError
        first = self.publish()
        target = next(key for bucket, key in self.s3.objects
                      if "/cache-bindings-v1/training/GetData/" in key)
        original_get = self.s3.get_object
        missed = []

        def get(**kwargs):
            if kwargs["Key"] == target and not missed:
                missed.append(True)
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return original_get(**kwargs)

        self.edit("registry/model_package.py")
        with patch.object(self.s3, "get_object", side_effect=get):
            second = self.publish()
        self.assertTrue(missed)
        self.assertTrue(all(item["source_uri"] == first["uri"] for item in second["published"]))

    def test_corrupt_binding_and_missing_retained_source_stop_publication(self):
        first = self.publish()
        target = next((bucket, key) for bucket, key in self.s3.objects
                      if "/cache-bindings-v1/training/GetData/" in key)
        original = self.s3.objects[target]
        record = json.loads(original)
        record["source_uri"] = "s3://another-bucket/source/sha256-" + record["source_sha256"]
        self.s3.objects[target] = json.dumps(record).encode()
        self.edit("registry/model_package.py")
        with self.assertRaisesRegex(ValueError, "outside this application's"):
            self.publish()
        self.s3.objects[target] = original
        bucket, key = source.s3_location(first["uri"] + "/source.json")
        del self.s3.objects[bucket, key]
        with self.assertRaisesRegex(ValueError, "missing or has an inconsistent"):
            self.publish()

    def test_development_loader_gets_cache_binding_even_with_artifact_default(self):
        from copy import deepcopy
        first = self.publish(model_code_source="artifact")
        self.assertNotEqual(first["definitions"]["inference"], first["original"]["inference"])
        self.assertEqual(next(p for p in first["definitions"]["inference"]["Parameters"]
                              if p["Name"] == "ModelCodeSource")["DefaultValue"], "artifact")
        disabled = deepcopy(first["original"])
        for body in disabled.values():
            for step in walk_steps(body["Steps"]):
                step.pop("CacheConfig", None)
        saved = deepcopy(disabled)
        self.assertEqual(source.configure_cache_sources(disabled), [])
        self.assertEqual(disabled, saved)
        self.session.reset_mock()
        self.assertEqual(source.publish_cache_sources(self.session, self.s3_root, first["uri"], []), [])
        self.session.client.assert_not_called()

    def test_execution_code_choice_changes_inference_keys_without_changing_the_model(self):
        from copy import deepcopy
        first = self.publish(model_code_source="artifact")
        body = deepcopy(first["definitions"]["inference"])
        before = self.cache_keys(body)
        next(p for p in body["Parameters"] if p["Name"] == "ModelCodeSource")["DefaultValue"] = "pipeline"
        after = self.cache_keys(body)
        self.assertEqual(set(before), {"GetData", "Preprocess", "Score"})
        for name in before:
            self.assertNotEqual(before[name]["Environment"], after[name]["Environment"])
            self.assertEqual(before[name]["ProcessingInputs"], after[name]["ProcessingInputs"])

    def test_predict_edits_refresh_score_source_without_retraining(self):
        for relative in ("entrypoints/predict.py", "model_base/inference/predict.py",
                         "models/usecase1/inference/predict.py"):
            with self.subTest(relative=relative):
                first = self.publish(model_code_source="pipeline")
                self.edit(relative)
                second = self.publish(model_code_source="pipeline")
                before = self.cache_keys(first["definitions"]["inference"])
                after = self.cache_keys(second["definitions"]["inference"])
                self.assertEqual({name for name in before if before[name] != after[name]}, {"Score"})
                selected = next(item for item in second["published"]
                                if item["kind"] == "inference" and item["step"] == "Score")
                self.assertEqual(selected["source_uri"], second["uri"])


if __name__ == "__main__":
    unittest.main()
