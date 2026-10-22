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
        for name in ("QualityGate", "RegisterModel"):
            self.assertNotIn("CacheConfig", steps[name])

    def test_batch_compute_is_cached_but_approval_lookup_is_live(self):
        for code in ("artifact", "pipeline"):
            body = self.definitions(model_code_source=code)["inference"]
            steps = {step["Name"]: step for step in body["Steps"]}
            for name in ("GetData", "Preprocess", "Score", "Postprocess"):
                self.assertTrue(steps[name]["CacheConfig"]["Enabled"])
                environment = steps[name]["Arguments"]["Environment"]
                # Registry JsonGet expressions contain the current decision
                # file URI, but resolve to the selected image/checkpoint value.
                self.assertNotIn("PIPELINE_EXECUTION_ID", environment)
                self.assertEqual(environment["MLOPS_MODEL_CODE_SOURCE"], "pipeline" if name == "GetData" else code)
            self.assertNotIn("CacheConfig", steps["ResolveModel"])

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
                job = {"ProcessingOutputConfig": {"Outputs": [{"OutputName": "evaluation", "S3Output": {
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


if __name__ == "__main__":
    unittest.main()
