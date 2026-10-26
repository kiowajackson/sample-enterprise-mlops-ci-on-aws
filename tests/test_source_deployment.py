"""Exercise the shared source pipeline contract and approved-model selection."""

import json
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mlops-models"), str(ROOT / "mlops-models/src")]
from mlops_models.registry.model_package import resolve_latest_approved_model
from mlops_models.pipelines.source import build_definitions, source_identity, upload_source
from mlops_skeleton.pipelines import walk_steps
from model_fixtures import MODELS
from test_execution_tracking import S3
from mlops_models.pipelines import source


class SourceDeployment(unittest.TestCase):
    def setUp(self):
        self.identity = {"mode": "direct-deploy", "name": "review", "commit": "a" * 40, "dirty": True}
        self.image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "b" * 64

    def definitions(self, runtime="transform", **settings):
        return build_definitions(
            "review", "arn:aws:iam::123456789012:role/jobs", self.image,
            "s3://artifacts/team", "s3://artifacts/team/source/direct-deploy/review",
            self.identity, region="us-east-1", batch_runtime=runtime,
            infrastructure={"network": {"isolation": False}},
            feature_store_config_uri="s3://data/features/config.json", **settings)

    def test_both_pipelines_publish_before_any_training_result_exists(self):
        definitions = self.definitions()
        for body in definitions.values():
            model = next(p for p in body["Parameters"] if p["Name"] == "ModelId")
            self.assertEqual(model["EnumValues"], [folder.name for folder in MODELS])
            self.assertNotIn("ModelDataUri", [p["Name"] for p in body["Parameters"]])
        steps = {s["Name"]: s for s in walk_steps(definitions["training"]["Steps"])}
        self.assertEqual(steps["RegisterModel"]["Type"], "Processing")
        self.assertEqual(steps["RegisterModel"]["Arguments"]["RoleArn"],
                         "arn:aws:iam::123456789012:role/jobs")
        args = steps["Train"]["Arguments"]
        self.assertEqual(args["InputDataConfig"][-1]["ChannelName"], "source")
        self.assertEqual(args["AlgorithmSpecification"]["ContainerEntrypoint"][-1], "train")
        inference = definitions["inference"]["Steps"]
        self.assertEqual(inference[0]["Name"], "ResolveModel")
        self.assertFalse(inference[0]["Arguments"]["NetworkConfig"]["EnableNetworkIsolation"])
        self.assertEqual(inference[0]["Arguments"]["Environment"]["AWS_DEFAULT_REGION"], "us-east-1")
        self.assertFalse(inference[1]["Arguments"]["NetworkConfig"]["EnableNetworkIsolation"])
        self.assertIn("Std:JsonGet", next(s for s in inference if s["Name"] == "CreateModel")["Arguments"]["PrimaryContainer"]["Image"])

    def test_runtime_hpo_uses_prepared_file_and_source(self):
        body = self.definitions(include_hpo=True)["training"]
        self.assertFalse(next(p for p in body["Parameters"] if p["Name"] == "UseHPO")["DefaultValue"])
        steps = {s["Name"]: s for s in walk_steps(body["Steps"])}
        mode = steps["TrainingMode"]["Arguments"]
        self.assertEqual([s["Name"] for s in mode["IfSteps"]], ["Tune", "EvaluateTuned"])
        self.assertEqual([s["Name"] for s in mode["ElseSteps"]], ["Train", "Evaluate"])
        args = steps["Tune"]["Arguments"]
        job = args["TrainingJobDefinition"]
        self.assertEqual({c["ChannelName"] for c in job["InputDataConfig"]}, {"prepared", "selection", "source"})
        self.assertNotIn("ContainerEntrypoint", job["AlgorithmSpecification"])
        self.assertEqual(job["Environment"]["PYTHONPATH"], "/opt/ml/input/data/source")
        self.assertFalse(job["EnableNetworkIsolation"])
        objective = args["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"]
        self.assertEqual(objective["MetricName"]["Std:JsonGet"]["Path"], "tuning.objective.name")
        self.assertEqual(objective["Type"]["Std:JsonGet"]["Path"], "tuning.objective.direction")
        self.assertEqual(steps["QualityGate"]["DependsOn"], ["Evaluate"])
        self.assertEqual(steps["QualityGateTuned"]["DependsOn"], ["EvaluateTuned"])
        self.assertIn("TrainingJobSummaries[0]",
                      json.dumps(steps["EvaluateTuned"]["Arguments"]["ProcessingInputs"]))
        self.assertEqual(steps["RegisterTunedModel"]["Type"], "Processing")
        self.assertTrue(steps["RegisterTunedModel"]["Arguments"]["AppSpecification"]["ContainerEntrypoint"][-1]
                        .endswith("/registry/model_package.py"))

    def test_source_parameter_supplies_training_jobs_and_hpo_configuration(self):
        body = self.definitions(include_hpo=True)["training"]
        reference = {"Get": "Parameters.PipelineSourceUri"}
        parameter = next(p for p in body["Parameters"] if p["Name"] == "PipelineSourceUri")
        self.assertEqual(parameter["DefaultValue"], "s3://artifacts/team/source/direct-deploy/review")
        for step in walk_steps(body["Steps"]):
            args = step["Arguments"]
            if step["Type"] == "Processing":
                mounted = next(i for i in args["ProcessingInputs"] if i["InputName"] == "source")
                self.assertEqual(mounted["S3Input"]["S3Uri"], reference, step["Name"])
            elif step["Type"] in {"Training", "Tuning"}:
                job = args["TrainingJobDefinition"] if step["Type"] == "Tuning" else args
                channel = next(i for i in job["InputDataConfig"] if i["ChannelName"] == "source")
                self.assertEqual(channel["DataSource"]["S3DataSource"]["S3Uri"], reference)
            if step["Name"] in {"RegisterModel", "RegisterTunedModel"}:
                self.assertEqual(args["Environment"]["SOURCE_URI"], reference)
            if step["Type"] == "Tuning":
                objective = args["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"]
                self.assertEqual(objective["MetricName"]["Std:JsonGet"]["S3Uri"]["Std:Join"]["Values"][0],
                                 reference)

    def test_source_parameter_does_not_replace_the_selected_models_archived_code(self):
        reference = {"Get": "Parameters.PipelineSourceUri"}
        for runtime, mode in (("processing", "pipeline"), ("processing", "artifact"),
                              ("both", "pipeline"), ("both", "artifact"), ("transform", "artifact")):
            with self.subTest(runtime=runtime, mode=mode):
                body = self.definitions(runtime, model_code_source=mode,
                                        monitoring={"enabled": runtime != "transform"})["inference"]
                parameters = {p["Name"]: p for p in body["Parameters"]}
                self.assertEqual(parameters["PipelineSourceUri"]["DefaultValue"],
                                 "s3://artifacts/team/source/direct-deploy/review")
                for step in walk_steps(body["Steps"]):
                    if step["Type"] != "Processing":
                        continue
                    args = step["Arguments"]
                    actual = next(i["S3Input"]["S3Uri"] for i in args["ProcessingInputs"]
                                  if i["InputName"] == "source")
                    uses_deployment = (step["Name"] == "ResolveModel" or
                                       (mode == "pipeline" and step["Name"] in {
                                           "GetData", "Preprocess", "Score", "Postprocess", "DriftComparison"}))
                    if uses_deployment:
                        self.assertEqual(actual, reference, step["Name"])
                    else:
                        self.assertEqual(actual["Std:JsonGet"]["Path"], "source_uri", step["Name"])
                    if step["Name"] != "ResolveModel":
                        self.assertEqual(args["Environment"]["MLOPS_MODEL_CODE_SOURCE"],
                                         "pipeline" if uses_deployment else "artifact")

    def test_processing_scoring_remains_a_distinct_runtime(self):
        body = self.definitions("processing")["inference"]
        self.assertEqual([s["Name"] for s in body["Steps"]],
                         ["ResolveModel", "GetData", "Preprocess", "Score", "Postprocess"])
        score = next(s for s in body["Steps"] if s["Name"] == "Score")
        self.assertEqual(score["Arguments"]["AppSpecification"]["ContainerEntrypoint"][-1], "score")
        self.assertNotIn("ContainerArguments", score["Arguments"]["AppSpecification"])
        self.assertIn("['prepared']", score["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"]["Get"])

    def test_development_defaults_to_uploaded_code_with_saved_model_artifacts(self):
        # Registry lifecycle and deployment route determine this choice, not the account label.
        for environment in ("dev", "prod"):
            body = self.definitions("processing", environment=environment,
                                    monitoring={"enabled": True})["inference"]
            self.assertEqual(body["Metadata"]["ModelCodeSource"], "pipeline")
            self.assertEqual(body["Metadata"]["RegistryStage"], "Development")
            for step in walk_steps(body["Steps"]):
                if step["Type"] != "Processing" or step["Name"] == "ResolveModel":
                    continue
                args = step["Arguments"]
                inputs = {i["InputName"]: i["S3Input"]["S3Uri"] for i in args["ProcessingInputs"]}
                self.assertEqual(inputs["source"], {"Get": "Parameters.PipelineSourceUri"})
                self.assertEqual(args["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "pipeline")
                self.assertEqual(inputs["checkpoint"]["Std:JsonGet"]["Path"], "checkpoint")
                self.assertEqual(args["AppSpecification"]["ImageUri"]["Std:JsonGet"]["Path"], "image")

    def test_ci_requires_saved_model_code_and_pins_the_helper_source(self):
        self.identity = {**self.identity, "mode": "ci-deploy", "dirty": False}
        for runtime in ("processing", "transform", "both"):
            with self.subTest(runtime=runtime):
                bodies = self.definitions(runtime)
                for body in bodies.values():
                    parameter = next(p for p in body["Parameters"] if p["Name"] == "PipelineSourceUri")
                    self.assertEqual(parameter["EnumValues"], [parameter["DefaultValue"]])
                inference = bodies["inference"]
                self.assertEqual(inference["Metadata"]["ModelCodeSource"], "artifact")
                for step in walk_steps(inference["Steps"]):
                    if step["Type"] != "Processing" or step["Name"] == "ResolveModel":
                        continue
                    args = step["Arguments"]
                    source_input = next(i for i in args["ProcessingInputs"] if i["InputName"] == "source")
                    self.assertEqual(source_input["S3Input"]["S3Uri"]["Std:JsonGet"]["Path"], "source_uri")
                    self.assertEqual(args["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "artifact")
                with self.assertRaisesRegex(ValueError, "CI must use"):
                    self.definitions(runtime, model_code_source="pipeline")

    def test_analysis_choice_reaches_preprocessing_and_postprocessing_for_both_runtimes(self):
        for runtime in ("processing", "transform", "both"):
            with self.subTest(runtime=runtime):
                definitions = self.definitions(runtime, monitoring={"enabled": runtime != "transform"})
                self.assertNotIn("PredictionAnalysis", [p["Name"] for p in definitions["training"]["Parameters"]])
                body = definitions["inference"]
                parameter = next(p for p in body["Parameters"] if p["Name"] == "PredictionAnalysis")
                self.assertEqual(parameter["DefaultValue"], "prediction_inference")
                self.assertEqual(parameter["EnumValues"], ["prediction_inference", "prediction_test"])
                consumers = []
                for step in walk_steps(body["Steps"]):
                    value = step.get("Arguments", {}).get("Environment", {}).get("MLOPS_ANALYSIS_MODE")
                    if value:
                        consumers.append(step["Name"])
                        self.assertEqual(value, {"Get": "Parameters.PredictionAnalysis"})
                expected = (["Preprocess", "Postprocess", "PreprocessTransform", "PostprocessTransform"]
                            if runtime == "both" else ["Preprocess", "Postprocess"])
                self.assertCountEqual(consumers, expected)

    def test_one_inference_pipeline_can_select_either_runtime_without_cross_branch_inputs(self):
        from checks import find_references
        definitions = self.definitions("both", model_code_source="pipeline")
        self.assertEqual(set(definitions), {"training", "inference"})
        self.assertEqual(definitions["training"],
                         self.definitions("processing", model_code_source="pipeline")["training"])
        body = definitions["inference"]
        self.assertEqual([s["Name"] for s in body["Steps"]], ["ResolveModel", "InferenceMode"])
        parameters = {p["Name"]: p for p in body["Parameters"]}
        self.assertEqual(parameters["InferenceRuntime"]["EnumValues"], ["processing", "transform"])
        self.assertEqual(parameters["InferenceRuntime"]["DefaultValue"], "processing")
        self.assertEqual(parameters["TransformOutputFile"]["DefaultValue"], "messaging_final.csv")
        branches = body["Steps"][-1]["Arguments"]
        self.assertEqual([s["Name"] for s in branches["ElseSteps"]], ["GetData", "Preprocess", "Score", "Postprocess"])
        self.assertEqual([s["Name"] for s in branches["IfSteps"]],
                         ["GetDataTransform", "PreprocessTransform", "CreateModel", "Transform", "PostprocessTransform"])
        for branch in ("IfSteps", "ElseSteps"):
            available = {"ResolveModel"} | {s["Name"] for s in branches[branch]}
            for reference in find_references(branches[branch]):
                kind, name, *_ = reference.split(".")
                self.assertIn(name, parameters if kind == "Parameters" else available if kind == "Steps" else {name})
            for step in branches[branch]:
                self.assertLessEqual(set(step.get("DependsOn", [])), available)
        steps = {s["Name"]: s for s in walk_steps(body["Steps"])}
        self.assertEqual(steps["Score"]["Arguments"]["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "pipeline")
        self.assertEqual(steps["PreprocessTransform"]["Arguments"]["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "artifact")
        self.assertEqual(steps["PostprocessTransform"]["Arguments"]["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "artifact")
        native = steps["CreateModel"]["Arguments"]["PrimaryContainer"]
        self.assertEqual(native["Environment"]["MLOPS_TRANSFORM_OUTPUT_FILE"],
                         {"Get": "Parameters.TransformOutputFile"})
        self.assertEqual(native["Environment"]["PYTHONPATH"], "/opt/ml/model/code")
        self.assertEqual(steps["Transform"]["Arguments"]["ModelName"], {"Get": "Steps.CreateModel.ModelName"})
        self.assertIn("Steps.PreprocessTransform.", json.dumps(steps["Transform"]["Arguments"]["TransformInput"]))
        self.assertNotIn("Steps.Preprocess.", json.dumps(branches["IfSteps"]))
        self.assertEqual(body["Metadata"]["TransformModelCodeSource"], "artifact")

    def test_runtime_choice_retains_compute_cache_and_fresh_registry_resolution(self):
        revisions = {name: name + "-revision" for name in
                     ("BatchGetData", "BatchPreprocess", "Score", "Postprocess")}
        body = build_definitions(
            "review", "arn:aws:iam::123456789012:role/jobs", self.image,
            "s3://artifacts/team", "s3://artifacts/team/source", self.identity,
            region="us-east-1", batch_runtime="both", model_code_source="pipeline",
            data_source="csv_direct", csv_input_uri="s3://data/scoring.csv",
            step_cache_revisions=revisions)["inference"]
        steps = {s["Name"]: s for s in walk_steps(body["Steps"])}
        for name in ("GetData", "Preprocess", "Score",
                     "PreprocessTransform", "PostprocessTransform"):
            self.assertTrue(steps[name]["CacheConfig"]["Enabled"])
        for name in ("ResolveModel", "InferenceMode", "CreateModel", "Transform", "Postprocess"):
            self.assertNotIn("CacheConfig", steps[name])
        self.assertIn("Steps.PreprocessTransform.", json.dumps(steps["PostprocessTransform"]["Arguments"]))

    def registry(self, summaries, package):
        client = Mock()
        client.exceptions = SimpleNamespace(ClientError=ClientError)
        client.get_paginator.return_value.paginate.return_value = [
            {"ModelPackageSummaryList": summaries[:1]}, {"ModelPackageSummaryList": summaries[1:]}]
        package.setdefault("ModelLifeCycle", {"Stage": "Development", "StageStatus": "Approved"})
        package.setdefault("CustomerMetadataProperties", {
            "DeploymentNamespace": "review", "ModelId": "usecase1", "Environment": "dev",
            "SourceUri": "s3://bucket/original-source", "SourceDigest": "a" * 64,
        })
        client.describe_model_package.return_value = package
        return client

    def test_latest_numeric_approved_completed_version_is_selected_once(self):
        summaries = [{"ModelPackageVersion": n, "ModelPackageArn": f"package/{n}",
                      "ModelPackageStatus": status} for n, status in [(2, "Completed"), (10, "Completed"), (11, "InProgress")]]
        package = {"ModelPackageArn": "package/10", "ModelPackageVersion": 10,
                   "ModelPackageGroupName": "review-usecase1", "ModelApprovalStatus": "Approved",
                   "ModelPackageStatus": "Completed", "InferenceSpecification": {"Containers": [
                       {"Image": self.image, "ModelDataUrl": "s3://models/10/model.tar.gz"}]}}
        client = self.registry(summaries, package)
        with TemporaryDirectory() as directory:
            result = resolve_latest_approved_model("review-usecase1", directory, client=client, namespace="review", model_id="usecase1", environment="dev")
            self.assertEqual(result["version"], 10)
            self.assertEqual(json.loads((Path(directory) / "package.json").read_text()), result)
        client.get_paginator.return_value.paginate.assert_called_once_with(
            ModelPackageGroupName="review-usecase1", ModelApprovalStatus="Approved")
        client.describe_model_package.assert_called_once_with(ModelPackageName="package/10")

    def test_no_approved_package_fails_before_model_execution(self):
        client = self.registry([], {})
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "No eligible"):
            resolve_latest_approved_model("review-usecase1", directory, client=client, namespace="review", model_id="usecase1", environment="dev")
        client.describe_model_package.assert_not_called()

    def test_permission_denial_is_not_reported_as_missing_approval(self):
        client = self.registry([], {})
        client.get_paginator.return_value.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "permission denied"}}, "ListModelPackages")
        with TemporaryDirectory() as directory, self.assertRaises(ClientError):
            resolve_latest_approved_model("review-usecase1", directory, client=client, namespace="review", model_id="usecase1", environment="dev")

    def test_missing_group_and_describe_denial_keep_the_original_registry_error(self):
        for operation in ("ListModelPackages", "DescribeModelPackage"):
            client = self.registry([{"ModelPackageVersion": 1, "ModelPackageArn": "package/1",
                                     "ModelPackageStatus": "Completed"}], {})
            error = ClientError(
                {"Error": {"Code": "ValidationException" if operation == "ListModelPackages" else "AccessDenied",
                           "Message": "Group does not exist" if operation == "ListModelPackages" else "Denied"}},
                operation,
            )
            if operation == "ListModelPackages":
                client.get_paginator.return_value.paginate.side_effect = error
            else:
                client.describe_model_package.side_effect = error
            with self.subTest(operation=operation), TemporaryDirectory() as directory:
                with self.assertRaises(ClientError) as caught:
                    resolve_latest_approved_model("review-usecase1", directory, client=client, namespace="review", model_id="usecase1", environment="dev")
                self.assertIs(caught.exception, error)
                self.assertFalse((Path(directory) / "package.json").exists())

    def test_changed_approval_or_group_fails_before_publishing_a_model_reference(self):
        for update in ({"ModelApprovalStatus": "Rejected"},
                       {"ModelPackageGroupName": "another-group"},
                       {"ModelPackageStatus": "Failed"}):
            package = {"ModelPackageGroupName": "review-usecase1", "ModelApprovalStatus": "Approved",
                       "ModelPackageStatus": "Completed", "InferenceSpecification": {"Containers": [
                           {"Image": self.image, "ModelDataUrl": "s3://models/model.tar.gz"}]}, **update}
            client = self.registry([{"ModelPackageVersion": 1, "ModelPackageArn": "package/1",
                                     "ModelPackageStatus": "Completed"}], package)
            with self.subTest(update=update), TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "No eligible"):
                    resolve_latest_approved_model("review-usecase1", directory, client=client, namespace="review", model_id="usecase1", environment="dev")
                self.assertFalse((Path(directory) / "package.json").exists())

    def test_ci_commit_is_supplied_and_matches_a_clean_checkout(self):
        with self.assertRaisesRegex(ValueError, "full reviewed commit"):
            source_identity("ci-deploy", "release", "main")
        with patch("subprocess.check_output", side_effect=["b" * 40, ""]):
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                source_identity("ci-deploy", "release", "a" * 40)
        with patch("subprocess.check_output", side_effect=["a" * 40, ""]):
            self.assertFalse(source_identity("ci-deploy", "release", "a" * 40)["dirty"])

    def test_source_upload_contains_importable_package_and_tracked_loader(self):
        s3 = S3()
        session = SimpleNamespace(client=lambda name: s3)
        uri = upload_source(session, "s3://bucket/app", self.identity)
        uploads = {key: body for (bucket, key), body in s3.objects.items()}
        self.assertTrue(uri.startswith("s3://bucket/app/source/direct-deploy/review/sha256-"))
        prefix = uri.removeprefix("s3://bucket/")
        self.assertIn(prefix + f"/mlops_models/models/{MODELS[0].name}/parameters.json", uploads)
        self.assertEqual(uploads[prefix + "/load_source.py"],
                         (ROOT / "mlops-models/src/mlops_models/runtime/load_source.py").read_bytes())
        self.assertIn(prefix + "/mlops_models/entrypoints/get_data.py", uploads)
        self.assertFalse(any("/mlops_features/" in key for key in uploads))
        self.assertEqual(json.loads(uploads[prefix + "/source.json"]),
                         {**self.identity, "uri": uri, "sha256": uri.rsplit("sha256-", 1)[-1]})
        self.assertEqual(uri, upload_source(session, "s3://bucket/app", self.identity))
        self.assertEqual(uploads, {key: body for (bucket, key), body in s3.objects.items()})

    def test_changed_preview_and_conflicting_objects_are_not_overwritten(self):
        s3 = S3()
        session = SimpleNamespace(client=lambda name: s3)
        with TemporaryDirectory() as directory:
            package = Path(directory) / "mlops_models"
            loader = package / "runtime/load_source.py"
            loader.parent.mkdir(parents=True)
            loader.write_text("original")
            with patch.object(source, "PACKAGE", package):
                before = source.source_location("s3://bucket/app", self.identity)
                loader.write_text("updated")
                with self.assertRaisesRegex(ValueError, "Source changed after preview"):
                    upload_source(session, "s3://bucket/app", self.identity, expected_uri=before)
                self.assertFalse(s3.objects)
                after = upload_source(session, "s3://bucket/app", self.identity)
                self.assertNotEqual(before, after)
                key = after.removeprefix("s3://bucket/") + "/mlops_models/runtime/load_source.py"
                s3.objects["bucket", key] = b"unexpected"
                with self.assertRaisesRegex(ValueError, "Refusing to replace different bytes"):
                    upload_source(session, "s3://bucket/app", self.identity)
                self.assertEqual(s3.objects["bucket", key], b"unexpected")

    def test_upload_uses_captured_bytes_and_can_resume_after_interruption(self):
        s3 = S3()
        session = SimpleNamespace(client=lambda name: s3)
        with TemporaryDirectory() as directory:
            package = Path(directory) / "mlops_models"
            loader = package / "runtime/load_source.py"
            loader.parent.mkdir(parents=True)
            loader.write_text("captured")
            original_put = s3.put_object
            def interrupt(**request):
                original_put(**request)
                raise RuntimeError("interrupted")
            with patch.object(source, "PACKAGE", package):
                with patch.object(s3, "put_object", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        upload_source(session, "s3://bucket/app", self.identity)
                self.assertFalse(any(key.endswith("/source.json") for bucket, key in s3.objects))
                before = dict(s3.objects)
                uri = upload_source(session, "s3://bucket/app", self.identity)
                self.assertTrue(all(s3.objects[key] == value for key, value in before.items()))
                snapshot = source.source_snapshot(self.identity)
                loader.write_text("changed after snapshot")
                with patch.object(source, "source_snapshot", return_value=snapshot):
                    self.assertEqual(uri, upload_source(session, "s3://bucket/app", self.identity))
                self.assertTrue(all(body == b"captured" for (bucket, key), body in s3.objects.items()
                                    if key.endswith(".py")))

    def test_evaluation_loader_uses_checkpoint_source_over_uploaded_source(self):
        with TemporaryDirectory() as work:
            folder = Path(work)
            def package(where, marker):
                code = where / "mlops_models/entrypoints"
                code.mkdir(parents=True)
                (where / "source.json").write_text(json.dumps({"revision": marker}))
                (where / "mlops_models/__init__.py").write_text("")
                (code / "__init__.py").write_text("")
                (code / "__main__.py").write_text(
                    f"def main(): print({marker!r})\n")
            package(folder, "WRONG-UPLOADED-SOURCE")
            package(folder / "checkpoint/code", "RETAINED-MODEL-SOURCE")
            with tarfile.open(folder / "model.tar.gz", "w:gz") as archive:
                archive.add(folder / "checkpoint/code", arcname="code")
            shutil.copyfile(ROOT / "mlops-models/src/mlops_models/runtime/load_source.py",
                            folder / "load_source.py")
            result = subprocess.run([sys.executable, str(folder / "load_source.py"), "evaluate",
                                     "--model-dir", str(folder / "model.tar.gz")],
                                    check=True, capture_output=True, text=True)
            self.assertIn("RETAINED-MODEL-SOURCE", result.stdout)
            self.assertNotIn("WRONG-UPLOADED-SOURCE", result.stdout)


class ExistingGroupRegistration(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "MODEL_PACKAGE_GROUP": "example-sm-dev", "MODEL_ID": "example",
            "IMAGE_URI": "image@sha256:abc", "PIPELINE_EXECUTION_ARN": "pipeline/execution/one",
            "REGISTRATION_OUTPUT_URI": "s3://bucket/models/example/runs/one/reports/registry",
            "EVALUATION_URI": "s3://bucket/models/example/runs/producer/reports/evaluation/evaluation.json",
            "SOURCE_URI": "s3://bucket/source", "DEPLOY_MODE": "direct-deploy",
            "DEPLOYMENT_NAMESPACE": "review", "ENVIRONMENT": "dev",
        }
        self.evaluation = {"checkpoint_uri": "s3://bucket/models/example/runs/producer/model/model.tar.gz",
                           "model_source_commit": "saved-model-commit", "model_source_dirty": "false",
                           "model_source_uri": "s3://bucket/original-source", "model_source_digest": "a" * 64}
        self.client, self.s3 = Mock(), Mock()
        self.evaluation["model_metadata_sha256"] = hashlib.sha256(b"{}").hexdigest()
        self.s3.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(b"{}")}
        self.client.describe_model_package_group.return_value = {"ModelPackageGroupArn": "existing-group-arn"}
        self.client.create_model_package.return_value = {"ModelPackageArn": "package-arn"}
        self.client.describe_model_package.return_value = {"ModelPackageStatus": "Completed"}

    def register(self):
        from mlops_models.registry.model_package import register_existing_group
        return register_existing_group(self.evaluation, self.settings, client=self.client, s3=self.s3)

    def test_registration_never_creates_group_and_keeps_cached_model_identity(self):
        record = self.register()
        request = self.client.create_model_package.call_args.kwargs
        self.assertEqual(request["ModelPackageGroupName"], "existing-group-arn")
        self.assertEqual(request["ModelApprovalStatus"], "PendingManualApproval")
        self.assertEqual(request["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
                         self.evaluation["checkpoint_uri"])
        self.assertEqual(request["MetadataProperties"]["CommitId"], "saved-model-commit")
        self.client.create_model_package_group.assert_not_called()
        self.assertEqual(record, json.loads(self.s3.put_object.call_args.kwargs["Body"]))
        token = request["ClientToken"]
        self.register()
        self.assertEqual(self.client.create_model_package.call_args.kwargs["ClientToken"], token)
        self.settings["PIPELINE_EXECUTION_ARN"] = "pipeline/execution/two"
        self.register()
        self.assertNotEqual(self.client.create_model_package.call_args.kwargs["ClientToken"], token)

    def test_missing_or_inaccessible_group_stops_before_registration(self):
        self.client.describe_model_package_group.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "group denied"}}, "DescribeModelPackageGroup")
        with self.assertRaises(ClientError):
            self.register()
        self.client.create_model_package.assert_not_called()
        self.s3.put_object.assert_not_called()

    def test_failed_registration_does_not_publish_success_receipt(self):
        for operation in (self.client.create_model_package, self.client.describe_model_package):
            with self.subTest(operation=operation):
                operation.side_effect = RuntimeError("registration failed")
                with self.assertRaisesRegex(RuntimeError, "registration failed"):
                    self.register()
                self.s3.put_object.assert_not_called()
                operation.side_effect = None


if __name__ == "__main__":
    unittest.main()
