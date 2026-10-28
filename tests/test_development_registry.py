"""Exercise shared-group isolation, exact replay and Development-only registration."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from mlops_models.registry.model_package import (
    approved_development_package, development_scope, resolve_latest_approved_model,
    validate_approved_model_package,
)
import test_source_deployment as existing_checks


def package(version, namespace="parul", stage="Development", stage_status="Approved"):
    return {
        "ModelPackageArn": f"arn:aws:sagemaker:us-east-1:123456789012:model-package/shared/{version}",
        "ModelPackageVersion": version, "ModelPackageGroupName": "shared",
        "ModelPackageStatus": "Completed", "ModelApprovalStatus": "Approved",
        "ModelLifeCycle": {"Stage": stage, "StageStatus": stage_status},
        "CustomerMetadataProperties": {
            "DeploymentNamespace": namespace, "Environment": "dev", "ModelId": "usecase2",
            "SourceUri": f"s3://bucket/source/release-{version}", "SourceDigest": f"{version:064x}",
        },
        "MetadataProperties": {"CommitId": f"{version:040x}"},
        "InferenceSpecification": {"Containers": [{
            "Image": "image@sha256:" + "b" * 64,
            "ModelDataUrl": f"s3://bucket/models/{version}/model.tar.gz",
        }]},
    }


class DevelopmentRegistry(unittest.TestCase):
    def resolve(self, versions, *, exact=""):
        client = Mock()
        client.get_paginator.return_value.paginate.return_value = [
            {"ModelPackageSummaryList": [
                {k: v[k] for k in ("ModelPackageArn", "ModelPackageVersion", "ModelPackageStatus") if k in v}
                for v in versions
            ]}
        ]
        by_arn = {v["ModelPackageArn"]: v for v in versions}
        client.describe_model_package.side_effect = lambda **kw: deepcopy(by_arn[kw["ModelPackageName"]])
        with TemporaryDirectory() as directory:
            result = resolve_latest_approved_model(
                "shared", directory, namespace="parul", model_id="usecase2", environment="dev",
                package_arn=exact, client=client)
            self.assertEqual(result, json.loads((Path(directory) / "package.json").read_text()))
        return result, client

    def test_newer_other_namespace_and_production_candidate_cannot_win(self):
        for selection in ("", "latest-approved"):
            with self.subTest(selection=selection):
                result, _ = self.resolve([package(2), package(20, namespace="other"),
                                          package(30, stage="ProductionCandidate"), package(4)],
                                         exact=selection)
                self.assertEqual(result["version"], 4)
                self.assertEqual(result["namespace"], "parul")

    def test_exact_replay_keeps_old_source_and_does_not_list_latest_models(self):
        old, newer = package(2), package(10)
        result, client = self.resolve([old, newer], exact=old["ModelPackageArn"])
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["source_uri"], old["CustomerMetadataProperties"]["SourceUri"])
        self.assertEqual(result["source_digest"], old["CustomerMetadataProperties"]["SourceDigest"])
        client.get_paginator.assert_not_called()

    def test_explicit_version_cannot_bypass_scope_stage_or_approval(self):
        cases = [
            package(1, namespace="other"), package(1, stage="ProductionCandidate"),
            package(1, stage="Production"), package(1, stage_status="PendingApproval"),
        ]
        for field, value in (("Environment", "prod"), ("ModelId", "usecase1")):
            candidate = package(1)
            candidate["CustomerMetadataProperties"][field] = value
            cases.append(candidate)
        rejected = package(1)
        rejected["ModelApprovalStatus"] = "Rejected"
        cases.append(rejected)
        unscoped = package(1)
        del unscoped["CustomerMetadataProperties"]["DeploymentNamespace"]
        cases.append(unscoped)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, "No eligible"):
                self.resolve([candidate], exact=candidate["ModelPackageArn"])

    def test_missing_saved_source_does_not_use_current_pipeline_source(self):
        candidate = package(1)
        del candidate["CustomerMetadataProperties"]["SourceUri"]
        with self.assertRaisesRegex(ValueError, "no saved source binding"):
            self.resolve([candidate])

    def test_no_own_model_never_falls_back_to_another_namespace(self):
        for selection in ("", "latest-approved"):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "No eligible"):
                self.resolve([package(10, namespace="another-person")], exact=selection)

    def test_each_failed_or_missing_field_names_actual_and_expected_values(self):
        fields = [
            (("ModelPackageGroupName",), "ModelPackageGroupName", "shared"),
            (("ModelPackageStatus",), "ModelPackageStatus", "Completed"),
            (("ModelLifeCycle", "Stage"), "ModelLifeCycle.Stage", "Development"),
            (("CustomerMetadataProperties", "DeploymentNamespace"), "DeploymentNamespace", "parul"),
            (("CustomerMetadataProperties", "ModelId"), "ModelId", "usecase2"),
            (("CustomerMetadataProperties", "Environment"), "Environment", "dev"),
            (("ModelApprovalStatus",), "ModelApprovalStatus", "Approved"),
            (("ModelLifeCycle", "StageStatus"), "ModelLifeCycle.StageStatus", "Approved"),
        ]
        for path, label, expected in fields:
            for missing in (False, True):
                candidate = package(1)
                target = candidate[path[0]] if len(path) == 2 else candidate
                actual = None if missing else expected + " "
                if missing:
                    del target[path[-1]]
                else:
                    target[path[-1]] = actual
                with self.subTest(field=label, missing=missing):
                    with self.assertRaises(ValueError) as error:
                        self.resolve([candidate], exact=candidate["ModelPackageArn"])
                    self.assertIn(candidate["ModelPackageArn"], str(error.exception))
                    self.assertIn(f"{label}: actual={actual!r}, expected={expected!r}", str(error.exception))

    def test_one_error_reports_all_failed_checks(self):
        candidate = package(1, namespace="previous", stage="Production", stage_status="PendingApproval")
        candidate["ModelApprovalStatus"] = "Rejected"
        del candidate["CustomerMetadataProperties"]["ModelId"]
        with self.assertRaises(ValueError) as error:
            self.resolve([candidate], exact=candidate["ModelPackageArn"])
        message = str(error.exception)
        for fragment in (
            "DeploymentNamespace: actual='previous', expected='parul'",
            "ModelLifeCycle.Stage: actual='Production', expected='Development'",
            "ModelLifeCycle.StageStatus: actual='PendingApproval', expected='Approved'",
            "ModelApprovalStatus: actual='Rejected', expected='Approved'",
            "ModelId: actual=None, expected='usecase2'",
        ):
            self.assertIn(fragment, message)

    def test_latest_logs_reasons_and_stops_at_the_highest_eligible_version(self):
        pending = package(30)
        pending["ModelPackageStatus"] = "InProgress"
        output = io.StringIO()
        with redirect_stdout(output):
            result, client = self.resolve([package(2), package(20, namespace="previous"), pending, package(4)])
        self.assertEqual(result["version"], 4)
        self.assertIn("ModelPackageStatus: actual='InProgress', expected='Completed'", output.getvalue())
        self.assertIn("DeploymentNamespace: actual='previous', expected='parul'", output.getvalue())
        self.assertEqual(
            [call.kwargs["ModelPackageName"] for call in client.describe_model_package.call_args_list],
            [package(version)["ModelPackageArn"] for version in (30, 20, 4)],
        )
        client.get_paginator.return_value.paginate.assert_called_once_with(
            ModelPackageGroupName="shared", ModelApprovalStatus="Approved")

    def test_empty_approved_list_explains_the_filter(self):
        with self.assertRaisesRegex(ValueError, "No versions with ModelApprovalStatus='Approved'"):
            self.resolve([])

    def test_validate_package_reports_scope_values_without_writing_authorization(self):
        candidate = package(1, namespace="previous")
        client = Mock()
        client.describe_model_package.return_value = candidate
        container = candidate["InferenceSpecification"]["Containers"][0]
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as error:
                validate_approved_model_package(
                    candidate["ModelPackageArn"], "shared", container["Image"],
                    container["ModelDataUrl"], directory, namespace="parul",
                    model_id="usecase2", environment="dev", client=client)
            self.assertIn("DeploymentNamespace: actual='previous', expected='parul'", str(error.exception))
            self.assertFalse((Path(directory) / "package.json").exists())

    def test_notebook_scope_check_still_allows_a_model_awaiting_approval(self):
        candidate = package(1, stage_status="PendingApproval")
        candidate["ModelApprovalStatus"] = "PendingManualApproval"
        self.assertTrue(development_scope(candidate, "shared", "parul", "usecase2", "dev"))
        self.assertFalse(approved_development_package(candidate, "shared", "parul", "usecase2", "dev"))
        with self.assertRaisesRegex(ValueError, "DeploymentNamespace, ModelId"):
            development_scope(candidate, "shared", "", None, "dev")

    def test_notebook_registration_stays_development_even_if_a_ci_flag_is_supplied(self):
        fixture = existing_checks.ExistingGroupRegistration()
        fixture.setUp()
        fixture.settings["DEPLOY_MODE"] = "ci-deploy"
        fixture.register()
        request = fixture.client.create_model_package.call_args.kwargs
        self.assertEqual(request["ModelLifeCycle"],
                         {"Stage": "Development", "StageStatus": "PendingApproval"})
        self.assertEqual(request["CustomerMetadataProperties"]["DeploymentNamespace"], "review")
        self.assertEqual(request["CustomerMetadataProperties"]["SourceUri"], "s3://bucket/original-source")
        self.assertNotEqual(request["CustomerMetadataProperties"]["SourceUri"], fixture.settings["SOURCE_URI"])

    def test_saved_code_replay_uses_the_model_source_and_image(self):
        fixture = existing_checks.SourceDeployment()
        fixture.setUp()
        fixture.identity["mode"] = "ci-deploy"
        body = fixture.definitions("processing", model_code_source="artifact")["inference"]
        parameters = {p["Name"]: p for p in body["Parameters"]}
        self.assertEqual(parameters["ModelPackageArn"]["DefaultValue"], "latest-approved")
        for step in body["Steps"][1:]:
            args = step["Arguments"]
            source = next(p for p in args["ProcessingInputs"] if p["InputName"] == "source")
            self.assertEqual(source["S3Input"]["S3Uri"]["Std:JsonGet"]["Path"], "source_uri")
            self.assertEqual(args["AppSpecification"]["ImageUri"]["Std:JsonGet"]["Path"], "image")
            self.assertEqual(args["Environment"]["MLOPS_MODEL_CODE_SOURCE"], "artifact")
        resolver = body["Steps"][0]["Arguments"]
        self.assertEqual(resolver["Environment"]["DEPLOYMENT_NAMESPACE"], "review")
        self.assertIn({"Get": "Parameters.ModelPackageArn"},
                      resolver["AppSpecification"]["ContainerArguments"])

    def test_csv_launch_defaults_can_all_be_submitted_by_studio(self):
        fixture = existing_checks.SourceDeployment()
        fixture.setUp()
        for runtime in ("processing", "both"):
            definitions = fixture.definitions(
                runtime, data_source="csv_direct", inference_data_source="csv_direct",
                csv_input_uri="s3://data/scoring.csv", monitoring={"enabled": True})
            for kind, body in definitions.items():
                with self.subTest(runtime=runtime, pipeline=kind):
                    overrides = {p["Name"]: p["DefaultValue"] for p in body["Parameters"]}
                    self.assertTrue(all(value != "" for value in overrides.values()), overrides)
                    if kind == "inference":
                        self.assertEqual(overrides["ModelPackageArn"], "latest-approved")
                        self.assertEqual(overrides["ReferenceExecutionArn"], "none")


if __name__ == "__main__":
    unittest.main()
