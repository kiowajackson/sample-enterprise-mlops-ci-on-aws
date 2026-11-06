"""Local checks for one-time secret creation and non-destructive workspace setup."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from botocore.exceptions import ClientError


SOURCE = Path(__file__).resolve().parents[1] / "notebooks/source/setup/platform_settings.py"
SPEC = importlib.util.spec_from_file_location("setup_settings_under_test", SOURCE)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)
ACCOUNT = "123456789012"
REGION = "us-east-1"
ARN = f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:wsf-mlops-setup-AbCdEf"


def values():
    return {
        "bucket": "example-workload-bucket", "s3_prefix": "wsf-mlops",
        "role_arn": f"arn:aws:iam::{ACCOUNT}:role/ExampleExecutionRole",
        "image_uri": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/example:review",
        "registry_group_prefix": "example-",
    }


def payload():
    return {"schema_version": 1, "account_id": ACCOUNT, "region": REGION, "settings": values()}


class Secrets:
    def __init__(self, existing=None):
        self.value = deepcopy(existing)
        self.calls = []
        self.deny = False
        self.concurrent = None

    def get_secret_value(self, **kwargs):
        self.calls.append(("get", kwargs))
        if self.deny:
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "GetSecretValue")
        if self.value is None:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "absent"}}, "GetSecretValue")
        return {"ARN": ARN, "VersionId": "a" * 32, "SecretString": json.dumps(self.value)}

    def create_secret(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.concurrent:
            self.value = deepcopy(self.concurrent)
            raise ClientError({"Error": {"Code": "ResourceExistsException", "Message": "exists"}}, "CreateSecret")
        self.value = json.loads(kwargs["SecretString"])
        return {"ARN": ARN, "VersionId": "a" * 32}


class Session:
    def __init__(self, secrets):
        self.secrets = secrets
        self.region_name = REGION

    def client(self, name):
        if name == "secretsmanager":
            return self.secrets
        if name == "sts":
            return self
        raise AssertionError(name)

    def get_caller_identity(self):
        return {"Account": ACCOUNT}


class NoSecretsSession(Session):
    def client(self, name):
        if name == "secretsmanager":
            raise AssertionError("Complete local config must not access Secrets Manager")
        return super().client(name)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "notebooks").mkdir()
        self.config = self.root / "notebooks/config.json"
        (self.root / "notebooks/config.example.json").write_text(json.dumps({
            "bucket": "", "s3_prefix": "wsf-mlops", "role_arn": "", "image_uri": "",
            "registry_group_prefix": "", "namespace": "",
            "user_alias": "", "csv_input": "", "test_csv_input": "",
            "inference_csv_input": "", "feature_store_config_uri": "", "metadata_csv": "",
        }))
        self.metadata = self.root / setup.DEFAULT_METADATA
        self.metadata.parent.mkdir(parents=True)
        self.metadata.write_text("column_name\nexample\n")

    def save(self, value):
        self.config.write_text(json.dumps(value))

    def test_create_uses_only_shared_fields_and_reads_exact_version(self):
        self.save(values() | {"namespace": "personal", "csv_input": "s3://example/a.csv", "extra": "local"})
        secrets = Secrets()
        result = setup.create_setup_secret(self.root, Session(secrets))
        self.assertEqual(result["status"], "created")
        self.assertEqual(secrets.value, payload())
        self.assertEqual(secrets.calls[-1], ("get", {"SecretId": ARN, "VersionId": "a" * 32}))
        self.assertNotIn("settings", result)

    def test_older_secret_primary_key_is_not_copied_into_local_settings(self):
        existing = payload()
        existing["settings"]["primary_key"] = "old-model-key"
        secrets = Secrets(existing)
        setup.load_workspace_settings(self.root, Session(secrets))
        self.assertNotIn("primary_key", json.loads(self.config.read_text()))
        self.assertEqual(secrets.value, existing)  # Do not rewrite the shared secret.
        result = setup.create_setup_secret(self.root, Session(secrets))
        self.assertEqual(result["status"], "reused")
        self.assertTrue(all(call[0] == "get" for call in secrets.calls))

    def test_identical_secret_reused_without_writes(self):
        self.save(values())
        secrets = Secrets(payload())
        result = setup.create_setup_secret(self.root, Session(secrets))
        self.assertEqual(result["status"], "reused")
        self.assertEqual([c[0] for c in secrets.calls], ["get"])

    def test_different_secret_not_overwritten(self):
        self.save(values())
        existing = payload()
        existing["settings"]["bucket"] = "another-bucket"
        secrets = Secrets(existing)
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            setup.create_setup_secret(self.root, Session(secrets))
        self.assertEqual([c[0] for c in secrets.calls], ["get"])
        self.assertEqual(secrets.value, existing)

    def test_concurrent_create_only_reuses_identical_values(self):
        self.save(values())
        secrets = Secrets()
        secrets.concurrent = payload()
        self.assertEqual(setup.create_setup_secret(self.root, Session(secrets))["status"], "reused")
        secrets = Secrets()
        secrets.concurrent = payload()
        secrets.concurrent["settings"]["bucket"] = "another-bucket"
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            setup.create_setup_secret(self.root, Session(secrets))

    def test_new_workspace_gets_config_metadata_and_version_receipt(self):
        secrets = Secrets(payload())
        result = setup.load_workspace_settings(self.root, Session(secrets))
        written = json.loads(self.config.read_text())
        self.assertEqual(written["bucket"], values()["bucket"])
        self.assertEqual(written["metadata_csv"], setup.DEFAULT_METADATA)
        self.assertEqual(written["namespace"], "")
        self.assertEqual(written["csv_input"], "")
        self.assertEqual(result["version_id"], "a" * 32)
        self.assertFalse(result["data_generated"])
        self.assertEqual([c[0] for c in secrets.calls], ["get"])
        self.assertEqual(list((self.root / "notebooks").glob(".setup-*")), [])

    def test_existing_personal_and_dataset_fields_are_preserved(self):
        local = values() | {
            "namespace": "my-namespace", "user_alias": "scientist",
            "csv_input": "s3://example/train.csv", "test_csv_input": "missing/test.csv",
            "inference_csv_input": "s3://example/infer.csv", "metadata_csv": "custom.csv",
            "custom_local_setting": "retained",
        }
        self.save(local)
        setup.load_workspace_settings(self.root, Session(Secrets(payload())))
        written = json.loads(self.config.read_text())
        for key, value in local.items():
            self.assertEqual(written[key], value)

    def test_platform_conflict_stops_until_explicit_refresh(self):
        local = values() | {"bucket": "old-bucket", "namespace": "my-namespace", "csv_input": "s3://old/train.csv"}
        del local["image_uri"]  # An incomplete setup still needs the shared secret.
        self.save(local)
        before = self.config.read_bytes()
        with self.assertRaisesRegex(ValueError, "bucket"):
            setup.load_workspace_settings(self.root, Session(Secrets(payload())))
        self.assertEqual(self.config.read_bytes(), before)
        setup.load_workspace_settings(self.root, Session(Secrets(payload())), refresh_shared=True)
        written = json.loads(self.config.read_text())
        self.assertEqual(written["bucket"], values()["bucket"])
        self.assertEqual(written["namespace"], "my-namespace")
        self.assertEqual(written["csv_input"], "s3://old/train.csv")

    def test_invalid_scope_schema_and_fields_never_write_config(self):
        bad = []
        for key, value in (("account_id", "999999999999"), ("region", "eu-west-1"), ("schema_version", 2)):
            item = payload()
            item[key] = value
            bad.append(item)
        item = payload()
        item["settings"]["namespace"] = "shared-by-mistake"
        bad.append(item)
        item = payload()
        item["settings"]["role_arn"] = "arn:aws:iam::999999999999:role/Wrong"
        bad.append(item)
        item = payload()
        item["settings"]["registry_group_prefix"] = ""
        bad.append(item)
        for item in bad:
            with self.subTest(payload=item), self.assertRaises(ValueError):
                setup.load_workspace_settings(self.root, Session(Secrets(item)))
            self.assertFalse(self.config.exists())

    def test_access_denied_does_not_fall_back_or_create(self):
        secrets = Secrets()
        secrets.deny = True
        self.save(values())
        with self.assertRaises(ClientError):
            setup.create_setup_secret(self.root, Session(secrets))
        self.assertEqual([c[0] for c in secrets.calls], ["get"])
        self.config.unlink()
        with self.assertRaises(ClientError):
            setup.load_workspace_settings(self.root, Session(secrets))
        self.assertFalse(self.config.exists())

    def test_shared_group_and_customer_kms_key(self):
        self.save(values() | {"registry_group_prefix": "", "model_package_group": "existing-group"})
        secrets = Secrets()
        setup.create_setup_secret(self.root, Session(secrets), kms_key_id="alias/approved-key")
        self.assertEqual(secrets.calls[1][1]["KmsKeyId"], "alias/approved-key")
        self.assertEqual(secrets.value["settings"]["model_package_group"], "existing-group")

    def test_existing_config_edit_during_fetch_is_preserved(self):
        self.save({key: value for key, value in values().items() if key != "image_uri"})
        secrets = Secrets(payload())
        original_get = secrets.get_secret_value

        def edit_then_get(**kwargs):
            self.save(values() | {"namespace": "new-edit"})
            return original_get(**kwargs)

        secrets.get_secret_value = edit_then_get
        with self.assertRaisesRegex(RuntimeError, "changed"):
            setup.load_workspace_settings(self.root, Session(secrets))
        self.assertEqual(json.loads(self.config.read_text())["namespace"], "new-edit")

    def test_complete_local_config_never_requests_secret_and_retains_bytes(self):
        local = values() | {
            "namespace": "mine", "metadata_csv": "mine.csv",
            "csv_input": "s3://example/train.csv", "test_csv_input": "s3://example/test.csv",
            "inference_csv_input": "s3://example/inference.csv",
            "custom_setting": "preserved",
        }
        self.config.write_text(json.dumps(local, indent=4) + "\n")
        before = self.config.read_bytes()
        secrets = Secrets()
        secrets.deny = True
        result = setup.load_workspace_settings(self.root, NoSecretsSession(secrets))
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse(result["config_changed"])
        self.assertEqual(result["settings_source"], "local")
        self.assertNotIn("secret_arn", result)
        self.assertNotIn("version_id", result)
        self.assertEqual(secrets.calls, [])
        saved = json.loads((self.root / ".runs/setup/platform-settings.json").read_text())
        self.assertEqual(saved, result)

    def test_complete_local_config_still_validates_scope_and_prefix(self):
        for key, value in (("role_arn", "arn:aws:iam::999999999999:role/Wrong"),
                           ("s3_prefix", "../wrong"), ("bucket", "s3://wrong")):
            with self.subTest(key=key):
                self.save(values() | {key: value, "metadata_csv": "mine.csv"})
                before = self.config.read_bytes()
                with self.assertRaises(ValueError):
                    setup.load_workspace_settings(self.root, NoSecretsSession(Secrets()))
                self.assertEqual(self.config.read_bytes(), before)

    def test_complete_local_config_can_explicitly_refresh_from_secret(self):
        self.save(values() | {"bucket": "old-bucket", "metadata_csv": "mine.csv",
                              "namespace": "mine", "csv_input": "s3://old/train.csv"})
        secrets = Secrets(payload())
        result = setup.load_workspace_settings(self.root, Session(secrets), refresh_shared=True)
        self.assertEqual([call[0] for call in secrets.calls], ["get"])
        self.assertEqual(result["settings_source"], "secret")
        self.assertEqual(result["secret_arn"], ARN)
        written = json.loads(self.config.read_text())
        self.assertEqual(written["bucket"], values()["bucket"])
        self.assertEqual(written["namespace"], "mine")
        self.assertEqual(written["csv_input"], "s3://old/train.csv")

    def test_refresh_permission_error_is_not_hidden_by_existing_config(self):
        self.save(values() | {"metadata_csv": "mine.csv"})
        before = self.config.read_bytes()
        secrets = Secrets()
        secrets.deny = True
        with self.assertRaises(ClientError):
            setup.load_workspace_settings(self.root, Session(secrets), refresh_shared=True)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual([call[0] for call in secrets.calls], ["get"])

    def test_new_workspace_without_bundled_metadata_does_not_write(self):
        self.metadata.unlink()
        with self.assertRaisesRegex(ValueError, "metadata"):
            setup.load_workspace_settings(self.root, Session(Secrets(payload())))
        self.assertFalse(self.config.exists())


if __name__ == "__main__":
    unittest.main()
