"""Check delivery selection and pipeline read boundaries without AWS credentials."""

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mlops-features"), str(ROOT / "mlops-models/src"), str(ROOT / "mlops-models")]
from mlops_models.entrypoints.get_data import resolve_delivery
from mlops_models.entrypoints import get_data, preprocess_inference
from mlops_models.pipelines.source import build_definitions
from mlops_skeleton.pipelines import walk_steps


class FeatureStorePipeline(unittest.TestCase):
    def test_csv_direct_reads_local_columns_and_records_provenance_without_aws(self):
        config = {"identifier": "id", "timestamp": "date", "label": "label",
                  "features": {"numeric": ["value"], "categorical": ["category"]}}
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MLOPS_DATA_SOURCE": "csv_direct", "MLOPS_CSV_INPUT_URI": "s3://data/sample.csv",
            "PIPELINE_EXECUTION_ID": "execution1",
        }), patch("boto3.Session", side_effect=AssertionError("No AWS reads in csv_direct")):
            path = Path(directory)
            (path / "sample.csv").write_text(
                "id,date,value,category,label,unused\n"
                "0002,2026-01-01,2,NA,0,ignored\n"
                "0001,2026-01-01,9007199254740993,001,1,ignored\n")
            df, provenance = get_data.get_csv_input(path, "example", config)
            self.assertEqual(list(df), ["id", "date", "value", "category", "label"])
            self.assertEqual(df["id"].tolist(), ["0001", "0002"])
            self.assertEqual(df["category"].tolist(), ["001", "NA"])
            self.assertEqual(df["value"].tolist(), [9007199254740993, 2])
            self.assertEqual(df["date"].tolist(), ["2026-01-01T00:00:00Z"] * 2)
            output = get_data.write_container_output(df, provenance, path / "outputs")
            self.assertEqual(json.loads((output / "provenance.json").read_text())["records"], 2)
            self.assertEqual(provenance["input_uri"], "s3://data/sample.csv")
            (path / "extra.csv").write_text("x\n1\n")
            with self.assertRaisesRegex(ValueError, "one flat CSV"):
                get_data.get_csv_input(path, "example", config)

    def test_csv_direct_inference_reads_saved_config_and_needs_no_label(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MLOPS_DATA_SOURCE": "csv_direct",
            "MLOPS_FEATURE_STORE_CONFIG_URI": "s3://ignored/stale-config.json",
        }), patch("boto3.Session", side_effect=AssertionError("No Feature Store calls")):
            root = Path(directory)
            inputs, model = root / "inputs", root / "model"
            inputs.mkdir()
            model.mkdir()
            (inputs / "sample.csv").write_text("id,value,unused\n0001,2,ignored\n")
            (model / "model-config.json").write_text(json.dumps({
                "identifier": "id", "features": {"numeric": ["value"], "categorical": []}}))
            shards = preprocess_inference.execute("example", inputs, root / "output", model_dir=model)
            self.assertEqual(json.loads((shards / "part-000000.jsonl").read_text()), {"id": "0001", "value": 2})
            self.assertEqual((root / "output/raw/data.csv").read_bytes(), (inputs / "sample.csv").read_bytes())
            self.assertEqual(json.loads((root / "output/raw/provenance.json").read_text())["data_source"], "csv_direct")

    def test_csv_direct_changes_only_the_data_readers_and_preserves_all_handoffs(self):
        args = ("review", "arn:aws:iam::123456789012:role/jobs", "image:tag",
                "s3://data/app", "s3://data/source",
                {"mode": "direct-deploy", "name": "test", "commit": "b" * 40, "dirty": True})
        store = build_definitions(*args, region="us-east-1", batch_runtime="processing",
                                  feature_store_config_uri="s3://data/config.json")
        direct = build_definitions(*args, region="us-east-1", batch_runtime="processing",
                                   data_source="csv_direct", csv_input_uri="s3://data/sample.csv")
        for kind, body in direct.items():
            reader = "GetData" if kind == "training" else "Preprocess"
            expected_steps = {step["Name"]: step for step in store[kind]["Steps"]}
            parameters = {p["Name"]: p for p in body["Parameters"]}
            self.assertEqual(parameters["CsvInputUri"]["DefaultValue"], "s3://data/sample.csv")
            self.assertNotIn("FeatureStoreConfigUri", parameters)
            self.assertNotIn("InputDataRoot", parameters)
            for step in body["Steps"]:
                if step["Name"] != reader:
                    self.assertEqual(step, expected_steps[step["Name"]])
                    continue
                arguments = step["Arguments"]
                self.assertEqual(arguments["Environment"]["MLOPS_DATA_SOURCE"], "csv_direct")
                self.assertNotIn("MLOPS_FEATURE_STORE_CONFIG_URI", arguments["Environment"])
                self.assertEqual(arguments["ProcessingOutputConfig"],
                                 expected_steps[reader]["Arguments"]["ProcessingOutputConfig"])
                inputs = {i["InputName"]: i["S3Input"] for i in arguments["ProcessingInputs"]}
                self.assertEqual(inputs["input"]["LocalPath"], "/opt/ml/processing/input")
                self.assertEqual(inputs["input"]["S3Uri"], {"Get": "Parameters.CsvInputUri"})
                self.assertEqual(inputs["input"]["S3InputMode"], "File")
        with self.assertRaisesRegex(ValueError, "feature_store_config_uri"):
            build_definitions(*args, region="us-east-1")
        with self.assertRaisesRegex(ValueError, "labels in the flat CSV"):
            build_definitions(*args, region="us-east-1", data_source="csv_direct",
                              csv_input_uri="s3://data/sample.csv", universe_data_root="s3://data/universe")

    def test_runtime_reads_each_group_once_without_ingestion_readiness_polling(self):
        for module in (get_data, preprocess_inference):
            with self.subTest(module=module.__name__):
                session = Mock()
                session.client.return_value.get_object.return_value = {
                    "Body": io.BytesIO(b"id,event_time,value\n0001,2026-01-01T00:00:00Z,9007199254740993\n")
                }
                settings = {"record_id": "id", "event_time": "event_time", "database": "db"}
                group = {"name": "test", "table": "feature_table", "database": "db",
                         "columns": {"id": "String", "event_time": "String", "value": "Integral"}}
                receipt = {"delivery_id": "delivery", "completed_at": "2026-01-01T01:00:00Z", "records": 100}
                with patch.object(module, "run_query", return_value=("query1", "s3://query/result.csv")) as query:
                    df, provenance = module.read_group(session, settings, group, list(group["columns"]), receipt)
                query.assert_called_once()
                self.assertEqual(df["id"].tolist(), ["0001"])
                self.assertEqual(df["value"].tolist(), [9007199254740993])
                self.assertEqual(provenance["query_id"], "query1")
                session.client.assert_called_once_with("s3")

    def test_latest_completed_receipt_or_explicit_replay(self):
        s3 = Mock()
        settings = {"bucket": "data", "prefix": "team/"}
        date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "team/deliveries/older/ingestion.json", "LastModified": date}]},
            {"Contents": [{"Key": "team/deliveries/new/ingestion.json", "LastModified": date.replace(day=2)},
                          {"Key": "team/deliveries/partial/source.json", "LastModified": date.replace(day=3)}]},
        ]
        def read(**kwargs):
            return {"Body": io.BytesIO(json.dumps({"key": kwargs["Key"]}).encode()), "VersionId": "v1"}
        s3.get_object.side_effect = read
        receipt, identity = resolve_delivery(s3, settings)
        self.assertEqual(receipt["key"], "team/deliveries/new/ingestion.json")
        self.assertEqual(identity["version_id"], "v1")
        s3.get_paginator.reset_mock()
        receipt, _ = resolve_delivery(s3, settings, "older")
        self.assertEqual(receipt["key"], "team/deliveries/older/ingestion.json")
        s3.get_paginator.assert_not_called()

    def test_no_completed_ingestion_and_permission_errors_are_explicit(self):
        s3 = Mock()
        settings = {"bucket": "data", "prefix": "team/"}
        s3.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
        with self.assertRaisesRegex(ValueError, "No completed"):
            resolve_delivery(s3, settings)
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")
        with self.assertRaises(ClientError):
            resolve_delivery(s3, settings, "known-delivery")
        with self.assertRaisesRegex(ValueError, "delivery ID"):
            resolve_delivery(s3, settings, "../other")

    def test_only_get_data_and_inference_preprocess_query_feature_store(self):
        for runtime in ("processing", "transform"):
            with self.subTest(runtime=runtime):
                definitions = build_definitions(
                    "review", "arn:aws:iam::123456789012:role/jobs",
                    "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "a" * 64,
                    "s3://data/app", "s3://data/app/source", {"mode": "direct-deploy", "name": "test", "commit": "b" * 40, "dirty": True},
                    region="us-east-1", batch_runtime=runtime,
                    feature_store_config_uri="s3://data/features/settings.json",
                )
                for kind, body in definitions.items():
                    parameters = {p["Name"]: p for p in body["Parameters"]}
                    self.assertNotIn("InputDataRoot", parameters)
                    self.assertEqual(parameters["FeatureStoreDeliveryId"]["DefaultValue"], "")
                    self.assertNotIn("Parameters.InputDataRoot", json.dumps(body))
                    readers = []
                    for step in walk_steps(body["Steps"]):
                        args = step["Arguments"]
                        if "MLOPS_FEATURE_STORE_CONFIG_URI" in args.get("Environment", {}):
                            readers.append(step["Name"])
                            paths = [item["S3Input"]["LocalPath"] for item in args["ProcessingInputs"]]
                            self.assertNotIn("/opt/ml/processing/input", paths)
                            self.assertIn("/opt/ml/processing/source", paths)
                            raw = next(o for o in args["ProcessingOutputConfig"]["Outputs"] if o["OutputName"] == "raw")
                            self.assertEqual(args["Environment"]["MLOPS_FEATURE_STORE_SNAPSHOT_URI"], raw["S3Output"]["S3Uri"])
                    self.assertEqual(readers, ["GetData"] if kind == "training" else ["Preprocess"])


if __name__ == "__main__":
    unittest.main()
