"""Check delivery selection and pipeline read boundaries without AWS credentials."""

import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from model_fixtures import REGRESSION_MODEL as FIRST, CLASSIFICATION_MODEL as SECOND
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mlops-features"), str(ROOT / "mlops-models/src"), str(ROOT / "mlops-models")]
from mlops_models.entrypoints.get_data import resolve_delivery
from mlops_models.entrypoints import get_data, preprocess_inference
from mlops_models.pipelines.source import build_definitions
from mlops_skeleton.pipelines import walk_steps


class FeatureStorePipeline(unittest.TestCase):
    def test_csv_direct_snapshots_every_column_and_value_without_model_config_or_aws(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MLOPS_DATA_SOURCE": "csv_direct", "MLOPS_CSV_INPUT_URI": "s3://data/sample.csv",
            "PIPELINE_EXECUTION_ID": "execution1",
        }), patch("boto3.Session", side_effect=AssertionError("No AWS reads in csv_direct")):
            path = Path(directory)
            sample = path / "sample.csv"
            sample.write_bytes(
                b'custom_id,AsOf,CustomValue,category,Target,extra\r\n'
                b'0002,2026-01-02,2,NA,0,"keep,\r\nthis"\r\n'
                b'0001,2026-01-01,9007199254740993,001,1,\\N\r\n')
            for input_path in (path, sample):
                with self.subTest(input_path=input_path):
                    output = get_data.execute("no-model-config-needed", input_path, path / "outputs")
                    self.assertEqual(output.read_bytes(), sample.read_bytes())
                    provenance = json.loads((output.parent / "provenance.json").read_text())
                    self.assertEqual(provenance["columns"],
                                     ["custom_id", "AsOf", "CustomValue", "category", "Target", "extra"])
                    self.assertEqual(provenance["records"], 2)
                    self.assertEqual(provenance["model_id"], "no-model-config-needed")
                    self.assertEqual(provenance["input_uri"], "s3://data/sample.csv")
                    self.assertEqual(provenance["pipeline_execution_id"], "execution1")
                    self.assertNotIn("selected_features", provenance)
            (path / "extra.csv").write_text("x\n1\n")
            with self.assertRaisesRegex(ValueError, "one flat CSV"):
                get_data.execute("example", path, path / "outputs")

    def test_custom_metadata_sample_keeps_names_through_selection_and_preprocessing(self):
        import pandas as pd
        from mlops_features.metadata import generate_sample
        from mlops_models.model_base.training.feature_selection import FeatureSelector, run_feature_selection
        from mlops_models.model_base.training.preprocess import process_train
        from mlops_models.runtime.load_source import upload_processing_outputs

        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MLOPS_DATA_SOURCE": "csv_direct",
        }), patch("boto3.Session", side_effect=AssertionError("No Feature Store calls")):
            root = Path(directory)
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["column_name", "column_map", "table_nm",
                                 "ref_val1", "ref_val2", "ref_val3", "ref_val4", "ref_val5"])
                writer.writerows([
                    ["CustomSignal", "signal", "source_table", 1, 2, 3, 4, 5],
                    ["aux_value", "aux", "source_table", 5, 1, 4, 2, 3],
                    ["CustomerBand", "band", "source_table", "A", "B", "A", "C", "B"],
                    ["Target", "outcome", "source_table", 10, 20, 30, 40, 50],
                ])
            sample = generate_sample(metadata, root / "samples", rows=30,
                                     record_id="record_key", event_time="AsOf")["sample"]
            raw = get_data.execute("custom", sample, root / "get-data")
            uploaded = {}
            def upload_file(filename, bucket, key, **kwargs):
                uploaded[bucket, key] = Path(filename).read_bytes()
            with patch("boto3.client") as client:
                client.return_value.upload_file.side_effect = upload_file
                upload_processing_outputs({"Outputs": [{"OutputName": "raw", "S3Output": {
                    "LocalPath": "/opt/ml/processing/output/raw",
                    "S3Uri": "s3://data/models/custom/runs/execution1/data/raw",
                }}]}, Path("/opt/ml/processing/output"), root / "get-data")
            # Simulate SageMaker downloading that exact S3 object for the next step.
            mounted = root / "feature-selection-input"
            mounted.mkdir()
            (mounted / "data.csv").write_bytes(
                uploaded["data", "models/custom/runs/execution1/data/raw/data.csv"])
            self.assertEqual((mounted / "data.csv").read_bytes(), sample.read_bytes())
            for categorical in ([], ["CustomerBand"]):
                with self.subTest(categorical=categorical):
                    config = {
                        "primary_key": "record_key", "timestamp": "AsOf", "target_key": "Target",
                        "features": {"numeric": ["CustomSignal", "aux_value"], "categorical": categorical},
                        "feature_selection": {"sample_rows": 30, "n_estimators": 3, "top_k": 2},
                        "seed": 1, "num_cpus": 1, "task_type": "regression",
                    }
                    selection = run_feature_selection(mounted, root / "selection", config, selector_cls=FeatureSelector)
                    selected = pd.read_csv(selection / "selected_features.csv")["feature"].tolist()
                    self.assertEqual(len(selected), 2)
                    self.assertTrue(set(selected) <= {"CustomSignal", "aux_value", *categorical})
                    self.assertTrue((selection / "random_forest_plots.pdf").read_bytes().startswith(b"%PDF"))
                    prepared = process_train(mounted, selection / "selected_features.csv", root / "preprocess", config)
                    df = pd.read_csv(prepared / "data.csv", dtype={"record_key": str})
                    self.assertEqual(list(df), ["record_key", "AsOf", *selected, "Target"])
                    self.assertEqual(len(df), 30)

    def test_csv_direct_inference_preserves_columns_without_config_features_or_label(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MLOPS_DATA_SOURCE": "csv_direct",
            "MLOPS_FEATURE_STORE_CONFIG_URI": "s3://ignored/stale-config.json",
        }), patch("boto3.Session", side_effect=AssertionError("No Feature Store calls")):
            root = Path(directory)
            inputs, model = root / "inputs", root / "model"
            inputs.mkdir()
            model.mkdir()
            (inputs / "sample.csv").write_text("id,value,unused\n0001,2,ignored\n")
            (model / "parameters.json").write_text(json.dumps({
                "primary_key": "id"}))
            (model / "model_metadata.json").write_text(json.dumps({"feature_mapping": {"value": "value"}}))
            shards = preprocess_inference.execute(FIRST, inputs, root / "output", model_dir=model)
            self.assertEqual(json.loads((shards / "part-000000.jsonl").read_text()), {"id": "0001", "value": 2, "unused": "ignored"})
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
