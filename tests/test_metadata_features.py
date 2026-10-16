"""Check metadata fidelity and model handoff without AWS credentials."""

import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlops-features"))
from mlops_features.metadata import generate_sample, sample_type
from mlops_features.feature_mapping import group_columns, read_mapping
from mlops_features.data import validate_delivery
from mlops_models.entrypoints import get_data, preprocess_inference
from unittest.mock import Mock, patch
import io
import os


class MetadataFeatures(unittest.TestCase):
    def metadata(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["column_name", "column_map", "table_nm", *[f"ref_val{i}" for i in range(1, 6)]])
            writer.writerows(rows)

    def test_references_repeat_and_category_spellings_survive(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            path = root / "schema.csv"
            self.metadata(path, [["flag", "raw_flag", "accounts", "0", "1", "0", "1", ""],
                                 ["code", "raw_code", "customers", "001", "NA", "a,b", "été", ""]])
            result = generate_sample(path, root / "result", rows=12)
            frame = pd.read_csv(result["sample"], dtype=str, keep_default_na=False)
            self.assertEqual(frame.code.tolist(), ["001", "NA", "a,b", "été", ""] * 2 + ["001", "NA"])
            self.assertEqual(frame.customer_id.nunique(), 12)
            meta = json.loads(result["metadata"].read_text())
            self.assertEqual(meta["columns"][0]["column_map"], "raw_flag")
            self.assertEqual(meta["columns"][1]["storage_type"], "string")

    def test_ambiguous_names_and_unknown_overrides_fail(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            path = root / "schema.csv"
            self.metadata(path, [["A", "a", "table", *["1"] * 5], ["a", "b", "table", *["0"] * 5]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                generate_sample(path, root / "duplicate")
            with self.assertRaisesRegex(ValueError, "Unknown type"):
                generate_sample(path, root / "override", type_overrides={"typo": "string"})

    def test_repeated_shared_keys_appear_once_in_csv_and_in_every_group(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            path = root / "schema.csv"
            self.metadata(path, [
                ["entity_id", "source_id", "accounts", *["old-id"] * 5],
                ["asof", "source_date", "accounts", *["old-date"] * 5],
                ["balance", "source_balance", "accounts", "1", "2", "3", "4", "5"],
                ["ENTITY_ID", "source_id", "customers", *["another-id"] * 5],
                ["ASOF", "source_date", "customers", *["another-date"] * 5],
                ["category", "source_category", "customers", "001", "NA", "a,b", "text", ""],
            ])
            result = generate_sample(path, root / "result", rows=12,
                                     record_id="entity_id", event_time="asof")
            df = pd.read_csv(result["sample"], dtype=str, keep_default_na=False)
            self.assertEqual(df.columns.tolist(), ["entity_id", "asof", "balance", "category"])
            self.assertEqual(df.entity_id.nunique(), 12)
            self.assertEqual(df["asof"].unique().tolist(), ["2026-01-01T00:00:00Z"])
            self.assertEqual(df.category.tolist(), ["001", "NA", "a,b", "text", ""] * 2 + ["001", "NA"])
            mapping = read_mapping(result["mapping"], "entity_id", "asof")
            groups = group_columns(mapping, "entity_id", "asof")
            self.assertEqual(len(groups), 2)
            for columns in groups.values():
                self.assertEqual(columns["entity_id"], "String")
                self.assertEqual(columns["asof"], "String")
                self.assertEqual(len(columns), 3)
            validate_delivery(result["sample"], {"record_id": "entity_id", "event_time": "asof",
                                                "mapping": mapping})
            meta = json.loads(result["metadata"].read_text())
            self.assertEqual(len(meta["source_key_columns"]), 4)
            self.assertEqual([row["column_name"] for row in meta["columns"]], ["balance", "category"])

    def test_large_table_splits_before_feature_store_limit(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            path = root / "schema.csv"
            self.metadata(path, [[f"f{i}", f"f{i}", "one_table", *["1"] * 5] for i in range(2500)])
            result = generate_sample(path, root / "result", rows=5)
            mapping = pd.read_csv(result["mapping"])
            self.assertEqual(sorted(mapping.groupby("group").size()), [3, 2497])
        self.assertEqual(sample_type(["001"]), "string")
        self.assertEqual(sample_type(["9223372036854775808"]), "string")
        self.assertEqual(sample_type(["NaN"]), "string")
        self.assertEqual(sample_type(["1.5", "0"]), "continuous")

    def test_training_targets_and_inference_populations_join_by_keys(self):
        df = pd.DataFrame({"id": ["001", "002"], "asof": ["2026-01-01T00:00:00Z"] * 2,
                           "value": [2, 3]})
        settings = {"record_id": "id", "event_time": "asof",
                    "groups": {"features": {"columns": {"id": "String", "asof": "String", "value": "Integral"}}}}
        config = {"primary_key": "id", "timestamp": "asof", "target_key": "target",
                  "features": {"numeric": ["value"], "categorical": []},
                  "model_metadata": {"feature_mapping": {"value": "value"}}}
        s3 = Mock()
        with patch.object(get_data.boto3, "Session") as session, \
             patch.object(get_data, "read_json", return_value=(settings, {})), \
             patch.object(get_data, "resolve_delivery", return_value=({}, {})), \
             patch.object(get_data, "read_group", return_value=(df, {})), \
             patch.dict(os.environ, {"MLOPS_FEATURE_STORE_UNIVERSE_ROOT": "s3://data/population"}):
            session.return_value.client.return_value = s3
            s3.get_object.return_value = {"Body": io.BytesIO(b"id,asof,target\n999,2026-01-01,1\n")}
            with self.assertRaisesRegex(ValueError, "no matching"):
                get_data.get_container_input("s3://data/config.json", "usecase1", config)
            s3.get_object.return_value = {"Body": io.BytesIO(b"id,asof,target\n002,2026-01-01,1\n")}
            selected, _ = get_data.get_container_input("s3://data/config.json", "usecase1", config)
            self.assertEqual(selected.id.tolist(), ["002"])
            self.assertEqual(selected.target.tolist(), [1])
            s3.get_object.return_value = {"Body": io.BytesIO(b"id,asof\n001,2026-01-01\n")}
            selected, _ = get_data.get_container_input("s3://data/config.json", "usecase1", config, inference=True)
            self.assertEqual(selected.id.tolist(), ["001"])
            self.assertNotIn("target", selected)



if __name__ == "__main__":
    unittest.main()
