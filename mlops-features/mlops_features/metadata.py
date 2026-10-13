"""Generate a repeated-reference sample and an explicit Feature Store mapping."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .feature_mapping import read_mapping


def sample_type(values):
    """Suggest storage from five examples; preserve category spellings such as 001."""
    values = [value for value in values if value != ""]
    if values and all(re.fullmatch(r"-?(0|[1-9][0-9]*)", value) for value in values):
        if all(-(2**63) <= int(value) < 2**63 for value in values):
            return "binary" if set(values) <= {"0", "1"} else "integer"
        return "string"
    number = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    if values and all(re.fullmatch(number, value) and math.isfinite(float(value)) for value in values):
        return "continuous"
    return "string"


def generate_sample(metadata_csv, output_dir, *, rows=100, record_id="customer_id",
                    event_time="feature_asof_dt", asof="2026-01-01T00:00:00Z", type_overrides=None,
                    deduplicate=False):
    """Repeat ref_val1..5 as rows, preserving column_name and retaining its source mapping.

    Return local sample, mapping and source-metadata paths. No AWS calls.
    Type/length codes are retained, not interpreted as a confirmed data dictionary.
    Configured ID/time columns may repeat across source tables. Generate them once
    in the sample; group_columns adds them to every Feature Store group.
    For dummy data, deduplicate=True keeps the first definition of each column.
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError("rows must be a positive integer.")
    with Path(metadata_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["column_name", "column_map", "table_nm", *[f"ref_val{i}" for i in range(1, 6)]]
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("The metadata CSV needs unique column headers.")
        if set(required) - set(reader.fieldnames):
            raise ValueError(f"Metadata CSV requires: {', '.join(required)}")
        metadata = list(reader)
    if not metadata or any(None in row or any(v is None for v in row.values()) for row in metadata):
        raise ValueError("Supply nonempty metadata with the same number of fields on every row.")
    overrides = type_overrides or {}
    unknown = set(overrides) - {row["column_name"] for row in metadata}
    if unknown:
        raise ValueError(f"Unknown type override columns: {sorted(unknown)}")

    if deduplicate:
        first_by_name = {}
        for row in metadata:
            row["column_name"] = row["column_name"].strip()
            first_by_name.setdefault(row["column_name"].lower(), row)
        print(f"Dummy metadata: removed {len(metadata) - len(first_by_name)} duplicate column definitions.")
        metadata = list(first_by_name.values())

    # Shared keys already have generated values; they are not ordinary features.
    key_names = {record_id.lower(), event_time.lower()}
    features = [row for row in metadata if row["column_name"].lower() not in key_names]

    # A source table maps to one group, or successive parts above 2,497 features.
    counts, mapping = {}, []
    for row in features:
        table = row["table_nm"]
        if not table.strip():
            raise ValueError(f"Missing table_nm for {row['column_name']}.")
        count = counts.get(table, 0)
        counts[table] = count + 1
        slug = re.sub(r"[^a-z0-9]+", "_", table.lower()).strip("_")[:18] or "table"
        group = f"t_{slug}_{hashlib.sha256(table.encode()).hexdigest()[:8]}_p{count // 2497 + 1}"
        references = [row[f"ref_val{i}"] for i in range(1, 6)]
        kind = overrides.get(row["column_name"], sample_type(references))
        mapping.append({"feature": row["column_name"], "group": group, "type": kind})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    mapping_path, sample_path = output / "feature-mapping.csv", output / "sample.csv"
    with mapping_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["feature", "group", "type"])
        writer.writeheader()
        writer.writerows(mapping)
    # Reuse the application's name, type, reserved-field and group-size checks.
    mapping = read_mapping(mapping_path, record_id, event_time)
    with sample_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([record_id, event_time, *(row["column_name"] for row in features)])
        for index in range(rows):
            reference = f"ref_val{index % 5 + 1}"
            writer.writerow([f"sample-{index + 1:06d}", asof, *(row[reference] for row in features)])
    details = {
        "generation": "Row i takes ref_val((i-1) modulo 5)+1; synthetic IDs and a fixed event time.",
        "metadata_sha256": hashlib.sha256(Path(metadata_csv).read_bytes()).hexdigest(),
        "source_key_columns": [
            {key: value for key, value in row.items() if not key.startswith("ref_val")}
            for row in metadata if row["column_name"].lower() in key_names
        ],
        "columns": [{**{key: value for key, value in row.items() if not key.startswith("ref_val")},
                     "group": mapped["group"], "storage_type": mapped["type"]}
                    for row, mapped in zip(features, mapping)],
    }
    source_path = output / "source-metadata.json"
    source_path.write_text(json.dumps(details, indent=2) + "\n")
    return {"sample": sample_path, "mapping": mapping_path, "metadata": source_path,
            "rows": rows, "features": len(mapping), "tables": len(counts)}
