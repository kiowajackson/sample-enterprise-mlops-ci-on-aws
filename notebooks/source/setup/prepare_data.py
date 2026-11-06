"""CSV preparation extracted from the detailed notebook; no notebook parsing at runtime."""
from platform_settings import _atomic_json


def prepare_data(root, *, options=None, progress=None):
    """Generate the three CSV datasets, update their local bindings, and return a report."""
    options = dict(options or {})
    unknown = set(options) - set(['demo_10000_features', 'demo_model_id', 'event_time', 'inference_rows', 'missing_values', 'numeric_zero_fraction', 'randomize_numeric', 'sample_asof', 'sample_rows', 'sample_scale', 'split_seed', 'test_fraction', 'type_overrides'])
    if unknown:
        raise ValueError(f"Unknown preparation options: {sorted(unknown)}")
    from pathlib import Path
    import json

    ROOT = Path(root).resolve()
    CONFIG_PATH = ROOT / "notebooks/config.json"
    TEMPLATE_PATH = ROOT / "notebooks/config.example.json"
    if not CONFIG_PATH.exists():
        raise ValueError(f"Load platform settings before calling this function: {CONFIG_PATH}")

    # Read supplied inputs; optional settings are declared in the Defaults cell below.
    notebook_config = json.loads(TEMPLATE_PATH.read_text()) | json.loads(CONFIG_PATH.read_text())
    metadata_csv = notebook_config["metadata_csv"]
    bucket = notebook_config["bucket"]
    role_arn = notebook_config["role_arn"]
    image_uri = notebook_config["image_uri"]

    model_keys = {
        path.parent.name: json.loads(path.read_text()).get("primary_key")
        for path in sorted((ROOT / "mlops-models/src/mlops_models/models").glob("*/parameters.json"))
    }
    if not model_keys or any(not isinstance(key, str) or not key.strip() for key in model_keys.values()):
        raise ValueError("Set primary_key in each model's parameters.json before generating sample data.")
    if len(set(model_keys.values())) != 1:
        raise ValueError(
            "Shared sample generation requires matching primary_key values in the model parameter files. "
            "For models with different ID columns, supply prepared CSV inputs. "
            f"Found: {model_keys}"
        )
    primary_key = next(iter(model_keys.values()))


    import hashlib
    import re

    namespace = notebook_config.get("namespace", "").strip()
    if not namespace:
        # A supplied alias also works outside Studio. Do not infer a person from
        # the shared container login or an assumed-role session name.
        deployment_user = notebook_config.get("user_alias", "").strip()
        identity = deployment_user
        if not deployment_user:
            metadata_path = Path("/opt/ml/metadata/resource-metadata.json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            deployment_user = metadata.get("UserProfileName", "").strip()
            identity = metadata.get("DomainId", "") + "/" + deployment_user
        if not deployment_user:
            raise ValueError(
                "No Studio user profile was found. Set user_alias once in config.json "
                "(for example your username), or supply namespace explicitly."
            )
        label = re.sub(r"[^a-z0-9]+", "-", deployment_user.lower()).strip("-")[:7].rstrip("-") or "user"
        namespace = "wsf-" + label + "-" + hashlib.sha256(identity.encode()).hexdigest()[:8]
        notebook_config["namespace"] = namespace
        _atomic_json(CONFIG_PATH, notebook_config)

    s3_prefix = notebook_config.get("s3_prefix", "wsf-mlops").strip().strip("/")
    if (not re.fullmatch(r"[A-Za-z0-9_./-]+", s3_prefix)
            or any(part in {"", ".", ".."} for part in s3_prefix.split("/"))):
        raise ValueError("s3_prefix must be a nonempty folder path without '.' or '..' components.")
    storage_prefix = f"{s3_prefix}/{namespace}"
    data_source = "csv_direct"
    sample_scale = options.get("sample_scale", 100)
    sample_rows = options.get("sample_rows", 100 * sample_scale)
    test_fraction = options.get("test_fraction", 0.2)
    inference_rows = options.get("inference_rows", 20 * sample_scale)
    split_seed = options.get("split_seed", 42)
    randomize_numeric = options.get("randomize_numeric", True)
    numeric_zero_fraction = options.get("numeric_zero_fraction", 0.10)
    missing_values = options.get("missing_values", ("", ".", r"\N"))
    record_id = primary_key  # Generated ID-column name read from model parameters.
    event_time = options.get("event_time", "feature_asof_dt")
    sample_asof = options.get("sample_asof", "2026-01-01T00:00:00Z")
    type_overrides = options.get("type_overrides", {})
    feature_store_role_arn = role_arn  # Set a different offline writer role here if required.
    athena_workgroup = "primary"
    kms_key = ""
    create_missing_groups = False
    verify_offline_readback = False

    demo_10000_features = options.get("demo_10000_features", False)
    demo_model_id = options.get("demo_model_id", "usecase2")


    from time import perf_counter

    preparation_started = perf_counter()


    def show_progress(message):
        """Print elapsed wall time immediately in notebooks and ordinary Python scripts."""
        print(f"[{perf_counter() - preparation_started:6.1f}s] {message}", flush=True)
        if progress is not None:
            progress(message)


    show_progress("Loading preparation libraries...")

    from uuid import uuid4
    import csv
    import hashlib
    import math
    import re
    import sys
    from random import Random
    import pandas as pd

    sys.path[:0] = [str(ROOT / "mlops-features"), str(ROOT / "mlops-models/src")]
    import mlops_features
    assert Path(mlops_features.__file__).resolve().is_relative_to(ROOT.resolve()), (
        f"STOP: this kernel loaded code from {mlops_features.__file__}. Restart this notebook's kernel for the new checkout."
    )
    from mlops_features.feature_mapping import read_mapping, group_columns, group_name
    from mlops_features.data import validate_delivery, ingest_csv, read_dataset

    # Sample-generation helpers live in this cell; no metadata module is imported.
    def sample_type(values, missing_values=("", ".", r"\N")):
        """Suggest storage from five examples; preserve category spellings such as 001."""
        values = [value.strip() for value in values if value.strip() not in missing_values]
        if values and all(re.fullmatch(r"[+-]?(0|[1-9][0-9]*)", value) for value in values):
            if all(-(2**63) <= int(value) < 2**63 for value in values):
                return "binary" if {int(value) for value in values} <= {0, 1} else "integer"
            return "string"
        number = r"[+-]?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
        if values and all(re.fullmatch(number, value) and math.isfinite(float(value)) for value in values):
            return "binary" if {float(value) for value in values} <= {0, 1} else "continuous"
        return "string"


    def generate_sample(metadata_csv, output_dir, *, rows=100, record_id="customer_id",
                        event_time="feature_asof_dt", asof="2026-01-01T00:00:00Z", type_overrides=None,
                        deduplicate=False, randomize_numeric=False, seed=42, target_columns=(),
                        numeric_zero_fraction=0.0, missing_values=("", ".", r"\N")):
        """Generate rows from ref_val1..5, retaining column names and their source mapping.

        Return local sample, mapping and source-metadata paths. No AWS calls.
        Optional numeric randomization samples uniformly between the reference min/max.
        Randomized mode samples reference values independently across columns/rows.
        Numeric missing markers become empty fields; string categories retain their spelling.
        Binary flags, constants and targets retain their allowed reference values.
        numeric_zero_fraction injects zeros among nonmissing randomized numeric values;
        those zeros can lie outside the reference range. It does not affect targets.
        Numeric columns representing categories can be marked string in type_overrides.
        Type/length codes are retained, not interpreted as a confirmed data dictionary.
        Configured ID/time columns may repeat across source tables. Generate them once
        in the sample; group_columns adds them to every Feature Store group.
        For dummy data, deduplicate=True keeps the first definition of each column.
        """
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise ValueError("rows must be a positive integer.")
        if isinstance(numeric_zero_fraction, bool) or not 0 <= numeric_zero_fraction < 1:
            raise ValueError("numeric_zero_fraction must be at least 0 and less than 1.")
        show_progress("Reading feature metadata...")
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
            metadata = list(first_by_name.values())

        # Shared keys already have generated values; they are not ordinary features.
        key_names = {record_id.lower(), event_time.lower()}
        features = [row for row in metadata if row["column_name"].lower() not in key_names]

        # A source table maps to one group, or successive parts above 2,497 features.
        counts, mapping, samplers, examples = {}, [], {}, []
        rng = Random(seed)
        targets = set(target_columns)
        for position, row in enumerate(features):
            table = row["table_nm"]
            if not table.strip():
                raise ValueError(f"Missing table_nm for {row['column_name']}.")
            count = counts.get(table, 0)
            counts[table] = count + 1
            slug = re.sub(r"[^a-z0-9]+", "_", table.lower()).strip("_")[:18] or "table"
            group = f"t_{slug}_{hashlib.sha256(table.encode()).hexdigest()[:8]}_p{count // 2497 + 1}"
            references = [row[f"ref_val{i}"] for i in range(1, 6)]
            kind = overrides.get(row["column_name"], sample_type(references, missing_values))
            mapping.append({"feature": row["column_name"], "group": group, "type": kind})
            if kind in {"binary", "integer", "continuous"}:
                references = ["" if value.strip() in missing_values else value.strip()
                              for value in references]
                if kind == "binary":
                    if any(float(value) not in {0, 1} for value in references if value):
                        raise ValueError(f"Binary feature {row['column_name']} has references other than 0/1.")
                    references = [str(int(float(value))) if value else "" for value in references]
            examples.append(references)
            if randomize_numeric and row["column_name"] not in targets and kind in {"integer", "continuous"}:
                number = int if kind == "integer" else float
                values = [number(value) for value in references if value != ""]
                if values and min(values) < max(values):
                    sample = rng.randint if kind == "integer" else rng.uniform
                    samplers[position] = (sample, min(values), max(values))

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        mapping_path, sample_path = output / "feature-mapping.csv", output / "sample.csv"
        with mapping_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["feature", "group", "type"])
            writer.writeheader()
            writer.writerows(mapping)
        # Reuse the application's name, type, reserved-field and group-size checks.
        mapping = read_mapping(mapping_path, record_id, event_time)
        show_progress(f"Generating {rows:,} rows x {len(features) + 2:,} columns...")
        progress_interval = max(1, (rows + 9) // 10)
        with sample_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([record_id, event_time, *(row["column_name"] for row in features)])
            for index in range(rows):
                values = ([rng.choice(reference) for reference in examples] if randomize_numeric else
                          [reference[index % 5] for reference in examples])
                for position, (sample, lower, upper) in samplers.items():
                    if values[position] != "":
                        values[position] = (0 if numeric_zero_fraction and rng.random() < numeric_zero_fraction
                                            else sample(lower, upper))
                writer.writerow([f"sample-{index + 1:06d}", asof, *values])
                completed = index + 1
                if completed % progress_interval == 0 or completed == rows:
                    show_progress(f"Generated {completed:,}/{rows:,} rows ({completed / rows:.0%})")
        details = {
            "generation": (
                "Seeded uniform numeric values within reference min/max, with optional injected zeros; "
                "reference values sampled independently for each column and row."
                if randomize_numeric else
                "Row i takes ref_val((i-1) modulo 5)+1; synthetic IDs and a fixed event time."
            ),
            "randomize_numeric": randomize_numeric,
            "seed": seed,
            "numeric_zero_fraction": numeric_zero_fraction,
            "numeric_missing_values": list(missing_values),
            "target_columns": sorted(targets),
            "metadata_sha256": hashlib.sha256(Path(metadata_csv).read_bytes()).hexdigest(),
            "source_key_columns": [
                {key: value for key, value in row.items() if not key.startswith("ref_val")}
                for row in metadata if row["column_name"].lower() in key_names
            ],
            "columns": [{**{key: value for key, value in row.items() if not key.startswith("ref_val")},
                         "group": mapped["group"], "storage_type": mapped["type"],
                         "sampling": "uniform" if position in samplers else (
                             "independent_reference" if randomize_numeric else "repeated_reference"),
                         "protected_target": row["column_name"] in targets,
                         "reference_min": samplers[position][1] if position in samplers else None,
                         "reference_max": samplers[position][2] if position in samplers else None}
                        for position, (row, mapped) in enumerate(zip(features, mapping))],
        }
        source_path = output / "source-metadata.json"
        source_path.write_text(json.dumps(details, indent=2) + "\n")
        return {"sample": sample_path, "mapping": mapping_path, "metadata": source_path,
                "rows": rows, "features": len(mapping), "tables": len(counts)}


    # Choose the input metadata, then generate the three pipeline datasets.
    assert data_source in {"csv_direct", "feature_store"}, "Choose csv_direct or feature_store."
    RUN = ROOT / ".runs/feature-store" / namespace / uuid4().hex
    if demo_10000_features:
        assert demo_model_id in {"usecase1", "usecase2"}, "Choose one of the included demo models."
        parameters = json.loads((ROOT / "mlops-models/src/mlops_models/models" / demo_model_id / "parameters.json").read_text())
        assert "features" not in parameters, "Remove the fixed features list to exercise all 10,000 demo predictors."
        assert parameters["primary_key"] == record_id, (
            f"Demo model {demo_model_id} expects primary_key={parameters['primary_key']!r}; "
            f"Sample generation uses {record_id!r} from the model parameter files."
        )
        event_time = parameters["timestamp"]
        RUN.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = RUN.with_name(RUN.name + "-demo-metadata.csv")
        rng = Random(split_seed)
        with metadata_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["column_name", "column_map", "table_nm", *[f"ref_val{i}" for i in range(1, 6)]])
            for index in range(1, 10001):
                name = f"demo_feature_{index:05d}"
                writer.writerow([name, name, "synthetic_width_demo", *[round(rng.uniform(-10, 10), 6) for _ in range(5)]])
            target = parameters["target_key"]
            labels = [0, 1, 0, 1, 1] if parameters["task_type"] == "classification" else [10, 30, 20, 50, 40]
            writer.writerow([target, target, "synthetic_width_demo", *labels])
        print("Explicit synthetic demo: 10,000 predictors + target + record ID + event time.")
        print("ModelId:", demo_model_id, "| Target:", target)
    else:
        assert metadata_csv.strip(), f"STOP: set metadata_csv in {CONFIG_PATH}. No example file is selected automatically."
        metadata_path = Path(metadata_csv).expanduser()
        if not metadata_path.is_absolute():
            metadata_path = ROOT / metadata_path
    metadata_path = metadata_path.resolve()
    assert metadata_path.is_file(), f"STOP: metadata CSV does not exist: {metadata_path}"
    metadata_csv = str(metadata_path)
    assert isinstance(sample_rows, int) and sample_rows >= 2, "Use at least two labelled sample rows."
    assert 0 < test_fraction < 1, "test_fraction must be between 0 and 1."
    assert isinstance(inference_rows, int) and inference_rows > 0, "Use at least one inference row."
    target_names = {
        json.loads(path.read_text())["target_key"]
        for path in (ROOT / "mlops-models/src/mlops_models/models").glob("*/parameters.json")
    }
    sample = generate_sample(
        metadata_csv, RUN, rows=sample_rows + inference_rows, record_id=record_id, event_time=event_time,
        asof=sample_asof, type_overrides=type_overrides, deduplicate=True,
        randomize_numeric=randomize_numeric, seed=split_seed, target_columns=target_names,
        numeric_zero_fraction=numeric_zero_fraction, missing_values=missing_values,
    )
    show_progress("Checking feature mapping...")
    mapping = read_mapping(sample["mapping"], record_id, event_time)
    groups = {
        key: {"name": group_name(namespace, key), "columns": columns}
        for key, columns in group_columns(mapping, record_id, event_time).items()
    }
    show_progress("Validating generated rows and feature values...")
    validate_delivery(sample["sample"], {"record_id": record_id, "event_time": event_time, "mapping": mapping})
    show_progress("Loading the generated CSV into pandas...")
    df = pd.read_csv(sample["sample"], dtype=str, keep_default_na=False)
    show_progress("Splitting training, testing and inference rows...")
    # Split before fitting anything. Generated IDs are unique across the whole pool.
    labelled = df.iloc[:sample_rows]
    test_count = min(sample_rows - 1, max(1, round(sample_rows * test_fraction)))
    testing = labelled.sample(n=test_count, random_state=split_seed)
    training = labelled.drop(testing.index)
    inference = df.iloc[sample_rows:].copy()
    inference = inference.drop(columns=list(target_names), errors="ignore")
    inference[event_time] = (pd.Timestamp(sample_asof) + pd.DateOffset(months=1)).isoformat().replace("+00:00", "Z")
    sample_files = {"training": RUN / "train.csv", "testing": RUN / "test.csv",
                    "inference": RUN / "inference.csv"}
    for kind, frame in (("training", training), ("testing", testing), ("inference", inference)):
        show_progress(f"Writing {sample_files[kind].name}: {len(frame):,} rows x {len(frame.columns):,} columns...")
        frame.to_csv(sample_files[kind], index=False)
    show_progress("Checking distinct values in training, testing and inference...")
    source_details = json.loads(sample["metadata"].read_text())
    summary = pd.DataFrame(source_details["columns"])[
        ["column_name", "storage_type", "sampling", "protected_target", "reference_min", "reference_max"]
    ].set_index("column_name")
    missing_cells = {}
    for kind, frame in (("training", training), ("testing", testing), ("inference", inference)):
        counts, missing = {}, {}
        for name in summary.index:
            if name not in frame:
                continue  # Inference intentionally omits target columns.
            present = frame[name] != ""
            counts[name] = frame.loc[present, name].nunique()
            missing[name] = int((~present).sum())
        summary[kind + "_unique"] = pd.Series(counts)
        missing_cells[kind] = {name: count for name, count in missing.items() if count}
    summary_path = RUN / "generation-summary.csv"
    summary.to_csv(summary_path)
    def file_sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    preparation_report = {
        "files": {kind: str(path.resolve()) for kind, path in sample_files.items()},
        "sha256": {kind: file_sha256(path) for kind, path in sample_files.items()},
        "rows": {"training": len(training), "testing": len(testing), "inference": len(inference)},
        "columns": {"training": len(training.columns), "testing": len(testing.columns),
                    "inference": len(inference.columns)},
        "missing_cells": {kind: sum(counts.values()) for kind, counts in missing_cells.items()},
        "missing_by_column": missing_cells,
        "preview": training.iloc[:5, :6].to_dict(orient="records"),
        "summary_path": str(summary_path.resolve()),
        "source_pool": str(sample["sample"].resolve()),
        "metadata_sha256": source_details["metadata_sha256"],
        "seed": split_seed,
        "settings": {
            "sample_rows": sample_rows, "inference_rows": inference_rows,
            "test_fraction": test_fraction, "randomize_numeric": randomize_numeric,
            "numeric_zero_fraction": numeric_zero_fraction, "missing_values": list(missing_values),
            "record_id": record_id, "event_time": event_time, "sample_asof": sample_asof,
            "type_overrides": type_overrides, "target_columns": sorted(target_names),
            "demo_10000_features": demo_10000_features, "demo_model_id": demo_model_id,
        },
        "environment": {"python": sys.version.split()[0], "pandas": pd.__version__,
                        "numpy": __import__("numpy").__version__},
        "elapsed_seconds": round(perf_counter() - preparation_started, 3),
        "report_path": str((RUN / "preparation-report.json").resolve()),
    }
    Path(preparation_report["report_path"]).write_text(json.dumps(preparation_report, indent=2) + "\n")
    show_progress("Data preparation complete.")
    print(f"Seed: {split_seed} | Processing time: {preparation_report['elapsed_seconds']:.1f}s")
    print("Rows: " + " | ".join(f"{kind}: {count:,}" for kind, count in preparation_report["rows"].items()))
    print("Missing cells: " + " | ".join(f"{kind}: {count:,}" for kind, count in preparation_report["missing_cells"].items()))
    if any(preparation_report["missing_cells"].values()):
        print("Source missing values were preserved; <missing> marks a blank CSV cell. Column counts are in the report.")
    print("\nTraining sample — first 5 rows, first 6 columns:")
    print(training.iloc[:5, :6].replace("", "<missing>").to_string(index=False, max_colwidth=24))
    print("\nDetails:", preparation_report["report_path"])

    if data_source == "feature_store":
        from mlops_features.data import write_once
        config_key = settings["prefix"] + "configurations/" + uuid4().hex + "/feature-store.json"
        write_once(session.client("s3"), bucket, config_key, settings, kms_key=kms_key)
        feature_store_config_uri = f"s3://{bucket}/{config_key}"
        print("Completed training delivery ID:", receipt["delivery_id"])
        print("Leave FeatureStoreDeliveryId blank for the latest completed delivery, or set this ID to replay it.")
    else:
        feature_store_config_uri = ""

    # Reload before saving so unrelated configuration edits are preserved.
    saved_config = json.loads(CONFIG_PATH.read_text())
    for _key in ("csv_input", "test_csv_input", "inference_csv_input", "metadata_csv"):
        if saved_config.get(_key, "") != notebook_config.get(_key, ""):
            raise RuntimeError(f"{_key} changed during preparation; generated files were retained but config was not overwritten.")
    for key, kind in (("csv_input", "training"), ("test_csv_input", "testing"),
                      ("inference_csv_input", "inference")):
        saved_config[key] = sample_files[kind].resolve().relative_to(ROOT.resolve()).as_posix()
    saved_config["feature_store_config_uri"] = feature_store_config_uri
    _atomic_json(CONFIG_PATH, saved_config)
    return preparation_report
