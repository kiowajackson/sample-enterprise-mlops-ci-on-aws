"""Read scoring data, apply the saved feature list, and write prepared shards."""

from pathlib import Path

import pandas as pd


def process_inference(data_directory, output_directory, config, *, shard_rows=10000):
    """Replace the projection with the model's process_inference; never fit a selector."""
    dfs = get_container_input(data_directory, config, shard_rows)
    mapping = config["model_metadata"]["feature_mapping"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("model_metadata.json needs a nonempty feature_mapping.")
    columns = list(dict.fromkeys([config["primary_key"], *mapping]))
    # Customer column normalization belongs here, before selecting columns.
    prepared = (df[columns] for df in dfs)
    return write_container_output(prepared, output_directory, mapping)


def get_container_input(data_directory, config, shard_rows):
    """Read the snapshot CSV in bounded dataframes without changing identifiers."""
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive.")
    with pd.read_csv(Path(data_directory) / "data.csv", dtype={config["primary_key"]: str},
                     keep_default_na=False, na_values=["", r"\N"],
                     chunksize=shard_rows) as reader:
        for df in reader:
            yield df


def write_container_output(dfs, output_directory, feature_mapping):
    """Save prepared shards and the exact feature list for downstream analysis."""
    output = Path(output_directory) / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty batch shard output directory.")
    shards = output / "shards"
    shards.mkdir()
    rows = 0
    for number, df in enumerate(dfs):
        df.to_csv(output / "data.csv", mode="a", header=rows == 0, index=False)
        df.to_json(shards / f"part-{number:06d}.jsonl",
                   orient="records", lines=True, double_precision=15)
        rows += len(df)
    if rows == 0:
        raise ValueError("Batch input contains no rows.")
    selection = Path(output_directory) / "selection"
    selection.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"features": list(feature_mapping),
                  "description": list(feature_mapping.values())}).to_csv(
                      selection / "selected_features.csv", index=False)
    return output, rows
