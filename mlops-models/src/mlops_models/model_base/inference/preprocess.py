"""Shard scoring rows; model-specific transformations belong in the predictor."""

from pathlib import Path

import pandas as pd


def process_inference(data_directory, output_directory, config, *, shard_rows=10000):
    """Preserve input columns and write bounded scoring shards without fitting transforms."""
    dfs = get_container_input(data_directory, config, shard_rows)
    return write_container_output(dfs, output_directory)


def get_container_input(data_directory, config, shard_rows):
    """Read the snapshot CSV in bounded dataframes without changing identifiers."""
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive.")
    with pd.read_csv(Path(data_directory) / "data.csv", dtype={config["primary_key"]: str},
                     keep_default_na=False, na_values=["", r"\N"],
                     chunksize=shard_rows) as reader:
        for df in reader:
            if config["primary_key"] not in df:
                raise ValueError(f"Scoring CSV is missing primary_key {config['primary_key']!r}.")
            yield df


def write_container_output(dfs, output_directory):
    """Save local shards; the loader uploads them for the next pipeline step."""
    output = Path(output_directory) / "shards"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty batch shard output directory.")
    rows = 0
    for number, df in enumerate(dfs):
        df.to_json(output / f"part-{number:06d}.jsonl",
                   orient="records", lines=True, double_precision=15)
        rows += len(df)
    if rows == 0:
        raise ValueError("Batch input contains no rows.")
    return output, rows
