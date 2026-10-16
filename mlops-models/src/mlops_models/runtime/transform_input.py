"""Convert prepared CSV to the JSON-lines input expected by the Transform server."""

import json
from pathlib import Path

import pandas as pd


def prepare_transform_input(prepared_directory, parameter_path):
    """Format an existing prepared CSV only when native Batch Transform is selected."""
    parameters = json.loads(Path(parameter_path).read_text())
    prepared = Path(prepared_directory)
    df = pd.read_csv(prepared / "data.csv", dtype={parameters["primary_key"]: str},
                     keep_default_na=False, na_values=["", r"\N"])
    output = prepared / "shards"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty Transform input directory.")
    df.to_json(output / "part-000000.jsonl", orient="records", lines=True, double_precision=15)
