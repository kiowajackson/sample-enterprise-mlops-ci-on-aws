"""Read scoring inputs and prediction files, run analysis, and publish the result."""

import json
from pathlib import Path

import pandas as pd


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory,
                            config, *, format_predictions, chunk_rows=10000, prediction_format="csv"):
    """Replace the analysis; its inputs come from this inference execution's producers."""
    data_path, selected_features, predictions = get_container_input(
        data_directory, selection_file, prediction_directory, config,
        prediction_format=prediction_format)
    result = format_predictions(data_path, selected_features, predictions, config)
    return write_container_output(result, output_directory, config)


def get_container_input(data_directory, selection_file, prediction_directory, config, *, prediction_format="csv"):
    """Return the raw CSV path, selected-feature dataframe and named prediction dataframes."""
    data_path = Path(data_directory) / "data.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the original GetData snapshot: {data_path}")
    selected = pd.read_csv(selection_file, keep_default_na=False)
    source = Path(prediction_directory)
    if prediction_format == "csv":
        predictions = {
            name: pd.read_csv(source / name, dtype={config["primary_key"]: str},
                              keep_default_na=False, na_values=["", r"\N"])
            for name in ("predictions.csv", "messaging.csv", "messaging_final.csv")}
    elif prediction_format == "jsonlines":
        # Native Transform returns line outputs, not the Processing model's three CSVs.
        frames = list(read_predictions(source, 10000))
        if not frames:
            raise ValueError("No Transform predictions were found.")
        result = pd.concat(frames, ignore_index=True)
        predictions = {"predictions.csv": result, "messaging_final.csv": result,
                       "messaging.csv": result[[config["primary_key"]]].copy()}
    else:
        raise ValueError("prediction_format must be csv or jsonlines.")
    return data_path, selected, predictions


def read_predictions(source, chunk_rows):
    """Read native Transform outputs in bounded chunks with their string IDs."""
    from itertools import islice
    for path in sorted(Path(source).rglob("*.out")):
        with path.open() as stream:
            while lines := list(islice(stream, chunk_rows)):
                records = [json.loads(line) for line in lines if line.strip()]
                if records:
                    yield pd.DataFrame(records)


def write_container_output(result, output, config):
    """Publish the final analysis/leads file; the loader uploads this output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if result.empty:
        raise ValueError("Prediction analysis returned no rows.")
    path = output / "prediction_analysis.csv"
    result.to_csv(path, index=False)
    (output / "summary.json").write_text(json.dumps({
        "rows": len(result), "task": config["task_type"], "file": path.name}) + "\n")
    return path


def postprocess(data_path, selected_features, predictions, config) -> pd.DataFrame:
    """Replace with customer analysis using the raw CSV, selection and prediction files."""
    df = predictions["messaging_final.csv"].copy()
    if "prediction" not in df:
        if config["task_type"] == "classification":
            df["probability"] = df["score"]
            df["prediction"] = (df["score"] >= config.get("threshold", 0.5)).astype(int)
            df = df.drop(columns="score")
        else:
            df = df.rename(columns={"score": "prediction"})
    return df
