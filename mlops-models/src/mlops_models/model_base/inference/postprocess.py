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
    """Read every prediction CSV, keyed by its path relative to the prediction root."""
    data_path = Path(data_directory) / "data.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the original GetData snapshot: {data_path}")
    selected = pd.read_csv(selection_file, keep_default_na=False)
    source = Path(prediction_directory)
    if prediction_format == "csv":
        files = sorted(path for path in source.rglob("*")
                       if path.is_file() and path.suffix.lower() == ".csv")
        if not files:
            raise FileNotFoundError(f"Postprocess found no prediction CSV files under: {source}")
        predictions = {}
        for path in files:
            name = path.relative_to(source).as_posix()
            print(json.dumps({"event": "prediction_input_csv", "path": str(path),
                              "relative_path": name}), flush=True)
            predictions[name] = pd.read_csv(path, dtype={config["primary_key"]: str},
                                           keep_default_na=False, na_values=["", r"\N"])
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
    # Only this example analysis chooses a particular input; the reader accepts all CSVs.
    matches = [name for name in predictions if Path(name).name.lower() == "messaging_final.csv"]
    if len(matches) != 1:
        raise ValueError("Example analysis needs exactly one messaging_final.csv at any depth; "
                         f"found {len(matches)}. Available CSVs: {list(predictions)}. "
                         "Adapt postprocess() to select the files required by your analysis.")
    df = predictions[matches[0]].copy()
    task = config["task_type"]
    print(json.dumps({"event": "postprocess_input_columns", "file": matches[0],
                      "task": task, "columns": list(df.columns)}), flush=True)
    # Normalize score/result names only; keep identifiers and explanation columns.
    renamed = {}
    for expected in ("SCORE", "PROBABILITY", "PREDICTION"):
        columns = [column for column in df if str(column).upper() == expected]
        if len(columns) > 1:
            raise ValueError(f"Ambiguous {expected} columns: {columns}")
        if columns:
            renamed[columns[0]] = expected
    df = df.rename(columns=renamed)
    if task not in ("classification", "regression"):
        raise ValueError("task_type must be classification or regression.")
    if "PREDICTION" not in df:
        if "SCORE" not in df:
            raise ValueError(f"{task} output needs PREDICTION or SCORE; found {list(df.columns)}")
        if task == "classification":
            probability = pd.to_numeric(df["SCORE"], errors="raise")
            threshold = float(config.get("threshold", 0.5))
            if not probability.between(0, 1).all() or not 0 <= threshold <= 1:
                raise ValueError("Classification SCORE and threshold must be probabilities between 0 and 1.")
            df["PROBABILITY"] = probability
            df["PREDICTION"] = (probability >= threshold).astype(int)
            df = df.drop(columns="SCORE")
        else:
            # Legacy/demo regression calls its estimate SCORE. No inverse transform.
            df = df.rename(columns={"SCORE": "PREDICTION"})
    sample = {column: str(df[column].iloc[0]) for column in ("PROBABILITY", "PREDICTION")
              if column in df and not df.empty}
    print(json.dumps({"event": "postprocess_output_columns", "columns": list(df.columns),
                      "first_scores": sample}), flush=True)
    return df
