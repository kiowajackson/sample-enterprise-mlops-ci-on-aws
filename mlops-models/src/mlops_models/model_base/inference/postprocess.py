"""Read scoring inputs and prediction files, run analysis, and publish the result."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory,
                            config, *, run_analysis, chunk_rows=10000, prediction_format="csv"):
    """Resolve input paths, let analysis write its files, then check the output directory."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="postprocess-inputs-") as temporary:
        source = Path(prediction_directory)
        if prediction_format == "jsonlines":
            # Preserve the existing Transform adapter. Only its selected response
            # file survives serving; these are temporary compatibility CSVs,
            # not the full set of files produced by Processing prediction.
            frames = list(read_predictions(source, chunk_rows))
            if not frames:
                raise ValueError("No Transform predictions were found.")
            result = pd.concat(frames, ignore_index=True)
            source = Path(temporary)
            result.to_csv(source / "predictions.csv", index=False)
            result.to_csv(source / "messaging_final.csv", index=False)
            result[[config["primary_key"]]].to_csv(source / "messaging.csv", index=False)
        elif prediction_format != "csv":
            raise ValueError("prediction_format must be csv or jsonlines.")
        data_path, selection_path, predictions = get_container_input(
            data_directory, selection_file, source)
        # The use-case wrapper supplies run_analysis=postprocess directly.
        # Customer analysis reads these paths and writes under output; no return is required.
        run_analysis(data_path, selection_path, predictions, output, config)
        # Postprocess owns prediction analysis and its execution report.
        from mlops_models.runtime.execution import write_inference_report
        write_inference_report(output, data_path=data_path, selection_file=selection_path,
                               predictions=predictions, config=config)
    return write_container_output(output)


def get_container_input(data_directory, selection_file, prediction_directory):
    """Resolve files without loading them; prediction keys retain their relative subfolders."""
    data_path = Path(data_directory) / "data.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the original GetData snapshot: {data_path}")
    selection_path = Path(selection_file)
    if not selection_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the saved selected-feature CSV: {selection_path}")
    source = Path(prediction_directory)
    files = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"Postprocess found no prediction CSV files under: {source}")
    predictions = {}
    for path in files:
        name = path.relative_to(source).as_posix()
        print(json.dumps({"event": "prediction_input_csv", "path": str(path),
                          "relative_path": name}), flush=True)
        predictions[name] = path
    return data_path, selection_path, predictions


def read_predictions(source, chunk_rows):
    """Read native Transform outputs in bounded chunks with their string IDs."""
    from itertools import islice
    for path in sorted(Path(source).rglob("*.out")):
        with path.open() as stream:
            while lines := list(islice(stream, chunk_rows)):
                records = [json.loads(line) for line in lines if line.strip()]
                if records:
                    yield pd.DataFrame(records)


def write_container_output(output):
    """Check and log all analysis files; the loader uploads them without rewriting them."""
    output = Path(output)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Prediction analysis wrote no files under: {output}")
    for path in files:
        print(json.dumps({"event": "prediction_analysis_output", "path": str(path),
                          "relative_path": path.relative_to(output).as_posix()}), flush=True)
    return output


def postprocess(data_path, selection_file, predictions, output_directory, config):
    """Replace this demo with prediction analysis using explicit input and output paths.

    predictions maps relative CSV names to Paths, for example:
    {"output/predictions.csv": Path(...), "output/messaging_final.csv": Path(...)}.
    Pass those paths to the customer's analysis instead of reconstructing old
    notebook folders. Set its destination to output_directory and retain its
    plot/report writes there. Model-specific names/settings belong in config.
    """
    # Only this example analysis chooses a particular input; the reader accepts all CSVs.
    matches = [name for name in predictions if Path(name).name.lower() == "messaging_final.csv"]
    if len(matches) != 1:
        raise ValueError("Example analysis needs exactly one messaging_final.csv at any depth; "
                         f"found {len(matches)}. Available CSVs: {list(predictions)}. "
                         "Adapt postprocess() to select the files required by your analysis.")
    df = pd.read_csv(predictions[matches[0]], dtype={config["primary_key"]: str},
                     keep_default_na=False, na_values=["", r"\N"])
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
    if df.empty:
        raise ValueError("Prediction analysis returned no rows.")
    # These are demo outputs, not mandatory filenames for a customer analysis.
    output = Path(output_directory)
    path = output / "prediction_analysis.csv"
    df.to_csv(path, index=False)
    (output / "summary.json").write_text(json.dumps({
        "rows": len(df), "task": task, "file": path.name}) + "\n")
