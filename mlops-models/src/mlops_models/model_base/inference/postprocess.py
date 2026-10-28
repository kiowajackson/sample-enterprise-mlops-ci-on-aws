"""Read scoring inputs and prediction files, run analysis, and publish the result."""

import json
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

import pandas as pd


# Analysis helpers can read this without receiving parameters in every call.
# One job runs one model; get_container_input refreshes the same dictionary per call.
parameters = {}


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory,
                            model_directory, *, analysis_mode="prediction_inference", chunk_rows=10000,
                            inference_fn=None, test_fn=None):
    """Read inputs, initialize parameters, run the chosen analysis, and write outputs."""
    if analysis_mode == "prediction_inference":
        analyze = prediction_inference if inference_fn is None else inference_fn
    elif analysis_mode == "prediction_test":
        analyze = prediction_test if test_fn is None else test_fn
    else:
        raise ValueError("analysis_mode must be prediction_inference or prediction_test.")
    with TemporaryDirectory(prefix="postprocess-inputs-") as converted_directory:
        data_path, selection_path, predictions, output, _ = get_container_input(
            data_directory, selection_file, prediction_directory, output_directory, model_directory,
            converted_directory=converted_directory, chunk_rows=chunk_rows)
        analyze(data_path, selection_path, predictions, output)
        return write_container_output(
            output, data_path=data_path, selection_file=selection_path,
            predictions=predictions, parameters=parameters)


def get_container_input(data_directory, selection_file, prediction_directory,
                        output_directory, model_directory, *, converted_directory=None, chunk_rows=10000):
    """Resolve CSV paths and the destination; read the saved model's parameters."""
    data_path = Path(data_directory) / "data.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the original GetData snapshot: {data_path}")
    selection_path = Path(selection_file)
    if not selection_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the saved selected-feature CSV: {selection_path}")
    source = Path(prediction_directory)
    prediction_format = os.getenv("MLOPS_PREDICTION_FORMAT", "csv")
    if prediction_format == "jsonlines":
        if converted_directory is None:
            raise ValueError("Native Transform inputs need a temporary CSV directory.")
        convert_transform_predictions(source, converted_directory, model_directory, chunk_rows)
        source = Path(converted_directory)
    elif prediction_format != "csv":
        raise ValueError("MLOPS_PREDICTION_FORMAT must be csv or jsonlines.")
    files = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"Postprocess found no prediction CSV files under: {source}")
    predictions = {}
    for path in files:
        name = path.relative_to(source).as_posix()
        print(f"Prediction input: {name} -> {path}", flush=True)
        predictions[name] = path
    # Update in place so use-case modules importing parameters see the same values.
    parameters.clear()
    parameters.update(load_model_parameters(model_directory))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Analysis data: {data_path}", flush=True)
    print(f"Selected features: {selection_path}", flush=True)
    print(f"Parameters: {model_directory} / parameters.json", flush=True)
    print(f"Project: {parameters.get('project')} | Test YMID: {parameters.get('test_ymid')}", flush=True)
    print(f"Analysis output directory: {output}", flush=True)
    return data_path, selection_path, predictions, output, parameters


def load_model_parameters(model_directory):
    """Read parameters.json and model_metadata.json from the selected checkpoint."""
    model = Path(model_directory)
    if model.is_file():
        with tarfile.open(model) as archive:
            parameters = json.load(archive.extractfile("parameters.json"))
            parameters["model_metadata"] = json.load(archive.extractfile("model_metadata.json"))
    else:
        parameters = json.loads((model / "parameters.json").read_text())
        parameters["model_metadata"] = json.loads((model / "model_metadata.json").read_text())
    return parameters


def read_predictions(source, chunk_rows):
    """Read native Transform outputs in bounded chunks with their string IDs."""
    from itertools import islice
    for path in sorted(Path(source).rglob("*.out")):
        with path.open() as stream:
            while lines := list(islice(stream, chunk_rows)):
                records = [json.loads(line) for line in lines if line.strip()]
                if records:
                    yield pd.DataFrame(records)


def convert_transform_predictions(source, destination, model_directory, chunk_rows):
    """Adapt native Batch Transform JSON-lines output before calling CSV analysis."""
    frames = list(read_predictions(source, chunk_rows))
    if not frames:
        raise ValueError("No Transform predictions were found.")
    result = pd.concat(frames, ignore_index=True)
    primary_key = load_model_parameters(model_directory)["primary_key"]
    destination = Path(destination)
    result.to_csv(destination / "predictions.csv", index=False)
    result.to_csv(destination / "messaging_final.csv", index=False)
    result[[primary_key]].to_csv(destination / "messaging.csv", index=False)


def write_container_output(output, *, data_path, selection_file, predictions, parameters):
    """Check analysis files and write run.json; the loader uploads the files unchanged."""
    output = Path(output)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Prediction analysis wrote no files under: {output}")
    for path in files:
        print(f"Analysis output: {path.relative_to(output).as_posix()} -> {path}", flush=True)
    # This records artifact locations for the execution, not customer analysis metrics.
    from mlops_models.pipelines.inspect import write_inference_report
    write_inference_report(output, data_path=data_path, selection_file=selection_file,
                           predictions=predictions, config=parameters)
    return output


def prediction_inference(data_path, selection_file, predictions, output_directory):
    """Replace this demo with prediction analysis using explicit input and output paths.

    predictions maps relative CSV names to Paths, for example:
    {"output/predictions.csv": Path(...), "output/messaging_final.csv": Path(...)}.
    Pass those paths to the customer's analysis instead of reconstructing old
    notebook folders. Set its destination to output_directory and retain its
    plot/report writes there. Project and test YMID can be read from parameters
    using the customer's actual keys (the example names are project and test_ymid).
    """
    config = parameters
    # Only this example analysis chooses a particular input; the reader accepts all CSVs.
    matches = [name for name in predictions if Path(name).name.lower() == "messaging_final.csv"]
    if len(matches) != 1:
        raise ValueError("Example analysis needs exactly one messaging_final.csv at any depth; "
                         f"found {len(matches)}. Available CSVs: {list(predictions)}. "
                         "Adapt prediction_inference() to select the files required by your analysis.")
    df = pd.read_csv(predictions[matches[0]], dtype={config["primary_key"]: str},
                     keep_default_na=False, na_values=["", r"\N"])
    task = config["task_type"]
    print(f"Analysis input {matches[0]} ({task}): {list(df.columns)}", flush=True)
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
    print(f"Analysis result columns: {list(df.columns)} | First scores: {sample}", flush=True)
    if df.empty:
        raise ValueError("Prediction analysis returned no rows.")
    # These are demo outputs, not mandatory filenames for a customer analysis.
    output = Path(output_directory)
    path = output / "prediction_analysis.csv"
    df.to_csv(path, index=False)
    (output / "summary.json").write_text(json.dumps({
        "rows": len(df), "task": task, "file": path.name}) + "\n")


def prediction_test(data_path, selection_file, predictions, output_directory):
    """Placeholder for customer test analysis; the demo writes the same example files."""
    prediction_inference(data_path, selection_file, predictions, output_directory)
