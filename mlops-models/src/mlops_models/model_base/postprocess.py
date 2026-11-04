"""Read scoring inputs and prediction files, run analysis, and publish the result."""

import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

import pandas as pd


def prediction_test(train_data_path, test_data_path, selection_file, predictions,
                    output_directory):
    """Read test inputs, run the model's analysis and publish its output files."""
    train_data_path = Path(train_data_path)
    if not train_data_path.is_file():
        raise FileNotFoundError(f"Raw training CSV not found: {train_data_path}")
    print(f"Training analysis data: {train_data_path}", flush=True)
    # The bootstrap supplies the saved model before importing model code.
    # These are internal dependencies of this example's reporting, not hook arguments.
    model_directory = Path(os.environ["MLOPS_MODEL_DIR"])
    parameter_path = Path(__file__).parents[1] / "models" / os.environ["MODEL_ID"] / "parameters.json"
    data_path, selection_path, predictions, output, parameters = get_container_input(
        test_data_path, selection_file, predictions, output_directory, model_directory)
    analyze_predictions(data_path, selection_path, predictions, output, parameters)
    return write_test_output(
        output, data_path=data_path, selection_file=selection_path, predictions=predictions,
        parameters=parameters, model_directory=model_directory, parameter_path=parameter_path)


def prediction_inference(data_directory, selection_file, prediction_directory, output_directory,
                            model_directory, *, chunk_rows=10000):
    """Load this step's inputs and run inference analysis; there is no mode selection."""
    with TemporaryDirectory(prefix="postprocess-inputs-") as converted_directory:
        data_path, selection_path, predictions, output, parameters = get_container_input(
            Path(data_directory) / "data.csv", selection_file, prediction_directory,
            output_directory, model_directory,
            converted_directory=converted_directory, chunk_rows=chunk_rows)
        analyze_predictions(data_path, selection_path, predictions, output, parameters)
        return write_inference_output(
            output, data_path=data_path, selection_file=selection_path,
            predictions=predictions, parameters=parameters)


def get_container_input(data_path, selection_file, prediction_directory,
                        output_directory, model_directory, *, converted_directory=None, chunk_rows=10000):
    """Resolve CSV paths and the destination; read the saved model's parameters."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the original GetData snapshot: {data_path}")
    selection_path = Path(selection_file)
    if not selection_path.is_file():
        raise FileNotFoundError(f"Postprocess needs the saved selected-feature CSV: {selection_path}")
    parameters = json.loads((Path(model_directory) / "parameters.json").read_text())
    source = Path(prediction_directory)
    prediction_format = os.getenv("MLOPS_PREDICTION_FORMAT", "csv")
    if prediction_format == "jsonlines":
        if converted_directory is None:
            raise ValueError("Native Transform inputs need a temporary CSV directory.")
        convert_transform_predictions(source, converted_directory, parameters["primary_key"], chunk_rows)
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
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Analysis data: {data_path}", flush=True)
    print(f"Selected features: {selection_path}", flush=True)
    print(f"Parameters: {model_directory} / parameters.json", flush=True)
    print(f"Project: {parameters.get('project')} | Test YMID: {parameters.get('test_ymid')}", flush=True)
    print(f"Analysis output directory: {output}", flush=True)
    return data_path, selection_path, predictions, output, parameters


def read_predictions(source, chunk_rows):
    """Read native Transform outputs in bounded chunks with their string IDs."""
    from itertools import islice
    for path in sorted(Path(source).rglob("*.out")):
        with path.open() as stream:
            while lines := list(islice(stream, chunk_rows)):
                records = [json.loads(line) for line in lines if line.strip()]
                if records:
                    yield pd.DataFrame(records)


def convert_transform_predictions(source, destination, primary_key, chunk_rows):
    """Adapt native Batch Transform JSON-lines output before calling CSV analysis."""
    frames = list(read_predictions(source, chunk_rows))
    if not frames:
        raise ValueError("No Transform predictions were found.")
    result = pd.concat(frames, ignore_index=True)
    destination = Path(destination)
    result.to_csv(destination / "predictions.csv", index=False)
    result.to_csv(destination / "messaging_final.csv", index=False)
    result[[primary_key]].to_csv(destination / "messaging.csv", index=False)


def write_test_output(output, *, data_path, selection_file, predictions, parameters,
                           model_directory, parameter_path=None):
    """Publish analysis, quality-gate results and the model files required by Registry."""
    output = Path(output)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Prediction analysis wrote no files under: {output}")
    for path in files:
        print(f"Analysis output: {path.relative_to(output).as_posix()} -> {path}", flush=True)
    # Metrics come only from this test analysis, never from the training report.
    report_path = output / "evaluation.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    policy = json.loads(Path(parameter_path).read_text()) if parameter_path else parameters
    model_id = Path(parameter_path).parent.name if parameter_path else parameters["model_id"]
    report.setdefault("model_id", model_id)
    report.setdefault("task", policy["task_type"])
    report.setdefault("metrics", {})
    report.setdefault("data_usage", "prediction_test")
    report["evaluation_mode"] = "prediction_test"
    if report["model_id"] != model_id or report["task"] != policy["task_type"]:
        raise ValueError("Test report model_id/task differs from the selected model.")
    if not isinstance(report["metrics"], dict) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in report["metrics"].values()
    ):
        raise ValueError("Test metrics must be a dictionary of finite numbers.")
    if report["metrics"] and not report.get("data_usage"):
        raise ValueError("Test metrics must identify their data_usage.")
    checks = []
    for metric, limits in policy.get("quality_gate", {}).items():
        if not limits or set(limits) - {"min", "max"}:
            raise ValueError(f"Set min and/or max for quality_gate.{metric}.")
        value = report["metrics"].get(metric)
        passed = value is not None and ("min" not in limits or value >= limits["min"]) and (
            "max" not in limits or value <= limits["max"])
        checks.append({"metric": metric, "value": value, "limits": limits, "passed": passed})
    report["quality"] = {"configured": bool(checks), "passed": all(c["passed"] for c in checks),
                         "checks": checks}

    # Copy original model files without reformatting; Registry verifies the metadata hash.
    metadata = read_model_file(model_directory, "model_metadata.json")
    json.loads(metadata)
    (output / "model_metadata.json").write_bytes(metadata)
    for source, destination in (("parameters.json", "parameters.json"),
                                ("_execution.json", "training_execution.json")):
        content = read_model_file(model_directory, source, optional=True)
        if content is not None:
            (output / destination).write_bytes(content)
    report["effective_parameters_available"] = (output / "parameters.json").is_file()
    identity = json.loads(read_model_file(model_directory, "code/source.json", optional=True) or b"{}")
    report.update(
        checkpoint_uri=os.getenv("MLOPS_MODEL_ARTIFACT_URI", os.getenv("CHECKPOINT_URI", str(model_directory))),
        model_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        model_source_commit=identity.get("commit", "unknown"),
        model_source_dirty=str(identity.get("dirty", True)).lower(),
        model_source_uri=identity.get("uri", ""), model_source_digest=identity.get("sha256", ""),
        evaluation_source_commit=os.getenv("SOURCE_COMMIT", "unknown"))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # This records artifact locations for the execution, not customer analysis metrics.
    from mlops_models.pipelines.inspect import write_inference_report
    write_inference_report(output, data_path=data_path, selection_file=selection_file,
                           predictions=predictions, config=parameters)
    return output


def write_inference_output(output, *, data_path, selection_file, predictions, parameters):
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


def read_model_file(model_directory, name, *, optional=False):
    """Read one packaged file, supporting the extracted directory or a local archive."""
    model = Path(model_directory)
    try:
        if model.is_file():
            with tarfile.open(model) as archive:
                return archive.extractfile(name).read()
        return (model / name).read_bytes()
    except (FileNotFoundError, KeyError):
        if not optional:
            raise
        return None


def analyze_predictions(data_path, selection_file, predictions, output_directory, parameters):
    """Replace this example with customer test analysis using explicit input and output paths.

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
    if not matches:
        matches = [name for name in predictions if Path(name).name.lower() == "predictions.csv"]
    if len(matches) != 1:
        raise ValueError("Example analysis needs exactly one predictions.csv (or legacy messaging_final.csv); "
                         f"found {len(matches)}. Available CSVs: {list(predictions)}. "
                         "Adapt prediction_test() to select the files required by your analysis.")
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
