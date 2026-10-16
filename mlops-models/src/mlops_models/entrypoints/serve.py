"""Serve Transform batches through the same file-based function as Processing."""

from copy import deepcopy
from importlib import import_module
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, Response, request
import pandas as pd
from werkzeug.exceptions import BadRequest


def execute(model_id, model_dir):
    """Load configuration once; give each request its own input/output directory.

    The model function owns model loading and prediction. This adapter does not
    construct a customer ModelPredictor or change the process working directory.
    """
    model_dir = Path(model_dir).resolve()
    config = json.loads((model_dir / "model_metadata.json").read_text())
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    parameters = json.loads(parameter_path.read_text())
    primary_key = parameters["primary_key"]
    predict_with_model = import_module(
        f"mlops_models.models.{model_id}.inference.predict"
    ).predict_with_model
    output_file = os.getenv("MLOPS_TRANSFORM_OUTPUT_FILE", "messaging_final.csv")
    selected = Path(output_file)
    if selected.is_absolute() or ".." in selected.parts or selected.suffix.lower() != ".csv":
        raise ValueError("TransformOutputFile must be a relative CSV filename or path.")

    def predict_batch(inputs, output):
        # Match the Processing call and isolate metadata mutations between requests.
        predict_with_model(inputs, model_dir, output, deepcopy(config), parameter_path)

    application = Flask(__name__)
    application.add_url_rule("/ping", view_func=ping, methods=["GET"])
    application.add_url_rule("/execution-parameters", view_func=execution_parameters, methods=["GET"])

    @application.post("/invocations")
    def invoke():
        if request.mimetype not in {"application/jsonlines", "application/jsonl",
                                    "application/x-ndjson"}:
            return Response("Expected application/jsonlines", status=415)
        return score_request(model_id, predict_batch, primary_key,
                             request.get_data(as_text=True), output_file)

    print(json.dumps({"event": "server_ready", "adapter": "file_prediction_v1",
                      "model": model_id, "model_dir": str(model_dir),
                      "result_file": output_file}), flush=True)
    return application


def ping():
    """Configuration and prediction imports are ready; model loading happens in the hook."""
    return Response(status=200)


def execution_parameters():
    """Use conservative request concurrency; the model function may load each batch."""
    return {"MaxConcurrentTransforms": int(os.getenv("MAX_CONCURRENT_TRANSFORMS", "1")),
            "BatchStrategy": "MULTI_RECORD", "MaxPayloadInMB": int(os.getenv("MAX_PAYLOAD_MB", "1"))}


def score_request(model_id, predict_batch, primary_key, payload, output_file="messaging_final.csv"):
    """Materialize one HTTP batch, call the model, and return its selected CSV rows."""
    try:
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except ValueError as error:
        raise BadRequest("Expected one JSON object per line.") from error
    if not rows or any(not isinstance(row, dict) or primary_key not in row for row in rows):
        raise BadRequest(f"Every input record must be an object containing {primary_key!r}.")
    df = pd.DataFrame(rows)
    if df[primary_key].isna().any():
        raise BadRequest("Input identifiers must not be null.")
    df[primary_key] = df[primary_key].astype(str)
    input_ids = df[primary_key].tolist()

    # Honors TMPDIR. Nothing is written under the registered model or source paths.
    with TemporaryDirectory(prefix="mlops-request-") as directory:
        work = Path(directory)
        inputs, output = work / "input", work / "output"
        inputs.mkdir()
        output.mkdir()
        df.to_csv(inputs / "data.csv", index=False)
        predict_batch(inputs, output)
        csv_files = sorted(path for path in output.rglob("*")
                           if path.is_file() and path.suffix.lower() == ".csv")
        selected = Path(output_file)
        matches = [path for path in csv_files if (
            path.relative_to(output) == selected if len(selected.parts) > 1
            else path.name.lower() == selected.name.lower())]
        if len(matches) != 1:
            raise ValueError(f"TransformOutputFile {output_file!r} matched {len(matches)} CSVs; "
                             f"available: {[p.relative_to(output).as_posix() for p in csv_files]}")
        result_path = matches[0]
        if not result_path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Transform output must stay inside the request output directory.")
        result = pd.read_csv(result_path, dtype={primary_key: str},
                             keep_default_na=False, na_values=["", r"\N"])
        if primary_key not in result or result[primary_key].isna().any():
            raise ValueError(f"Transform output must retain {primary_key!r}.")
        result_ids = result[primary_key].tolist()
        if len(result) != len(df):
            raise ValueError("Transform output must contain one prediction per input record.")
        if result_ids != input_ids:
            if len(set(input_ids)) != len(input_ids) or set(result_ids) != set(input_ids):
                raise ValueError("Transform output identifiers must match the input records.")
            # Customer code may sort by score. Transform responses follow input order.
            result = result.set_index(primary_key).loc[input_ids].reset_index()
        body = result.to_json(orient="records", lines=True, double_precision=15)
        print(json.dumps({"event": "success", "step": "Predict", "model": model_id,
                          "rows": len(result), "result_file": result_path.relative_to(output).as_posix(),
                          "csv_files": [p.relative_to(output).as_posix() for p in csv_files]}), flush=True)
        return Response(body, mimetype="application/jsonlines")
