"""Publish shared training and approved-model inference pipelines from supplied source."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from mlops_skeleton.names import discover_models, validate_namespace
from mlops_skeleton.pipelines import load_definition, walk_steps, apply_job_settings
from mlops_models.storage import data_prefix, s3_location

APPLICATION = Path(__file__).resolve().parents[3]
PACKAGE = APPLICATION / "src/mlops_models"


def source_identity(mode, name, commit=""):
    """CI supplies its reviewed commit; direct deploy records the local checkout when available."""
    if mode not in {"direct-deploy", "ci-deploy"}:
        raise ValueError("Choose direct-deploy or ci-deploy.")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,63}", name):
        raise ValueError("Use a short release/development name containing letters, numbers and hyphens.")
    if mode == "ci-deploy":
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise ValueError("CI must supply the full reviewed commit hash.")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=APPLICATION, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                        cwd=APPLICATION, text=True).strip()
        if head != commit or dirty:
            raise ValueError("CI deploy requires a clean checkout of the supplied commit.")
        return {"mode": mode, "name": name, "commit": commit, "dirty": False}
    if shutil.which("git") is None:
        return {"mode": mode, "name": name, "commit": "uncommitted", "dirty": True}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=APPLICATION, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=APPLICATION, capture_output=True, text=True)
    return {"mode": mode, "name": name, "commit": result.stdout.strip() or "uncommitted",
            "dirty": bool(status.returncode or status.stdout)}


def source_location(s3_root, identity):
    """Return the planned S3 source prefix without reading or writing S3."""
    data_prefix({"prefix": s3_location(s3_root)[1]})
    suffix = f"source/{identity['mode']}/{identity['name']}"
    if identity["mode"] == "ci-deploy":
        suffix += "/" + identity["commit"]
    else:
        # A new set of files needs a fresh prefix: S3 uploads do not remove old names.
        names = [path.relative_to(PACKAGE).as_posix() for path in source_files()]
        layout = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
        suffix += "/files-" + layout
    return s3_root.rstrip("/") + "/" + suffix


def source_files():
    """Return only the runtime files included in source publication."""
    return [path for path in sorted(PACKAGE.rglob("*"))
            if path.is_file() and not path.is_symlink() and path.suffix != ".pyc"
            and not any(part.startswith(".") or part == "__pycache__"
                        for part in path.relative_to(PACKAGE).parts)]


def upload_source(session, s3_root, identity):
    """Upload the complete importable package and the tracked source loader to one prefix."""
    uri = source_location(s3_root, identity)
    bucket, prefix = s3_location(uri)
    s3 = session.client("s3")
    for path in source_files():
        relative = path.relative_to(PACKAGE)
        s3.upload_file(str(path), bucket, f"{prefix}/{PACKAGE.name}/{relative.as_posix()}")
    s3.upload_file(str(PACKAGE / "runtime/load_source.py"), bucket, prefix + "/load_source.py")
    s3.put_object(Bucket=bucket, Key=prefix + "/source.json",
                  Body=(json.dumps(identity, indent=2) + "\n").encode())
    return uri


def preparation_fingerprint(session, csv_uri, local_csv=None):
    """Hash each preparation step separately; native input references propagate upstream changes."""
    digest = hashlib.sha256()
    if local_csv is not None:
        with Path(local_csv).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    elif csv_uri:
        bucket, key = s3_location(csv_uri)
        info = session.client("s3").head_object(Bucket=bucket, Key=key)
        digest.update(json.dumps({
            "uri": csv_uri, "etag": info["ETag"], "version": info.get("VersionId"),
        }, sort_keys=True).encode())
    digests = {name: digest.copy() for name in ("GetData", "FeatureSelection", "Preprocess")}
    entrypoints = {"get_data.py": "GetData", "feature_selection.py": "FeatureSelection",
                   "preprocess_training.py": "Preprocess"}
    training = {"feature_selection.py": "FeatureSelection", "preprocess.py": "Preprocess"}
    for path in source_files():
        relative = path.relative_to(PACKAGE)
        # These modules do not run in preparation jobs. Unknown shared helpers do.
        if relative.parts[0] in {"pipelines", "registry"} or "inference" in relative.parts:
            continue
        steps = digests.keys()
        if relative.parent == Path("entrypoints"):
            if path.name in {"train.py", "evaluate.py", "predict.py", "postprocess.py",
                             "preprocess_inference.py", "serve.py"}:
                continue
            if path.name in entrypoints:
                steps = (entrypoints[path.name],)
        elif relative.parent.name == "training":
            if path.name in {"train.py", "evaluate.py", "definition.json"}:
                continue
            if path.name in training:
                steps = (training[path.name],)
        elif path.name in {"parameters.json", "model_metadata.json"}:
            # CSV-direct GetData copies bytes; it does not read model settings.
            steps = ("FeatureSelection", "Preprocess")
        content = b"\0" + relative.as_posix().encode() + b"\0" + path.read_bytes()
        for step in steps:
            digests[step].update(content)
    return {step: value.hexdigest() for step, value in digests.items()}


def pipeline_fingerprints(session, csv_uri="", local_csv=None):
    """Identify the source consumed by each expensive step, independently."""
    revisions = preparation_fingerprint(session, csv_uri, local_csv)
    digests = {name: hashlib.sha256() for name in
               ("Train", "Evaluate", "BatchPreprocess", "Score", "Postprocess")}
    entrypoints = {"train.py": ("Train",), "evaluate.py": ("Evaluate",),
                   "preprocess_inference.py": ("Train", "BatchPreprocess"),
                   "predict.py": ("Train", "Evaluate", "Score"),
                   "postprocess.py": ("Train", "Evaluate", "Postprocess"),
                   "serve.py": ("Train",)}
    inference = {"preprocess.py": ("Train", "BatchPreprocess"),
                 "predict.py": ("Train", "Evaluate", "Score"),
                 "postprocess.py": ("Train", "Evaluate", "Postprocess")}
    for path in source_files():
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] in {"pipelines", "registry"} or path.name == "definition.json":
            continue
        steps = digests.keys()
        if relative.parent == Path("entrypoints") and path.name in entrypoints:
            steps = entrypoints[path.name]
        elif relative.parent == Path("entrypoints") and path.name in {
                "get_data.py", "feature_selection.py", "preprocess_training.py"}:
            continue
        elif relative.parent.name == "training" and path.name in {
                "train.py", "evaluate.py", "feature_selection.py", "preprocess.py"}:
            steps = {"train.py": ("Train",), "evaluate.py": ("Evaluate",)}.get(path.name, ())
        elif relative.parent.name == "inference" and path.name in inference:
            steps = inference[path.name]
        content = b"\0" + relative.as_posix().encode() + b"\0" + path.read_bytes()
        for step in steps:
            digests[step].update(content)
    # The inference CSV can share the notebook's uploaded input. Changed bytes
    # must invalidate its first consumer even when its S3 URI stays the same.
    revisions["BatchGetData"] = revisions["GetData"]
    revisions.update({step: digest.hexdigest() for step, digest in digests.items()})
    return revisions


def configure_step_cache(body, revisions, kind, data_source):
    """Cache compute steps; keep registry checks and side effects live."""
    for step in walk_steps(body["Steps"]):
        name = ({"GetData": "BatchGetData", "Preprocess": "BatchPreprocess"}.get(step["Name"], step["Name"])
                if kind == "inference" else step["Name"])
        if name not in revisions or step["Type"] not in {"Processing", "Training"}:
            continue
        # A latest-delivery query has changing external state, not a fixed input.
        if data_source == "feature_store" and name in {"GetData", "BatchGetData"}:
            continue
        environment = step["Arguments"].setdefault("Environment", {})
        for key in ("PIPELINE_EXECUTION_ID", "SOURCE_COMMIT", "SOURCE_DIRTY"):
            environment.pop(key, None)
        environment["MLOPS_STEP_REVISION"] = revisions[name]
        step["CacheConfig"] = {"Enabled": True, "ExpireAfter": "PT24H"}

    if kind == "training":
        # Both quality gates and registration must read the actual Evaluate
        # producer, which can belong to an earlier execution after a cache hit.
        replacements = []
        for step in walk_steps(body["Steps"]):
            if step["Name"] not in {"Evaluate", "EvaluateTuned"}:
                continue
            output = step["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]
            replacements.append((
                join(output["S3Output"]["S3Uri"], "evaluation.json"),
                join({"Get": f"Steps.{step['Name']}.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri"},
                     "evaluation.json"),
            ))
            template_uri = deepcopy(output["S3Output"]["S3Uri"])
            template_uri["Std:Join"]["Values"][-1] += "/evaluation.json"
            replacements.append((template_uri, replacements[-1][1]))
        # Templates sometimes express a filename inside the last path segment.
        body.update(replace(flatten_joins(body), [(flatten_joins(before), flatten_joins(after))
                                                for before, after in replacements]))
        for step in walk_steps(body["Steps"]):
            if step["Type"] != "RegisterModel":
                continue
            args = step["Arguments"]
            report_uri = args["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
            args["MetadataProperties"]["CommitId"] = {
                "Std:JsonGet": {"S3Uri": report_uri, "Path": "model_source_commit"}}
            args["CustomerMetadataProperties"]["SourceDirty"] = {
                "Std:JsonGet": {"S3Uri": report_uri, "Path": "model_source_dirty"}}


def upload_examples(session, s3_root, models):
    """Put each selected model's sample under inputs/<model>/ for the ModelId selector."""
    s3 = session.client("s3")
    uri = s3_root.rstrip("/") + "/inputs"
    bucket, prefix = s3_location(uri)
    for model in models:
        config = json.loads((PACKAGE / "models" / model / "parameters.json").read_text())
        path = APPLICATION.parent / "examples/data" / config["example_input"]
        s3.upload_file(str(path), bucket, f"{prefix}/{model}/{path.name}")
    return uri


def join(*values, on="/"):
    """Build a native pipeline string expression."""
    return {"Std:Join": {"On": on, "Values": list(values)}}


def replace(value, replacements):
    """Replace exact native expressions while retaining the common pipeline templates."""
    for before, after in replacements:
        if value == before:
            return deepcopy(after)
    if isinstance(value, dict):
        return {key: replace(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace(item, replacements) for item in value]
    return value


def source_input(uri):
    """Mount source separately from Processing data inputs."""
    return {"InputName": "source", "S3Input": {
        "S3Uri": uri, "LocalPath": "/opt/ml/processing/source",
        "S3DataType": "S3Prefix", "S3InputMode": "File",
        "S3DataDistributionType": "FullyReplicated"}}


def flatten_joins(value):
    """SageMaker JsonGet paths require flat joins of literals and parameter references."""
    if isinstance(value, list):
        return [flatten_joins(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = {key: flatten_joins(item) for key, item in value.items()}
    if "Std:Join" in value:
        spec = value["Std:Join"]
        spec["Values"] = [part for item in spec["Values"]
                          for part in (item["Std:Join"]["Values"]
                                       if isinstance(item, dict) and "Std:Join" in item
                                       and item["Std:Join"]["On"] == spec["On"] else [item])]
    return value


def build_definitions(namespace, job_role, image, s3_root, source_uri, identity, *,
                      region, models=None, infrastructure=None, batch_runtime="transform",
                      feature_store_config_uri="", universe_data_root="",
                      data_source="feature_store", inference_data_source=None, csv_input_uri="", preparation_cache_revision="",
                      model_package_group="", environment="dev", registry_group_prefix="",
                      include_hpo=False, step_cache_revisions=None, model_code_source="artifact"):
    """Render training registration and inference from an approved Registry version."""
    data_sources = {"training": data_source, "inference": inference_data_source or data_source}
    if any(value not in {"feature_store", "csv_direct"} for value in data_sources.values()):
        raise ValueError("Choose feature_store or csv_direct for each pipeline data source.")
    model_code_source = model_code_source or "artifact"
    if model_code_source not in {"pipeline", "artifact"}:
        raise ValueError("Choose pipeline or artifact for model_code_source.")
    if batch_runtime == "transform" and model_code_source != "artifact":
        raise ValueError("Batch Transform serves the code in the model artifact; use model_code_source='artifact'.")
    if preparation_cache_revision and data_source != "csv_direct":
        raise ValueError("Preparation caching currently requires a fixed csv_direct input.")
    if "feature_store" in data_sources.values() and not feature_store_config_uri:
        raise ValueError("Supply feature_store_config_uri from notebooks/01_prepare_data.ipynb.")
    if "csv_direct" in data_sources.values():
        s3_location(csv_input_uri)
        if not csv_input_uri.endswith(".csv"):
            raise ValueError("csv_direct needs the S3 URI of one flat .csv file.")
        if universe_data_root and "feature_store" not in data_sources.values():
            raise ValueError("csv_direct uses labels in the flat CSV; leave universe_data_root empty.")
    validate_namespace(namespace)
    validate_namespace(environment)
    data_prefix({"prefix": s3_location(s3_root)[1]})
    if "feature_store" in data_sources.values():
        s3_location(feature_store_config_uri)
        if universe_data_root:
            s3_location(universe_data_root)
        if (infrastructure or {}).get("network", {}).get("isolation"):
            raise ValueError("Feature Store queries need network access to Athena, S3 and SageMaker.")
    available = [p.name for p in discover_models(PACKAGE / "models")]
    models = available if models is None else models
    if not models or any(model not in available for model in models):
        raise ValueError("Select model folders containing parameters.json.")
    if batch_runtime not in {"transform", "processing"}:
        raise ValueError("Choose transform or processing for batch_runtime.")
    if model_package_group and len(models) != 1:
        raise ValueError("An explicit registry group requires one selected model.")
    if model_package_group and registry_group_prefix:
        raise ValueError("Use either an exact model_package_group or registry_group_prefix, not both.")
    if not isinstance(registry_group_prefix, str):
        raise ValueError("registry_group_prefix must be a string, including any separator.")
    configs = [json.loads((PACKAGE / "models" / name / "parameters.json").read_text()) for name in models]
    if include_hpo:
        if not all(config.get("tuning") for config in configs):
            raise ValueError("include_hpo requires tuning settings in every selected model's parameters.json.")
        search = configs[0]["tuning"]
        if any(c["tuning"]["parameters"] != search["parameters"] or
               c["tuning"].get("fixed", {}) != search.get("fixed", {}) for c in configs):
            raise ValueError("Shared HPO requires the same search space and fixed parameters. "
                             "Select a compatible model subset when publishing.")
    model = {"Get": "Parameters.ModelId"}
    output = join({"Get": "Parameters.OutputDataRoot"}, model, "runs")
    group = model_package_group or join(model, "sm", environment, on="-")
    if registry_group_prefix:
        group = join(registry_group_prefix, group, on="")
    template_model = models[0] if len(models) == 1 else "shared"
    definitions = {}
    for kind in ("training", "inference"):
        body = load_definition(
            template_model, kind, namespace=namespace, role=job_role, image=image,
            model_package_group=namespace + "-" + template_model,
            input_uri=s3_root.rstrip("/") + "/inputs",
            output_uri=s3_root.rstrip("/") + "/models",
            source_commit=identity["commit"], source_dirty=str(identity["dirty"]).lower(),
            source_repository="", image_digest=image.split("@")[-1],
            definition_path=PACKAGE / "model_base" / kind / "definition.json",
            tuning_config=configs[0], include_hpo=include_hpo,
        )
        body = replace(body, [
            (template_model, model), (namespace + "-" + template_model, group),
            ({"Get": "Parameters.InputDataUri"}, join({"Get": "Parameters.InputDataRoot"}, model)),
            ({"Get": "Parameters.OutputDataUri"}, output),
        ])
        for parameter in body["Parameters"]:
            parameter["Name"] = {"InputDataUri": "InputDataRoot", "OutputDataUri": "OutputDataRoot"}.get(
                parameter["Name"], parameter["Name"])
        body["Parameters"].insert(0, {"Name": "ModelId", "Type": "String",
                                      "DefaultValue": models[0], "EnumValues": models})
        body["Metadata"].update(SourceUri=source_uri, DeployMode=identity["mode"],
                                Release=identity["name"], BatchRuntime=batch_runtime, DataSource=data_sources[kind],
                                RegistryMode="model_registry", Environment=environment,
                                ModelCodeSource=model_code_source,
                                HpoEnabled=str(include_hpo).lower())
        for step in walk_steps(body["Steps"]):
            args = step["Arguments"]
            if step["Type"] == "Processing":
                action = args["AppSpecification"]["ContainerEntrypoint"][-1]
                args["AppSpecification"]["ContainerEntrypoint"] = [
                    "python", "/opt/ml/processing/source/load_source.py", action]
                arguments = args["AppSpecification"].get("ContainerArguments", [])
                args["AppSpecification"]["ContainerArguments"] = [
                    item for index in range(0, len(arguments), 2)
                    if arguments[index] not in {"--input", "--output", "--model-dir"}
                    for item in arguments[index:index + 2]]
                if not args["AppSpecification"]["ContainerArguments"]:
                    del args["AppSpecification"]["ContainerArguments"]
                args.setdefault("ProcessingInputs", []).append(source_input(source_uri))
            elif step["Type"] == "Training":
                args["AlgorithmSpecification"]["ContainerEntrypoint"] = [
                    "python", "/opt/ml/input/data/source/load_source.py", "train"]
                args["InputDataConfig"].append({"ChannelName": "source", "DataSource": {
                    "S3DataSource": {"S3Uri": source_uri, "S3DataType": "S3Prefix",
                                     "S3DataDistributionType": "FullyReplicated"}}})
            elif step["Type"] == "Tuning":
                # HPO cannot override image ENTRYPOINT. A compatible image launches
                # `python -m mlops_models.entrypoints train` using this mounted source.
                job = args["TrainingJobDefinition"]
                job["InputDataConfig"].append({"ChannelName": "source", "DataSource": {
                    "S3DataSource": {"S3Uri": source_uri, "S3DataType": "S3Prefix",
                                     "S3DataDistributionType": "FullyReplicated"}}})
                job["Environment"].update(PYTHONPATH="/opt/ml/input/data/source", PYTHONSAFEPATH="1")
                config_uri = join(source_uri, "mlops_models/models", model, "parameters.json")
                def setting(path):
                    return {"Std:JsonGet": {"S3Uri": config_uri, "Path": "tuning.objective." + path}}
                args["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"] = {
                    "MetricName": setting("name"), "Type": setting("direction")}
                job["AlgorithmSpecification"]["MetricDefinitions"] = [
                    {"Name": setting("name"), "Regex": r"tuning_objective=([0-9.eE+-]+);"}]
                job["Environment"]["TUNING_OBJECTIVE"] = "enabled"
            elif step["Type"] == "RegisterModel":
                args["InferenceSpecification"]["Containers"][0]["Environment"].update(
                    PYTHONPATH="/opt/ml/model/code", PYTHONSAFEPATH="1")
                args["CustomerMetadataProperties"].update(SourceUri=source_uri,
                                                          DeployMode=identity["mode"])
        if kind == "training":
            from .training_files import configure_file_training
            configure_file_training(body)
        if kind == "inference":
            configure_approved_inference(body, group, source_uri, batch_runtime, region)
        if data_sources[kind] == "feature_store":
            configure_feature_store(body, feature_store_config_uri, universe_data_root, region)
        else:
            configure_csv_direct(body, csv_input_uri, region)
        apply_job_settings(body, infrastructure or {})
        if kind == "training":
            train = next(step for step in walk_steps(body["Steps"]) if step["Name"] == "Train")["Arguments"]
            if not train.get("EnableNetworkIsolation", False):
                # Source-delivered ordinary Training uploads its artifacts from writable scratch.
                # Keep these separate from SageMaker's native <job>/output artifacts.
                run = join(output, {"Get": "Execution.PipelineExecutionId"})
                model_uri = join(run, "model/application/model.tar.gz")
                if step_cache_revisions:
                    train["Environment"]["MLOPS_TRAINING_OUTPUT_UPLOAD"] = "application"
                    model_uri = join({"Get": "Steps.Train.OutputDataConfig.S3OutputPath"},
                                     "application/model.tar.gz")
                else:
                    train["Environment"].update(
                        MLOPS_TRAINING_MODEL_URI=model_uri,
                        MLOPS_TRAINING_REPORTS_URI=join(run, "reports/training/output.tar.gz"),
                        MLOPS_TRAINING_KMS_KEY=train["OutputDataConfig"].get("KmsKeyId", ""))
                # Evaluate already depends on Train; registry reads its reported checkpoint_uri.
                body = replace(body, [({"Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"}, model_uri)])
        for step in walk_steps(body["Steps"]):
            if step["Type"] != "Processing":
                continue
            args = step["Arguments"]
            # Isolated containers cannot upload to S3; retain SageMaker's uploader.
            if args.get("NetworkConfig", {}).get("EnableNetworkIsolation", False):
                continue
            for processing_output in args["ProcessingOutputConfig"]["Outputs"]:
                processing_output["AppManaged"] = True
            args.setdefault("Environment", {})["MLOPS_PROCESSING_OUTPUT_UPLOAD"] = "application"
        if kind == "training" and preparation_cache_revision:
            from .training_files import configure_preparation_cache
            configure_preparation_cache(body, preparation_cache_revision)
        for step in walk_steps(body["Steps"]):
            if step["Type"] == "Processing" and step["Name"] in {
                    "Evaluate", "EvaluateTuned", "GetData", "Preprocess", "Score", "Postprocess"}:
                if kind == "training" and step["Name"] in {"GetData", "Preprocess"}:
                    continue
                # Evaluation is independently editable. Batch defaults to code
                # packaged with the approved Registry model.
                step["Arguments"]["Environment"]["MLOPS_MODEL_CODE_SOURCE"] = (
                    "pipeline" if kind == "training" or step["Name"] == "GetData" else model_code_source)
        if step_cache_revisions:
            configure_step_cache(body, step_cache_revisions, kind, data_sources[kind])
        if kind == "training":
            configure_registration(body, source_uri, region)
        definitions[kind] = flatten_joins(body)
    return definitions


def configure_registration(body, source_uri, region):
    """Register through the API without granting native RegisterModel group-creation rights."""
    steps = {step["Name"]: step for step in walk_steps(body["Steps"])}
    for step in steps.values():
        if step["Type"] != "RegisterModel":
            continue
        request = step["Arguments"]
        evaluation = steps["EvaluateTuned" if step["Name"] == "RegisterTunedModel" else "Evaluate"]
        args = deepcopy(evaluation["Arguments"])
        args["ProcessingJobName"] = args["ProcessingJobName"].replace("-eval", "-reg")
        args["AppSpecification"]["ContainerEntrypoint"] = [
            "python", "/opt/ml/processing/source/mlops_models/registry/model_package.py"]
        args["AppSpecification"].pop("ContainerArguments", None)
        report_uri = request["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
        output_uri = join(request["CustomerMetadataProperties"]["RunArtifacts"], "reports/registry")
        args["ProcessingInputs"] = [source_input(source_uri), {
            "InputName": "evaluation", "S3Input": {
                "S3Uri": report_uri, "LocalPath": "/opt/ml/processing/evaluation",
                "S3DataType": "S3Prefix", "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated"}}]
        args["ProcessingOutputConfig"]["Outputs"] = [{
            "OutputName": "registration", "AppManaged": True, "S3Output": {
                "S3Uri": output_uri, "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"}}]
        args["Environment"] = {
            "MODEL_ID": {"Get": "Parameters.ModelId"},
            "MODEL_PACKAGE_GROUP": request["ModelPackageGroupName"],
            "IMAGE_URI": request["InferenceSpecification"]["Containers"][0]["Image"],
            "EVALUATION_URI": report_uri, "REGISTRATION_OUTPUT_URI": output_uri,
            "SOURCE_URI": source_uri, "DEPLOY_MODE": body["Metadata"]["DeployMode"],
            "PIPELINE_EXECUTION_ARN": {"Get": "Execution.PipelineExecutionArn"},
            "AWS_DEFAULT_REGION": region, "PYTHONUNBUFFERED": "1",
            "S3_KMS_KEY": args["ProcessingOutputConfig"].get("KmsKeyId", ""),
        }
        step.update(Type="Processing", Arguments=args)


def configure_csv_direct(body, csv_uri, region):
    """Mount one flat CSV for either pipeline's GetData step."""
    body["Parameters"] = [p for p in body["Parameters"] if p["Name"] != "InputDataRoot"]
    body["Parameters"].append({"Name": "CsvInputUri", "Type": "String", "DefaultValue": csv_uri})
    for step in walk_steps(body["Steps"]):
        if step["Type"] != "Processing":
            continue
        args = step["Arguments"]
        if args["AppSpecification"]["ContainerEntrypoint"][-1] not in {"get-data", "get-inference-data"}:
            continue
        args["ProcessingInputs"] = [item for item in args["ProcessingInputs"]
                                   if item["S3Input"]["LocalPath"] != "/opt/ml/processing/input"]
        args["ProcessingInputs"].append({
            "InputName": "input", "S3Input": {
                "S3Uri": {"Get": "Parameters.CsvInputUri"}, "LocalPath": "/opt/ml/processing/input",
                "S3DataType": "S3Prefix", "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated"}})
        args.setdefault("Environment", {}).update(
            MLOPS_DATA_SOURCE="csv_direct", MLOPS_CSV_INPUT_URI={"Get": "Parameters.CsvInputUri"},
            AWS_DEFAULT_REGION=region,
        )



def configure_feature_store(body, config_uri, universe_root, region):
    """Configure either pipeline's GetData to reconstruct its Feature Store snapshot."""
    body["Parameters"] = [p for p in body["Parameters"] if p["Name"] != "InputDataRoot"]
    body["Parameters"].extend([
        {"Name": "FeatureStoreConfigUri", "Type": "String", "DefaultValue": config_uri},
        {"Name": "FeatureStoreDeliveryId", "Type": "String", "DefaultValue": ""},
        {"Name": "UniverseDataRoot", "Type": "String", "DefaultValue": universe_root},
    ])
    for step in walk_steps(body["Steps"]):
        if step["Type"] != "Processing" or step["Arguments"]["AppSpecification"]["ContainerEntrypoint"][-1] not in {"get-data", "get-inference-data"}:
            continue
        args = step["Arguments"]
        args["ProcessingInputs"] = [item for item in args["ProcessingInputs"]
                                   if item["S3Input"]["LocalPath"] != "/opt/ml/processing/input"]
        raw = next(item for item in args["ProcessingOutputConfig"]["Outputs"] if item["OutputName"] == "raw")
        args.setdefault("Environment", {}).update(
            MLOPS_DATA_SOURCE="feature_store",
            MLOPS_FEATURE_STORE_CONFIG_URI={"Get": "Parameters.FeatureStoreConfigUri"},
            MLOPS_FEATURE_STORE_DELIVERY_ID={"Get": "Parameters.FeatureStoreDeliveryId"},
            MLOPS_FEATURE_STORE_UNIVERSE_ROOT={"Get": "Parameters.UniverseDataRoot"},
            MLOPS_FEATURE_STORE_SNAPSHOT_URI=raw["S3Output"]["S3Uri"],
            AWS_DEFAULT_REGION=region,
        )


def configure_approved_inference(body, group, source_uri, batch_runtime, region):
    """Resolve one approved package, then feed its exact image and checkpoint to all scoring steps."""
    steps = {step["Name"]: step for step in body["Steps"]}
    resolver = steps["ValidatePackage"]
    resolver["Name"] = "ResolveModel"
    args = resolver["Arguments"]
    args["Environment"] = {"MODEL_ID": {"Get": "Parameters.ModelId"},
                           "MODEL_PACKAGE_GROUP": group, "PYTHONUNBUFFERED": "1",
                           "AWS_DEFAULT_REGION": region}
    args["AppSpecification"]["ContainerEntrypoint"][-1] = "resolve-model"
    args["AppSpecification"]["ContainerArguments"] = ["--output", "/opt/ml/processing/output"]
    args["ProcessingInputs"] = [source_input(source_uri)]
    args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["LocalPath"] = "/opt/ml/processing/output"
    decision_uri = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    def selected(field):
        return {"Std:JsonGet": {"S3Uri": join(decision_uri, "package.json"), "Path": field}}
    body["Steps"] = replace(body["Steps"], [
        ("ValidatePackage", "ResolveModel"),
        ({"Get": "Parameters.ModelDataUri"}, selected("checkpoint")),
        ({"Get": "Parameters.InferenceImageUri"}, selected("image")),
    ])
    body["Parameters"] = [p for p in body["Parameters"]
                           if p["Name"] not in {"ModelDataUri", "InferenceImageUri", "ModelPackageArn"}]
    steps = {step["Name"]: step for step in body["Steps"]}
    for name in ("GetData", "Preprocess", "CreateModel", "Postprocess"):
        dependencies = steps[name].setdefault("DependsOn", [])
        if "ResolveModel" not in dependencies:
            dependencies.append("ResolveModel")
        args = steps[name]["Arguments"]
        if steps[name]["Type"] == "Processing":
            args["AppSpecification"]["ImageUri"] = selected("image")
    steps["CreateModel"]["Arguments"]["PrimaryContainer"]["Environment"].update(
        PYTHONPATH="/opt/ml/model/code", PYTHONSAFEPATH="1")
    if batch_runtime == "processing":
        steps["Preprocess"]["Arguments"]["Environment"].pop("MLOPS_PREPARE_TRANSFORM_INPUT", None)
        steps["Postprocess"]["Arguments"]["Environment"]["MLOPS_PREDICTION_FORMAT"] = "csv"
        score = deepcopy(steps["Preprocess"])
        score["Name"] = "Score"
        score["DependsOn"] = ["Preprocess", "ResolveModel"]
        args = score["Arguments"]
        args["ProcessingJobName"] = args["ProcessingJobName"].removesuffix("-prep") + "-score"
        args["AppSpecification"]["ContainerEntrypoint"][-1] = "score"
        args["AppSpecification"].pop("ContainerArguments", None)
        args["ProcessingInputs"][0]["S3Input"].update(
            S3Uri={"Get": "Steps.Preprocess.ProcessingOutputConfig.Outputs['prepared'].S3Output.S3Uri"},
            LocalPath="/opt/ml/processing/prepared")
        args["ProcessingOutputConfig"]["Outputs"] = [{
            "OutputName": "scores", "S3Output": {
                "S3Uri": steps["Transform"]["Arguments"]["TransformOutput"]["S3OutputPath"],
                "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"}}]
        body["Steps"] = [steps["ResolveModel"], steps["GetData"], steps["Preprocess"], score, steps["Postprocess"]]
        body["Steps"] = replace(body["Steps"], [
            ("Transform", "Score"),
            ({"Get": "Steps.Transform.TransformOutput.S3OutputPath"},
             {"Get": "Steps.Score.ProcessingOutputConfig.Outputs['scores'].S3Output.S3Uri"}),
        ])
        body["Parameters"] = [p for p in body["Parameters"] if p["Name"] not in {
            "TransformInstanceType", "TransformInstanceCount", "MaxConcurrentTransforms",
            "MaxPayloadInMB", "ModelServerWorkers"}]


def publish(session, namespace, pipeline_role, definitions, identity):
    """Create/update both owned pipelines; label their numeric versions with the supplied source commit."""
    client = session.client("sagemaker")
    published = {}
    for kind, body in definitions.items():
        name = f"{namespace}-{kind}"
        request = dict(PipelineName=name, RoleArn=pipeline_role, PipelineDefinition=json.dumps(body))
        try:
            existing = client.describe_pipeline(PipelineName=name)
        except client.exceptions.ClientError as error:
            if "does not exist" not in error.response["Error"]["Message"]:
                raise
            response = client.create_pipeline(**request, Tags=[
                {"Key": "Deployment", "Value": namespace}, {"Key": "SourceDeploy", "Value": identity["mode"]}])
        else:
            tags = {t["Key"]: t["Value"] for t in client.list_tags(ResourceArn=existing["PipelineArn"])["Tags"]}
            if tags.get("Deployment") != namespace or tags.get("SourceDeploy") != identity["mode"]:
                raise ValueError(f"{name} belongs to another deployment.")
            response = client.update_pipeline(**request)
        versions = client.list_pipeline_versions(PipelineName=name, SortOrder="Descending")
        version = versions["PipelineVersionSummaries"][0]["PipelineVersionId"]
        actual = client.describe_pipeline(PipelineName=name, PipelineVersionId=version)
        if json.loads(actual["PipelineDefinition"]) != body:
            raise RuntimeError("The published definition changed concurrently; inspect before executing.")
        client.update_pipeline_version(
            PipelineArn=response["PipelineArn"], PipelineVersionId=version,
            PipelineVersionDisplayName=f"{identity['mode']}-{identity['commit'][:12]}",
            PipelineVersionDescription=json.dumps(identity),
        )
        published[kind] = {"name": name, "arn": response["PipelineArn"], "version": version}
        print(f"Studio → Pipelines → {name} | version {version} | commit {identity['commit']}")
    return published
