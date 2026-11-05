"""Publish verified pipeline definitions using the detailed notebook's existing flow."""
from platform_settings import _atomic_json


def publish_pipelines(root, session, *, options=None, setup_receipt=None, progress=None):
    """Validate, upload and publish. This does not execute training or approve models."""
    def show_progress(message):
        if progress is not None:
            progress(message)

    show_progress("Checking data and model settings...")
    options = dict(options or {})
    unknown = set(options) - set(['batch_runtime', 'cache_steps', 'completion_permissions_boundary_arn', 'data_source', 'deploy_name', 'environment', 'include_hpo', 'inference_data_source', 'infrastructure', 'job_role_arn', 'model_code_source', 'model_ids', 'monitoring', 'universe_data_root'])
    if unknown:
        raise ValueError(f"Unknown publication options: {sorted(unknown)}")
    from pathlib import Path
    import json

    published = {}
    verified_source_uri = ""

    ROOT = Path(root).resolve()
    CONFIG_PATH = ROOT / "notebooks/config.json"
    TEMPLATE_PATH = ROOT / "notebooks/config.example.json"
    if not CONFIG_PATH.exists():
        raise ValueError(f"Load platform settings before calling this function: {CONFIG_PATH}")

    # Read supplied inputs; optional settings are declared in the Defaults cell below.
    notebook_config = json.loads(TEMPLATE_PATH.read_text()) | json.loads(CONFIG_PATH.read_text())
    print("Notebook configuration:", CONFIG_PATH)
    bucket = notebook_config["bucket"]
    role_arn = notebook_config["role_arn"]
    image_uri = notebook_config["image_uri"]
    csv_input = notebook_config["csv_input"]
    test_csv_input = notebook_config.get("test_csv_input", "").strip()
    inference_csv_input = notebook_config.get("inference_csv_input", "").strip()
    test_csv_input_uri, inference_csv_input_uri = test_csv_input, inference_csv_input
    feature_store_config_uri = notebook_config["feature_store_config_uri"]
    registry_group_prefix = notebook_config["registry_group_prefix"]
    tracking_uri = notebook_config["tracking_uri"].strip()

    completion_code_uri = notebook_config.get("completion_code_uri", "").strip()


    import hashlib
    import re

    namespace = notebook_config.get("namespace", "").strip()
    if not namespace:
        # A supplied alias also works outside Studio. Do not infer a person from
        # the shared container login or an assumed-role session name.
        deployment_user = notebook_config.get("user_alias", "").strip()
        identity = deployment_user
        if not deployment_user:
            metadata_path = Path("/opt/ml/metadata/resource-metadata.json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            deployment_user = metadata.get("UserProfileName", "").strip()
            identity = metadata.get("DomainId", "") + "/" + deployment_user
        if not deployment_user:
            raise ValueError(
                "No Studio user profile was found. Set user_alias once in config.json "
                "(for example your username), or supply namespace explicitly."
            )
        label = re.sub(r"[^a-z0-9]+", "-", deployment_user.lower()).strip("-")[:7].rstrip("-") or "user"
        namespace = "wsf-" + label + "-" + hashlib.sha256(identity.encode()).hexdigest()[:8]
        notebook_config["namespace"] = namespace
        _atomic_json(CONFIG_PATH, notebook_config)
        print("Deployment user:", deployment_user)
    print("Reusing deployment namespace:", namespace)

    s3_prefix = notebook_config.get("s3_prefix", "wsf-mlops").strip().strip("/")
    if (not re.fullmatch(r"[A-Za-z0-9_./-]+", s3_prefix)
            or any(part in {"", ".", ".."} for part in s3_prefix.split("/"))):
        raise ValueError("s3_prefix must be a nonempty folder path without '.' or '..' components.")
    storage_prefix = f"{s3_prefix}/{namespace}"
    print("Shared S3 folder:", s3_prefix + "/")
    print("Namespace S3 folder:", storage_prefix + "/")
    data_source = options.get("data_source", "csv_direct")
    inference_data_source = options.get("inference_data_source", "csv_direct")
    environment = options.get("environment", "dev")
    deploy_name = options.get("deploy_name", "my-review")
    model_ids = options.get("model_ids", None)
    include_hpo = options.get("include_hpo", False)
    tracking_experiment_prefix = namespace
    # Adds HasReference → DriftComparison. Requires Evidently in the approved image.
    monitoring = options.get("monitoring", {"enabled": True, "prediction_file": "messaging_final.csv", "score_column": "score",
                  "threshold": 0.2, "drift_share": 0.5})
    cache_steps = options.get("cache_steps", True)
    model_code_source = options.get("model_code_source", "artifact")
    universe_data_root = options.get("universe_data_root", "")
    job_role_arn = options.get("job_role_arn", role_arn)
    batch_runtime = options.get("batch_runtime", "processing")
    infrastructure = options.get("infrastructure", {
        "network": {"isolation": False, "subnet_ids": [], "security_group_ids": []},
        "output_kms_key_arn": "",
        "volume_kms_key_arn": "",
    })
    csv_preflight_ok = csv_upload_verified = False
    print("Training input:", data_source, "| inference input:", inference_data_source, "| cache_steps:", cache_steps)
    print("Inference runtime:", batch_runtime, "| default ModelCodeSource:", model_code_source)

    completion_permissions_boundary_arn = options.get("completion_permissions_boundary_arn", "")


    from uuid import uuid4
    import csv
    import hashlib
    import io
    import sys
    import boto3
    import pandas as pd

    csv_preflight_ok = csv_upload_verified = False
    verified_source_uri = ""
    published = {}
    bucket = bucket.strip().removeprefix("s3://").rstrip("/")
    role_arn, image_uri = role_arn.strip(), image_uri.strip()
    csv_input = csv_input.strip()
    print("Effective csv_input:", repr(csv_input))
    print("Current notebook directory:", Path.cwd())
    print("cache_steps:", cache_steps)
    assert bucket and "/" not in bucket and role_arn and image_uri, f"STOP: fill in bucket, role_arn and image_uri in {CONFIG_PATH}."
    data_sources = {"training": data_source, "testing": inference_data_source, "inference": inference_data_source}
    assert all(value in {"feature_store", "csv_direct"} for value in data_sources.values()), "STOP: choose csv_direct or feature_store for each pipeline."
    uses_csv = "csv_direct" in data_sources.values()
    if "feature_store" in data_sources.values():
        assert feature_store_config_uri, f"STOP: supply feature_store_config_uri in {CONFIG_PATH}, or complete notebook 01 in feature_store mode."
    if uses_csv:
        assert csv_input, f"STOP: csv_input is empty in {CONFIG_PATH}. Run notebook 01, or set an existing CSV path in that file. Nothing will be published."

    sys.path[:0] = [str(ROOT / "mlops-models"), str(ROOT / "mlops-models/src"), str(ROOT / "mlops-features")]
    import mlops_models
    import mlops_skeleton
    for package in (mlops_models, mlops_skeleton):
        assert Path(package.__file__).resolve().is_relative_to(ROOT.resolve()), (
            f"STOP: this kernel loaded code from {package.__file__}. Restart this notebook's kernel for the new checkout."
        )
    from mlops_models.pipelines.source import (
        source_identity, source_location, upload_source, put_immutable, build_definitions, pipeline_fingerprints,
        configure_cache_sources, publish_cache_sources, preparation_fingerprint,
    )
    from mlops_skeleton.names import discover_models, validate_namespace
    from mlops_models.pipelines import source as source_module

    print("Packaging module:", Path(source_module.__file__).resolve())
    print("Packaging source directory:", source_module.PACKAGE.resolve())
    assert source_module.PACKAGE.resolve() == (ROOT / "mlops-models/src/mlops_models").resolve(), (
        "STOP: the packaging module points at another checkout. Restart the kernel in the intended project."
    )

    validate_namespace(namespace)
    available_models = [folder.name for folder in discover_models(ROOT / "mlops-models/src/mlops_models/models")]
    model_ids = available_models if model_ids is None else model_ids
    assert model_ids and all(model in available_models for model in model_ids), f"STOP: select model folders from {available_models}."
    def csv_settings():
        # Recheck if the CSV, model list or any use-case parameters changed.
        return csv_input, test_csv_input, inference_csv_input, test_csv_input_uri, inference_csv_input_uri, data_source, inference_data_source, tuple(
            (model, (ROOT / "mlops-models/src/mlops_models/models" / model / "parameters.json").read_text())
            for model in model_ids
        )

    model_targets = {}
    for model in model_ids:
        parameters_path = ROOT / "mlops-models/src/mlops_models/models" / model / "parameters.json"
        parameters = json.loads(parameters_path.read_text())
        for key in ("target_key", "primary_key"):
            assert isinstance(parameters.get(key), str) and parameters[key].strip(), f"STOP: set {key} in {parameters_path}."
        assert parameters.get("task_type") in {"classification", "regression"}, f"STOP: task_type must be classification or regression in {parameters_path}."
        model_targets[model] = parameters["target_key"]
        print("Model:", model, "| target_key:", repr(parameters["target_key"]),
              "| primary_key:", repr(parameters["primary_key"]), "| task_type:", parameters["task_type"])
        print("Parameters:", parameters_path)

    for code_file in (ROOT / "mlops-models/src/mlops_models").rglob("*.py"):
        compile(code_file.read_text(), str(code_file), "exec")
    print("Model Python syntax checks passed:", model_ids)
    pipeline_role_arn = role_arn
    job_role_arn = job_role_arn.strip() or role_arn
    sm = session.client("sagemaker")
    s3_root = f"s3://{bucket}/{storage_prefix}"
    identity = source_identity("direct-deploy", deploy_name)
    source_uri = source_location(s3_root, identity)
    csv_input_uri, csv_local_path, csv_check = "", None, None
    data_preview = []
    if uses_csv:
        if csv_input.startswith("s3://"):
            csv_input_uri = csv_input
        else:
            csv_local_path = Path(csv_input).expanduser()
            if not csv_local_path.is_absolute():
                csv_local_path = ROOT / csv_local_path
            csv_local_path = csv_local_path.resolve()
            print("Resolved local CSV:", csv_local_path)
            assert csv_local_path.is_file(), f"STOP: CSV file does not exist: {csv_local_path}. Nothing will be published."
        try:
            if csv_local_path is not None:
                csv_bytes = csv_local_path.read_bytes()
                csv_input_uri = f"{s3_root}/inputs/csv-direct/{deploy_name}/sha256-{hashlib.sha256(csv_bytes).hexdigest()}/data.csv"
            else:
                csv_bucket, csv_key = csv_input_uri.removeprefix("s3://").split("/", 1)
                response = session.client("s3").get_object(Bucket=csv_bucket, Key=csv_key)
                with response["Body"] as stream:
                    csv_bytes = stream.read()
        except Exception as error:
            raise RuntimeError(f"STOP: cannot read the requested CSV {csv_local_path or csv_input_uri}. Nothing will be published: {error}") from error
        columns = next(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"))), [])
        print("CHECKED CSV:", str(csv_local_path or csv_input_uri))
        print("Bytes:", len(csv_bytes), "| Columns:", len(columns))
        print("First 30 column names:", columns[:30])
        print("Model targets from parameters.json:", model_targets)
        assert columns and all(columns), "STOP: CSV has no valid header."
        assert len(set(columns)) == len(columns), "STOP: CSV contains duplicate column names."
        target_presence = {model: target in columns for model, target in model_targets.items()}
        for model, present in target_presence.items():
            print(model, "| target in this CSV:", present, "|", repr(model_targets[model]))
        if not all(target_presence.values()):
            print("This CSV does not contain every configured target. Publication can proceed.")
            print("When training, choose the intended ModelId and its CsvInputUri; that model's code handles its target.")
        preview = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig", nrows=5, dtype=str, keep_default_na=False)
        data_preview = preview.iloc[:, :6].to_dict(orient="records")
        assert len(preview), "STOP: CSV has a header but no data rows."
        print(preview.iloc[:, :10].to_string(index=False))
        csv_check = {"source": str(csv_local_path or csv_input_uri), "sha256": hashlib.sha256(csv_bytes).hexdigest(),
                     "bytes": len(csv_bytes), "columns": columns, "model_targets": model_targets, "target_presence": target_presence}
        csv_preflight_ok = True
        print("CSV FILE CHECK PASSED | SHA-256:", csv_check["sha256"])
        print("Will use S3 CSV:", csv_input_uri)
        del csv_bytes

    print("Checkout:", ROOT)
    print("Region:", session.region_name)
    print("Pipeline role:", pipeline_role_arn)
    print("Job role:", job_role_arn)
    print("Models:", model_ids)
    print("Source upload destination:", source_uri)
    print("Training input:", csv_input_uri if data_source == "csv_direct" else feature_store_config_uri)
    print("Testing input:", test_csv_input_uri or "required TestCsvInputUri when starting training")
    print("Inference input:", (inference_csv_input_uri or csv_input_uri) if inference_data_source == "csv_direct" else feature_store_config_uri)
    print("Pipelines:", namespace + "-training", "|", namespace + "-inference")
    print("Run outputs:", s3_root + "/models/<ModelId>/<pipeline-name>/runs/<execution-id>/")
    print("Source identity:", identity)
    additional_csv_inputs = {}
    for label, value in (("testing", test_csv_input), ("inference", inference_csv_input)):
        if not value:
            continue  # An explicit execution-time CsvInputUri can still be supplied.
        local_path = None
        if value.startswith("s3://"):
            uri = value
            if not uri.endswith(".csv"):
                raise ValueError(f"{label} input must name a CSV file.")
            source_bucket, source_key = uri.removeprefix("s3://").split("/", 1)
            with session.client("s3").get_object(Bucket=source_bucket, Key=source_key)["Body"] as stream:
                content = stream.read()
        else:
            local_path = Path(value).expanduser()
            local_path = (local_path if local_path.is_absolute() else ROOT / local_path).resolve()
            content = local_path.read_bytes()
            uri = f"{s3_root}/inputs/csv-direct/{deploy_name}/sha256-{hashlib.sha256(content).hexdigest()}/data.csv"
        header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))), [])
        assert header and all(header) and len(header) == len(set(header)), f"{label}: invalid CSV header."
        preview = pd.read_csv(io.BytesIO(content), nrows=1, dtype=str, keep_default_na=False)
        assert len(preview), f"{label}: CSV has no data rows."
        additional_csv_inputs[label] = {
            "uri": uri, "local_path": local_path, "sha256": hashlib.sha256(content).hexdigest(),
        }
        print(label, "| source:", local_path or uri, "| S3 input:", uri)
    test_csv_input_uri = additional_csv_inputs.get("testing", {}).get("uri", "")
    inference_csv_input_uri = additional_csv_inputs.get("inference", {}).get("uri", "")
    if uses_csv:
        csv_checked_settings = csv_settings()


    show_progress("Resolving image and building pipeline definitions...")
    if "@sha256:" not in image_uri:
        registry, repository = image_uri.split("/", 1)
        repository, _, tag = repository.partition(":")
        ecr = session.client("ecr")
        if tag:
            detail = ecr.describe_images(
                registryId=registry.split(".")[0], repositoryName=repository,
                imageIds=[{"imageTag": tag}],
            )["imageDetails"][0]
        else:
            # Repository address only: use its newest tagged image,.
            images = []
            for page in ecr.get_paginator("describe_images").paginate(
                    registryId=registry.split(".")[0], repositoryName=repository,
                    filter={"tagStatus": "TAGGED"}):
                images.extend(page["imageDetails"])
            assert images, "That repository has no tagged images; paste a full image URI instead."
            detail = max(images, key=lambda item: item["imagePushedAt"])
            print("Pinned the newest tagged image:", detail.get("imageTags"))
        image_uri = registry + "/" + repository + "@" + detail["imageDigest"]
    print("Image:", image_uri)

    if uses_csv:
        assert csv_preflight_ok and csv_checked_settings == csv_settings(), "STOP: rerun the CSV check after changing the input, selected models or their targets."

    validate_namespace(environment)
    tracking = None
    tracking_check = None
    tracking_artifact_uris = []
    if tracking_uri:
        print("Preparing MLflow notebook check...", flush=True)
        from mlops_models.pipelines.tracking import check_tracking_notebook, validate_tracking
        tracking = validate_tracking({"uri": tracking_uri, "experiment_prefix": tracking_experiment_prefix})
        tracking_check = check_tracking_notebook(tracking, [
            f"{tracking_experiment_prefix}/{environment}/{model}" for model in model_ids
        ], timeout_seconds=120)
        tracking["client_version"] = tracking_check["client_version"]
        tracking_artifact_uris = [item["artifact_location"] for item in tracking_check["experiments"]]
        print("MLflow notebook write/read check passed:", json.dumps(tracking_check, indent=2))
        print("Jobs use the MLflow client and sagemaker-mlflow plugin supplied in their image.")
        print("Job client compatibility, role access and connectivity are verified when jobs execute; UI access is separate.")
    else:
        print("MLflow is disabled. Set tracking_uri in config.json to enable it.")
    model_package_group = notebook_config.get("model_package_group", "").strip()  # Optional existing shared group.
    registry_groups = {}
    for model in model_ids:
        group_name = model_package_group or f"{registry_group_prefix}{model}-sm-{environment}"
        print("Checking existing Registry group:", group_name)
        group = sm.describe_model_package_group(ModelPackageGroupName=group_name)
        registry_groups[model] = group["ModelPackageGroupArn"]
        print("Registry group:", registry_groups[model])
    print("Group reads passed for the notebook caller. Pipeline registration and job reads are verified when executed.")

    step_cache_revisions = {}
    if cache_steps:
        step_cache_revisions = pipeline_fingerprints(session, csv_input_uri, csv_local_path)
        for kind, item in additional_csv_inputs.items():
            step = "TestGetData" if kind == "testing" else "BatchGetData"
            step_cache_revisions[step] = preparation_fingerprint(
                session, item["uri"], item["local_path"])["GetData"]
    print("Compute step cache:", "enabled for 24 hours" if step_cache_revisions else "disabled")
    if step_cache_revisions:
        print("Step revisions:", step_cache_revisions)
        print("Training: GetData, FeatureSelection, Preprocess, Train, GetTestData, PreprocessTest, PredictTest, PostprocessTest.")
        print("Processing batch inference: GetData, Preprocess, Score. Postprocess runs for every execution.")
        print("Registry checks, registration and conditions run normally. Feature Store reads remain uncached.")
        print("Unchanged compute steps retain their source snapshot across publications.")

    definitions = build_definitions(
        namespace, job_role_arn, image_uri, s3_root, source_uri, identity,
        region=session.region_name, models=model_ids, infrastructure=infrastructure, batch_runtime=batch_runtime,
        feature_store_config_uri=feature_store_config_uri, universe_data_root=universe_data_root,
        data_source=data_source, inference_data_source=inference_data_source, csv_input_uri=csv_input_uri,
        step_cache_revisions=step_cache_revisions,
        test_csv_input_uri=test_csv_input_uri, inference_csv_input_uri=inference_csv_input_uri,
        model_code_source=model_code_source,
        environment=environment, model_package_group=model_package_group,
        registry_group_prefix="" if model_package_group else registry_group_prefix, include_hpo=include_hpo,
        tracking=tracking, monitoring=monitoring,
    )
    cache_source_bindings = configure_cache_sources(definitions)
    published_cache_sources = []
    WORK = ROOT / ".runs/source-deploy" / namespace
    WORK.mkdir(parents=True, exist_ok=True)
    for kind, body in definitions.items():
        path = WORK / (kind + ".json")
        path.write_text(json.dumps(body, indent=2) + "\n")
        print(namespace + "-" + kind, "→", " → ".join(step["Name"] for step in body["Steps"]))
        print("Definition:", path.resolve())
    print("Model Registry: Development versions only; inference requires namespace, stage and model approval.")
    print("Registry namespace:", namespace)
    print("Registry groups:", list(registry_groups.values()))

    if monitoring.get("enabled"):
        print("After Postprocess: ReferenceExecutionArn=none skips DriftComparison; supply a prior execution ARN to compare.")
    print("Batch inference runtime:", batch_runtime)
    print("Default ModelCodeSource:", model_code_source)
    print("Development Processing executions can choose pipeline to test published edits without training.")
    if batch_runtime == "both":
        print("InferenceRuntime=processing: ResolveModel → GetData → Preprocess → Score → Postprocess")
        print("InferenceRuntime=transform: ResolveModel → GetDataTransform → PreprocessTransform → CreateModel → Transform → PostprocessTransform")
        print("Start two executions with the same ModelId and CsvInputUri to test the routes concurrently.")
    if batch_runtime in {"both", "transform"}:
        print("Transform uses archived code. Train/package and approve a version containing the updated serve adapter.")
        print("The image must route serve to the model server and include Flask/Gunicorn and model dependencies.")
        print("TransformOutputFile defaults to messaging_final.csv; use a relative path if needed.")
    if cache_steps and not infrastructure.get("network", {}).get("isolation", False):
        print("Cached Training needs sagemaker:DescribeTrainingJob on the job execution role.")
    print("HPO branch:", "included; UseHPO defaults to false" if include_hpo else "hidden; ordinary training only")

    csv_upload_verified = False
    verified_source_uri = ""
    published = {}
    published_cache_sources = []
    if uses_csv:
        assert csv_preflight_ok and csv_checked_settings == csv_settings(), "STOP: rerun the CSV check after changing the input, selected models or their targets."
        csv_bucket, csv_key = csv_input_uri.removeprefix("s3://").split("/", 1)
        try:
            if csv_local_path is not None:
                csv_upload_bytes = csv_local_path.read_bytes()
                assert hashlib.sha256(csv_upload_bytes).hexdigest() == csv_check["sha256"], "STOP: local CSV changed after validation; rerun the notebook."
                kms = infrastructure.get("output_kms_key_arn", "")
                encryption = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms} if kms else {}
                put_immutable(session.client("s3"), csv_bucket, csv_key, csv_upload_bytes, **encryption)
                del csv_upload_bytes
            response = session.client("s3").get_object(Bucket=csv_bucket, Key=csv_key)
            with response["Body"] as stream:
                uploaded_sha256 = hashlib.sha256(stream.read()).hexdigest()
            assert uploaded_sha256 == csv_check["sha256"], "S3 CSV differs from the file checked above."
        except Exception as error:
            raise RuntimeError(f"STOP: CSV upload/readback verification failed at {csv_input_uri}. Pipelines will not be published: {error}") from error
        csv_upload_verified = True
        print("S3 CSV VERIFIED:", csv_input_uri)
        print("SHA-256:", uploaded_sha256, "| Columns:", len(csv_check["columns"]))
    show_progress("Verifying test/inference inputs and publishing code...")
    # Upload and verify each additional file before publishing any pipeline.
    csv_upload_verified = False
    for kind, item in additional_csv_inputs.items():
        source_bucket, source_key = item["uri"].removeprefix("s3://").split("/", 1)
        if item["local_path"] is not None:
            content = item["local_path"].read_bytes()
            assert hashlib.sha256(content).hexdigest() == item["sha256"], f"STOP: {kind} CSV changed; rerun validation."
            kms = infrastructure.get("output_kms_key_arn", "")
            encryption = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms} if kms else {}
            put_immutable(session.client("s3"), source_bucket, source_key, content, **encryption)
        with session.client("s3").get_object(Bucket=source_bucket, Key=source_key)["Body"] as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == item["sha256"], f"STOP: {kind} CSV differs from validation."
        print("S3 CSV VERIFIED:", kind, item["uri"])
    csv_upload_verified = True
    uploaded_source = upload_source(session, s3_root, identity, expected_uri=source_uri)
    source_bucket, source_prefix = uploaded_source.removeprefix("s3://").split("/", 1)
    with session.client("s3").get_object(Bucket=source_bucket, Key=source_prefix.rstrip("/") + "/source.json")["Body"] as stream:
        source_receipt = json.load(stream)
    assert source_receipt["uri"] == source_uri, "STOP: source upload receipt names a different location."
    assert source_receipt["sha256"] == source_uri.rsplit("/sha256-", 1)[1], "STOP: source upload receipt has a different hash."
    assert all(source_receipt.get(key) == value for key, value in identity.items()), "STOP: source upload receipt has a different identity."
    published_cache_sources = publish_cache_sources(session, s3_root, uploaded_source, cache_source_bindings)
    for binding in published_cache_sources:
        print("Step source:", binding["kind"], binding["step"], "→", binding["source_uri"],
              "(retained)" if binding["reused_source"] else "(current publication)")
    verified_source_uri = uploaded_source
    print("SOURCE UPLOAD VERIFIED:", verified_source_uri)
    print("Source SHA-256:", source_receipt["sha256"])
    print("GetData sources:", data_sources)

    published = {}
    assert verified_source_uri == source_uri, "STOP: complete the source upload cell successfully before publishing."
    assert source_location(s3_root, identity) == verified_source_uri, "STOP: local source changed after upload; rerun the notebook."
    assert {(b["kind"], b["step"], b["revision"]) for b in published_cache_sources} == {
        (b["kind"], b["step"], b["revision"]) for b in cache_source_bindings
    }, "STOP: complete source upload and cache bindings before publishing."
    for kind, body in definitions.items():
        saved_source = next((p.get("DefaultValue") for p in body["Parameters"]
                             if p["Name"] == "PipelineSourceUri"), None)
        assert saved_source == verified_source_uri, (
            f"STOP: {kind} PipelineSourceUri differs from the uploaded source. "
            "Restart the kernel and run all cells with the current pipelines/source.py."
        )
    if uses_csv:
        assert csv_upload_verified and csv_checked_settings == csv_settings(), "STOP: complete CSV verification before publishing."
        for kind, body in definitions.items():
            parameters = {p["Name"]: p.get("DefaultValue", "") for p in body["Parameters"]}
            if body["Metadata"]["DataSource"] == "csv_direct":
                expected_csv = (inference_csv_input_uri or csv_input_uri) if kind == "inference" else csv_input_uri
                assert parameters["CsvInputUri"] == expected_csv, "STOP: definition uses another CSV; rerun all cells."
            if kind == "training" and body["Metadata"].get("TestDataSource") == "csv_direct":
                assert parameters["TestCsvInputUri"] == test_csv_input_uri, "STOP: definition uses another test CSV."
    completion = {"deployment": "skipped"}
    if completion_code_uri:
        sys.path.insert(0, str(ROOT))
        from operations.completion.deploy import deploy_completion
        completion = deploy_completion(
            session, namespace, s3_root, completion_code_uri, tracking_uri=tracking_uri,
            tracking_artifact_uris=tracking_artifact_uris,
            registry_prefix=model_package_group or registry_group_prefix, kms_key=infrastructure.get("output_kms_key_arn", ""),
            permissions_boundary_arn=completion_permissions_boundary_arn,
        )
    else:
        print("Completion service deployment skipped. Existing services are unchanged.")
        print("Jobs and Model Registry work independently. Automatic final S3/MLflow status requires the Platform service.")
    published = {}
    for kind, body in definitions.items():
        name = f"{namespace}-{kind}"
        show_progress(f"Publishing {name}...")
        request = {
            "PipelineName": name, "RoleArn": pipeline_role_arn,
            "PipelineDefinition": json.dumps(body),
        }
        try:
            existing = sm.describe_pipeline(PipelineName=name)
        except sm.exceptions.ClientError as error:
            if "does not exist" not in error.response["Error"]["Message"]:
                raise
            response = sm.create_pipeline(**request, Tags=[
                {"Key": "Deployment", "Value": namespace},
                {"Key": "SourceDeploy", "Value": identity["mode"]},
            ])
            version = 1  # The initial version of a newly created pipeline.
        else:
            tags = {t["Key"]: t["Value"] for t in sm.list_tags(ResourceArn=existing["PipelineArn"])["Tags"]}
            if tags.get("Deployment") != namespace or tags.get("SourceDeploy") != identity["mode"]:
                raise ValueError(f"{name} belongs to another deployment.")
            response = sm.update_pipeline(**request)
            version = response["PipelineVersionId"]
        actual = sm.describe_pipeline(PipelineName=name, PipelineVersionId=version)
        assert json.loads(actual["PipelineDefinition"]) == body, "Published definition differs; inspect before executing."
        assert actual["RoleArn"] == pipeline_role_arn, "Saved pipeline role differs; inspect before executing."
        published[kind] = {"name": name, "arn": response["PipelineArn"], "version": version}
        print("Published:", name, "| version:", version, "| saved role:", actual["RoleArn"])

    publication = {
        "namespace": namespace, "registry_stage": "Development", "pipelines": published, "models": model_ids,
        "tracking": tracking, "tracking_check": tracking_check, "completion": completion, "monitoring": monitoring, "job_role_arn": job_role_arn, "infrastructure": infrastructure,
        "source": identity, "source_uri": source_uri, "image": image_uri,
        "s3_root": s3_root, "batch_runtime": batch_runtime, "region": session.region_name,
        "environment": environment, "model_package_group": model_package_group, "registry_group_prefix": registry_group_prefix, "registry_groups": registry_groups, "include_hpo": include_hpo,
        "feature_store_config_uri": feature_store_config_uri, "universe_data_root": universe_data_root,
        "data_source": data_source, "inference_data_source": inference_data_source, "csv_input_uri": csv_input_uri,
        "test_csv_input_uri": test_csv_input_uri, "inference_csv_input_uri": inference_csv_input_uri,
        "step_cache_revisions": step_cache_revisions, "cache_sources": published_cache_sources,
        "model_code_source": definitions["inference"]["Metadata"]["ModelCodeSource"],
        "csv_check": csv_check,
        "data_preview": data_preview,
        "data_sha256": {
            **({"training": csv_check["sha256"]} if csv_check else {}),
            **{kind: item["sha256"] for kind, item in additional_csv_inputs.items()},
        },
    }
    if setup_receipt is not None:
        publication["setup"] = setup_receipt
    path = WORK / "publication.json"
    _atomic_json(path, publication)

    # Save verified inputs only after every pipeline has published successfully.
    # Reload to preserve unrelated settings edited while this notebook was running.
    saved_config = json.loads(CONFIG_PATH.read_text())
    _config_before_save = saved_config.copy()
    for _key in ("csv_input", "test_csv_input", "inference_csv_input"):
        if saved_config.get(_key, "") != notebook_config.get(_key, ""):
            raise RuntimeError(f"{_key} changed during publication; see publication.json and reconcile config explicitly.")
    for key, uri in (("csv_input", csv_input_uri),
                     ("test_csv_input", test_csv_input_uri),
                     ("inference_csv_input", inference_csv_input_uri)):
        if uri:
            saved_config[key] = uri
            print(key + ":", uri)
    if saved_config != _config_before_save:
        _atomic_json(CONFIG_PATH, saved_config)
    print("Saved published S3 inputs:", CONFIG_PATH)
    print("Publication complete:", path.resolve())
    print("Training (including held-out testing) and inference pipelines are ready in Studio.")
    print("ModelId dropdown:", model_ids)
    print("Train here, then approve both Development stage status and model approval for the exact version.")
    print("For an already approved model, start Processing inference with its ModelId and scoring CsvInputUri.")
    if batch_runtime == "both":
        print("InferenceRuntime dropdown: processing (default), transform. Each execution runs only its chosen branch.")
    print("PipelineSourceUri in both pipelines:", source_uri)
    print("Compare that parameter with Uploaded source above; CsvInputUri is a separate data input.")
    if model_code_source == "artifact":
        print("ModelCodeSource defaults to artifact: saved model code. Select pipeline on a Development execution to test edits.")
        print("To test published Processing edits: select ModelCodeSource=pipeline when starting inference; no training needed.")
    else:
        print("Processing uses the source bindings published for this pipeline version.")
        print("Unchanged cached steps can retain an older source snapshot; see Step source above.")
        print("Native Transform uses the selected model's saved code for its entire route.")
    print("Batch inference resolves the newest approved Development version in this namespace, or the exact ModelPackageArn.")
    if uses_csv:
        print("Use the newly printed pipeline version and this CsvInputUri:", csv_input_uri)
        print("A previous execution/version or a Studio parameter override can select a different CSV.")

    return {**publication, "publication_path": str(path.resolve())}
