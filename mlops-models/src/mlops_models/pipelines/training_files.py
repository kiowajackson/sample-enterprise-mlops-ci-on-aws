"""Wire the source-deployed training pipeline through explicit CSV file contracts."""

from copy import deepcopy
from pathlib import Path
from mlops_skeleton.pipelines import walk_steps


def configure_file_training(body):
    """Replace Prepare with GetData → FeatureSelection → Preprocess; Train consumes data and selection.

    Model hooks own file contents and the internals of the model artifact. The
    orchestration only supplies input mounts, output locations and dependencies.
    """
    from .source import join

    original = body["Steps"][0]
    raw_output = next(item for item in original["Arguments"]["ProcessingOutputConfig"]["Outputs"]
                      if item["OutputName"] == "raw")
    run = join({"Get": "Parameters.OutputDataRoot"}, {"Get": "Parameters.ModelId"},
               {"Get": "Parameters.PipelineName"}, "runs", {"Get": "Execution.PipelineExecutionId"})
    source = next(item for item in original["Arguments"]["ProcessingInputs"] if item["InputName"] == "source")

    def output_of(step, name):
        return {"Get": f"Steps.{step}.ProcessingOutputConfig.Outputs['{name}'].S3Output.S3Uri"}

    def input_from(step, name):
        item = deepcopy(source)
        item["InputName"] = name
        item["S3Input"].update(S3Uri=output_of(step, name), LocalPath=f"/opt/ml/processing/{name}")
        return item

    get_data = deepcopy(original)
    get_data["Name"] = "GetData"
    args = get_data["Arguments"]
    args["ProcessingJobName"] = args["ProcessingJobName"].removesuffix("-prep") + "-get"
    args["AppSpecification"]["ContainerEntrypoint"][-1] = "get-data"
    args["ProcessingOutputConfig"]["Outputs"] = [deepcopy(raw_output)]

    selection = deepcopy(get_data)
    selection["Name"] = "FeatureSelection"
    selection["DependsOn"] = ["GetData"]
    selection["Arguments"]["Environment"]["MLOPS_LOG_ARTIFACTS"] = "1"
    selection["Arguments"]["StoppingCondition"]["MaxRuntimeInSeconds"] = 3600
    preprocess = deepcopy(get_data)
    preprocess["Name"] = "Preprocess"
    preprocess["DependsOn"] = ["GetData", "FeatureSelection"]
    for step, action, name, suffix, inputs in (
        (selection, "select-features", "selection", "-sel", [input_from("GetData", "raw")]),
        (preprocess, "preprocess-training", "prepared", "-prep", [input_from("GetData", "raw"),
                                                                input_from("FeatureSelection", "selection")]),
    ):
        args = step["Arguments"]
        args["ProcessingJobName"] = args["ProcessingJobName"].removesuffix("-get") + suffix
        args["AppSpecification"]["ContainerEntrypoint"][-1] = action
        args["ProcessingInputs"] = [*inputs, deepcopy(source)]
        output = args["ProcessingOutputConfig"]["Outputs"][0]
        output["OutputName"] = name
        output["S3Output"].update(S3Uri=join(run, "data", name),
                                 LocalPath=f"/opt/ml/processing/output/{name}")

    body["Steps"] = [get_data, selection, preprocess, *body["Steps"][1:]]
    for step in walk_steps(body["Steps"]):
        if "Prepare" in step.get("DependsOn", []):
            step["DependsOn"] = ["Preprocess" if name == "Prepare" else name for name in step["DependsOn"]]
        args = step["Arguments"]
        if step["Type"] in {"Training", "Tuning"}:
            job = args if step["Type"] == "Training" else args["TrainingJobDefinition"]
            data_channels = [{"ChannelName": name, "DataSource": {"S3DataSource": {
                "S3Uri": output_of(producer, name), "S3DataType": "S3Prefix",
                "S3DataDistributionType": "FullyReplicated"}}}
                for name, producer in (("prepared", "Preprocess"), ("selection", "FeatureSelection"))]
            job["InputDataConfig"] = [*data_channels,
                *[channel for channel in job["InputDataConfig"] if channel["ChannelName"] == "source"]]
            job["Environment"]["MLOPS_TRAINING_INPUT"] = "prepared-directory"
    body["Metadata"]["TrainingInputContract"] = "prepared-directory-and-selection"


def configure_testing(body):
    """Use approved-model retrieval and finish testing with the model's postprocessing."""
    from .source import replace, join
    body.update(replace(body, [("get-inference-data", "get-test-data"),
                               ("preprocess", "preprocess-test"),
                               ("postprocess", "postprocess-test")]))
    for step in walk_steps(body["Steps"]):
        if step["Name"] == "ResolveModel":
            step["Arguments"]["AppSpecification"]["ContainerArguments"].append("--resolve-training-data")
        if step["Name"] in {"GetData", "Preprocess", "Score", "Postprocess"}:
            step["Arguments"]["Environment"]["MLOPS_ANALYSIS_MODE"] = "prediction_test"
        if step["Name"] == "Postprocess":
            args = step["Arguments"]
            resolver = next(item for item in body["Steps"] if item["Name"] == "ResolveModel")
            decision = resolver["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            training = deepcopy(next(item for item in args["ProcessingInputs"] if item["InputName"] == "raw"))
            training["InputName"] = "training-raw"
            training["S3Input"].update(
                S3Uri={"Std:JsonGet": {"S3Uri": join(decision, "package.json"), "Path": "training_raw_uri"}},
                LocalPath="/opt/ml/processing/training-raw")
            args["ProcessingInputs"].append(training)
            args["AppSpecification"]["ContainerArguments"].extend(
                ["--train-data", "/opt/ml/processing/training-raw"])
            output = step["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]
            output["OutputName"] = "evaluation"
            output["S3Output"]["LocalPath"] = "/opt/ml/processing/evaluation"
    body["Metadata"].update(PipelineKind="testing", EvaluationInputContract="separately-selected-test-data")


def combine_training_testing(training, testing):
    """Test each newly trained checkpoint before registering it, without approval.

    Reuse the same four Processing contracts as standalone testing. Ordinary and
    HPO branches each mount their own checkpoint; neither reads the other branch.
    The test CSV/delivery is explicit and independent of the training input.
    """
    from .source import join, replace

    # Normalize the test outputs while names are still distinct from training.
    # Renaming below also renames their S3 step folders and cached references.
    configure_output_roots(testing)
    test_steps = {step["Name"]: step for step in testing["Steps"]}
    resolver_output = test_steps["ResolveModel"]["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]

    def selected(field):
        return {"Std:JsonGet": {"S3Uri": join(resolver_output, "package.json"), "Path": field}}

    # Keep test-data controls separate even when both inputs use Feature Store.
    data_parameters = {
        "CsvInputUri": "TestCsvInputUri",
        "FeatureStoreConfigUri": "TestFeatureStoreConfigUri",
        "FeatureStoreDeliveryId": "TestFeatureStoreDeliveryId",
        "UniverseDataRoot": "TestUniverseDataRoot",
    }
    existing = {item["Name"]: item for item in training["Parameters"]}
    data_replacements = []
    for parameter in testing["Parameters"]:
        name = parameter["Name"]
        if name in data_parameters:
            new = deepcopy(parameter)
            new["Name"] = data_parameters[name]
            new["Description"] = "Held-out test data; separate from the training input."
            training["Parameters"].append(new)
            data_replacements.append(({"Get": "Parameters." + name},
                                      {"Get": "Parameters." + new["Name"]}))
        elif name not in existing and name not in {"ModelCodeSource", "ModelMetadataUri", "ModelPackageArn"}:
            training["Parameters"].append(deepcopy(parameter))

    def add_test_steps(steps):
        result = []
        for step in steps:
            if step["Type"] == "Condition":
                for branch in ("IfSteps", "ElseSteps"):
                    step["Arguments"][branch] = add_test_steps(step["Arguments"][branch])
            if step["Name"] in {"RegisterModel", "RegisterTunedModel"}:
                suffix = "Tuned" if step["Name"] == "RegisterTunedModel" else ""
                names = {"GetData": "GetTestData" + suffix,
                         "Preprocess": "PreprocessTest" + suffix,
                         "Score": "PredictTest" + suffix,
                         "Postprocess": "PostprocessTest" + suffix}

                def rename(value):
                    if isinstance(value, dict):
                        return {key: rename(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [rename(item) for item in value]
                    if isinstance(value, str):
                        if value in names:
                            return names[value]
                        for old, new in names.items():
                            if value.startswith(f"Steps.{old}."):
                                return f"Steps.{new}." + value[len(f"Steps.{old}."):]
                    return value

                args = step["Arguments"]
                checkpoint = args["Environment"]["CHECKPOINT_URI"]
                chain = rename([deepcopy(test_steps[name]) for name in names])
                chain = replace(chain, data_replacements + [
                    (selected("checkpoint"), checkpoint),
                    (selected("image"), args["Environment"]["IMAGE_URI"]),
                    (selected("source_uri"), {"Get": "Parameters.PipelineSourceUri"}),
                    (selected("model_metadata"), ""),
                    (selected("model_metadata_sha256"), ""),
                    (selected("package"), ""),
                    (selected("training_raw_uri"),
                     {"Get": "Steps.GetData.ProcessingOutputConfig.Outputs['raw'].S3Output.S3Uri"}),
                    ({"Get": "Parameters.ModelCodeSource"}, "pipeline"),
                ])
                job_suffixes = ("get", "pre", "scr", "post")
                for item, job_suffix in zip(chain, job_suffixes):
                    item["DependsOn"] = [name for name in item.get("DependsOn", []) if name != "ResolveModel"]
                    job = item["Arguments"]
                    job["Environment"]["MLOPS_MODEL_CODE_SOURCE"] = "pipeline"
                    job["Environment"].pop("MLOPS_MODEL_PACKAGE", None)
                    job["Environment"].pop("MLOPS_MODEL_METADATA_URI", None)
                    job["Environment"].pop("MLOPS_MODEL_METADATA_SHA256", None)
                    prefix = args["ProcessingJobName"].rsplit("-", 1)[0]
                    job["ProcessingJobName"] = prefix + ("-h" if suffix else "-t") + job_suffix
                chain[0]["DependsOn"] = list(step["DependsOn"])
                chain[-1]["DependsOn"] = [names["Score"]]
                step["DependsOn"] = [names["Postprocess"]]
                result.extend(chain)
            result.append(step)
        return result

    training["Steps"] = add_test_steps(training["Steps"])
    # Keep descriptive contract prose out of the 20-entry pipeline metadata map.
    # Tracking adds one entry; retain the fields used to identify this workflow.
    training["Metadata"].update(
        TrainingWorkflow="train-test-register",
        TestDataSource=testing["Metadata"]["DataSource"])


def configure_output_roots(body):
    """Publish each Processing step's whole output directory under one S3 prefix."""
    from .source import join, replace
    from mlops_models.runtime.job_paths import job_paths
    replacements = []
    for step in walk_steps(body["Steps"]):
        if step["Type"] != "Processing" or step["Name"] in {"RegisterModel", "RegisterTunedModel"}:
            continue
        args = step["Arguments"]
        action = args["AppSpecification"]["ContainerEntrypoint"][-1]
        root = Path(job_paths(action)[1])
        outputs = args["ProcessingOutputConfig"]["Outputs"]
        first = outputs[0]
        destination = join({"Get": "Parameters.OutputDataRoot"}, {"Get": "Parameters.ModelId"},
                           {"Get": "Parameters.PipelineName"}, "runs",
                           {"Get": "Execution.PipelineExecutionId"}, "steps", step["Name"])
        reference = {"Get": f"Steps.{step['Name']}.ProcessingOutputConfig.Outputs['{first['OutputName']}'].S3Output.S3Uri"}
        for output in outputs:
            relative = Path(output["S3Output"]["LocalPath"]).relative_to(root).as_posix()
            before = {"Get": f"Steps.{step['Name']}.ProcessingOutputConfig.Outputs['{output['OutputName']}'].S3Output.S3Uri"}
            after = reference if relative == "." else join(reference, relative)
            replacements.append((before, after))
            replacements.append((output["S3Output"]["S3Uri"],
                                 destination if relative == "." else join(destination, relative)))
        first["S3Output"].update(LocalPath=str(root), S3Uri=destination)
        args["ProcessingOutputConfig"]["Outputs"] = [first]
    body.update(replace(body, replacements))


def configure_preparation_cache(body, revision):
    """Reuse successful CSV preparation; Train and downstream steps still execute."""
    for step in body["Steps"]:
        if step["Name"] not in {"GetData", "FeatureSelection", "Preprocess"}:
            continue
        environment = step["Arguments"]["Environment"]
        # These are restored by load_source.py from the job output and source identity.
        for name in ("PIPELINE_EXECUTION_ID", "SOURCE_COMMIT", "SOURCE_DIRTY"):
            environment.pop(name, None)
        environment["MLOPS_PREPARATION_REVISION"] = (
            revision[step["Name"]] if isinstance(revision, dict) else revision)
        step["CacheConfig"] = {"Enabled": True, "ExpireAfter": "PT24H"}
