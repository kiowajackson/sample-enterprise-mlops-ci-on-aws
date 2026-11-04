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
               "runs", {"Get": "Execution.PipelineExecutionId"})
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
    preprocess = deepcopy(get_data)
    preprocess["Name"] = "Preprocess"
    preprocess["DependsOn"] = ["GetData", "FeatureSelection"]
    for step, action, name, suffix, inputs in (
        (selection, "select-features", "selection", "-select", [input_from("GetData", "raw")]),
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
    """Use approved-model retrieval with test data, test preprocessing and test analysis."""
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
    report = join({"Get": "Steps.Postprocess.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri"},
                  "evaluation.json")
    body["Steps"].append({"Name": "QualityGate", "Type": "Condition", "DependsOn": ["Postprocess"],
        "Arguments": {"Conditions": [{"Type": "Equals", "LeftValue": {
            "Std:JsonGet": {"S3Uri": report, "Path": "quality.passed"}}, "RightValue": True}],
            "IfSteps": [], "ElseSteps": [{"Name": "QualityFailed", "Type": "Fail",
                "Arguments": {"ErrorMessage": "Test quality thresholds failed; inspect Postprocess evaluation.json."}}]}})
    body["Metadata"].update(PipelineKind="testing", EvaluationInputContract="separately-selected-test-data")


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
        destination = join({"Get": "Parameters.OutputDataRoot"}, {"Get": "Parameters.ModelId"}, "runs",
                           {"Get": "Execution.PipelineExecutionId"}, "steps", step["Name"])
        reference = {"Get": f"Steps.{step['Name']}.ProcessingOutputConfig.Outputs['{first['OutputName']}'].S3Output.S3Uri"}
        for output in outputs:
            relative = Path(output["S3Output"]["LocalPath"]).relative_to(root).as_posix()
            before = {"Get": f"Steps.{step['Name']}.ProcessingOutputConfig.Outputs['{output['OutputName']}'].S3Output.S3Uri"}
            after = reference if relative == "." else join(reference, relative)
            replacements.append((before, after))
            replacements.append((output["S3Output"]["S3Uri"],
                                 destination if relative == "." else join(destination, relative)))
        first["S3Output"].update(LocalPath=str(root), S3Uri=join(
            {"Get": "Parameters.OutputDataRoot"}, {"Get": "Parameters.ModelId"}, "runs",
            {"Get": "Execution.PipelineExecutionId"}, "steps", step["Name"]))
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
