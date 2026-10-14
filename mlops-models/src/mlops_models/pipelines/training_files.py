"""Wire the source-deployed training pipeline through explicit CSV file contracts."""

from copy import deepcopy
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
        if step["Name"] in {"Evaluate", "EvaluateTuned"}:
            args["ProcessingInputs"] = [item for item in args["ProcessingInputs"]
                                        if item["S3Input"]["LocalPath"] != "/opt/ml/processing/test"]
            args["Environment"]["MLOPS_EVALUATION_INPUT"] = "model-artifact"
    body["Metadata"]["TrainingInputContract"] = "prepared-directory-and-selection"


def configure_preparation_cache(body, revision):
    """Reuse successful CSV preparation; Train and downstream steps still execute."""
    for step in body["Steps"]:
        if step["Name"] not in {"GetData", "FeatureSelection", "Preprocess"}:
            continue
        environment = step["Arguments"]["Environment"]
        # These are restored by load_source.py from the job output and source identity.
        for name in ("PIPELINE_EXECUTION_ID", "SOURCE_COMMIT", "SOURCE_DIRTY"):
            environment.pop(name, None)
        environment["MLOPS_PREPARATION_REVISION"] = revision
        step["CacheConfig"] = {"Enabled": True, "ExpireAfter": "PT24H"}
