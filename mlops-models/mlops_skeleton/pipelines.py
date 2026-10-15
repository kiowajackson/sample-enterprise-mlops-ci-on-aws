"""Render native SageMaker pipeline definitions and job settings."""

import json
import math
from pathlib import Path
from string import Template

from .names import job_prefix, tuning_prefix, validate_job_names

PACKAGE = Path(__file__).resolve().parents[1] / "src/mlops_models"


def load_definition(model_id, kind, *, definition_path=None, infrastructure=None,
                    tuning_config=None, include_hpo=False, **values):
    """Take a use case, pipeline kind and deployment values; return the rendered definition."""
    values = {"source_commit": "uncommitted", "source_repository": "",
              "source_dirty": "true", "image_digest": "", **values}
    values["job_prefix"] = job_prefix(values["namespace"], model_id, kind)
    values["tuning_prefix"] = tuning_prefix(values["namespace"], model_id)
    path = Path(definition_path) if definition_path else PACKAGE / "models" / model_id / kind / "definition.json"
    if not definition_path and not path.is_file():
        path = PACKAGE / "model_base" / kind / "definition.json"

    # Substitute individual JSON strings so paths and names remain correctly escaped.
    def substitute(value):
        """Return a JSON value with deployment placeholders replaced inside its strings."""
        if isinstance(value, str):
            return Template(value).substitute(model_id=model_id, **values)
        if isinstance(value, list):
            return [substitute(item) for item in value]
        if isinstance(value, dict):
            return {key: substitute(item) for key, item in value.items()}
        return value

    definition = substitute(json.loads(path.read_text()))
    if kind == "training" and not include_hpo:
        mode = next(step for step in definition["Steps"] if step["Name"] == "TrainingMode")
        ordinary_steps = mode["Arguments"]["ElseSteps"]
        ordinary_steps[0]["DependsOn"] = mode["DependsOn"]
        definition["Steps"] = [
            child
            for step in definition["Steps"]
            if step["Name"] != "QualityGateTuned"
            for child in (ordinary_steps if step["Name"] == "TrainingMode" else [step])
        ]
        definition["Parameters"] = [
            item for item in definition["Parameters"]
            if item["Name"] not in {"UseHPO", "TuningTrials", "TuningParallel",
                                    "TuningSeconds", "TrialSeconds", "TuningSeed"}
        ]
    if any(step["Type"] == "Tuning" for step in walk_steps(definition["Steps"])):
        config = tuning_config if tuning_config is not None else json.loads(
            (PACKAGE / "models" / model_id / "parameters.json").read_text())
        configure_tuning(definition, config)
        validate_tuning(definition, config)
    apply_job_settings(definition, infrastructure or {})
    validate_job_names(definition)
    return definition


def walk_steps(steps):
    """Yield native steps including those inside either conditional branch."""
    for step in steps:
        yield step
        if step["Type"] == "Condition":
            for branch in ("IfSteps", "ElseSteps"):
                yield from walk_steps(step["Arguments"].get(branch, []))


def job_vpc(settings):
    """Take environment settings and return the paired subnet and security-group lists."""
    network = settings.get("network", {})
    vpc = {"Subnets": network.get("subnet_ids", []),
           "SecurityGroupIds": network.get("security_group_ids", [])}
    if bool(vpc["Subnets"]) != bool(vpc["SecurityGroupIds"]):
        raise ValueError("Configure both subnet_ids and security_group_ids for VPC jobs.")
    return vpc


def apply_job_settings(definition, settings):
    """Apply explicitly configured network and encryption settings to the appropriate job APIs."""
    network, vpc = settings.get("network", {}), job_vpc(settings)
    for step in definition["Steps"]:
        arguments, kind = step["Arguments"], step["Type"]
        if kind == "Tuning":
            apply_job_settings({"Steps": [{"Name": step["Name"], "Type": "Training",
                                          "Arguments": arguments["TrainingJobDefinition"]}]}, settings)
            continue
        if kind == "Condition":
            for branch in ("IfSteps", "ElseSteps"):
                apply_job_settings({"Steps": arguments.get(branch, [])}, settings)
            continue
        if network and kind in {"Processing", "Training", "Model"}:
            target = arguments.setdefault("NetworkConfig", {}) if kind == "Processing" else arguments
            target["EnableNetworkIsolation"] = (
                False if step["Name"] in {"ValidatePackage", "ResolveModel"} else network.get("isolation", False)
            )
            if vpc["Subnets"]:
                target["VpcConfig"] = vpc
        if key := settings.get("output_kms_key_arn"):
            field = {"Processing": "ProcessingOutputConfig", "Training": "OutputDataConfig",
                     "Transform": "TransformOutput"}.get(kind)
            if field and field in arguments:
                arguments[field]["KmsKeyId"] = key
        if key := settings.get("volume_kms_key_arn"):
            if kind == "Processing":
                arguments["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"] = key
            elif kind in {"Training", "Transform"}:
                field = "ResourceConfig" if kind == "Training" else "TransformResources"
                arguments[field]["VolumeKmsKeyId"] = key


def configure_tuning(definition, config):
    """Apply a model's search settings to the native tuning fields and pipeline parameter defaults."""
    settings = config["tuning"]
    objective = settings["objective"]
    name = objective["name"]
    if objective["direction"] not in {"Minimize", "Maximize"}:
        raise ValueError("tuning.objective.direction must be Minimize or Maximize.")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", name):
        raise ValueError("Use a metric name containing letters, digits, ':', '.', '_' or '-'.")
    ranges = {}
    for key, values in settings["parameters"].items():
        kind = values["type"]
        if kind not in {"Continuous", "Integer", "Categorical"}:
            raise ValueError("Tuning parameter type must be Continuous, Integer or Categorical.")
        item = {"Name": key}
        if kind == "Categorical":
            item["Values"] = [str(value) for value in values["values"]]
            if not 1 <= len(item["Values"]) <= 30 or any(len(value) > 256 for value in item["Values"]):
                raise ValueError("Categorical searches allow 1–30 values of at most 256 characters.")
        else:
            lower, upper = values["min"], values["max"]
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError("Numeric tuning bounds must be finite and increasing.")
            if kind == "Integer" and (type(lower) is not int or type(upper) is not int):
                raise ValueError("Integer tuning bounds must be integers.")
            item.update(MinValue=str(lower), MaxValue=str(upper), ScalingType=values.get("scaling", "Auto"))
        ranges.setdefault(kind + "ParameterRanges", []).append(item)
    if not 1 <= len(settings["parameters"]) <= 30:
        raise ValueError("Configure between 1 and 30 tuned parameters.")
    if set(settings.get("fixed", {})) & set(settings["parameters"]):
        raise ValueError("A parameter cannot be both fixed and tuned.")
    for step in walk_steps(definition["Steps"]):
        if step["Type"] != "Tuning":
            continue
        args = step["Arguments"]
        args["HyperParameterTuningJobConfig"].update(
            Strategy=settings.get("strategy", "Random"), ParameterRanges=ranges,
            HyperParameterTuningJobObjective={"MetricName": name, "Type": objective["direction"]})
        job = args["TrainingJobDefinition"]
        job["StaticHyperParameters"] = {key: value if isinstance(value, str) else json.dumps(value)
                                         for key, value in settings.get("fixed", {}).items()}
        job["Environment"]["TUNING_OBJECTIVE"] = name
        job["AlgorithmSpecification"]["MetricDefinitions"] = [
            {"Name": name, "Regex": re.escape(name) + r"=([0-9.eE+-]+);"}]
    defaults = {"TuningTrials": settings.get("max_jobs", 3),
                "TuningParallel": settings.get("max_parallel_jobs", 1),
                "TuningSeconds": settings.get("max_seconds", 1800),
                "TrialSeconds": settings.get("trial_seconds", 600), "TuningSeed": config.get("seed", 42)}
    for parameter in definition["Parameters"]:
        if parameter["Name"] in defaults:
            parameter["DefaultValue"] = defaults[parameter["Name"]]


def validate_tuning(definition, config, parameters=None):
    """Reject unbounded sample searches or test-data channels before publishing or starting a tuning run."""
    values = {item["Name"]: item.get("DefaultValue") for item in definition["Parameters"]}
    values.update(parameters or {})
    jobs, parallel = int(values["TuningTrials"]), int(values["TuningParallel"])
    total, trial = int(values["TuningSeconds"]), int(values["TrialSeconds"])
    if not 0 <= int(values["TuningSeed"]) <= 2147483647:
        raise ValueError("TuningSeed must be a nonnegative 32-bit integer.")
    if not (1 <= jobs <= 20 and 1 <= parallel <= min(jobs, 4)
            and 120 <= total <= 7200 and 60 <= trial <= min(total, 1800)):
        raise ValueError("Use 1–20 trials, at most 4 parallel, 120–7200 total seconds and 60–1800 seconds per trial.")
    for step in walk_steps(definition["Steps"]):
        if step["Type"] == "Tuning":
            channels = step["Arguments"]["TrainingJobDefinition"]["InputDataConfig"]
            if {item["ChannelName"] for item in channels} != {"train", "validation"}:
                raise ValueError("Tuning receives training and validation data only, never test rows.")
