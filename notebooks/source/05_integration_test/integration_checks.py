"""Run the ordinary Development training/inference paths and retain their evidence.

The notebook presents the results. This module submits, waits and checks outputs;
it neither deploys pipelines nor changes model code.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

from mlops_models.registry.model_package import check_model_package


PRELIMINARY = "Preliminary models with validation"
FINAL = "Final model with all data"
PHASES = ("preliminary", "full", "prediction_test", "prediction_inference")
TERMINAL = {"Succeeded", "Failed", "Stopped"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _string(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def _location(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a complete S3 URI, got {uri!r}.")
    return parsed.netloc, parsed.path.lstrip("/")


def _steps(steps):
    for step in steps:
        yield step
        if step["Type"] == "Condition":
            for branch in ("IfSteps", "ElseSteps"):
                yield from _steps(step["Arguments"].get(branch, []))


def _value(expression, parameters):
    if isinstance(expression, str):
        return expression
    if "Get" in expression and expression["Get"].startswith("Parameters."):
        return parameters[expression["Get"].split(".", 1)[1]]
    if "Std:Join" in expression:
        join = expression["Std:Join"]
        return join["On"].join(_value(v, parameters) for v in join["Values"])
    raise ValueError(f"Unsupported Registry group expression: {expression!r}")


class IntegrationSuite:
    """One resumable test label, with parallel executions inside each phase."""

    def __init__(self, root, publication, *, run_label, model_ids=(), csv_inputs=None,
                 pipelines=None, namespace="", region="", model_code_source="artifact",
                 poll_seconds=30, timeout_minutes=120, session=None, on_update=None):
        self.root = Path(root)
        self.publication = publication
        self.namespace = namespace or publication.get("namespace", "")
        for name, value in (("namespace", self.namespace), ("run_label", run_label)):
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,78}[A-Za-z0-9])?", value):
                raise ValueError(f"Set {name} to letters/digits/hyphens, starting and ending with a letter/digit.")
        if not 1 <= poll_seconds <= 60 or timeout_minutes <= 0:
            raise ValueError("Use poll_seconds 1–60 and a positive timeout_minutes.")
        if model_code_source not in {"artifact", "pipeline"}:
            raise ValueError("model_code_source must be artifact or pipeline.")
        targets = deepcopy(publication.get("pipelines", {}))
        targets.update(pipelines or {})
        if not all(targets.get(kind) for kind in ("training", "inference")):
            raise ValueError("Run notebook 02 first, or provide both pipeline names/ARNs in PIPELINES.")
        self.targets = {
            kind: ({"name": target} if isinstance(target, str) else target)
            for kind, target in targets.items() if kind in {"training", "inference"}
        }
        self.settings = {"namespace": self.namespace, "run_label": run_label,
                         "model_ids": list(model_ids), "csv_inputs": csv_inputs or {},
                         "model_code_source": model_code_source,
                         "targets": {k: v.get("arn") or v.get("name") for k, v in self.targets.items()}}
        self.directory = self.root / ".runs/integration-tests" / self.namespace / run_label
        self.path = self.directory / "state.json"
        self.state = json.loads(self.path.read_text()) if self.path.is_file() else {
            "settings": self.settings, "executions": {}, "checks": {}, "status": "Not started",
        }
        if self.state["settings"] != self.settings:
            raise ValueError("This RUN_LABEL has different saved inputs. Restore those inputs or use a new label.")
        if session is None:
            import boto3
            session = boto3.Session(region_name=region or publication.get("region") or None)
        from botocore.config import Config
        options = Config(connect_timeout=10, read_timeout=30,
                         retries={"mode": "standard", "total_max_attempts": 3})
        self.sm = session.client("sagemaker", config=options)
        self.s3 = session.client("s3", config=options)
        self.region = session.region_name
        self.poll_seconds, self.timeout_minutes = poll_seconds, timeout_minutes
        self.on_update = on_update
        self.prepared = False

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix("." + uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, default=str) + "\n")
        temporary.replace(self.path)

    def prepare(self):
        """Read and pin both definitions, validate the matrix; start no jobs."""
        pinned = {}
        for kind, target in self.targets.items():
            name = target.get("arn") or target["name"]
            previous = self.state.get("pipelines", {}).get(kind, {})
            version = previous.get("version") or target.get("version")
            if version is None:
                versions = self.sm.list_pipeline_versions(
                    PipelineName=name, SortOrder="Descending", MaxResults=1)
                version = versions["PipelineVersionSummaries"][0]["PipelineVersionId"]
            described = self.sm.describe_pipeline(PipelineName=name, PipelineVersionId=int(version))
            body = json.loads(described["PipelineDefinition"])
            meta = body.get("Metadata", {})
            for field, expected in (("DeploymentNamespace", self.namespace), ("RegistryStage", "Development")):
                if meta.get(field) != expected:
                    raise ValueError(f"{kind}: {field}={meta.get(field)!r}, expected {expected!r}.")
            if kind == "inference" and meta.get("BatchRuntime") != "processing":
                raise ValueError("This first integration suite covers Processing inference; deploy that runtime.")
            required = ({"Train", "Evaluate", "RegisterModel"} if kind == "training"
                        else {"ResolveModel", "Score", "Postprocess"})
            if not required.issubset(s["Name"] for s in _steps(body["Steps"])):
                raise ValueError(f"{kind} definition does not contain the ordinary {sorted(required)} steps.")
            pinned[kind] = {"arn": described["PipelineArn"], "version": int(version), "definition": body}
        if pinned["training"]["arn"].split(":")[:5] != pinned["inference"]["arn"].split(":")[:5]:
            raise ValueError("Training and inference must be in the same account and Region.")
        if pinned["training"]["arn"].split(":")[3] != self.region:
            raise ValueError("The client Region does not match the pipeline Region.")
        environments = {p["definition"]["Metadata"].get("Environment") for p in pinned.values()}
        if len(environments) != 1 or not next(iter(environments)):
            raise ValueError("Both pipelines must declare the same nonempty Environment.")
        self.pipelines = pinned
        available = self._declared("training")["ModelId"].get("EnumValues", [])
        self.models = self.settings["model_ids"] or available
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Select at least one distinct ModelId.")
        if set(self.settings["csv_inputs"]) - set(self.models):
            raise ValueError("CSV_INPUTS contains a model not selected in MODEL_IDS.")
        for model in self.models:
            for phase in PHASES:
                self._parameters(model, phase, metadata_uri="s3://placeholder/metadata.json",
                                 package_arn="pending-package")
        if self.state.get("pipelines") and self.state["pipelines"] != pinned:
            raise ValueError("A pinned pipeline definition changed. Use a new RUN_LABEL.")
        self.state["pipelines"] = pinned
        self.state["models"] = self.models
        self._save()
        self.prepared = True
        return [{"model": model, "phase": phase,
                 "input": self._csv(model, phase) or "published default",
                 "metadata/model": ("none" if phase == "preliminary" else
                                    "this test's preliminary metadata" if phase == "full" else
                                    "this test's approved full model")}
                for phase in PHASES for model in self.models]

    def _declared(self, kind):
        return {p["Name"]: p for p in self.pipelines[kind]["definition"]["Parameters"]}

    def _csv(self, model, phase):
        values = self.settings["csv_inputs"].get(model, {})
        unknown = set(values) - {"training", "prediction_test", "prediction_inference"}
        if unknown:
            raise ValueError(f"{model}: unknown CSV input names: {sorted(unknown)}")
        return values.get(phase, values.get("training", "")) if phase.startswith("prediction") else values.get("training", "")

    def _parameters(self, model, phase, *, metadata_uri="", package_arn=""):
        kind = "inference" if phase.startswith("prediction") else "training"
        declared = self._declared(kind)
        overrides = {"ModelId": model}
        if kind == "training":
            overrides.update(TrainingMode=PRELIMINARY if phase == "preliminary" else FINAL,
                             ModelMetadataUri="none" if phase == "preliminary" else metadata_uri)
            if "UseHPO" in declared:
                overrides["UseHPO"] = False
        else:
            overrides.update(ModelPackageArn=package_arn, ModelMetadataUri="registry",
                             PredictionAnalysis=phase)
            if "ModelCodeSource" in declared:
                overrides["ModelCodeSource"] = self.settings["model_code_source"]
            elif self.settings["model_code_source"] != "artifact":
                raise ValueError("This pipeline does not expose ModelCodeSource=pipeline.")
            if "ReferenceExecutionArn" in declared:
                overrides["ReferenceExecutionArn"] = "none"
        csv = self._csv(model, phase)
        if csv:
            _location(csv)
            overrides["CsvInputUri"] = csv
        if set(overrides) - set(declared):
            raise ValueError(f"{kind}: unsupported parameters {sorted(set(overrides) - set(declared))}.")
        effective = {name: p["DefaultValue"] for name, p in declared.items() if "DefaultValue" in p}
        effective.update(overrides)
        if set(declared) - set(effective):
            raise ValueError(f"{kind}: missing required parameters {sorted(set(declared) - set(effective))}.")
        for name, p in declared.items():
            if "EnumValues" in p and effective[name] not in p["EnumValues"]:
                raise ValueError(f"{kind}: {name}={effective[name]!r} must be one of {p['EnumValues']}.")
        return kind, overrides, effective

    def run(self):
        """Run all phases; retain evidence and surface failures without hiding them."""
        if not self.prepared:
            raise ValueError("Run prepare() before run().")
        self.state.setdefault("started_at", _now())
        self.state.update(status="Running", error="", ended_at=None)
        self.deadline = time.monotonic() + self.timeout_minutes * 60
        try:
            self._phase("preliminary")
            self._phase("full")
            self._approve_full_models()
            self._phase("prediction_test", "prediction_inference")
            self.state["status"] = "Passed"
        except BaseException as error:
            self.state.update(status="Interrupted" if isinstance(error, KeyboardInterrupt) else "Failed",
                              error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.state["ended_at"] = _now()
            self._save()
            (self.directory / "results.json").write_text(json.dumps(self.report(), indent=2) + "\n")
            self._notify()
        return self.report()

    def _phase(self, *phases):
        cases = []
        # Submit the whole phase before waiting, so independent models run together.
        for phase in phases:
            for model in self.models:
                key = f"{model}/{phase}"
                prior = self.state["executions"].get(f"{model}/preliminary", {})
                full = self.state["executions"].get(f"{model}/full", {})
                kind, overrides, effective = self._parameters(
                    model, phase, metadata_uri=prior.get("metadata_uri", ""),
                    package_arn=full.get("package_arn", ""))
                request = {"PipelineName": self.pipelines[kind]["arn"],
                           "PipelineVersionId": self.pipelines[kind]["version"],
                           "PipelineExecutionDisplayName": (
                               self.settings["run_label"][:40] + "-" + phase.replace("_", "-")[:20] +
                               "-" + hashlib.sha256(model.encode()).hexdigest()[:8]),
                           "PipelineParameters": [{"Name": k, "Value": _string(v)}
                                                  for k, v in sorted(overrides.items())]}
                case = self.state["executions"].setdefault(key, {
                    "model": model, "phase": phase, "kind": kind, "request": request,
                    "effective": effective,
                    "token": uuid5(NAMESPACE_URL, self.settings["run_label"] +
                                   json.dumps(request, sort_keys=True)).hex,
                    "status": "Pending"})
                if case["request"] != request:
                    raise ValueError(f"{key}: saved request differs. Use a new RUN_LABEL.")
                self._save()  # Persist the token before any possible start request.
                if not case.get("execution_arn"):
                    response = self.sm.start_pipeline_execution(
                        **request, ClientRequestToken=case["token"])
                    case.update(execution_arn=response["PipelineExecutionArn"], status="Executing",
                                started_at=_now())
                    self._save()
                cases.append(case)
        self._wait(cases)
        errors = []
        for case in cases:
            prefix = f"{case['model']}/{case['phase']}/"
            self.state["checks"] = {k: v for k, v in self.state["checks"].items()
                                    if not k.startswith(prefix)}
            try:
                self._check(case, "Pipeline succeeded", case["status"] == "Succeeded",
                            case.get("failure", case["status"]))
                self._verify_parameters(case)
                if case["kind"] == "training":
                    self._verify_training(case)
                else:
                    self._verify_inference(case)
            except Exception as error:
                case["verification"] = "Failed"
                self.state["checks"][f"{case['model']}/{case['phase']}/validation"] = {
                    "model": case["model"], "phase": case["phase"], "check": "Required outputs",
                    "result": "FAIL", "detail": str(error)}
                errors.append(f"{case['model']}/{case['phase']}: {error}")
            else:
                case["verification"] = "Passed"
            self._save()
            self._notify()
        if errors:
            raise RuntimeError("\n".join(errors))

    def _wait(self, cases):
        while True:
            pending = []
            for case in cases:
                detail = self.sm.describe_pipeline_execution(PipelineExecutionArn=case["execution_arn"])
                case["status"] = detail["PipelineExecutionStatus"]
                case["steps"] = self._execution_steps(case["execution_arn"])
                if detail.get("CreationTime"):
                    case["started_at"] = detail["CreationTime"].isoformat()
                if case["status"] in TERMINAL:
                    case["ended_at"] = (detail["LastModifiedTime"].isoformat()
                                        if detail.get("LastModifiedTime") else _now())
                    failures = [f"{s['StepName']}: {s.get('FailureReason', s['StepStatus'])}"
                                for s in case["steps"] if s["StepStatus"] in {"Failed", "Stopped"}]
                    case["failure"] = detail.get("FailureReason") or "; ".join(failures)
                else:
                    pending.append(case)
            self._save()
            self._notify()
            if not pending:
                return
            self._pause("Waiting for " + ", ".join(c["model"] + "/" + c["phase"] for c in pending))

    def _pause(self, reason):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(reason + ". Jobs are not cancelled; rerun with the same RUN_LABEL to resume.")
        time.sleep(min(self.poll_seconds, remaining))

    def _execution_steps(self, arn):
        return [s for page in self.sm.get_paginator("list_pipeline_execution_steps").paginate(
            PipelineExecutionArn=arn) for s in page["PipelineExecutionSteps"]]

    def _job(self, case, step_name):
        arn, seen = case["execution_arn"], set()
        while arn and arn not in seen:
            seen.add(arn)
            steps = case["steps"] if arn == case["execution_arn"] else self._execution_steps(arn)
            step = next((s for s in steps if s["StepName"] == step_name), None)
            if step is None or step["StepStatus"] != "Succeeded":
                raise ValueError(f"{step_name}: no successful step found in {arn}.")
            job_arn = step.get("Metadata", {}).get("ProcessingJob", {}).get("Arn")
            if job_arn:
                return self.sm.describe_processing_job(ProcessingJobName=job_arn.rsplit("/", 1)[-1])
            arn = step.get("CacheHitResult", {}).get("SourcePipelineExecutionArn")
        raise ValueError(f"{step_name}: no producing Processing job found.")

    def _output(self, case, step_name, output_name):
        job = self._job(case, step_name)
        outputs = {o["OutputName"]: o["S3Output"]["S3Uri"].rstrip("/")
                   for o in job["ProcessingOutputConfig"]["Outputs"]}
        if output_name not in outputs:
            raise ValueError(f"{step_name}: expected output {output_name!r}; found {sorted(outputs)}.")
        return outputs[output_name], job

    def _json(self, uri):
        bucket, key = _location(uri)
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            with response["Body"] as stream:
                content = stream.read()
            value = json.loads(content)
        except Exception as error:
            raise ValueError(f"Cannot read required JSON {uri}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{uri}: expected a JSON object.")
        return value, hashlib.sha256(content).hexdigest()

    def _check(self, case, name, passed, detail):
        key = f"{case['model']}/{case['phase']}/{name}"
        self.state["checks"][key] = {"model": case["model"], "phase": case["phase"],
                                     "check": name, "result": "PASS" if passed else "FAIL",
                                     "detail": str(detail)}
        if not passed:
            raise ValueError(f"{name}: {detail}")

    def _verify_parameters(self, case):
        parameters = {p["Name"]: p["Value"] for page in
                      self.sm.get_paginator("list_pipeline_parameters_for_execution").paginate(
                          PipelineExecutionArn=case["execution_arn"]) for p in page["PipelineParameters"]}
        mismatch = {name: {"actual": parameters.get(name), "expected": _string(expected)}
                    for name, expected in case["effective"].items()
                    if parameters.get(name) != _string(expected)}
        self._check(case, "Execution parameters", not mismatch, mismatch or "Match the pinned request and defaults")

    def _verify_training(self, case):
        output, _ = self._output(case, "Evaluate", "evaluation")
        metadata_uri = output + "/model_metadata.json"
        metadata, digest = self._json(metadata_uri)
        mapping = metadata.get("feature_mapping")
        self._check(case, "Metadata has features", isinstance(mapping, dict) and bool(mapping), metadata_uri)
        report, _ = self._json(output + "/evaluation.json")
        self._check(case, "Evaluation metadata checksum", report.get("model_metadata_sha256") == digest, metadata_uri)
        registration, _ = self._output(case, "RegisterModel", "registration")
        receipt, _ = self._json(registration + "/package.json")
        arn = receipt["package"]
        package = self.sm.describe_model_package(ModelPackageName=arn)
        body = self.pipelines["training"]["definition"]
        registration_step = next(s for s in _steps(body["Steps"]) if s["Name"] == "RegisterModel")
        group = _value(registration_step["Arguments"]["Environment"]["MODEL_PACKAGE_GROUP"], case["effective"])
        issues = check_model_package(package, group, self.namespace, case["model"],
                                     body["Metadata"]["Environment"], require_approval=False)
        self._check(case, "Registry scope and readiness", not issues, "\n".join(issues) or arn)
        generated_by = package.get("MetadataProperties", {}).get("GeneratedBy")
        self._check(case, "Registry execution", generated_by == case["execution_arn"],
                    f"actual={generated_by!r}, expected={case['execution_arn']!r}")
        properties = package.get("CustomerMetadataProperties", {})
        containers = package.get("InferenceSpecification", {}).get("Containers", [])
        checkpoint = report.get("checkpoint_uri")
        matches = (len(containers) == 1 and containers[0].get("ModelDataUrl") == checkpoint
                   and receipt.get("checkpoint") == checkpoint
                   and properties.get("ModelMetadataUri") == metadata_uri
                   and receipt.get("model_metadata") == metadata_uri
                   and properties.get("ModelMetadataSha256") == digest
                   and receipt.get("model_metadata_sha256") == digest)
        self._check(case, "Registry artifact and metadata", matches, arn)
        bucket, key = _location(checkpoint)
        head = self.s3.head_object(Bucket=bucket, Key=key)
        self._check(case, "Model archive exists", head["ContentLength"] > 0, checkpoint)
        case.update(metadata_uri=metadata_uri, metadata_sha256=digest, feature_count=len(mapping),
                    package_arn=arn, checkpoint_uri=checkpoint, registry_group=group,
                    evaluation_uri=output + "/evaluation.json")

    def _approve_full_models(self):
        for model in self.models:
            case = self.state["executions"][f"{model}/full"]
            # Recheck the exact version immediately before mutating its two approval fields.
            self._verify_training(case)
            arn = case["package_arn"]
            package = self.sm.describe_model_package(ModelPackageName=arn)
            if (package.get("ModelApprovalStatus") != "Approved" or
                    package.get("ModelLifeCycle", {}).get("StageStatus") != "Approved"):
                self.sm.update_model_package(
                    ModelPackageArn=arn, ModelApprovalStatus="Approved",
                    ModelLifeCycle={"Stage": "Development", "StageStatus": "Approved"},
                    ApprovalDescription=f"Integration test {self.settings['run_label']}: verified full training outputs")
            while True:
                package = self.sm.describe_model_package(ModelPackageName=arn)
                issues = check_model_package(
                    package, case["registry_group"], self.namespace, model,
                    self.pipelines["training"]["definition"]["Metadata"]["Environment"])
                if not issues:
                    break
                self._pause(f"Waiting for both approvals on {arn}: " + "; ".join(issues))
            self._check(case, "Both Development approvals", True, arn)
            self._save()
            self._notify()

    def _files(self, uri):
        bucket, prefix = _location(uri.rstrip("/") + "/")
        ignored = {"_execution.json", "run.json", "_SUCCESS", "parameters.json", "model_metadata.json"}
        return [{"uri": f"s3://{bucket}/{o['Key']}", "bytes": o["Size"]}
                for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                for o in page.get("Contents", [])
                if o["Size"] > 0 and Path(o["Key"]).name not in ignored]

    def _verify_inference(self, case):
        full = self.state["executions"][f"{case['model']}/full"]
        output, _ = self._output(case, "ResolveModel", "authorization")
        decision, _ = self._json(output + "/package.json")
        self._check(case, "Exact approved model resolved",
                    decision.get("package") == full["package_arn"]
                    and decision.get("checkpoint") == full["checkpoint_uri"]
                    and decision.get("model_metadata_sha256") == full["metadata_sha256"],
                    full["package_arn"])
        scores, score_job = self._output(case, "Score", "scores")
        inputs = [i["S3Input"]["S3Uri"] for i in score_job.get("ProcessingInputs", []) if "S3Input" in i]
        self._check(case, "Score input model archive", full["checkpoint_uri"] in inputs, inputs)
        record, _ = self._json(scores + "/_execution.json")
        binding = record.get("model_metadata", {})
        self._check(case, "Scoring metadata matches training",
                    binding.get("sha256") == full["metadata_sha256"]
                    and binding.get("checkpoint_uri") == full["checkpoint_uri"]
                    and binding.get("feature_count") == full["feature_count"],
                    binding)
        analysis, _ = self._output(case, "Postprocess", "predictions")
        score_files, analysis_files = self._files(scores), self._files(analysis)
        # Score often writes predictions/raw beneath Postprocess's predictions prefix.
        # Those earlier files must not make a missing analysis output appear successful.
        analysis_files = [f for f in analysis_files if not f["uri"].startswith(scores + "/")]
        self._check(case, "Prediction files exist",
                    any(f["uri"].endswith((".csv", ".jsonl", ".parquet")) for f in score_files), scores)
        self._check(case, "Analysis files exist", bool(analysis_files), analysis)
        case.update(package_arn=full["package_arn"], feature_count=full["feature_count"],
                    score_uri=scores, analysis_uri=analysis,
                    output_files=score_files + analysis_files)

    def report(self):
        ended = self.state.get("ended_at") or _now()
        elapsed = lambda start, end: round((datetime.fromisoformat(end) -
                                            datetime.fromisoformat(start)).total_seconds() / 60, 2)
        rows = []
        for case in self.state["executions"].values():
            cached = [s["StepName"] for s in case.get("steps", []) if s.get("CacheHitResult")]
            active = [s["StepName"] for s in case.get("steps", []) if s["StepStatus"] == "Executing"]
            rows.append({"model": case["model"], "phase": case["phase"], "status": case["status"],
                         "checks": case.get("verification", ""), "active_steps": ", ".join(active),
                         "minutes": elapsed(case["started_at"], case.get("ended_at", ended))
                         if case.get("started_at") else 0,
                         "cached_steps": ", ".join(cached), "features": case.get("feature_count", ""),
                         "execution_arn": case.get("execution_arn", ""),
                         "model_package": case.get("package_arn", ""),
                         "metadata_uri": case.get("metadata_uri", ""),
                         "analysis_uri": case.get("analysis_uri", ""),
                         "failure": case.get("failure", "")})
        checks = list(self.state["checks"].values())
        return {"status": self.state["status"], "error": self.state.get("error", ""),
                "elapsed_minutes": elapsed(self.state["started_at"], ended) if self.state.get("started_at") else 0,
                "passed_checks": sum(c["result"] == "PASS" for c in checks),
                "failed_checks": sum(c["result"] == "FAIL" for c in checks),
                "executions": rows, "checks": checks, "evidence_directory": str(self.directory)}

    def _notify(self):
        report = self.report()
        if self.on_update:
            self.on_update(report)
        else:
            print(f"{report['status']} | {report['elapsed_minutes']:.1f} min | " +
                  ", ".join(f"{r['model']}/{r['phase']}: {r['status']}" for r in report["executions"]),
                  flush=True)
