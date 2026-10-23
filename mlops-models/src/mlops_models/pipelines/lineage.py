"""Publish a small native SageMaker graph from the jobs and locations already recorded."""

import hashlib

from botocore.config import Config
from mlops_models.pipelines.tracking import collect_execution, _put_json, READ_CONFIG

LINEAGE_CONFIG = Config(connect_timeout=5, read_timeout=15,
                        retries={"mode": "adaptive", "total_max_attempts": 8})


def pages(client, operation, key, **request):
    """Read an API's remaining pages without changing the original request."""
    while True:
        page = getattr(client, operation)(**request)
        yield from page.get(key, [])
        if not page.get("NextToken"):
            return
        request["NextToken"] = page["NextToken"]


def job_inputs(step):
    """Return the locations the native job declared as inputs."""
    job = step["job"]
    if step["job_type"] == "ProcessingJob":
        return [item["S3Input"]["S3Uri"] for item in job.get("ProcessingInputs", []) if "S3Input" in item]
    if step["job_type"] == "TrainingJob":
        return [item["DataSource"]["S3DataSource"]["S3Uri"] for item in job.get("InputDataConfig", [])
                if "S3DataSource" in item.get("DataSource", {})]
    return [job["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"]]


def job_outputs(step):
    """Return successful output locations, including the application's model archive."""
    return list(step.get("outputs", {}).values()) + [
        step[key] for key in ("model_artifact", "transform_output") if step.get(key)]


def publish_lineage(session, execution_arn, *, prefix="mlops", kms_key=""):
    """Connect artifacts to existing native job entities and save a graph receipt.

    This reads existing job/output references. It does not reread datasets, calculate
    data hashes, rerun models, or change Registry approval.
    """
    report = collect_execution(session, execution_arn)
    sm = session.client("sagemaker", region_name=execution_arn.split(":")[3], config=LINEAGE_CONFIG)
    steps = [step for step in report["steps"] if step["status"] == "Succeeded" and step.get("job_arn")]
    jobs = {}
    # Resolve every job before creating metadata. Never invent a second job entity
    # when native metadata is delayed, unavailable or ambiguous.
    for step in steps:
        job_arn = step["job_arn"]
        matches = list(pages(sm, "list_trial_components", "TrialComponentSummaries", SourceArn=job_arn))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one native lineage component for {job_arn}; found {len(matches)}. "
                "Check native lineage availability and retry; no replacement job record was created."
            )
        jobs[job_arn] = matches[0]["TrialComponentArn"]
    outputs = {uri.rstrip("/") for step in steps for uri in job_outputs(step)}
    model_uris = {item["package"]["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
                  for item in report["registry"]}
    model_uris.update(step["model_artifact"] for step in steps if step.get("model_artifact"))
    code_uris = {item["S3Input"]["S3Uri"].rstrip("/") for step in steps
                 for item in step["job"].get("ProcessingInputs", [])
                 if item.get("InputName") in {"source", "code"} and "S3Input" in item}
    artifacts, edges, created = {}, [], {"artifacts": [], "associations": []}
    owned_artifacts = []
    tags = [{"Key": "Purpose", "Value": "mlops-lineage-example"}]

    def name(uri):
        return prefix + "-" + hashlib.sha256(uri.encode()).hexdigest()[:24]

    def artifact(uri, kind=None):
        uri = uri.rstrip("/")
        if uri in artifacts:
            return artifacts[uri]
        kind = kind or ("Model" if uri in model_uris else "Code" if uri in code_uris else "Dataset")
        matches = list(pages(sm, "list_artifacts", "ArtifactSummaries", SourceUri=uri))
        preferred = [item for item in matches if item["ArtifactType"].lower() == kind.lower()]
        matches = preferred or matches
        if len(matches) > 1:
            raise ValueError(f"More than one native artifact describes {uri}; select the intended existing graph.")
        if matches:
            arn = matches[0]["ArtifactArn"]
            if matches[0].get("ArtifactName") == name(uri):
                owned_artifacts.append(arn)
        else:
            arn = sm.create_artifact(ArtifactName=name(uri), ArtifactType=kind,
                                     Source={"SourceUri": uri}, Tags=tags)["ArtifactArn"]
            created["artifacts"].append(arn)
            owned_artifacts.append(arn)
        artifacts[uri] = arn
        return arn

    def associate(source, destination, kind):
        edge = {"SourceArn": source, "DestinationArn": destination, "AssociationType": kind}
        if edge in edges:
            return
        present = any(item["DestinationArn"] == destination for item in pages(
            sm, "list_associations", "AssociationSummaries", SourceArn=source, AssociationType=kind))
        if not present:
            sm.add_association(**edge)
            created["associations"].append(edge)
        edges.append(edge)

    def producer_location(uri):
        # A downstream channel can select a file inside the upstream output prefix.
        candidates = [root for root in outputs if uri.rstrip("/") == root or uri.startswith(root + "/")]
        return max(candidates, key=len) if candidates else uri

    for step in steps:
        job = jobs[step["job_arn"]]
        for uri in job_inputs(step):
            associate(artifact(producer_location(uri)), job, "ContributedTo")
        for uri in job_outputs(step):
            associate(job, artifact(uri), "Produced")

    package_roots = []
    for reference in report["registry"]:
        package = reference["package"]
        model = artifact(package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"], "Model")
        registered = artifact(package["ModelPackageArn"], "Model")
        associate(model, registered, "ContributedTo")
        package_roots.append(registered)
        # ResolveModel binds the selected registered model to this inference execution.
        for step in steps:
            if step["name"] == "ResolveModel":
                associate(registered, jobs[step["job_arn"]], "ContributedTo")
            elif step["name"] in {"RegisterModel", "RegisterTunedModel"}:
                associate(jobs[step["job_arn"]], registered, "Produced")

    for dataset in report["datasets"]:
        raw = artifact(dataset["uri"])
        producer = artifact(producer_location(dataset["uri"]))
        if raw != producer:
            associate(producer, raw, "Produced")
        provenance = dataset["provenance"]
        get_data = next((jobs[step["job_arn"]] for step in steps
                         if step["name"] == "GetData" and step["producer_execution_arn"] ==
                         dataset["producer_execution_arn"]), raw)
        for query in provenance.get("queries", []):
            group_uri = execution_arn.split(":pipeline/")[0] + ":feature-group/" + query["group"]
            associate(artifact(group_uri, "FeatureGroup"), get_data, "ContributedTo")
        if provenance.get("universe_uri"):
            associate(artifact(provenance["universe_uri"]), get_data, "ContributedTo")

    parameters = report["parameters"]
    root = (parameters["OutputDataRoot"].rstrip("/") + "/" + parameters["ModelId"] +
            "/runs/" + execution_arn.rsplit("/", 1)[-1])
    result = {"execution_arn": execution_arn, "artifacts": artifacts, "jobs": jobs,
              "associations": edges, "package_roots": package_roots, "created": created,
              "owned_artifacts": owned_artifacts,
              "receipt_uri": root + "/lineage.json"}
    _put_json(session.client("s3", config=READ_CONFIG), result["receipt_uri"], result, kms_key)
    return result


def query_lineage(session, start_arn):
    """Read the native upstream graph from one artifact/action (service depth limit: ten)."""
    sm = session.client("sagemaker", region_name=start_arn.split(":")[3], config=LINEAGE_CONFIG)
    request = {"StartArns": [start_arn], "Direction": "Ascendants", "IncludeEdges": True,
               "MaxDepth": 10, "MaxResults": 50}
    vertices, edges = {}, {}
    while True:
        page = sm.query_lineage(**request)
        vertices.update({item["Arn"]: item for item in page.get("Vertices", [])})
        edges.update({(item["SourceArn"], item["DestinationArn"], item["AssociationType"]): item
                      for item in page.get("Edges", [])})
        if not page.get("NextToken"):
            break
        request["NextToken"] = page["NextToken"]
    return {"Vertices": list(vertices.values()), "Edges": list(edges.values())}
