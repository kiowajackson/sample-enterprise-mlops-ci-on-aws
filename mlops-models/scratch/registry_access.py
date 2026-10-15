"""Paste this entire file into one notebook cell and run it.

This checks the current notebook credentials, deployed role bindings and Registry
read access. It does not assume another role or create, approve or delete models.
Configured pipeline/job roles are reported, not impersonated or permission-tested.

AWS role documentation:
https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-access.html
https://docs.aws.amazon.com/STS/latest/APIReference/API_GetCallerIdentity.html
"""

# INPUTS — edit this block only. Use the exact deployed names.
pipeline_names = ["wsf-dev-training", "wsf-dev-inference"]
model_package_groups = []  # Paste the exact dev group names; empty lists available groups.
region_name = None  # None uses this notebook's configured AWS Region.

import json
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def walk_steps(value):
    """Include steps inside the per-model and quality-gate branches."""
    if isinstance(value, dict):
        if "Name" in value and "Type" in value and "Arguments" in value:
            yield value
        for child in value.values():
            yield from walk_steps(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_steps(child)


session = boto3.Session(region_name=region_name)
identity = session.client("sts").get_caller_identity()
report = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "region": session.region_name,
    "caller": {key: identity[key] for key in ("Account", "Arn", "UserId")},
    "pipelines": [],
    "checks": [],
    "write_access": "NOT TESTED",
}
print("AWS account:", identity["Account"])
print("AWS Region:", session.region_name)
print("NOTEBOOK CALLER:", identity["Arn"])
print("All AWS checks below use this caller. Setting role_arn does not change it.")
if ":assumed-role/" in identity["Arn"]:
    print("Notebook role name:", identity["Arn"].split(":assumed-role/", 1)[1].split("/")[0])
assert session.region_name, "Set region_name in the inputs to the groups' AWS Region."
sm = session.client("sagemaker")


def check(operation, **arguments):
    """Keep access denial distinct from an empty Registry."""
    item = {"operation": operation, "arguments": arguments}
    try:
        response = getattr(sm, operation)(**arguments)
    except ClientError as error:
        item.update(status="FAILED", error=error.response["Error"])
        print(f"FAILED {operation}: {item['error']['Code']}: {item['error']['Message']}")
        response = None
    else:
        item["status"] = "ALLOWED for notebook caller"
        print(f"ALLOWED {operation}: {arguments}")
    report["checks"].append(item)
    return response


for name in pipeline_names:
    pipeline = check("describe_pipeline", PipelineName=name)
    if pipeline is None:
        continue
    binding = {"name": name, "pipeline_role": pipeline["RoleArn"], "steps": []}
    print("\nPIPELINE:", name)
    print("Configured PIPELINE execution role:", pipeline["RoleArn"])
    print("These are the currently deployed bindings, not a historical execution.")
    definition = json.loads(pipeline["PipelineDefinition"])
    for step in walk_steps(definition):
        arguments = step["Arguments"]
        role = arguments.get("RoleArn")
        if step["Type"] == "RegisterModel":
            role = pipeline["RoleArn"]
            print("Registration group configured in", step["Name"], ":",
                  json.dumps(arguments.get("ModelPackageGroupName")))
        if role:
            binding["steps"].append({"name": step["Name"], "type": step["Type"], "role": role})
            print(" ", step["Name"], "|", step["Type"], "|", role)
    report["pipelines"].append(binding)

if not model_package_groups:
    groups = check("list_model_package_groups", MaxResults=100)
    if groups is not None:
        names = [group["ModelPackageGroupName"] for group in groups["ModelPackageGroupSummaryList"]]
        report["available_groups_first_page"] = names
        print("\nAvailable groups (first page):", names)
        if groups.get("NextToken"):
            print("More groups exist; this discovery list is limited to 100.")
    print("Paste the exact dev names into model_package_groups and rerun to test their reads.")

for group in model_package_groups:
    details = check("describe_model_package_group", ModelPackageGroupName=group)
    if details is not None:
        print("GROUP ARN:", details["ModelPackageGroupArn"])
    versions = check(
        "list_model_packages", ModelPackageGroupName=group,
        SortBy="CreationTime", SortOrder="Descending", MaxResults=10,
    )
    if versions is not None:
        packages = versions["ModelPackageSummaryList"]
        print("Recent versions (up to 10):", [
            (p["ModelPackageArn"], p.get("ModelApprovalStatus"), p["ModelPackageStatus"])
            for p in packages
        ])
        if packages:
            check("describe_model_package", ModelPackageName=packages[0]["ModelPackageArn"])
        else:
            print("Group is readable and empty. Version read access has not been tested.")

report_path = Path.cwd() / (
    "registry-access-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
)
report_path.write_text(json.dumps(report, indent=2) + "\n")
print("\nSaved report:", report_path.resolve())
print("WRITE ACCESS NOT TESTED: successful reads do not prove registration, approval or deletion.")
print("Verify pipeline write access by registering one dev candidate as PendingManualApproval.")
print("The pipeline's RegisterModel result must contain the new ModelPackageArn.")
print("Notebook/Studio approval and deletion require separate operator permissions.")
print("For a Studio/browser action, its CloudTrail event identifies the actual caller role.")
