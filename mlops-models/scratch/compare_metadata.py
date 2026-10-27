"""Paste into one notebook cell INSIDE BMO and set the two identifiers below.

Uses that notebook's existing AWS credentials. Reads SageMaker descriptions and
S3 objects only. No job starts, approvals, uploads or model execution.
The model archives are streamed, not extracted or unpickled.
"""

SCORE_JOB_NAME = ""
MODEL_PACKAGE_ARN = ""

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.parse import urlparse


def s3_object(s3, uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a complete S3 object URI: {uri!r}")
    return s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))


def read_metadata(s3, uri, *, archive=False):
    response = s3_object(s3, uri)
    limit = 4 * 1024 * 1024
    with response["Body"] as stream:
        if archive:
            content, matches = None, 0
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    if PurePosixPath(member.name) != PurePosixPath("model_metadata.json"):
                        continue
                    matches += 1
                    if not member.isfile() or member.size > limit:
                        raise ValueError(f"Unexpected metadata archive member: {member.name}")
                    with tar.extractfile(member) as metadata:
                        content = metadata.read(limit + 1)
            if matches != 1:
                raise ValueError(f"{uri} contains {matches} root model_metadata.json entries; expected one.")
        else:
            content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Metadata exceeds the diagnostic's 4 MiB text limit.")
    document = json.loads(content)
    mapping = document.get("feature_mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"Metadata has no feature_mapping dictionary: {uri}")
    return content, {
        "uri": uri, "member": "model_metadata.json" if archive else None,
        "sha256": hashlib.sha256(content).hexdigest(),
        "feature_count": len(mapping), "feature_mapping": mapping,
        "object_version": response.get("VersionId"), "etag": response.get("ETag"),
        "last_modified": str(response.get("LastModified", "")),
    }


def compare_metadata(job_name, package_arn, *, session=None):
    if not job_name or not package_arn or ":model-package/" not in package_arn:
        raise ValueError("Set SCORE_JOB_NAME and the exact version's MODEL_PACKAGE_ARN.")
    if session is None:
        import boto3
        session = boto3.Session(region_name=package_arn.split(":")[3])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    job = sm.describe_processing_job(ProcessingJobName=job_name)
    package = sm.describe_model_package(ModelPackageName=package_arn)
    inputs = [item["S3Input"] for item in job["ProcessingInputs"]
              if item.get("S3Input", {}).get("LocalPath", "").rstrip("/") == "/opt/ml/processing/model"]
    if len(inputs) != 1:
        raise ValueError("Expected one model input mounted at /opt/ml/processing/model.")
    actual_uri = inputs[0]["S3Uri"]
    if not actual_uri.endswith(".tar.gz"):
        raise ValueError(f"Expected an exact model archive URI; inspect this input: {inputs[0]}")
    containers = package["InferenceSpecification"]["Containers"]
    if len(containers) != 1:
        raise ValueError("Expected one registered inference container.")
    registered_uri = containers[0]["ModelDataUrl"]
    properties = package.get("CustomerMetadataProperties", {})
    report_uri = properties["ModelMetadataUri"]
    env = job.get("Environment", {})
    details = {
        "score_job": job_name, "job_created": str(job["CreationTime"]),
        "job_status": job["ProcessingJobStatus"],
        "package_checked": package_arn, "package_recorded_by_job": env.get("MLOPS_MODEL_PACKAGE"),
        "code_source": env.get("MLOPS_MODEL_CODE_SOURCE", "not recorded"),
        "metadata_uri_recorded_by_job": env.get("MLOPS_MODEL_METADATA_URI"),
        "model_input": inputs[0],
        "registered_archive_uri": registered_uri,
        "score_archive_uri_matches_registry": actual_uri == registered_uri,
    }
    print("JOB AND MODEL REFERENCES\n" + json.dumps(details, indent=2), flush=True)
    output = Path.cwd() / ".runs/metadata-comparison" / job_name
    output.mkdir(parents=True, exist_ok=True)
    objects, loaded = {}, {}
    for label, uri, is_archive in (
            ("registry_report", report_uri, False),
            ("registered_archive", registered_uri, True),
            ("score_archive", actual_uri, True)):
        key = (uri, is_archive)
        if key not in loaded:
            print(f"Reading {label}: {uri}", flush=True)
            loaded[key] = read_metadata(s3, uri, archive=is_archive)
        content, description = loaded[key]
        objects[label] = description
        (output / (label + ".json")).write_bytes(content)
        print(label, "| features:", description["feature_count"], "| SHA-256:", description["sha256"], flush=True)
    comparisons = {}
    for left, right in (("registry_report", "registered_archive"),
                        ("registered_archive", "score_archive"),
                        ("registry_report", "score_archive")):
        a, b = objects[left], objects[right]
        comparison = {
            "identical_bytes": a["sha256"] == b["sha256"],
            "identical_feature_mapping": a["feature_mapping"] == b["feature_mapping"],
            "only_in_" + left: sorted(set(a["feature_mapping"]) - set(b["feature_mapping"])),
            "only_in_" + right: sorted(set(b["feature_mapping"]) - set(a["feature_mapping"])),
        }
        comparisons[left + "_vs_" + right] = comparison
        print(left + " vs " + right + "\n" + json.dumps(comparison, indent=2), flush=True)
    registered_sha = properties.get("ModelMetadataSha256")
    report = {
        **details, "objects": objects, "comparisons": comparisons,
        "registered_metadata_sha256": registered_sha,
        "report_matches_registered_hash": objects["registry_report"]["sha256"] == registered_sha
        if registered_sha else None,
        "note": "S3 bytes as stored now; job-time bytes can differ if objects were overwritten. "
                "An intentional metadata override can make the Score archive differ from Registry's archive.",
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Report matches Registry's recorded hash:", report["report_matches_registered_hash"])
    print("Saved three exact metadata files and comparison:", output.resolve())
    print(report["note"])
    return report


if __name__ == "__main__":
    metadata_comparison = compare_metadata(SCORE_JOB_NAME, MODEL_PACKAGE_ARN)
