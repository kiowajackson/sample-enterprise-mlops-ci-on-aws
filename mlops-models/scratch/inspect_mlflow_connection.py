# November 2, 2026. Paste this ENTIRE file into one cell in the failing notebook.
TRACKING_URI = ""  # Required: the same arn:aws:sagemaker:...:mlflow-tracking-server/... used by MLflow.
NOTEBOOK_VPC_ID = ""  # Optional. Otherwise discovered from Studio metadata when available.

# Read-only: no experiments, runs, artifacts, permissions, or infrastructure are created.
# Each probe has a wait limit. A timed-out read may finish in a daemon thread.
# MLflow URI and HTTP timeout settings remain set in this notebook after the cell.
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
import requests

match = re.fullmatch(
    r"arn:aws:sagemaker:([a-z0-9-]+):(\d{12}):mlflow-tracking-server/([A-Za-z0-9-]+)",
    TRACKING_URI.strip(),
)
if not match:
    raise ValueError("Set TRACKING_URI to the tracking SERVER ARN, not a URL or MLflow App ARN.")
if NOTEBOOK_VPC_ID and not re.fullmatch(r"vpc-[a-f0-9]+", NOTEBOOK_VPC_ID):
    raise ValueError("NOTEBOOK_VPC_ID must be empty or a VPC ID such as vpc-0123456789abcdef0.")
TRACKING_URI = TRACKING_URI.strip()
region, server_account, server_name = match.groups()
host = f"{region}.experiments.sagemaker.aws"
url = f"https://{host}/api/2.0/mlflow/experiments/search"
endpoint_service = f"aws.sagemaker.{region}.experiments"
started_at = time.monotonic()
deadline = started_at + 180
checks = {}
mlflow_diagnostic = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "tracking_server_arn": TRACKING_URI,
    "region": region,
    "host": host,
    "endpoint_service": endpoint_service,
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "checks": checks,
}


def safe_text(value):
    text = str(value)
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1<credentials-redacted>@", text)
    text = re.sub(r"(https?://[^\s?'\"\)]+)\?[^\s'\"\)]+", r"\1?<query-redacted>", text)
    text = re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "<access-key-redacted>", text)
    text = re.sub(
        r"(?im)((?:authorization|x-amz-security-token|aws_secret_access_key|"
        r"aws_session_token)\s*[:=]\s*)[^\n]+", r"\1<redacted>", text
    )
    return text[:2000]


def error_details(error):
    details = {"type": type(error).__name__, "message": safe_text(error)}
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        details["aws_error"] = response.get("Error", {}).get("Code")
        details["http_status"] = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        details["request_id"] = response.get("ResponseMetadata", {}).get("RequestId")
    elif response is not None:
        details["http_status"] = getattr(response, "status_code", None)
        details["request_id"] = response.headers.get("x-amzn-requestid")
    cause = error.__cause__ or error.__context__
    if cause:
        details["cause"] = {"type": type(cause).__name__, "message": safe_text(cause)}
    return details


def check(label, operation, seconds=15):
    remaining = min(seconds, deadline - time.monotonic())
    if remaining <= 0:
        checks[label] = {"status": "SKIPPED", "reason": "180-second diagnostic budget exhausted"}
        print(f"\nSKIPPED {label}: diagnostic budget exhausted", flush=True)
        return None
    print(f"\nSTART {label} (wait limit {remaining:.0f}s)", flush=True)
    result_queue = queue.Queue(maxsize=1)

    def run():
        try:
            result_queue.put(("OK", operation()))
        except Exception as error:
            result_queue.put(("FAILED", error_details(error)))

    worker = threading.Thread(target=run, daemon=True, name="mlflow-diagnostic")
    before = time.monotonic()
    worker.start()
    try:
        status, details = result_queue.get(timeout=remaining)
    except queue.Empty:
        status, details = "TIMEOUT", {"reason": "Probe did not finish within the wait limit"}
        frame = sys._current_frames().get(worker.ident)
        if frame is not None:
            details["waiting_stack"] = [
                f"{Path(item.filename).name}:{item.lineno} in {item.name}"
                for item in traceback.extract_stack(frame)[-8:]
            ]
    checks[label] = {"status": status, "seconds": round(time.monotonic() - before, 2),
                     "details": details}
    print(status, label, json.dumps(details, default=str), flush=True)
    return details if status == "OK" else None


versions = {}
for package in ("mlflow", "sagemaker-mlflow", "boto3", "botocore", "requests", "urllib3"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "NOT INSTALLED"
mlflow_diagnostic["versions"] = versions
print("CONFIG", json.dumps({k: v for k, v in mlflow_diagnostic.items() if k != "checks"}), flush=True)

effective_proxy = requests.utils.select_proxy(url, requests.utils.get_environ_proxies(url))
proxy_address, proxy_port, proxy_parse_error = None, None, None
try:
    proxy_address = urlsplit(effective_proxy) if effective_proxy else None
    if proxy_address:
        proxy_port = proxy_address.port or (443 if proxy_address.scheme == "https" else 80)
        if not proxy_address.hostname:
            raise ValueError("Configured proxy has no hostname.")
except ValueError as error:
    proxy_address = None
    proxy_parse_error = safe_text(error)
mlflow_diagnostic["network_settings"] = {
    "uses_environment_proxy": bool(effective_proxy),
    "proxy_host": proxy_address.hostname if proxy_address else None,
    "proxy_port": proxy_port,
    "proxy_parse_error": proxy_parse_error,
    "no_proxy_configured": bool(os.getenv("NO_PROXY") or os.getenv("no_proxy")),
    "custom_ca_bundle": os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("CURL_CA_BUNDLE"),
}
print("NETWORK SETTINGS", json.dumps(mlflow_diagnostic["network_settings"]), flush=True)


def resolve(name, port):
    entries = socket.getaddrinfo(name, port, type=socket.SOCK_STREAM)
    return [dict(family=family, ip=ip) for family, ip in
            sorted({(entry[0], entry[4][0]) for entry in entries})]


def connect(address, port):
    with socket.socket(address["family"], socket.SOCK_STREAM) as connection:
        connection.settimeout(4)
        connection.connect((address["ip"], port))
    return {"ip": address["ip"], "port": port, "connected": True}


addresses = check("Endpoint DNS", lambda: resolve(host, 443), seconds=8)
for address in (addresses or [])[:4]:
    check(f"Direct TCP {address['ip']}:443 (bypasses proxy)",
          lambda address=address: connect(address, 443), seconds=6)
if addresses and len(addresses) > 4:
    print("Direct TCP probes cover only the first four DNS addresses.", flush=True)
if proxy_address:
    proxy_port = mlflow_diagnostic["network_settings"]["proxy_port"]
    proxy_ips = check("Proxy DNS", lambda: resolve(proxy_address.hostname, proxy_port), seconds=8)
    if proxy_ips:
        check("Proxy TCP", lambda: connect(proxy_ips[0], proxy_port), seconds=6)


def unsigned_search():
    with requests.Session() as http:
        http.auth = lambda request: request  # Prevent implicit .netrc authentication.
        http.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        with http.post(url, json={"max_results": 1}, timeout=(5, 10),
                       allow_redirects=False, stream=True) as response:
            return {
                "http_status": response.status_code,
                "server": response.headers.get("Server"),
                "via": bool(response.headers.get("Via")),
                "request_id": response.headers.get("x-amzn-requestid"),
                "aws_error_type": response.headers.get("x-amzn-errortype"),
                "meaning": "HTTP responder reached; may be a proxy. Unsigned 401/403 is NOT an IAM test.",
            }


check("Unsigned HTTPS through configured route", unsigned_search, seconds=20)
client_config = Config(connect_timeout=5, read_timeout=8, retries={"total_max_attempts": 1})
session = boto3.Session(region_name=region)


def credentials_ready():
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials found in this notebook's credential chain.")
    credentials.get_frozen_credentials()  # Force retrieval/refresh without displaying any key/token.
    return {"provider": credentials.method,
            "expiry": str(getattr(credentials, "_expiry_time", None))}


credentials = check("AWS credential resolution", credentials_ready)
server = None
if credentials:
    check("Notebook caller identity", lambda: {
        key: value for key, value in session.client("sts", config=client_config).get_caller_identity().items()
        if key in {"Arn", "Account", "UserId"}
    })

    def describe_server():
        response = session.client("sagemaker", config=client_config).describe_mlflow_tracking_server(
            TrackingServerName=server_name)
        details = {key: response.get(key) for key in (
            "TrackingServerArn", "TrackingServerStatus", "IsActive", "MlflowVersion",
            "RoleArn", "ArtifactStoreUri"
        )}
        details["arn_matches_requested"] = response.get("TrackingServerArn") == TRACKING_URI
        return details

    server = check("Describe tracking server (management API)", describe_server)

mlflow_modules = {}


def load_mlflow():
    for name in ("mlflow", "sagemaker_mlflow"):
        mlflow_modules[name] = importlib.import_module(name)
    return {name: module.__file__ for name, module in mlflow_modules.items()}


if credentials and check("MLflow and AWS plugin imports", load_mlflow):
    # The plugin reads the GLOBAL URI; passing only MlflowClient(tracking_uri=...) is insufficient.
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    mlflow_modules["mlflow"].set_tracking_uri(TRACKING_URI)
    check("SIGNED SearchExperiments", lambda: {
        "experiment_count": len(mlflow_modules["mlflow"].search_experiments(max_results=1))
    }, seconds=20)
else:
    checks["SIGNED SearchExperiments"] = {
        "status": "SKIPPED", "reason": "Credentials or required imports unavailable"
    }

# Optional infrastructure reads. A denied inspection is a gap in evidence, not proof of bad networking.
vpc_id = NOTEBOOK_VPC_ID
studio_file = Path("/opt/ml/metadata/resource-metadata.json")
studio = {}
if studio_file.is_file():
    try:
        studio = json.loads(studio_file.read_text())
    except (OSError, ValueError) as error:
        print("Studio metadata could not be read:", type(error).__name__, flush=True)
domain_id = studio.get("DomainId") or os.getenv("SAGEMAKER_DOMAIN_ID")
mlflow_diagnostic["studio"] = {key: studio[key] for key in
                              ("DomainId", "UserProfileName", "SpaceName") if key in studio}
if credentials and domain_id:
    def describe_domain():
        domain = session.client("sagemaker", config=client_config).describe_domain(DomainId=domain_id)
        return {key: domain.get(key) for key in ("DomainId", "VpcId", "SubnetIds", "AppNetworkAccessType")}

    domain = check("Studio domain network", describe_domain)
    if domain:
        if vpc_id and vpc_id != domain["VpcId"]:
            print("WARNING: supplied VPC differs from Studio domain VPC; inspect both bindings.", flush=True)
        vpc_id = vpc_id or domain["VpcId"]
mlflow_diagnostic["notebook_vpc_id"] = vpc_id or "UNKNOWN"

if credentials:
    def describe_endpoints():
        filters = [{"Name": "service-name", "Values": [endpoint_service]}]
        if vpc_id:
            filters.append({"Name": "vpc-id", "Values": [vpc_id]})
        endpoints = []
        for page in session.client("ec2", config=client_config).get_paginator(
                "describe_vpc_endpoints").paginate(Filters=filters):
            for endpoint in page["VpcEndpoints"]:
                endpoints.append({key: endpoint.get(key) for key in (
                    "VpcEndpointId", "VpcId", "State", "PrivateDnsEnabled", "SubnetIds",
                    "Groups", "NetworkInterfaceIds", "DnsEntries", "PolicyDocument"
                )})
        return endpoints

    endpoints = check("MLflow interface endpoint inventory", describe_endpoints)
    if endpoints:
        group_ids = sorted({group["GroupId"] for endpoint in endpoints for group in (endpoint["Groups"] or [])})
        interface_ids = sorted({interface for endpoint in endpoints for interface in (endpoint["NetworkInterfaceIds"] or [])})
        if interface_ids:
            def describe_interfaces():
                interfaces = session.client("ec2", config=client_config).describe_network_interfaces(
                    NetworkInterfaceIds=interface_ids)["NetworkInterfaces"]
                return [{key: interface.get(key) for key in (
                    "NetworkInterfaceId", "VpcId", "SubnetId", "PrivateIpAddress", "Ipv6Addresses", "Groups"
                )} for interface in interfaces]

            check("Endpoint interface IPs (compare with DNS)", describe_interfaces)
        if group_ids:
            def describe_groups():
                groups = session.client("ec2", config=client_config).describe_security_groups(
                    GroupIds=group_ids)["SecurityGroups"]
                return [{key: group.get(key) for key in
                         ("GroupId", "VpcId", "IpPermissions", "IpPermissionsEgress")} for group in groups]

            check("Endpoint security-group rules", describe_groups)

print("\nSTANDUP SUMMARY", flush=True)
signed = checks["SIGNED SearchExperiments"]
print("Signed search:", signed["status"], flush=True)
if signed["status"] != "OK":
    print(json.dumps(signed.get("details", signed.get("reason")), default=str), flush=True)
signed_error = json.dumps(signed.get("details", {})).lower()
if signed["status"] == "OK":
    conclusion = "Signed experiment search succeeded from this notebook; the previous failure was not reproduced."
elif signed["status"] == "SKIPPED":
    conclusion = "Signed search was not attempted. Resolve the reported credential/import/budget issue first."
elif signed["status"] == "TIMEOUT":
    conclusion = "Signed search exceeded its wait limit. The captured stack shows where its thread was waiting."
elif "sslerror" in signed_error or "certificate_verify_failed" in signed_error:
    conclusion = "Signed search failed at TLS/certificate validation. Inspect the CA bundle and TLS inspection."
elif "connecttimeout" in signed_error or "name or service not known" in signed_error:
    conclusion = "Signed search failed while establishing the connection. Investigate DNS/proxy/network access."
elif "readtimeout" in signed_error:
    conclusion = "Signed search timed out waiting for a response. Check the responder/proxy/service path."
elif (any(word in signed_error for word in ("accessdenied", "permission_denied"))
      or re.search(r"\b(?:401|403)\b", signed_error)):
    conclusion = "Signed search returned an authorization-related error. Inspect the exact error, caller IAM and endpoint policy."
else:
    conclusion = "Signed search failed; use the exact error above to identify the next check."
mlflow_diagnostic["conclusion"] = conclusion
print("Finding:", conclusion, flush=True)
inventory = checks.get("MLflow interface endpoint inventory", {})
if inventory.get("status") == "OK" and not inventory["details"]:
    print(f"No {endpoint_service} endpoint found in "
          f"{vpc_id or 'the caller account/region (notebook VPC unknown)'}.", flush=True)
    print("A missing endpoint matters only if the intended connection uses PrivateLink.", flush=True)
if inventory.get("status") == "OK" and vpc_id:
    for endpoint in inventory["details"]:
        if endpoint["State"] != "available" or not endpoint["PrivateDnsEnabled"]:
            print("Endpoint needs review:", endpoint["VpcEndpointId"],
                  "state=", endpoint["State"], "private DNS=", endpoint["PrivateDnsEnabled"], flush=True)
interfaces = checks.get("Endpoint interface IPs (compare with DNS)", {})
if interfaces.get("status") == "OK" and addresses:
    endpoint_ips = {item["PrivateIpAddress"] for item in interfaces["details"]}
    endpoint_ips.update(ip["Ipv6Address"] for item in interfaces["details"]
                        for ip in (item["Ipv6Addresses"] or []))
    matching_ips = sorted({item["ip"] for item in addresses} & endpoint_ips)
    print("Endpoint DNS addresses matching inspected endpoint interfaces:", matching_ips, flush=True)
if server and str(server.get("MlflowVersion", "")).startswith("3.0"):
    print("AWS documents mlflow==3.0.0 for a 3.0.x tracking server; installed:",
          versions["mlflow"], flush=True)
if server and not server["arn_matches_requested"]:
    print("WARNING: described server ARN differs from requested ARN; check the caller account and server name.",
          flush=True)
print("ConnectTimeout/DNS/TCP failures: investigate the network route before changing IAM.", flush=True)
print("SSLError: inspect certificate trust or TLS inspection. Signed authorization errors: inspect IAM/endpoint policy.", flush=True)
print("A direct TCP failure may be expected when a proxy is required.", flush=True)
print("Security-group rules alone do not establish routing, NACL access, or the notebook's effective source security group.", flush=True)
print("Read-only inspection failures and skipped checks remain unresolved, not successful.", flush=True)
print(f"Finished in {time.monotonic() - started_at:.1f}s. Copy the cell output for standup.", flush=True)
# The structured results also remain available as mlflow_diagnostic in this kernel.
