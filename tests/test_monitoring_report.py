"""Exercise the optional report against actual Evidently and its publication boundary."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from mlops_models.monitoring.report import publish_report, write_report


class Services:
    def __init__(self):
        self.objects, self.messages, self.metric_calls = {}, [], []

    def client(self, service):
        return self

    def upload_file(self, filename, bucket, key, **kwargs):
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def put_metric_data(self, **request):
        self.metric_calls.append(request)

    def publish(self, **request):
        self.messages.append(json.loads(request["Message"]))
        return {"MessageId": str(len(self.messages))}


@unittest.skipUnless(importlib.util.find_spec("evidently"), "Install requirements-monitoring.txt")
class MonitoringReportTest(unittest.TestCase):
    def test_real_unchanged_and_shifted_reports_publish_only_a_breach(self):
        reference = pd.DataFrame({"activity": [float(i % 50) for i in range(500)],
                                  "segment": ["a", "b"] * 250})
        services = Services()
        context = {"model_id": "usecase2", "environment": "dev",
                   "model_package_arn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/example/1",
                   "inference_execution_arn": "retained-batch", "reference_uri": "s3://data/reference.csv",
                   "current_uri": "s3://data/current.csv"}
        with tempfile.TemporaryDirectory() as temporary:
            for name, current in (("unchanged", reference), ("shifted", reference.assign(activity=reference.activity + 100))):
                output = Path(temporary) / name
                summary = write_report(
                    reference, current, output, numerical=["activity"], categorical=["segment"],
                    context=context, quality_metrics={"auc": None},
                )
                self.assertEqual(summary["quality_status"], "unavailable")
                self.assertEqual(summary["drift_detected"], name == "shifted")
                self.assertGreater((output / "report.html").stat().st_size, 1000)
                receipt = publish_report(
                    services, output, "s3://reports/" + name,
                    cloudwatch_namespace="Example/Monitoring", sns_topic_arn="arn:aws:sns:us-east-1:123456789012:example",
                )
                self.assertEqual("sns_message_id" in receipt, name == "shifted")
                self.assertNotIn("quality.auc", json.loads((output / "metrics.json").read_text()))
            self.assertEqual(len(services.metric_calls), 2)
            self.assertEqual(len(services.messages), 1)
            message = services.messages[0]
            self.assertEqual(message["summary_uri"], "s3://reports/shifted/summary.json")
            self.assertEqual(message["context"], context)
            self.assertEqual(message["drifted_columns"], ["activity"])
            self.assertEqual(message["metrics"]["drifted_fraction"], 0.5)
            self.assertIn(("reports", "shifted/receipt.json"), services.objects)


if __name__ == "__main__":
    unittest.main()
