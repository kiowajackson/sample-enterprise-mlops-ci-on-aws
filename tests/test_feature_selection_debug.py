"""Exercise local selection using actual entrypoints and mocked read-only AWS APIs."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "notebooks/source/feature_selection/support.py"
SPEC = importlib.util.spec_from_file_location("selection_support", HELPER)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)
EXECUTION = "arn:aws:sagemaker:us-east-1:123456789012:pipeline/debug-training/execution/newrun"
PRODUCER = EXECUTION.replace("newrun", "cachedrun")


class FeatureSelectionDebug(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "mlops-models/src"
        package = self.source / "mlops_models"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        for relative in (
            "entrypoints/__init__.py", "entrypoints/feature_selection.py",
            "model_base/training/feature_selection.py",
        ):
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "mlops-models/src/mlops_models" / relative, destination)
        self.parameters = package / "models/demo/parameters.json"
        self.parameters.parent.mkdir(parents=True)
        self.settings = {
            "primary_key": "id", "target_key": "target", "task_type": "classification",
            "features": {"numeric": ["x", "y"], "categorical": []},
            "feature_selection": {"sample_rows": 8, "n_estimators": 2, "top_k": 2},
            "num_cpus": 1,
        }
        self.parameters.write_text(json.dumps(self.settings))
        self.override = self.parameters.parent / "training/feature_selection.py"
        self.override.parent.mkdir()
        self.override.write_text(
            "import json, os\n"
            "from mlops_models.model_base.training.feature_selection import FeatureSelector, run_feature_selection as base_run\n"
            "MARKER = 'original'\n"
            "def run_feature_selection(data_directory, output_directory, config, *, selector_cls):\n"
            "    (output_directory / 'observed.json').write_text(json.dumps({\n"
            "        'marker': MARKER, 'metadata': os.environ['MLOPS_MODEL_METADATA_URI'],\n"
            "        'mode': os.environ['MLOPS_TRAINING_MODE'],\n"
            "        'uploads': os.environ['MLOPS_WRITE_RUN_REPORT'],\n"
            "        'task_type': config['task_type']}))\n"
            "    return base_run(data_directory, output_directory, config, selector_cls=selector_cls)\n"
        )
        (self.root / "notebooks").mkdir()
        (self.root / "notebooks/config.json").write_text(json.dumps({"namespace": "debug"}))
        self.raw = b"id,x,y,target\n001,1,8,0\n002,2,7,0\n003,3,6,1\n004,4,5,1\n005,5,4,0\n006,6,3,1\n"
        self.etag = '"raw-v1"'
        self.sm = Mock()
        self.s3 = Mock()
        self.session = Mock()
        self.session.client.side_effect = lambda name, **kwargs: {"sagemaker": self.sm, "s3": self.s3}[name]
        self.sm.describe_pipeline.return_value = {"PipelineArn": EXECUTION.rsplit("/execution/", 1)[0]}
        self.job = {
            "ProcessingJobName": "selection-job",
            "ProcessingJobStatus": "Stopped",
            "Environment": {
                "MODEL_ID": "demo",
                "MLOPS_MODEL_METADATA_URI": "s3://bucket/preliminary/model_metadata.json",
                "MLOPS_TRAINING_SELECTION": "Final model with all data",
            },
            "ProcessingInputs": [{
                "InputName": "raw",
                "S3Input": {"S3Uri": "s3://bucket/older-get-data/steps/GetData/raw",
                            "LocalPath": "/opt/ml/processing/raw"},
            }],
        }
        self.sm.describe_processing_job.return_value = self.job
        step = {"StepName": "FeatureSelection", "Metadata": {"ProcessingJob": {"Arn": "arn:job/selection-job"}}}
        self.steps = {EXECUTION: [step], PRODUCER: [step]}
        self.sm.get_paginator.return_value.paginate.side_effect = (
            lambda **kwargs: [{"PipelineExecutionSteps": self.steps[kwargs["PipelineExecutionArn"]]}]
        )
        self.s3.head_object.side_effect = lambda **kwargs: {"ETag": self.etag, "ContentLength": len(self.raw)}
        self.s3.download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(self.raw)
        self.saved_modules = {key: value for key, value in sys.modules.items()
                              if key == "mlops_models" or key.startswith("mlops_models.")}
        self.saved_path = sys.path[:]
        for key in self.saved_modules:
            del sys.modules[key]
        self.session_patch = patch.object(support.boto3, "Session", return_value=self.session)
        self.session_factory = self.session_patch.start()

    def tearDown(self):
        self.session_patch.stop()
        for key in tuple(sys.modules):
            if key == "mlops_models" or key.startswith("mlops_models."):
                del sys.modules[key]
        sys.modules.update(self.saved_modules)
        sys.path[:] = self.saved_path
        self.temporary.cleanup()

    def run_selection(self, execution=EXECUTION):
        with redirect_stdout(io.StringIO()):
            step = support.prepare(self.root, "demo", execution)
            with support.local_environment(step):
                from mlops_models.entrypoints import feature_selection
                return importlib.reload(feature_selection).execute(model_id=step.model_id, input_path=step.data_directory,
                               output_dir=step.output_directory)

    def prepare_step(self):
        with redirect_stdout(io.StringIO()):
            return support.prepare(self.root, "demo", EXECUTION)

    def execution_cell(self):
        notebook = json.loads((ROOT / "notebooks/steps/feature_selection.ipynb").read_text())
        return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == "execute"))

    def test_stopped_job_runs_real_override_with_actual_raw_mount_and_restores_environment(self):
        with patch.dict(os.environ, {"MLOPS_MODEL_METADATA_URI": "previous", "SM_OUTPUT_DIR": "previous-output"}):
            result = self.run_selection()
            self.assertEqual(os.environ["MLOPS_MODEL_METADATA_URI"], "previous")
            self.assertEqual(os.environ["SM_OUTPUT_DIR"], "previous-output")
        self.assertTrue((result / "selected_features.csv").is_file())
        observed = json.loads((result.parent / "observed.json").read_text())
        self.assertEqual(observed["metadata"], self.job["Environment"]["MLOPS_MODEL_METADATA_URI"])
        self.assertEqual(observed["mode"], "2")
        self.assertEqual(observed["uploads"], "0")
        self.assertEqual(observed["marker"], "original")
        bucket, key, _ = self.s3.download_file.call_args.args
        self.assertEqual((bucket, key), ("bucket", "older-get-data/steps/GetData/raw/data.csv"))
        self.session_factory.assert_called_once_with(region_name="us-east-1")
        self.assertFalse(self.sm.start_pipeline_execution.called)
        self.assertFalse(self.sm.stop_pipeline_execution.called)
        self.assertFalse(self.s3.upload_file.called)

    def test_bare_id_uses_training_pipeline_in_configured_namespace(self):
        self.run_selection("newrun")
        self.sm.describe_pipeline.assert_called_once_with(PipelineName="debug-training")

    def test_full_arn_does_not_require_local_config(self):
        (self.root / "notebooks/config.json").unlink()
        self.run_selection()
        self.sm.describe_pipeline.assert_not_called()

    def test_cached_step_follows_producer_before_reading_the_job(self):
        self.steps[EXECUTION] = [{"StepName": "FeatureSelection", "Metadata": {},
                                 "CacheHitResult": {"SourcePipelineExecutionArn": PRODUCER}}]
        self.run_selection()
        calls = self.sm.get_paginator.return_value.paginate.call_args_list
        self.assertEqual([call.kwargs["PipelineExecutionArn"] for call in calls], [EXECUTION, PRODUCER])

    def test_cached_step_with_job_metadata_reports_the_actual_producer(self):
        # SageMaker can return both the cached job and its producing execution.
        self.steps[EXECUTION] = [{**self.steps[EXECUTION][0],
                                  "CacheHitResult": {"SourcePipelineExecutionArn": PRODUCER}}]
        producer, job = support._processing_job(self.sm, EXECUTION)
        self.assertEqual(producer, PRODUCER)
        self.assertIs(job, self.job)
        calls = self.sm.get_paginator.return_value.paginate.call_args_list
        self.assertEqual([call.kwargs["PipelineExecutionArn"] for call in calls], [EXECUTION, PRODUCER])
        self.sm.describe_processing_job.assert_called_once_with(ProcessingJobName="selection-job")

    def test_wrong_model_stops_before_downloading_or_running_code(self):
        self.job["Environment"]["MODEL_ID"] = "another_model"
        with self.assertRaisesRegex(ValueError, "another_model"):
            self.run_selection()
        self.s3.download_file.assert_not_called()
        self.assertFalse((self.root / ".runs").exists())

    def test_missing_job_and_cache_cycle_fail_without_executing(self):
        self.steps[EXECUTION] = [{"StepName": "FeatureSelection", "Metadata": {}}]
        with self.assertRaisesRegex(ValueError, "no Processing job"):
            self.run_selection()
        self.steps[EXECUTION][0]["CacheHitResult"] = {"SourcePipelineExecutionArn": EXECUTION}
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.run_selection()
        self.s3.download_file.assert_not_called()

    def test_rerun_reuses_raw_data_but_reloads_override_and_current_parameters(self):
        first = self.run_selection()
        self.override.write_text(self.override.read_text().replace("'original'", "'edited-override'"))
        self.settings["task_type"] = "regression"
        self.settings["feature_selection"]["top_k"] = 1
        self.parameters.write_text(json.dumps(self.settings))
        second = self.run_selection()
        self.assertNotEqual(first, second)
        self.assertEqual(self.s3.download_file.call_count, 1)
        observed = json.loads((second.parent / "observed.json").read_text())
        self.assertEqual(observed["marker"], "edited-override")
        self.assertEqual(observed["task_type"], "regression")
        self.assertEqual(len((second / "selected_features.csv").read_text().splitlines()), 2)
        self.assertTrue((first / "selected_features.csv").is_file())

    def test_changed_s3_object_is_downloaded_again_and_failed_download_never_executes(self):
        self.run_selection()
        self.etag = '"raw-v2"'
        self.s3.download_file.side_effect = RuntimeError("download interrupted")
        code = support._vscode_code("demo", EXECUTION)
        with patch.object(support, "_vscode_code", return_value=code), \
                patch.object(support, "_prepare_imports") as load:
            with self.assertRaisesRegex(RuntimeError, "download interrupted"):
                self.run_selection()
            load.assert_not_called()
        self.assertEqual(self.s3.download_file.call_count, 2)

    def test_exception_keeps_traceback_inputs_and_restores_environment(self):
        failure = RuntimeError("selector failed here")
        step = self.prepare_step()
        with patch.dict(os.environ, {"MLOPS_TRACKING_URI": "original-tracking"}):
            with self.assertRaises(RuntimeError) as caught:
                with support.local_environment(step):
                    raise failure
            self.assertIs(caught.exception, failure)
            self.assertEqual(os.environ["MLOPS_TRACKING_URI"], "original-tracking")
        self.assertEqual((self.root / ".runs/step-debug/feature-selection/selection-job/raw/data.csv").read_bytes(), self.raw)

    def test_other_checkout_import_is_rejected(self):
        foreign = ModuleType("mlops_models")
        foreign.__file__ = "/different-checkout/mlops_models/__init__.py"
        sys.modules["mlops_models"] = foreign
        with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
            support._prepare_imports(self.root, "demo")

    def notebook_cell(self, cell_id):
        notebook = json.loads((ROOT / "notebooks/steps/feature_selection.ipynb").read_text())
        return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == cell_id))

    def copy_helper(self):
        helper = self.root / "notebooks/source/feature_selection/support.py"
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HELPER, helper)

    def export_namespace(self, code, script=None):
        namespace = {
            "__name__": "__main__",
            "__file__": str(script or self.source / "mlops_models/entrypoints/feature_selection.py"),
        }
        with patch("tempfile.tempdir", str(self.root)), redirect_stdout(io.StringIO()):
            exec(code, namespace)
        return namespace

    def test_four_cells_separate_step_output_from_copyable_export(self):
        notebook = json.loads((ROOT / "notebooks/steps/feature_selection.ipynb").read_text())
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(not cell["outputs"] and cell["execution_count"] is None for cell in cells))
        self.copy_helper()
        steps = self.root / "notebooks/steps"
        steps.mkdir()
        namespace = {}
        exec("".join(cells[0]["source"]), namespace)
        namespace.update(MODEL_ID="demo", EXECUTION_ID=EXECUTION)
        output = io.StringIO()
        with patch.object(Path, "cwd", return_value=steps), redirect_stdout(output):
            exec("".join(cells[1]["source"]), namespace)
            self.assertNotIn("result", namespace)
            exec("".join(cells[2]["source"]), namespace)
        self.assertTrue((namespace["result"] / "selected_features.csv").is_file())
        self.assertNotIn('if __name__ == "__main__":', output.getvalue())
        exported = io.StringIO()
        with redirect_stdout(exported):
            exec("".join(cells[3]["source"]), namespace)
        self.assertEqual(exported.getvalue(), namespace["vscode_code"])
        compile(exported.getvalue(), "copied-output.py", "exec")

    def test_preparation_does_not_import_or_run_the_model(self):
        code = support._vscode_code("demo", EXECUTION)
        with patch.object(support, "_vscode_code", return_value=code), \
                patch.object(support, "_prepare_imports") as load:
            step = self.prepare_step()
        load.assert_not_called()
        self.assertFalse((step.output_directory / "selection").exists())
        self.assertEqual((step.output_directory / "vscode_main.py").read_text(), step.vscode_code)

    def test_notebook_continues_to_clean_export_after_failure_interrupt_and_import_error(self):
        step = self.prepare_step()
        real_reload = importlib.reload
        real_prepare = support._prepare_imports
        for failure, during_import in [
            (RuntimeError("selection failed"), False),
            (KeyboardInterrupt(), False),
            (ModuleNotFoundError("model dependency missing"), True),
        ]:
            with self.subTest(failure=type(failure).__name__):
                namespace = {
                    "step": step, "support": {"local_environment": support.local_environment},
                    "result": "stale", "vscode_code": step.vscode_code,
                    "MODEL_ID": "demo", "EXECUTION_ID": EXECUTION,
                    "prepared_inputs": ("demo", EXECUTION),
                }
                execute = Mock(side_effect=failure)
                load = Mock(side_effect=failure) if during_import else real_prepare

                def reload_module(module):
                    if module.__name__ == "mlops_models.entrypoints.feature_selection":
                        return SimpleNamespace(__file__=module.__file__, execute=execute)
                    return real_reload(module)

                output = io.StringIO()
                with patch.object(support, "_prepare_imports", load), \
                        patch.object(importlib, "reload", side_effect=reload_module), \
                        patch.dict(os.environ, {"MLOPS_TRACKING_URI": "before"}), \
                        redirect_stdout(output):
                    exec(compile(self.execution_cell(), "feature_selection_notebook", "exec"), namespace)
                    self.assertEqual(os.environ["MLOPS_TRACKING_URI"], "before")
                self.assertIs(namespace["execution_error"], failure)
                self.assertIsNone(namespace["result"])
                self.assertIn(type(failure).__name__, output.getvalue())
                self.assertIn("feature_selection_notebook", output.getvalue())
                self.assertNotIn(step.vscode_code, output.getvalue())
                exported = io.StringIO()
                with redirect_stdout(exported):
                    exec(self.notebook_cell("export-vscode"), namespace)
                self.assertEqual(exported.getvalue(), step.vscode_code)

    def test_environment_prepares_imports_without_loading_an_entrypoint(self):
        step = self.prepare_step()
        with redirect_stdout(io.StringIO()), support.local_environment(step) as value:
            self.assertIsNone(value)
            self.assertNotIn("mlops_models.entrypoints.feature_selection", sys.modules)

    def test_download_failure_saves_export_without_printing_it_among_logs(self):
        self.s3.download_file.side_effect = RuntimeError("download failed")
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "download failed"):
            support.prepare(self.root, "demo", EXECUTION)
        self.assertNotIn('if __name__ == "__main__":', output.getvalue())
        saved = list((self.root / ".runs").rglob("vscode_main.py"))
        self.assertEqual(len(saved), 1)
        compile(saved[0].read_text(), "saved-export.py", "exec")

    def test_failed_lookup_clears_previous_step_and_still_reaches_export(self):
        self.copy_helper()
        namespace = {"step": self.prepare_step(), "MODEL_ID": "demo", "EXECUTION_ID": "invalid"}
        self.sm.get_paginator.return_value.paginate.side_effect = RuntimeError("execution missing")
        output = io.StringIO()
        with patch.object(Path, "cwd", return_value=self.root), redirect_stdout(output):
            exec(self.notebook_cell("prepare"), namespace)
            exec(self.execution_cell(), namespace)
        self.assertIsNone(namespace["step"])
        self.assertIsNone(namespace["result"])
        self.assertIn("execution missing", output.getvalue())
        self.assertNotIn('if __name__ == "__main__":', output.getvalue())
        exported = io.StringIO()
        with redirect_stdout(exported):
            exec(self.notebook_cell("export-vscode"), namespace)
        self.assertEqual(exported.getvalue(), namespace["vscode_code"])
        self.assertIn("EXECUTION_ID = 'invalid'", exported.getvalue())

    def test_changed_notebook_inputs_do_not_execute_the_previously_prepared_model(self):
        step = self.prepare_step()
        environment = Mock()
        namespace = {
            "step": step, "support": {"local_environment": environment},
            "MODEL_ID": "ICLV", "EXECUTION_ID": EXECUTION,
            "prepared_inputs": ("demo", EXECUTION), "result": "stale",
        }
        with redirect_stdout(io.StringIO()):
            exec(self.execution_cell(), namespace)
        environment.assert_not_called()
        self.assertIsNone(namespace["result"])

    def test_export_resolves_current_job_and_calls_real_entrypoint(self):
        step = self.prepare_step()
        self.job["ProcessingInputs"][0]["S3Input"]["S3Uri"] = "s3://bucket/new/raw"
        self.assertNotIn("s3://bucket/new/raw", step.vscode_code)
        self.assertNotIn("s3://bucket/older-get-data", step.vscode_code)
        with patch.dict(os.environ, {"MLOPS_MODEL_METADATA_URI": "before"}):
            namespace = self.export_namespace(step.vscode_code)
            self.assertEqual(os.environ["MLOPS_MODEL_METADATA_URI"], "before")
        self.assertTrue((namespace["result"] / "selected_features.csv").is_file())
        observed = json.loads((namespace["output_directory"] / "observed.json").read_text())
        self.assertEqual(observed["task_type"], "classification")
        self.assertEqual((namespace["data_directory"] / "data.csv").read_bytes(), self.raw)
        self.assertEqual(self.s3.download_file.call_args.args[:2], ("bucket", "new/raw/data.csv"))

    def test_switch_from_classifier_to_uppercase_regression_model_runs_correct_code(self):
        step = self.prepare_step()
        first = self.export_namespace(step.vscode_code)
        first_observed = json.loads((first["output_directory"] / "observed.json").read_text())
        self.assertEqual(first_observed["task_type"], "classification")
        second_model = self.parameters.parent.parent / "ICLV"
        second_model.mkdir()
        second_parameters = second_model / "parameters.json"
        second_parameters.write_text(json.dumps({**self.settings, "task_type": "regression"}))
        (second_model / "training").mkdir()
        shutil.copyfile(self.override, second_model / "training/feature_selection.py")
        before = {path: path.read_bytes() for path in (self.parameters, second_parameters)}
        second_execution = EXECUTION.replace("us-east-1", "us-west-2").replace("newrun", "secondrun")
        second_producer = second_execution.replace("secondrun", "secondproducer")
        self.steps[second_execution] = [{
            **self.steps[EXECUTION][0], "CacheHitResult": {"SourcePipelineExecutionArn": second_producer},
        }]
        self.steps[second_producer] = [{
            "StepName": "FeatureSelection", "Metadata": {"ProcessingJob": {"Arn": "arn:job/second-job"}},
        }]
        self.job = {
            **self.job, "ProcessingJobName": "second-job",
            "Environment": {
                "MODEL_ID": "ICLV", "MLOPS_TRAINING_SELECTION": "Preliminary models with validation",
                "MLOPS_MODEL_METADATA_URI": "none",
            },
            "ProcessingInputs": [{"InputName": "raw", "S3Input": {"S3Uri": "s3://second-bucket/second/data.csv"}}],
        }
        self.sm.describe_processing_job.return_value = self.job
        # Continuous targets would fail if the old classifier were still selected.
        second_raw = b"id,x,y,target\n001,1,8,0.15\n002,2,7,0.35\n003,3,6,1.25\n004,4,5,1.75\n005,5,4,0.55\n006,6,3,1.65\n"
        self.s3.download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(second_raw)
        self.session_factory.reset_mock()
        code = step.vscode_code.replace("MODEL_ID = 'demo'", "MODEL_ID = 'ICLV'").replace(
            f"EXECUTION_ID = {EXECUTION!r}", f"EXECUTION_ID = {second_execution!r}")
        with patch.dict(os.environ, {"MODEL_ID": "demo", "MLOPS_MODEL_DIR": str(self.parameters.parent)}):
            namespace = self.export_namespace(code)
            self.assertEqual(os.environ["MODEL_ID"], "demo")
        self.session_factory.assert_called_once_with(region_name="us-west-2")
        self.sm.describe_processing_job.assert_called_with(ProcessingJobName="second-job")
        self.assertEqual(namespace["producer_arn"], second_producer)
        observed = json.loads((namespace["output_directory"] / "observed.json").read_text())
        self.assertEqual(observed["task_type"], "regression")
        self.assertEqual(observed["mode"], "1")
        self.assertEqual(observed["metadata"], "none")
        self.assertEqual(namespace["parameter_path"], second_parameters.resolve())
        self.assertTrue((namespace["result"] / "selected_features.csv").is_file())
        self.assertEqual((namespace["data_directory"] / "data.csv").read_bytes(), second_raw)
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))

    def test_export_uses_current_checkout_config_for_short_id(self):
        step = self.prepare_step()
        self.sm.describe_pipeline.reset_mock()
        new_execution = EXECUTION.replace("debug-training", "new-namespace-training").replace("123456789012", "999999999999")
        self.steps[new_execution] = self.steps[EXECUTION]
        self.sm.describe_pipeline.return_value = {"PipelineArn": new_execution.rsplit("/execution/", 1)[0]}
        (self.root / "notebooks/config.json").write_text(json.dumps({"namespace": "new-namespace"}))
        code = step.vscode_code.replace(f"EXECUTION_ID = {EXECUTION!r}", "EXECUTION_ID = 'newrun'")
        self.assertNotIn("123456789012", code)
        self.assertNotIn("debug-training", code)
        namespace = self.export_namespace(code)
        self.sm.describe_pipeline.assert_called_once_with(PipelineName="new-namespace-training")
        self.assertEqual(namespace["execution_arn"], new_execution)
        self.assertTrue((namespace["result"] / "selected_features.csv").is_file())

    def test_full_arn_occurs_only_in_editable_input_and_needs_no_config(self):
        step = self.prepare_step()
        self.assertEqual(step.vscode_code.count(EXECUTION), 1)
        self.assertEqual(step.vscode_code.count("123456789012"), 1)
        (self.root / "notebooks/config.json").unlink()
        self.sm.describe_pipeline.reset_mock()
        namespace = self.export_namespace(step.vscode_code)
        self.sm.describe_pipeline.assert_not_called()
        self.assertEqual(namespace["execution_arn"], EXECUTION)

    def test_short_id_without_config_reports_required_path(self):
        (self.root / "notebooks/config.json").unlink()
        with self.assertRaisesRegex(ValueError, "full execution ARN.*notebooks/config.json"):
            self.export_namespace(support._vscode_code("demo", "newrun"))

    def test_export_rejects_mismatched_job_model_before_download_or_execution(self):
        step = self.prepare_step()
        self.job["Environment"]["MODEL_ID"] = "different-model"
        self.s3.download_file.reset_mock()
        with self.assertRaisesRegex(ValueError, "different-model"):
            self.export_namespace(step.vscode_code)
        self.s3.download_file.assert_not_called()
        self.assertNotIn("mlops_models.entrypoints.feature_selection", sys.modules)

    def test_model_lookup_uses_checkout_not_fixed_parent_depth(self):
        code = support._vscode_code("demo", EXECUTION)
        nested_script = self.parameters.parent / "training/local_run.py"
        namespace = self.export_namespace(code, script=nested_script)
        self.assertEqual(namespace["parameter_path"], self.parameters.resolve())
        self.assertTrue((namespace["result"] / "selected_features.csv").is_file())

    def test_missing_model_reports_exact_path_and_available_models(self):
        with self.assertRaises(ValueError) as caught:
            self.export_namespace(support._vscode_code("ICLV", EXECUTION))
        self.assertIn("MODEL_ID='ICLV'", str(caught.exception))
        self.assertIn(str(self.parameters.parent.parent.resolve() / "ICLV/parameters.json"), str(caught.exception))
        self.assertIn("demo", str(caught.exception))
        self.s3.download_file.assert_not_called()

    def test_export_rejects_foreign_import_instead_of_using_its_model_code(self):
        foreign = ModuleType("mlops_models")
        foreign.__file__ = "/different-checkout/mlops_models/__init__.py"
        sys.modules["mlops_models"] = foreign
        with self.assertRaisesRegex(RuntimeError, "/different-checkout/.*but this run uses"):
            self.export_namespace(support._vscode_code("demo", EXECUTION))

    def test_exported_main_runs_real_entrypoint_in_relocated_clone_without_launch_config(self):
        step = self.prepare_step()
        clone = self.root / "relocated clone"
        shutil.copytree(self.source, clone / "mlops-models/src")
        entrypoint = clone / "mlops-models/src/mlops_models/entrypoints/feature_selection.py"
        entrypoint.write_text(entrypoint.read_text() + "\n\n" + step.vscode_code)
        # The copy intentionally has no notebook, helper, or notebook config.
        (clone / "boto3.py").write_text(
            "from pathlib import Path\n"
            f"JOB = {self.job!r}\n"
            "class Paginator:\n"
            "    def paginate(self, **kwargs):\n"
            "        return [{'PipelineExecutionSteps': [{'StepName': 'FeatureSelection',\n"
            "            'Metadata': {'ProcessingJob': {'Arn': 'arn:job/selection-job'}}}]}]\n"
            "class SM:\n"
            "    def get_paginator(self, name):\n"
            "        return Paginator()\n"
            "    def describe_processing_job(self, **kwargs):\n"
            "        return JOB\n"
            "class S3:\n"
            "    def download_file(self, bucket, key, path):\n"
            "        assert (bucket, key) == ('bucket', 'older-get-data/steps/GetData/raw/data.csv')\n"
            f"        Path(path).write_bytes({self.raw!r})\n"
            "class Session:\n"
            "    def __init__(self, region_name):\n"
            "        assert region_name == 'us-east-1'\n"
            "    def client(self, name, **kwargs):\n"
            "        return {'sagemaker': SM(), 's3': S3()}[name]\n"
        )
        process = subprocess.run(
            [sys.executable, "-m", "mlops_models.entrypoints.feature_selection"], cwd=clone,
            env={**os.environ, "PYTHONPATH": str(clone / "mlops-models/src"), "PYTHONDONTWRITEBYTECODE": "1",
                 "TMPDIR": str(clone)},
            text=True, capture_output=True, timeout=45,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        outputs = list(clone.glob("feature-selection-*/output/selection/selected_features.csv"))
        self.assertEqual(len(outputs), 1, process.stdout)
        observed = json.loads((outputs[0].parents[1] / "observed.json").read_text())
        self.assertEqual(observed["mode"], "2")
        self.assertEqual(observed["marker"], "original")


if __name__ == "__main__":
    unittest.main()
