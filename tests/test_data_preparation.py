"""Check seeded input identity and honest previews in both preparation interfaces."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = r'''
from contextlib import redirect_stdout
from pathlib import Path
import io, json, os, sys
root, source, mode, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
options = {"sample_rows": 200, "inference_rows": 40, "split_seed": seed,
           "type_overrides": {"unavailable": "continuous"}}
output = io.StringIO()
progress = []
with redirect_stdout(output):
    if mode == "function":
        sys.path.insert(0, str(source / "notebooks/source/setup"))
        from prepare_data import prepare_data
        result = prepare_data(root, options=options, progress=progress.append)
    else:
        os.chdir(root / "notebooks")
        notebook = json.loads((source / "notebooks/01_prepare_data.ipynb").read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        scope = {}
        exec(cells[0], scope)
        exec(cells[1], scope)
        scope.update(options)
        exec(cells[2], scope)
        exec(cells[6], scope)
        result = scope["preparation_report"]
print(json.dumps({"result": result, "output": output.getvalue(), "progress": progress}))
'''


class DataPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def run_preparation(self, name, *, seed=42, mode="function"):
        root = self.base / name
        (root / "notebooks").mkdir(parents=True)
        models = root / "mlops-models/src/mlops_models/models/example"
        models.mkdir(parents=True)
        (models / "parameters.json").write_text(json.dumps({"target_key": "outcome", "primary_key": "customer_id"}))
        shutil.copytree(ROOT / "mlops-features", root / "mlops-features",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "notebooks/config.example.json", root / "notebooks/config.example.json")
        (root / "notebooks/config.json").write_text(json.dumps({
            "namespace": name, "metadata_csv": "metadata.csv",
            "bucket": "fixture", "role_arn": "fixture", "image_uri": "fixture",
        }))
        (root / "metadata.csv").write_text(
            "column_name,column_map,table_nm,ref_val1,ref_val2,ref_val3,ref_val4,ref_val5\n"
            "amount,amount,example,1.1,2.2,3.3,4.4,5.5\n"
            "flag,flag,example,0,1,0,1,1\n"
            "postal,postal,example,001,002,003,004,005\n"
            "unavailable,unavailable,example,.,.,.,.,.\n"
            "outcome,outcome,example,0,1,0,1,1\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", RUNNER, str(root), str(ROOT), mode, str(seed)],
            text=True, capture_output=True,
            env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true"},
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_same_seed_reproduces_csv_bytes_across_workspaces_and_notebook(self):
        first = self.run_preparation("alice")
        second = self.run_preparation("bob")
        notebook = self.run_preparation("carol", mode="notebook")
        different = self.run_preparation("dana", seed=43)
        for other in (second, notebook):
            self.assertEqual(first["result"]["sha256"], other["result"]["sha256"])
            for kind, file in first["result"]["files"].items():
                self.assertEqual(Path(file).read_bytes(),
                                 Path(other["result"]["files"][kind]).read_bytes())
        self.assertNotEqual(first["result"]["sha256"], different["result"]["sha256"])
        self.assertIn("Generated 240/240 rows (100%)", first["progress"])

    def test_missing_values_are_counted_and_preview_is_actual_data(self):
        run = self.run_preparation("inspect")
        result = run["result"]
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["missing_cells"], {"training": 160, "testing": 40, "inference": 40})
        self.assertEqual(result["missing_by_column"]["training"], {"unavailable": 160})
        self.assertIn("<missing>", run["output"])
        self.assertNotIn("reference_min", run["output"])
        self.assertNotIn("NaN", run["output"])
        self.assertTrue(result["environment"]["pandas"])
        self.assertGreater(result["elapsed_seconds"], 0)
        import pandas as pd
        data = pd.read_csv(result["files"]["training"], dtype=str, keep_default_na=False)
        self.assertEqual(result["preview"], data.iloc[:5, :6].to_dict(orient="records"))
        self.assertTrue(data["postal"].str.startswith("00").all())
        saved = json.loads(Path(result["report_path"]).read_text())
        self.assertEqual(saved, result)
        for kind, file in result["files"].items():
            self.assertEqual(result["sha256"][kind], hashlib.sha256(Path(file).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
