# Source export

From the repository root:

```bash
python release/export.py --output /your/output/source.zip
```

The exporter packages current working files, including staged edits. It does
not imply that those bytes match the last commit or have passed a cloud run.
The checksum identifies the exact ZIP. An existing ZIP is not overwritten.

The source includes the root README, the implementation and integration
guides under docs/, code, templates, configurations, dependency declarations
(including optional requirements-tracking.txt), synthetic CSVs and clean notebooks.
Only the explicitly listed docs/ guides are included; other planning
documents stay excluded. Non-README Markdown in the source folders, hidden
working directories, deployment receipts, generated definitions and Git history
are excluded. Notebook outputs, execution counts and transient metadata are
removed during export.

Review the extracted archive before transfer, especially configuration values and
notebook source cells. Clearing outputs does not sanitize literals in source.
Customer-specific settings and credentials should be supplied in the receiving
environment.

## Adoption stages

Export the checked-out branch for the stage being delivered. No feature filtering
is performed: the branch itself contains the intended files and working code.
Keep the source commit and ZIP checksum with the receiving team's acceptance record.
Transfer the ZIP, not the development repository's Git history. Initialize the
receiving repository there, then apply the next layer through its normal review process.

## Model runtime image

The example model image uses Python 3.12.14 and AutoGluon 1.2. Its model-library
versions come from
`mlops-models/src/mlops_models/model_base/requirements.constraints.txt`.
Both example models share the same resolved, hashed lockfile. The image build
installs that lockfile and runs `pip check`:

```bash
docker build --platform linux/amd64 \
  -f release/model-runtime.Dockerfile \
  -t mlops-model-runtime:ag12 .
```

The recipe pins the Python base image by digest and runs as user 10001. It
includes this template's runtime and two example model packages. It does not
install a platform's private launcher or recreate its complete Studio/tooling
image. Platform teams can install the same model lock into their approved base.

After publishing a tested image through the approved container process, supply
its ECR digest URI as `image_uri` in the deployment notebook configuration.
Existing registered models retain their recorded image and dependencies;
training with a new image creates a new model artifact.

MLflow processing jobs use their own `tracking_image_uri` and
`requirements-tracking.txt`. Notebook and tracking SDK dependencies are separate
from these model-image pins. See the
[implementation reference](../docs/implementation/README.md) for the model pins
and the public XGBoost contribution call.
