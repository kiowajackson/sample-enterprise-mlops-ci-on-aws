# Latest complete handoff

[Download the complete password-protected ZIP](mlops-models/scratch/wsf-full-handoff-2026-10-15T190829Z-password-protected.zip) — password: `pass`.

Source commit: `17a39de55e1fe7e6b44318e5fd0e1274ee87c3b3`. Contains 109 source files, including the two current notebooks. This replaces the earlier T145812Z delivery and includes the later cache improvements and required Model Registry integration.

[Exact changed-file list](mlops-models/scratch/wsf-full-handoff-2026-10-15T190829Z-password-protected-changes.md) · [ZIP checksum](mlops-models/scratch/wsf-full-handoff-2026-10-15T190829Z-password-protected.zip.sha256)

Unpack into a fresh folder and carry over your customer model code and configuration through version control. The ZIP is the complete generic baseline. Loose files on this branch are matching replacements for review; older timestamped ZIPs are historical.

Registration and inference now require Model Registry. RegisterModel is an uncached Processing job which writes a pending version into an existing group. The job role needs DescribeModelPackageGroup, CreateModelPackage and DescribeModelPackage; ResolveModel also needs ListModelPackages. Approval is an operator action. No Registry fallback or group creation is performed.

Preserve your notebook input cell. Before rerunning notebook 02, ensure model folder names and environment match the pre-created `<ModelId>-sm-<environment>` groups. Republish both pipelines to apply the changes.

The source contains generic code and synthetic examples. Notebook connection inputs are blank and saved execution outputs are cleared; no customer logs or captured permission values are included.
