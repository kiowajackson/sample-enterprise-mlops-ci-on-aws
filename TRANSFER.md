# Latest complete handoff

[Download the complete password-protected ZIP](mlops-models/scratch/wsf-full-handoff-2026-10-15T195327Z-password-protected.zip) — password: `pass`.

Source commit: `00011f94548bcd29e29f42d0ac301eef441e2b3f`. Contains 111 source files, including the two current notebooks. This replaces T190829Z and retains required Model Registry integration and independent compute-step caching.

[Exact changed-file list](mlops-models/scratch/wsf-full-handoff-2026-10-15T195327Z-password-protected-changes.md) · [ZIP checksum](mlops-models/scratch/wsf-full-handoff-2026-10-15T195327Z-password-protected.zip.sha256)

Unpack into a fresh folder and carry over your customer model code and populated notebooks/config.json through version control. The ZIP is the complete generic baseline. Loose files on this branch are matching replacements for review; older timestamped ZIPs are historical. Model algorithm files are unchanged in this update.

Copy notebooks/config.example.json to notebooks/config.json once and fill in the required inputs. Both notebooks read that file. Notebook 01 saves csv_input or feature_store_config_uri on success; notebook 02 reads it directly. No notebook cell edits or copied variables are needed. Relative input paths start at the project root. Existing config.json is never replaced by startup, is Git-ignored, and is excluded from exported archives.

Set registry_group_prefix to the exact prefix, including its trailing separator. Groups are `<prefix><ModelId>-sm-<environment>`. Keep model IDs/folders and environment lowercase. Preflight, registration and inference use the same group name. The generic examples remain usecase1/usecase2; rename them inside your environment. Rerun notebook 02 to publish the pipeline changes.

Registration and inference require Model Registry. RegisterModel writes a pending version to an existing group under the job role; approval is an operator action. No Registry fallback or group creation is performed.

92 source tests and 25 simulated-AWS notebook cases passed; the 92 tests also passed from the extracted archive. This configuration update did not launch managed jobs. The source contains generic code and synthetic examples, blank connection placeholders and cleared notebook outputs. No customer logs, populated config or captured permission values are included.
