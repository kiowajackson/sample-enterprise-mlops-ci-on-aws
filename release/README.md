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
 Only the two explicitly listed docs/ guides are included; other planning
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
