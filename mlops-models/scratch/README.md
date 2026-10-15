# Scratch diagnostics

This folder is for temporary debugging scripts and notebook cells. It sits outside
`src/mlops_models`, so the normal pipeline source uploader does not include it.

- `cell.py`: the current ad hoc diagnostic. Copy the whole file into a notebook
  cell and run it. Its initial version prints Python and the working directory.
- `job_log_cell.py`: copy the whole file into a notebook cell, set the job type,
  exact job name and region, then run it to save and print the complete job log.
- `training_paths.py`: two marked blocks to paste inside an existing train_model
  function. This file relies on that function's directory variables and is not
  intended to be run as a standalone notebook cell.

The root README and two notebooks on dev match the complete handoff archive
`wsf-full-handoff-2026-10-15T142349Z-password-protected.zip`. If that archive is
already extracted, those three files need no additional replacement. Future fixes
will appear on dev at their exact paths; scratch cells stay in this folder.
