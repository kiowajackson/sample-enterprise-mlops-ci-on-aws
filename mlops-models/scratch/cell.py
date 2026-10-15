# Copy this whole file into one notebook cell and run it.
# This file is the reusable place for the current diagnostic.
# The starter cell only displays the active Python environment and directory.

from pathlib import Path
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("Working directory:", Path.cwd())
