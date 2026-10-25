"""Paste this file into a notebook cell and set the two values below.

In a notebook, DATA_PATH must be the prepared data.csv S3 URI recorded on the
failed Score job. The /opt/ml path below exists inside that job's container.
PRIMARY_KEY must match the exact printed self.primary_key, including its case.
Reads only the CSV header; does not modify files, pad columns or run prediction.
"""

DATA_PATH = "/opt/ml/processing/prepared/data.csv"
PRIMARY_KEY = "REPLACE_WITH_EXACT_SELF_PRIMARY_KEY"

from difflib import get_close_matches

import pandas as pd

if PRIMARY_KEY == "REPLACE_WITH_EXACT_SELF_PRIMARY_KEY":
    raise ValueError("Set PRIMARY_KEY to the exact value of self.primary_key.")

header = pd.read_csv(DATA_PATH, nrows=0)
original_columns = list(header.columns)
header.columns = header.columns.map(str.upper)  # Same operation as the predictor.
uppercase_columns = list(header.columns)

print("CSV:", DATA_PATH)
print("Expected primary key:", repr(PRIMARY_KEY))
print("Column count:", len(original_columns))
print("Key before uppercase:", PRIMARY_KEY in original_columns)
print("Key after uppercase:", PRIMARY_KEY in uppercase_columns)
print("First 20 original headers:", [repr(c) for c in original_columns[:20]])
matches = [
    (repr(original), repr(upper))
    for original, upper in zip(original_columns, uppercase_columns)
    if upper.strip() == PRIMARY_KEY.strip().upper()
]
print("Case/whitespace matches (original, uppercase):", matches)
print("Closest uppercase headers:", [
    repr(c) for c in get_close_matches(PRIMARY_KEY.upper(), uppercase_columns, n=5)
])

if PRIMARY_KEY in uppercase_columns:
    print("The CSV contains the key. Compare the actual dataframe at the failing line.")
elif matches:
    print("The names differ in case or whitespace. Inspect the quoted names above.")
else:
    print("The key is absent from this CSV header. Check Preprocess's input and output.")

# If the CSV contains the key, paste these inside the predictor before selection:
# print("Actual primary key:", repr(self.primary_key))
# print("Key in df_test:", self.primary_key in df_test.columns)
# print("Index names:", df_test.index.names)
