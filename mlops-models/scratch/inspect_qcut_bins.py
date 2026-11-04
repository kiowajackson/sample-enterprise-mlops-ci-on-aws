# %% Diagnose qcut at the failing line.
# Paste into the notebook/debug console where top_decile already exists.
import pandas as pd

scores = top_decile["SCORE"]

print("Non-missing rows:", scores.count())
print("Unique scores:", scores.nunique())
print("\nMost frequent scores:")
print(scores.value_counts(dropna=False).head(10))

requested_edges = scores.quantile([i / 10 for i in range(11)])
print("\nRequested quantile edges:")
print(requested_edges.to_string(float_format=lambda value: f"{value:.17g}"))
print("\nRepeated quantile edges:")
print(requested_edges[requested_edges.duplicated(keep=False)])

groups, edges = pd.qcut(scores, q=10, duplicates="drop", retbins=True)
print("\nBins remaining after duplicate edges are dropped:", len(edges) - 1)
print("Remaining edges:", edges)
print("\nRows per bin:")
print(groups.value_counts(sort=False))

# %% Optional fix: use only if fewer than ten groups are acceptable.
# Dropping duplicate edges changes the number of bins, so label AFTER binning.
# Highest-score group gets label 1. Equal scores stay together.
# This defines groups for inspection; it does not overwrite top_decile.
groups = groups.cat.rename_categories(range(len(groups.cat.categories), 0, -1))
print("\nDescending group labels (1 = highest scores):")
print(groups.value_counts(sort=False))

# If downstream code requires exactly ten groups, agree on how to handle ties.
# rank(method="first") would split tied scores according to their row order.
# Do not silently use that if the report expects equal scores to stay together.
# Also trace unexpectedly frequent ties to the scoring output and any rounding.
