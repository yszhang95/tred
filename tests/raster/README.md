# Raster Test Notes

`tests/raster/` currently contains visualization-oriented smoke coverage around
the interaction between `Drifter` and `Raster`.

## Current Status

- `tests/raster/test_raster_graph.py` is useful for manual inspection.
- It is not yet a strong unit test because it mainly plots and prints instead
  of asserting detailed numerical behavior.

## Plotting Guidance

- These plots are best treated as diagnostic artifacts, not as the primary test
  oracle.
- When turning a plotting workflow into a unit test, the preferred structure
  is:
  - compute the numeric outputs in a helper
  - assert on those outputs in pytest
  - keep plotting as an opt-in debug path

## Plot Collections

Curated plot sets for routine inspection are still being standardized. For now,
the existing plotting helpers remain useful for ad hoc debugging, but they are
not yet organized as a stable reference-figure collection.
