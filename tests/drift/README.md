# Drift Test Notes

`tests/drift/` contains drift-focused tests and several plotting helpers used to
inspect transport behavior, timing, charge attenuation, and diffusion trends.

## Current Status

- The assertion-driven tests remain the primary correctness checks.
- The plot-producing helpers are still mainly diagnostic tools for manual
  inspection.

## Plot Collections

Standardized collections of inspection plots are still being developed. At the
moment, the plotting helpers in this directory are useful for targeted
debugging, but they should not yet be treated as a stable reference-image
suite.
