#!/usr/bin/env bash
# Point-charge study: 8 ke- point charges, 10000 iterations, noises off, TPC 0.
# Run from the repo root so the relative geometry yaml paths resolve.
set -e
cd "$(git rev-parse --show-toplevel)"
uv run tred -L warning -c tests/pointq/pointq_8ke/config.yaml \
    fullsim -o tests/pointq/pointq_8ke/pointq_8ke_10000.npz
