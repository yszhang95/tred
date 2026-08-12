#!/bin/bash
# Muon-track lifetime fit, wide starts (2.0 / 0.5 ms), sigma = 170 e-/tick.
# All outputs (log, npz) stay in this directory.
cd "$(dirname "$0")"
uv run tred -l optimized_widestart_170.log -c config.yaml train -i ../muon_long_track/muon_track.hdf5
