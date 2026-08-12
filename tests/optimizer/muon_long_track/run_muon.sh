#!/bin/bash
# Long-muon-track lifetime fit. All outputs (log, npz) stay in this directory.
cd "$(dirname "$0")"
uv run tred -l optimized_muon.log -c config_muon_long_track.yaml train -i muon_track.hdf5
