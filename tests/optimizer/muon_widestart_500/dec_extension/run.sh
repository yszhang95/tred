#!/bin/bash
# Dec-arm extension for the 500 e-/tick wide-start test.
cd "$(dirname "$0")"
uv run tred -l optimized_dec_extension.log -c config.yaml train -i ../../muon_long_track/muon_track.hdf5
