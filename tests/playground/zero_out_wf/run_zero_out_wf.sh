#!/bin/bash
OutputFile="$1"
uv run tred -c config.yaml fullsim \
   -i /home/yousen/Public/ndlar_shared/data/MR5_2x2_single_particles/segments_pid13.hdf5 \
   -o $OutputFile
