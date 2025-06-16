#!/bin/bash
set -euo pipefail

# ?config_for_sp_unipolar_thres0p5k.yaml? USER SETTINGS ??????????????????????????????????????????????????????????????

# directory where your .hdf5 files live
INPUT_DIR="/nfs/data/1/yousen/signal_processing/20250616"

# suffixes for your thresholded configs:
#   ""               ? config_for_sp.yaml
#   "_thres0p5k"     ? config_for_sp_thres0p5k.yaml
#   "_thres7k"       ? config_for_sp_thres7k.yaml
SUFFIXES=( "" "_thres0p5k" "_thres7k" )

# ?? LOOP & LAUNCH ??????????????????????????????????????????????????????????????

for file in "${INPUT_DIR}"/segments_pid*_angle*.hdf5; do
  # get just the filename, e.g. segments_pid13_angle30.hdf5
  fname=$(basename "$file")
  # strip extension
  base=${fname%.hdf5}
  # extract pid and angle via parameter expansion
  #   from "segments_pid13_angle30" ? pid=13, angle=30
  pid=${base#segments_pid}; pid=${pid%%_angle*}
  angle=${base##*_angle}

  for suf in "${SUFFIXES[@]}"; do
    # build config + output names for the "sp" run
    cfg_sp="config_for_sp${suf}.yaml"
    out_sp="single_track_for_sp${suf}_pid${pid}_angle${angle}.npz"

    echo "? Running SP:   uv run tred -c ${cfg_sp} fullsim -i ${file} -o ${out_sp}"
    uv run tred -c "${cfg_sp}" fullsim -i "${file}" -o "${out_sp}"
    mv ${out_sp} $INPUT_DIR

    # build config + output names for the "unipolar_sp" run
    cfg_up="config_for_sp_unipolar${suf}.yaml"
    out_up="single_track_for_sp_unipolar${suf}_pid${pid}_angle${angle}.npz"

    echo "? Running UNI:  uv run tred -c ${cfg_up} fullsim -i ${file} -o ${out_up}"
    uv run tred -c "${cfg_up}" fullsim -i "${file}" -o "${out_up}"
    mv ${out_up} $INPUT_DIR
  done
done
