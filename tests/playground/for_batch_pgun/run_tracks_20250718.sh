#!/bin/bash
# skip # in the file
event_ids=( $(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250708/event_list.txt) ) # works
for i in ${event_ids[@]}; do
    # skip # in the file
    if [[ $i == \#* ]]; then
        echo "Skipped comment line: $i"
        continue
    fi
  echo "Processing event ID: $i"
  InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_event_id${i}.hdf5"
  OutFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_event_id${i}_tred.npz"
  echo "uv run tred -c config_for_pgun.yaml fullsim -i $InFile -o $OutFile"
  CUDA_LAUNCH_BLOCKING=1 uv run tred -c config_for_pgun.yaml fullsim -i $InFile -o $OutFile
done
