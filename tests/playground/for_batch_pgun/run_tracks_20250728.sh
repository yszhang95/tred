#!/bin/bash
# skip # in the file
# edit 20250728
event_ids=( $(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/event_list.txt) ) # works
for i in ${event_ids[@]}; do
    # skip # in the file
    if [[ $i == \#* ]]; then
        echo "Skipped comment line: $i"
        continue
    fi
  echo "Processing event ID: $i"
  # edit: no reset time
  # InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}.hdf5"
  # OutFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}_noreset_tred.npz"
  # echo "uv run tred -c config_for_pgun_noreset.yaml fullsim -i $InFile -o $OutFile"
  # CUDA_LAUNCH_BLOCKING=1 uv run tred -c config_for_pgun_noreset.yaml fullsim -i $InFile -o $OutFile

  # edit: no reset time + adc_hold_delay 1.8us
  InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}.hdf5"
  OutFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_event_id${i}_delay18_noreset_tred.npz"
  ConfigFile="config_for_pgun_delay18_noreset.yaml "
  echo "uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile"
  CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile
done
