#!/bin/bash
# skip # in the file
# edit 20250730
event_ids=( $(grep -v '^#' /home/yousen/Public/ndlar_shared/data_reflowv5_20250722/event_list.txt) ) # works
for i in ${event_ids[@]}; do
    # skip # in the file
    if [[ $i == \#* ]]; then
        echo "Skipped comment line: $i"
        continue
    fi
  echo "Processing event ID: $i"

  # edit 20250730; xoffset 0.5cm to anode
  InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_event_id${i}.hdf5"
  OutFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_0p5cm_event_id${i}_tred.npz"
  ConfigFile="config_for_pgun.yaml "
  echo "uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile"
  CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile

  # edit 20250730; xoffset 1cm to anode
  InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_1p0cm_event_id${i}.hdf5"
  OutFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_1p0cm_event_id${i}_tred.npz"
  ConfigFile="config_for_pgun.yaml "
  echo "uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile"
  CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile

  # edit 20250730; xoffset 2cm to anode
  InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_2p0cm_event_id${i}.hdf5"
  OutFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/build/pgun_mu_3GeV_2mm_xoffset_2p0cm_event_id${i}_tred.npz"
  ConfigFile="config_for_pgun.yaml "
  echo "uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile"
  CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile
done
