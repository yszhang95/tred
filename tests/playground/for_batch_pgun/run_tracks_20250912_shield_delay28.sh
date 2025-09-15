#!/bin/bash

ConfigFile="config_for_pgun_shield_delay28.yaml"
while IFS= read -r InFile; do
    echo "Processing $InFile"
    # python prepare.py "$InFile"
    bname=$(basename "$InFile")
    odir="${bname%.hdf5}"
    if [[ -d $odir ]]; then
        echo "Output directory $odir already exists."
    else
        mkdir "$odir"
    fi
    InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/${odir}/pgun_mu_3GeV_2mm.hdf5"
    OutFile="${odir}/pgun_mu_3GeV_2mm_shield_delay28.npz"
    if [[ -f $OutFile ]]; then
        echo "Output file $OutFile already exists. Skipping."
        continue
    fi
    echo "uv run tred -c $ConfigFile fullsim -i $InFile -o $OutFile"
    CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile fullsim -i $InFile -o "$OutFile"
done < "/home/yousen/Documents/NDLAr2x2/MuonLArSim/run_list.txt"
