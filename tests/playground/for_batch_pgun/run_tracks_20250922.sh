#!/bin/bash

# run with additional noise on threshold at the end of ADC_HOLD_DELAY

ConfigFile="config_for_pgun.yaml"
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
    OutFile="${odir}/pgun_mu_3GeV_2mm_20250922.npz"
    OutLog="${OutFile/npz/log}"
    if [[ -f $OutFile ]]; then
        echo "Output file $OutFile already exists. Skipping."
        continue
    fi
    echo "uv run tred -c $ConfigFile -l $OutLog fullsim -i $InFile -o $OutFile"
    CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile -l $OutLog fullsim -i $InFile -o "$OutFile"
done < "/home/yousen/Documents/NDLAr2x2/MuonLArSim/run_list.txt"
