#!/bin/bash
#
# Same 2-pass prompt-hits readout as run_tracks_thres_prompt_hits_2pass_edgetrim5_20260701.sh,
# but triggering on the peak-mode threshold table
#   threshold_peakmode_edgetrim5_20260701.hdf5
# (peakmode table consumed directly; no edgetrim variant — see
#  2x2_ql/threshold_studies/algorithm.md §4 and REPORT.md for why the
#  edgetrim step was dropped).
# The old cluster_highstat table sat ~1.6 ke below the operating thresholds
# (inflection estimator lands at the foot of the noise-peak edge); this run
# tests whether the corrected table aligns the low-Q totQ peak with data.

ConfigFile="config_for_pgun_thres_prompt_hits_2pass_peakmode_dbin_hd18_prcsync_thres01_20260714.yaml"
while IFS= read -r InFile; do
    echo "Processing $InFile"
    bname=$(basename "$InFile")
    odir="${bname%.hdf5}"
    if [[ -d $odir ]]; then
        echo "Output directory $odir already exists."
    else
        mkdir "$odir"
    fi
    InFile="/home/yousen/Documents/NDLAr2x2/MuonLArSim/${odir}/pgun_mu_3GeV_2mm.hdf5"
    OutFile="${odir}/pgun_mu_3GeV_2mm_thres_prompt_hits_2pass_peakmode_dbin_hd18_prcsync_thres01_20260714.npz"
    OutLog="${OutFile/npz/log}"
    if [[ -f $OutFile ]]; then
        echo "Output file $OutFile already exists. Skipping."
        continue
    fi
    echo "uv run tred -c $ConfigFile -l $OutLog fullsim -i $InFile -o $OutFile"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile -l $OutLog fullsim -i $InFile -o "$OutFile"
done < "/home/yousen/Documents/NDLAr2x2/MuonLArSim/run_list.txt"
echo "SIM DONE"
