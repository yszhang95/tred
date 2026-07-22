#!/bin/bash
#
# Same 2-pass prompt-hits readout as run_tracks_thres_prompt_hits_2pass_20260629.sh,
# but with the edge-trimmed threshold map:
#   threshold_july10to11_bin0p2_cluster_highstat_edgetrim5.hdf5
# In that file the 5-channel border is set to Q=1E16 (disabled) and low
# thresholds are raised, so the code no longer disables low channels itself:
#
# diff --git a/src/tred/plots/graph_effq.py b/src/tred/plots/graph_effq.py
# --- a/src/tred/plots/graph_effq.py
# +++ b/src/tred/plots/graph_effq.py
# @@ -452,8 +452,8 @@ def runit(device='cpu'):
#                  # if isinstance(threshold, str):
#                  #     raise NotImplementedError("To add support for loading a threshold file.")
#                  thres = thresholds[tpcdataset.tpc_id].to(device)
# -                if thres.ndim > 0:
# -                    thres[thres<2] = 1E16 # FIXME: Temporarily disable low threshold channels
# +                # if thres.ndim > 0:
# +                #     thres[thres<2] = 1E16 # FIXME: Temporarily disable low threshold channels
#                 hits = nd_readout(currents, thres, adc_hold_delay, adc_down_time, csa_reset_time, one_tick=one_tick,
#                                     offset_to_align=0, # FIXME: how to calculate properly?
#                                     pixel_axes=(1,2), uncorr_noise=uncorr_noise, thres_noise=thres_noise, reset_noise=reset_noise)

# run with additional noise on threshold at the end of ADC_HOLD_DELAY

ConfigFile="config_for_pgun_thres_prompt_hits_2pass_edgetrim5_20260701.yaml"
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
    OutFile="${odir}/pgun_mu_3GeV_2mm_thres_prompt_hits_2pass_edgetrim5_20260701.npz"
    OutLog="${OutFile/npz/log}"
    if [[ -f $OutFile ]]; then
        echo "Output file $OutFile already exists. Skipping."
        continue
    fi
    echo "uv run tred -c $ConfigFile -l $OutLog fullsim -i $InFile -o $OutFile"
    CUDA_LAUNCH_BLOCKING=1 uv run tred -c $ConfigFile -l $OutLog fullsim -i $InFile -o "$OutFile"
done < "/home/yousen/Documents/NDLAr2x2/MuonLArSim/run_list.txt"
