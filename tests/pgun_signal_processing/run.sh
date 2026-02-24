#!/bin/bash

uv run tred -c config_noises.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_positron_3gev.hdf5 -o pgun_positron_3gev_tred_noises.npz
uv run tred -c config_noises.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o pgun_muplus_3gev_tred_noises.npz
uv run tred -c config_nburst4_noises.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_positron_3gev.hdf5 -o pgun_positron_3gev_tred_nburst4_noises.npz
uv run tred -c config_nburst4_noises.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o pgun_muplus_3gev_tred_nburst4_noises.npz
uv run tred -c config_nburst4_nonoises_nd_readout.yaml fullsim -i  /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o  pgun_muplus_3gev_tred_nburst4_nonoises_nd_readout.npz
uv run tred -c config_nburst4_noises_nd_readout.yaml fullsim -i  /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o  pgun_muplus_3gev_tred_nburst4_noises_nd_readout.npz
