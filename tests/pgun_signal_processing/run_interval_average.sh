#!/bin/bash

uv run tred -c config_noises.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o pgun_muplus_3gev_noises_interval_average.npz
uv run tred -c config_noises_offset8.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o pgun_muplus_3gev_noises_offset8_interval_average.npz
uv run tred -c config_noises_offset16.yaml fullsim -i /home/yousen/Documents/NDLAr2x2/MuonLArSim/pgun_muplus_3gev.hdf5 -o pgun_muplus_3gev_noises_offset16_interval_average.npz
