#!/bin/bash
uv run tred -c config_10x10_0p05us.yaml -l output_10x10_0p05us.log fullsim -i /nfs/data/1/yousen/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o output_10x10_0p05us.npz
uv run tred -c config_10x10_0p1us.yaml -l output_10x10_0p1us.log fullsim -i /nfs/data/1/yousen/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o output_10x10_0p1us.npz
