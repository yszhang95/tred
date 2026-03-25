#!/bin/bash
uv run tred -c config_10x10_300_50.yaml -l output_10x10_300_50.log fullsim -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o test_output.npz
uv run tred -c config_10x10_100_50.yaml -l output_10x10_100_50.log fullsim -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o test_output.npz
uv run tred -c config_10x10_100_10.yaml -l output_10x10_100_10.log fullsim -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o test_output.npz
