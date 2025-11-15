#!/bin/bash
rm *.log test_output1.npz test_output2.npz
uv run tred -c config_8x8x6912.yaml -l output_8x8x6912.log fullsim -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o test_output.npz
uv run tred -c config_8x8x2560.yaml -l output_8x8x2560.log fullsim -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 -o test_output.npz

