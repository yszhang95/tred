#!/bin/bash
ConfigYaml=$1
OutputFile="$2"
uv run tred -c ${ConfigYaml} fullsim \
   -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
   -o ${OutputFile}
