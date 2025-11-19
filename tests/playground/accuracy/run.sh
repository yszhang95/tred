#!/bin/bash
ConfigYaml=$1
OutputFile="$2"
BaseName=${OutputFile%.npz}
rm "${BaseName}.log"
uv run tred -c ${ConfigYaml} -l ${BaseName}.log fullsim \
   -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
   -o ${OutputFile}
