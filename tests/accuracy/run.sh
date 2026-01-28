uv run tred -c config_10x10_4pt.yaml -l output_10x10_4pt.log fullsim \
    -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
   -o output_10x10_4pt.npz

uv run tred -c config_10x10_2pt.yaml -l output_10x10_2pt.log fullsim \
    -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
   -o output_10x10_2pt.npz

uv run tred -c config_10x10_1pt.yaml -l output_10x10_1pt.log fullsim \
    -i /home/yousen/Public/ndlar_shared/data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
   -o output_10x10_1pt.npz
