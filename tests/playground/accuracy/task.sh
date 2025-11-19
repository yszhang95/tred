set -x
./run.sh config_10x10_4pt.yaml output_10x10_4pt.npz
./run.sh config_10x10.yaml output_10x10.npz
./run.sh config_10x10_1pt.yaml output_10x10_1pt.npz

./run.sh config_8x8.yaml output_8x8.npz
./run.sh config_6x6.yaml output_6x6.npz

./run.sh config_8x8_1pt.yaml output_8x8_1pt.npz
./run.sh config_6x6_1pt.yaml output_6x6_1pt.npz

./run.sh config_8x8_4pt.yaml output_8x8_4pt.npz
./run.sh config_6x6_4pt.yaml output_6x6_4pt.npz
set +x
