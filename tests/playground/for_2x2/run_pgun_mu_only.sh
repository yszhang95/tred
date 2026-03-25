#!/bin/bash
# uv run tred -c config_for_pgun_mu_only.yaml fullsim -i /home/yousen/Public/ndlar_shared/data/tred_particle_gun_20250625/particle_gun_mu_only.hdf5 -o pgun_mu_only.npz
uv run tred -c config_for_pgun_mu_only.yaml fullsim -i /home/yousen/Public/ndlar_shared/data/tred_particle_gun_20250625/pgun_mu_only_transformed.hdf5 -o pgun_mu_only_transformed.npz
