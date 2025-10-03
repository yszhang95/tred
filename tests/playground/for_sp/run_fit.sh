#!/bin/bash
# smooth_kernel_size: [1, 11, 15]
# gaussian_sigma: [0.0, 1.0, 2.0]
# lam_l1: [1E-8, 1E-2, 10]
# lam_l2: [1E-8, ]
# lam_dx: [1E-2, ]
# lam_a0: [1E-8, 1E-2, 10]
# noise: [1.0]
# pre_n: [5, 15]
# post_n: [5,]
for smooth_kernel_size in 1 11 15; do
    for gaussian_sigma in 0.0 1.0 2.0; do
        for lam_l1 in 1E-8 1E-2 10; do
            for lam_l2 in 1E-8; do
                for lam_dx in 1E-2; do
                    for lam_a0 in 1E-8 1E-2 10; do
                        for noise in 1.0; do
                            for pre_n in 5 15; do
                                for post_n in 5; do
                                    echo "Running with parameters: smooth_kernel_size=$smooth_kernel_size, gaussian_sigma=$gaussian_sigma, lam_l1=$lam_l1, lam_l2=$lam_l2, lam_dx=$lam_dx, lam_a0=$lam_a0, noise=$noise, pre_n=$pre_n, post_n=$post_n"
                                    uv run --with plotly --with kaleido python fit_deconv3d.py --smooth_kernel_size $smooth_kernel_size --gaussian_kernel_sigma $gaussian_sigma --lam_l1 $lam_l1 --lam_l2 $lam_l2 --lam_dx $lam_dx --lam_a0 $lam_a0 --noise $noise --pre_n $pre_n --post_n $post_n
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
# how many events do I run from combinations above?
