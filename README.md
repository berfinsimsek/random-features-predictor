# Random Features (RF) Predictor

This is the repo for our ICML paper "Implicit Regularization of Random Feature Models"

https://proceedings.icml.cc/paper/2020/hash/de043a5e421240eb846da8effe472ff1

We provide Jupyter notebooks to replicate some plots from the paper and see how the effective ridge behaves for different spectrums, ridge, and number of features. In addition, we present the mean and the variance of the RF Predictor in the function space on three Kernels.

## Content

(1) evolution-eff-ridge.ipynb

Some figures from the appendix are presented. In particular, we investigate the behavior of the effective ridge and its derivative for eigenvalue spectrums with exponential and polynomial decays.

(2) movie-RF-pred.ipynb

The RF predictor on sinusodial dataset is presented.

(3) movie-eigs-A.ipynb

We show two approximations for the eigenvalues of the average hat matrix: (1) the classical approximation when the number of features is infinity and when the effective ridge converges to the original ridge, (2) our approximation when the number of features is finite with the effective ridge bigger than the original ridge.

(4) utils.py

Helper functions in particular a fixed point solver for the effective ridge.
