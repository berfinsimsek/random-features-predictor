# Implicit Regularization in Random Feature Models
We analyze the distribution of $\lambda$-RF$ (Random Features) predictor and in particular, how close it is to the $\tilde{\lambda}$-KRR (Kernel Ridge Regression) predictor.

We provide Jupyter notebooks to replicate the plots found in the paper and explore the regimes beyond what is presented in the paper. In particular, we provide:

(1) RF-predictor.ipynb

For a sinusoidal data, presents how the distribution of the $\lambda$-RF$ predictor changes for different $\lambda$ and $P$.

(2) evolution-plots.ipynb

We investigate the behavior of the effective and its derivative for different eigenvalue spectrums $D$ (i.e. with exponential and polynomial decays). We also show that the eigenvalues of $\mathbb E A_{\lambda}$ converge to $d_i / d_i + \tilde{\lambda}$.
