import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import fsolve

def make_eig_spect(N, exp=True, exp_decay=0.5, pol_decay=1):
    # makes an eigenvalue spectrum (of the Kernel) with desired decay properties:
    # flag=1 for the exponential decay with exp(-beta*(i-1)), and
    # flag=0 is for the polynomial decay with i**(-alpha) for i=1,...,N.
    if(exp): 
        return np.exp(-exp_decay*np.linspace(0,N-1,N))
    else: 
        return np.linspace(1,N,N)**(-pol_decay)     

def calc_emp_eigs(D, P, lambd, num_trials=1000):
    # for given eigenvalue spectrum D and a ridge lambd, calculates the empirical average and variance   
    # of the diagonal entries of A_\lambda for P=1,...,N. 
    N = D.size
    A = np.zeros((N, num_trials))
    for it in range(num_trials):
        W = np.random.normal(0,1,size=(N,P))
        B = np.linalg.inv(np.matmul(np.matmul(W.T,np.diag(D)),W)/P + lambd*np.eye(P))
        for i in range(N):
            A[i, it] = D[i]*np.inner(W[i,:],np.matmul(B, W[i,:]))/P
    return np.mean(A, axis=1), np.sqrt(np.var(A, axis=1))

def calc_kernel_eigs(D, lambd):
    # for given eigenvalue spectrum D and a ridge lambd, 
    # returns the effective eigenvalue spectrum
    N = D.size
    C = np.zeros(N)
    for i in range(N):
        C[i] = D[i] / (D[i] + lambd)
    return C

def solve_eff_ridge(D, P, lambd):
    # for given eigenvalues D, number of features P, and ridge lambd,
    # numerically solves for the effective ridge.
    N = D.size
    def f(z):
        s = 0;
        for i in range(N):
            s += D[i] / (D[i] + z) 
        return P*(1-lambd/z) - s
    sol = np.asscalar(fsolve(f, 1e-4))
    eff_ridge = max(sol, 0) # if removed small negative solutions can be found due to numerical precision
    eff_dim2 = 0
    for i in range(N):
        eff_dim2 += D[i]**2 / (D[i]+eff_ridge)**2
    eff_ridge_der = 1 / (1 - eff_dim2 / P)    
    return eff_ridge, eff_ridge_der

def calc_eff_ridges(D, lambd_list, P_list):
    # for given eigenvalues D, and a list of ridge parameters lambd_list,
    # and of number of features P, calculates the corresponding effective ridge and its derivative.
    N = D.size; 
    num_lambd = lambd_list.size;
    num_P = P_list.size;
    E = np.zeros((num_lambd, num_P));
    E_der = np.zeros((num_lambd, num_P));
    for i in range(num_lambd):
        lambd = lambd_list[i]
        for j in range(num_P):
            P = P_list[j]
            eff_ridge, eff_ridge_der = solve_eff_ridge(D,P,lambd)
            E[i, j] = eff_ridge
            E_der[i, j] = eff_ridge_der  
    return E, E_der

def pred(X1, X2, Y1, P, lambd):
    # for given training input and output data (X1, Y1), returns sample predictions 
    # for the linear regression with Gaussian features on the training and test data (X1, X2)
    X_all = np.concatenate((X1, X2), axis=0)
    C = make_cov_matrix(X_all)
    D, U = np.linalg.eig(C); 
    D[D < 0] = 0; D = D.real; # fixing small numerical perturbations
    R = np.matmul(U, np.diag(np.sqrt(D)))
    F_all = np.matmul(R, np.random.normal(0, 1, size=(X_all.shape[0], P))) # sample Gaussian features
    F1 = F_all[:X1.shape[0], :]; 
    F2 = F_all[X1.shape[0]:, :];
    B = np.linalg.pinv(np.matmul(F1.T, F1) / P + lambd*np.eye(P))
    theta = np.matmul(np.matmul(B, F1.T), Y1) # the optimal parameter 
    Y1_pred = np.matmul(F1, theta) / P
    Y2_pred = np.matmul(F2, theta) / P 
    return Y1_pred, Y2_pred

def make_cov_matrix(X):
    # making a covariance matrix for a given dataset with the RBF Kernel
    def RBF_ker(x1, x2, l=1):
        # RBF Kernel function
        return np.exp(-np.linalg.norm(x1-x2)**2/(2*l))
    
    C = np.zeros((X.shape[0], X.shape[0])) 
    for i1 in range(X.shape[0]):
        for i2 in range(X.shape[0]):
            if(X.ndim == 1):
                C[i1,i2] = RBF_ker(X[i1], X[i2])
            else:
                C[i1,i2] = RBF_ker(X[i1,:], X[i2,:])
    return C


