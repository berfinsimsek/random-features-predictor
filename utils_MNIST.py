import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from utils import *

def normalize_data(X, cut=2):
    # normalising the MNIST dataset (X)
    Xnorm = np.zeros((X.shape[0], (28-2*cut)**2))
    for i in range(X.shape[0]):
        im = np.reshape(X[i][:], (28,28))
        im_cut = im[cut:28-cut, cut:28-cut].flatten()
        Xnorm[i][:] = im_cut / 255 - np.mean(im_cut / 255)
    return Xnorm

def make_two_classes(label1, label2, X, Y, X2, Y2):
    # given training (X,Y) and test datasets (X2, Y2), we choose two classes,
    # apply normalization and return the (modified) training and test datasets
    c7 = Y == label1; c9 = Y == label2;
    Y_trn = np.concatenate((np.ones(Y[c7].shape), -np.ones(Y[c9].shape)))
    X_trn = np.concatenate((X[c7][:], X[c9][:]), axis=0) 

    c7_t = Y2 == label1; c9_t = Y2 == label2;
    Y_tst = np.concatenate((np.ones(Y2[c7_t].shape), -np.ones(Y2[c9_t].shape)))
    X_tst = np.concatenate((X2[c7_t][:], X2[c9_t][:]), axis=0)

    X_trn = normalize_data(X_trn); X_tst = normalize_data(X_tst); 
    return X_trn, Y_trn, X_tst, Y_tst

def KernelPredictor(X, Y, X2, lambd=0):
    # given training dataset (X, Y), test inputs X2, and ridge lambd,
    # returns the predictions of the Kernel function on the test input 
    C = make_cov_matrix(X); 
    C_inv = np.linalg.inv(C + lambd*np.eye(X.shape[0]))
    Y_pred = np.zeros(X2.shape[0])
    for i in range(X2.shape[0]):
        x = X2[i,:]
        Cx = np.zeros(X.shape[0]);
        for j in range(X.shape[0]):
            Cx[j] = gauss_ker(x, X[j,:]);
        Y_pred[i] = np.matmul(np.matmul(Cx, C_inv), Y)
    return Y_pred

def expectedRFPredictor(X, X2, Y, Y2, P, lambd=0, num_trials = 50):
    # returns some qualitative metrics on test input data
    Y1_all = np.zeros((Y.size, num_trials))
    Y2_all = np.zeros((Y2.size, num_trials))
    aver_trn_err = 0; 
    aver_tst_err = 0;

    for i in range(num_trials):
        Y1_pred, Y2_pred = pred(X, X2, Y, Y2, P)
        Y1_all[:,i] = Y1_pred; Y2_all[:,i] = Y2_pred;
        aver_trn_err += np.linalg.norm(Y1_pred - Y)**2 / Y.size
        aver_tst_err += np.linalg.norm(Y2_pred - Y2)**2 / Y2.size

    aver_trn_err = aver_trn_err / num_trials
    aver_tst_err = aver_tst_err / num_trials
   
    Y2_mean = np.mean(Y2_all, axis=1)
    bias = np.linalg.norm(Y2_mean - Y2)**2 / Y2.size
    var = 0;
    for i in range(Y2.size):
        var += np.linalg.norm(Y2_all[i,:] - Y2_mean[i])**2 / num_trials
    var = var / Y2.size
    return aver_trn_err, aver_tst_err, bias, var