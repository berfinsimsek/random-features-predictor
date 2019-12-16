import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def normalize_data(X, cut=2):
    X2 = np.zeros((X.shape[0], (28-2*cut)**2))
    for i in range(X.shape[0]):
        im = np.reshape(X[i][:], (28,28))
        im_cut = im[cut:28-cut, cut:28-cut].flatten()
        X2[i][:] = im_cut / 255 - np.mean(im_cut / 255)
    return X2

def make_two_classes(label1, label2, X, Y, X2, Y2):
    c7 = Y == label1; c9 = Y == label2;
    Y_trn = np.concatenate((np.ones(Y[c7].shape), -np.ones(Y[c9].shape)))
    X_trn = np.concatenate((X[c7][:], X[c9][:]), axis=0) 

    c7_t = Y2 == label1; c9_t = Y2 == label2;
    Y_tst = np.concatenate((np.ones(Y2[c7_t].shape), -np.ones(Y2[c9_t].shape)))
    X_tst = np.concatenate((X2[c7_t][:], X2[c9_t][:]), axis=0)

    X_trn = normalize_data(X_trn); X_tst = normalize_data(X_tst); 
    return X_trn, Y_trn, X_tst, Y_tst

def gauss_ker(x1,x2,l=0.1):
    return np.exp(-np.linalg.norm(x1-x2)**2/(2*l))

def make_cov_matrix(X):
    C = np.zeros((X.shape[0], X.shape[0])) # covariance matrix
    for i1 in range(X.shape[0]):
        for i2 in range(X.shape[0]):
            C[i1,i2] = gauss_ker(X[i1,:], X[i2,:])
    return C

def pred(X1, X2, Y1, Y2, P):
    X_all = np.concatenate((X1, X2), axis=0)
    C = make_cov_matrix(X_all)
    F_all = np.zeros((X_all.shape[0], X1.shape[1])) # make random features 
    D, U = np.linalg.eig(C); D[D < 0] = 0;
    R = np.matmul(U, np.diag(np.sqrt(D)))
    F_all = np.matmul(R, np.random.normal(0, 1, size=(X_all.shape[0],P)))
    F1 = F_all[:X1.shape[0],:]; F2 = F_all[X1.shape[0]:,:];
    B = np.linalg.pinv(np.matmul(F1.T, F1)) 
    theta = np.matmul(np.matmul(B, F1.T), Y1) # the optimal predictor 
    Y1_pred = np.matmul(F1, theta); Y2_pred = np.matmul(F2, theta) # predictions on training and test data
    return Y1_pred, Y2_pred

def KernelPredictor(X, Y, X2, lambd=0):
    # for lambd /neq 0, non-interpolating solutions on training data should be coded
    C = make_cov_matrix(X); C_inv = np.linalg.inv(C + lambd*np.eye(X.shape[0]))
    Y_pred = np.zeros(X2.shape[0])
    for i in range(X2.shape[0]):
        x = X2[i,:]
        Cx = np.zeros(X.shape[0]);
        for j in range(X.shape[0]):
            Cx[j] = gauss_ker(x, X[j,:]);
        Y_pred[i] = np.matmul(np.matmul(Cx, C_inv), Y)
    return Y_pred

def expectedRFPredictor(X, X2, Y, Y2, P, lambd=0, num_trials = 50):
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