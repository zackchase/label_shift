# From scratch implementation of two-sample tests.
# author: yu-xiang wang
#  hotelling t^2 test:  see wikipedia: https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
#  btest_mmd:  the  subquadratic and consistent mmd test:
#               https://papers.nips.cc/paper/5081-b-test-a-non-parametric-low-variance-kernel-two-sample-test.pdf

from __future__ import division, print_function
import os
import math

import numpy as np
from scipy import linalg, stats

def linear_hotelling_test(X, Y, reg=0):
    n, p = X.shape
    Z = X - Y
    Z_bar = Z.mean(axis=0)

    Z -= Z_bar
    S = Z.T.dot(Z)
    S /= (n - 1)
    if reg:
        S[::p + 1] += reg
    # z' inv(S) z = z' inv(L L') z = z' inv(L)' inv(L) z = ||inv(L) z||^2
    L = linalg.cholesky(S, lower=True, overwrite_a=True)
    Linv_Z_bar = linalg.solve_triangular(L, Z_bar, lower=True, overwrite_b=True)
    stat = n * Linv_Z_bar.dot(Linv_Z_bar)

    p_val = stats.chi2.sf(stat, p)
    return p_val, stat


def btest_mmd_python(X,Y, bandwidth='median',median_samples=100):
    if bandwidth == 'median':
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        from sklearn.metrics.pairwise import euclidean_distances
        Z = np.r_[sub(X, median_samples // 2), sub(Y, median_samples // 2)]
        D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = np.median(upper, overwrite_input=True)/math.sqrt(2)
        bandwidth = np.sqrt(kernel_width / 2)
        # sigma = median / sqrt(2); works better, sometimes at least
        del Z, D2, upper
    else:
        kernel_width = 2 * bandwidth**2

    m = X.shape[0]
    Z = np.concatenate((X,Y),axis=0)
    ker =lambda a,b: np.exp(-np.sum((a-b)**2,axis=1)/kernel_width*0.5).reshape((a.shape[0],1))
    blocksize = int(round(math.sqrt(m)))

    assert(blocksize >= 2 and blocksize < m/3),"Data size too small!"

    m2 = int(m//blocksize)

    hh = np.zeros(shape=(m2,1))
    hh_null = np.zeros(shape=(m2,1))

    subidx = lambda feats,n : np.random.choice(
        feats.shape[0], min(feats.shape[0], n), replace=False)

    for i in range(blocksize):
        for j in range(i+1,blocksize):
            idx1 = range(m2*i,m2*(i+1))
            idx2 = range(m2*j,m2*(j+1))
            hh += ker(X[idx1,:],X[idx2,:])
            hh += ker(Y[idx1,:],Y[idx2,:])
            hh -= ker(X[idx1,:],Y[idx2,:])
            hh -= ker(Y[idx1,:],X[idx2,:])

            idx1X = subidx(Z,m2)
            idx2X = subidx(Z,m2)
            idx1Y = subidx(Z,m2)
            idx2Y = subidx(Z,m2)

            hh_null += ker(Z[idx1X,:], Z[idx2X,:])
            hh_null += ker(Z[idx1Y,:], Z[idx2Y,:])
            hh_null -= ker(Z[idx1X,:], Z[idx2Y,:])
            hh_null -= ker(Z[idx1Y,:], Z[idx2X,:])

    testStat = np.mean(hh)
    std_est = np.std(hh_null)/math.sqrt(m2)
    p = stats.t.sf(testStat/std_est,m2-1)

    return p, testStat/std_est
