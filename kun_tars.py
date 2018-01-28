# Python API to call the matlab implementation the TargetShift and LS_TargetShift in Kun et. al. ICML'2013
# author: yu-xiang wang
# Reference:
# - Zhang, Scholkopf, Muandet, Wang "Domain Adaptation under Target and Conditional Shift" ICML'13
#  URL: http://proceedings.mlr.press/v28/zhang13d.pdf
# Source code for the matlab version: http://people.tuebingen.mpg.de/kzhang/Code-TarS.zip

# We used the more modern way of choosing kernel bandwith via the median trick.

from __future__ import division, print_function
import os
import math

import numpy as np
from scipy import linalg, stats


import matlab
import matlab.engine

eng = matlab.engine.start_matlab()

eng.add_path(nargout=0)  # call the matlab script that adds path


def py_betaKMM_targetshift(X, Y, Xtst, sigma='median', lambda_beta=0.1):
    # sigma             ------    kernel bandwidth
    # width_L_beta      ------    kernel bandwidth for Y  (only needed for continuous Y)
    # lambda_beta       ------    regularization parameter for Y (only needed for continuous Y)
    # Other inputs are optional

    median_samples = 1000
    if sigma == 'median':
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        from sklearn.metrics.pairwise import euclidean_distances
        Z = np.r_[sub(X, median_samples // 2), sub(Xtst, median_samples // 2)]
        D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = np.median(upper, overwrite_input=True)/math.sqrt(2)
        sigma = np.sqrt(kernel_width / 2)
        # sigma = median / sqrt(2); works better, sometimes at least
        del Z, D2, upper

    sigma = float(sigma)
    mX = matlab.double(X.tolist())
    mY = matlab.double(Y.tolist())
    mXtst = matlab.double(Xtst.tolist())
    if sigma is None:
        mbeta = eng.betaKMM_targetshift(mX, mY, mXtst, [], 0, lambda_beta)
        return np.array(mbeta._data).reshape(mbeta.size, order='F')
    else:
        width_L_beta = 3 * sigma
        mbeta = eng.betaKMM_targetshift(mX, mY, mXtst, sigma, width_L_beta, lambda_beta)
        return np.array(mbeta._data).reshape(mbeta.size, order='F')


def py_betaEM_targetshift(X, Y, Xtst):
    mX = matlab.double(X.tolist())
    mY = matlab.double(Y.tolist())
    mXtst = matlab.double(Xtst.tolist())
    mbeta = eng.betaEM_targetshift(mX, mY, mXtst, [])
    return np.array(mbeta._data).reshape(mbeta.size, order='F')
