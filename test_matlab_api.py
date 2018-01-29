from __future__ import division, print_function
from scipy import stats
import scipy as sp
from statsmodels.distributions.empirical_distribution import ECDF
#from kernel_two_sample_test.kernel_two_sample_test import *

#from mmd_test_with_Shogun import *
from twosample_tests import *


import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon

import pickle


# two customized modules
from labelshift import *
from utils4gluon import *
from kun_tars import *

# generate data
n=500
m=500
d=2
p=0.1
q=0.7
y = np.random.rand(n,1)<p
ytest = np.random.rand(m,1)<q


X = np.random.randn(n,d)
X_test = np.random.randn(n,d)

#broad cast
X += np.sqrt(d) * y
X_test += np.sqrt(d) * ytest



beta1 = py_betaKMM_targetshift(X, y, X_test, sigma=[], lambda_beta=0.1)

beta2 = py_betaEM_targetshift(X, y, X_test)




w0  = [(1-q)/(1-p),q/p]
w1 = beta_to_w(beta1,y,2)
w2 = beta_to_w(beta2,y,2)
weightfunc = w_to_weightfunc(w1)

# checking the correctness
print(beta1 - weightfunc([],y)) # this should be all zeros
print('Ground truth = '+repr(w0) + ', KMM =' +repr(w1) +', EM =' + repr(w2))