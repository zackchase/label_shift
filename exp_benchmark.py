# this script runs the experiments where we compare us v.s. Kun's approach and the Gaussian-EM approach.




from __future__ import division, print_function
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

import pickle

# three customized modules
from labelshift import *
from utils4gluon import *
from data_shift import *
from data import *
from ls_correction import *
from kun_tars import *
import pickle
np.random.seed(112358)


ctx=mx.cpu()

dataset_name = 'cifar10'  # choices: 'mnist', 'cifar10'
num_labels = 10
results_dict = {}

alpha_range = [1, .1, .01, .001]
num_runs = 2

# Tweak train data
tweak_train = True  # options include
p_P = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]

# Tweak test data
tweak_test = True

allresults = {}

unweighted = lambda X, y, Xtest: np.ones(shape=(X.shape[0], 1))
ours = lambda X, y, Xtest: BBSE(X, y, Xtest, ctx, num_hidden=num_hidden, epochs=epochs)
KMM_ts = lambda X, y, Xtest: py_betaKMM_targetshift(X, y, Xtest, sigma='median', lambda_beta=0.1)

methods = [unweighted, ours, KMM_ts]
methods_name = ["unweighted", "BBSE", "KMM-TarS"]

# maybe also adding


counter = 0
for alpha in alpha_range:
    allresults[alpha]=[]
    for run in range(num_runs):
        counter += 1
        print("Experiment: ", counter, "alpha =", alpha, "run =", run)
        p_Q = np.random.dirichlet([alpha] * 10)

        # sample data
        num_train_samples = 3000
        # num_val_samples = 3000
        num_test_samples = 3000

        # NN config
        num_hidden = 64
        epochs = 5
        batch_size = 64

        #########################################
        #  Invoke experiment code
        #########################################

        results = correction_experiment_benchmark(methods, dataset_name='mnist',
                                                  tweak_train=tweak_train,
                                                  p_P=p_P, tweak_test=tweak_test, p_Q=p_Q,
                                                  num_train_samples=num_train_samples,
                                                  num_test_samples=num_test_samples,
                                                  num_hidden=num_hidden,
                                                  epochs=epochs,
                                                  batch_size=batch_size)

        allresults[alpha].append([results,p_P,p_Q])


ToPickle = [alpha_range, num_runs, methods_name, allresults]

pickle.dump( ToPickle, open( "results_exp_benchmarking.p", "wb" ) )