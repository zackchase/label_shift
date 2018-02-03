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

dataset_name = 'mnist'  # choices: 'mnist', 'cifar10'
num_labels = 10
results_dict = {}


TWEAK_ONE = False # use knockout scheme, rather than dirichlet
MODIFY_P = False # if set to false, we use uniform P and modify Q.
EXTEND = True

if TWEAK_ONE:
    alpha_range = [0.9, 0.7, 0.5, 0.3, 0.1]
else:
    alpha_range = [.1, 1.0, 10.0] # small shift to large shift

num_runs = 20# repeat 5 times
num_runs_slow = 5 # number of runs for those slow ones..
nlist = [500, 1000, 2000, 4000, 8000] # a list of n
if EXTEND:
    nlist =[16000,32000]
cnn_flag=False # Use two-layer perceptron if this is False. otherwise use CNN

allresults = {}

unweighted = lambda X, y, Xtest: np.ones(shape=(X.shape[0], 1))
ours = lambda X, y, Xtest: BBSE(X, y, Xtest, ctx, num_hidden=num_hidden, epochs=epochs, cnn_flag=cnn_flag)
ours1 = lambda X, y, Xtest: BBSE(X, y, Xtest, ctx, num_hidden=num_hidden, epochs=epochs, useProb=True, cnn_flag=cnn_flag)
KMM_ts = lambda X, y, Xtest: py_betaKMM_targetshift(X, y, Xtest, sigma='median', lambda_beta=0.1)
#EM_ts = lambda X, y, Xtest: py_betaEM_targetshift(X, y, Xtest)
# the EM approach is not implemented in a way that handles more than 2 classes.

methods_dict = {"unweighted": unweighted, "KMM-Tars": KMM_ts, "BBSE": ours, "BBSE-prob": ours1}
methods_slowflag = {"unweighted": False, "KMM-Tars": True, "BBSE": False, "BBSE-prob": False}
methods_name = ["unweighted", "BBSE","BBSE-prob","KMM-Tars"]
if EXTEND:
    methods_name = ["unweighted", "BBSE", "BBSE-prob"]
methods = []
methods_slow = []
methods_name_fast = []
for item in methods_name:
    methods.append(methods_dict[item])
    if methods_slowflag[item] is False:
        methods_slow.append(methods_dict[item])
        methods_name_fast.append(item)



# maybe also adding

counter = 0
for n in nlist:
    for alpha in alpha_range:
        allresults[(alpha,n)]=[]
        for run in range(num_runs):
            counter += 1
            print("Experiment: ", counter, "n =", n, "alpha =", alpha, "run =", run)

            # Tweak train data
            tweak_train = True  # options include
            # Tweak test data
            tweak_test = True
            if MODIFY_P:
                p_Q = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
                if TWEAK_ONE:
                    p = alpha#(1-alpha)/num_labels
                    knockout_label = 5
                    p_P = np.full(num_labels, (1. - p) / (num_labels - 1))
                    p_P[knockout_label] = p
                else:
                    p_P = np.random.dirichlet([alpha] * 10)
            else:
                p_P = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
                if TWEAK_ONE:
                    p = alpha#(1 - alpha) / num_labels
                    knockout_label = 5
                    p_Q = np.full(num_labels, (1. - p) / (num_labels - 1))
                    p_Q[knockout_label] = p
                else:
                    p_Q = np.random.dirichlet([alpha] * 10)

            print(np.array(p_Q)/p_P)
            # sample data
            num_train_samples = n
            # num_val_samples = 3000
            num_test_samples = n

            # NN config
            num_hidden = 256
            epochs = 20
            batch_size = 64

            #########################################
            #  Invoke experiment code
            #########################################
            if run < num_runs_slow:
                methods_list = methods
            else:
                methods_list = methods_slow

            results = correction_experiment_benchmark(methods_list, dataset_name=dataset_name,
                                                      tweak_train=tweak_train,
                                                      p_P=p_P, tweak_test=tweak_test, p_Q=p_Q,
                                                      num_train_samples=num_train_samples,
                                                      num_test_samples=num_test_samples,
                                                      num_hidden=num_hidden,
                                                      epochs=epochs,
                                                      batch_size=batch_size,
                                                      cnn_flag=cnn_flag)

            allresults[(alpha,n)].append([results,p_P,p_Q])

            ToPickle = [alpha_range, nlist, num_runs, methods_name, allresults, methods_name_fast, num_runs_slow, num_runs]
            if EXTEND:
                pickle.dump( ToPickle, open( "results_exp_benchmarking_dirichlet_extend.p", "wb" ) )