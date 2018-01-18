#  This script runs experiments and generates plots for the hypothesis testing problem


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




deltalist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
powerlist = []
powerlist_oracle = []
cdflist = []
cdflist_oracle = []

powerlist_x = []
cdflist_x = []


for delta in deltalist:
    #-------------------- preparing data ----------------------------------------

    ctx = mx.cpu()


    mnist = mx.test_utils.get_mnist()
    num_inputs = 784
    num_outputs = 10

    dfeat = 784
    nclass = 10


    batch_size = 64

    dataset = mnist

    X = dataset["train_data"]
    y = dataset["train_label"]

    # make the training data slightly unbalanced by knocking out the class distribution a little bit
    # split it into train and validation

    # The test set will have a uniform distribution over y
    # the train will not.

    n = X.shape[0]

    # Random permutation of the data
    idx = np.random.permutation(n)
    X = X[idx,...]
    y = y[idx]

    # split the data into training and testing
    num = 3

    Xtest = X[(2*n // num):, :, :, :]
    ytest = y[(2*n // num):]

    X = X[:(n // num), :, :, :]
    y = y[:(n // num)]

    n = X.shape[0]

    # Now adding perturbation to the train distribution
    # the amount to perturb in one class is delta.
    #delta = 0.5
    ko_class = 5

    idx = (y==ko_class).nonzero()
    nn=len(idx[0])
    nnko=np.round(delta*len(idx[0])).astype(int)
    ko_idx = np.random.choice(nn,nnko)

    mask = np.ones(n)>0
    mask[idx[0][ko_idx]] = False

    X = X[mask, ...]
    y = y[mask]

    # further splitting the training data into train and val

    n = X.shape[0]

    num = 2

    Xtrain = X[:(n//num),:,:,:]
    ytrain = y[:(n//num)]
    Xval = X[(n//num):(2*n//num),:,:,:]
    yval = y[(n//num):(2*n//num):]

    # we will ignore the standard test data, which I believe has a different feature distribution.
    #Xtest = dataset["test_data"]
    #ytest = dataset["test_label"]

    sz = 10000

    #
    # ------------------ Training a classifier -----------------
    #
    num_hidden = 256
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))

    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})


    # get the p-values for ground-truth tests
    pvlist_gnd = []
    for i in range(500):
        yboot_s = np.random.choice(yval, size=sz, replace=True)
        yboot_t = np.random.choice(ytest, size=sz, replace=True)
        D, pv = stats.ks_2samp(yboot_s, yboot_t)
        pvlist_gnd.append(pv.astype(float))
        #D, tmp, pv = stats.anderson_ksamp([yboot_s, yboot_t])
        #pvlist_gnd.append(pv)

    cdfpval_gnd = ECDF(pvlist_gnd)

    # get the p-values for p(x), q(x) test using kernel two-sample test

    # reshape the data points first
    Xval = Xval.reshape((-1, dfeat))
    Xtest = Xtest.reshape((-1, dfeat))


    r = 50
    v = np.random.rand(dfeat, r)
    DIM_REDUCTION = False

    pvlist_x = []
    for i in range(100):
        idx1 = np.random.choice(range(len(Xval)), size=sz, replace=True)
        idx2 = np.random.choice(range(len(Xtest)), size=sz, replace=True)
        xboot_s = Xval[idx1,:]
        xboot_t = Xtest[idx2,:]

        if DIM_REDUCTION:
            # dimension reduction
            xboot_s = np.dot(xboot_s, v)
            xboot_t = np.dot(xboot_t, v)

        # how to choose kernel bandwidth?
        # use the median trick!
        # they are already randomized
        sq_dist = np.sum((xboot_s[:sz,]-xboot_t[:sz,])**2,1)
        h = np.sqrt(np.median(sq_dist)/2)

        #print("start kernel test"+repr(i))
        #pval = linear_time_rbf_mmd_test(xboot_s, xboot_t, bandwidth="median")
        #print("kernel test returns pval = " + repr(pval))
        #tmp1,tmp2, pval = kernel_two_sample_test(xboot_s, xboot_t, kernel_function='rbf', iterations=100,
        #                       verbose=False, random_state=None, gamma=1/h)

        # using btest-mmd
        pval, tmp = btest_mmd_python(xboot_s, xboot_t, bandwidth="median")
        # using t-test
        #pval, tmp = linear_hotelling_test(xboot_s, xboot_t, reg=0.1)
        print("testing p(x) returns stat = " + repr(tmp)+ "and pval = " + repr(pval))
        pvlist_x.append(pval)

    cdfpval_x = ECDF(pvlist_x)


    temp = []
    # get the p-values for better and better classifiers.
    for i in range(1,10,2):

        if i == 1:
            epochs=1
        else:
            epochs=2
        # Training
        weighted_train(net, softmax_cross_entropy, trainer, Xtrain, ytrain, Xval, yval, ctx, dfeat, epoch=epochs, weightfunc=None)


        # Prediction
        ypred_s, ypred_s_soft = predict_all(Xval, net, ctx, dfeat)
        ypred_t, ypred_t_soft = predict_all(Xtest, net, ctx, dfeat)


        # Converting to numpy array for later convenience
        ypred_s= ypred_s.asnumpy()
        ypred_s_soft = ypred_s_soft.asnumpy()
        ypred_t=ypred_t.asnumpy()

        # use bootstrap and plot p-values

        pvlist =[]
        for i in range(500):
            yboot_s = np.random.choice(ypred_s, size=sz, replace=True)
            yboot_t = np.random.choice(ypred_t, size=sz, replace=True)
            D, pv = stats.ks_2samp(yboot_s, yboot_t)
            pvlist.append(pv.astype(float))
            #D, tmp, pv = stats.anderson_ksamp([yboot_s, yboot_t])
            #pvlist.append(pv)

        cdfpval = ECDF(pvlist)

        temp.append(cdfpval(0.05))


    powerlist.append(temp)
    powerlist_oracle.append(cdfpval_gnd(0.05))

    powerlist_x.append(cdfpval_x(0.05))

    cdflist.append(cdfpval)
    cdflist_oracle.append(cdfpval_gnd)
    cdflist_x.append(cdfpval_x)


results = [deltalist, powerlist, powerlist_oracle, cdflist, cdflist_oracle, powerlist_x, cdflist_x]

pickle.dump( results, open( "results_exp_mnist_full.p", "wb" ) )

