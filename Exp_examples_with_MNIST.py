#  This script provide demos on how to do: label-shift detection, importance-weight estimation
# and label-shift correction via importance weighted learning.

from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF


import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon

# two customized modules
from labelshift import *
from utils4gluon import *





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

# the amount to perturb in one calss
delta = 0.98
ko_class = 5

idx = (y==ko_class).nonzero()

nn=len(idx[0])
nnko=np.round(delta*len(idx[0])).astype(int)
ko_idx = np.random.choice(nn,nnko)

mask = np.ones(n)>0
mask[idx[0][ko_idx]] = False

X = X[mask, ...]
y = y[mask]

# do the standard data splitting

n = X.shape[0]

# Random permutation of the data
idx = np.random.permutation(n)
X = X[idx,...]
y = y[idx]

num = 2

Xtrain = X[:(n//num),:,:,:]
ytrain = y[:(n//num)]
Xval = X[(n//num):(2*n//num),:,:,:]
yval = y[(n//num):(2*n//num):]

# lastly get the test set and its corresponding iterator
Xtest = dataset["test_data"]
ytest = dataset["test_label"]


#
# ------------------ Training a classifier -----------------
#
num_hidden = 256
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

epochs = 5

# Training
weighted_train(net, softmax_cross_entropy, trainer, Xtrain, ytrain, Xval, yval, ctx, dfeat, epoch=epochs, weightfunc=None)


# Prediction
ypred_s, ypred_s_soft = predict_all(Xval, net, ctx, dfeat)
ypred_t, ypred_t_soft = predict_all(Xtest, net, ctx, dfeat)


# Converting to numpy array for later convenience
ypred_s= ypred_s.asnumpy()
ypred_s_soft = ypred_s_soft.asnumpy()
ypred_t=ypred_t.asnumpy()


#
# ------------------ Applications -----------------
#
#----------------------------------------------------------------------------
# Example code for estimating Wt and Py
#----------------------------------------------------------------------------


wt = estimate_labelshift_ratio(yval, ypred_s, ypred_t,num_outputs)

Py_est = estimate_target_dist(wt, yval,num_outputs)

Py_true =calculate_marginal(ytest,num_outputs)
Py_base =calculate_marginal(yval,num_outputs)

print(Py_true)

print(Py_base)

wt_true = Py_true/Py_base

print(np.concatenate((wt,wt_true),axis=1))
print(np.concatenate((Py_est,Py_true),axis=1))

print("||wt - wt_true||^2  = " + repr(np.sum((wt-wt_true)**2)/np.linalg.norm(wt_true)**2))

print("KL(Py_est|| Py_true) = " + repr(stats.entropy(Py_est,Py_base)))



#----------------------------------------------------------------------------
# Example code for Detect nonstationarity using KS-test
#----------------------------------------------------------------------------

D, pv = stats.ks_2samp(ypred_s, ypred_t)
D, pv2 = stats.ks_2samp(yval, ytest)

print("The p-value " + repr(pv) +", p-value for testing directly "+ repr(pv2))

# use bootstrap and plot p-values

pvlist =[]
for i in range(500):
    yboot_s = np.random.choice(ypred_s, size=len(ypred_s)//2, replace=True)
    yboot_t = np.random.choice(ypred_t, size=len(ypred_t)//2, replace=True)
    D, tmp, pv=stats.anderson_ksamp([yboot_s,yboot_t])
    #D, pv = stats.ks_2samp(yboot_s, yboot_t)
    pvlist.append(pv)

pvlist_gnd = []
for i in range(500):
    yboot_s = np.random.choice(yval, size=len(yval)//2, replace=True)
    yboot_t = np.random.choice(ytest, size=len(ytest)//2, replace=True)
    #D, pv = stats.ks_2samp(yboot_s, yboot_t)
    #pvlist_gnd.append(pv.astype(float))
    D, tmp, pv = stats.anderson_ksamp([yboot_s, yboot_t])
    pvlist_gnd.append(pv)



import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
xx=  np.linspace(0, 1, num=len(pvlist))

cdfpval = ECDF(pvlist)
cdfpval_gnd = ECDF(pvlist_gnd)
plt.plot(xx, xx)
plt.plot(xx, cdfpval(xx), linewidth=2)
plt.plot(xx, cdfpval_gnd(xx), linewidth=2)

plt.legend(['Uniform', 'p-value of AD-test with blackbox predictor', 'p-value of oracle AD-test using target labels'], loc='best')
plt.title('qq-plot on the power of hypothesis testing')
fig.savefig("AD-test"+"delta="+repr(delta)+".pdf", bbox_inches='tight')
#
# uniformdist = stats.uniform(loc=0,scale=1)
#
# ax1 = plt.subplot(211)
# res = stats.probplot(pvlist, dist=uniformdist, plot=ax1)
#
# ax2 = plt.subplot(212)
# res2 = stats.probplot(pvlist_gnd, dist=uniformdist, plot=ax2)





#----------------------------------------------------------------------------
# Example code for solving weighted ERM and compare to previously trained models
#----------------------------------------------------------------------------

data_test = mx.io.NDArrayIter(Xtest, ytest, batch_size, shuffle=False)

acc_unweighted =  evaluate_accuracy(data_test, net, ctx, dfeat) # in fact, drawing confusion matrix maybe more informative

print(acc_unweighted)

wt_ndarray = nd.array(wt,ctx=ctx)

weightfunc = lambda x,y: wt_ndarray[y.asnumpy().astype(int)]

# Train a model using the following!
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
weighted_train(net, softmax_cross_entropy, trainer, Xtrain, ytrain, Xval, yval, ctx, dfeat, epoch=epochs, weightfunc=weightfunc)

data_test.reset()
acc_weighted = evaluate_accuracy(data_test, net, ctx, dfeat)

print(acc_weighted)