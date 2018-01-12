#  This script runs experiments and generates plots for hypothesis testing

from scipy import stats

import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon
from labelshift import *


#------------------- utility functions -----------------

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx).reshape((-1, dfeat))
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def weighted_train(net, lossfunc, trainer, Xtrain, ytrain, Xval, yval, epoch=1, weightfunc=None):
    # declare data iterators
    train_data = mx.io.NDArrayIter(Xtrain, ytrain, batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(Xval, yval, batch_size, shuffle=False)

    smoothing_constant = .01
    moving_loss = 0


    for e in range(epochs):
        train_data.reset()
        for i, batch in enumerate(train_data):
            data = batch.data[0].as_in_context(ctx).reshape((-1, dfeat))
            label = batch.label[0].as_in_context(ctx)
            if weightfunc is None:
                wt_batch = nd.ones_like(label) # output an ndarray of importance weight
            else:
                wt_batch = weightfunc(data,label)

            with autograd.record():
                output = net(data)
                loss = lossfunc(output, label)
                loss = loss*wt_batch
                loss.backward()
            trainer.step(data.shape[0])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        val_data.reset()
        train_data.reset()
        val_accuracy = evaluate_accuracy(val_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
              (e, moving_loss, train_accuracy, val_accuracy))

def predict_all(X, net, dfeat, batchsize=64):
    '''
    :param X: an ndarray containing the data. The first axis is over examples
    :param net: trained model
    :param dfeat: the dimensionality of the vectorized feature
    :param batchsize: batchsize used in iterators. default is 64.
    :return: Two ndarrays containing the soft and hard predictions of the classifier.
    '''

    data_iterator = mx.io.NDArrayIter(X, None, batch_size, shuffle=False)
    ypred_soft=[]
    ypred=[]
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx).reshape((-1, dfeat))
        output = net(data)
        softpredictions = nd.softmax(output, axis=1)
        predictions = nd.argmax(output, axis=1)
        ypred_soft.append(softpredictions)
        ypred.append(predictions)

    ypred_soft_all = nd.concatenate(ypred_soft, axis=0)
    ypred_all = nd.concatenate(ypred, axis=0)

    # iterator automatically pads the last minibatch, so the length of the vectors might be different.
    ypred_all = ypred_all[:X.shape[0]]
    ypred_soft_all = ypred_soft_all [:X.shape[0], ]

    return ypred_all, ypred_soft_all




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
delta = 0.5
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
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

epochs = 5

# Training
weighted_train(net, softmax_cross_entropy, trainer, Xtrain, ytrain, Xval, yval, epoch=epochs, weightfunc=None)


# Prediction
ypred_s, ypred_s_soft = predict_all(Xval, net, dfeat)
ypred_t, ypred_t_soft = predict_all(Xtest, net, dfeat)


# Converting to numpy array for later convenience
ypred_s= ypred_s.asnumpy()
ypred_s_soft = ypred_s_soft.asnumpy()
ypred_t=ypred_t.asnumpy()


#
# ------------------ Applications -----------------
#



#----------------------------------------------------------------------------
# Example code for Detect nonstationarity using KS-test
#----------------------------------------------------------------------------

D, pv = stats.ks_2samp(ypred_s, ypred_t)
D, pv2 = stats.ks_2samp(yval, ytest)

print("The p-value " + repr(pv) +", p-value for testing directly "+ repr(pv2))

# use bootstrap and plot p-values

pvlist =[]
for i in range(200):
    yboot_s = np.random.choice(ypred_s, size=len(ypred_s), replace=True)
    yboot_t = np.random.choice(ypred_t, size=len(ypred_t), replace=True)
    D, pv = stats.ks_2samp(yboot_s, yboot_t)
    pvlist.append(pv.astype(float))

pvlist_gnd = []
for i in range(200):
    yboot_s = np.random.choice(yval, size=len(yval), replace=True)
    yboot_t = np.random.choice(ytest, size=len(ytest), replace=True)
    D, pv = stats.ks_2samp(yboot_s, yboot_t)
    pvlist_gnd.append(pv.astype(float))



import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
xx=  np.linspace(0, 1, num=len(pvlist))
plt.plot(xx, xx)
plt.plot(xx, sorted(pvlist),linewidth=2)
plt.plot(xx, sorted(pvlist_gnd),linewidth=2)

plt.legend(['Uniform', 'p-value of KS-test with blackbox predictor', 'p-value of oracle KS-test using target labels'], loc='best')
plt.title('qq-plot on the power of hypothesis testing')
fig.savefig("Hypothesis-Testing-Power.pdf", bbox_inches='tight')
#
# uniformdist = stats.uniform(loc=0,scale=1)
#
# ax1 = plt.subplot(211)
# res = stats.probplot(pvlist, dist=uniformdist, plot=ax1)
#
# ax2 = plt.subplot(212)
# res2 = stats.probplot(pvlist_gnd, dist=uniformdist, plot=ax2)


#----------------------------------------------------------------------------
# Example code for estimating Wt and Py
#----------------------------------------------------------------------------


wt = estimate_labelshift_ratio(yval, ypred_s, ypred_t,num_outputs)

Py_est = estimate_target_dist(wt, yval,num_outputs)

Py_true =calculate_marginal(ytest,num_outputs)
Py_base =calculate_marginal(yval,num_outputs)

wt_true = Py_true/Py_base

print(np.concatenate((wt,wt_true),axis=1))
print(np.concatenate((Py_est,Py_true),axis=1))

print("||wt - wt_true||^2  = " + repr(np.sum((wt-wt_true)**2)/np.linalg.norm(wt_true)**2))

print("KL(Py_est|| Py_true) = " + repr(stats.entropy(Py_est,Py_base)))



#----------------------------------------------------------------------------
# Example code for solving weighted ERM and compare to previously trained models
#----------------------------------------------------------------------------

data_test = mx.io.NDArrayIter(Xtest, ytest, batch_size, shuffle=False)

acc_unweighted =  evaluate_accuracy(data_test, net) # in fact, drawing confusion matrix maybe more informative

print(acc_unweighted)

wt_ndarray = nd.array(wt,ctx=ctx)

weightfunc = lambda x,y: wt_ndarray[y.asnumpy().astype(int)]

# Train a model using the following!
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
weighted_train(net, softmax_cross_entropy, trainer, Xtrain, ytrain, Xval, yval, epoch=epochs, weightfunc=weightfunc)

data_test.reset()
acc_weighted = evaluate_accuracy(data_test, net)

print(acc_weighted)