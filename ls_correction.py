from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

# three customized modules
from labelshift import *
from utils4gluon import *
from data_shift import *
from data import *



def correction_experiment(dataset_name=None, 
                     tweak_train=None, 
                     p_P=None, tweak_test=None, p_Q=None, 
                     num_train_samples=None,
                     num_val_samples=None,
                     num_test_samples=None,
                     num_hidden=None, 
                     epochs=None,
                     batch_size=None):

    # set the context for compute
    ctx = mx.gpu()
    
    # set the context for data
    data_ctx = mx.gpu()

    # load the dataset
    X, y, Xtest, ytest = load_data(dataset_name)

    n = X.shape[0]
    dfeat = np.prod(X.shape[1:])

    # NOTE FOR IMPROVEMENT: eventually this should be returned by the data library
    num_labels = 10

    ################################################
    # Random permutation of the data
    ################################################

    rand_idx = np.random.permutation(n)
    X = X[rand_idx,...]
    y = y[rand_idx]

    ################################################
    #  First split examples between train and validation
    ################################################
    num = 2  
    Xtrain_source = X[:(n//num),:,:,:]
    ytrain_source = y[:(n//num)]
    Xval_source = X[(n//num):(2*n//num),:,:,:]
    yval_source = y[(n//num):(2*n//num):]

    ################################################
    #  Set the label distribution at train time
    ################################################
    if tweak_train:
#         print("Sampling training and validation data from p_P")
#         print("Current p_P: ", p_P)
        Xtrain, ytrain = tweak_dist(Xtrain_source, ytrain_source, num_labels, num_train_samples, p_P)
        Xval, yval = tweak_dist(Xval_source, yval_source, num_labels, num_val_samples, p_P)
    else:
        Xtrain, ytrain = Xtrain_source, ytrain_source
        Xval, yval = Xval_source, yval_source

    ################################################
    #  Set the label distribution for test data
    ################################################
    if tweak_test:
#         print("Sampling test data from p_Q")
#         print("Current p_Q: ", p_Q)
        Xtest, ytest = tweak_dist(Xtest, ytest, num_labels, num_test_samples, p_Q)
          
    ####################################
    # Train on p_P
    ####################################
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_labels))

    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
    net.hybridize()
    
    # Training
    weighted_train(net, softmax_cross_entropy, trainer, Xtrain, ytrain, Xval, yval, ctx, dfeat, epoch=epochs, weightfunc=None, data_ctx=data_ctx)


    # Prediction
    ypred_s, ypred_s_soft = predict_all(Xval, net, ctx, dfeat)
    ypred_t, ypred_t_soft = predict_all(Xtest, net, ctx, dfeat)


    # Converting to numpy array for later convenience
    ypred_s= ypred_s.asnumpy()
    ypred_s_soft = ypred_s_soft.asnumpy()
    ypred_t = ypred_t.asnumpy()
    ypred_t_soft = ypred_t_soft.asnumpy()
    
    ####################################
    # Estimate Wt and Py
    ####################################
    wt = estimate_labelshift_ratio(yval, ypred_s, ypred_t,num_labels)

    Py_est = estimate_target_dist(wt, yval,num_labels)

    Py_true = calculate_marginal(ytest,num_labels)
    Py_base = calculate_marginal(yval,num_labels)

    wt_true = Py_true/Py_base

    print(np.concatenate((wt,wt_true),axis=1))
    print(np.concatenate((Py_est,Py_true),axis=1))

#     print("||wt - wt_true||^2  = " + repr(np.sum((wt-wt_true)**2)/np.linalg.norm(wt_true)**2))
#     print("KL(Py_est|| Py_true) = " + repr(stats.entropy(Py_est,Py_base)))
    
    
    ####################################
    # Solve weighted ERM and compare to previously trained models
    ####################################
    
    data_test = mx.io.NDArrayIter(Xtest, ytest, batch_size, shuffle=False)

    acc_unweighted =  evaluate_accuracy(data_test, net, ctx, dfeat) # in fact, drawing confusion matrix maybe more informative

    print("Accuracy unweighted", acc_unweighted)

    training_weights=np.maximum(wt, 0)
    wt_ndarray = nd.array(training_weights,ctx=ctx)


    weightfunc = lambda x,y: wt_ndarray[y.asnumpy().astype(int)]

    # Train a model using the following!
    net2 = gluon.nn.HybridSequential()
    with net2.name_scope():
        net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net2.add(gluon.nn.Dense(num_labels))

    net2.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer2 = gluon.Trainer(net2.collect_params(), 'sgd', {'learning_rate': .1})
    net.hybridize()
    
    # NOTE WE ASSUME SAME NUMBER OF EPOCHS IN PERIOD 1 and PERIOD 2
    
    # Training
    weighted_train(net2, softmax_cross_entropy, trainer2, Xtrain, ytrain, 
                   Xval, yval, ctx, dfeat, epoch=epochs, weightfunc=weightfunc, data_ctx=data_ctx)

    data_test.reset()
    acc_weighted = evaluate_accuracy(data_test, net2, ctx, dfeat)

    print("Accuracy weighted", acc_weighted)
    
    return {"acc_unweighted": acc_unweighted, 
            "acc_weighted": acc_weighted,
            "wt": wt, 
            "wt_true": wt_true, 
            "wt_l2": np.sum((wt-wt_true)**2)/np.linalg.norm(wt_true)**2, 
            "kl_div": stats.entropy(Py_est,Py_base),
            "ypred_s": ypred_s,
            "ypred_s_soft": ypred_s_soft,
            "ypred_t:": ypred_t,
            "ypred_t_soft": ypred_t_soft,
            }

    
    