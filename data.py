
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon


def load_data(domain):
    if domain == "mnist":
        return load_mnist()
    elif domain == "cifar10":
        return load_cifar10()
    else:
        print("ERROR: INVALID DATASET SPECIFIED")

    
def load_mnist():
    mnist = mx.test_utils.get_mnist()
    num_inputs = 784
    num_outputs = 10
    dfeat = 784
    nclass = 10
    dataset = mnist
    X = dataset["train_data"]
    y = dataset["train_label"]
    # lastly get the test set and its corresponding iterator
    Xtest = dataset["test_data"]
    ytest = dataset["test_label"]
    return (X, y, Xtest, ytest)

    
def load_cifar10():
    num_inputs = 3072
    num_outputs = 10
    dfeat = 3072
    nclass = 10
    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
    train_DS = gluon.data.vision.CIFAR10(train=True, transform=transform)
    test_DS = gluon.data.vision.CIFAR10(train=False, transform=transform)
    
    def transform_data(data):
        return nd.transpose(data.astype(np.float32), axes=(0,3,1,2))/255

    def transform_label(label):
        return nd.transpose(label.astype(np.float32))

    X = transform_data(train_DS._data).asnumpy()
    y = transform_label(nd.array(train_DS._label)).asnumpy()

    Xtest = transform_data(test_DS._data).asnumpy()
    ytest = transform_label(nd.array(test_DS._label)).asnumpy()
    
    return (X, y, Xtest, ytest)
