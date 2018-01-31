import mxnet as mx
from mxnet import nd, autograd
import numpy as np


##################################3
#  X, y - training data
#  n - number of data points in dataset
#  Py - desired label distribution
###################################
def tweak_dist(X, y, num_labels, n, Py):
    shape = (n, *X.shape[1:])
    Xshift = np.zeros(shape)
    yshift = np.zeros(n, dtype=np.int8)

    # get indices for each label
    indices_by_label = [(y==k).nonzero()[0] for k in range(10)]
    
    labels = np.argmax(
        np.random.multinomial(1, Py, n), axis=1)
        
    for i in range(n):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[labels[i]])
        Xshift[i] = X[idx]
        yshift[i] = y[idx]
    
    return Xshift, yshift


def tweak_one(X, y, num_labels, n, knockout_label, p):
    # create Py
    # call down to tweak_dist
    Py = np.full(num_labels, (1.-p)/(num_labels-1))
    Py[knockout_label] = p
    print(Py)
    return tweak_dist(X, y, num_labels, n, Py)
    


