import mxnet as mx
from mxnet import nd, autograd


#------------------- utility functions -----------------

##################################################
#
##################################################
# def evaluate_accuracy(data_iterator, net, ctx, dfeat):
#     acc = mx.metric.Accuracy()
#     for i, batch in enumerate(data_iterator):
#         data = batch.data[0].as_in_context(ctx).reshape((-1, dfeat))
#         label = batch.label[0].as_in_context(ctx)
#         output = net(data)
#         predictions = nd.argmax(output, axis=1)
#         acc.update(preds=predictions, labels=label)
#     return acc.get()[1]


def evaluate_accuracy(data_iterator, net, ctx, dfeat):
    data_iterator.reset()
    numerator = 0.
    denominator = 0.
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx).reshape((-1, dfeat))
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

def weighted_train(net, lossfunc, trainer, Xtrain, ytrain, Xval, yval, ctx, dfeat, epoch=1, batch_size=64,weightfunc=None, data_ctx=mx.cpu()):
    '''

    :param net:             Forward pass model with NDarray parameters attached.
    :param lossfunc:        Loss function used
    :param trainer:         A declared optimizer.
    :param Xtrain:          Training data features
    :param ytrain:          Training data labels
    :param Xval:            Validation data features
    :param yval:            Validation data labels
    :param epoch:           The number of data passes to run in training
    :param weightfunc:      An optional lambda function that takes in a vector of X and y
                            and outputs a vector of weights to assign to those examples.
    :return:

    The function does not return anything. Trained parameters will remain in net.

    '''
    # declare data iterators
    train_data = mx.io.NDArrayIter(nd.array(Xtrain, ctx=data_ctx), nd.array(ytrain, ctx=data_ctx), batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(nd.array(Xval, ctx=data_ctx), nd.array(yval, ctx=data_ctx), batch_size, shuffle=False)
    
    smoothing_constant = .01
    moving_loss = 0

    #params = net.collect_params()
    #for param in params.values():
    #    ctx = param.list_ctx()[0]
    #    break # assuming all parameters are on the same context

    for e in range(epoch):
        train_data.reset()
        for i, batch in enumerate(train_data):
            data = batch.data[0].as_in_context(ctx).reshape((-1, dfeat))
            label = batch.label[0].as_in_context(ctx)

            if weightfunc is None:
                wt_batch = nd.ones_like(label) # output an ndarray of importance weight
            else:
                wt_batch = weightfunc(data,label).reshape((-1,))

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
        val_accuracy = evaluate_accuracy(val_data, net, ctx, dfeat)
        train_accuracy = evaluate_accuracy(train_data, net, ctx, dfeat)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
              (e, moving_loss, train_accuracy, val_accuracy))

def predict_all(X, net, ctx, dfeat, batch_size=64):
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
    ypred_soft_all = ypred_soft_all[:X.shape[0], ]

    return ypred_all, ypred_soft_all
