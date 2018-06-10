import os
import pickle

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import torch
import torch.utils.data

from utils import get_data_loader


def flatten_dataset(data_loader: torch.utils.data.DataLoader, cast_numpy=True, take_first=None):
    if take_first is None:
        take_first = float('inf')
    images_all, labels_all = [], []
    for images, labels in data_loader:
        images_all.append(images)
        labels_all.append(labels)
        if len(labels_all) * data_loader.batch_size > take_first:
            break
    images_all = torch.cat(images_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    images_all = images_all.view(len(images_all), -1)
    if cast_numpy:
        images_all = images_all.numpy()
        labels_all = labels_all.numpy()
    return images_all, labels_all


def prepare_data(train=True, onehot=False, take_first=None):
    loader = get_data_loader(dataset="MNIST", train=train)
    x_data, y_data = flatten_dataset(data_loader=loader, cast_numpy=True, take_first=take_first)
    x_data = (x_data > 0).astype(np.float32)
    if onehot:
        n_samples = len(y_data)
        labels_onehot = np.zeros(shape=(n_samples, 10), dtype=np.int8)
        labels_onehot[range(n_samples), y_data] = 1
        y_data = labels_onehot
    return x_data, y_data


def predict(x_data, w):
    """
    :param x_data: matrix of size (n, 784)
    :param w: matrix of size (784, 10)
    :return: vector of size n of predicted labels
    """
    logit_vec = np.dot(x_data, w)
    y_pred = logit_vec.argmax(axis=1)
    return y_pred


def softmax(vec):
    """
    :param vec: Theano tensor of logit activations
    :return: softmax probability vector
    """
    vec_stable = vec - vec.max(axis=1, keepdims=True)
    vec_exp = tt.exp(vec_stable)
    return vec_exp / vec_exp.sum(axis=1, keepdims=True)


def main():
    np.random.seed(113)
    x_train, y_train = prepare_data(train=True, onehot=True, take_first=500)
    print(f"Using {len(x_train)} train samples.")
    fpath_trace = os.path.join(os.path.dirname(__file__), "mnist_trace.pkl")
    model = pm.Model()
    with model:
        w = pm.Bernoulli('w', p=0.5, shape=(784, 10))
        logit_vec = tt.dot(x_train, w)
        proba = softmax(logit_vec)
        y_obs = pm.Multinomial('y_obs', n=1, p=proba, observed=y_train)
        trace = None
        if os.path.exists(fpath_trace):
            with open(fpath_trace, 'rb') as f:
                trace = pickle.load(f)
        trace = pm.sample(draws=3, njobs=1, chains=1, n_init=100, tune=0, trace=trace)
    with open(fpath_trace, 'wb') as f:
        pickle.dump(trace, f)
    w_mean = trace.get_values('w').mean(axis=0)
    w_binary = (w_mean > 0.5).astype(int)
    x_test, y_test = prepare_data(train=False, onehot=False)
    y_pred = predict(x_data=x_test, w=w_binary)
    accuracy = (y_pred == y_test).sum() / len(y_test)
    print("Test accuracy: {:.3f}".format(accuracy))


if __name__ == '__main__':
    main()
