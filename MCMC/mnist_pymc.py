import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import torch
import torch.utils.data
from tqdm import tqdm

from utils import get_data_loader

fpath_trace = os.path.join(os.path.dirname(__file__), "mnist_trace.pkl")


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


def convergence_plot(trace=None, train=False):
    if trace is None:
        assert os.path.exists(fpath_trace), f"{fpath_trace} does not exist. Train the model first."
        with open(fpath_trace, 'rb') as f:
            trace = pickle.load(f)
    x_data, y_data = prepare_data(train=train, onehot=False)
    w_traces = trace.get_values('w', combine=False)
    if trace.nchains == 1:
        w_traces = [w_traces]
    for chain_id, w_chain in enumerate(tqdm(w_traces, desc='Running convergence diagnostics')):
        chain_accuracy = []
        for w_iter in w_chain:
            y_pred = predict(x_data=x_data, w=w_iter)
            accuracy = (y_data == y_pred).mean()
            chain_accuracy.append(accuracy)
        plt.plot(np.arange(1, len(chain_accuracy)+1), chain_accuracy, label=f'chain {chain_id}')
    plt.xlabel('Epoch (draw)')
    plt.ylabel('Accuracy')
    plt.title(f"Convergence plot, MNIST {'train' if train else 'test'}")
    plt.legend()
    plt.show()


def main(n_chains=3):
    np.random.seed(113)
    x_train, y_train_onehot = prepare_data(train=True, onehot=True, take_first=500)
    print(f"Using {len(x_train)} train samples.")
    model = pm.Model()
    with model:
        w = pm.Bernoulli('w', p=0.5, shape=(784, 10))
        logit_vec = tt.dot(x_train, w)
        proba = softmax(logit_vec)
        y_obs = pm.Multinomial('y_obs', n=1, p=proba, observed=y_train_onehot)
        trace = None
        if os.path.exists(fpath_trace):
            with open(fpath_trace, 'rb') as f:
                trace = pickle.load(f)
            if trace.nchains != n_chains:
                print(f"Reset previous progress {trace} to match n_chains={n_chains}")
                trace = None
        trace = pm.sample(draws=3, njobs=1, chains=n_chains, tune=0, trace=trace)
    if trace.nchains == n_chains:
        # we didn't stop the training process
        with open(fpath_trace, 'wb') as f:
            pickle.dump(trace, f)
    convergence_plot(trace=trace, train=True)


if __name__ == '__main__':
    main()
    convergence_plot(train=False)
