import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

mnist56_url = {
    "train": "https://www.dropbox.com/s/l7uppxi1wvfj45z/MNIST56_train.csv?dl=1",
    "test": "https://www.dropbox.com/s/399gkdk9bhqvz86/MNIST56_test.csv?dl=1"
}


def prepare_data(fold_name="train"):
    """
    Processes MNIST56 dataset.
    :param fold_name: either 'train' or 'test'
    :return: binary pixels matrix of size (n, 25) and corresponding labels (vector of size n)
    """
    dataframe = pd.read_csv(mnist56_url[fold_name])
    x = dataframe.iloc[:, :25]  # MNIST 5x5 flatten
    x = (x > 0).astype(np.float32)  # binarize pixels
    y = dataframe.iloc[:, 25].astype(np.int8)  # labels 0 or 1
    return x, y


def predict(x_data, w):
    """
    :param x_data: matrix of size (n, 25)
    :param w: matrix of size (25, 2)
    :return: vector of size n of predicted labels
    """
    logit_vec = np.dot(x_data, w)
    y_pred = logit_vec.argmax(axis=1)
    return y_pred


def main():
    np.random.seed(113)
    x_train, y_train = prepare_data(fold_name="train")
    model = pm.Model()

    with model:
        w = pm.Bernoulli('w', p=0.5, shape=(25, 2))
        logit_vec = tt.dot(x_train, w)
        logit_p = logit_vec[:, 1] - logit_vec[:, 0]  # logit of p(y=1)
        y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y_train)
        trace = pm.sample(draws=101, njobs=1, chains=3, tune=10)
    print(pm.summary(trace))
    w_mean = trace.get_values('w').mean(axis=0)
    w_binary = (w_mean > 0.5).astype(int)
    x_test, y_test = prepare_data(fold_name="test")
    y_pred = predict(x_data=x_test, w=w_binary)
    accuracy = (y_pred == y_test).mean()
    print("\nTest accuracy: {:.3f}".format(accuracy))


if __name__ == '__main__':
    main()
