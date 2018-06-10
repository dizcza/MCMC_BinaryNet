# MCMC simulation in R and Python

Here you'll find a _true MCMC_ simulation of drawing binary weights to fit binarized MNIST data, using one fully connected layer, followed by softmax. A _true MCMC_ simulation means using true (but slow) Bayesian optimization frameworks like PyMC3 and JAGS, not the approximation that you can find in the [main page](https://github.com/dizcza/MCMC_BinaryNet).

It's recommended to start with training on the truncated MNIST56 dataset.

## Experiments

| Dataset | Overview | Train/test images | Image size | Layer size | One MCMC draw duration | Accuracy |
| ------- | -------- | ----------------- | ---------- | --------------- | ---------------------- | -------- |
| MNIST   | Full [MNIST](http://yann.lecun.com/exdb/mnist) dataset | 60000/10000 | 28x28 | 784x10 | ~15 min | 0.802 |
| MNIST56 | A subset of MNIST dataset of digits 5 and 6 | 11339/1850 | resized to 5x5 | 25x2 | <1 sec | 0.913 |

## Prerequisites

* PyMC3 (Python)

`conda install -c conda-forge pymc3 theano pandas`

* JAGS (R)

```
sudo add-apt-repository ppa:marutter/rrutter
sudo apt-get update
sudo apt-get install r-cran-rjags
```

Inside _R_ environment run `install.packages("rjags")`.
