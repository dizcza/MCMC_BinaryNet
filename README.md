# MCMC Binary Net optimization

This repo shows an alternative optimization of binary neural nets that uses forward pass only. No backward passes. No gradients. Instead, we can use Gibbs sampling to randomly select `flip_ratio` weights (connections) in any binary network and flip their signs (multiply by `-1`). Then, we can accept or reject a new candidate (new model weights) at MCMC step. Convergence is determined by the temperature as a function of `flip_ratio` that slowly decreases with time.

## Requirements

* Python 3.5+
* [requirements.txt](requirements.txt)


## Quick start

Before running any experiment, make sure you've started the visdom server:

`python3 -m visdom.server`

```
>>> import torch.nn as nn
>>> from layers import binarize_model
>>> from trainer import TrainerMCMC
>>> class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_sequential = nn.Sequential(
                nn.BatchNorm2d(num_features=1),
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, bias=False),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(inplace=True),
            )
            self.linear = nn.Linear(in_features=1690, out_features=10, bias=False)
    
        def forward(self, x):
            x = self.conv_sequential(x)
            x = x.view(x.shape[0], -1)
            x = self.linear(x)
            return x

>>> model = Net()
>>> model_binary = binarize_model(model)
>>> print(model_binary)
Net(
  (conv_sequential): Sequential(
    (0): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)
    (1): [Binary]Conv2d (1, 10, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (3): ReLU(inplace)
  )
  (linear): Sequential(
    (0): BatchNorm1d(1690, eps=1e-05, momentum=0.1, affine=True)
    (1): [Binary]Linear(in_features=1690, out_features=10)
  )
)

>>> trainer = TrainerMCMC(model_binary,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name="MNIST",
                          temperature=1e-3,  # environment temperature (decreases with time)
                          flip_ratio=3e-3)  # flip how many signs of binary weights at MCMC step
>>> trainer.train(n_epoch=100, debug=False)
Training progress http://localhost:8097
```

## Results

Navigate to [http://ec2-34-227-113-244.compute-1.amazonaws.com:8099](http://ec2-34-227-113-244.compute-1.amazonaws.com:8099) and choose the Environment you want (`main` env is empty).

For local results, go to [http://localhost:8097](http://localhost:8097)