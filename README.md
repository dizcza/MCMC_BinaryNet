# MCMC Binary Net optimization

This repo shows an alternative optimization of binary neural nets that uses forward pass only. No backward passes. No gradients. Instead, we can use MCMC sampler to randomly select `flip_ratio` weights (connections) in any binary network and flip their signs (multiply by `-1`). Then, we can accept or reject a new candidate (new model weights) at MCMC step, based on the loss. Convergence is determined by the temperature as a function of `flip_ratio` that slowly decreases with time.

## Requirements

* Python 3.6+
* [requirements.txt](requirements.txt)


## Quick start

Before running any experiment, make sure you've started the visdom server:

`python3 -m visdom.server`

```python
import torch.nn as nn
from layers import binarize_model
from trainer import TrainerMCMC
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28**2, 10, bias=False)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

model = Perceptron()
model_binary = binarize_model(model)
print(model_binary)
# Perceptron(
#   (linear): [Binary]Linear(in_features=784, out_features=10, bias=False)
# )

trainer = TrainerMCMC(model_binary,
                      criterion=nn.CrossEntropyLoss(),
                      dataset_name="MNIST",
                      flip_ratio=0.1)  # flip how many signs of binary weights at MCMC step
trainer.train(n_epoch=100)
# Training progress http://localhost:8097
```

## Results

* Train plots. Navigate to [http://ec2-34-227-113-244.compute-1.amazonaws.com:8099](http://ec2-34-227-113-244.compute-1.amazonaws.com:8099) and choose the Environment you want (`main` env is empty).
* For your local results, go to [http://localhost:8097](http://localhost:8097)
* JAGS simulation in R: [paper](JAGS/paper.pdf), [code](JAGS/mcmc_jags.R)