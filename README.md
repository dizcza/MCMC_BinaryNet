# MCMC Binary Net optimization

This repository demonstrates an alternative optimization of binary neural nets with forward pass in mind only. No backward passes. No gradients. Instead, we use Metropolis-Hasting sampler to randomly select 1 % of weights (connections) in a binary network and flip them (multiply by `-1`). Then, we can accept or reject a new candidate (new model weights) at MCMC step, based on the loss and the surrounding `temperature` (which defines how many weights to flip). Convergence is obtained by freezing the model (temperature goes to zero). Loss plays a role of model state energy, and you're free to choose any conventional loss you might like: Cross-Entropy loss, Contrastive loss, Triplet loss, etc.


## Quick start

### Setup

* `pip3 install -r requirements.txt`
* start visdom server with `python3 -m visdom.server -port 8097`

```python
import torch.nn as nn
from torchvision.datasets import MNIST
from mighty.utils.data import DataLoader, TransformDefault
from mighty.models import MLP

from trainer import TrainerMCMCGibbs


model = MLP(784, 10)
# MLP(
#   (classifier): Sequential(
#     (0): [Binary][Compiled]Linear(in_features=784, out_features=10, bias=False)
#   )
# )

data_loader = DataLoader(MNIST, transform=TransformDefault.mnist())
trainer = TrainerMCMCGibbs(model,
                           criterion=nn.CrossEntropyLoss(),
                           data_loader=data_loader)
trainer.train(n_epochs=100, mutual_info_layers=0)

# Training progress http://localhost:8097
```

For more examples, refer to [main.py](main.py).

## Results

A snapshot of training binary MLP 784 -> 10 (binary weights and binary activations) with `TrainerMCMCGibbs` on MNIST:

![](images/mnist_TrainerMCMC.png)

More results:

* Navigate to http://visdom.kyivaigroup.com:8097/. Give your browser a few minutes to parse the json data. Choose environments with `TrainerMCMC`. 
* For your local results, go to [http://localhost:8097](http://localhost:8097)
* JAGS simulation in _R_: [paper](MCMC/paper.pdf), [source](MCMC/mnist56_jags.R)
* PyMC3 simulation in Python: [readme](MCMC/README.md)
