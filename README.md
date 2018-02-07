# MCMC Binary Net optimization

This repo shows an alternative optimization of binary neural nets that uses forward pass only. No backward passes. No gradients. Instead, we can use Gibbs sampling to randomly select `flip_ratio` weights (connections) in any binary network and flip their signs (multiply by `-1`). Then, we can accept or reject a new candidate (new model weights) at MCMC step. Convergence is determined by `temperature` that slowly decreases with time.

## Requirements

* Python 3.5+
* [requirements.txt](requirements.txt)


## Quick start

```
>>> from torchvision.models.alexnet import AlexNet
>>> from layers import binarize_model
>>> from trainer import TrainerMCMC
>>> model = AlexNet()
>>> model_binary = binarize_model(model)
>>> print(model_binary)
AlexNet(
  (features): Sequential(
    (0): [Binary]Conv2d (3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
    (3): [Binary]Conv2d (64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
    (6): [Binary]Conv2d (192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): [Binary]Conv2d (384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): [Binary]Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
  )
  (classifier): Sequential(
    (1): Sequential(
      (0): BatchNorm1d(9216, eps=1e-05, momentum=0.1, affine=True)
      (1): [Binary]Linear(in_features=9216, out_features=4096)
    )
    (2): ReLU(inplace)
    (4): Sequential(
      (0): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)
      (1): [Binary]Linear(in_features=4096, out_features=4096)
    )
    (5): ReLU(inplace)
    (6): Sequential(
      (0): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)
      (1): [Binary]Linear(in_features=4096, out_features=1000)
    )
  )
)

>>> trainer = TrainerMCMC(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name="MNIST",
                          temperature=1e-3,  # environment temperature (decreases with time)
                          flip_ratio=3*1e-3)  # flip how many signs of binary weights at MCMC step
>>> trainer.train(n_epoch=100, debug=0)
Training progress http://localhost:8097
```

## Results

Navigate to [http://ec2-34-227-113-244.compute-1.amazonaws.com:8099](http://ec2-34-227-113-244.compute-1.amazonaws.com:8099) and choose the Environment you want (`main` env is empty).