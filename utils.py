import os
from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets

from constants import MODELS_DIR


def parameters_binary(model: nn.Module):
    for name, param in named_parameters_binary(model):
        yield param


def named_parameters_binary(model: nn.Module):
    return filter(lambda named_param: getattr(named_param[1], "is_binary", False), model.named_parameters())


def get_data_loader2(train=True, batch_size=256) -> torch.utils.data.DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_data_loader(train=True, batch_size=256) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.CIFAR10('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def load_model(model_name: str) -> Union[nn.Module, None]:
    if not model_name.endswith('.pt'):
        model_name += '.pt'
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return None
    return torch.load(model_path)


class StepLRClamp(torch.optim.lr_scheduler.StepLR):

    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-4):
        self.min_lr = min_lr
        super().__init__(optimizer, step_size, gamma, last_epoch=-1)

    def get_lr(self):
        learning_rates = super().get_lr()
        learning_rates = [max(lr, self.min_lr) for lr in learning_rates]
        return learning_rates
