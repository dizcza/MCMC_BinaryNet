from pathlib import Path
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


def get_data_loader(dataset: str, train=True, batch_size=256) -> torch.utils.data.DataLoader:
    if dataset == "MNIST":
        dataset_cls = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == "CIFAR10":
        dataset_cls = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise NotImplementedError()
    dataset = dataset_cls('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def load_model(dataset: str, model_name: str) -> Union[nn.Module, None]:
    model_path = MODELS_DIR.joinpath(dataset, Path(model_name).with_suffix('.pt'))
    if not model_path.exists():
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
