import os

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'

import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets import MNIST
from torchvision import transforms
from mighty.utils.data import DataLoader
from mighty.monitor.monitor import MonitorLevel

from trainer import *
from utils.layers import ScaleLayer


class NetBinary(nn.Module):
    def __init__(self, fc_sizes, batch_norm=True, scale_layer=False):
        super().__init__()
        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.Linear(in_features, out_features, bias=False))
            if batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*fc_layers)
        if scale_layer:
            self.scale_layer = ScaleLayer(size=fc_sizes[-1])
        else:
            self.scale_layer = None

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if self.scale_layer is not None:
            x = self.scale_layer(x)
        return x


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    return optimizer, scheduler


def train_gradient_full_precision(model: nn.Module, dataset_cls=MNIST):
    optimizer, scheduler = get_optimizer_scheduler(model)
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerGrad(model,
                          criterion=nn.CrossEntropyLoss(),
                          data_loader=data_loader,
                          optimizer=optimizer,
                          scheduler=scheduler)
    # trainer.restore()  # uncomment to restore the saved state
    trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epoch=10, mutual_info_layers=2)
    return model


def train_gradient_binary(model: nn.Module, dataset_cls=MNIST):
    optimizer, scheduler = get_optimizer_scheduler(model)
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerGradBinary(model,
                                criterion=nn.CrossEntropyLoss(),
                                data_loader=data_loader,
                                optimizer=optimizer,
                                scheduler=scheduler)
    trainer.train(n_epoch=100, mutual_info_layers=0)
    return model


def train_binsoft(model: nn.Module, dataset_cls=MNIST):
    optimizer, scheduler = get_optimizer_scheduler(model)
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerGradBinarySoft(model,
                                    criterion=nn.CrossEntropyLoss(),
                                    data_loader=data_loader,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    hardness_scheduler=HardnessScheduler(model=model, step_size=5))
    trainer.train(n_epoch=100)
    return model


def train_mcmc(model: nn.Module, dataset_cls=MNIST):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerMCMCGibbs(model,
                               criterion=nn.CrossEntropyLoss(),
                               data_loader=data_loader)
    # trainer.restore("2019.01.02 NetBinary: MNIST TrainerGradBinary CrossEntropyLoss.pt", restore_env=False)
    trainer.train(n_epoch=100, mutual_info_layers=0)
    return model


def train_tempering(model: nn.Module, dataset_cls=MNIST):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = ParallelTempering(model,
                                criterion=nn.CrossEntropyLoss(),
                                data_loader=data_loader,
                                trainer_cls=TrainerMCMCGibbs,
                                n_chains=5)
    trainer.train(n_epoch=100)
    return model


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    set_seed(seed=113)
    # train_gradient_binary(NetBinary(fc_sizes=(784, 10), batch_norm=False))
    train_mcmc(NetBinary(fc_sizes=(784, 10), batch_norm=False, scale_layer=False))
    # train_mcmc(NetBinary(fc_sizes=(25, 2), batch_norm=False, scale_layer=False))
    # train_binsoft(NetBinary((784, 10), batch_norm=False))
    # train_tempering(NetBinary(fc_sizes=(784, 10), batch_norm=False, scale_layer=False))
