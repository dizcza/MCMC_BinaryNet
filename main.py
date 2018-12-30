import os

import torch
import torch.nn as nn
import torch.utils.data

from trainer import *
from utils.layers import ScaleLayer

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'


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


def train_gradient_full_precision(model: nn.Module, dataset_name="MNIST"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    trainer = TrainerGrad(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name=dataset_name,
                          optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epoch=100)
    return model


def train_gradient_binary(model: nn.Module, dataset_name="MNIST"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    trainer = TrainerGradBinary(model,
                                criterion=nn.CrossEntropyLoss(),
                                dataset_name=dataset_name,
                                optimizer=optimizer,
                                scheduler=scheduler)
    trainer.train(n_epoch=100)
    return model


def train_binsoft(model: nn.Module, dataset_name="MNIST"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    trainer = TrainerGradBinarySoft(model,
                                    criterion=nn.CrossEntropyLoss(),
                                    dataset_name=dataset_name,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    hardness_scheduler=HardnessScheduler(model=model, step_size=5))
    trainer.train(n_epoch=100)
    return model


def train_mcmc(model: nn.Module, dataset_name="MNIST"):
    trainer = TrainerMCMCGibbs(model,
                               criterion=nn.CrossEntropyLoss(),
                               dataset_name=dataset_name,
                               flip_ratio=0.01)
    trainer.train(n_epoch=100, mutual_info_layers=0)
    return model


def train_tempering(model: nn.Module, dataset_name="MNIST"):
    trainer = ParallelTempering(model, criterion=nn.CrossEntropyLoss(), dataset_name=dataset_name,
                                trainer_cls=TrainerMCMCGibbs, n_chains=5, monitor_kwargs=dict(watch_parameters=False))
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
    # model = train_gradient(NetBinary(fc_sizes=(784, 10), batch_norm=True), is_binary=True, dataset_name="MNIST")
    # model = train_mcmc(model=None, dataset_name="MNIST56FullSize")
    # train_tempering(NetBinary(fc_sizes=(784, 10), batch_norm=False, scale_layer=False), dataset_name="MNIST")
    # train_mcmc(NetBinary(fc_sizes=(784, 10), batch_norm=False, scale_layer=False), dataset_name="MNIST")
    train_mcmc(NetBinary(fc_sizes=(25, 2), batch_norm=False, scale_layer=False), dataset_name="MNIST56")
    # train_binsoft(NetBinary((784, 10), batch_norm=False), dataset_name="MNIST")
