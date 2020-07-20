import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets import MNIST

from mighty.models import MLP
from mighty.monitor.monitor import MonitorLevel
from mighty.monitor.mutual_info import *
from mighty.utils.data import DataLoader, TransformDefault
from trainer import *


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    return optimizer, scheduler


def train_gradient_full_precision(model: nn.Module, dataset_cls=MNIST):
    data_loader = DataLoader(dataset_cls, transform=TransformDefault.mnist())
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainerGrad(model,
                          criterion=nn.CrossEntropyLoss(),
                          data_loader=data_loader,
                          optimizer=optimizer,
                          scheduler=scheduler)
    # trainer.restore()  # uncomment to restore the saved state
    trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epochs=10, mutual_info_layers=2)
    return model


def train_gradient_binary(model: nn.Module, dataset_cls=MNIST):
    data_loader = DataLoader(dataset_cls, transform=TransformDefault.mnist())
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainerGradBinary(model,
                                criterion=nn.CrossEntropyLoss(),
                                data_loader=data_loader,
                                optimizer=optimizer,
                                scheduler=scheduler)
    trainer.train(n_epochs=100, mutual_info_layers=0)
    return model


def train_binsoft(model: nn.Module, dataset_cls=MNIST):
    data_loader = DataLoader(dataset_cls, transform=TransformDefault.mnist())
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainerGradBinarySoft(model,
                                    criterion=nn.CrossEntropyLoss(),
                                    data_loader=data_loader,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    hardness_scheduler=HardnessScheduler(model=model, step_size=5))
    trainer.train(n_epochs=100)
    return model


def train_mcmc(model: nn.Module, dataset_cls=MNIST):
    data_loader = DataLoader(dataset_cls, transform=TransformDefault.mnist())
    trainer = TrainerMCMCGibbs(model,
                               criterion=nn.CrossEntropyLoss(),
                               mutual_info=MutualInfoKMeans(data_loader),
                               data_loader=data_loader)
    trainer.train(n_epochs=100, mutual_info_layers=1)
    return model


def train_tempering(model: nn.Module, dataset_cls=MNIST):
    data_loader = DataLoader(dataset_cls, transform=TransformDefault.mnist())
    trainer = ParallelTempering(model,
                                criterion=nn.CrossEntropyLoss(),
                                data_loader=data_loader,
                                trainer_cls=TrainerMCMCGibbs,
                                n_chains=5)
    trainer.train(n_epochs=100)
    return model


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    set_seed(seed=113)
    # train_gradient_binary(NetBinary(fc_sizes=(784, 10)))
    train_mcmc(MLP(784, 10))
    # train_binsoft(MLP(784, 10))
    # train_tempering(MLP(784, 10))
