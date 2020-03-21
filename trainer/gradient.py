from typing import Union

import torch.nn as nn
import torch.utils.data
from mighty.trainer import TrainerGrad
from mighty.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from utils.binary_param import parameters_binary
from utils.layers import binarize_model


class TrainerGradBinary(TrainerGrad):

    def __init__(self, model: nn.Module, criterion: nn.Module, data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 **kwargs):
        model = binarize_model(model)
        super().__init__(model=model, criterion=criterion, data_loader=data_loader, optimizer=optimizer,
                         scheduler=scheduler, **kwargs)

    def train_batch(self, images, labels):
        outputs, loss = super().train_batch(images, labels)
        for param in parameters_binary(self.model):
            param.data.clamp_(min=-1, max=1)
        return outputs, loss
