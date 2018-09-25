from typing import Union

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from layers import BinaryDecoratorSoft
from trainer.gradient import TrainerGradFullPrecision
from utils import find_layers


class HardnessScheduler:
    """
    BinaryDecoratorSoft hardness scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma_hardness=2.0, max_hardness=10):
        self.binsoft_layers = tuple(find_layers(model, layer_class=BinaryDecoratorSoft))
        self.step_size = step_size
        self.gamma_hardness = gamma_hardness
        self.max_hardness = max_hardness
        self.epoch = 0
        self.last_epoch_update = -1

    def need_update(self):
        return self.epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        if self.need_update():
            for layer in self.binsoft_layers:
                layer.hardness = min(layer.hardness * self.gamma_hardness, self.max_hardness)
            self.last_epoch_update = self.epoch
        self.epoch += 1

    def extra_repr(self):
        return f"step_size={self.step_size}, gamma_hardness={self.gamma_hardness}, max_hardness={self.max_hardness}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerGradBinarySoft(TrainerGradFullPrecision):

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 hardness_scheduler: HardnessScheduler = None,
                 **kwargs):
        super().__init__(model, criterion, dataset_name, optimizer=optimizer, scheduler=scheduler, **kwargs)
        self.hardness_scheduler = hardness_scheduler
        if self.hardness_scheduler is not None:
            self.monitor.register_func(lambda: list(layer.hardness for layer in self.hardness_scheduler.binsoft_layers),
                                       opts=dict(
                                           xlabel='Epoch',
                                           ylabel='hardness',
                                           title='BinaryDecoratorSoft tanh hardness',
                                           ytype='log',
                                       ))

    def _epoch_finished(self, epoch, outputs, labels):
        loss = super()._epoch_finished(epoch, outputs, labels)
        if self.hardness_scheduler is not None:
            self.hardness_scheduler.step(epoch=epoch)
        return loss
