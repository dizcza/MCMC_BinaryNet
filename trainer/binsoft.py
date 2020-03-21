from typing import Union

import torch.nn as nn
import torch.utils.data
from mighty.trainer.gradient import TrainerGrad
from mighty.utils.common import find_layers
from mighty.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from utils.layers import BinaryDecoratorSoft, binarize_model


class HardnessScheduler:
    """
    BinaryDecoratorSoft hardness scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma_hardness=2.0, max_hardness=10):
        self.binsoft_layers = tuple(find_layers(model, layer_class=BinaryDecoratorSoft))
        self.step_size = step_size
        self.gamma_hardness = gamma_hardness
        self.max_hardness = max_hardness
        self.last_epoch_update = -1

    def need_update(self, epoch: int):
        return epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch: int):
        if self.need_update(epoch):
            for layer in self.binsoft_layers:
                layer.hardness = min(layer.hardness * self.gamma_hardness, self.max_hardness)
            self.last_epoch_update = epoch

    def extra_repr(self):
        return f"step_size={self.step_size}, gamma_hardness={self.gamma_hardness}, max_hardness={self.max_hardness}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerGradBinarySoft(TrainerGrad):

    def __init__(self, model: nn.Module, criterion: nn.Module, data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 hardness_scheduler: HardnessScheduler = None,
                 **kwargs):
        model = binarize_model(model, binarizer=BinaryDecoratorSoft)
        super().__init__(model, criterion, data_loader=data_loader, optimizer=optimizer, scheduler=scheduler, **kwargs)
        self.hardness_scheduler = hardness_scheduler

    def monitor_functions(self):
        super().monitor_functions()

        def hardness(viz):
            viz.line_update(y=list(layer.hardness for layer in self.hardness_scheduler.binsoft_layers), opts=dict(
                xlabel='Epoch',
                ylabel='hardness',
                title='BinaryDecoratorSoft tanh hardness',
                ytype='log',
            ))

        if self.hardness_scheduler is not None:
            self.monitor.register_func(hardness)

    def _epoch_finished(self, epoch, outputs, labels):
        loss = super()._epoch_finished(epoch, outputs, labels)
        if self.hardness_scheduler is not None:
            self.hardness_scheduler.step(epoch=epoch)
        return loss
