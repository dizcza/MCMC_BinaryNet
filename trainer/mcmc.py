import math
import random

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from mighty.trainer.trainer import Trainer
from mighty.utils.data import DataLoader, get_normalize_inverse

from monitor.monitor_mcmc import MonitorMCMC
from utils.binary_param import named_parameters_binary
from utils.layers import compile_inference, binarize_model


class TemperatureScheduler:
    """
    Flip ratio (temperature) scheduler.
    """

    def __init__(self, temperature_init=0.05, step_size=10, gamma_temperature=0.5, min_temperature=1e-4,
                 boltzmann_const=1.):
        """
        :param temperature_init: initial temperature
        :param step_size: epoch steps
        :param gamma_temperature: temperature down-factor
        :param min_temperature: min temperature
        :param boltzmann_const: Boltzmann's constant
        """
        self.temperature = temperature_init
        self.step_size = step_size
        self.gamma_temperature = gamma_temperature
        self.min_temperature = min_temperature
        self.last_epoch_update = -1
        self.boltzmann_const = boltzmann_const

    @property
    def energy(self):
        return self.temperature * self.boltzmann_const

    def need_update(self, epoch: int):
        return epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch: int):
        if self.need_update(epoch):
            self.temperature = max(self.temperature * self.gamma_temperature, self.min_temperature)
            self.last_epoch_update = epoch

    def state_dict(self):
        return {
            'last_epoch_update': self.last_epoch_update
        }

    def load_state_dict(self, state_dict: dict):
        self.last_epoch_update = state_dict['last_epoch_update']

    def extra_repr(self):
        return f"step_size={self.step_size}, gamma_flip={self.gamma_temperature}, " \
            f"min_flip={self.min_temperature}, boltzmann_const={self.boltzmann_const}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TemperatureSchedulerConstant(TemperatureScheduler):
    def __init__(self, temperature: float, boltzmann_const=1.):
        super().__init__(temperature_init=temperature, min_temperature=temperature, boltzmann_const=boltzmann_const)


class ParameterFlip(object):
    def __init__(self, name: str, param: nn.Parameter, source: torch.LongTensor, sink: torch.LongTensor):
        """
        :param name: (sink) parameter's name
        :param param: nn.Parameter
        :param source: input layer neuron indices
        :param sink: output layer neuron indices
        """
        assert param.ndimension() == 2, "For now, only nn.Linear is supported"
        self.name = name
        self.param = param
        self.source = source
        self.sink = sink
        self.is_flipped = False

    def construct_flip(self):
        idx_connection_flip = torch.zeros(self.param.shape, dtype=torch.int32)
        idx_connection_flip[self.sink, self.source] = 1
        return idx_connection_flip

    def flip(self):
        self.param[self.sink, self.source] *= -1
        self.is_flipped = not self.is_flipped

    def restore(self):
        if self.is_flipped:
            self.flip()


class ParameterFlipCached(ParameterFlip):
    def __init__(self, name: str, param: nn.Parameter, source: torch.LongTensor, sink: torch.LongTensor):
        """
        :param name: (sink) parameter's name
        :param param: nn.Parameter
        :param source: input layer neuron indices
        :param sink: output layer neuron indices
        """
        super().__init__(name, param, source, sink)
        self.data_backup = self.param.data.clone()

    def restore(self):
        self.param.data = self.data_backup
        self.is_flipped = False


class TrainerMCMC(Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, data_loader: DataLoader,
                 temperature_scheduler=TemperatureScheduler(), **kwargs):
        model = binarize_model(model)
        compile_inference(model)
        super().__init__(model=model, criterion=criterion, data_loader=data_loader, **kwargs)
        self.temperature_scheduler = temperature_scheduler
        self.accepted_count = 0
        for param in model.parameters():
            param.requires_grad_(False)

    def _init_monitor(self, mutual_info):
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorMCMC(
            model=self.model,
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(self.temperature_scheduler)

    def get_acceptance_ratio(self) -> float:
        return self.accepted_count / (self.timer.batch_id + 1)

    def neurons_to_flip(self, size: int) -> int:
        return math.ceil(size * self.temperature_scheduler.temperature)

    def accept(self, loss_new: torch.Tensor, loss_old: torch.Tensor) -> float:
        loss_delta = (loss_new - loss_old).item()
        if loss_delta < 0:
            proba_accept = 1.0
        else:
            proba_accept = math.exp(-loss_delta / self.temperature_scheduler.energy)
        return proba_accept

    def train_batch_mcmc(self, images, labels, named_params):
        outputs_orig = self.model(images)
        loss_orig = self.criterion(outputs_orig, labels)

        param_flips = []
        source = None
        for name, param in named_params:
            size_output, size_input = param.shape
            sink = torch.randint(low=0, high=size_output, size=(self.neurons_to_flip(size_output), 1),
                                 device=param.device)
            if source is None:
                source = torch.randint(low=0, high=size_input,
                                       size=(sink.shape[0], self.neurons_to_flip(size_input)),
                                       device=param.device)
            pflip = ParameterFlipCached(name, param, source, sink)
            param_flips.append(pflip)
            source = sink

        for pflip in param_flips:
            pflip.flip()

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        proba_accept = self.accept(loss_new=loss, loss_old=loss_orig)
        proba_draw = random.random()
        if proba_draw <= proba_accept:
            self.accepted_count += 1
        else:
            # reject
            for pflip in param_flips:
                pflip.restore()
            outputs = outputs_orig
            loss = loss_orig
        self.monitor.mcmc_step(param_flips)

        del param_flips

        return outputs, loss

    def train_batch(self, images, labels):
        return self.train_batch_mcmc(images, labels, named_params=[
            random.choice(named_parameters_binary(self.model))
        ])

    def _epoch_finished(self, epoch, outputs, labels):
        loss = super()._epoch_finished(epoch=epoch, outputs=outputs, labels=labels)
        self.temperature_scheduler.step(epoch=epoch)
        return loss

    def monitor_functions(self):
        super().monitor_functions()

        def acceptance_ratio(viz):
            viz.line_update(y=self.get_acceptance_ratio(), opts=dict(
                xlabel='Epoch',
                ylabel='Acceptance ratio',
                title='MCMC accepted / total_tries'
            ))

        def temperature(viz):
            viz.line_update(y=self.temperature_scheduler.temperature, opts=dict(
                xlabel='Epoch',
                ylabel='Temperature',
                title='Surrounding temperature',
                ytype='log',
            ))

        self.monitor.register_func(acceptance_ratio)
        self.monitor.register_func(temperature)


class TrainerMCMCTree(TrainerMCMC):

    def train_batch(self, images, labels):
        return self.train_batch_mcmc(images, labels, named_params=named_parameters_binary(self.model))


class TrainerMCMCGibbs(TrainerMCMC):
    """
    Probability to find a model in a given state is ~ exp(-loss/kT).
    """

    def accept(self, loss_new: torch.Tensor, loss_old: torch.Tensor) -> float:
        loss_delta = (loss_old - loss_new).item()
        try:
            proba_accept = 1 / (1 + math.exp(-loss_delta / self.temperature_scheduler.energy))
        except OverflowError:
            proba_accept = int(loss_delta > 0)
        return proba_accept
