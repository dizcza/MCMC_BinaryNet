import math
import random
from typing import List

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from utils.layers import compile_inference, binarize_model
from monitor.monitor_mcmc import MonitorMCMC
from trainer.trainer import Trainer
from utils.binary_param import named_parameters_binary
from utils.common import get_data_loader


class ParameterFlip(object):
    def __init__(self, name: str, param: nn.Parameter, source: List[int], sink: List[int]):
        """
        :param name: (sink) parameter's name
        :param param: nn.Parameter
        :param source: input layer neuron indices
        :param sink: output layer neuron indices
        """
        assert param.ndimension() == 2, "For now, only nn.Linear is supported"
        self.name = name
        self.param = param
        self.source = torch.as_tensor(source, dtype=torch.int64, device=self.param.device)
        self.sink = torch.as_tensor(sink, dtype=torch.int64, device=self.param.device).unsqueeze(dim=1)

    @property
    def source_expanded(self):
        return self.source.expand(len(self.sink), -1)

    def construct_flip(self) -> torch.ByteTensor:
        idx_connection_flip = torch.ByteTensor(self.param.data.shape).fill_(0)
        idx_connection_flip[self.sink, self.source_expanded] = 1
        return idx_connection_flip

    def flip(self):
        self.param[self.sink, self.source_expanded] *= -1

    def restore(self):
        self.flip()


class ParameterFlipCached(ParameterFlip):
    def __init__(self, name: str, param: nn.Parameter, source: List[int], sink: List[int]):
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


class TrainerMCMC(Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, flip_ratio=0.1, **kwargs):
        model = binarize_model(model)
        compile_inference(model)
        super().__init__(model=model, criterion=criterion, dataset_name=dataset_name, **kwargs)
        self.flip_ratio = flip_ratio
        self.accepted_count = 0
        self.update_calls = 0
        for param in model.parameters():
            param.requires_grad_(False)

    def _init_monitor(self):
        monitor = MonitorMCMC(test_loader=get_data_loader(self.dataset_name, train=False),
                              accuracy_measure=self.accuracy_measure, model=self.model)
        return monitor

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(f"Flip ratio: {self.flip_ratio}")

    def get_acceptance_ratio(self) -> float:
        if self.update_calls == 0:
            return 0
        else:
            return self.accepted_count / self.update_calls

    def sample_neurons(self, size) -> List[int]:
        return random.sample(range(size), k=math.ceil(size * self.flip_ratio))

    def accept(self, loss_new: torch.Tensor, loss_old: torch.Tensor) -> float:
        loss_delta = (loss_new - loss_old).item()
        if loss_delta < 0:
            proba_accept = 1.0
        else:
            proba_accept = math.exp(-loss_delta / (self.flip_ratio * 1))
        return proba_accept

    def train_batch_mcmc(self, images, labels, named_params):
        outputs_orig = self.model(images)
        loss_orig = self.criterion(outputs_orig, labels)

        param_flips = []
        source = None
        for name, param in named_params:
            size_output, size_input = param.data.shape
            if source is None:
                source = self.sample_neurons(size_input)
            sink = self.sample_neurons(size_output)
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

        self.update_calls += 1
        return outputs, loss

    def train_batch(self, images, labels):
        return self.train_batch_mcmc(images, labels, named_params=[
            random.choice(named_parameters_binary(self.model))
        ])

    def monitor_functions(self):
        super().monitor_functions()

        def acceptance_ratio(viz):
            viz.line_update(y=self.get_acceptance_ratio(), opts=dict(
                xlabel='Epoch',
                ylabel='Acceptance ratio',
                title='MCMC accepted / total_tries'
            ))

        self.monitor.register_func(acceptance_ratio)


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
            proba_accept = 1 / (1 + math.exp(-loss_delta / (self.flip_ratio * 1)))
        except OverflowError:
            proba_accept = int(loss_delta > 0)
        return proba_accept
