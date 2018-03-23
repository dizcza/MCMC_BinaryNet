import math
import random
from typing import List

import torch
import torch.nn as nn
import torch.utils.data

from layers import compile_inference
from monitor.monitor import MonitorMCMC
from trainer.trainer import Trainer
from utils import named_parameters_binary


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
        self.source = source
        self.sink = sink

    def construct_flip(self) -> torch.ByteTensor:
        # hack to select and modify a sub-matrix
        idx_connection_flip = torch.ByteTensor(self.param.data.shape).fill_(0)
        idx_connection_flip_output = idx_connection_flip[self.sink, :]
        idx_connection_flip_output[:, self.source] = True
        idx_connection_flip[self.sink, :] = idx_connection_flip_output
        return idx_connection_flip

    def flip(self):
        idx_flipped = self.get_idx_flipped()
        idx_flipped_cuda = idx_flipped
        if self.param.is_cuda:
            idx_flipped_cuda = idx_flipped.cuda()
        self.param[idx_flipped_cuda] *= -1
        del idx_flipped_cuda

    def get_idx_flipped(self) -> torch.ByteTensor:
        return self.construct_flip()

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
        self.idx_flipped = self.construct_flip()

    def get_idx_flipped(self):
        return self.idx_flipped

    def restore(self):
        self.param.data = self.data_backup


class TrainerMCMC(Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, flip_ratio=0.1):
        compile_inference(model)
        super().__init__(model, criterion, dataset_name, monitor_cls=MonitorMCMC)
        self.volatile = True
        self.flip_ratio = flip_ratio
        self.monitor.log(f"Flip ratio: {flip_ratio}")
        self.accepted_count = 0
        self.update_calls = 0
        self.loss_delta_mean = 0
        self.patience = 5
        self.num_bad_epochs = 0
        self.best_loss = float('inf')
        for param in model.parameters():
            param.requires_grad = False
            param.volatile = True
        self._monitor_functions()

    def get_acceptance_ratio(self) -> float:
        if self.update_calls == 0:
            return 0
        else:
            return self.accepted_count / self.update_calls

    def sample_neurons(self, size) -> List[int]:
        return random.sample(range(size), k=math.ceil(size * self.flip_ratio))

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
        loss_delta = (loss - loss_orig).data[0]
        if loss_delta < 0:
            mcmc_proba_accept = 1.0
        else:
            mcmc_proba_accept = math.exp(-loss_delta / (self.flip_ratio * len(param_flips) ** 2))
        proba_draw = random.random()
        if proba_draw < mcmc_proba_accept:
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
        self.loss_delta_mean += (abs(loss_delta) - self.loss_delta_mean) / self.update_calls
        return outputs, loss

    def train_batch(self, images, labels):
        return self.train_batch_mcmc(images, labels, named_params=[
            random.choice(named_parameters_binary(self.model))
        ])

    def reset(self):
        self.accepted_count = 0
        self.update_calls = 0
        self.loss_delta_mean = 0
        self.num_bad_epochs = 0

    def _epoch_finished(self, epoch, outputs, labels):
        loss = self.criterion(outputs, labels).data[0]
        self.monitor.update_loss(loss, mode='full train')
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            self.flip_ratio = max(self.flip_ratio * 0.7, 1e-4)
            self.reset()

    def _monitor_functions(self):
        self.monitor.register_func(self.get_acceptance_ratio, opts=dict(
            xlabel='Epoch',
            ylabel='Acceptance ratio',
            title='MCMC accepted / total_tries'
        ))
        self.monitor.register_func(lambda: self.flip_ratio * 100., opts=dict(
            xlabel='Epoch',
            ylabel='Sign flip ratio, %',
            title='MCMC flipped / total_neurons per layer'
        ))
        # self.monitor.register_func(lambda: self.loss_delta_mean, opts=dict(
        #     xlabel='Epoch',
        #     ylabel='|ΔL|',
        #     title='MCMC |Loss(flipped) - Loss(origin)|'
        # ))


class TrainerMCMCTree(TrainerMCMC):

    def train_batch(self, images, labels):
        return self.train_batch_mcmc(images, labels, named_params=named_parameters_binary(self.model))
