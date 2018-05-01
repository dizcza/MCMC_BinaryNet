import math
import random
from typing import List
import copy

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

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
        self.patience = 5
        self.num_bad_epochs = 0
        self.best_loss = float('inf')
        self.best_model_state = self.model.state_dict()
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

    def accept(self, loss_new: Variable, loss_old: Variable) -> float:
        loss_delta = (loss_new - loss_old).data[0]
        if loss_delta < 0:
            proba_accept = 1.0
        else:
            proba_accept = math.exp(-loss_delta / (self.flip_ratio * 1))
        return proba_accept

    def train_batch_mcmc(self, images: Variable, labels: Variable, named_params):
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

    def reset(self):
        self.accepted_count = 0
        self.update_calls = 0
        self.num_bad_epochs = 0
        self.model.load_state_dict(self.best_model_state)

    def _epoch_finished(self, epoch, outputs, labels):
        loss = self.criterion(outputs, labels).data[0]
        self.monitor.update_loss(loss, mode='full train')
        if (epoch + 1) % 10 == 0:
            for name, param in named_parameters_binary(self.model):
                if name in self.monitor.param_data_online:
                    # continue
                    mean, std = self.monitor.param_data_online[name].get_mean_std()
                    inactive = mean.abs() / (std + 1e-7) < 0.5
                    # mean = mean.sign()
                    # mean[inactive] = 0
                    # param.data = mean
                    self.monitor.viz.text(f'{name}: {inactive.sum()} / {inactive.numel()} are turned off', win='status2')
                    param.data[inactive] = 0
                    self.monitor.param_data_online[name].mean[inactive] = 0
                    self.monitor.param_data_online[name].var[inactive] = 0
                    # if inactive.any():
                    #     self.monitor.param_data_online[name].reset()
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
            self.best_model_state = copy.deepcopy(self.model.state_dict())
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


class TrainerMCMCTree(TrainerMCMC):

    def train_batch(self, images, labels):
        return self.train_batch_mcmc(images, labels, named_params=named_parameters_binary(self.model))


class TrainerMCMCGibbs(TrainerMCMC):
    """
    Probability to find a model in a given state is ~ exp(-loss/kT).
    """

    def accept(self, loss_new: Variable, loss_old: Variable) -> float:
        loss_delta = (loss_old - loss_new).data[0]
        try:
            proba_accept = 1 / (1 + math.exp(-loss_delta / (self.flip_ratio * 0.1)))
        except OverflowError:
            proba_accept = int(loss_delta > 0)
        return proba_accept
