import copy
import math
import random

import torch.nn as nn

from utils.layers import compile_inference, binarize_model
from trainer.mcmc import TrainerMCMC
from trainer.trainer import Trainer
from utils.binary_param import named_parameters_binary


def clone_model(model: nn.Module) -> nn.Module:
    names_binary = set(name for name, param in named_parameters_binary(model))
    model_copied = copy.deepcopy(model)
    for name, param in model_copied.named_parameters():
        if name in names_binary:
            param.is_binary = True
    return model_copied


class ParallelTempering(Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, trainer_cls=TrainerMCMC,
                 n_chains=5, **kwargs):
        model = binarize_model(model)
        compile_inference(model)
        super().__init__(model=model, criterion=criterion, dataset_name=dataset_name, **kwargs)
        self.monitor.log(f"Parallel tempering chain trainer: {trainer_cls.__name__}")
        flip_min = 0.001
        flip_max = 0.9
        temperature_multiplier = math.pow(flip_max / flip_min, 1 / (n_chains - 1))
        flip_ratio_chains = []
        self.trainers = []
        for trainer_id in range(n_chains):
            flip_ratio = flip_min * temperature_multiplier ** trainer_id
            flip_ratio_chains.append(flip_ratio)
            trainer = trainer_cls(model=model, criterion=self.criterion,
                                  dataset_name=self.dataset_name, flip_ratio=flip_ratio,
                                  monitor_kwargs=dict(is_active=False, watch_parameters=False))
            self.trainers.append(trainer)
            model = clone_model(model)
        self.monitor.log(f"Started {n_chains} chains with flip ratios (temperatures between 0 and 1): "
                         f"{flip_ratio_chains}")

    @staticmethod
    def to_temperature(flip_ratio):
        return flip_ratio * math.log(10)

    def train_batch(self, images, labels):
        best_outputs, best_loss = self.trainers[0].train_batch(images, labels)
        prev_loss = best_loss
        for trainer_prev, trainer in zip(self.trainers[:-1], self.trainers[1:]):
            outputs, loss = trainer.train_batch(images, labels)
            if loss.item() < best_loss.item():
                best_loss = loss
                best_outputs = outputs
                self.model = trainer.model
            energy_diff = loss.item() - prev_loss.item()
            temperature_curr = self.to_temperature(trainer.flip_ratio)
            temperature_prev = self.to_temperature(trainer_prev.flip_ratio)
            proba_swap = math.exp(energy_diff / (1 / temperature_curr - 1 / temperature_prev))
            if random.random() < proba_swap:
                model_curr = trainer.model
                trainer.model = trainer_prev.model
                trainer_prev.model = model_curr
        return best_outputs, best_loss
