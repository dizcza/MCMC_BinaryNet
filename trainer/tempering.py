import copy
import math
import random

import torch.nn as nn

from utils.layers import compile_inference, binarize_model
from trainer.mcmc import TrainerMCMC, TemperatureSchedulerConstant
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
        temperature_min = 0.001
        temperature_max = 0.9
        temperature_multiplier = math.pow(temperature_max / temperature_min, 1 / (n_chains - 1))
        self.exchanges_count = 0
        self.trainers = []
        for trainer_id in range(n_chains):
            temperature = temperature_min * temperature_multiplier ** trainer_id
            trainer = trainer_cls(model=clone_model(model), criterion=self.criterion,
                                  dataset_name=self.dataset_name,
                                  temperature_scheduler=TemperatureSchedulerConstant(temperature))
            self.trainers.append(trainer)

    def log_trainer(self):
        super().log_trainer()
        temperatures = (trainer.temperature_scheduler.temperature for trainer in self.trainers)
        self.monitor.log(f"Started {len(self.trainers)} {self.trainers[0].__class__.__name__} chains with flip ratios "
                         f"(temperatures between 0 and 1): {temperatures}")

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
            temperature_curr = trainer.temperature_scheduler.temperature
            temperature_prev = trainer_prev.temperature_scheduler.temperature
            proba_swap = math.exp(energy_diff / (1 / temperature_curr - 1 / temperature_prev))
            if random.random() < proba_swap:
                model_curr = trainer.model
                trainer.model = trainer_prev.model
                trainer_prev.model = model_curr
                self.exchanges_count += 1
        return best_outputs, best_loss

    def get_exchanges_ratio(self):
        """
        :return: Number of chain pairs which exchanged their temperatures per batch.
        """
        return self.exchanges_count / (self.timer.batch_id + 1)

    def monitor_functions(self):
        super().monitor_functions()

        def exchanges_ratio(viz):
            viz.line_update(y=self.get_exchanges_ratio(), opts=dict(
                xlabel='Epoch',
                ylabel='Exchanged temperatures ratio',
                title='Replica exchange MCMC sampling'
            ))

        self.monitor.register_func(exchanges_ratio)
