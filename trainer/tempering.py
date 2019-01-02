import copy
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn

from monitor.var_online import MeanOnline
from trainer.mcmc import TrainerMCMC, TemperatureSchedulerConstant
from trainer.trainer import Trainer
from utils.binary_param import named_parameters_binary
from utils.layers import compile_inference, binarize_model


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
        self.exchanges = defaultdict(int)
        self.trainers = []
        self.proba_exchange = MeanOnline()
        for trainer_id in range(n_chains):
            temperature = temperature_min * temperature_multiplier ** trainer_id
            trainer = trainer_cls(model=clone_model(model), criterion=self.criterion,
                                  dataset_name=self.dataset_name,
                                  temperature_scheduler=TemperatureSchedulerConstant(temperature, boltzmann_const=10))
            self.trainers.append(trainer)

    @property
    def temperatures(self):
        return list(f"{trainer.temperature_scheduler.temperature:.3f}" for trainer in self.trainers)

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(f"Started {len(self.trainers)} {self.trainers[0].__class__.__name__} chains "
                         f"with such temperatures: {self.temperatures}")

    def train_batch(self, images, labels):
        best_outputs, best_loss = self.trainers[0].train_batch(images, labels)
        prev_loss = best_loss
        self.model = self.trainers[0].model
        for trainer_id in range(1, len(self.trainers)):
            trainer = self.trainers[trainer_id]
            trainer_prev = self.trainers[trainer_id - 1]
            outputs, loss = trainer.train_batch(images, labels)
            if loss.item() < best_loss.item():
                best_loss = loss
                best_outputs = outputs
                self.model = trainer.model
            energy_diff = loss.item() - prev_loss.item()
            energy_curr = trainer.temperature_scheduler.energy
            energy_prev = trainer_prev.temperature_scheduler.energy
            proba_swap = math.exp(energy_diff / (1 / energy_curr - 1 / energy_prev))
            proba_swap = min(proba_swap, 1.0)
            self.proba_exchange.update(torch.Tensor([proba_swap]))
            do_exchange = random.random() < proba_swap
            self.exchanges[(trainer_id - 1, trainer_id)] += do_exchange
            if do_exchange:
                model_curr = trainer.model
                trainer.model = trainer_prev.model
                trainer_prev.model = model_curr
        return best_outputs, best_loss

    def monitor_functions(self):
        super().monitor_functions()

        def proba_exchange(viz):
            viz.line_update(y=self.proba_exchange.get_mean(), opts=dict(
                xlabel='Epoch',
                ylabel='Probability of chain temperatures exchange',
                title='Replica exchange MCMC proba sampling'
            ))

        def acceptance_ratio(viz):
            temperatures = [f"T {t}" for t in self.temperatures]
            viz.line_update(y=[trainer.get_acceptance_ratio() for trainer in self.trainers], opts=dict(
                xlabel='Epoch',
                ylabel='Acceptance ratio',
                title='MCMC accepted / total_tries',
                legend=temperatures,
            ))

        def chain_exchanges(viz):
            _chain_exchanges = list(value / (self.timer.batch_id + 1) for value in self.exchanges.values())
            temperatures = []
            for trainer_id in range(1, len(self.trainers)):
                t_prev = self.trainers[trainer_id - 1].temperature_scheduler.temperature
                t_curr = self.trainers[trainer_id].temperature_scheduler.temperature
                temperatures.append(f"T {t_prev:.3f}-{t_curr:.3f}")
            viz.line_update(y=_chain_exchanges, opts=dict(
                xlabel='Epoch',
                title='Chain exchange acceptance',
                legend=temperatures,
                ytype='log',
            ))

        self.monitor.register_func(proba_exchange)
        self.monitor.register_func(acceptance_ratio)
        self.monitor.register_func(chain_exchanges)
