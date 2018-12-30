import time
import warnings
from abc import ABC, abstractmethod
from functools import partial, update_wrapper
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from loss import PairLoss
from monitor.accuracy import full_forward_pass, AccuracyEmbedding, AccuracyArgmax, Accuracy, calc_accuracy
from monitor.batch_timer import timer
from monitor.monitor import Monitor
from monitor.var_online import MeanOnline
from utils.common import get_data_loader
from utils.constants import CHECKPOINTS_DIR
from utils.layers import find_named_layers


class Trainer(ABC):
    watch_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, accuracy_measure: Accuracy = None,
                 env_suffix='', checkpoint_dir=CHECKPOINTS_DIR):
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.train_loader = get_data_loader(dataset_name, train=True)
        self.timer = timer
        self.timer.init(batches_in_epoch=len(self.train_loader))
        self.env_name = f"{time.strftime('%Y.%m.%d')} {self.model.__class__.__name__}: " \
            f"{self.dataset_name} {self.__class__.__name__} {self.criterion.__class__.__name__}"
        if env_suffix:
            self.env_name = self.env_name + f' {env_suffix}'
        if accuracy_measure is None:
            if isinstance(self.criterion, PairLoss):
                accuracy_measure = AccuracyEmbedding()
            else:
                # cross entropy loss
                accuracy_measure = AccuracyArgmax()
        self.accuracy_measure = accuracy_measure
        self.monitor = self._init_monitor()
        for name, layer in find_named_layers(self.model, layer_class=self.watch_modules):
            self.monitor.register_layer(layer, prefix=name)

    def _init_monitor(self) -> Monitor:
        monitor = Monitor(test_loader=get_data_loader(self.dataset_name, train=False),
                          accuracy_measure=self.accuracy_measure)
        return monitor

    @property
    def checkpoint_path(self):
        return self.checkpoint_dir / (self.env_name + '.pt')

    def monitor_functions(self):
        pass

    def log_trainer(self):
        self.monitor.log(f"Criterion: {self.criterion}")

    @abstractmethod
    def train_batch(self, images, labels):
        raise NotImplementedError()

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.state_dict(), self.checkpoint_path)
        except PermissionError as error:
            print(error)

    def state_dict(self):
        return {
            "model_state": self.model.state_dict(),
            "epoch": self.timer.epoch,
            "env_name": self.env_name,
        }

    def restore(self, checkpoint_path=None, strict=True):
        """
        :param checkpoint_path: train checkpoint path to restore
        :param strict: model's load_state_dict strict argument
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        if not checkpoint_path.exists():
            print(f"Checkpoint '{checkpoint_path}' doesn't exist. Nothing to restore.")
            return None
        map_location = None
        if not torch.cuda.is_available():
            map_location = 'cpu'
        checkpoint_state = torch.load(checkpoint_path, map_location=map_location)
        try:
            self.model.load_state_dict(checkpoint_state['model_state'], strict=strict)
        except RuntimeError as error:
            print(f"Error is occurred while restoring {checkpoint_path}: {error}")
            return None
        self.env_name = checkpoint_state['env_name']
        self.timer.set_epoch(checkpoint_state['epoch'])
        self.monitor.open(env_name=self.env_name)
        print(f"Restored model state from {checkpoint_path}.")
        return checkpoint_state

    def _epoch_finished(self, epoch, outputs, labels):
        loss = self.criterion(outputs, labels)
        self.monitor.update_loss(loss, mode='full train')
        self.save()
        return loss

    def update_batch_accuracy(self, outputs, labels):
        self.accuracy_measure.save(outputs, labels)
        labels_predicted = self.accuracy_measure.predict(outputs)
        self.monitor.update_accuracy(accuracy=calc_accuracy(labels, labels_predicted), mode='batch')

    def train_epoch(self, epoch):
        """
        :param epoch: epoch id
        :return: last batch loss
        """
        loss_batch_average = MeanOnline()
        outputs = None
        use_cuda = torch.cuda.is_available()
        for images, labels in tqdm(self.train_loader,
                                   desc="Epoch {:d}".format(epoch),
                                   leave=False):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs, loss = self.train_batch(images, labels)
            loss_batch_average.update(loss.detach().cpu())
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    warnings.warn(f"NaN parameters in '{name}'")
            self.monitor.batch_finished(self.model)

            # uncomment to see more detailed progress - at each batch instead of epoch
            # self.monitor.update_loss(loss=loss, mode='batch')
            # self.update_batch_accuracy(outputs, labels)
            # self.monitor.update_sparsity(outputs, mode='batch')
            # self.monitor.update_density(outputs, mode='batch')
            # self.monitor.activations_heatmap(outputs, labels)

        self.monitor.update_loss(loss=loss_batch_average.get_mean(), mode='batch')
        if not isinstance(self.accuracy_measure, AccuracyArgmax):
            self.monitor.update_sparsity(outputs, mode='batch')
            self.monitor.update_density(outputs, mode='batch')

    def train(self, n_epoch=10, epoch_update_step=1, mutual_info_layers=1):
        """
        :param n_epoch: number of training epochs
        :param epoch_update_step: epoch step to run full evaluation
        :param mutual_info_layers: number of last layers to be monitored for mutual information;
                                   pass '0' to turn off this feature.
        """
        print(self.model)
        if not self.monitor.is_active:
            # new environment
            self.monitor.open(env_name=self.env_name)
            self.monitor.clear()
        self.monitor_functions()
        self.monitor.log_model(self.model)
        self.monitor.log_self()
        self.log_trainer()
        print(f"Training '{self.model.__class__.__name__}'")

        eval_loader = torch.utils.data.DataLoader(dataset=self.train_loader.dataset,
                                                  batch_size=self.train_loader.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.train_loader.num_workers)

        full_forward_pass_eval = partial(full_forward_pass, loader=eval_loader)
        update_wrapper(wrapper=full_forward_pass_eval, wrapped=full_forward_pass)

        if mutual_info_layers > 0:
            full_forward_pass_eval = self.monitor.mutual_info.decorate_evaluation(full_forward_pass_eval)
            self.monitor.mutual_info.prepare(eval_loader, model=self.model, monitor_layers_count=mutual_info_layers)

        for epoch in range(self.timer.epoch, self.timer.epoch + n_epoch):
            self.train_epoch(epoch=epoch)
            if epoch % epoch_update_step == 0:
                outputs_full, labels_full = full_forward_pass_eval(self.model)
                self.accuracy_measure.save(outputs_train=outputs_full, labels_train=labels_full)
                self.monitor.epoch_finished(self.model, outputs_full, labels_full)
                self._epoch_finished(epoch, outputs_full, labels_full)
