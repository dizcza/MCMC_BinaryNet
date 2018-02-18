import copy
import random
import math
from typing import Iterable, Union

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

from constants import MODELS_DIR
from monitor import Monitor, calc_accuracy, get_outputs, get_softmax_accuracy
from utils import get_data_loader, named_parameters_binary, parameters_binary, load_model_state, MNISTSmall
from layers import compile_inference


class _Trainer(object):

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str):
        self.model = model
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.train_loader = get_data_loader(dataset_name, train=True)
        self.monitor = Monitor(self)
        self._register_monitor_parameters()

    def save_model(self, accuracy: float = None):
        model_path = MODELS_DIR.joinpath(self.dataset_name, self.model.__class__.__name__).with_suffix('.pt')
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), model_path)
        msg = f"Saved to {model_path}"
        if accuracy is not None:
            msg += f" (train accuracy: {accuracy:.4f})"
        print(msg)

    def load_best_accuracy(self, debug=False) -> float:
        best_accuracy = 0.
        if not debug:
            try:
                model_state = load_model_state(self.dataset_name, self.model.__class__.__name__)
                loaded_model = copy.deepcopy(self.model)
                loaded_model.load_state_dict(model_state)
                loaded_model.eval()
                for param in loaded_model.parameters():
                    param.requires_grad = False
                    param.volatile = True
                best_accuracy = calc_accuracy(loaded_model, self.train_loader)
                del loaded_model
            except Exception as e:
                print(f"Couldn't estimate the best accuracy for {self.model.__class__.__name__}. Reset to 0.")
        return best_accuracy

    def _register_monitor_parameters(self):
        for name, param in named_parameters_binary(self.model):
            self.monitor.register_param(name, param)

    def _train_batch(self, images, labels):
        raise NotImplementedError()

    def _epoch_finished(self, epoch, outputs, labels):
        pass
    
    def train(self, n_epoch=10, debug=False):
        print(self.model)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        best_accuracy = self.load_best_accuracy(debug)
        self.monitor.log(f"Best train accuracy so far: {best_accuracy:.4f}")
        print(f"Training '{self.model.__class__.__name__}'. "
              f"Best {self.dataset_name} train accuracy so far: {best_accuracy:.4f}")

        for epoch in range(n_epoch):
            for images, labels in tqdm(self.train_loader,
                                       desc="Epoch {:d}/{:d}".format(epoch, n_epoch),
                                       leave=False):
                images = Variable(images)
                labels = Variable(labels)
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs, loss = self._train_batch(images, labels)
                self.monitor.batch_finished(outputs, labels, loss)

            outputs_full, labels_full = get_outputs(self.model, self.train_loader)
            accuracy = get_softmax_accuracy(outputs_full, labels_full)
            is_best = accuracy > best_accuracy
            self.monitor.update_train_accuracy(accuracy, is_best)
            if is_best:
                if not debug:
                    self.save_model(accuracy)
                best_accuracy = accuracy

            self._epoch_finished(epoch, outputs_full, labels_full)
            self.monitor.epoch_finished()


class TrainerGradFullPrecision(_Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None):
        super().__init__(model, criterion, dataset_name)
        self.optimizer = optimizer
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.monitor.register_func(lambda: list(group['lr'] for group in self.optimizer.param_groups), opts=dict(
                xlabel='Epoch',
                ylabel='Learning rate',
                title='Learning rate'
            ))

    def _register_monitor_parameters(self):
        for name, param in self.model.named_parameters():
            self.monitor.register_param(name, param)

    def _train_batch(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step(closure=None)
        return outputs, loss

    def _epoch_finished(self, epoch, outputs, labels):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            loss = self.criterion(outputs, labels).data[0]
            self.scheduler.step(metrics=loss, epoch=epoch)
        elif isinstance(self.scheduler, _LRScheduler):
            self.scheduler.step(epoch=epoch)


class TrainerGradBinary(TrainerGradFullPrecision):

    def _train_batch(self, images, labels):
        outputs, loss = super()._train_batch(images, labels)
        for param in parameters_binary(self.model):
            param.data.clamp_(min=-1, max=1)
        return outputs, loss

    def _register_monitor_parameters(self):
        for name, param in named_parameters_binary(self.model):
            self.monitor.register_param(name, param)


class TrainerMCMC(_Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, flip_ratio=0.1):
        compile_inference(model)
        super().__init__(model, criterion, dataset_name)
        self.flip_ratio = flip_ratio
        self.monitor.log(f"Flip ratio: {flip_ratio}")
        self.accepted_count = 0
        self.update_calls = 0
        self.loss_delta_mean = 0
        self.patience = 5
        self.num_bad_epochs = 0
        self.plot_autocorrelation = isinstance(self.train_loader.dataset, MNISTSmall)
        self.best_loss = float('inf')
        for param in model.parameters():
            param.requires_grad = False
            param.volatile = True

    def get_acceptance_ratio(self) -> float:
        if self.update_calls == 0:
            return 0
        else:
            return self.accepted_count / self.update_calls

    def _train_batch(self, images, labels):
        outputs_orig = self.model(images)
        loss_orig = self.criterion(outputs_orig, labels)

        param_modified, idx_to_flip = self.choose_weights_to_flip(parameters_binary(self.model))
        data_orig = param_modified.data.clone()
        idx_to_flip_cuda = idx_to_flip
        if torch.cuda.is_available():
            idx_to_flip_cuda = idx_to_flip.cuda()
        param_modified[idx_to_flip_cuda] *= -1

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss_delta = (loss - loss_orig).data[0]
        if loss_delta < 0:
            mcmc_proba_accept = 1.0
        else:
            mcmc_proba_accept = math.exp(-loss_delta / self.flip_ratio)
        proba_draw = random.random()
        if proba_draw < mcmc_proba_accept:
            self.accepted_count += 1
            if self.plot_autocorrelation:
                self.monitor.update_autocorrelation(idx_to_flip)
        else:
            # reject
            param_modified.data = data_orig
            outputs = outputs_orig
            loss = loss_orig
        self.update_calls += 1

        self.loss_delta_mean += (abs(loss_delta) - self.loss_delta_mean) / self.update_calls

        return outputs, loss

    def reset(self):
        # self.accepted_count = 0
        # self.update_calls = 0
        # self.loss_delta_mean = 0
        self.num_bad_epochs = 0

    def _epoch_finished(self, epoch, outputs, labels):
        loss = self.criterion(outputs, labels).data[0]
        self.monitor.update_loss(loss, mode='full dataset')
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            self.flip_ratio = max(self.flip_ratio * 0.7, 1e-4)
            self.reset()

    def choose_weights_to_flip(self, parameters: Iterable[nn.Parameter]):
        param_modified = random.choice(list(parameters))
        assert param_modified.ndimension() == 2, "For now, only nn.Linear is supported"

        def sample_neurons(size):
            return random.sample(range(size), k=math.ceil(size * self.flip_ratio))

        size_output, size_input = param_modified.data.shape
        idx_output = sample_neurons(size_output)
        idx_input = sample_neurons(size_input)

        # hack to select and modify a sub-matrix
        idx_connection_flip = torch.ByteTensor(param_modified.data.shape).fill_(0)
        idx_connection_flip_output = idx_connection_flip[idx_output, :]
        idx_connection_flip_output[:, idx_input] = True
        idx_connection_flip[idx_output, :] = idx_connection_flip_output

        return param_modified, idx_connection_flip

    def _register_monitor_parameters(self):
        super()._register_monitor_parameters()
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
        self.monitor.register_func(lambda: self.loss_delta_mean, opts=dict(
            xlabel='Epoch',
            ylabel='|ΔL|',
            title='MCMC |Loss(flipped) - Loss(origin)|'
        ))
