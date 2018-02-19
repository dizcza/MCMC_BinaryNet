import time
import math
from statsmodels.tsa.stattools import acf, ccf
from typing import Union, List, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import visdom
from torch.autograd import Variable

from utils import get_data_loader, parameters_binary, find_param_by_name


def get_softmax_accuracy(outputs, labels) -> float:
    _, labels_predicted = torch.max(outputs.data, 1)
    softmax_accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return softmax_accuracy


def timer(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:e} sec")
        return result

    return wrapped


def get_outputs(model: nn.Module, loader: torch.utils.data.DataLoader):
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in iter(loader):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable(inputs, volatile=True))
        outputs_full.append(outputs)
        labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    labels_full = Variable(labels_full, volatile=True)
    return outputs_full, labels_full


def calc_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    if model is None:
        return 0.0
    outputs, labels = get_outputs(model, loader)
    accuracy = get_softmax_accuracy(outputs, labels)
    return accuracy


def test(model: nn.Module, dataset_name: str,  train=False):
    loader = get_data_loader(dataset=dataset_name, train=train)
    accur = calc_accuracy(model, loader)
    print(f"Model={model.__class__.__name__} dataset={dataset_name} train={train} accuracy: {accur:.4f}")


class VarianceOnline(object):

    """
    Online updating sample mean and unbiased variance in a single pass.
    """

    def __init__(self):
        self.mean = None
        self.var = None
        self.count = 0

    def update(self, new_tensor: torch.FloatTensor):
        self.count += 1
        if self.mean is None:
            self.mean = new_tensor.clone()
            self.var = torch.zeros_like(self.mean)
        else:
            self.var = (self.count - 2) / (self.count - 1) * self.var + torch.pow(new_tensor - self.mean, 2) / self.count
            self.mean += (new_tensor - self.mean) / self.count

    def get_mean_std(self):
        if self.mean is None:
            return None, None
        else:
            return self.mean.clone(), torch.sqrt(self.var)


class UpdateTimer(object):

    def __init__(self, max_skip: int):
        self.max_skip = max_skip
        self.next_update = 10

    def need_update(self, batch_id: int):
        if batch_id >= self.next_update:
            self.next_update = min(int((batch_id + 1) ** 1.1), batch_id + self.max_skip)
            return True
        return False


class ParamRecord(object):
    def __init__(self, name: str, param: nn.Parameter):
        self.name = name
        self.param = param
        self.variance = VarianceOnline()
        self.prev_sign = param.data.clone()  # clone is faster


class ParamList(list):
    def __init__(self):
        super().__init__()
        self.sign_flips = 0
        self.n_updates = 0

    # todo move auto- & cross-correlation here
    def batch_finished(self):
        self.n_updates += 1
        for param_record in self:
            param = param_record.param
            self.sign_flips += torch.sum((param.data * param_record.prev_sign) < 0)
            param_record.prev_sign = param.data.clone()

    def get_sign_flips(self):
        if len(self) == 0:
            # haven't registered any param yet
            return None
        param_count = 0
        for param_record in self:
            param_count += torch.numel(param_record.param)
        flips = self.sign_flips
        flips /= param_count  # per param
        flips /= self.n_updates  # per update
        flips *= 100.  # percents
        self.sign_flips = 0
        self.n_updates = 0
        return flips


class Autocorrelation(object):
    def __init__(self):
        self.samples = []

    def add_sample(self, new_sample):
        self.samples.append(new_sample)

    def plot_acf_ccf(self, viz: visdom.Visdom, nlags=30):
        if len(self.samples) == 0:
            return

        def strongest_correlation_id(coef_vars_lags) -> int:
            accumulated_per_variable = np.sum(np.abs(coef_vars_lags), axis=1)
            return np.argmax(accumulated_per_variable)

        observations = np.vstack(self.samples).T
        n_variables = len(observations)
        acf_variables = []
        ccf_variable_pairs = {}
        for left in range(n_variables):
            variable_samples = observations[left]
            nlags = min(len(variable_samples) - 1, nlags)
            acf_lags = acf(variable_samples, nlags=nlags)
            acf_variables.append(acf_lags)
            for right in range(left + 1, n_variables):
                ccf_lags = ccf(observations[left], observations[right])
                ccf_variable_pairs[(left, right)] = ccf_lags[: nlags]

        variable_most_autocorr = strongest_correlation_id(acf_variables)
        viz.bar(X=acf_variables[variable_most_autocorr], win='autocorr',
                opts=dict(
                    xlabel='Lag',
                    ylabel='ACF',
                    title=f'Autocorrelation of weight #{variable_most_autocorr}'
                ))

        variable_most_crosscorr = strongest_correlation_id(list(ccf_variable_pairs.values()))
        key_most_crosscorr_pair = list(ccf_variable_pairs.keys())[variable_most_crosscorr]
        viz.bar(X=ccf_variable_pairs[key_most_crosscorr_pair], win='crosscorr',
                opts=dict(
                    xlabel='Lag',
                    ylabel='CCF',
                    title=f'Cross-Correlation of weights {key_most_crosscorr_pair}'
                ))
        return acf_variables[variable_most_autocorr], ccf_variable_pairs[key_most_crosscorr_pair]


class Monitor(object):
    # todo: feature maps

    def __init__(self, trainer):
        """
        :param trainer: _Trainer instance
        """
        self.viz = visdom.Visdom(env=f"{trainer.dataset_name} "
                                     f"{trainer.__class__.__name__} "
                                     f"{time.strftime('%b-%d %H:%M')}")
        self.model = trainer.model
        self.batches_in_epoch = len(trainer.train_loader)
        self.batch_id = 0
        self._registered_params = ParamList()
        self._registered_functions = []
        self.autocorrelation = Autocorrelation()
        self.timer_update = UpdateTimer(max_skip=self.batches_in_epoch // 2)
        self.log_model(self.model)
        self.log_binary_ratio()
        self.log_trainer(trainer)
        print(f"Monitor is opened at http://localhost:8097. Choose environment '{self.viz.env}'.")

    def log_trainer(self, trainer):
        self.log(f"Criterion: {trainer.criterion}")
        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is not None:
            optimizer_str = f"Optimizer {optimizer.__class__.__name__}:"
            for group_id, group in enumerate(optimizer.param_groups):
                optimizer_str += f"\n\tgroup {group_id}: lr={group['lr']}, weight_decay={group['weight_decay']}"
            self.log(optimizer_str)

    def log_binary_ratio(self):
        n_params_full = sum(map(torch.numel, self.model.parameters()))
        n_params_binary = sum(map(torch.numel, parameters_binary(self.model)))
        self.log(f"Parameters binary={n_params_binary:e} / total={n_params_full}"
                 f" = {100. * n_params_binary / n_params_full:.2f} %")

    def log_model(self, model: nn.Module, space='-'):
        self.viz.text("", win='model')
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            self.viz.text(line, win='model', append=True)

    def _draw_line(self, y: Union[float, List[float]], win: str, opts: dict):
        y = np.array([y])
        size = y.shape[-1]
        if size == 0:
            return
        if y.ndim > 1 and size == 1:
            y = y[0]
        epoch_progress = self.batch_id / self.batches_in_epoch
        x = np.full_like(y, epoch_progress)
        self.viz.line(Y=y,
                      X=x,
                      win=win,
                      opts=opts,
                      update='append' if self.viz.win_exists(win) else None)

    def log(self, text: str):
        self.viz.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}", win='log', append=self.viz.win_exists(win='log'))

    def batch_finished(self, outputs: Variable, labels: Variable, loss: Variable):
        self._registered_params.batch_finished()
        if self.timer_update.need_update(self.batch_id):
            self.update_batch_accuracy(batch_accuracy=get_softmax_accuracy(outputs, labels))
            self.update_loss(loss.data[0], mode='batch')
            self.update_distribution()
            self.update_gradient_mean_std()
            self._draw_line(y=self._registered_params.get_sign_flips(), win='sign', opts=dict(
                xlabel='Epoch',
                ylabel='Sign flips, %',
                title="Sign flips after optimizer.step()",
            ))
            for func_id, (func, opts) in enumerate(self._registered_functions):
                self._draw_line(y=func(), win=f"func_{func_id}", opts=opts)
        self.batch_id += 1

    def update_batch_accuracy(self, batch_accuracy: float):
        self._draw_line(batch_accuracy, win='batch_accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title='Train batch accuracy',
        ))

    def update_loss(self, loss: float, mode='batch'):
        self._draw_line(loss, win=f'{mode} loss', opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'{mode} loss'
        ))

    def register_param(self, name: str, param: nn.Parameter):
        self._registered_params.append(ParamRecord(name, param))

    def register_func(self, func: Callable, opts: dict = None):
        self._registered_functions.append((func, opts))

    def update_distribution(self):
        for param_record in self._registered_params:
            name, param = param_record.name, param_record.param
            if param.numel() == 1:
                self._draw_line(y=param.data[0], win=name, opts=dict(
                    xlabel='Epoch',
                    ylabel='Value',
                    title=name,
                ))
            else:
                self.viz.histogram(X=param.data.view(-1), win=name, opts=dict(
                    xlabel='Param norm',
                    ylabel='# bins (distribution)',
                    title=name,
                ))

    def update_gradient_mean_std(self):
        for param_record in self._registered_params:
            name, param = param_record.name, param_record.param
            if param.grad is None:
                continue
            param_record.variance.update(param.grad.data)
            mean, std = param_record.variance.get_mean_std()
            param_norm = param.data.norm(p=2)
            mean = mean.abs().mean() / param_norm
            std = std.mean() / param_norm
            self._draw_line(y=[mean, std], win=f"grad_mean_std_{name}", opts=dict(
                xlabel='Epoch',
                ylabel='Normalized mean and STD',
                title=name,
                legend=['mean', 'std'],
                xtype='log',
                ytype='log',
            ))

    def update_train_accuracy(self, accuracy: float, is_best=False):
        self._draw_line(accuracy, win='train_accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title='Train full dataset accuracy',
            markers=True,
        ))
        if is_best:
            epoch = self.batch_id // self.batches_in_epoch
            self.log(f"Epoch {epoch}. Best train accuracy so far: {accuracy:.4f}")

    def epoch_finished(self, epoch: int = None):
        if epoch is None:
            epoch = self.batch_id // self.batches_in_epoch
        if (epoch + 1) % 5 == 0:
            self.autocorrelation.plot_acf_ccf(self.viz, nlags=self.batches_in_epoch)
