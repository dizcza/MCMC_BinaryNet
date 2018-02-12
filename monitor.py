import time
import math
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
        self.next_update = 1

    def need_update(self, batch_id: int):
        if batch_id >= self.next_update:
            self.next_update = min(int((batch_id + 1) ** 1.1), batch_id + self.max_skip)
            return True
        return False


class SignMonitor(object):
    def __init__(self):
        self.sign_flips = 0
        self.calls = 0
        self.param_sign_before = {}

    def batch_finished(self, named_params: Dict[str, nn.Parameter]):
        self.calls += 1
        for name, param in named_params.items():
            self.sign_flips += torch.sum((param.data * self.param_sign_before[name]) < 0)
            self.param_sign_before[name] = param.data.clone()

    def get_sign_flips(self):
        if len(self.param_sign_before) == 0:
            # haven't registered any param yet
            return None
        param_count = sum(map(torch.numel, self.param_sign_before.values()))
        flips = self.sign_flips
        flips /= param_count  # per param
        flips /= self.calls  # per update
        flips *= 100.  # percents
        self.sign_flips = 0
        self.calls = 0
        return flips


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
        self._registered_params = {}
        self._registered_functions = []
        self.sign_monitor = SignMonitor()
        self.timer_update = UpdateTimer(max_skip=self.batches_in_epoch // 2)
        self.log_model(self.model)
        self.log_binary_ratio()

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
        self.sign_monitor.batch_finished(self._registered_params)
        if self.timer_update.need_update(self.batch_id):
            self.update_batch_accuracy(batch_accuracy=get_softmax_accuracy(outputs, labels))
            self.update_loss(loss.data[0], mode='batch')
            self.update_distribution()
            self.update_gradients()
            self._draw_line(y=self.sign_monitor.get_sign_flips(), win='sign', opts=dict(
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

    def register_param(self, param_name: str, param: nn.Parameter = None):
        if param is None:
            param = find_param_by_name(self.model, param_name)
        if param is None:
            raise ValueError(f"Illegal parameter name to register: {param_name}")
        self._registered_params[param_name] = param
        self.sign_monitor.param_sign_before[param_name] = param.data.clone()

    def register_func(self, func: Callable, opts: dict = None):
        self._registered_functions.append((func, opts))

    def update_distribution(self):
        for name, param in self._registered_params.items():
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

    def update_gradients(self):
        norms = []
        legend = []
        for name, param in self._registered_params.items():
            if param.grad is None:
                continue
            norms.append(param.grad.data.norm(p=2) / math.sqrt(param.numel()))
            legend.append(name)
        self._draw_line(y=norms, win='grad.norm', opts=dict(
            xlabel='Epoch',
            ylabel='Gradient L2 norm / sqrt(n)',
            legend=legend,
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
