import time
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import visdom
from torch.autograd import Variable

from constants import MODELS_DIR
from utils import get_data_loader, parameters_binary, named_parameters_binary, find_param_by_name


def get_softmax_accuracy(outputs, labels):
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


def calc_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader):
    if model is None:
        return 0.0
    correct_count = 0
    total_count = len(loader.dataset)
    if loader.drop_last:
        # drop the last incomplete batch
        total_count -= len(loader.dataset) % loader.batch_size
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    for batch_id, (images, labels) in enumerate(iter(loader)):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(Variable(images, volatile=True))
        _, labels_predicted = torch.max(outputs.data, 1)
        correct_count += torch.sum(labels_predicted == labels)
    model.train(mode_saved)
    return correct_count / total_count


def test(train=False):
    print(f"{'train' if train else 'test'} accuracy:")
    for dataset_path in MODELS_DIR.iterdir():
        if not dataset_path.is_dir():
            continue
        print(f"\t{dataset_path.name}:")
        for model_path in dataset_path.iterdir():
            test_loader = get_data_loader(dataset=dataset_path.name, train=train)
            try:
                model = torch.load(model_path)
                accur = calc_accuracy(model, test_loader)
                print(f"\t\t{model}: {accur:.4f}")
            except Exception:
                print(f"Skipped evaluating {model_path} model")


class Metrics(object):
    # todo: plot gradients, feature maps

    def __init__(self, model: nn.Module, dataset_name: str, batches_in_epoch: int):
        """
        :param model: network to monitor
        :param dataset_name: 'MNIST', 'CIFAR10', etc.
        :param batches_in_epoch: number of batches in one epoch
        """
        self.viz = visdom.Visdom(env=f"{dataset_name} {time.strftime('%Y-%b-%d %H:%M')}")
        self.model = model
        self.batches_in_epoch = batches_in_epoch
        self._update_step = max(batches_in_epoch // 10, 1)
        self.sign_flips = 0
        self.batch_id = 0
        self.param_sign_before = {}
        self._registered_params = {}
        self.log_model(model)
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
        epoch_progress = self.batch_id / self.batches_in_epoch
        if isinstance(y, list):
            x = np.column_stack([epoch_progress] * len(y))
            y = np.column_stack(y)
        else:
            x = np.array([epoch_progress])
            y = np.array([y])
        self.viz.line(Y=y,
                      X=x,
                      win=win,
                      opts=opts,
                      update='append' if self.viz.win_exists(win) else None)

    def log(self, text: str):
        self.viz.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}", win='log', append=self.viz.win_exists(win='log'))

    def batch_finished(self, outputs: Variable, labels: Variable, loss: Variable):
        self.update_signs()
        if self.batch_id % self._update_step == 0:
            self.update_batch_accuracy(batch_accuracy=get_softmax_accuracy(outputs, labels))
            self.update_loss(loss.data[0])
            self.update_distribution()
            self.update_gradients()
        self.batch_id += 1

    def update_batch_accuracy(self, batch_accuracy: float):
        self._draw_line(batch_accuracy, win='batch_accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title='Train batch accuracy',
        ))

    def update_loss(self, loss: float):
        self._draw_line(loss, win='loss', opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
        ))

    def update_signs(self):
        if len(self._registered_params) == 0:
            return
        registered_count = 0
        for name, param in self._registered_params.items():
            registered_count += param.numel()
            self.sign_flips += torch.sum((param.data * self.param_sign_before[name]) < 0)
            self.param_sign_before[name] = param.data.clone()
        if self.batch_id % self._update_step == 0:
            self.sign_flips /= self._update_step
            self.sign_flips *= 100. / registered_count
            self._draw_line(y=self.sign_flips, win='sign', opts=dict(
                xlabel='Epoch',
                ylabel='Sign flips, %',
                title="Sign flips after optimizer.step()",
            ))
            self.sign_flips = 0

    def register_param(self, param_name: str, param: nn.Parameter = None):
        if param is None:
            param = find_param_by_name(self.model, param_name)
        if param is None:
            raise ValueError(f"Illegal parameter name to register: {param_name}")
        self._registered_params[param_name] = param
        self.param_sign_before[param_name] = param.data.clone()

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
        for name, param in self._registered_params.items():
            if param.grad is None:
                continue
            grad = param.grad.data
            y = grad.norm(p=2)
            legend = ['norm']
            if param.numel() > 1:
                y = [y, grad.min(), grad.max()]
                legend = ['norm', 'min', 'max']
            self._draw_line(y=y, win=f'{name}.grad.norm', opts=dict(
                xlabel='Epoch',
                ylabel='Gradient L2 norm',
                title=name,
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
            self.log(f"Best train accuracy so far: {accuracy:.4f}")
