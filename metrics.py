import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import visdom
from torch.autograd import Variable

from constants import MODELS_DIR
from utils import get_data_loader, load_model, parameters_binary, named_parameters_binary


def get_softmax_accuracy(outputs, labels):
    _, labels_predicted = torch.max(outputs.data, 1)
    softmax_accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return softmax_accuracy


def timer(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:.3f} sec")
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
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        test_loader = get_data_loader(train)
        try:
            model = torch.load(model_path)
            accur = calc_accuracy(model, test_loader)
            print(f"\t{model}: {accur:.4f}")
        except Exception:
            print(f"Skipped evaluating {model_path} model")


class Metrics(object):
    # todo: plot gradients, feature maps

    def __init__(self, model: nn.Module, loader: torch.utils.data.DataLoader, monitor_sign: str = 'binary'):
        """
        :param model: network to monitor
        :param loader: DataLoader to get its size and name
        :param monitor_sign:
            What layers to monitor for sign changes after optimizer.step()?
            Possible modes:
             * 'all' - all layers
             * 'binary' - binary layers
             * None - don't monitor any layers for sign changes
        """
        dataset_name = loader.dataset.__class__.__name__
        self.viz = visdom.Visdom(env=f"{dataset_name} {time.strftime('%Y-%b-%d %H:%M')}")
        self.model = model
        self.total_binary_params = sum(map(torch.numel, parameters_binary(model)))
        self.batches_in_epoch = len(loader)
        self.update_step = max(len(loader) // 10, 1)
        self.batch_id = 0
        if monitor_sign == 'all':
            named_params = model.named_parameters()
        elif monitor_sign == 'binary':
            named_params = named_parameters_binary(model)
        else:
            named_params = []
        self.param_sign_before = {
            name: param.data.sign() for name, param in named_params
        }
        self.log_model(model)
        n_params_full = sum(map(torch.numel, model.parameters()))
        self.log(f"Parameters total={n_params_full:e}, "
                 f"binary={self.total_binary_params:e} ({100. * self.total_binary_params / n_params_full:.2f} %)")

    def log_model(self, model: nn.Module, space='-'):
        self.viz.text("", win='model')
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            self.viz.text(line, win='model', append=True)

    @staticmethod
    def load_best_accuracy(model_name: str, debug: bool = False) -> float:
        train_loader = get_data_loader(train=True)
        best_accuracy = 0.
        if not debug:
            try:
                loaded_model = load_model(model_name)
                best_accuracy = calc_accuracy(loaded_model, train_loader)
            except Exception:
                print(f"Couldn't estimate the best accuracy for {model_name}. Reset to 0.")
        return best_accuracy

    def _draw_line(self, y, win: str, opts: dict):
        epoch_progress = self.batch_id / self.batches_in_epoch
        self.viz.line(Y=np.array([y]),
                      X=np.array([epoch_progress]),
                      win=win,
                      opts=opts,
                      update='append' if self.viz.win_exists(win) else None)

    def log(self, text: str):
        self.viz.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}", win='log', append=self.viz.win_exists(win='log'))

    def batch_finished(self, outputs: Variable, labels: Variable, loss: Variable):
        if self.batch_id % self.update_step == 0:
            self.update_signs()
            batch_accuracy = get_softmax_accuracy(outputs, labels)
            self.update_batch_accuracy(batch_accuracy)
            self.update_loss(loss.data[0])
            self.update_weight_distribution()
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

    def update_weight_distribution(self):
        for name, param in named_parameters_binary(self.model):
            self.viz.histogram(param.data.view(-1), win=name, opts=dict(
                xlabel='Param norm',
                ylabel='# bins (distribution)',
                title=name,
            ))

    def update_signs(self):
        if len(self.param_sign_before) == 0:
            # don't monitor sign changes
            return
        named_parameters_binary_list = list(named_parameters_binary(self.model))
        names_binary = set()
        sign_flips = 0
        if len(named_parameters_binary_list) > 0:
            for name, param in named_parameters_binary_list:
                names_binary.add(name)
                new_sign = param.data.sign()
                sign_flips += torch.sum((new_sign * self.param_sign_before[name]) < 0)
                self.param_sign_before[name] = new_sign
            self._draw_line(y=sign_flips * 100. / self.total_binary_params, win='sign_binary', opts=dict(
                xlabel='Epoch',
                ylabel='Sign flips, %',
                title="[BINARY] Sign flips after optimizer.step()",
            ))
        for name, param in self.model.named_parameters():
            if name in names_binary:
                # already computed
                continue
            if name not in self.param_sign_before:
                # we set don't monitor all layers signs
                return
            new_sign = param.data.sign()
            sign_flips += torch.sum((new_sign * self.param_sign_before[name]) < 0)
            self.param_sign_before[name] = new_sign
        self._draw_line(y=sign_flips * 100. / self.total_binary_params, win='sign_all', opts=dict(
            xlabel='Epoch',
            ylabel='Sign flips, %',
            title="[ALL LAYERS] Sign flips after optimizer.step()",
        ))

    def update_gradients(self):
        # todo make it real
        for name, param in named_parameters_binary(self.model):
            print(name, float(param.grad.min()), float(param.grad.max()), float(param.data.min()), float(param.data.max()))

    def update_train_accuracy(self, accuracy: float, is_best=False):
        self._draw_line(accuracy, win='train_accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title='Train full dataset accuracy',
            markers=True,
        ))
        if is_best:
            self.log(f"Best train accuracy so far: {accuracy:.4f}")
