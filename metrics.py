import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import time
import visdom
import numpy as np

from constants import MODELS_DIR
from utils import get_data_loader, load_model


def get_softmax_accuracy(outputs, labels):
    _, labels_predicted = torch.max(outputs.data, 1)
    softmax_accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return softmax_accuracy


def timer(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print("{}: {:.3f} sec".format(func.__name__, elapsed))
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
    print("{} accuracy:".format("train" if train else "test"))
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        test_loader = get_data_loader(train)
        try:
            model = torch.load(model_path)
            accur = calc_accuracy(model, test_loader)
            print("\t{}: {:.4f}".format(model, accur))
        except Exception:
            print("Skipped evaluating {} model".format(model_path))


class Metrics(object):
    # todo: plot gradients, feature maps

    def __init__(self, model: nn.Module, loader: torch.utils.data.DataLoader):
        dataset_name = loader.dataset.__class__.__name__
        env_dated = "{} {}".format(dataset_name, time.strftime('%Y-%b-%d %H:%M'))
        self.viz = visdom.Visdom(env=env_dated)
        self.model = model
        self.total_binary_params = sum(map(torch.numel, model.parameters_binary()))
        self.batches_in_epoch = len(loader)
        self.update_step = max(len(loader) // 10, 1)
        self.batch_id = 0
        self.param_sign_before = {
            name: param.data.sign() for name, param in model.named_parameters_binary()
        }

    @staticmethod
    def load_best_accuracy(model_name: str, debug: bool) -> float:
        train_loader = get_data_loader(train=True)
        best_accuracy = 0.
        if not debug:
            try:
                loaded_model = load_model(model_name)
                best_accuracy = calc_accuracy(loaded_model, train_loader)
            except Exception:
                print("Couldn't estimate the best accuracy for {}. Rest to 0.".format(model_name))
        return best_accuracy

    def _draw_line(self, y, win: str, opts: dict):
        epoch_progress = self.batch_id / self.batches_in_epoch
        self.viz.line(Y=np.array([y]),
                      X=np.array([epoch_progress]),
                      win=win,
                      opts=opts,
                      update='append' if self.viz.win_exists(win) else None)

    def batch_finished(self, outputs: Variable, labels: Variable, loss: Variable):
        if self.batch_id % self.update_step == 0:
            sign_flips = self.update_signs() * 100. / self.total_binary_params
            self._draw_line(sign_flips, win='sign', opts=dict(
                xlabel='Epoch',
                ylabel='Sign flips, %',
                title="Sign flips after optimizer.step()",
            ))
            batch_accuracy = get_softmax_accuracy(outputs, labels)
            self._draw_line(batch_accuracy, win='batch_accuracy', opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='Train batch accuracy',
            ))
            self._draw_line(loss.data[0], win='loss', opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
            ))
            for name, param in self.model.named_parameters_binary():
                self.viz.histogram(param.data.view(-1), win=name, opts=dict(
                    xlabel='Param norm',
                    ylabel='# bins (distribution)',
                    title=name,
                ))
        self.batch_id += 1

    def update_signs(self) -> int:
        sign_flips = 0
        # for name, param in self.model.named_parameters_binary():
        #     print(name, float(param.grad.min()), float(param.grad.max()), float(param.data.min()), float(param.data.max()))
        for name, param in self.model.named_parameters_binary():
            new_sign = param.data.sign()
            sign_flips += torch.sum((new_sign * self.param_sign_before[name]) < 0)
            self.param_sign_before[name] = new_sign
        return sign_flips

    def log_best_accuracy(self, best_accuracy: float):
        self.viz.text("{} Best train accuracy: {:.4f}".format(time.strftime('%Y-%b-%d %H:%M'), best_accuracy),
                      win='best_accuracy',
                      append=self.viz.win_exists(win='best_accuracy'))

    def update_train_accuracy(self, accuracy: float, is_best=False):
        self._draw_line(accuracy, win='train_accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title='Train full dataset accuracy',
            markers=True,
        ))
        if is_best:
            self.log_best_accuracy(accuracy)
