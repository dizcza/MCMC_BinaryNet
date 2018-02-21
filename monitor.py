import time
from typing import Union, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import visdom
from numpy.core.defchararray import zfill
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import acf, ccf
from torch.autograd import Variable

from utils import get_data_loader, parameters_binary


def argmax_accuracy(outputs, labels) -> float:
    _, labels_predicted = torch.max(outputs.data, 1)
    accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return accuracy


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
    accuracy = argmax_accuracy(outputs, labels)
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


class BatchTimer(object):

    def __init__(self, batches_in_epoch: int):
        self.batches_in_epoch = batches_in_epoch
        self.batch_id = 0
        self.max_skip = batches_in_epoch
        self.next_update = 10

    def need_update(self):
        if self.batch_id >= self.next_update:
            self.next_update = min(int((self.batch_id + 1) ** 1.1), self.batch_id + self.max_skip)
            return True
        return False

    def need_epoch_update(self, epoch_update):
        return (int(self.epoch_progress()) + 1) % epoch_update == 0

    def epoch_progress(self):
        return self.batch_id / self.batches_in_epoch

    def tick(self):
        self.batch_id += 1


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
    def __init__(self, timer: BatchTimer, epoch_update=5):
        self.timer = timer
        self.samples = []
        self.epoch_update = epoch_update

    def add_sample(self, new_sample):
        self.samples.append(new_sample)

    def plot(self, viz: visdom.Visdom):
        if len(self.samples) == 0:
            return

        if not self.timer.need_epoch_update(self.epoch_update):
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
            nlags = min(len(variable_samples) - 1, self.timer.batches_in_epoch)
            acf_lags = acf(variable_samples, nlags=nlags)
            acf_variables.append(acf_lags)
            for right in range(left + 1, n_variables):
                ccf_lags = ccf(observations[left], observations[right])
                ccf_variable_pairs[(left, right)] = ccf_lags[: nlags]

        variable_most_autocorr = strongest_correlation_id(acf_variables)
        viz.bar(X=acf_variables[variable_most_autocorr], win='autocorr', opts=dict(
            xlabel='Lag',
            ylabel='ACF',
            title=f'Autocorrelation of weight #{variable_most_autocorr}'
        ))

        variable_most_crosscorr = strongest_correlation_id(list(ccf_variable_pairs.values()))
        key_most_crosscorr_pair = list(ccf_variable_pairs.keys())[variable_most_crosscorr]
        viz.bar(X=ccf_variable_pairs[key_most_crosscorr_pair], win='crosscorr', opts=dict(
            xlabel='Lag',
            ylabel='CCF',
            title=f'Cross-Correlation of weights {key_most_crosscorr_pair}'
        ))
        return acf_variables[variable_most_autocorr], ccf_variable_pairs[key_most_crosscorr_pair]


class MutualInfo(object):
    def __init__(self, timer: BatchTimer, quantize=10, epoch_update=1):
        """
        :param timer: timer to schedule updates
        :param quantize: #bins to split the activations interval into
        :param epoch_update: timer epoch step
        """
        self.timer = timer
        self.quantize = quantize
        self.layers = {}
        self.input_layer_name = None
        self.activations = {
            'input': [],
            'target': [],
        }
        self.information = None
        self.is_active = False
        self.epoch_update = epoch_update

    def register(self, layer: nn.Module, name: str):
        self.layers[name] = (layer, layer.forward)  # immutable
        self.activations[name] = []

    def start(self):
        if not self.timer.need_epoch_update(self.epoch_update):
            return
        for name, (layer, forward_orig) in self.layers.items():
            if layer.forward == forward_orig:
                layer.forward = self.wrap_forward(layer_name=name, forward_orig=forward_orig)
        self.is_active = True

    def finish(self, targets: Variable):
        if not self.is_active:
            return
        self.activations['target'] = targets.data
        for name, (layer, forward_orig) in self.layers.items():
            layer.forward = forward_orig
        self.is_active = False
        self.estimate_mutual_info()

    def wrap_forward(self, layer_name, forward_orig):
        def forward_and_save(input):
            assert self.is_active, 'Did you forget to start the job?'
            if len(self.activations['input']) == 0:
                self.input_layer_name = layer_name
            if layer_name == self.input_layer_name:
                self.save_activations(layer_name='input', tensor_variable=input)
            output = forward_orig(input)
            self.save_activations(layer_name, output)
            return output
        return forward_and_save

    def save_activations(self, layer_name, tensor_variable):
        self.activations[layer_name].append(tensor_variable.data.clone())

    def reset(self):
        for name in self.activations:
            self.activations[name] = []
        self.information = None

    def quantize_activations(self):
        for layer_name, activations in self.activations.items():
            if layer_name == 'target':
                assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
                quantized = activations.numpy()
            else:
                activations = torch.cat(activations, dim=0)
                activations = activations.view(activations.shape[0], -1)
                bins = np.linspace(start=activations.min(), stop=activations.max(), num=self.quantize, endpoint=True)
                quantized = np.digitize(activations.numpy(), bins, right=True)
            largest = quantized.max()
            largest_width = len(str(largest))
            quantized = zfill(quantized.astype(str), width=largest_width)
            patterns = map(''.join, quantized)
            self.activations[layer_name] = list(patterns)

    def estimate_mutual_info(self):
        if len(self.activations['input']) == 0:
            return None
        self.quantize_activations()
        self.information = {}
        hidden_layer_names = set(self.activations.keys()).difference({'input', 'target'})
        for hname in hidden_layer_names:
            info_x = mutual_info_score(self.activations['input'], self.activations[hname])
            info_y = mutual_info_score(self.activations['target'], self.activations[hname])
            self.information[hname] = (info_x, info_y)

    def plot(self, viz: visdom.Visdom):
        if self.information is None:
            return
        legend = []
        ys = []
        xs = []
        for layer_name, (info_x, info_y) in self.information.items():
            ys.append(info_y)
            xs.append(info_x)
            legend.append(layer_name)
        title = 'Mutual information plane'
        viz.line(Y=np.array([ys]), X=np.array([xs]), win=title, opts=dict(
                     xlabel='I(X, T)',
                     ylabel='I(T, Y)',
                     title=title,
                     legend=legend,
                 ),
                 update='append' if viz.win_exists(title) else None)
        self.reset()


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
        self.timer = BatchTimer(batches_in_epoch=len(trainer.train_loader))
        self.params = ParamList()
        self.functions = []
        self.autocorrelation = Autocorrelation(self.timer)
        self.mutual_info = MutualInfo(self.timer)
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
        x = np.full_like(y, self.timer.epoch_progress())
        self.viz.line(Y=y,
                      X=x,
                      win=win,
                      opts=opts,
                      update='append' if self.viz.win_exists(win) else None)

    def log(self, text: str):
        self.viz.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}", win='log', append=self.viz.win_exists(win='log'))

    def batch_finished(self, outputs: Variable, labels: Variable, loss: Variable):
        self.params.batch_finished()
        if self.timer.need_update():
            self.update_batch_accuracy(batch_accuracy=argmax_accuracy(outputs, labels))
            self.update_loss(loss.data[0], mode='batch')
            self.update_distribution()
            self.update_gradient_mean_std()
            self._draw_line(y=self.params.get_sign_flips(), win='sign', opts=dict(
                xlabel='Epoch',
                ylabel='Sign flips, %',
                title="Sign flips after optimizer.step()",
            ))
            for func_id, (func, opts) in enumerate(self.functions):
                self._draw_line(y=func(), win=f"func_{func_id}", opts=opts)
        self.timer.tick()

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

    def register_func(self, func: Callable, opts: dict = None):
        self.functions.append((func, opts))

    def update_distribution(self):
        for param_record in self.params:
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
        for param_record in self.params:
            name, param = param_record.name, param_record.param
            if param.grad is None:
                continue
            param_record.variance.update(param.grad.data)
            mean, std = param_record.variance.get_mean_std()
            param_norm = param.data.norm(p=2)
            mean = mean.norm(p=2) / param_norm
            std = std.mean() / param_norm
            self._draw_line(y=[mean, std], win=f"grad_mean_std_{name}", opts=dict(
                xlabel='Epoch',
                ylabel='Normalized Mean and STD',
                title=name,
                legend=['||Mean(∇Wi)||', 'STD(∇Wi)'],
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
            epoch = int(self.timer.epoch_progress())
            self.log(f"Epoch {epoch}. Best train accuracy so far: {accuracy:.4f}")

    def epoch_finished(self):
        self.autocorrelation.plot(self.viz)
        self.mutual_info.plot(self.viz)

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            self.params.append(ParamRecord(name, param))
