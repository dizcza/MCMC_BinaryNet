import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score
from torch.autograd import Variable

from monitor.batch_timer import BatchTimer
from monitor.mutual_info.kraskov_knn import get_mi as mutual_info_score_knn
from monitor.viz import VisdomMighty


class MutualInfo(ABC):

    log2e = math.log2(math.e)

    def __init__(self, viz: VisdomMighty, timer: BatchTimer, epoch_update=10, compression_range=(0.05, 0.95)):
        """
        :param viz: Visdom logger
        :param timer: BatchTimer to schedule updates
        :param epoch_update: timer epoch step
        :param compression_range: min & max acceptable quantization compression range
        """
        self.viz = viz
        self.timer = timer
        self.epoch_update = epoch_update
        self.compression_range = compression_range
        self.n_bins = {}
        self.n_bins_default = 20  # will be adjusted
        self.max_trials_adjust = 10
        self.layers = {}
        self.input_layer_name = None
        self.activations = {
            'input': [],
            'target': [],
        }
        self.information = None
        self.is_active = False

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
        self.activations['target'] = targets.data.cpu()
        for name, (layer, forward_orig) in self.layers.items():
            layer.forward = forward_orig
        self.is_active = False
        self.save_mutual_info()

    def wrap_forward(self, layer_name, forward_orig):
        def forward_and_save(input):
            assert self.is_active, 'Did you forget to start the job?'
            if len(self.activations['input']) == 0:
                self.input_layer_name = layer_name
            if layer_name == self.input_layer_name:
                self.save_activations(layer_name='input', tensor=input)
            output = forward_orig(input)
            self.save_activations(layer_name, output)
            return output
        return forward_and_save

    def save_activations(self, layer_name: str, tensor: torch.autograd.Variable):
        self.activations[layer_name].append(tensor.data.cpu().clone())

    def reset(self):
        for name in self.activations:
            self.activations[name] = []
        self.information = None

    @abstractmethod
    def preprocess_activations(self):
        raise NotImplementedError()

    def hidden_layer_names(self):
        hidden_names = []
        for name in self.activations:
            if name in ('input', 'target'):
                continue
            hidden_names.append(name)
        return hidden_names

    def compute_mutual_info(self, x, y) -> float:
        return mutual_info_score(x, y) * self.log2e

    def save_mutual_info(self):
        if len(self.activations['input']) == 0:
            return None
        self.preprocess_activations()
        self.information = {}
        for hname in self.hidden_layer_names():
            info_x = self.compute_mutual_info(self.activations['input'], self.activations[hname])
            info_y = self.compute_mutual_info(self.activations['target'], self.activations[hname])
            self.information[hname] = (info_x, info_y)

    def plot(self):
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
        if len(ys) > 1:
            ys = [ys]
            xs = [xs]
        self.viz.line(Y=np.array(ys), X=np.array(xs), win=title, opts=dict(
            xlabel='I(X, T), bits',
            ylabel='I(T, Y), bits',
            title=title,
            legend=legend,
        ), update='append' if self.viz.win_exists(title) else None)
        self.reset()


class MutualInfoBin(MutualInfo):

    def preprocess_activations(self):
        for layer_name, activations in self.activations.items():
            if layer_name == 'target':
                assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
                self.activations[layer_name] = activations.numpy()
            else:
                activations = torch.cat(activations, dim=0)
                activations = activations.view(activations.shape[0], -1)
                if layer_name not in self.n_bins:
                    self.n_bins[layer_name] = self.adjust_bins(activations)
                    self.viz.log(f"[{self.__class__.__name__}] {layer_name}: set n_bins={self.n_bins[layer_name]}")
                digitized = self.digitize(activations, n_bins=self.n_bins[layer_name])
                unique, inverse = np.unique(digitized, return_inverse=True, axis=0)
                self.activations[layer_name] = inverse

    def adjust_bins(self, activations: torch.FloatTensor) -> int:
        n_bins = self.n_bins_default
        compression_min, compression_max = self.compression_range
        for trial in range(self.max_trials_adjust):
            digitized = self.digitize(activations, n_bins)
            unique = np.unique(digitized, axis=0)
            compression = (len(activations) - len(unique)) / len(activations)
            if compression > compression_max:
                n_bins *= 2
            elif compression < compression_min:
                n_bins = max(2, int(n_bins / 2))
                if n_bins == 2:
                    break
            else:
                break
        return n_bins

    @abstractmethod
    def digitize(self, activations: torch.FloatTensor, n_bins: int):
        raise NotImplementedError()


class MutualInfoBinFixed(MutualInfoBin):

    def digitize(self, activations: torch.FloatTensor, n_bins: int) -> torch.FloatTensor:
        mean = activations.mean(dim=0)
        sig = activations.std(dim=0)
        dim0_min, dim0_max = mean - 2 * sig, mean + 2 * sig
        digitized = n_bins * (activations - dim0_min) / (dim0_max - dim0_min)
        digitized.clamp_(min=0, max=n_bins)
        digitized = digitized.type(torch.LongTensor)
        return digitized


class MutualInfoBinFixedFlat(MutualInfoBinFixed):

    def digitize(self, activations: torch.FloatTensor, n_bins: int) -> torch.FloatTensor:
        shape = activations.shape
        return super().digitize(activations.view(-1), n_bins).view(shape)


class MutualInfoQuantile(MutualInfoBin):

    def digitize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        activations = activations.cpu()
        bins = np.percentile(activations,
                             q=np.linspace(start=0, stop=100, num=n_bins, endpoint=True),
                             axis=0)
        digitized = np.empty_like(activations, dtype=np.int32)
        for dim in range(activations.shape[1]):
            digitized[:, dim] = np.digitize(activations[:, dim], bins[:, dim], right=True)
        return digitized


class MutualInfoKNN(MutualInfo):

    def preprocess_activations(self):
        for layer_name, activations in self.activations.items():
            if layer_name == 'target':
                assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
                activations.unsqueeze_(dim=1)
            else:
                activations = torch.cat(activations, dim=0)
                activations = activations.view(activations.shape[0], -1)
            self.activations[layer_name] = activations.numpy()

    def compute_mutual_info(self, x, y) -> float:
        return mutual_info_score_knn(x, y, k=3, estimator='ksg')
