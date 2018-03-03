import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mutual_info_score

from monitor.batch_timer import BatchTimer
from monitor.mutual_info.kraskov_knn import get_mi as mutual_info_score_knn
from monitor.viz import VisdomMighty


class MutualInfo(ABC):

    log2e = math.log2(math.e)

    def __init__(self, viz: VisdomMighty, timer: BatchTimer, estimate_size: int = np.inf,
                 compression_range=(0.05, 0.95), epoch_update=10):
        """
        :param viz: Visdom logger
        :param timer: BatchTimer to schedule updates
        :param estimate_size: number of samples to estimate MI from
        :param compression_range: min & max acceptable quantization compression range
        :param epoch_update: timer epoch step
        """
        self.viz = viz
        self.timer = timer
        self.estimate_size = estimate_size
        self.compression_range = compression_range
        self.epoch_update = epoch_update
        self.n_bins = {}
        self.n_bins_default = 20  # will be adjusted
        self.max_trials_adjust = 10
        self.layers = {}
        self.activations = defaultdict(list)
        self.information = {}
        self.is_active = False

    def register(self, layer: nn.Module, name: str):
        self.layers[name] = (layer, layer.forward)  # immutable

    def decorate_evaluation(self, func: Callable):
        def wrapped(*args, **kwargs):
            self.start_listening()
            res = func(*args, **kwargs)
            self.finish_listening()
            return res
        print(f"Decorated '{func.__name__}' function to save layers' activations for MI estimation")
        return wrapped

    def capture_input_output(self, loader: torch.utils.data.DataLoader):
        inputs = []
        targets = []
        for images, labels in iter(loader):
            inputs.append(images)
            targets.append(labels)
            if len(inputs) * loader.batch_size >= self.estimate_size:
                break
        self.activations['input'] = self.process(layer_name='input', activations=inputs)
        self.activations['target'] = self.process(layer_name='target', activations=targets)

    def start_listening(self):
        if not self.timer.need_epoch_update(self.epoch_update):
            return
        for name, (layer, forward_orig) in self.layers.items():
            if layer.forward == forward_orig:
                layer.forward = self.wrap_forward(layer_name=name, forward_orig=forward_orig)
        self.is_active = True

    def finish_listening(self):
        for name, (layer, forward_orig) in self.layers.items():
            layer.forward = forward_orig
        self.is_active = False
        self.save_information()

    def wrap_forward(self, layer_name, forward_orig):
        def forward_and_save(input):
            assert self.is_active, 'Did you forget to start the job?'
            output = forward_orig(input)
            self.save_activations(layer_name, output)
            return output
        return forward_and_save

    def save_activations(self, layer_name: str, tensor: torch.autograd.Variable):
        if sum(map(len, self.activations[layer_name])) < self.estimate_size:
            self.activations[layer_name].append(tensor.data.cpu().clone())

    @abstractmethod
    def process(self, layer_name: str, activations):
        activations = torch.cat(activations, dim=0)
        size = min(len(activations), self.estimate_size)
        activations = activations[: size]
        return activations

    def hidden_layer_names(self):
        return [name for name in self.activations if name not in ('input', 'target')]

    def compute_mutual_info(self, x, y) -> float:
        return mutual_info_score(x, y) * self.log2e

    def save_information(self):
        for hname in self.hidden_layer_names():
            if hname not in self.activations:
                continue
            hidden_activations = self.process(hname, self.activations[hname])
            del self.activations[hname]
            info_x = self.compute_mutual_info(self.activations['input'], hidden_activations)
            info_y = self.compute_mutual_info(self.activations['target'], hidden_activations)
            self.information[hname] = (info_x, info_y)

    def plot(self):
        assert not self.is_active, "Wait, not finished yet."
        if len(self.information) == 0:
            return
        legend = []
        ys = []
        xs = []
        for layer_name, (info_x, info_y) in list(self.information.items()):
            ys.append(info_y)
            xs.append(info_x)
            legend.append(layer_name)
            del self.information[layer_name]
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


class MutualInfoBin(MutualInfo):

    def process(self, layer_name: str, activations) -> np.ndarray:
        activations = super().process(layer_name, activations)
        if layer_name == 'target':
            assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
            activations = activations.numpy()
        else:
            activations = activations.view(activations.shape[0], -1)
            if layer_name not in self.n_bins:
                self.n_bins[layer_name] = self.adjust_bins(activations)
                self.viz.log(f"[{self.__class__.__name__}] {layer_name}: set n_bins={self.n_bins[layer_name]}")
            digitized = self.digitize(activations, n_bins=self.n_bins[layer_name])
            unique, inverse = np.unique(digitized, return_inverse=True, axis=0)
            activations = inverse
        return activations

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

    def process(self, layer_name: str, activations) -> np.ndarray:
        activations = super().process(layer_name, activations)
        if layer_name == 'target':
            assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
            activations.unsqueeze_(dim=1)
        else:
            activations = activations.view(activations.shape[0], -1)
        return activations.numpy()

    def compute_mutual_info(self, x, y) -> float:
        return mutual_info_score_knn(x, y, k=3, estimator='ksg')
