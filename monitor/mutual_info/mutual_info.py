import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mutual_info_score
from sklearn import cluster
from torch.autograd import Variable

from monitor.batch_timer import BatchTimer, Schedulable
from monitor.mutual_info.kraskov_knn import get_mi as mutual_info_score_knn
from monitor.viz import VisdomMighty


class MutualInfo(Schedulable):

    log2e = math.log2(math.e)

    def __init__(self, viz: VisdomMighty, estimate_size: int = np.inf, compression_range=(0.50, 0.999), debug=False):
        """
        :param viz: Visdom logger
        :param estimate_size: number of samples to estimate MI from
        :param compression_range: min & max acceptable quantization compression range
        """
        self.viz = viz
        self.viz.log(f"MI estimate_size={estimate_size}")
        self.estimate_size = estimate_size
        self.compression_range = compression_range
        self.debug = debug
        self.n_bins = {}
        self.max_trials_adjust = 10
        self.layers = {}
        self.activations = defaultdict(list)
        self.information = {}
        self.is_active = False
        self.eval_loader = None

    @property
    def n_bins_default(self):
        return 20

    def schedule(self, timer: BatchTimer, epoch_update: int = 1, batch_update: int = 0):
        """
        :param timer: timer to schedule updates
        :param epoch_update: epochs between updates
        :param batch_update: batches between updates (additional to epochs)
        """
        self.start_listening = timer.schedule(self.start_listening, epoch_update=epoch_update,
                                              batch_update=batch_update)
        self.update = timer.schedule(self.update, epoch_update=epoch_update, batch_update=batch_update)
        self.plot_quantized_dispersion = timer.schedule(self.plot_quantized_dispersion, epoch_update=1)

    def register(self, layer: nn.Module, name: str):
        self.layers[name] = (layer, layer.forward)  # immutable

    def update(self, model: nn.Module):
        if self.eval_loader is None:
            # did you forget to call .prepare()?
            return
        self.start_listening()
        use_cuda = torch.cuda.is_available()
        for batch_id, (images, labels) in enumerate(iter(self.eval_loader)):
            if use_cuda:
                images = images.cuda()
            model(Variable(images, volatile=True))
            if batch_id * self.eval_loader.batch_size >= self.estimate_size:
                break
        self.finish_listening()
        self.plot()

    def decorate_evaluation(self, func: Callable):
        def wrapped(*args, **kwargs):
            self.start_listening()
            res = func(*args, **kwargs)
            self.finish_listening()
            return res
        print(f"Decorated '{func.__name__}' function to save layers' activations for MI estimation")
        return wrapped

    def prepare(self, loader: torch.utils.data.DataLoader):
        self.eval_loader = loader
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
        for name, (layer, forward_orig) in self.layers.items():
            if layer.forward == forward_orig:
                layer.forward = self.wrap_forward(layer_name=name, forward_orig=forward_orig)
        self.is_active = True

    def finish_listening(self):
        if not self.is_active:
            return
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
    def process(self, layer_name: str, activations: List[torch.FloatTensor]):
        activations = torch.cat(activations, dim=0)
        size = min(len(activations), self.estimate_size)
        activations = activations[: size]
        return activations

    def hidden_layer_names(self):
        return [name for name in self.activations if name not in ('input', 'target')]

    def compute_mutual_info(self, x, y) -> float:
        return mutual_info_score(x, y) * self.log2e

    def plot_quantized_hist(self, quantized: Dict[str, np.ndarray]):
        for name, layer_quantized in quantized.items():
            _, counts = np.unique(layer_quantized, return_counts=True)
            counts.sort()
            counts = counts[::-1]
            self.viz.bar(Y=np.arange(len(counts), dtype=int), X=counts, win=f'{name} MI hist', opts=dict(
                xlabel='bin ID',
                ylabel='# items',
                title=f'{name} MI quantized histogram',
            ))

    def plot_quantized_dispersion(self, quantized: Dict[str, np.ndarray]):
        if len(set(self.n_bins[name] for name in quantized.keys())) == 1:
            # all layers have the same n_bins
            counts = []
            for name, layer_quantized in quantized.items():
                _, layer_counts = np.unique(layer_quantized, return_counts=True)
                counts.append(layer_counts)
            self.viz.boxplot(X=np.vstack(counts).transpose(), win='MI hist', opts=dict(
                ylabel='# items in one bin',
                title='MI quantized dispersion (smaller is better)',
                legend=list(quantized.keys()),
            ))
        else:
            self.viz.boxplot(X=np.vstack(quantized.values()).transpose(), win='MI hist', opts=dict(
                ylabel='bin ID dispersion',
                title='MI inverse quantized dispersion (smaller is worse)',
                legend=list(quantized.keys()),
            ))
        if self.debug:
            self.plot_quantized_hist(quantized)

    def save_information(self):
        quantized = dict(input=self.activations['input'])
        for hname in self.hidden_layer_names():
            quantized[hname] = self.process(hname, self.activations.pop(hname))
            info_x = self.compute_mutual_info(self.activations['input'], quantized[hname])
            info_y = self.compute_mutual_info(self.activations['target'], quantized[hname])
            self.information[hname] = (info_x, info_y)
        self.plot_quantized_dispersion(quantized)

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

    def process(self, layer_name: str, activations: List[torch.FloatTensor]) -> np.ndarray:
        activations = super().process(layer_name, activations)
        if layer_name == 'target':
            assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
            activations = activations.numpy()
        else:
            activations = activations.view(activations.shape[0], -1)
            if layer_name not in self.n_bins:
                self.n_bins[layer_name] = self.adjust_bins(layer_name, activations)
                self.viz.log(f"[{self.__class__.__name__}] {layer_name}: set n_bins={self.n_bins[layer_name]}")
            activations = self.quantize(layer_name, activations, n_bins=self.n_bins[layer_name])
        return activations

    def adjust_bins(self, layer_name: str, activations: torch.FloatTensor) -> int:
        n_bins = self.n_bins_default
        compression_min, compression_max = self.compression_range
        for trial in range(self.max_trials_adjust):
            digitized = self.digitize(layer_name, activations, n_bins)
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
    def digitize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        pass

    def quantize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        digitized = self.digitize(layer_name, activations, n_bins=n_bins)
        unique, inverse = np.unique(digitized, return_inverse=True, axis=0)
        return inverse


class MutualInfoBinFixed(MutualInfoBin):

    def digitize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        mean = activations.mean(dim=0)
        sig = activations.std(dim=0)
        dim0_min, dim0_max = mean - 2 * sig, mean + 2 * sig
        digitized = n_bins * (activations - dim0_min) / (dim0_max - dim0_min)
        digitized.clamp_(min=0, max=n_bins)
        digitized = digitized.type(torch.LongTensor).numpy()
        return digitized


class MutualInfoBinFixedFlat(MutualInfoBinFixed):

    def digitize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        shape = activations.shape
        return super().digitize(layer_name, activations.view(-1), n_bins).view(shape)


class MutualInfoQuantile(MutualInfoBin):

    def digitize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        activations = activations.cpu()
        bins = np.percentile(activations,
                             q=np.linspace(start=0, stop=100, num=n_bins, endpoint=True),
                             axis=0)
        digitized = np.empty_like(activations, dtype=np.int32)
        for dim in range(activations.shape[1]):
            digitized[:, dim] = np.digitize(activations[:, dim], bins[:, dim], right=True)
        return digitized


class MutualInfoKMeans(MutualInfoBin):

    def __init__(self, viz: VisdomMighty, estimate_size: int = np.inf, compression_range=(0.50, 0.999), debug=False,
                 patience: int = 2):
        """
        :param viz: Visdom logger
        :param estimate_size: number of samples to estimate MI from
        :param compression_range: min & max acceptable quantization compression range
        :param patience: reuse previously fit model for n='patience' iterations
        """
        super().__init__(viz, estimate_size, compression_range, debug)
        self.model = {}
        self.count_iterations = defaultdict(int)
        self.patience = patience

    def digitize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        if layer_name not in self.model or self.model[layer_name].n_clusters != n_bins:
            self.model[layer_name] = cluster.MiniBatchKMeans(n_clusters=n_bins)
            labels = self.model[layer_name].fit_predict(activations)
        else:
            labels = self.model[layer_name].predict(activations)
        return labels

    def quantize(self, layer_name: str, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        self.count_iterations[layer_name] += 1
        if self.count_iterations[layer_name] % self.patience == 0:
            del self.model[layer_name]
        return self.digitize(layer_name, activations, n_bins=n_bins)


class MutualInfoKNN(MutualInfo):

    def process(self, layer_name: str, activations: List[torch.FloatTensor]) -> np.ndarray:
        activations = super().process(layer_name, activations)
        if layer_name == 'target':
            assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
            activations.unsqueeze_(dim=1)
        else:
            activations = activations.view(activations.shape[0], -1)
        return activations.numpy()

    def compute_mutual_info(self, x, y) -> float:
        return mutual_info_score_knn(x, y, k=3, estimator='ksg')
