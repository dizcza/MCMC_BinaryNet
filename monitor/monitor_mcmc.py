import torch.nn as nn
from mighty.monitor.accuracy import Accuracy
from mighty.monitor.monitor import MonitorEmbedding

from monitor.autocorrelation import Autocorrelation
from monitor.graph import GraphMCMC
from utils.binary_param import named_parameters_binary


class MonitorMCMC(MonitorEmbedding):

    def __init__(self, model: nn.Module, accuracy_measure: Accuracy, mutual_info=None, normalize_inverse=None):
        super().__init__(accuracy_measure=accuracy_measure,
                         mutual_info=mutual_info,
                         normalize_inverse=normalize_inverse)
        self.autocorrelation = Autocorrelation(n_lags=self.timer.batches_in_epoch)
        named_param_shapes = iter((name, param.shape) for name, param in named_parameters_binary(model))
        self.graph_mcmc = GraphMCMC(named_param_shapes=named_param_shapes, timer=self.timer,
                                    history_heatmap=True)

    def log_self(self):
        super().log_self()
        self.log(repr(self.autocorrelation))

    def mcmc_step(self, param_flips):
        self.autocorrelation.add_samples(param_flips)
        self.graph_mcmc.add_samples(param_flips)

    def epoch_finished(self):
        super().epoch_finished()
        self.autocorrelation.plot(self.viz)
        # self.graph_mcmc.render(self.viz)
