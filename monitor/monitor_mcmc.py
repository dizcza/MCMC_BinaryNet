import torch.nn as nn
import torch.utils.data

from monitor.accuracy import Accuracy
from monitor.autocorrelation import Autocorrelation
from monitor.graph import GraphMCMC
from monitor.monitor import Monitor
from utils.binary_param import named_parameters_binary
from utils.datasubset import DataSubset


class MonitorMCMC(Monitor):

    def __init__(self, test_loader: torch.utils.data.DataLoader, accuracy_measure: Accuracy, model: nn.Module):
        super().__init__(test_loader=test_loader, accuracy_measure=accuracy_measure)
        self.autocorrelation = Autocorrelation(n_lags=self.timer.batches_in_epoch,
                                               with_autocorrelation=isinstance(self.test_loader.dataset, DataSubset))
        named_param_shapes = iter((name, param.shape) for name, param in named_parameters_binary(model))
        self.graph_mcmc = GraphMCMC(named_param_shapes=named_param_shapes, timer=self.timer,
                                    history_heatmap=True)

    def mcmc_step(self, param_flips):
        self.autocorrelation.add_samples(param_flips)
        self.graph_mcmc.add_samples(param_flips)

    def epoch_finished(self, model: nn.Module, outputs_full, labels_full):
        super().epoch_finished(model=model, outputs_full=outputs_full, labels_full=labels_full)
        self.autocorrelation.plot(self.viz)
        # self.graph_mcmc.render(self.viz)
