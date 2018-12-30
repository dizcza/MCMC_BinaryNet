from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import visdom
from statsmodels.tsa.stattools import acf, ccf

from monitor.batch_timer import ScheduleStep
from utils.common import clone_cpu


class Autocorrelation(object):
    def __init__(self, n_lags: int = 40, with_autocorrelation=True, with_cross_correlation=False):
        """
        Auto- & cross-correlation for the flipped weight connections that have been chosen by the TrainerMCMC.
        Cross-correlation is calculated for lag 0 only.
        Estimation is based on the `n_lags` latest samples.
        """
        self.n_lags = n_lags
        self.with_autocorrelation = with_autocorrelation
        self.with_cross_correlation = with_cross_correlation
        self.n_observations = max(100, 5 * self.n_lags)  # num of subsequent parameter snapshots
        self.ccf_lags = {}  # mean Cross-Correlation
        self.calls = 0
        self.calls_ccf = {}
        self.param_history = defaultdict(list)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_lags={self.n_lags})"

    def add_samples(self, param_flips: Iterable):
        """
        :param param_flips: Iterable of ParameterFLip
        """
        if self.with_autocorrelation or self.with_cross_correlation:
            for pflip in param_flips:
                if self.with_cross_correlation and pflip.name not in self.ccf_lags:
                    n_elements = pflip.param.numel()
                    self.ccf_lags[pflip.name] = np.zeros((n_elements, n_elements, self.n_lags), dtype=np.float32)
                    self.calls_ccf[pflip.name] = np.zeros((n_elements, n_elements), dtype=np.int32)
                observations = self.param_history[pflip.name]
                if len(observations) < self.n_observations:
                    param_clone = clone_cpu(pflip.param.data)
                    observations.append(param_clone)

    @ScheduleStep(epoch_step=1)
    def plot(self, viz: visdom.Visdom):
        self.calls += 1

        for name, observations in self.param_history.items():
            if len(observations) < self.n_lags:
                continue
            acf_weights = {}
            observations = torch.stack(observations, dim=0).flatten(start_dim=1)
            observations.t_()
            observations = observations.numpy()
            active_rows_mask = list(map(np.any, np.diff(observations, axis=1)))
            active_rows = np.where(active_rows_mask)[0]
            for i, active_row in enumerate(active_rows):
                if self.with_autocorrelation:
                    acf_lags = acf(observations[active_row], unbiased=False, nlags=self.n_lags, fft=True,
                                   missing='raise')
                    acf_weights[active_row] = acf_lags
                if self.with_cross_correlation:
                    for paired_row in active_rows[i + 1:]:
                        ccf_lags = ccf(observations[active_row], observations[paired_row], unbiased=False)
                        ccf_lags = ccf_lags[: self.n_lags]
                        ccf_old_mean = self.ccf_lags[name][active_row, paired_row]
                        self.calls_ccf[name][active_row, paired_row] += 1
                        self.ccf_lags[name][active_row, paired_row] += (ccf_lags - ccf_old_mean) / self.calls_ccf[name][
                            active_row, paired_row]

            if len(acf_weights) > 0:
                acf_mean = np.mean(list(acf_weights.values()), axis=0)
                viz.bar(X=acf_mean, win=f"ACF {name}", opts=dict(
                    xlabel='Lag',
                    ylabel='ACF',
                    title=f"Mean Autocorrelation: {name}",
                ))

            if len(self.ccf_lags) > 0:
                for lag in range(0, self.n_lags, 5):
                    viz.surf(self.ccf_lags[name][::, lag], win=f'CCF {name} lag {lag}', opts=dict(
                        xlabel='neuron id',
                        ylabel='neuron id',
                        title=f'Mean CCF {name} Lag {lag}',
                        colormap='Hot',
                    ))

        self.param_history.clear()
