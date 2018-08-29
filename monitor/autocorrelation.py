from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import torch
import visdom
from statsmodels.tsa.stattools import acf, ccf

from monitor.batch_timer import Schedule


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
        deque_length = max(100, 5 * self.n_lags)
        self.ccf_lag0 = defaultdict(lambda: defaultdict(float))  # track mean of zero-lag cross-correlation
        self.calls = 0
        self.samples = defaultdict(lambda: deque(maxlen=deque_length))

    def add_samples(self, param_flips: Iterable):
        """
        :param param_flips: Iterable of ParameterFLip
        """
        if self.with_autocorrelation or self.with_cross_correlation:
            for pflip in param_flips:
                self.samples[pflip.name].append(pflip.param.data.cpu().view(-1))

    @Schedule(epoch_update=1)
    def plot(self, viz: visdom.Visdom):
        self.calls += 1

        for name, samples in self.samples.items():
            acf_weights = {}
            if len(samples) < self.n_lags:
                continue
            observations = torch.stack(samples, dim=0)
            observations.t_()
            observations = observations.numpy()
            active_rows_mask = list(map(np.any, np.diff(observations, axis=1)))
            active_rows = np.where(active_rows_mask)[0]
            ccf_normalizer = np.count_nonzero(observations, axis=1).max()
            for i, active_row in enumerate(active_rows):
                if self.with_autocorrelation:
                    acf_lags = acf(observations[active_row], unbiased=False, nlags=self.n_lags, fft=True,
                                   missing='raise')
                    acf_weights[active_row] = acf_lags
                if self.with_cross_correlation:
                    for paired_row in active_rows[i+1:]:
                        ccf_lag0 = np.sum(observations[active_row] * observations[paired_row]) / ccf_normalizer
                        ccf_old_mean = self.ccf_lag0[name][(active_row, paired_row)]
                        self.ccf_lag0[name][(active_row, paired_row)] += (ccf_lag0 - ccf_old_mean) / self.calls

            if len(acf_weights) > 0:
                acf_mean = np.mean(list(acf_weights.values()), axis=0)
                acf_title = f"Mean Autocorrelation: {name}"
                viz.bar(X=acf_mean, win=acf_title, opts=dict(
                    xlabel='Lag',
                    ylabel='ACF',
                    title=acf_title,
                ))

            if len(self.ccf_lag0) > 0:
                ccf_vals = list(self.ccf_lag0[name].values())
                viz.bar(X=ccf_vals, win='crosscorr', opts=dict(
                    xlabel='neuron pairs (i, j)',
                    ylabel='CCF',
                    ytickmin=-1.,
                    ytickmax=1.,
                    title=f'Mean Cross-Correlation at Lag 0'
                ))
