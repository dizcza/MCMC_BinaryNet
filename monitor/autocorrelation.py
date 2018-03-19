from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import torch
import visdom
from statsmodels.tsa.stattools import acf, ccf

from monitor.batch_timer import BatchTimer, Schedulable


class Autocorrelation(Schedulable):
    def __init__(self, n_lags: int = 40, with_autocorrelation=True, with_cross_correlation=False):
        """
        Auto- & cross-correlation for the flipped weight connections that have been chosen by the TrainerMCMC.
        Estimation is based on the latest ~`n_lags` samples.
        """
        with_cross_correlation &= with_autocorrelation
        self.n_lags = n_lags
        self.with_autocorrelation = with_autocorrelation
        self.with_cross_correlation = with_cross_correlation
        deque_length = max(100, 5 * self.n_lags)
        self.samples = defaultdict(lambda: deque(maxlen=deque_length))

    def add_samples(self, param_flips: Iterable):
        """
        :param param_flips: Iterable of ParameterFLip
        """
        if self.with_autocorrelation:
            for pflip in param_flips:
                self.samples[pflip.name].append(pflip.param.data.cpu().view(-1))

    def schedule(self, timer: BatchTimer, epoch_update: int = 1, batch_update: int = 0):
        """
        :param timer: timer to schedule updates
        :param epoch_update: epochs between updates
        :param batch_update: batches between updates (additional to epochs)
        """
        self.plot = timer.schedule(self.plot, epoch_update=epoch_update, batch_update=batch_update)

    def plot(self, viz: visdom.Visdom):

        def strongest_correlation(coef_vars_lags: dict):
            values = list(coef_vars_lags.values())
            keys = list(coef_vars_lags.keys())
            accumulated_per_variable = np.sum(np.abs(values), axis=1)
            strongest_id = np.argmax(accumulated_per_variable)
            return keys[strongest_id], values[strongest_id]

        acf_variables = {}
        ccf_variable_pairs = {}
        for name, samples in self.samples.items():
            if len(samples) < self.n_lags + 1:
                continue
            observations = torch.stack(samples, dim=0)
            observations.t_()
            observations = observations.numpy()
            active_rows_mask = list(map(np.any, np.diff(observations, axis=1)))
            active_rows = np.where(active_rows_mask)[0]
            for i, active_row in enumerate(active_rows):
                acf_lags = acf(observations[active_row], unbiased=False, nlags=self.n_lags, fft=True, missing='raise')
                acf_variables[f'{name}.{active_row}'] = acf_lags
                if self.with_cross_correlation:
                    for paired_row in active_rows[i+1:]:
                        ccf_lags = ccf(observations[active_row], observations[paired_row], unbiased=False)
                        ccf_variable_pairs[(active_row, paired_row)] = ccf_lags

        if len(acf_variables) > 0:
            acf_mean = np.mean(list(acf_variables.values()), axis=0)
            viz.bar(X=acf_mean, win='autocorr', opts=dict(
                xlabel='Lag',
                ylabel='ACF',
                title=f'mean Autocorrelation'
            ))

        if len(ccf_variable_pairs) > 0:
            shortest_length = min(map(len, ccf_variable_pairs.values()))
            for key, values in ccf_variable_pairs.items():
                ccf_variable_pairs[key] = values[: shortest_length]
            ccf_mean = np.mean(list(ccf_variable_pairs.values()), axis=0)
            viz.bar(X=ccf_mean, win='crosscorr', opts=dict(
                xlabel='Lag',
                ylabel='CCF',
                ytickmin=0.,
                ytickmax=1.,
                title=f'mean Cross-Correlation'
            ))
