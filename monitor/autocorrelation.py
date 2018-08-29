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
        Estimation is based on the `n_lags` latest samples.
        """
        with_cross_correlation &= with_autocorrelation
        self.n_lags = n_lags
        self.with_autocorrelation = with_autocorrelation
        self.with_cross_correlation = with_cross_correlation
        deque_length = max(100, 5 * self.n_lags)
        self.ccf_variable_pairs = defaultdict(float)
        self.calls = 0
        self.samples = defaultdict(lambda: deque(maxlen=deque_length))

    def add_samples(self, param_flips: Iterable):
        """
        :param param_flips: Iterable of ParameterFLip
        """
        if self.with_autocorrelation:
            for pflip in param_flips:
                self.samples[pflip.name].append(pflip.param.data.cpu().view(-1))

    @Schedule(epoch_update=1)
    def plot(self, viz: visdom.Visdom, ccf_top_n=10000):
        self.calls += 1

        def strongest_correlation(coef_vars_lags: dict):
            values = list(coef_vars_lags.values())
            keys = list(coef_vars_lags.keys())
            accumulated_per_variable = np.sum(np.abs(values), axis=1)
            strongest_id = np.argmax(accumulated_per_variable)
            return keys[strongest_id], values[strongest_id]

        acf_variables = {}
        # ccf_variable_pairs = {}
        for name, samples in self.samples.items():
            if len(samples) < self.n_lags + 1:
                continue
            observations = torch.stack(samples, dim=0)
            observations.t_()
            observations = observations.numpy()
            active_rows_mask = list(map(np.any, np.diff(observations, axis=1)))
            active_rows = np.where(active_rows_mask)[0]
            ccf_normalizer = np.count_nonzero(observations, axis=1).max()
            for i, active_row in enumerate(active_rows):
                acf_lags = acf(observations[active_row], unbiased=False, nlags=self.n_lags, fft=True, missing='raise')
                acf_variables[f'{name}.{active_row}'] = acf_lags
                if self.with_cross_correlation:
                    for paired_row in active_rows[i+1:]:
                        # ccf_lags = ccf(observations[active_row], observations[paired_row], unbiased=False)
                        ccf_lags0 = np.sum(observations[active_row] * observations[paired_row])
                        ccf_lags0 /= ccf_normalizer
                        self.ccf_variable_pairs[(active_row, paired_row)] += (ccf_lags0 -
                            self.ccf_variable_pairs[(active_row, paired_row)]) / self.calls

        if len(acf_variables) > 0:
            acf_mean = np.mean(list(acf_variables.values()), axis=0)
            viz.bar(X=acf_mean, win='autocorr', opts=dict(
                xlabel='Lag',
                ylabel='ACF',
                title=f'mean Autocorrelation'
            ))

        if len(self.ccf_variable_pairs) > 0:
            # shortest_length = min(map(len, ccf_variable_pairs.values()))
            # ccf_top_n = min(ccf_top_n, shortest_length)
            # for key, ccf_lags in ccf_variable_pairs.items():
            #     ccf_variable_pairs[key] = ccf_lags[: shortest_length]
            # ccf_lag0_dict = {key: ccf_lags[0] for key, ccf_lags in ccf_variable_pairs.items()}
            # ccf_lag0_vals = np.fromiter(ccf_lag0_dict.values(), dtype=np.float32)
            # argsort = np.argsort(ccf_lag0_vals)[-ccf_top_n:][::-1]
            # ccf_lag0_keys = np.take(list(ccf_lag0_dict.keys()), argsort).tolist()
            # x = ccf_lag0_vals[argsort]
            # x = list(self.ccf_variable_pairs.values())
            # x = np.sort(x)[::-1][:ccf_top_n]
            x = list(self.ccf_variable_pairs.values())
            viz.bar(X=x, win='crosscorr', opts=dict(
                xlabel='neuron pairs (i, j)',
                ylabel='CCF',
                ytickmin=-1.,
                ytickmax=1.,
                title=f'Mean Cross-Correlation'
            ))
