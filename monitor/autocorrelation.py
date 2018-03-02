from collections import defaultdict, deque

import numpy as np
import torch
import visdom
from statsmodels.tsa.stattools import acf, ccf

from monitor.batch_timer import BatchTimer


class Autocorrelation(object):
    def __init__(self, timer: BatchTimer, epoch_update=10):
        """
        Auto- & cross-correlation for the flipped weight connections that have been chosen by the TrainerMCMC.
        Estimation is done for `n_batch` lags, based on the latest `5 * n_batch` samples, where `n_batch` - number of
        batches in one epoch.
        :param timer: timer to schedule updates
        :param epoch_update: see timer.need_epoch_update
        """
        self.timer = timer
        deque_length = max(100, 5 * self.timer.batches_in_epoch)
        self.samples = defaultdict(lambda: deque(maxlen=deque_length))
        self.epoch_update = epoch_update

    def add_sample(self, name: str, new_sample: torch.ByteTensor):
        """
        :param name: param name
        :param new_sample: boolean matrix of flipped (1) and remained (0) weights
        """
        self.samples[name].append(new_sample.cpu().view(-1))

    def plot(self, viz: visdom.Visdom):
        if len(self.samples) == 0:
            return

        if not self.timer.need_epoch_update(self.epoch_update):
            return

        def strongest_correlation(coef_vars_lags: dict):
            values = list(coef_vars_lags.values())
            keys = list(coef_vars_lags.keys())
            accumulated_per_variable = np.sum(np.abs(values), axis=1)
            strongest_id = np.argmax(accumulated_per_variable)
            return keys[strongest_id], values[strongest_id]

        acf_variables = {}
        ccf_variable_pairs = {}
        for name, samples in self.samples.items():
            if len(samples) < self.timer.batches_in_epoch + 1:
                continue
            observations = torch.cat(samples, dim=0)
            observations.t_()
            observations = observations.numpy()
            variables_active = np.where([len(np.where(diffs)[0]) > 0 for diffs in np.diff(observations, axis=1)])[0]
            observations = np.take(observations, variables_active, axis=0)
            n_variables = len(observations)
            for true_id, left in zip(variables_active, range(n_variables)):
                acf_lags = acf(observations[left], unbiased=True, nlags=self.timer.batches_in_epoch)
                acf_variables[f'{name}.{true_id}'] = acf_lags
                for right in range(left + 1, n_variables):
                    ccf_lags = ccf(observations[left], observations[right], unbiased=True)
                    ccf_variable_pairs[(left, right)] = ccf_lags

        if len(acf_variables) > 0:
            weight_name, weight_acf = strongest_correlation(acf_variables)
            viz.bar(X=weight_acf, win='autocorr', opts=dict(
                xlabel='Lag',
                ylabel='ACF',
                title=f'Autocorrelation of {weight_name}'
            ))

        if len(ccf_variable_pairs) > 0:
            shortest_length = min(map(len, ccf_variable_pairs.values()))
            for key, values in ccf_variable_pairs.items():
                ccf_variable_pairs[key] = values[: shortest_length]
            pair_ids, pair_ccf = strongest_correlation(ccf_variable_pairs)
            viz.bar(X=pair_ccf, win='crosscorr', opts=dict(
                xlabel='Lag',
                ylabel='CCF',
                ytickmin=0.,
                ytickmax=1.,
                title=f'Cross-Correlation of weights {pair_ids}'
            ))
