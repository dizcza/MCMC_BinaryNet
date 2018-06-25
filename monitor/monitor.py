import time
from typing import Callable
from collections import UserDict

import torch
import torch.nn as nn
import torch.utils.data
import math

from monitor.autocorrelation import Autocorrelation
from monitor.batch_timer import timer, Schedule
from monitor.graph import GraphMCMC
from monitor.mutual_info.mutual_info import MutualInfoKMeans, MutualInfoSign, MutualInfoKNN
from monitor.var_online import VarianceOnline
from monitor.viz import VisdomMighty
from monitor.accuracy import calc_accuracy
from utils import named_parameters_binary, parameters_binary, MNISTSmall, get_data_loader, factors_root


def timer_profile(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        elapsed /= len(args[1])  # fps
        elapsed *= 1e3
        print(f"{func.__name__} {elapsed} ms")
        return res

    return wrapped


class ParamRecord(object):
    def __init__(self, param: nn.Parameter, monitor=False):
        self.param = param
        self.prev_sign = param.data.cpu().clone()  # clone is faster
        self.is_monitored = monitor
        self.variance = VarianceOnline(tensor=param.data.cpu(), is_active=self.is_monitored)
        self.grad_variance = VarianceOnline(is_active=self.is_monitored)
        if self.is_monitored:
            self.initial_data = param.data.clone()
            self.initial_norm = self.initial_data.norm(p=2)
            self.inactive = torch.ByteTensor(self.param.shape).fill_(0)

    def freeze(self, tstat_min: float):
        """
        Freezes insignificant parameters.
        :param tstat_min: t-statistics threshold
        """
        assert self.is_monitored, "Parameter is not monitored!"
        self.inactive |= self.tstat() < tstat_min

    def tstat(self) -> torch.FloatTensor:
        """
        :return: t-statistics of the parameters history
        """
        assert self.is_monitored, "Parameter is not monitored!"
        mean, std = self.variance.get_mean_std()
        tstat = mean.abs() / std
        isnan = std == 0
        if isnan.all():
            tstat.fill_(0)
        else:
            tstat_nonnan = tstat[~isnan]
            tstat_max = tstat_nonnan.mean() + 2 * tstat_nonnan.std()
            tstat_nonnan.clamp_(max=tstat_max)
            tstat[~isnan] = tstat_nonnan
            tstat[isnan] = tstat_max
        return tstat


class ParamsDict(UserDict):
    def __init__(self):
        super().__init__()
        self.sign_flips = 0
        self.n_updates = 0

    def batch_finished(self):
        self.n_updates += 1
        for param_record in self.values():
            param = param_record.param
            new_data = param.data.cpu()
            if new_data is param.data:
                new_data = new_data.clone()
            self.sign_flips += torch.sum((new_data * param_record.prev_sign) < 0)
            param_record.prev_sign = new_data
            param_record.variance.update(new_data)

    def plot_sign_flips(self, viz: VisdomMighty):
        if len(self) == 0:
            # haven't registered any param yet
            return
        param_count = 0
        for param_record in self.values():
            param_count += torch.numel(param_record.param)
        viz.line_update(y=self.sign_flips / self.n_updates, win='sign', opts=dict(
            xlabel='Epoch',
            ylabel='Sign flips',
            title="Sign flips after optimizer.step()",
        ))
        self.sign_flips = 0
        self.n_updates = 0

    def items_monitored(self):
        def pass_monitored(pair):
            name, param_record = pair
            return param_record.is_monitored

        return filter(pass_monitored, self.items())


class Monitor(object):
    # todo: feature maps

    def __init__(self, trainer):
        """
        :param trainer: Trainer instance
        """
        self.watch_parameters = True
        self.timer = timer
        self.timer.init(batches_in_epoch=len(trainer.train_loader))
        self.viz = VisdomMighty(env=f"{time.strftime('%Y-%b-%d')} "
                                    f"{trainer.dataset_name} "
                                    f"{trainer.__class__.__name__}", timer=self.timer)
        self.model = trainer.model
        self.test_loader = get_data_loader(dataset=trainer.dataset_name, train=False)
        self.param_records = ParamsDict()
        self.mutual_info = MutualInfoKMeans(estimate_size=int(1e3), compression_range=(0.5, 0.999))
        self.functions = []
        self.log_model(self.model)
        self.log_binary_ratio()
        self.log_trainer(trainer)

    def log_trainer(self, trainer):
        self.log(f"Criterion: {trainer.criterion}")
        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is not None:
            optimizer_str = f"Optimizer {optimizer.__class__.__name__}:"
            for group_id, group in enumerate(optimizer.param_groups):
                optimizer_str += f"\n\tgroup {group_id}: lr={group['lr']}, weight_decay={group['weight_decay']}"
            self.log(optimizer_str)

    def log_binary_ratio(self):
        n_params_full = sum(map(torch.numel, self.model.parameters()))
        n_params_binary = sum(map(torch.numel, parameters_binary(self.model)))
        self.log(f"Parameters binary={n_params_binary} / total={n_params_full}"
                 f" = {100. * n_params_binary / n_params_full:.2f} %")

    def log_model(self, model: nn.Module, space='-'):
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            self.viz.text(line, win='log', append=self.viz.win_exists('log'))

    def log(self, text: str):
        self.viz.log(text)

    def batch_finished(self):
        self.param_records.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
            self.mutual_info.update(self.model)
            self.mutual_info.plot(self.viz)

    def start_training(self):
        self.mutual_info.update(self.model)
        self.mutual_info.plot(self.viz)

    def update_loss(self, loss: float, mode='batch'):
        self.viz.line_update(loss, win=f'loss', opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'Loss'
        ), name=mode)

    def update_accuracy(self, accuracy: float, mode='batch'):
        self.viz.line_update(accuracy, win=f'accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title=f'Accuracy'
        ), name=mode)

    @Schedule(epoch_update=1)
    def update_accuracy_test(self):
        self.update_accuracy(accuracy=calc_accuracy(self.model, self.test_loader), mode='full test')

    def register_func(self, func: Callable, opts: dict = None):
        self.functions.append((func, opts))

    def update_distribution(self):
        for name, param_record in self.param_records.items():
            param = param_record.param
            if param.numel() == 1:
                self.viz.line_update(y=param.data[0], win=name, opts=dict(
                    xlabel='Epoch',
                    ylabel='Value',
                    title=name,
                ))
            else:
                self.viz.histogram(X=param.data.cpu().view(-1), win=name, opts=dict(
                    xlabel='Param norm',
                    ylabel='# bins (distribution)',
                    title=name,
                ))

    def update_gradient_mean_std(self):
        for name, param_record in self.param_records.items_monitored():
            param = param_record.param
            if param.grad is None:
                continue
            param_record.grad_variance.update(param.grad.data.cpu())
            mean, std = param_record.grad_variance.get_mean_std()
            param_norm = param.data.norm(p=2)
            mean = mean.norm(p=2) / param_norm
            std = std.mean() / param_norm
            self.viz.line_update(y=[mean, std], win=f"grad_mean_std_{name}", opts=dict(
                xlabel='Epoch',
                ylabel='Normalized Mean and STD',
                title=name,
                legend=['||Mean(∇Wi)||', 'STD(∇Wi)'],
                xtype='log',
                ytype='log',
            ))

    def update_timings(self):
        self.viz.text(f'Batch duration: {self.timer.batch_duration(): .2e} sec', win='status')
        self.viz.text(f'Epoch duration: {self.timer.epoch_duration(): .2e} sec', win='status', append=True)
        self.viz.text(f'Training time: {self.timer.training_time()}', win='status', append=True)

    def epoch_finished(self):
        self.update_accuracy_test()
        self.update_distribution()
        self.mutual_info.plot(self.viz)
        self.param_records.plot_sign_flips(self.viz)
        for func_id, (func, opts) in enumerate(self.functions):
            self.viz.line_update(y=func(), win=f"func_{func_id}", opts=opts)
        # statistics below require monitored parameters
        self.update_gradient_mean_std()
        self.update_heatmap_history()
        self.update_active_count()
        self.update_initial_difference()
        self.update_grad_norm()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            self.param_records[name] = ParamRecord(param, monitor=self.watch_parameters)

    def update_heatmap_history(self, by_dim=False):
        """
        :param by_dim: use hitmap_by_dim for the last layer's weights
        """
        def heatmap(tensor: torch.FloatTensor, win: str):
            self.viz.heatmap(X=tensor, win=win, opts=dict(
                colormap='Jet',
                title=win,
                xlabel='input dimension',
                ylabel='output dimension',
                ytickstep=1,
            ))

        def heatmap_by_dim(tensor: torch.FloatTensor, win: str):
            for dim, x_dim in enumerate(tensor):
                factors = factors_root(x_dim.shape[0])
                x_dim = x_dim.view(factors)
                heatmap(x_dim, win=f'{win}: dim {dim}')

        names_backward = list(name for name, _ in self.model.named_parameters())[::-1]
        name_last = None
        for name in names_backward:
            if name in self.param_records:
                name_last = name
                break

        for name, param_record in self.param_records.items_monitored():
            heatmap_func = heatmap_by_dim if by_dim and name == name_last else heatmap
            heatmap_func(tensor=param_record.tstat(), win=f'Heatmap {name} t-statistics')

    def update_active_count(self):
        legend = []
        active_percents = []
        for name, param_record in self.param_records.items_monitored():
            legend.append(name)
            total = param_record.param.numel()
            n_active = total - param_record.inactive.sum()
            active_percents.append(100 * n_active / total)
        self.viz.line_update(y=active_percents, win='active weights', opts=dict(
            xlabel='Epoch',
            ylabel='active, %',
            legend=legend,
            title='% of active weights',
        ))

    def update_initial_difference(self):
        legend = []
        dp_normed = []
        for name, param_record in self.param_records.items_monitored():
            legend.append(name)
            dp = param_record.param.data - param_record.initial_data
            dp = dp.norm(p=2) / param_record.initial_norm
            dp_normed.append(dp)
        self.viz.line_update(y=dp_normed, win='w_initial', opts=dict(
            xlabel='Epoch',
            ylabel='||W - W_initial|| / ||W_initial||',
            title='How far the current weights are from the initial?',
            legend=legend,
        ))

    def update_grad_norm(self):
        grad_norms = []
        for name, param_record in self.param_records.items_monitored():
            grad = param_record.param.grad
            if grad is not None:
                grad_norms.append(grad.data.norm(p=2))
        if len(grad_norms) > 0:
            norm_mean = sum(grad_norms) / len(grad_norms)
            self.viz.line_update(y=norm_mean, win='grad_norm', opts=dict(
                xlabel='Epoch',
                ylabel='Gradient norm, L2',
                title='Average grad norm of all params',
            ))


class MonitorMCMC(Monitor):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.autocorrelation = Autocorrelation(n_lags=self.timer.batches_in_epoch,
                                               with_autocorrelation=isinstance(trainer.train_loader.dataset,
                                                                               MNISTSmall))
        self.graph_mcmc = GraphMCMC(named_params=named_parameters_binary(self.model), timer=self.timer,
                                    history_heatmap=True)

    def mcmc_step(self, param_flips):
        self.autocorrelation.add_samples(param_flips)
        self.graph_mcmc.add_samples(param_flips)

    def epoch_finished(self):
        super().epoch_finished()
        # self.autocorrelation.plot(self.viz)
        # self.graph_mcmc.render(self.viz)
