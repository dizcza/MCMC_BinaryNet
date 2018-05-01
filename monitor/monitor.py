import time
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data
import scipy.stats

from monitor.autocorrelation import Autocorrelation
from monitor.batch_timer import timer, Schedule
from monitor.graph import GraphMCMC
from monitor.mutual_info.mutual_info import MutualInfoKMeans, MutualInfoSign, MutualInfoKNN
from monitor.var_online import VarianceOnline
from monitor.viz import VisdomMighty
from monitor.accuracy import calc_accuracy
from utils import named_parameters_binary, parameters_binary, MNISTSmall, get_data_loader


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
    def __init__(self, name: str, param: nn.Parameter):
        self.name = name
        self.param = param
        self.variance = VarianceOnline()
        self.prev_sign = param.data.cpu().clone()  # clone is faster


class ParamList(list):
    def __init__(self):
        super().__init__()
        self.sign_flips = 0
        self.n_updates = 0

    # todo move auto- & cross-correlation here
    def batch_finished(self):
        self.n_updates += 1
        for param_record in self:
            param = param_record.param
            new_data = param.data.cpu()
            if new_data is param.data:
                new_data = new_data.clone()
            self.sign_flips += torch.sum((new_data * param_record.prev_sign) < 0)
            param_record.prev_sign = new_data

    def plot_sign_flips(self, viz: VisdomMighty):
        if len(self) == 0:
            # haven't registered any param yet
            return
        param_count = 0
        for param_record in self:
            param_count += torch.numel(param_record.param)
        flips = self.sign_flips
        flips /= param_count  # per param
        flips /= self.n_updates  # per update
        flips *= 100.  # percents
        self.sign_flips = 0
        self.n_updates = 0
        viz.line_update(y=flips, win='sign', opts=dict(
            xlabel='Epoch',
            ylabel='Sign flips, %',
            title="Sign flips after optimizer.step()",
        ))


class Monitor(object):
    # todo: feature maps

    def __init__(self, trainer):
        """
        :param trainer: Trainer instance
        """
        self.timer = timer
        self.timer.init(batches_in_epoch=len(trainer.train_loader))
        self.viz = VisdomMighty(env=f"{time.strftime('%Y-%b-%d')} "
                                    f"{trainer.dataset_name} "
                                    f"{trainer.__class__.__name__}", timer=self.timer)
        self.model = trainer.model
        self.test_loader = get_data_loader(dataset=trainer.dataset_name, train=False)
        self.params = ParamList()
        self.mutual_info = MutualInfoKMeans(estimate_size=int(1e4), compression_range=(0.5, 0.999))

        # remove if memory consumption is concerned
        self.param_data_online = {
            name: VarianceOnline(tensor=param.data) for name, param in named_parameters_binary(self.model)
        }

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
        self.log(f"Parameters binary={n_params_binary:e} / total={n_params_full}"
                 f" = {100. * n_params_binary / n_params_full:.2f} %")

    def log_model(self, model: nn.Module, space='-'):
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            self.viz.text(line, win='log', append=self.viz.win_exists('log'))

    def log(self, text: str):
        self.viz.log(text)

    def batch_finished(self):
        self.params.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
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
        for param_record in self.params:
            name, param = param_record.name, param_record.param
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
        for param_record in self.params:
            name, param = param_record.name, param_record.param
            if param.grad is None:
                continue
            param_record.variance.update(param.grad.data.cpu())
            mean, std = param_record.variance.get_mean_std()
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
        self.update_timings()
        self.update_accuracy_test()
        self.mutual_info.plot(self.viz)
        self.params.plot_sign_flips(self.viz)
        for func_id, (func, opts) in enumerate(self.functions):
            self.viz.line_update(y=func(), win=f"func_{func_id}", opts=opts)
        self.update_gradient_mean_std()
        self.update_heatmap_online()
        # self.update_distribution()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            self.params.append(ParamRecord(name, param))

    def update_heatmap_online(self):
        def heatmap(X, win, **opts_kwargs):
            self.viz.heatmap(X=X, win=win, opts=dict(
                colormap='Jet',
                title=win,
                xlabel='input dimension',
                ylabel='output dimension',
                ytickstep=1,
                **opts_kwargs
            ))

        for name, var_online in self.param_data_online.items():
            mean, std = var_online.get_mean_std()
            heatmap(X=mean, win=f'Heatmap {name} Mean')
            # heatmap(X=std, win=f'Heatmap {name} STD')
            # heatmap(X=std / mean.abs(), win=f'Heatmap {name} Coef of variation')
            isnan = std == 0
            if not isnan.all():
                tstat = mean.abs() / std
                tstat[isnan] = tstat[~isnan].max()
                heatmap(X=tstat, win=f'Heatmap {name} t-statistics')
            # p = 1 - scipy.stats.norm.cdf(0, loc=mean.abs(), scale=std)
            # heatmap(X=p, win=f'Heatmap {name} P(abs(w) > 0)')
            if mean.shape[0] == 2:
                # binary classifier
                diff = mean[1, ] - mean[0, ]
                # heatmap(X=diff.view(1, -1), win=f'Heatmap {name} W[1,] - W[0,]', ytick=False)


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
        for pflip in param_flips:
            self.param_data_online[pflip.name].update(pflip.param.data)

    def epoch_finished(self):
        super().epoch_finished()
        # self.autocorrelation.plot(self.viz)
        # self.graph_mcmc.render(self.viz)
