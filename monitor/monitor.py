import time
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data

from monitor.autocorrelation import Autocorrelation
from monitor.batch_timer import BatchTimer
from monitor.graph import GraphMCMC
from monitor.mutual_info.mutual_info import MutualInfoKMeans
from monitor.var_online import VarianceOnline
from monitor.viz import VisdomMighty
from utils import named_parameters_binary, parameters_binary, MNISTSmall


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
        self.timer = BatchTimer(batches_in_epoch=len(trainer.train_loader))
        self.viz = VisdomMighty(env=f"{trainer.dataset_name} "
                                    f"{trainer.__class__.__name__} "
                                    f"{time.strftime('%b-%d %H:%M')}", timer=self.timer)
        self.model = trainer.model
        self.params = ParamList()
        self.mutual_info = MutualInfoKMeans(self.viz, estimate_size=int(1e4), debug=False)
        self.mutual_info.schedule(self.timer, epoch_update=0, batch_update=2)
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
        self.mutual_info.update(self.model)

    def update_loss(self, loss: float, mode='batch'):
        self.viz.line_update(loss, win=f'loss', opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'Loss'
        ), name=mode)

    def update_accuracy(self, accuracy: float, mode='batch'):
        self.viz.line_update(accuracy, win=f'accuracy', opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'Train accuracy'
        ), name=mode)

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
        self.viz.text(f'Batch duration: {self.timer.batch_duration(): .2e} ms', win='status')
        self.viz.text(f'Epoch duration: {self.timer.epoch_duration(): .2e} ms', win='status', append=True)

    def epoch_finished(self):
        self.update_timings()
        self.mutual_info.plot()
        self.params.plot_sign_flips(self.viz)
        for func_id, (func, opts) in enumerate(self.functions):
            self.viz.line_update(y=func(), win=f"func_{func_id}", opts=opts)
        self.update_gradient_mean_std()
        # self.update_distribution()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            self.params.append(ParamRecord(name, param))


class MonitorMCMC(Monitor):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.autocorrelation = Autocorrelation(n_lags=self.timer.batches_in_epoch,
                                               with_autocorrelation=isinstance(trainer.train_loader.dataset,
                                                                               MNISTSmall))
        self.autocorrelation.schedule(self.timer, epoch_update=10)
        self.graph_mcmc = GraphMCMC(named_params=named_parameters_binary(self.model), timer=self.timer,
                                    history_heatmap=True)
        self.graph_mcmc.schedule(self.timer, epoch_update=1)

    def mcmc_step(self, param_flips):
        self.autocorrelation.add_samples(param_flips)
        self.graph_mcmc.add_samples(param_flips)

    def epoch_finished(self):
        super().epoch_finished()
        self.autocorrelation.plot(self.viz)
        self.graph_mcmc.render(self.viz)
