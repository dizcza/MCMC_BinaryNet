import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm

from constants import MODELS_DIR
from monitor import Monitor, calc_accuracy
from utils import get_data_loader, named_parameters_binary, parameters_binary, load_model, find_param_by_name


class _Trainer(object):

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str):
        self.model = model
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.train_loader = get_data_loader(dataset_name, train=True)
        self.monitor = Monitor(model, dataset_name, batches_in_epoch=len(self.train_loader))

    def save_model(self, accuracy: float = None):
        model_path = MODELS_DIR.joinpath(self.dataset_name, self.model.__class__.__name__).with_suffix('.pt')
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model, model_path)
        msg = f"Saved to {model_path}"
        if accuracy is not None:
            msg += f" (train accuracy: {accuracy:.4f})"
        print(msg)

    def load_best_accuracy(self, debug=False) -> float:
        train_loader = get_data_loader(self.dataset_name, train=True)
        best_accuracy = 0.
        if not debug:
            try:
                loaded_model = load_model(self.dataset_name, self.model.__class__.__name__)
                best_accuracy = calc_accuracy(loaded_model, train_loader)
            except Exception:
                print(f"Couldn't estimate the best accuracy for {self.model}. Reset to 0.")
        return best_accuracy

    def _train_batch(self, images, labels):
        raise NotImplementedError()

    def _epoch_started(self, epoch):
        pass

    def _epoch_finished(self, epoch):
        pass
    
    def train(self, n_epoch=10, debug=False):
        print(self.model)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        for name, param in named_parameters_binary(self.model):
            self.monitor.register_param(name, param)
        scale_param = find_param_by_name(self.model, name_search='scale_layer.scale')
        if scale_param is not None:
            self.monitor.register_param(param_name='scale_layer.scale', param=scale_param)
        best_accuracy = self.load_best_accuracy(debug)
        self.monitor.log(f"Best train accuracy so far: {best_accuracy:.4f}")
        print(f"Training '{self.model}'. Best {self.dataset_name} train accuracy so far: {best_accuracy:.4f}")

        for epoch in range(n_epoch):
            self._epoch_started(epoch)
            for images, labels in tqdm(self.train_loader,
                                       desc="Epoch {:d}/{:d}".format(epoch, n_epoch),
                                       leave=False):
                images = Variable(images)
                labels = Variable(labels)
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs, loss = self._train_batch(images, labels)
                self.monitor.batch_finished(outputs, labels, loss)

            if not debug:
                accuracy = calc_accuracy(self.model, self.train_loader)
                is_best = accuracy > best_accuracy
                self.monitor.update_train_accuracy(accuracy, is_best)
                if is_best:
                    self.save_model(accuracy)
                    best_accuracy = accuracy

            self._epoch_finished(epoch)


class TrainerGradFullPrecision(_Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, optimizer: torch.optim.Optimizer,
                 scheduler=None):
        super().__init__(model, criterion, dataset_name)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _train_batch(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step(closure=None)
        return outputs, loss

    def _epoch_started(self, epoch):
        self.monitor.log(f"Epoch {epoch}. Learning rate {self.scheduler.get_lr()}")

    def _epoch_finished(self, epoch):
        if self.scheduler is not None:
            self.scheduler.step(epoch)


class TrainerGradBinary(TrainerGradFullPrecision):
    def _train_batch(self, images, labels):
        outputs, loss = super()._train_batch(images, labels)
        for param in parameters_binary(self.model):
            param.data.clamp_(min=-1, max=1)
        return outputs, loss


class TrainerMCMC(_Trainer):
    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, temperature=0.1, flip_ratio=0.1):
        super().__init__(model, criterion, dataset_name)
        self.temperature = temperature
        self.flip_ratio = flip_ratio
        self.monitor.log(f"Temperature: {self.temperature}; flip ratio: {flip_ratio}")
        self.accepted_count = 0
        self.update_calls = 0
        for param in model.parameters():
            param.requires_grad = False
            param.volatile = True

    def get_acceptance_ratio(self) -> float:
        if self.update_calls == 0:
            return 0
        else:
            return self.accepted_count / self.update_calls

    def _train_batch(self, images, labels):
        outputs_orig = self.model(images)
        loss_orig = self.criterion(outputs_orig, labels)

        param_modified, data_orig = flip_signs(parameters_binary(self.model), self.flip_ratio)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss_delta = (loss - loss_orig).data
        self.monitor._draw_line(y=loss_delta[0], win='loss_delta', opts=dict(
            xlabel='Epoch',
            ylabel='ΔL',
            title='loss_flipped - loss_orig'
        ))
        mcmc_proba_accept = torch.exp(-loss_delta / self.temperature)[0]
        proba_draw = random.random()
        if proba_draw < mcmc_proba_accept:
            self.accepted_count += 1
        else:
            # reject
            param_modified.data = data_orig
            outputs = outputs_orig
            loss = loss_orig
        self.update_calls += 1

        self.monitor._draw_line(y=self.get_acceptance_ratio(), win='accp', opts=dict(
            xlabel='Epoch',
            ylabel='Acceptance ratio',
            title='MCMC accepted / total_tries'
        ))

        return outputs, loss

    def _epoch_finished(self, epoch):
        if (epoch + 1) % 10 == 0:
            self.flip_ratio = max(self.flip_ratio / 3, 1e-4)
            self.temperature /= 2.7**3


def flip_signs(parameters, flip_ratio=0.1):
    param_modified = random.choice(list(parameters))
    idx_to_flip = torch.rand(param_modified.data.shape) < flip_ratio
    if param_modified.data.is_cuda:
        idx_to_flip = idx_to_flip.cuda()
    data_orig = param_modified.data.clone()
    param_modified.data[idx_to_flip] *= -1
    return param_modified, data_orig
