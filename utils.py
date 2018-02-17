import math
from tqdm import tqdm
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets

from constants import MODELS_DIR


def parameters_binary(model: nn.Module):
    for name, param in named_parameters_binary(model):
        yield param


def named_parameters_binary(model: nn.Module):
    return filter(lambda named_param: getattr(named_param[1], "is_binary", False), model.named_parameters())


def find_param_by_name(model: nn.Module, name_search: str) -> Union[nn.Parameter, None]:
    for name, param in model.named_parameters():
        if name == name_search:
            return param
    return None


def get_data_loader(dataset: str, train=True, batch_size=256, transform=None) -> torch.utils.data.DataLoader:
    transform_list = []
    if dataset == "MNIST":
        dataset_cls = datasets.MNIST
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.1307,), std=(0.3081,)))
    elif dataset == "MNIST56":
        dataset_cls = MNIST56
    elif dataset == "CIFAR10":
        dataset_cls = datasets.CIFAR10
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
        raise NotImplementedError()
    if transform is not None:
        transform_list.insert(0, transform)
    dataset = dataset_cls('data', train=train, download=True, transform=transforms.Compose(transform_list))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def load_model_state(dataset_name: str, model_name: str):
    model_path = MODELS_DIR.joinpath(dataset_name, Path(model_name).with_suffix('.pt'))
    if not model_path.exists():
        return None
    return torch.load(model_path)


class StepLRClamp(torch.optim.lr_scheduler.StepLR):

    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-4):
        self.min_lr = min_lr
        super().__init__(optimizer, step_size, gamma, last_epoch=-1)

    def get_lr(self):
        learning_rates = super().get_lr()
        learning_rates = [max(lr, self.min_lr) for lr in learning_rates]
        return learning_rates


class AdamCustomDecay(torch.optim.Adam):

    def step(self, closure=None):
        """Performs a single optimization step.

                Arguments:
                    closure (callable, optional): A closure that reevaluates the model
                        and returns the loss.
                """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data * (p.data.pow(2) - 0.25))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class MNISTSmall(torch.utils.data.TensorDataset):
    def __init__(self, labels_keep=(5, 6), resize_to=(5, 5),  train=True):
        mnist = get_data_loader(dataset='MNIST', train=train, transform=transforms.Resize(size=resize_to))
        data = []
        targets = []
        for images, labels in tqdm(mnist, desc=f"Preparing {self.__class__.__name__} dataset"):
            for _image, _label in zip(images, labels):
                if _label in labels_keep:
                    new_label = labels_keep.index(_label)
                    targets.append(new_label)
                    data.append(_image)
        data = torch.cat(data, dim=0)
        targets = torch.LongTensor(targets)
        super().__init__(data_tensor=data, target_tensor=targets)


class MNIST56(MNISTSmall):
    """
    MNIST 5 and 6 digits, resized to 5x5.
    """
    def __init__(self, *args, train=True, **kwargs):
        super().__init__(labels_keep=(5, 6), resize_to=(5, 5), train=train)
