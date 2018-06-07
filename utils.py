import math
from functools import lru_cache
from tqdm import tqdm
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets

from constants import DATA_DIR, MODELS_DIR
from monitor.var_online import dataset_mean_std


@lru_cache(maxsize=32, typed=False)
def factors_root(number: int):
    """
    :param number: an integer value
    :return: two integer factors, closest to the square root of the input
    """
    root = int(math.sqrt(number))
    for divisor in range(root, 0, -1):
        if number % divisor == 0:
            return divisor, number // divisor
    return 1, number


def parameters_binary(model: nn.Module):
    for name, param in named_parameters_binary(model):
        yield param


def is_binary(param: nn.Parameter):
    return getattr(param, 'is_binary', False)


def named_parameters_binary(model: nn.Module):
    return [(name, param) for name, param in model.named_parameters() if is_binary(param)]


def find_param_by_name(model: nn.Module, name_search: str) -> Union[nn.Parameter, None]:
    for name, param in model.named_parameters():
        if name == name_search:
            return param
    return None


def has_binary_params(layer: nn.Module):
    if layer is None:
        return False
    return all(map(is_binary, layer.parameters()))


def get_data_loader(dataset: str, train=True, batch_size=256) -> torch.utils.data.DataLoader:
    transform_list = []
    if dataset == "MNIST":
        dataset_cls = datasets.MNIST
        transform_list.append(transforms.ToTensor())
        transform_list.append(NormalizeFromDataset(dataset_cls=dataset_cls))
    elif dataset == "MNIST56":
        dataset_cls = MNIST56
    elif dataset == "CIFAR10":
        dataset_cls = datasets.CIFAR10
        transform_list.append(transforms.ToTensor())
        transform_list.append(NormalizeFromDataset(dataset_cls=dataset_cls))
    else:
        raise NotImplementedError()
    dataset = dataset_cls(DATA_DIR, train=train, download=True, transform=transforms.Compose(transform_list))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def load_model_state(dataset_name: str, model_name: str):
    model_path = MODELS_DIR.joinpath(dataset_name, Path(model_name).with_suffix('.pt'))
    if not model_path.exists():
        return None
    return torch.load(model_path)


class NormalizeFromDataset(transforms.Normalize):

    def __init__(self, dataset_cls: type):
        mean, std = dataset_mean_std(dataset_cls=dataset_cls)
        std[std == 0] = 1  # normalized values will be zeros
        super().__init__(mean=mean, std=std)


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
        self.train = train
        data_path = self.get_data_path()
        if not data_path.exists():
            mnist = datasets.MNIST(DATA_DIR, train=train, transform=transforms.Compose(
                [transforms.Resize(size=resize_to),
                 transforms.ToTensor()]
            ))
            self.process_mnist(mnist, labels_keep)
        with open(data_path, 'rb') as f:
            data, targets = torch.load(f)
        super().__init__(data_tensor=data, target_tensor=targets)

    def get_data_path(self):
        return Path('./data').joinpath(self.__class__.__name__, 'train.pt' if self.train else 'test.pt')

    def process_mnist(self, mnist: torch.utils.data.Dataset, labels_keep: tuple):
        data = []
        targets = []
        for image, label_old in tqdm(mnist, desc=f"Preparing {self.__class__.__name__} dataset"):
            if label_old in labels_keep:
                label_new = labels_keep.index(label_old)
                targets.append(label_new)
                data.append(image)
        data = torch.cat(data, dim=0)
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data = (data - data_mean) / data_std
        targets = torch.LongTensor(targets)
        data_path = self.get_data_path()
        data_path.parent.mkdir(exist_ok=True, parents=True)
        with open(data_path, 'wb') as f:
            torch.save((data, targets), f)
        print(f"Saved preprocessed data to {data_path}")


class MNIST56(MNISTSmall):
    """
    MNIST 5 and 6 digits, resized to 5x5.
    """
    def __init__(self, *args, train=True, **kwargs):
        super().__init__(labels_keep=(5, 6), resize_to=(5, 5), train=train)
