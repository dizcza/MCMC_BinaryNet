from typing import Union

import torch.nn as nn


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
