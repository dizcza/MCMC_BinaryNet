import copy
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.modules.conv
import torch.utils.data


class Identity(nn.Module):
    def forward(self, x):
        return x


def find_layers(model: nn.Module, layer_class):
    for name, layer in find_named_layers(model, layer_class=layer_class):
        yield layer


def find_named_layers(model: nn.Module, layer_class, name_prefix=''):
    for name, layer in model.named_children():
        yield from find_named_layers(layer, layer_class, name_prefix=f"{name_prefix}.{name}")
    if isinstance(model, layer_class):
        yield name_prefix.lstrip('.'), model


def replace_relu(model: nn.Module, new_relu, drop_layers=()):
    """
    :param model: network model
    :param new_relu: new relu activation function
    :param drop_layers: drop these layers; you can try dropping batch norm
    :return: model with relu replaced by kWTA activation function
    """
    if isinstance(model, new_relu.__class__):
        return model
    for name, child in list(model.named_children()):
        if isinstance(child, drop_layers):
            setattr(model, name, Identity())
            continue
        child_new = replace_relu(model=child, new_relu=new_relu, drop_layers=drop_layers)
        if child_new is not child:
            setattr(model, name, child_new)
    if isinstance(model, (nn.ReLU, nn.RReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
        model = copy.deepcopy(new_relu)
    return model


def compile_inference(model: nn.Module):
    for name, child in list(model.named_children()):
        compile_inference(child)
    if isinstance(model, BinaryDecorator):
        model.compile_inference()


class BinaryFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        tensor = tensor > 0
        tensor = 2 * tensor.type(torch.float32) - 1
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        return grad_output


class BinaryDecorator(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        for param in layer.parameters():
            param.is_binary = True
        self.layer = layer
        self.is_inference = False

    def compile_inference(self):
        for param in self.layer.parameters():
            param.data.sign_()
        self.is_inference = True

    def forward(self, x):
        x = BinaryFunc.apply(x)
        if self.is_inference:
            x = self.layer(x)
        else:
            weight_full = self.layer.weight.data.clone()
            self.layer.weight.data.sign_()
            x = self.layer(x)
            self.layer.weight.data = weight_full
        return x

    def __repr__(self):
        tag = "[Binary]"
        if self.is_inference:
            tag += '[Compiled]'
        return tag + repr(self.layer)


class BinaryDecoratorSoft(BinaryDecorator):

    def __init__(self, layer: nn.Module):
        super().__init__(layer=layer)
        self.weight_threshold = nn.Parameter(torch.randn(layer.weight.shape))
        self.activation_threshold = nn.Parameter(torch.randn(layer.in_features))
        self.hardness = 1

    def binary(self, tensor, soft=None):
        if soft is None:
            soft = self.training
        if soft:
            return (tensor * self.hardness).tanh()
        else:
            return tensor.sign()

    def compile_inference(self):
        self.layer.weigh.data = self.binary(self.layer.weight.data - self.weight_threshold.data, soft=False)
        self.is_inference = True

    def forward(self, x):
        x = self.binary(x - self.activation_threshold)
        if self.is_inference:
            x = self.layer(x)
        else:
            weights_binary = self.binary(self.layer.weight - self.weight_threshold)
            x = x @ weights_binary.t()
        return x

    def __repr__(self):
        return "[Soft]" + super().__repr__()


class ScaleLayer(nn.Module):

    def __init__(self, size: int, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(size).fill_(init_value))

    def forward(self, x):
        return self.scale * x

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.scale.numel()})"


def binarize_model(model: nn.Module, drop_layers=(nn.ReLU, nn.PReLU), binarizer=BinaryDecorator) -> nn.Module:
    """
    :param model: net model
    :param drop_layers: remove these layers from the input model
    :param binarizer: what binarization to use: soft or hard
    :return: model with linear and conv layers wrapped in BinaryDecorator
    """
    if isinstance(model, BinaryDecorator):
        print("Layer is already binarized.")
        return model
    for name, child in list(model.named_children()):
        if isinstance(child, drop_layers):
            delattr(model, name)
            continue
        child_new = binarize_model(model=child, drop_layers=drop_layers, binarizer=binarizer)
        if child_new is not child:
            setattr(model, name, child_new)
    if isinstance(model, (nn.modules.conv._ConvNd, nn.Linear)):
        if hasattr(model, 'bias'):
            delattr(model, 'bias')
            model.register_parameter(name='bias', param=None)
        model = binarizer(model)
    return model


class SerializableModule(nn.Module, ABC):
    state_attr = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for attribute in self.state_attr:
            destination[prefix + attribute] = getattr(self, attribute)
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict_keys = list(state_dict.keys())
        for attribute in self.state_attr:
            key = prefix + attribute
            if key in state_dict_keys:
                setattr(self, attribute, state_dict.pop(key))
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata,
                                      strict=strict, missing_keys=missing_keys, unexpected_keys=unexpected_keys,
                                      error_msgs=error_msgs)
