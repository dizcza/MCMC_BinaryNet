import torch
import torch.nn as nn
import torch.nn.modules.conv
import torch.nn.functional as F
import torch.utils.data


def binarize_model(model: nn.Module, drop_layers=(nn.Dropout,)) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, drop_layers):
            delattr(model, name)
            continue
        child_new = binarize_model(child, drop_layers)
        if child_new is not child:
            setattr(model, name, child_new)
    if isinstance(model, (nn.modules.conv._ConvNd, nn.Linear)):
        if hasattr(model, 'bias'):
            delattr(model, 'bias')
            model.register_parameter(name='bias', param=None)
        model = BinaryDecorator(model)
    return model


def compile_inference(model: nn.Module):
    for name, child in list(model.named_children()):
        compile_inference(child)
    if isinstance(model, BinaryDecorator):
        model.compile_inference()


class BinaryFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

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
        x_mean = torch.mean(torch.abs(x))
        x = BinaryFunc.apply(x)
        if self.is_inference:
            x = self.layer(x)
        else:
            weight_full = self.layer.weight.data.clone()
            self.layer.weight.data.sign_()
            x = self.layer(x)
            self.layer.weight.data = weight_full
        x = F.mul(x, x_mean)
        return x

    def __repr__(self):
        tag = "[Binary]"
        if self.is_inference:
            tag += '[Compiled]'
        return tag + repr(self.layer)


class ScaleLayer(nn.Module):

    def __init__(self, size: int, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(size).fill_(init_value))

    def forward(self, x):
        return F.mul(x, self.scale)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.scale.numel()})"
