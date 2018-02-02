import torch
import torch.nn as nn
import torch.nn.modules.conv
import torch.nn.functional as F
import torch.utils.data


def binarize_model(model: nn.Module, ignore=(nn.Dropout,)) -> nn.Module:
    def _binary_decor(model: nn.Module) -> nn.Module:
        if isinstance(model, nn.modules.conv._ConvNd):
            model = BinaryDecorator(model)
        elif isinstance(model, nn.Linear):
            model = nn.Sequential(
                nn.BatchNorm1d(model.in_features),
                BinaryDecorator(model)
            )
        return model

    for name, child in list(model.named_children()):
        if isinstance(child, ignore):
            delattr(model, name)
            continue
        child_new = binarize_model(child, ignore)
        if child_new is not child:
            setattr(model, name, child_new)
    model = _binary_decor(model)

    return model


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

    def compile_inference(self):
        self.layer.weight.data.sign_()
        self.forward = self.layer.forward

    def forward(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = BinaryFunc.apply(x)
        weight_full = self.layer.weight.data.clone()
        self.layer.weight.data.sign_()
        x = self.layer.forward(x)
        self.layer.weight.data = weight_full
        x = F.mul(x, x_mean)
        return x

    def __repr__(self):
        return "[Binary]" + repr(self.layer)


class SignPassGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryDecoratorConv2d(BinaryDecorator):
    def forward(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = BinaryFunc.apply(x)
        x = F.conv2d(x, SignPassGrad.apply(self.layer.weight), self.layer.bias, self.layer.stride,
                     self.layer.padding, self.layer.dilation, self.layer.groups)
        x = F.mul(x, x_mean)
        return x


class BinaryDecoratorLinear(BinaryDecorator):
    def forward(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = BinaryFunc.apply(x)
        x = F.linear(x, SignPassGrad.apply(self.layer.weight), self.layer.bias)
        x = F.mul(x, x_mean)
        return x


class ScaleFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input, scale)
        return input * scale

    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_variables
        return grad_output * scale, torch.mean(grad_output * input)


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(init_value))

    def forward(self, input):
        return ScaleFunc.apply(input, self.scale)

    def __repr__(self):
        return self.__class__.__name__ + f"({self.scale.numel()} parameters)"
