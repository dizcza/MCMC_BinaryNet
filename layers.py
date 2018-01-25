import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data


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


class _BinaryWrapper(object):

    def __init__(self, cls, weight: nn.Parameter):
        self.cls = cls
        matrix_proba = torch.FloatTensor(weight.data.shape).fill_(0.5)
        weight.data = torch.bernoulli(matrix_proba) * 2 - 1
        self.weight_clone = weight.clone()
        self.weight = weight

    def forward_binary(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = BinaryFunc.apply(x)
        self.weight_clone = self.weight.clone()
        self.weight.data.sign_()
        x = self.cls.forward(self, x)
        self.weight.data = self.weight_clone.data
        x = F.mul(x, x_mean)
        return x


class BinaryLinear(nn.Linear, _BinaryWrapper):

    def __init__(self, in_features, out_features):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        _BinaryWrapper.__init__(self, nn.Linear, self.weight)

    def forward(self, x):
        return self.forward_binary(x)

    def named_parameters(self, memo=None, prefix=''):
        for name, param in super().named_parameters(memo, prefix):
            param.is_binary = True
            yield name, param


class BinaryConv2d(nn.Conv2d, _BinaryWrapper):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, padding=padding, bias=False)
        _BinaryWrapper.__init__(self, nn.Conv2d, self.weight)

    def forward(self, x):
        return self.forward_binary(x)

    def named_parameters(self, memo=None, prefix=''):
        for name, param in super().named_parameters(memo, prefix):
            param.is_binary = True
            yield name, param


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
