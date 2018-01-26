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


class BinaryDecorator(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        matrix_proba = torch.FloatTensor(layer.weight.data.shape).fill_(0.5)
        layer.weight.data = torch.bernoulli(matrix_proba) * 2 - 1
        layer.weight_clone = layer.weight.clone()
        for param in layer.parameters():
            param.is_binary = True
        self.layer = layer

    def forward(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = BinaryFunc.apply(x)
        self.layer.weight_clone = self.layer.weight.clone()
        self.layer.weight.data.sign_()
        if (self.layer.weight.data == 0).any():
            print((self.layer.weight.data == 0).sum())
            print('weight')
        if (x == 0).any():
            print((x == 0).sum())
            print('x')
        assert not (self.layer.weight.data == 0).any()
        assert not (x == 0).any()
        x = self.layer.forward(x)
        self.layer.weight.data = self.layer.weight_clone.data
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
