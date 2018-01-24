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


class BinaryLinear(nn.Linear):

    def forward(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = F.linear(BinaryFunc.apply(x), self.weight.sign(), self.bias)
        x = F.mul(x, x_mean)
        return x

    def parameters(self):
        for p in super().parameters():
            p.is_binary = True
            yield p


class BinaryConv2d(nn.Conv2d):

    def forward(self, x):
        x_mean = torch.mean(torch.abs(x))
        x = F.conv2d(BinaryFunc.apply(x), self.weight.sign(), self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        x = F.mul(x, x_mean)
        return x

    def parameters(self):
        for p in super().parameters():
            p.is_binary = True
            yield p


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
