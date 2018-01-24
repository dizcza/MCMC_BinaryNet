import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from utils import StepLRClamp, Trainer, test


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
    def forward(ctx, tensor, scale):
        ctx.save_for_backward(tensor, scale)
        return tensor * scale

    @staticmethod
    def backward(ctx, grad_output):
        tensor, scale = ctx.saved_variables
        return grad_output * scale, torch.mean(grad_output * tensor)


class ScaleLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(1e-3))

    def forward(self, input):
        return ScaleFunc.apply(input, self.scale)


class NetBinary(nn.Module):
    def __init__(self, conv_channels, fc_sizes):
        super().__init__()

        conv_layers = []
        for (in_features, out_features) in zip(conv_channels[:-1], conv_channels[1:]):
            conv_layers.append(nn.BatchNorm2d(in_features))
            conv_layers.append(BinaryConv2d(in_features, out_features, kernel_size=3, padding=1, bias=False))
            conv_layers.append(nn.PReLU(out_features))
        self.conv_sequential = nn.Sequential(*conv_layers)

        fc_in_features = 28 ** 2 * conv_channels[-1]
        fc_sizes = [fc_in_features, *fc_sizes]
        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.BatchNorm1d(in_features))
            fc_layers.append(BinaryLinear(in_features, out_features, bias=False))
            fc_layers.append(nn.PReLU(out_features))
        self.fc_sequential = nn.Sequential(*fc_layers)
        self.scale_layer = ScaleLayer()

    def __str__(self):
        return type(self).__name__

    def parameters_binary(self):
        return filter(lambda param: getattr(param, "is_binary", False), self.parameters())

    def forward(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_sequential(x)
        x = self.scale_layer(x)
        return x


def train_binary(n_epoch=10):
    conv_channels = [1, 2]
    fc_sizes = [1024, 10]
    model = NetBinary(conv_channels, fc_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLRClamp(optimizer, step_size=1, gamma=0.5, min_lr=1e-6)
    trainer = Trainer(model, criterion, optimizer, scheduler)
    trainer.train(n_epoch, debug=True)


if __name__ == '__main__':
    train_binary()
    test()
