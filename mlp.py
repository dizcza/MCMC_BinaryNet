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


class Binarize(nn.Module):

    def forward(self, input):
        return BinaryFunc.apply(input)


class BinaryLinear(nn.Linear):

    def forward(self, input):
        return F.linear(input, self.weight.sign(), self.bias)

    def parameters(self):
        for p in super().parameters():
            p.is_binary = True
            yield p


class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        return F.conv2d(input, self.weight.sign(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

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
        return grad_output * scale, torch.sum(grad_output * tensor)


class ScaleLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(1))

    def forward(self, input):
        return ScaleFunc.apply(input, self.scale)


class NetBinary(nn.Module):
    def __init__(self, conv_channels, fc_sizes):
        super().__init__()

        conv_layers = []
        for (in_features, out_features) in zip(conv_channels[:-1], conv_channels[1:]):
            conv_layers.append(Binarize())
            conv_layers.append(BinaryConv2d(in_features, out_features, kernel_size=3, padding=1, bias=False))
            conv_layers.append(nn.PReLU(out_features))
            conv_layers.append(nn.BatchNorm2d(out_features))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.fc_in_features = 28 ** 2 * conv_channels[-1]
        fc_sizes = [self.fc_in_features, *fc_sizes]
        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(Binarize())
            fc_layers.append(BinaryLinear(in_features, out_features, bias=False))
            fc_layers.append(nn.PReLU(out_features))
            fc_layers.append(nn.BatchNorm1d(out_features))
        fc_layers.append(ScaleLayer())
        self.fc_chain = nn.Sequential(*fc_layers)

    def __str__(self):
        return type(self).__name__

    def parameters_binary(self):
        return filter(lambda param: getattr(param, "is_binary", False), self.parameters())

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_in_features)
        x = self.fc_chain(x)
        return x


def train_binary(n_epoch=10):
    conv_channels = [1, 2]
    fc_sizes = [2048, 10]
    model = NetBinary(conv_channels, fc_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scheduler = StepLRClamp(optimizer, step_size=1, gamma=0.5, min_lr=1e-6)
    trainer = Trainer(model, criterion, optimizer, scheduler)
    trainer.train(n_epoch)


if __name__ == '__main__':
    train_binary()
    test()
