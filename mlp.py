import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from layers import ScaleLayer, BinaryConv2d, BinaryLinear
from trainer import StepLRClamp, Trainer, test


class NetBinary(nn.Module):
    def __init__(self, conv_channels, fc_sizes):
        super().__init__()

        conv_layers = []
        for (in_features, out_features) in zip(conv_channels[:-1], conv_channels[1:]):
            conv_layers.append(nn.BatchNorm2d(in_features))
            conv_layers.append(BinaryConv2d(in_features, out_features, kernel_size=5, padding=2))
            conv_layers.append(nn.PReLU())
        self.conv_sequential = nn.Sequential(*conv_layers)

        fc_in_features = 28 ** 2 * conv_channels[-1]
        fc_sizes = [fc_in_features, *fc_sizes]
        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.BatchNorm1d(in_features))
            fc_layers.append(BinaryLinear(in_features, out_features))
            fc_layers.append(nn.PReLU())
        self.fc_sequential = nn.Sequential(*fc_layers)
        self.scale_layer = ScaleLayer()

    def __str__(self):
        return type(self).__name__

    def parameters_binary(self):
        for name, param in self.named_parameters_binary():
            yield param

    def named_parameters_binary(self):
        return filter(lambda named_param: getattr(named_param[1], "is_binary", False), self.named_parameters())

    def forward(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_sequential(x)
        x = self.scale_layer(x)
        return x


def train_binary(n_epoch=100):
    conv_channels = [1, 10, 20]
    fc_sizes = [320, 50, 10]
    model = NetBinary(conv_channels, fc_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = StepLRClamp(optimizer, step_size=2, gamma=0.5, min_lr=1e-6)
    trainer = Trainer(model, criterion, optimizer, scheduler)
    trainer.train(n_epoch, debug=False)


if __name__ == '__main__':
    train_binary()
    # test()
