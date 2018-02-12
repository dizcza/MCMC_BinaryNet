import torch
import torch.nn as nn
import torch.utils.data

from layers import ScaleLayer, BinaryDecorator, binarize_model
from trainer import TrainerGradFullPrecision, TrainerMCMC, TrainerGradBinary
from utils import StepLRClamp


class NetBinary(nn.Module):
    def __init__(self, conv_channels, fc_sizes, conv_kernel=3):
        super().__init__()

        conv_layers = []
        for (in_features, out_features) in zip(conv_channels[:-1], conv_channels[1:]):
            # conv_layers.append(nn.BatchNorm2d(in_features))
            layer = nn.Conv2d(in_features, out_features, kernel_size=conv_kernel, padding=0, bias=False)
            conv_layers.append(layer)
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            conv_layers.append(nn.PReLU())
        self.conv_sequential = nn.Sequential(*conv_layers)

        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.BatchNorm1d(in_features))
            layer = nn.Linear(in_features, out_features, bias=False)
            fc_layers.append(layer)
            fc_layers.append(nn.PReLU(out_features))
        self.fc_sequential = nn.Sequential(*fc_layers)
        self.scale_layer = ScaleLayer(size=fc_sizes[-1])

    def forward(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_sequential(x)
        x = self.scale_layer(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(3 * 32 ** 2, 10, bias=False),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)


def train_gradient(model, is_binary=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-3, min_lr=1e-4)
    if is_binary:
        model = binarize_model(model)
        trainer_cls = TrainerGradBinary
    else:
        trainer_cls = TrainerGradFullPrecision
    trainer = trainer_cls(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name="CIFAR10",
                          optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epoch=200, debug=0)


def train_mcmc(model):
    model = binarize_model(model)
    trainer = TrainerMCMC(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name="CIFAR10",
                          flip_ratio=1e-2)
    trainer.train(n_epoch=500, debug=0)


if __name__ == '__main__':
    # train_gradient(model=NetBinary(conv_channels=[], fc_sizes=[3 * 32 ** 2, 10]), is_binary=True)
    train_mcmc(model=MLP())

