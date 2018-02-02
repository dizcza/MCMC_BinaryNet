import torch
import torch.nn as nn
import torch.utils.data

from layers import ScaleLayer, BinaryDecorator, binarize_model
from metrics import test
from trainer import TrainerGradFullPrecision, TrainerMCMC, TrainerGradBinary
from utils import StepLRClamp, load_model


class NetBinary(nn.Module):
    def __init__(self, conv_channels, fc_sizes, conv_kernel=3):
        super().__init__()

        conv_layers = []
        for (in_features, out_features) in zip(conv_channels[:-1], conv_channels[1:]):
            conv_layers.append(nn.BatchNorm2d(in_features))
            layer = nn.Conv2d(in_features, out_features, kernel_size=conv_kernel, padding=0, bias=False)
            layer = BinaryDecorator(layer)
            conv_layers.append(layer)
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            conv_layers.append(nn.PReLU())
        self.conv_sequential = nn.Sequential(*conv_layers)

        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.BatchNorm1d(in_features))
            layer = nn.Linear(in_features, out_features, bias=False)
            layer = BinaryDecorator(layer)
            fc_layers.append(layer)
            fc_layers.append(nn.PReLU())
        self.fc_sequential = nn.Sequential(*fc_layers)
        self.scale_layer = ScaleLayer()

    def forward(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_sequential(x)
        x = self.scale_layer(x)
        return x


def train_gradient():
    conv_channels = []
    fc_sizes = [28**2, 10]
    model = NetBinary(conv_channels, fc_sizes)
    # for n, p in model.named_parameters():
    #     print(n)
    # quit()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
    scheduler = StepLRClamp(optimizer, step_size=3, gamma=0.5, min_lr=1e-4)
    trainer = TrainerGradBinary(model, criterion, dataset_name="MNIST", optimizer=optimizer, scheduler=scheduler)
    trainer.train(n_epoch=200, debug=0)


def train_mcmc(dataset_name="MNIST"):
    conv_channels = []
    fc_sizes = [28**2, 10]
    model = NetBinary(conv_channels, fc_sizes)
    # model_pretrained = load_model(dataset_name, model_name="NetBinary")
    # model.load_state_dict(model_pretrained.state_dict())
    # del model_pretrained
    trainer = TrainerMCMC(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name=dataset_name,
                          temperature=1e-3,
                          flip_ratio=3*1e-3)
    trainer.train(n_epoch=100, debug=0)


class FullPrecisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28**2, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)


def train_full_precision(dataset_name="MNIST"):
    model = FullPrecisionNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    trainer = TrainerGradFullPrecision(model,
                                       criterion=nn.CrossEntropyLoss(),
                                       dataset_name=dataset_name,
                                       optimizer=optimizer,
                                       scheduler=StepLRClamp(optimizer, step_size=5, min_lr=1e-4))
    trainer.train(n_epoch=100, debug=1)


if __name__ == '__main__':
    # train_gradient()
    train_mcmc()
    # test(train=True)
    # train_full_precision()
