import torch
import torch.nn as nn
import torch.utils.data

from layers import ScaleLayer, BinaryDecorator, binarize_model
from trainer import TrainerGradFullPrecision, TrainerMCMC, TrainerGradBinary
from utils import StepLRClamp, AdamCustomDecay


linear_features = {
    "MNIST": (28 ** 2, 10),
    "MNIST56": (5 ** 2, 2),
    "CIFAR10": (3 * 32 ** 2, 10),
}


class NetBinary(nn.Module):
    def __init__(self, conv_channels=(), fc_sizes=(), conv_kernel=3, with_scale_layer=False):
        super().__init__()

        conv_layers = []
        for (in_features, out_features) in zip(conv_channels[:-1], conv_channels[1:]):
            conv_layers.append(nn.BatchNorm2d(in_features))
            conv_layers.append(nn.Conv2d(in_features, out_features, kernel_size=conv_kernel, padding=0, bias=False))
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            conv_layers.append(nn.PReLU())
        self.conv_sequential = nn.Sequential(*conv_layers)

        fc_layers = []
        for (in_features, out_features) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.BatchNorm1d(in_features))
            fc_layers.append(nn.Linear(in_features, out_features, bias=False))
            fc_layers.append(nn.PReLU(out_features))
        self.fc_sequential = nn.Sequential(*fc_layers)
        if with_scale_layer:
            self.scale_layer = ScaleLayer(size=fc_sizes[-1])
        else:
            self.scale_layer = None

    def forward(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_sequential(x)
        if self.scale_layer is not None:
            x = self.scale_layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features=28**2, out_features=10, n_hidden=0):
        super().__init__()
        step = int((out_features - in_features) / (n_hidden + 1))
        assert step < 0, "Too much hidden layers"
        fc_sizes = list(range(in_features, out_features, step))
        fc_layers = []
        for (hidden_in, hidden_out) in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.Linear(hidden_in, hidden_out, bias=False))
            fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.Linear(in_features=fc_sizes[-1], out_features=out_features, bias=False))
        self.fc_sequential = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc_sequential(x)


def train_gradient(model: nn.Module = None, is_binary=True, dataset_name="MNIST", n_hidden=0):
    if model is None:
        # model = NetBinary(conv_channels=[], fc_sizes=linear_features[dataset_name])
        model = MLP(*linear_features[dataset_name], n_hidden=n_hidden)
    optimizer = AdamCustomDecay(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    if is_binary:
        model = binarize_model(model, keep_data=False)
        trainer_cls = TrainerGradBinary
    else:
        trainer_cls = TrainerGradFullPrecision
    trainer = trainer_cls(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name=dataset_name,
                          optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epoch=500, save=False, with_mutual_info=True)


def train_mcmc(model: nn.Module = None, dataset_name="MNIST", n_hidden=0):
    if model is None:
        model = MLP(*linear_features[dataset_name], n_hidden=n_hidden)
    model = binarize_model(model)
    trainer = TrainerMCMC(model,
                          criterion=nn.CrossEntropyLoss(),
                          dataset_name=dataset_name,
                          flip_ratio=0.1)
    trainer.train(n_epoch=500, save=False, with_mutual_info=False)


if __name__ == '__main__':
    # train_gradient(is_binary=False, dataset_name="MNIST56")
    train_gradient(dataset_name="MNIST56", is_binary=False, n_hidden=2)
    # train_mcmc(dataset_name="MNIST56")
    # train_mcmc(model=NetBinary(fc_sizes=[5**2, 10, 2]), dataset_name="MNIST56")
