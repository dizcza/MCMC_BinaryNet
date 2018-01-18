import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data

from utils import StepLRClamp, Trainer, test


class BinaryFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.sign()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        return grad_output


class BinaryLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        linear_output = super().forward(input)
        binary_output = BinaryFunc.apply(linear_output)
        return binary_output


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output=10, linear_cls=nn.Linear):
        super().__init__()
        self.n_input = n_input
        self.linear_cls = linear_cls
        self.fc1 = linear_cls(n_input, n_hidden)
        self.fc2 = linear_cls(n_hidden, n_output)

    def __str__(self):
        return self.linear_cls.__name__

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = self.fc1(x)
        if type(self.fc1) is nn.Linear:
            x = F.relu(x)
        x = self.fc2(x)
        return x


def train(linear_cls=BinaryLinear, n_epoch=10):
    model = MLP(n_input=784, n_hidden=500, linear_cls=linear_cls)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    scheduler = StepLRClamp(optimizer, step_size=2, gamma=0.1, min_lr=1e-4)
    trainer = Trainer(model, criterion, optimizer, scheduler)
    trainer.train(n_epoch)


if __name__ == '__main__':
    train(linear_cls=BinaryLinear)
    # train(linear_cls=nn.Linear)
    test()
