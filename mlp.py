import os

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from torch.autograd import Variable

from utils import StepLRClamp


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_bin")


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

    def save_model(self):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        model_path = os.path.join(MODELS_DIR, self.linear_cls.__name__ + '.pt')
        torch.save(self, model_path)
        print("Saved to {}".format(model_path))

    def __str__(self):
        return self.linear_cls.__name__

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = self.fc1(x)
        if type(self.fc1) is nn.Linear:
            x = F.relu(x)
        x = self.fc2(x)
        return x


def get_data_loader(train=True, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def calc_accuracy(model, loader, n_batches=-1):
    correct_count = 0
    total_count = 0
    mode_saved = model.training
    model.train(False)
    for batch_id, (images, labels) in enumerate(iter(loader)):
        total_count += len(labels)
        outputs = model(Variable(images))
        _, labels_predicted = torch.max(outputs.data, 1)
        correct_count += torch.sum(labels_predicted == labels)
        if n_batches > 0 and batch_id > n_batches:
            break
    model.train(mode_saved)
    return correct_count / total_count


def get_softmax_accuracy(outputs, labels):
    _, labels_predicted = torch.max(outputs.data, 1)
    softmax_accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return softmax_accuracy


def train(linear_cls=BinaryLinear, n_epoch=10):
    train_loader = get_data_loader(train=True)
    test_loader = get_data_loader(train=False)

    model = MLP(n_input=784, n_hidden=500, linear_cls=linear_cls)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    scheduler = StepLRClamp(optimizer, step_size=2, gamma=0.1, min_lr=1e-4)
    log_step = len(train_loader) // 10

    print("Training {}".format(str(model)))
    for epoch in range(n_epoch):
        scheduler.step()
        for batch_id, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_id % log_step == 0:
                batch_accuracy = get_softmax_accuracy(outputs, labels)
                test_accuracy = calc_accuracy(model, test_loader, n_batches=20)
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, accuracy: train=%.4f, test=%.4f' % (
                    epoch, n_epoch, batch_id, len(train_loader), loss.data[0], batch_accuracy, test_accuracy))
    model.save_model()


def test():
    print("Test accuracy:")
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        test_loader = get_data_loader(train=False)
        model = torch.load(model_path)
        accur = calc_accuracy(model, test_loader)
        print("\t{}: {:.4f}".format(str(model), accur))


if __name__ == '__main__':
    train(linear_cls=BinaryLinear)
    train(linear_cls=nn.Linear)
    test()
