import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from torch.autograd import Variable, Function


class BinaryFunc(Function):

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
        self.fc1 = linear_cls(n_input, n_hidden)
        self.fc2 = linear_cls(n_hidden, n_output)

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = self.fc1(x)
        if type(self.fc1) is nn.Linear:
            x = F.relu(x)
        x = self.fc2(x)
        return x


def get_data_loader(train=True, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def calc_accuracy(model, loader, n_batches=20):
    correct_count = 0
    total_count = 0
    mode_saved = model.training
    model.train(False)
    for batch_id, (images, labels) in enumerate(iter(loader)):
        total_count += len(labels)
        outputs = model(Variable(images))
        _, labels_predicted = torch.max(outputs.data, 1)
        correct_count += torch.sum(labels_predicted == labels)
        if batch_id > n_batches:
            break
    model.train(mode_saved)
    return correct_count / total_count


def get_softmax_accuracy(outputs, labels):
    _, labels_predicted = torch.max(outputs.data, 1)
    softmax_accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return softmax_accuracy


def train():
    train_loader = get_data_loader(train=True)
    test_loader = get_data_loader(train=False)

    model = MLP(n_input=784, n_hidden=500, linear_cls=BinaryLinear)
    model.train(True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)

    for epoch in range(5):
        for batch_id, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (batch_id + 1) % 100 == 0:
                batch_accuracy = get_softmax_accuracy(outputs, labels)
                test_accuracy = calc_accuracy(model, test_loader)
                print('Epoch [%d], Step [%d/%d], Loss: %.4f, accuracy: train=%.4f, test=%.4f' % (
                    epoch, batch_id, len(train_loader), loss.data[0], batch_accuracy, test_accuracy))


if __name__ == '__main__':
    train()
