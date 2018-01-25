import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms, datasets
from typing import Union

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_bin")


def get_data_loader(train=True):
    if torch.cuda.is_available():
        batch_size = 256
    else:
        batch_size = 16
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def calc_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader):
    if model is None:
        return 0.0
    correct_count = 0
    total_count = 0
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    for batch_id, (images, labels) in enumerate(iter(loader)):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        total_count += len(labels)
        outputs = model(Variable(images, volatile=True))
        _, labels_predicted = torch.max(outputs.data, 1)
        correct_count += torch.sum(labels_predicted == labels)
    model.train(mode_saved)
    return correct_count / total_count


def get_softmax_accuracy(outputs, labels):
    _, labels_predicted = torch.max(outputs.data, 1)
    softmax_accuracy = torch.sum(labels.data == labels_predicted) / len(labels)
    return softmax_accuracy


class StepLRClamp(torch.optim.lr_scheduler.StepLR):

    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-4):
        self.min_lr = min_lr
        super().__init__(optimizer, step_size, gamma, last_epoch=-1)

    def get_lr(self):
        learning_rates = super().get_lr()
        learning_rates = [max(lr, self.min_lr) for lr in learning_rates]
        return learning_rates


class Trainer(object):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def load_model(model_name) -> Union[nn.Module, None]:
        model_path = os.path.join(MODELS_DIR, model_name + '.pt')
        if not os.path.exists(model_path):
            return None
        return torch.load(model_path)

    def save_model(self):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        model_path = os.path.join(MODELS_DIR, str(self.model) + '.pt')
        torch.save(self.model, model_path)
        print("Saved to {}".format(model_path))

    def optimizer_step_analyze(self) -> int:
        param_sign_before = {}
        for name, param in self.model.named_parameters_binary():
            param_sign_before[name] = param.data.clone()
        self.optimizer.step(closure=None)
        sign_flips = 0
        # for name, param in self.model.named_parameters_binary():
        #     print(name, float(param.grad.min()), float(param.grad.max()), float(param.data.min()), float(param.data.max()))
        for name, param in self.model.named_parameters_binary():
            sign_flips += torch.sum((param.data * param_sign_before[name]) < 0)
        return sign_flips

    def load_best_accuracy(self, debug: bool):
        train_loader = get_data_loader(train=True)
        best_accuracy = 0.
        if not debug:
            try:
                loaded_model = self.load_model(str(self.model))
                best_accuracy = calc_accuracy(loaded_model, train_loader)
            except Exception:
                print("Couldn't estimate the best accuracy for {}. Rest to 0.".format(self.model))
        return best_accuracy

    def train(self, n_epoch=10, debug=False):
        train_loader = get_data_loader(train=True)
        log_step = len(train_loader) // 10
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        best_accuracy = self.load_best_accuracy(debug)
        dataset_name = type(train_loader.dataset).__name__
        print("Training '{}'. Best {} train accuracy so far: {:.4f}".format(
            str(self.model), dataset_name, best_accuracy))

        total_binary_params = 0
        for param in self.model.parameters_binary():
            total_binary_params += param.numel()

        for epoch in range(n_epoch):
            sign_flips = 0
            if self.scheduler is not None:
                self.scheduler.step()
            for batch_id, (images, labels) in enumerate(train_loader, start=1):
                images = Variable(images)
                labels = Variable(labels)
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                sign_flips += self.optimizer_step_analyze()

                # todo: do we actually need weight clipping?
                for param in self.model.parameters_binary():
                    param.data.clamp_(min=-1, max=1)

                if batch_id % log_step == 0:
                    batch_accuracy = get_softmax_accuracy(outputs, labels)
                    sign_flips_relative = sign_flips / (batch_id * total_binary_params)
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Batch accuracy: %.4f, Sign flips: %.2e' % (
                           epoch, n_epoch, batch_id, len(train_loader), loss.data[0], batch_accuracy,
                           sign_flips_relative))
            if not debug:
                accuracy = calc_accuracy(self.model, train_loader)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save_model()


def test():
    print("Test accuracy:")
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        test_loader = get_data_loader(train=False)
        try:
            model = torch.load(model_path)
            accur = calc_accuracy(model, test_loader)
            print("\t{}: {:.4f}".format(model, accur))
        except Exception:
            print("Skipped evaluating {} model".format(model_path))
