import os
import torch
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms, datasets

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_bin")


def get_data_loader(train=True, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def calc_accuracy(model, loader):
    if model is None:
        return 0.0
    correct_count = 0
    total_count = 0
    mode_saved = model.training
    model.train(False)
    for batch_id, (images, labels) in enumerate(iter(loader)):
        total_count += len(labels)
        outputs = model(Variable(images))
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

    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def load_model(model_name):
        model_path = os.path.join(MODELS_DIR, model_name + '.pt')
        if os.path.exists(model_path):
            return torch.load(model_path)
        else:
            return None

    def save_model(self):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        model_path = os.path.join(MODELS_DIR, str(self.model) + '.pt')
        torch.save(self.model, model_path)
        print("Saved to {}".format(model_path))

    def train(self, n_epoch=10):
        train_loader = get_data_loader(train=True)
        log_step = len(train_loader) // 10
        loaded_model = self.load_model(str(self.model))
        best_accuracy = calc_accuracy(loaded_model, train_loader)
        dataset_name = type(train_loader.dataset).__name__
        print("Training '{}'. Best {} train accuracy so far: {:.4f}".format(
            str(self.model), dataset_name, best_accuracy))
        for epoch in range(n_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            for batch_id, (images, labels) in enumerate(train_loader):
                images = Variable(images)
                labels = Variable(labels)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if batch_id % log_step == 0:
                    batch_accuracy = get_softmax_accuracy(outputs, labels)
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Batch accuracy: %.4f' % (
                        epoch, n_epoch, batch_id, len(train_loader), loss.data[0], batch_accuracy))
            accuracy = calc_accuracy(self.model, train_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model()


def test():
    print("Test accuracy:")
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        test_loader = get_data_loader(train=False)
        model = torch.load(model_path)
        accur = calc_accuracy(model, test_loader)
        print("\t{}: {:.4f}".format(str(model), accur))
