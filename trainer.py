import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm

from constants import MODELS_DIR
from metrics import Metrics, calc_accuracy
from utils import get_data_loader


class Trainer(object):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, accuracy: float = None):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        model_path = os.path.join(MODELS_DIR, str(self.model) + '.pt')
        torch.save(self.model, model_path)
        msg = "Saved to {}".format(model_path)
        if accuracy is not None:
            msg += " (train accuracy: {:.4f})".format(accuracy)
        print(msg)

    def train(self, n_epoch=10, debug=False):
        print(repr(self.model))
        use_cuda = torch.cuda.is_available()
        train_loader = get_data_loader(train=True)
        if use_cuda:
            self.model.cuda()
        metrics = Metrics(self.model, train_loader)
        best_accuracy = metrics.load_best_accuracy(str(self.model), debug)
        dataset_name = type(train_loader.dataset).__name__
        print("Training '{}'. Best {} train accuracy so far: {:.4f}".format(
            str(self.model), dataset_name, best_accuracy))

        for epoch in range(n_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            for images, labels in tqdm(train_loader,
                                       desc="Epoch {:d}/{:d}".format(epoch, n_epoch),
                                       leave=False):
                images = Variable(images)
                labels = Variable(labels)
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step(closure=None)

                metrics.batch_finished(outputs, labels, loss)

                # todo: do we actually need weight clipping?
                for param in self.model.parameters_binary():
                    param.data.clamp_(min=-1, max=1)

            if not debug:
                accuracy = calc_accuracy(self.model, train_loader)
                metrics.update_train_accuracy(accuracy)
                if accuracy > best_accuracy:
                    self.save_model(accuracy)
                    best_accuracy = accuracy
