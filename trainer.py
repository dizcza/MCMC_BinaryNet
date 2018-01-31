import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm

from constants import MODELS_DIR
from metrics import Metrics, calc_accuracy
from utils import get_data_loader, parameters_binary, load_model


class Trainer(object):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataset: str,
                 scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.scheduler = scheduler

    def save_model(self, accuracy: float = None):
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR.joinpath(self.dataset, self.model.__class__.__name__).with_suffix('.pt')
        model_path.parent.mkdir(exist_ok=True)
        torch.save(self.model, model_path)
        msg = f"Saved to {model_path}"
        if accuracy is not None:
            msg += f" (train accuracy: {accuracy:.4f})"
        print(msg)

    def load_best_accuracy(self, debug=False) -> float:
        train_loader = get_data_loader(self.dataset, train=True)
        best_accuracy = 0.
        if not debug:
            try:
                loaded_model = load_model(self.dataset, self.model.__class__.__name__)
                best_accuracy = calc_accuracy(loaded_model, train_loader)
            except Exception:
                print(f"Couldn't estimate the best accuracy for {self.model}. Reset to 0.")
        return best_accuracy

    def train(self, n_epoch=10, debug=False):
        print(self.model)
        use_cuda = torch.cuda.is_available()
        train_loader = get_data_loader(self.dataset, train=True)
        if use_cuda:
            self.model.cuda()
        metrics = Metrics(self.model, train_loader, monitor_sign='all')
        best_accuracy = self.load_best_accuracy(debug)
        metrics.log(f"Best train accuracy so far: {best_accuracy:.4f}")
        dataset_name = type(train_loader.dataset).__name__
        print(f"Training '{self.model}'. Best {dataset_name} train accuracy so far: {best_accuracy:.4f}")

        for epoch in range(n_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            metrics.log(f"Epoch {epoch}. Learning rate {self.scheduler.get_lr()}")
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

                for param in parameters_binary(self.model):
                    param.data.clamp_(min=-1, max=1)

            if not debug:
                accuracy = calc_accuracy(self.model, train_loader)
                is_best = accuracy > best_accuracy
                metrics.update_train_accuracy(accuracy, is_best)
                if is_best:
                    self.save_model(accuracy)
                    best_accuracy = accuracy
