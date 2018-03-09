from typing import Callable
from abc import ABC, abstractmethod


class BatchTimer(object):

    def __init__(self, batches_in_epoch: int):
        self.batches_in_epoch = batches_in_epoch
        self.batch_id = 0

    def schedule(self, func: Callable, epoch_update: int = 1, batch_update: int = 0):
        """
        :param func: monitored function with void return
        :param epoch_update: +epochs between updates
        :param batch_update: +batches between updates with fixed epoch
        :return: decorated `func`
        """
        batch_update = epoch_update * self.batches_in_epoch + batch_update
        assert batch_update > 0, "At least 1 batch step"
        next_batch = batch_update

        def wrapped(*args, **kwargs):
            nonlocal next_batch
            if self.batch_id >= next_batch:
                func(*args, **kwargs)
                next_batch += batch_update

        return wrapped

    @property
    def epoch(self):
        return int(self.epoch_progress())

    def epoch_progress(self):
        return self.batch_id / self.batches_in_epoch

    def tick(self):
        self.batch_id += 1


class Schedulable(ABC):

    @abstractmethod
    def schedule(self, timer: BatchTimer, epoch_update: int = 1):
        """
        :param timer: timer to schedule updates
        :param epoch_update: epochs between updates
        """
        raise NotImplementedError()
