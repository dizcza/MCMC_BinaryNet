from typing import Callable
from abc import ABC, abstractmethod


class BatchTimer(object):

    def __init__(self, batches_in_epoch: int):
        self.batches_in_epoch = batches_in_epoch
        self.batch_id = 0

    def schedule(self, func: Callable, epoch_update: int):
        """
        :param func: monitored function with void return
        :param epoch_update: epochs between updates
        :return: decorated `func`
        """
        assert epoch_update > 0, "At least 1 epoch step"
        next_update = epoch_update

        def wrapped(*args, **kwargs):
            nonlocal next_update
            if self.epoch >= next_update:
                func(*args, **kwargs)
                next_update += epoch_update

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
