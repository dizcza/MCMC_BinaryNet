class BatchTimer(object):

    def __init__(self, batches_in_epoch: int):
        self.batches_in_epoch = batches_in_epoch
        self.batch_id = 0
        self.max_skip = batches_in_epoch
        self.next_update = 10

    def need_update(self):
        if self.batch_id >= self.next_update:
            self.next_update = min(int((self.batch_id + 1) ** 1.1), self.batch_id + self.max_skip)
            return True
        return False

    def need_epoch_update(self, epoch_update):
        return int(self.epoch_progress()) % epoch_update == 0

    def epoch_progress(self):
        return self.batch_id / self.batches_in_epoch

    def tick(self):
        self.batch_id += 1
