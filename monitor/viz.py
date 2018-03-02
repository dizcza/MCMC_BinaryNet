import time
from typing import Union, List

import numpy as np
import visdom

from monitor.batch_timer import BatchTimer


class VisdomMighty(visdom.Visdom):
    def __init__(self, env: str, timer: BatchTimer):
        super().__init__(env=env)
        self.timer = timer

    def line_update(self, y: Union[float, List[float]], win: str, opts: dict):
        y = np.array([y])
        size = y.shape[-1]
        if size == 0:
            return
        if y.ndim > 1 and size == 1:
            y = y[0]
        x = np.full_like(y, self.timer.epoch_progress())
        self.line(Y=y, X=x, win=win, opts=opts, update='append' if self.win_exists(win) else None)

    def log(self, text: str):
        self.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}", win='log', append=self.win_exists(win='log'))
