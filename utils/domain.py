from enum import Enum, unique


@unique
class MonitorLevel(Enum):
    DISABLED = 0
    SIGNAL_TO_NOISE = 1  # signal-to-noise ratio
    FULL = 2  # sign flips, initial point difference
