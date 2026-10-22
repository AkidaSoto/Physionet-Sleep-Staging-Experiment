import numpy as np

from physionet_sleep.algorithms.topography import slow_wave_coverage
from physionet_sleep.labels.cinc2018 import stage_epochs


def test_stage_epochs_hold_until_next_label_and_leave_gaps_unscored():
    changes = [(60.0, "W"), (120.0, "N1"), (180.0, "N2"), (240.0, "R"), (300.0, "?")]
    codes = stage_epochs(changes, length_seconds=360.0)
    assert codes.tolist() == [-1, -1, 0, 0, 1, 1, 2, 2, 4, 4, -1, -1]


def test_slow_wave_coverage_uses_the_75_microvolt_rule():
    fs = 200.0
    t = np.arange(0, 30, 1 / fs)
    big = 50.0 * np.sin(2 * np.pi * 1.0 * t)    # 100 uV peak-to-peak, 1 Hz
    small = 25.0 * np.sin(2 * np.pi * 1.0 * t)  # 50 uV peak-to-peak
    fast = 50.0 * np.sin(2 * np.pi * 6.0 * t)   # large but outside 0.5-2 Hz
    assert slow_wave_coverage(big, fs).mean() > 0.8
    assert slow_wave_coverage(small, fs).mean() == 0.0
    assert slow_wave_coverage(fast, fs).mean() < 0.05
