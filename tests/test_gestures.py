import numpy as np

from stretch_demo.tracking.gestures import DoubleShrug, FlickDetector, FlickSettings
from stretch_demo.tracking.mobility import GestureSettings

D = np.deg2rad
FLICK = FlickSettings(fire=D(300.0), hold_s=0.04, dominance=2.0)


def run_flick(detector, rates_dps, dt=0.01, t0=0.0):
    """Feed (x, y, z) deg/s samples; return the non-zero outputs."""
    fired = []
    for k, rates in enumerate(rates_dps):
        out = detector.update(D(np.array(rates)), t0 + k * dt)
        if out:
            fired.append(out)
    return fired


def test_flick_fires_once_on_a_sustained_fast_yaw():
    swing = [(0, 0, 400)] * 6 + [(0, 0, -100)] * 10 + [(0, 0, 0)] * 5
    assert run_flick(FlickDetector(2, 1.0, FLICK), swing) == [1.0]


def test_flick_ignores_a_one_sample_spike():
    spike = [(0, 0, 0), (0, 0, 900), (0, 0, 0), (0, 0, 0)]
    assert run_flick(FlickDetector(2, 1.0, FLICK), spike) == []


def test_flick_ignores_fast_motion_dominated_by_another_axis():
    tilt = [(300, 0, 400)] * 8
    assert run_flick(FlickDetector(2, 1.0, FLICK), tilt) == []


def test_flick_must_rearm_before_firing_again():
    held = [(0, 0, 400)] * 60  # never drops below rearm
    assert run_flick(FlickDetector(2, 1.0, FLICK), held) == [1.0]


def shrug_sequence(peaks_dps, gap_s=0.4, dt=0.01, swing_s=0.08):
    """Upward swings at the given peak rates, separated by rest."""
    samples = []
    for peak in peaks_dps:
        samples += [peak] * int(swing_s / dt) + [-30.0] * 5 + [0.0] * int(gap_s / dt)
    return samples


def run_shrugs(samples, other_dps=0.0, dt=0.01):
    g, shrugs, events = GestureSettings(), DoubleShrug(), []
    for k, dps in enumerate(samples):
        event = shrugs.update(D(dps), D(other_dps), g, k * dt)
        if event:
            events.append(event)
    return events, shrugs


def test_double_shrug_is_recognized():
    events, _ = run_shrugs(shrug_sequence([90.0, 90.0]))
    assert events == ["first", "double"]


def test_single_shrug_is_a_miss_with_a_reason():
    events, shrugs = run_shrugs(shrug_sequence([90.0], gap_s=1.5))
    assert events == ["first"]
    assert "only one quick shrug" in shrugs.miss


def test_slow_shrug_explains_it_was_not_fast_enough():
    events, shrugs = run_shrugs(shrug_sequence([45.0]))
    assert events == []
    assert "not fast enough" in shrugs.miss


def test_shrug_mixed_with_other_motion_does_not_count():
    events, shrugs = run_shrugs(shrug_sequence([90.0, 90.0]), other_dps=80.0)
    assert events == []
    assert "mixed with other motion" in shrugs.miss


def test_two_shrugs_too_far_apart_are_two_firsts():
    events, _ = run_shrugs(shrug_sequence([90.0, 90.0], gap_s=1.3))
    assert events == ["first", "first"]
