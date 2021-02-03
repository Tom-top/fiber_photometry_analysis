import random

import pytest

import numpy as np

from fiber_photometry_analysis.behavior_preprocessing import create_bool_map


@pytest.fixtute
def bool_arr():
    return np.zeros(1000)


@pytest.fixture
def starts():
    n_bouts = 50
    max_bout_size = 20
    starts = np.random.randint(1000, size=n_bouts)
    ends = starts.copy()
    for i, start in enumerate(starts):
        ends[i] = start + random.randint(0, max_bout_size)
    bool_arr[np.r_(np.column_stack((starts, ends)))] = 1

# def test_extract_behavior_data():
# def test_estimate_minimal_resolution(start, end):


def test_create_bool_map():
    # assert create_bool_map(bouts_positions, total_duration) == bool_map
    pass


def test_combine_ethograms():
    pass

# def test_trim_behavioral_data(bool_map, **kwargs):
# def test_extract_manual_bouts(start, end, **kwargs):


def test_merge_neighboring_bouts():
    pass

# def test_detect_major_bouts(position_bouts, **kwargs):
# def test_extract_peri_event_photmetry_data(position_bouts, **kwargs):
# def test_reorder_by_bout_size(delta_f_around_peaks, length_bouts):
