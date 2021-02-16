#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:16:29 2020

@author: thomas.topilko
"""
import logging
from decimal import Decimal

import numpy as np
import pandas as pd

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import photometry_io as io
from fiber_photometry_analysis.exceptions import FiberPhotometryDimensionError


def extract_row(data_frame, row_name):
    """
    Extract the row by name in the behaviour dataframe

    :param pd.DataFrame data_frame:
    :param str row_name:
    :return:
    """

    row = data_frame[data_frame[0] == row_name].values
    row_values = row[0][1:].astype(np.float64)
    row_values = row_values[row_values > 0]
    return row_values

def extract_column(data_frame, column_name):
    column_values = data_frame[column_name]
    return column_values[column_values > 0]


def extract_behavior_data(file_path, behaviour_name, transpose=True):
    """Extracts the raw behavior data from an excel file and cleans it.

    :param str file_path:  the path to the excel file
    :param str behaviour_name: The name of the behaviour type to be analysed
    :return:start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts
    """
    
    # df = pd.read_excel(file_path, header=None)  # FIXME: replace with rewrite binary dataframe
    df = pd.read_excel(file_path, header=None)
    if transpose:
        df = df.T
        column_names = df.iloc[0].to_list()
        df.columns = column_names
        df = df.drop(0)
        df = io.reset_dataframe_index(df)
        starts = extract_column(df, "tStart{0}".format(behaviour_name))
        ends = extract_column(df, "tEnd{0}".format(behaviour_name))
    else:
        starts = extract_row(df, "tStart{0}".format(behaviour_name))
        ends = extract_row(df, "tEnd{0}".format(behaviour_name))
    return starts, ends


def estimate_behavior_time_resolution(starts, ends):
    """Estimates the minimal time increment needed to fully capture the data.

    :param np.array starts: the starting timepoints for the behavioral bouts
    :param np.array ends:  the ending timepoints for the behavioral bouts
    :return:  minimal_resolution: the starting timepoints for the behavioral bouts
    :rtype: float
    """

    diff = ends - starts
    min_step = max(diff % 1)
    if min_step == 0:
        res = 1
    else:
        remaining = float(Decimal(str(min_step)) % Decimal("0.1"))
        if min_step - remaining == 0:
            res = 0.1
        else:
            res = min_step - remaining

    logging.info("The resolution for the behavioral is {}s".format(res))
    
    return 1/res


def estimate_length_bool_map(recording_duration, res):
    return int(recording_duration*res)


def match_time_behavior_to_photometry(starts, ends, recording_duration, video_duration, res):
    ratio = recording_duration / video_duration
    starts = np.round((starts*ratio).to_numpy().astype(float), 1) * res
    ends = np.round((ends*ratio).to_numpy().astype(float), 1) * res
    return starts, ends


def create_bool_map(bouts_positions, total_duration, sample_interval):
    """Creates a boolean map of the time during which the animal performs a given behavior.

    :param np.array bouts_positions: the starting/ending timepoints for the behavioral bouts
    :param int total_duration:
    :param float sample_interval:
    :return: bool_map = the boolean map of the time during which the animal performs the behavior
    :rtype: np.array
    """
    n_points = int(total_duration * sample_interval)
    bool_map = np.zeros(n_points)
    for bout in bouts_positions:  # TODO: check slices
        start_s, end_s = bout
        start_p = start_s * sample_interval
        end_p = end_s * sample_interval
        bool_map[start_p: end_p] = 1
    
    return bool_map.astype(bool)


def combine_ethograms(ethogerams):
    """
    Combine the ethograms so that each ethogram gets it order in the list as the value in the resulting
    combined ethogram

    :param [np.array, np.array, ] ethogerams: A list of ethograms as boolean arrays
    :return: combined ethogram
    :rtype: np.array
    """
    if min([eth.size for eth in ethogerams]) != max([eth.size for eth in ethogerams]):
        raise FiberPhotometryDimensionError('All ethograms must have same length')
    out = np.array(ethogerams[0].size, dtype=np.int)
    for i, ethogram in enumerate(ethogerams):
        tmp = ethogram * (i+1)  # Give it the value of it's order in the list
        out += tmp
    return out


def trim_behavioral_data(bool_map, crop_start, crop_end, res):
    """Trims the behavioral data to fit the pre-processed photometry data.
    It will remove the parts at the beginning and the end that correspond to the fluorescence
    settling time

    :param np.array mask: A boolean array
    :param dict params: dictionary with additional parameters
    :return: a trimmed mask
    """

    start = crop_start * res
    end = crop_end * res
    return bool_map[int(start) : -int(end)]


def extract_manual_bouts(starts, ends):  # FIXME: rename
    """Extracts the time and length of each behavioral bout.

    :param np.array starts: the starting timepoints for the behavioral bouts
    :param np.array ends: the ending timepoints for the behavioral bouts
    :return: position_bouts np.array = list of start and end of each behavioral bout
             length_bouts np.array = list of the length of each behavioral bout

    """
    position_bouts = np.column_stack(starts, ends)
    length_bouts = ends - starts
    return position_bouts, length_bouts


def check_starts_high(event_starts, event_ends):
    """
    Checks whether the first event is already ongoing

    :param np.array event_starts: the starting timepoints for the behavioral bouts
    :param np.array event_ends: the ending timepoints for the behavioral bouts
    :return:
    """
    return event_starts[0] < event_ends[0]


def check_ends_high(event_starts, event_ends):
    """
    Checks whether the last event is still ongoing

    :param np.array event_starts: the starting timepoints for the behavioral bouts
    :param np.array event_ends: the ending timepoints for the behavioral bouts
    :return:
    """
    return event_starts[-1] > event_ends[-1]


def get_up_durations(event_starts, event_ends, total_length):
    """
    Compute the duration of each event between starts and ends taking into account border cases
    (i.e. ongoing events at the beginning and end)

    :param np.array event_starts: the starting timepoints for the behavioral bouts
    :param np.array event_ends: the ending timepoints for the behavioral bouts
    :param int total_length: The size in points of the
    :return:
    """
    if check_starts_high(event_starts, event_ends):
        _event_ends = event_ends.copy()
    else:
        _event_ends = event_ends[1:]
    if check_ends_high(event_starts, event_ends):
        _event_ends = np.append(_event_ends, total_length)
    return _event_ends - event_starts


def get_down_durations(event_starts, event_ends, total_length):
    """
    Compute the duration of each inter-event (between ends and starts) taking into account border cases
    (i.e. ongoing events at the beginning and end)

    :param np.array event_starts: the starting timepoints for the behavioral bouts
    :param np.array event_ends: the ending timepoints for the behavioral bouts
    :param int total_length: The size in points of the
    :return:
    """
    if check_starts_high(event_starts, event_ends):
        _event_starts = event_starts[1:]
    else:
        _event_starts = event_starts.copy()
    if not check_ends_high(event_starts, event_ends):
        _event_starts = np.append(_event_starts, total_length)
    return _event_starts - event_ends


def extract_short_down_time_ranges(event_starts, event_ends, down_times, total_length, max_bout_gap):
    """
    Extract the ranges corresponding to inter-event of a duration below max_bout_gap

    :param np.array event_starts: the starting timepoints for the behavioral bouts
    :param np.array event_ends: the ending timepoints for the behavioral bouts
    :param np.array down_times: the list of (start, end) of each inter-event
    :param int total_length: The size of the source array
    :param int max_bout_gap: The maximum number of points between 2 events
    :return:
    """
    short_gaps_idx = np.where(down_times < max_bout_gap)[0]
    if check_starts_high(event_starts, event_ends):
        _event_starts = event_starts[1:]
    else:
        _event_starts = event_starts.copy()
    if not check_ends_high(event_starts, event_ends):
        _event_starts = np.append(_event_starts, total_length)
    short_down_time_ranges = np.column_stack((event_ends[short_gaps_idx],
                                              _event_starts[short_gaps_idx]))
    return short_down_time_ranges


def set_ranges_high(src_arr, ranges):
    """
    Set the ranges specified by ranges (pairs of start, end) to 1 in src_arr (does a copy)

    :param np.array src_arr: The Boolean array to modify
    :param np.array ranges: a list of ranges (start, end)
    :return:
    """
    out_arr = src_arr.copy()
    for s, e in ranges:
        out_arr[int(s):int(e)] = 1
    return out_arr


def detect_major_bouts(position_bouts, min_bout_length):
    """
    Algorithm that detects major behavioral bouts based on size. (Seeds are
    behavioral bouts that are too short to be considered a "major" event)

    :param np.array position_bouts:  list of merged start and end of each behavioral bout
    :param min_bout_length:
    :return: position_major_bouts (list) = list of start and end of each major behavioral bout
             length_major_bouts (list) = list of the length of each major behavioral bout
             position_seed_bouts (list) = list of start and end of each seed behavioral bout
             length_seed_bouts (list) = list of the length of each seed behavioral bout
    """
    event_starts = position_bouts[:, 0]
    event_ends = position_bouts[:, 1]

    up_times = event_ends - event_starts  # FIXME: Extract + depends on start low or high

    short_bouts_idx = np.where(up_times < min_bout_length)[0]  # FIXME: use function
    short_bout_ranges = np.column_stack((event_ends[short_bouts_idx],
                                         event_starts[short_bouts_idx]))
    short_bout_durations = short_bout_ranges[:, 1] - short_bout_ranges[:, 0]

    long_bouts_idx = np.where(up_times >= min_bout_length)[0]
    long_bout_ranges = np.column_stack((event_ends[long_bouts_idx],
                                        event_starts[long_bouts_idx]))
    long_bout_durations = long_bout_ranges[:, 1] - long_bout_ranges[:, 0]  # FIXME: start high ?
            
    return long_bout_ranges, long_bout_durations, short_bout_ranges, short_bout_durations


def extract_peri_event_photometry_data(position_bouts, **params):
    """
    Algorithm that extracts the photometry data at the moment when the animal behaves.
    
    :param np.array position_bouts:  list of start and end of each major behavioral bout
    :param dict params: A dictionary of parameters
    :return: df_around_peaks (list) = list of photometry data at the moment when the animal behaves
    """
    df_around_peaks = []
    sample_rate = params["recording_sampling_rate"]
    n_pnts_pre = params["peri_event"]["graph_distance_pre"] * sample_rate
    n_pnts_post = (params["peri_event"]["graph_distance_post"] * sample_rate) + 1  # Add 1 pnt
    delta_f = params["photometry_data"]["dFF"]["dFF"]
    for bout in position_bouts:
        event_start = bout[0] * sample_rate
        start_idx = int(event_start - n_pnts_pre)
        end_idx = int(event_start + n_pnts_post)
        df_around_peaks.append(delta_f[start_idx:end_idx])

    return df_around_peaks


def reorder_by_bout_size(delta_f_around_peaks, length_bouts):
    """
    Simple algorithm that re-orders the peri-event photometry data based on the size of the behavioral event.

    :param np.array delta_f_around_peaks:  photometry data at the moment when the animal behaves
    :param np.array length_bouts: list of the length of each behavioral bout
    :return: dd_around_peaks_ordered (list) = list of ordered photometry data at the moment when the animal behaves
             length_bouts_ordered (list) = list of the length of each behavioral bout
    """
    order = np.argsort( -np.array(length_bouts))
    
    dd_around_peaks_ordered = np.array(delta_f_around_peaks)[order]
    length_bouts_ordered = np.array(length_bouts)[order]
    
    return dd_around_peaks_ordered, length_bouts_ordered

def get_position_from_bool_map(bool_map, res):
    return np.where(bool_map)[0]/res
