#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:16:29 2020

@author: thomas.topilko
"""

import numpy as np
import pandas as pd

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis.exceptions import FiberPhotometryDimensionError


def extract_behavior_data(file, **kwargs):
    """Extracts the raw behavior data from an excel file and cleans it.
    
    Args :  file (str) = the path to the excel file
            kwargs (dict) = dictionary with additional parameters

    Returns : start (arr) = the starting timepoints for the behavioral bouts
              end (arr) = the ending timepoints for the behavioral bouts
    """
    
    f = pd.read_excel(file, header=None)  # FIXME: replace with rewrite binary dataframe
    
    start_pos = np.where(f[0] == "tStart{0}".format(kwargs["behavior_to_segment"]))[0]
    end_pos = np.where(f[0] == "tEnd{0}".format(kwargs["behavior_to_segment"]))[0]
    
    start = f.iloc[start_pos[0]][1:][f.iloc[start_pos[0]][1:] > 0]
    end = f.iloc[end_pos[0]][1:][f.iloc[end_pos[0]][1:] > 0]
    
    return start, end


def estimate_minimal_resolution(start, end):
    """Estimates the minimal time increment needed to fully capture the data.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts

    Returns : minimal_resolution (float) = the starting timepoints for the behavioral bouts
    """
    
    minimal_resolution = 1 / min(end - start)
    
    utils.print_in_color("The smallest behavioral bout is {0}s long".format(1 / minimal_resolution), "GREEN")
    
    return minimal_resolution


def create_bool_map(bouts_positions, total_duration, **kwargs):
    """Creates a boolean map of the time during which the animal performs a given behavior.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts

    Returns : bool_map (arr) = the boolean map of the time during which the animal performs the behavior
    """
    sample_interval = 1 / kwargs["recording_sampling_rate"]
    n_points = total_duration * sample_interval
    bool_map = np.zeros(n_points)
    for bout in bouts_positions:  # TODO: check slices
        start_s, end_s = bout
        start_p = start_s * sample_interval
        end_p = end_s * sample_interval
        bool_map[start_p: end_p] = 1

    utils.print_in_color("\nBehavioral data extracted. Behavior = {0}".format(kwargs["behavior_to_segment"]), "GREEN")
    
    return bool_map.astype(bool)


def combine_ethograms(ethogerams):
    """
    """
    if min([eth.size for eth in ethogerams]) != max([eth.size for eth in ethogerams]):
        raise FiberPhotometryDimensionError('All ethograms must have same length')
    out = np.array(ethogerams[0].size, dtype=np.int)
    for i, ethogram in enumerate(ethogerams):
        tmp = ethogram * (i+1)  # Give it the value of it's order in the list
        out += tmp
    return out


def trim_behavioral_data(bool_map, **kwargs):
    """Trims the behavioral data to fit the pre-processed photometry data.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts
            kwargs (dict) = dictionary with additional parameters

    Returns : start (arr) = trimmed list of start and end of each behavioral bout
              end (arr) = trimmed list of the length of each behavioral bout
    """

    # The duration that is cropped from the photometry data because of initial fluo drop
    sampling = 1 / kwargs["recording_sampling_rate"]
    time_lost_start_in_points = int(kwargs["photometry_data"]["start_time_lost"] * sampling)
    time_lost_end_in_points = int(kwargs["photometry_data"]["start_time_lost"] * sampling)
    return bool_map[time_lost_start_in_points:-time_lost_end_in_points]


def extract_manual_bouts(start, end, **kwargs):
    """Extracts the time and length of each behavioral bout.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts
            kwargs (dict) = dictionary with additional parameters

    Returns : position_bouts (list) = list of start and end of each behavioral bout
              length_bouts (list) = list of the length of each behavioral bout
    """
    position_bouts = np.column_stack(start, end)
    length_bouts = end - start
      
    print("\n")
    utils.print_in_color("Behavioral data extracted. Behavior = {0}".format(kwargs["behavior_to_segment"]), "GREEN")
    
    return position_bouts, length_bouts


def merge_neighboring_bouts(position_bouts, **kwargs):
    """Algorithm that merges behavioral bouts that are close together.
    
    Args :  position_bouts (arr) = list of start and end of each behavioral bout
            kwargs (dict) = dictionary with additional parameters

    Returns : position_bouts_merged (list) = list of merged start and end of each behavioral bout
              length_bouts_merged (list) = list of the length of each merged behavioral bout
    """
    position_bouts_merged = position_bouts.copy()  # FIXME: check point/time
    event_starts = position_bouts[:, 0]
    event_ends = position_bouts[:, 1]
    max_bout_gap = kwargs["peak_merging_distance"]  # FIXME: check point/time

    down_times = event_starts - event_ends  # FIXME: depends on start low or high
    short_gaps_idx = np.where(down_times < max_bout_gap)[0]
    short_down_time_ranges = np.column_stack((event_ends[short_gaps_idx],
                                              event_starts[short_gaps_idx]))  # FIXME: depends of low or high start
    position_bouts_merged[np.r_[short_down_time_ranges]] = 1
    return position_bouts_merged


def detect_major_bouts(position_bouts, **kwargs):
    """Algorithm that detects major behavioral bouts based on size. (Seeds are
    behavioral bouts that are too short to be considered a "major" event)
    
    Args :  position_bouts (arr) = list of merged start and end of each behavioral bout
            kwargs (dict) = dictionary with additional parameters

    Returns : position_major_bouts (list) = list of start and end of each major behavioral bout
              length_major_bouts (list) = list of the length of each major behavioral bout
              position_seed_bouts (list) = list of start and end of each seed behavioral bout
              length_seed_bouts (list) = list of the length of each seed behavioral bout
    """
    event_starts = position_bouts[:, 0]
    event_ends = position_bouts[:, 1]
    min_bout_length = kwargs["minimal_bout_length"]

    up_times = event_ends - event_starts  # FIXME: depends on start low or high

    short_bouts_idx = np.where(up_times < min_bout_length)[0]
    short_bout_ranges = np.column_stack((event_ends[short_bouts_idx],
                                         event_starts[short_bouts_idx]))
    short_bout_durations = short_bout_ranges[:, 1] - short_bout_ranges[:, 0]

    long_bouts_idx = np.where(up_times >= min_bout_length)[0]
    long_bout_ranges = np.column_stack((event_ends[long_bouts_idx],
                                        event_starts[long_bouts_idx]))
    long_bout_durations = long_bout_ranges[:, 1] - long_bout_ranges[:, 0]
            
    return long_bout_ranges, long_bout_durations, short_bout_ranges, short_bout_durations


def extract_peri_event_photmetry_data(position_bouts, **kwargs):
    """Algorithm that extracts the photometry data at the moment when the animal behaves.

    Args :  position_bouts (arr) = list of start and end of each major behavioral bout
            kwargs (dict) = dictionary with additional parameters

    Returns : df_around_peaks (list) = list of photometry data at the moment when the animal behaves
    """
    df_around_peaks = []
    
    for bout in position_bouts:
        segment = kwargs["photometry_data"]["dFF"]["dFF"][int((bout[0]*kwargs["recording_sampling_rate"]) -
                                                              (kwargs["peri_event"]["graph_distance_pre"] * kwargs["recording_sampling_rate"])):
                                                          int((bout[0]*kwargs["recording_sampling_rate"]) + ((kwargs["peri_event"]["graph_distance_post"]+1) * kwargs["recording_sampling_rate"]))]
        
        df_around_peaks.append(segment)
        
    return df_around_peaks


def reorder_by_bout_size(delta_f_around_peaks, length_bouts):
    """Simple algorithm that re-orders the peri-event photometry data based on the size of the behavioral event.
    
    Args :  dF_around_peaks (arr) = list of photometry data at the moment when the animal behaves
            length_bouts (list) = list of the length of each behavioral bout

    Returns : dd_around_peaks_ordered (list) = list of ordered photometry data at the moment when the animal behaves
              length_bouts_ordered (list) = list of the length of each behavioral bout
    """
    order = np.argsort(-np.array(length_bouts))
    
    dd_around_peaks_ordered = np.array(delta_f_around_peaks)[order]
    length_bouts_ordered = np.array(length_bouts)[order]
    
    return dd_around_peaks_ordered, length_bouts_ordered
