#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:16:29 2020

@author: thomas.topilko
"""

import numpy as np
import pandas as pd
from functools import reduce

import Utilities as ut

def extract_behavior_data(file, **kwargs) :
    
    """Extracts the raw behavior data from an excel file and cleans it.
    
    Args :  file (str) = the path to the excel file
            kwargs (dict) = dictionnary with additional parameters

    Returns : start (arr) = the starting timepoints for the behavioral bouts
              end (arr) = the ending timepoints for the behavioral bouts
    """
    
    f = pd.read_excel(file, header=None)
    
    start_pos = np.where(f[0] == "tStart{0}".format(kwargs["behavior_to_segment"]))[0]
    end_pos = np.where(f[0] == "tEnd{0}".format(kwargs["behavior_to_segment"]))[0]
    
    start = f.iloc[start_pos[0]][1:][f.iloc[start_pos[0]][1:] > 0]
    end = f.iloc[end_pos[0]][1:][f.iloc[end_pos[0]][1:] > 0]
    
    return start, end

def estimate_minimal_resolution(start, end) : 
    
    """Estimates the minimal time increment needed to fully capture the data.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts

    Returns : minimal_resolution (float) = the starting timepoints for the behavioral bouts
    """
    
    minimal_resolution = 1/min(end-start)
    
    ut.print_in_color("The smallest behavioral bout is {0}s long".format(1/minimal_resolution), "GREEN")
    
    return minimal_resolution

def create_bool_map(start, end, **kwargs) :
    
    """Creates a boolean map of the time during which the animal performs a given behavior.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts

    Returns : bool_map (arr) = the boolean map of the time during which the animal performs the behavior
    """
    
    bool_map = []

    for s, e in zip(start, end) :
        
        for i in np.arange(len(bool_map)/kwargs["recording_sampling_rate"], s, 1/kwargs["recording_sampling_rate"]) :
            
            bool_map.append(False)
        
        for i in np.arange(s, e, 1/kwargs["recording_sampling_rate"]) :
            
            bool_map.append(True)

    for i in np.arange(i+(1/kwargs["recording_sampling_rate"]), round(kwargs["video_end"]), 1/kwargs["recording_sampling_rate"]) :
        
        bool_map.append(False)
        
    if i != round(kwargs["video_end"]) :
        
        if not bool_map[-1] :
        
            bool_map.append(False)
        
        else :
            
            bool_map.append(True)
      
    print("\n")
    ut.print_in_color("Behavioral data extracted. Behavior = {0}".format(kwargs["behavior_to_segment"]), "GREEN")
    
    return bool_map

def trim_behavioral_data(start, end, **kwargs) : 
    
    """Trims the behavioral data to fit the pre-processed photometry data.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts
            kwargs (dict) = dictionnary with additional parameters

    Returns : start (arr) = trimmed list of start and end of each behavioral bout
              end (arr) = trimmed list of the length of each behavioral bout
    """
    
    half_time_lost = int(kwargs["photometry_data"]["time_lost"]/2)
    
    masks = [start > half_time_lost, end > half_time_lost,\
             start < kwargs["video_end"]-half_time_lost, end < kwargs["video_end"]-half_time_lost]
    total_mask = reduce(np.logical_and, masks)
    
    start = start[total_mask] - half_time_lost
    end = end[total_mask] - half_time_lost
    
    return start, end

def extract_manual_bouts(start, end, **kwargs) :
    
    """Extracts the time and length of each behavioral bout.
    
    Args :  start (arr) = the starting timepoints for the behavioral bouts
            end (arr) = the ending timepoints for the behavioral bouts
            kwargs (dict) = dictionnary with additional parameters

    Returns : position_bouts (list) = list of start and end of each behavioral bout
              length_bouts (list) = list of the length of each behavioral bout
    """
    
    position_bouts = []
    length_bouts = []
    
    for s, e in zip(start, end) :
            
        position_bouts.append((s, e))
        length_bouts.append(e-s)
      
    print("\n")
    ut.print_in_color("Behavioral data extracted. Behavior = {0}".format(kwargs["behavior_to_segment"]), "GREEN")
    
    return position_bouts, length_bouts

def merge_neighboring_bouts(position_bouts, **kwargs) :
    
    """Algorithm that merges behavioral bouts that are close together.
    
    Args :  position_bouts (arr) = list of start and end of each behavioral bout
            kwargs (dict) = dictionnary with additional parameters

    Returns : position_bouts_merged (list) = list of merged start and end of each behavioral bout
              length_bouts_merged (list) = list of the length of each merged behavioral bout
    """
    
    position_bouts_merged = []
    length_bouts_merged = []

    merged = False
    
    for i in np.arange(0, len(position_bouts)-1, 1) :
            
        if not merged :
            
            cur_bout = position_bouts[i]
        
        next_bout = position_bouts[i+1]
        
        if next_bout[0] - cur_bout[1] <= kwargs["peak_merging_distance"] :
            
            merged = True
            cur_bout = (cur_bout[0], next_bout[1])
            
            if i == len(position_bouts)-2 :
                
                position_bouts_merged.append(cur_bout)
                length_bouts_merged.append(cur_bout[1] - cur_bout[0])
        
        elif next_bout[0] - cur_bout[1] > kwargs["peak_merging_distance"] :
            
            merged = False
            position_bouts_merged.append(cur_bout)
            length_bouts_merged.append(cur_bout[1] - cur_bout[0])
            
            if i == len(position_bouts)-2 :
                
                position_bouts_merged.append(next_bout)
                length_bouts_merged.append(next_bout[1] - next_bout[0])
                
    print("\n")
    ut.print_in_color("Merged neighboring peaks that were closer than {0}s away".format(kwargs["peak_merging_distance"]), "GREEN")
                
    return position_bouts_merged, length_bouts_merged

def detect_major_bouts(position_bouts_merged, **kwargs) :
    
    """Algorithm that detects major behavioral bouts based on size. (Seeds are
    behavioral bouts that are too short to be considered a "major" event)
    
    Args :  position_bouts_merged (arr) = list of merged start and end of each behavioral bout
            kwargs (dict) = dictionnary with additional parameters

    Returns : position_major_bouts (list) = list of start and end of each major behavioral bout
              length_major_bouts (list) = list of the length of each major behavioral bout
              position_seed_bouts (list) = list of start and end of each seed behavioral bout
              length_seed_bouts (list) = list of the length of each seed behavioral bout
    """
    
    position_major_bouts = []
    length_major_bouts = []
    
    position_seed_bouts = []
    length_seed_bouts = []
    
    for i in position_bouts_merged :
        
        bout_duration = i[1] - i[0]
        
        if bout_duration >= kwargs["minimal_bout_length"] :
            
            if i[0] >= kwargs["peri_event"]["graph_distance_pre"] and i[0] <= kwargs["video_end"]-kwargs["peri_event"]["graph_distance_post"]-kwargs["photometry_data"]["time_lost"] :
                
                position_major_bouts.append((i[0], i[1]))
                length_major_bouts.append(bout_duration)

            else :
                
                position_seed_bouts.append((i[0], i[1]))
                length_seed_bouts.append(bout_duration)

        elif bout_duration < kwargs["minimal_bout_length"] :
            
            position_seed_bouts.append((i[0], i[1]))
            length_seed_bouts.append(bout_duration)
            
    return position_major_bouts, length_major_bouts, position_seed_bouts, length_seed_bouts
    
def extract_peri_event_photmetry_data(position_bouts, **kwargs) :
    
    """Algorithm that extracts the photometry data at the moment when the animal behaves.
    
    Args :  position_bouts (arr) = list of start and end of each major behavioral bout
            kwargs (dict) = dictionnary with additional parameters

    Returns : dF_around_peaks (list) = list of photometry data at the moment when the animal behaves
    """
    
    dF_around_peaks = []
    
    for bout in position_bouts :  
        
        segment = kwargs["photometry_data"]["dFF"]["dFF"][int((bout[0]*kwargs["recording_sampling_rate"])-\
                        (kwargs["peri_event"]["graph_distance_pre"]*kwargs["recording_sampling_rate"])) :\
                        int((bout[0]*kwargs["recording_sampling_rate"])+((kwargs["peri_event"]["graph_distance_post"]+1)*kwargs["recording_sampling_rate"]))]
        
        dF_around_peaks.append(segment);
        
    return dF_around_peaks

def reorder_by_bout_size(dF_around_peaks, length_bouts) :
    
    """Simple algorithm that re-orders the peri-event photometry data based on the size of the behavioral event.
    
    Args :  dF_around_peaks (arr) = list of photometry data at the moment when the animal behaves
            length_bouts (list) = list of the length of each behavioral bout

    Returns : dF_around_peaks_ordered (list) = list of ordered photometry data at the moment when the animal behaves
              length_bouts_ordered (list) = list of the length of each behavioral bout
    """
    
    order = np.argsort(-np.array(length_bouts))
    
    dF_around_peaks_ordered = np.array(dF_around_peaks)[order]
    length_bouts_ordered = np.array(length_bouts)[order]
    
    return dF_around_peaks_ordered, length_bouts_ordered







