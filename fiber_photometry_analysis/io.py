#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:15:03 2020

@author: thomas.topilko
"""

import os

import pandas as pd
import numpy as np

import moviepy.editor as mpy

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import signal_preprocessing as preproc
from fiber_photometry_analysis.exceptions import FiberFotometryIoFileNotFoundError


def convert_to_npy(file_path, **kwargs):
    """Function that takes a csv file as an input, extract useful data from it
    and saves it as a npy file
    
    Args :      file (str) = the path to the csv file
    
    Returns :   npy_file_path (str) = The path to the npy file
    """
    
    npy_file_path = os.path.join(os.path.dirname(file_path), "{0}.npy".format(os.path.basename(file_path).split(".")[0]))
    
    if not os.path.exists(file_path):
        raise FiberFotometryIoFileNotFoundError(file_path)
    
    if os.path.exists(npy_file_path):  # FIXME: extract
        os.remove(os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0] + ".npy"))  # If you want to remove a pre-existing npy file
        
    print("Converting CSV photometry data into NPY")

    photometry_sheet = pd.read_csv(file_path, header=1, usecols=np.arange(0, 3))  # Load the data
    
    non_mask_time = np.isnan(photometry_sheet["Time(s)"])  # F iltering NaN values (missing values)
    non_nan_mask_ch1 = np.isnan(photometry_sheet["AIn-1 - Dem (AOut-1)"])  # Filtering NaN values (missing values)
    non_nan_mask_ch2 = np.isnan(photometry_sheet["AIn-1 - Dem (AOut-2)"])  # Filtering NaN values (missing values)
    non_nan_mask = ~np.logical_or(np.logical_or(non_mask_time, non_nan_mask_ch1), non_nan_mask_ch2)
    
    filtered_photometry_sheet = photometry_sheet[non_nan_mask]  # Filter the data
    photometry_data_npy = filtered_photometry_sheet.to_numpy()  # Convert to numpy for speed
    
    seconds_ramaining = len(photometry_data_npy) % kwargs["recording_sampling_rate"]
    photometry_data_cropped = photometry_data_npy[:int(len(photometry_data_npy) - seconds_ramaining)]
    photometry_data_transposed = np.transpose(photometry_data_cropped)  # Transpose data
    
    x = np.array(photometry_data_transposed[0])  # time data
    
    if not kwargs["video"]:
        x_adjusted = np.linspace(0, kwargs["recording_duration"], len(x))  # adjusted time data to fit the "real" video time
        x_target = np.arange(0, kwargs["recording_duration"], 1/kwargs["recording_sampling_rate"])  # time to be extracted
    else:
        x_adjusted = np.linspace(0, kwargs["video_duration"], len(x))  # adjusted time data to fit the "real" video time
        x_target = np.arange(kwargs["video_start"], kwargs["video_end"], 1/kwargs["recording_sampling_rate"])  # time to be extracted
        
    isosbestic = np.array(photometry_data_transposed[1])  # isosbestic data
    isosbestic_adjusted = preproc.adjust_signal_to_video_time(x_adjusted, x_target, isosbestic)  # data compressed in time
    
    calcium = np.array(photometry_data_transposed[2])  # calcium data
    calcium_adjusted = preproc.adjust_signal_to_video_time(x_adjusted, x_target, calcium)  # data compressed in time
    
    x_final = x_target - x_target[0]

    sink = np.array([x_final, isosbestic_adjusted, calcium_adjusted])
    
    np.save(npy_file_path, sink)  # Save the data as numpy file
    
    print("Filetered : {0} points".format(len(photometry_sheet)-len(filtered_photometry_sheet)))

    return npy_file_path


def get_recording_duration_and_sampling_rate(file_path, allow_downsampling=True):
    """Function that takes a csv file as an input, extract the time data from it,
    computes an estimate of the sampling rate and returns it

    Args :      file (str) = the path to the csv file

    Returns :   sr (float) = The estimate sampling rate of the photometry system
    """
    if not os.path.exists(file_path):
        raise FiberFotometryIoFileNotFoundError(file_path)

    photometry_sheet = pd.read_csv(file_path, header=1, usecols=[0])  # Load the data
    photometry_data_npy = photometry_sheet.to_numpy()  # Convert to numpy for speed
    photometry_data_transposed = np.transpose(photometry_data_npy)  # Transpose data

    x = np.array(photometry_data_transposed[0])  # time data

    sr = round(len(x) / x[-1])

    if sr >= 250:
        print("\n")
        utils.print_in_color("The sampling rate of the recording is pretty high : {0}. "
                             "We suggest to downsample the data using the pp.down_sample_signal function (250Hz)"
                             .format(sr), "RED")

        if allow_downsampling:
            factor = int(sr/250)
            sr = sr/factor
            utils.print_in_color("Downsampling was enabled by user. New sampling rate of data : {0}Hz"
                                 .format(sr), "GREEN")
        else:
            factor = None
    else:
        factor = None

    utils.print_in_color("Lenght of recording : {0}s, estimated sampling rate of the system : {1}"
                         .format(round(x[-1]), sr), "GREEN")
    return round(x[-1]), sr, factor
    

def get_video_duration_and_framerate(file_path):
    if not os.path.exists(file_path):
        raise FiberFotometryIoFileNotFoundError(file_path)
    
    video_clip = mpy.VideoFileClip(file_path)
    duration = video_clip.duration
    fps = video_clip.fps
    
    return duration, fps
