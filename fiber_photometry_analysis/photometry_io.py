#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:15:03 2020

@author: thomas.topilko
"""
import logging
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

import pandas as pd
import numpy as np

import moviepy.editor as mpy

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import signal_preprocessing as preproc
from fiber_photometry_analysis.exceptions import FiberFotometryIoFileNotFoundError
from fiber_photometry_analysis.utilities import replace_ext


def validate_path(file_path):
    if file_path == None:
        raise FiberFotometryIoFileNotFoundError(file_path)
    if not os.path.exists(file_path):
        raise FiberFotometryIoFileNotFoundError(file_path)


def convert_photometry_data_to_dataframe(file_path):
    photometry_data = pd.read_csv(file_path, header=1, usecols=np.arange(0, 3))
    filtered_data = photometry_data.dropna(thresh=1)
    return filtered_data


def save_dataframe_to_feather(source_file_path, saving_path, overwrite=False):
    saving_name = os.path.splitext(os.path.basename(source_file_path))[0]
    saving_file_path = os.path.join(saving_path, "results/data/{}.feather".format(saving_name))
    if not os.path.exists(saving_file_path):
        df = convert_photometry_data_to_dataframe(source_file_path)
        df.to_feather(saving_file_path)
    else:
        if overwrite:
            df = convert_photometry_data_to_dataframe(source_file_path)
            df.to_feather(saving_file_path)
    return saving_file_path


def deprecated_convert_to_npy(file_path, **kwargs):  # FIXME: replace by save to dataframe binary
    """Function that takes a csv file as an input, extract useful data from it
    and saves it as a npy file
    
    Args :      file (str) = the path to the csv file
    
    Returns :   npy_file_path (str) = The path to the npy file
    """
    
    validate_path(file_path)

    npy_file_path = replace_ext(file_path, 'npy')
    if os.path.exists(npy_file_path):
        os.remove(npy_file_path)
        
    logging.info("Converting CSV photometry data into NPY")

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

    recording_freq = 1 / kwargs["recording_sampling_rate"]
    if not kwargs["video"]:
        x_adjusted = np.linspace(0, kwargs["recording_duration"], len(x))  # adjusted time data to fit the "real" video time
        x_target = np.arange(0, kwargs["recording_duration"], recording_freq)  # time to be extracted
    else:
        x_adjusted = np.linspace(0, kwargs["video_duration"], len(x))  # adjusted time data to fit the "real" video time
        x_target = np.arange(kwargs["video_start"], kwargs["video_end"], recording_freq)  # time to be extracted
        
    isosbestic = np.array(photometry_data_transposed[1])  # isosbestic data
    isosbestic_adjusted = preproc.adjust_signal_to_video_time(x_adjusted, x_target, isosbestic)  # data compressed in time
    
    calcium = np.array(photometry_data_transposed[2])  # calcium data
    calcium_adjusted = preproc.adjust_signal_to_video_time(x_adjusted, x_target, calcium)  # data compressed in time
    
    x_final = x_target - x_target[0]

    sink = np.array([x_final, isosbestic_adjusted, calcium_adjusted])
    
    np.save(npy_file_path, sink)  # Save the data as numpy file
    
    logging.info("Filetered : {0} points".format(len(photometry_sheet) - len(filtered_photometry_sheet)))

    return npy_file_path


def doric_csv_to_dataframe(file, **args):
    photometry_sheet = pd.read_csv(file, header=1, usecols=np.arange(0, 3))  # Load the data
    
    non_mask_time = np.isnan(photometry_sheet["Time(s)"])  # Filtering NaN values (missing values)
    non_nan_mask_isosbestic = np.isnan(photometry_sheet["AIn-1 - Dem (AOut-{})".format(args["isosbestic_channel"])])  # Filtering NaN values (missing values)
    non_nan_mask_calcium = np.isnan(photometry_sheet["AIn-1 - Dem (AOut-{})".format(args["calcium_channel"])])  # Filtering NaN values (missing values)
    non_nan_mask = ~np.logical_or(np.logical_or(non_mask_time, non_nan_mask_isosbestic), non_nan_mask_calcium)
    
    filtered_photometry_sheet = photometry_sheet[non_nan_mask]  # Filter the data
    
    seconds_remaining = len(filtered_photometry_sheet) % args["recording_sampling_rate"]
    photometry_data_cropped = photometry_data_npy[:int(len(filtered_photometry_sheet) - seconds_remaining)]
    photometry_data_transposed = np.transpose(photometry_data_cropped)  # Transpose data
    
    return photometry_data_transposed


def convert_to_feather(file, recompute=True, **kwargs):
    
    feather_file_path = os.path.join(os.path.dirname(file), "{0}.feather".format(os.path.basename(file).split(".")[0]))
    if os.path.exists(feather_file_path):
        if recompute:
            pass
    else:
        pass


def save_perievent_data_to_pickle(metadata, experiment, mouse, params):
    file_saving_path = os.path.join(params["general"]["behavior_saving_directory"], "perievent_data_{}_{}.pickle".format(experiment, mouse))
    with open(file_saving_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_perievent_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_recording_duration_and_sampling_rate(file_path, allow_downsampling=True):
    """Function that takes a csv file as an input, extract the time data from it,
    computes an estimate of the sampling rate and returns it

    Args :      file (str) = the path to the csv file

    Returns :   sr (float) = The estimate sampling rate of the photometry system
    """
    validate_path(file_path)

    photometry_sheet = pd.read_csv(file_path, header=1, usecols=[0])  # Load the data
    photometry_data_npy = photometry_sheet.to_numpy()  # Convert to numpy for speed
    photometry_data_transposed = np.transpose(photometry_data_npy)  # Transpose data

    x = np.array(photometry_data_transposed[0])  # time data

    x_max = x[-1]
    sr = round(len(x) / x_max)

    factor = None
    if sr >= 250:
        logging.error("\nThe sampling rate of the recording is pretty high : {}. "
                      "We suggest to downsample the data using the pp.down_sample_signal function (250Hz)"
                      .format(sr))

        if allow_downsampling:
            factor = int(sr / 250)
            sr /= factor
            logging.info("Downsampling was enabled by user. New sampling rate of data : {0}Hz".format(sr))

    sampling_info = {"recording_duration": round(x_max),
                     "recording_sampling_rate": sr,
                     "downsampling_factor": factor
                     }
    logging.info("Length of recording : {}s, estimated sampling rate of the system : {}".format(round(x_max), sr))

    return sampling_info
    

def get_video_duration_and_framerate(file_path):
    validate_path(file_path)
    
    video_clip = mpy.VideoFileClip(file_path)
    duration = video_clip.duration
    fps = video_clip.fps
    
    return duration, fps


def reset_dataframe_index(df):
    length = len(df)
    idx = pd.Index(np.arange(0, length, 1))
    df = df.set_index(idx)
    return df
