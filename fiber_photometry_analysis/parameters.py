#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:41:56 2020

@author: thomas.topilko
"""
import os
import shutil

import logging
from configobj import ConfigObj
import validate

from fiber_photometry_analysis import photometry_io
from fiber_photometry_analysis import utilities as utils


def load_config(working_directory, experiment_tag, animal_tag, overwrite=True):
    config_file_path = os.path.join(working_directory, "config.cfg")
    configspec_file_path = os.path.join(os.getcwd(), "config.configspec")
    if not os.path.exists(config_file_path) or overwrite:
        generic_config_file_path = os.path.join(os.getcwd(), "config.cfg")
        shutil.copy(generic_config_file_path, config_file_path)
        print("No config file was detected, generated default config")
    configspec = ConfigObj(configspec_file_path, encoding='UTF8',
                           list_values=False, _inspec=True, stringify=True)
    config = ConfigObj(config_file_path, unrepr=True, configspec=configspec)
    config.validate(validate.Validator(), preserve_errors=True, copy=True)

    workspace = utils.load_workspace(working_directory, experiment_tag, animal_tag)
    config["general"].update(workspace)
    allow_downsampling = config["general"]["photometry"]["allow_downsampling"]
    sampling_info = photometry_io.get_recording_duration_and_sampling_rate(workspace["photometry_file"],
                                                                           allow_downsampling=allow_downsampling)
    config["general"].update(sampling_info)

    if config["general"]["color_video"] is not None:
        config["general"]["video"] = True  # Bool. True if a video is to be analyzed, otherwise False
        video_duration, video_sampling_rate = photometry_io.get_video_duration_and_framerate(workspace["color_video"])
        config["general"]["video_duration"] = video_duration
        config["general"]["video_start"] = 0
        config["general"]["video_end"] = video_duration
        config["general"]["video_sampling_rate"] = video_sampling_rate
    else:
        config["general"]["video"] = False  # If no video
        config["general"]["video_sampling_rate"] = 0

    config["signal_pp"]["smoothing"]["smoothing_window"] = int(config["general"]["recording_sampling_rate"])

    config["plotting"]["peri_event"]["resample_graph"] = config["general"]["recording_sampling_rate"]
    config["plotting"]["peri_event"]["resample_heatmap"] = config["general"]["recording_sampling_rate"] / 100

    config["live_video"]["general"]["display_threshold"] = int(5 * config["general"]["video_sampling_rate"])
    config["live_video"]["general"]["plot_acceleration"] = config["general"]["recording_sampling_rate"]

    print("\nParameters loaded successfully\n")

    return config
