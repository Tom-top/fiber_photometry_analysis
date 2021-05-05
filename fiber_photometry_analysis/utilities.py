#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:02:35 2020

@author: thomas.topilko
"""
import logging
import os
import math

import numpy as np

from fiber_photometry_analysis.exceptions import FiberPhotometryTypeError

colors = {
    "WHITE": '\033[1;37m',
    "GREEN": '\033[0;32m',
    "YELLOW": '\033[1;33;48m',
    "RED": '\033[1;31;48m',
    "BLINK": '\33[5m',
    "BLINK2": '\33[6m',
    "RESET": '\033[1;37;0m'
}


def colorize(msg, color):
    color = color.upper()
    color = colors[color]
    return "{color}{msg}{reset_color}".format(color=color, msg=msg, reset_color=colors["RESET"])


def print_in_color(message, color):  # TODO: call when logging.note and red when error...
    """
    Function that receives a string as an argument
    and prints the message in color

    :param str message: Message to be printed
    :param str color: The color of the message to be printed. One of
    ("GREEN", "YELLOW", "RED", "BLINK", "BLINK2" "RESET")
    """
    print(colorize(message, color))


def safe_int(val):
    try:
        return int(val)
    except ValueError:
        raise FiberPhotometryTypeError(val)


def h_m_s(time, add_tags=True):
    """Function that receives a time in seconds as an argument
    and returns the time in h, m, s format
    
    Args :      time (int) = time in seconds
    
    Returns :   h (int) = hours
                m (int) = minutes
                s (int) = seconds
    """
    time = safe_int(time)

    delta = time % 3600
    h = (time - delta) / 3600
    m = (delta - (delta % 60)) / 60
    s = round((delta % 60), 2)

    if add_tags:
        return "{0}h".format(h), "{0}min".format(m), "{0}s".format(s)
    else:
        return h, m, s


def seconds(h, m, s):
    """Function that receives a time in h, m, s format as an argument
    and returns the time in seconds
    
    Args :      h (int) = hours
                m (int) = minutes
                s (int) = seconds
    
    Returns :   h*3600+m*60+s (int) = time in seconds
    """
    h, m, s = (safe_int(val) for val in (h, m, s))
    return h * 3600 + m * 60 + s


def check_if_path_exists(path):
    """Function that receives a path as an argument and prints a warning
    message in case the path is not existent
    
    Args :      path (str) = the path to be tested
    """
    
    if not os.path.exists(path):
        logging.warning('{} does not exist!'.format(path))
        return False
    else:
        logging.info("{} was correctly detected".format(path))
        return True


def create_workspace(working_directory):
    saving_directory = os.path.join(working_directory, "results")
    directories_to_create = {"saving_directory": saving_directory,
                             "plot_directory": os.path.join(saving_directory, "plots"),
                             "data_directory": os.path.join(saving_directory, "data"),
                             }

    for dtc in directories_to_create.values():
        if not os.path.exists(dtc):
            os.mkdir(dtc)
    return directories_to_create


def load_workspace(working_directory, experiment, mouse):
    """Function that receives the working directory as an argument as well as
    the experiment name and the mouse name and returns the paths of the files for
    analysis
    
    Args :      working_directory (str) = the working directory
                experiment (str) = the experiment date (format yymmdd)
                mouse (str) = the animals name
    
    Returns :   photometry_file_csv (str) = path to the photometry file
                video_file (str) = path to the video file
                behavior_automatic_file (str) = path to the automatic behavior file
                behavior_manual_file (str) = the working manual behavior file
    """

    def get_generic_file_path(base_name, ext):
        return os.path.join(working_directory,
                            "{}_{}_{}.{}".format(base_name, experiment, mouse, ext))

    workspace = {"photometry_file": get_generic_file_path("photometry", "csv"),
                 "color_video": get_generic_file_path("color_video", "mp4"),
                 "depth_video": get_generic_file_path("depth_video", "mp4"),
                 "behavior_automatic": get_generic_file_path("behavior_automatic", "npy"),
                 "behavior_manual": get_generic_file_path("behavior_manual", "xlsx"),
                 }

    workspace_directories = create_workspace(working_directory)
    workspace.update(workspace_directories)

    for n, p in workspace.items():
        exists = check_if_path_exists(p)
        if not exists:
            workspace[n] = None

    return workspace


def generate_xticks_and_labels(time, params, video_time=False):  # FIXME: extract unit computation
    """Small funtion that automatically generates xticks and labels for plots 
    depending on the length (s) of the data.
    
    Args :      time (int) = the time (s)
    
    Returns :   xticks (arr) = the xticks
                xticklabels (arr) = the labels
                unit (str) = the unit
                :param params:
    """
    
    n_mins = (time - time % 60) / 60
    
    if n_mins > 1:
        if n_mins < 10:
            step_ticks = 60
        else:
            step_ticks = (time - time % 600) / 10
        xticks = np.arange(0, (time - (time % 60)) + step_ticks, step_ticks)
        step_labels = step_ticks / 60
        xticklabels = np.arange(0, ((time - (time % 60)) / 60) + step_labels, step_labels)
        if video_time:
            xticklabels = xticklabels + (params["signal_pp"]["general"]["crop_start"] / 60)
        unit = "min"
    else:
        if time >= 10:
            step_ticks = 5
            xticks = np.arange(0, (time - time % 5) + step_ticks, step_ticks)
        else:
            step_ticks = 1
            xticks = np.arange(0, time + step_ticks, step_ticks)
        xticklabels = xticks
        if video_time:
            xticklabels = xticklabels + params["signal_pp"]["general"]["crop_start"]
        unit = "sec"
    
    return xticks, xticklabels, unit


def generate_yticks(source, delta):
    """Small funtion that automatically generates yticks for plots 
    depending on the range of the data.
    
    Args :      source (arr) = the data for which the yticks have to be generated
                delta = the value used for offsetting
    
    Returns :   y_min (float) = the minimum value of the yticks
                y_max (float) = the maximum value of the yticks
                round_factor (float) = the yticks step
    """
    y_max = offset(max(source), delta, "+")
    y_min = offset(min(source), delta, "-")
    
    round_factor = 1
    while round_factor > (abs(y_max - y_min) - abs(y_max - y_min) * 0.5):
        round_factor /= 10
    y_max = (math.ceil(y_max / round_factor)) * round_factor
    y_min = (math.floor(y_min / round_factor)) * round_factor
        
    return y_min, y_max, round_factor


def offset(value, offset, sign):  # FIXME: shadows name
    """Small function that offsets a value.
    
    Args :      value (float) = the value to be changed
                offset (float) = the percentage of offset to be used
                sign (str) = the type of offset to be performed
    
    Returns :   sink (float) = the offset value
    """
    signed_offset = (abs(value) * offset)
    if sign == "-":
        signed_offset *= -1

    return value + signed_offset


def replace_ext(file_path, file_ext):  # TODO: check if exists
    return os.path.join(os.path.dirname(file_path),
                        "{}.{}".format(os.path.basename(file_path).split(".")[0], file_ext))

def replace_photometry_files_into_folders(working_directory):
    photometry_files = []
    for f in os.listdir(working_directory):
        if f.endswith(".csv") and f.startswith("photometry"):
            photometry_files.append(f)
    for f in photometry_files:
        file_path = os.path.join(working_directory, f)
        file_name = os.path.splitext(f)[0]
        folder_name = "_".join(file_name.split("_")[2:])
        folder_path = os.path.join(working_directory, folder_name)
        os.rename(file_path, os.path.join(folder_path, f))

def rename_video_names(working_directory):
    for folder in os.listdir(working_directory):
        path_to_foler = os.path.join(working_directory, folder)
        if os.path.isdir(path_to_foler):
            for file in os.listdir(path_to_foler):
                path_to_file = os.path.join(path_to_foler, file)
                if file.endswith(".mp4") and not file.startswith("."):
                    file_name = os.path.splitext(file)[0]
                    file_ext = os.path.splitext(file)[1]
                    split_name = file_name.split("_")
                    new_name = "_".join(split_name[:-2]+split_name[::-1][:2]) + file_ext
                    new_path_to_foler = os.path.join(path_to_foler, new_name)
                    os.rename(path_to_file, new_path_to_foler)