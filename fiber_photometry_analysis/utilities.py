#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:02:35 2020

@author: thomas.topilko
"""

import os
import math

import numpy as np


colors = {
    "GREEN": '\33[32m',
    "YELLOW": '\033[1;33;48m',
    "RED": '\033[1;31;48m',
    "BLINK": '\33[5m',
    "BLINK2": '\33[6m',
    "END": '\033[1;37;0m'
}


def print_in_color(message, color):
    """Function that receives a string as an argument
    and prints the message in color
    
    Args :      message (str) = Message to be printed
    
    Returns :   color (str) = The color of the message to be printed (colors)
    """
    color = colors[color]
    print("{0}{1}{2}".format(color, message, colors["END"])) 


def h_m_s(time, add_tags=True):
    """Function that receives a time in seconds as an argument
    and returns the time in h, m, s format
    
    Args :      time (int) = time in seconds
    
    Returns :   h (int) = hours
                m (int) = minutes
                s (int) = seconds
    """
    try:
        int(time)
    except:
        raise RuntimeError("The input value is not of the correct type, required type = int")

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
    
    if all(isinstance(i, int) for i in [h, m, s]):
        return h*3600+m*60+s
    else:
        raise RuntimeError("The input values are not of the correct type, required type = int")


def check_if_path_exists(path):
    """Function that receives a path as an argument and prints a warning
    message in case the path is not existant
    
    Args :      path (str) = the path to be tested
    """
    
    if not os.path.exists(path):
        print_in_color("[WARNING] {0} does not exist!".format(path), "RED")
        return False
    else:
        print_in_color("{0} was correctly detected".format(path), "GREEN")
        return True


def set_file_paths(working_directory, experiment, mouse):
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
    
    photometry_file_csv = os.path.join(working_directory, "photometry_{0}_{1}.csv".format(experiment, mouse))
    video_file = os.path.join(working_directory, "video_{0}_{1}.avi".format(experiment, mouse))
    behavior_automatic_file = os.path.join(working_directory, "behavior_automatic_{0}_{1}.npy".format(experiment, mouse))
    behavior_manual_file = os.path.join(working_directory, "behavior_manual_{0}_{1}.xlsx".format(experiment, mouse))
    saving_directory = os.path.join(working_directory, "Results")

    print("\n")

    files = [photometry_file_csv, video_file, behavior_automatic_file, behavior_manual_file, saving_directory]
    for n, p in enumerate(files):
        if p == saving_directory:
            if not os.path.exists(p):
                os.mkdir(saving_directory)
        else:
            exists = check_if_path_exists(p)
            if not exists:
                files[n] = None

    return files


def generate_xticks_and_labels(time):
    """Small funtion that automatically generates xticks and labels for plots 
    depending on the length (s) of the data.
    
    Args :      time (int) = the time (s)
    
    Returns :   xticks (arr) = the xticks
                xticklabels (arr) = the labels
                unit (str) = the unit
    """
    
    n_mins = (time - time % 60)/60
    
    if n_mins > 1:
        if n_mins < 10:
            step_ticks = 60
            xticks = np.arange(0, (time - (time%60))+step_ticks, step_ticks)
            step_labels = step_ticks/60
            xticklabels = np.arange(0, ((time - (time%60))/60)+step_labels, step_labels)
            unit = "min"
        elif n_mins >= 10:
            step_ticks = (time - time%600)/10
            xticks = np.arange(0, (time - (time%60))+step_ticks, step_ticks)
            step_labels = step_ticks/60
            xticklabels = np.arange(0, ((time - (time%60))/60)+step_labels, step_labels)
            unit = "min"
    elif n_mins <= 1:
        if time >= 10:
            step_ticks = 5
            xticks = np.arange(0, (time - time%5)+step_ticks, step_ticks)
            xticklabels = xticks
            unit = "sec"
        elif time < 10:
            step_ticks = 1
            xticks = np.arange(0, time+step_ticks, step_ticks)
            xticklabels = xticks
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
    if y_max >= 0:
        y_max = (math.ceil(y_max/round_factor))*round_factor
    else:
        y_max = (math.ceil(y_max/round_factor))*round_factor
        
    if y_min >= 0:
        y_min = (math.floor(y_min/round_factor))*round_factor
    else:
        y_min = (math.floor(y_min/round_factor))*round_factor
        
    return y_min, y_max, round_factor


def offset(value, offset, sign):  # FIXME:
    """Small funtion that offsets a value.
    
    Args :      value (float) = the value to be changed
                offset (float) = the percentage of offset to be used
                sign (str) = the type of offset to be performed
    
    Returns :   sink (float) = the offset value
    """
    if sign == "-":
        sink = value - (abs(value) * offset)
        return sink
    elif sign == "+":
        sink = value + (abs(value) * offset)

    return sink


    
    
    
    
    
    
    
    
    
    
    
    
    
    