#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:08:31 2020

@author: thomas.topilko
"""

import os
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import numpy as np
import datetime
from scipy import signal
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.linear_model import Lasso, LinearRegression

import Utilities as ut

#########################################################################################################
# Functions for signal processing
#########################################################################################################

def down_sample_signal(source, factor) : 
    
    """Downsample the data using a certain factor.
    
    Args :  source (arr) = The input signal 
            factor (int) = The factor of down sampling

    Returns : sink = The downsampled signal
    """
    
    if source.ndim != 1:
        
        raise ValueError("downsample only accepts 1 dimension arrays.")
    
    sink = np.mean(source.reshape(-1, factor), axis=1)
    
    return sink

def smooth_signal(source, window_len=10, window='flat') :
    
    """Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Args :  source (arr) = the input signal 
            window_len (int) = the dimension of the smoothing window; should be an odd integer
            window (str) = the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns : sink = the smoothed signal
    
    Code taken from (https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)
    """

    if source.ndim != 1:
        
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if source.size < window_len:
        
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len <3 :
        
        return source


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[source[window_len-1:0:-1], source ,source[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    sink = np.convolve(w/w.sum(),s,mode='valid')
    
    return sink

def trailing_moving_average(source, window=1) :
    
    """Function that takes an 1D array as an argument and returns its trailing
    moving average by computing the unweighted mean for the previous n = window data
    
    [[WARNING] This shifts the mean in time, use "centered_moving_average" for an aligned moving average]
    
    Args :      source (np.array) = The signal for which the moving average has to be computed
                window (int) = The window of values that the algorithm takes to compute the moving average
    
    Returns :   sink (np.array) = The trailing moving average of the input signal
    """
    
    source = np.array(source)
    
    if len(source.shape) == 1 :
    
        cumsum = np.cumsum(source)
        sink = (cumsum[window:] - cumsum[:-window]) / float(window)
        source = source[window:]
        return source, sink
    
    else :
        
        raise RuntimeError("The input array has too many dimensions. Input : {0}D, Requiered : 1D".format(len(source.shape)))

def centered_moving_average(source, window=1) :
    
    """Function that takes an 1D array as an argument and returns its centered
    moving average by computing the unweighted mean for the previous and next n = window data
    
    Args :      source (np.array) = The signal for which the moving average has to be computed
                window (int) = The window of values that the algorithm takes to compute the moving average
    
    Returns :   sink (np.array) = The centered moving average of the input signal
    """
    
    source = np.array(source)
    
    if len(source.shape) == 1 :
        
        cumsum = np.cumsum(source)
        sink = (cumsum[window:] - cumsum[:-window]) / float(window)
        source = source[int(window/2):-int(window/2)]
        return source, sink
    
    else :
        
        raise RuntimeError("The input array has too many dimensions. Input : {0}D, Requiered : 1D".format(len(source.shape)))
        
def low_pass_filter(source, sr, cutoff_freq, order) :
    
    """Function that takes an 1D array as an argument and returns the filtered 
    signal following the application of a low pass filter with a cutoff frequency = cutoff_freq and
    order = order
    
    Args :      source (np.array) = The signal to be filtered
                sr (int) = The sampling rate of the signal
                cutoff_freq (int) = the cutoff frequency of the lowpass filter
                order (int) = the order of the lowpass filter
                
    
    Returns :   sink (np.array) = The centered moving average of the input signal
    """
    
    for name, arg in zip(["sr", "cutoff_freq", "order"], [sr, cutoff_freq, order]) :
        
        if type(arg) != int :
        
            raise RuntimeError("The argument {0} is of incorrect type. Input : {1}, Requiered : int".format(name, type(arg)))
    
    source = np.array(source)
    
    if len(source.shape) == 1 :
    
        nyq = sr/2 #The nyquist frequency
        normalized_cutoff = cutoff_freq / nyq
        
        z, p, k = signal.butter(order, normalized_cutoff, output="zpk")
        lesos = signal.zpk2sos(z, p, k)
        filtered = signal.sosfilt(lesos, source)
        sink = np.array(filtered)
        
        return sink
    
    else :
        
        raise RuntimeError("The input array has too many dimensions. Input : {0}D, Requiered : 1D".format(len(source.shape)))

def adjust_signal_to_video_time(time_video, time_final, source) :
    
    """Function that adjusts the time stamps of the source signal to the time
    stamps of the video. It then extracts the signal with the same sampling rate
    as the video to simplify the analysis with videos.
    
    Args :      time_video (arr) = The time of the video (photometry sampling rate)
                time_final (arr) = The time of the video (real fps recording)
                source (arr) = the signal to be adjusted

    Returns :   sink (arr) = The adjusted signal
    """
    
    spl = interp1d(time_video, source)
    sink = spl(time_final)
    
    return sink

def extract_raw_data(file, **kwargs) :
    
    """Function that extracts the raw data from the csv file and displays it
    in a plot.
    
    Args :      file (str) = The input photometry file for analysis
                kwargs (dict) = Dictionnary with the parameters

    Returns :   x (arr) = The time data in X
                isosbestic_adjusted (arr) = The adjusted isosbestic signal (time fitted to the video)
                calcium_adjusted (arr) = The adjusted calcium signal (time fitted to the video)
    """
    
    print("\nExtracting raw data for Isosbestic and Calcium recordings !")
    
    photometry_data = np.load(file) #Load the NPY file

    x = photometry_data[0] #time to be extracted
    isosbestic_adjusted = photometry_data[1] #data compressed in time
    calcium_adjusted = photometry_data[2] #data compressed in time
    x_max = x[-1]
#    print("Data length : {0}".format(ut.h_m_s(x_max, add_tags=True)))
    
    if kwargs["photometry_pp"]["plots_to_display"][0] :
        
        xticks, xticklabels, unit = ut.generate_xticks_and_labels(x_max)
        
        fig = plt.figure(figsize=(10, 5), dpi=200.)
        ax0 = plt.subplot(211)
        p, = ax0.plot(x, isosbestic_adjusted, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"])
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(isosbestic_adjusted, 0.1)
        ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel("mV", fontsize=kwargs["fsl"])
        ax0.legend(handles=[p], labels=["isosbestic"], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title("Raw Isosbestic and Calcium signals", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        ax1 = plt.subplot(212, sharex=ax0)
        b, = ax1.plot(x, calcium_adjusted, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"])
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax1.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(calcium_adjusted, 0.1)
        ax1.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax1.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax1.set_ylim(y_min, y_max)
        ax1.set_ylabel("mV", fontsize=kwargs["fsl"])
        ax1.legend(handles=[b], labels=["calcium"], loc=2, fontsize=kwargs["fsl"])
        ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
        ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["photometry_pp"]["multicursor"] :
            
            multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "Raw_Signals.{0}".format(kwargs["extension"])), dpi=200.)
        
    return x, isosbestic_adjusted, calcium_adjusted

def down_sample(x, isosbestic, calcium, factor) : 
    
    """Downsample the two channels using a certain factor.
    
    Args :  x (arr) = The time data in X
            isosbestic (arr) = The adjusted isosbestic signal (time fitted to the video)
            calcium (arr) = The adjusted calcium signal (time fitted to the video)
            factor (int) = The factor of down sampling

    Returns : sink = The downsampled signal
    """
    
    delta = len(x)%factor
    
    x = x[:-delta][::factor]
    isosbestic = down_sample_signal(isosbestic[:-delta], factor)
    calcium = down_sample_signal(calcium[:-delta], factor)
    
    return x, isosbestic, calcium

def smooth(x, isosbestic, calcium, **kwargs) :
    
    """Function that smooths the raw data and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The adjusted isosbestic signal (time fitted to the video)
                calcium (arr) = The adjusted calcium signal (time fitted to the video)
                kwargs (dict) = Dictionnary with the parameters

    Returns :   x (arr) = The new time data in X
                isosbestic_smoothed (arr) = The smoothed isosbestic signal
                calcium_smoothed (arr) = The smoothed calcium signal
    """
    
    print("\nStarting smoothing for Isosbestic and Calcium signals !")
    
    isosbestic_smoothed = smooth_signal(isosbestic, window_len=kwargs["smoothing_window"])[:-kwargs["smoothing_window"]+1]
    calcium_smoothed = smooth_signal(calcium, window_len=kwargs["smoothing_window"])[:-kwargs["smoothing_window"]+1]
    x_max = x[-1]
#    print("Data length : {0}".format(ut.h_m_s(x_max, add_tags=True)))
    
    if kwargs["photometry_pp"]["plots_to_display"][1] :
        
        xticks, xticklabels, unit = ut.generate_xticks_and_labels(x_max)
                
        fig = plt.figure(figsize=(10, 5), dpi=200.)
        ax0 = plt.subplot(211)
        p, = ax0.plot(x, isosbestic_smoothed, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"])
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(isosbestic_smoothed, 0.1)
        ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel("mV", fontsize=kwargs["fsl"])
        ax0.legend(handles=[p], labels=["isosbestic"], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title("Smoothed Isosbestic and Calcium signals", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        ax1 = plt.subplot(212, sharex=ax0)
        b, = ax1.plot(x, calcium_smoothed, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"])
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax1.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(calcium_smoothed, 0.1)
        ax1.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax1.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax1.set_ylim(y_min, y_max)
        ax1.set_ylabel("mV", fontsize=kwargs["fsl"])
        ax1.legend(handles=[b], labels=["calcium"], loc=2, fontsize=kwargs["fsl"])
        ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
        ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["photometry_pp"]["multicursor"] :
            
            multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "Smoothed_Signals.{0}".format(kwargs["extension"])), dpi=200.)
        
    return x, isosbestic_smoothed, calcium_smoothed

def find_baseline(x, isosbestic, calcium, **kwargs) :
    
    """Function that estimates the baseline of the smoothed signals and 
    displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The smoothed isosbestic signal (time fitted to the video)
                calcium (arr) = The smoothed calcium signal (time fitted to the video)
                kwargs (dict) = Dictionnary with the parameters

    Returns :   x (arr) = The new time data in X
                isosbestic (arr) = The cropped isosbestic signal
                calcium (arr) = The cropped calcium signal
                function_isosbestic (arr) = The baseline for the isosbestic signal
                function_calcium (arr) = The baseline for the calcium signal
    """
    
    print("\nStarting baseline computation for Isosbestic and Calcium signals !")
                   
    isosbestic, function_isosbestic = centered_moving_average(isosbestic, window=kwargs["moving_average_window"]) #moving average for isosbestic data
    calcium, function_calcium = centered_moving_average(calcium, window=kwargs["moving_average_window"]) #moving average for calcium data
    x = x[int(kwargs["moving_average_window"]/2):-int(kwargs["moving_average_window"]/2)]
    x = x - x[0]
    x_max = x[-1]
#    print("Data length : {0}".format(ut.h_m_s(x_max, add_tags=True)))
    
    if kwargs["photometry_pp"]["plots_to_display"][2] :
        
        xticks, xticklabels, unit = ut.generate_xticks_and_labels(x_max)
            
        fig = plt.figure(figsize=(10, 5), dpi=200.)
        ax0 = plt.subplot(211)
        p, = ax0.plot(x, isosbestic, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"])
        ma, = ax0.plot(x, function_isosbestic, alpha=0.8, c="orange", lw=2)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(isosbestic, 0.1)
        ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel("mV", fontsize=kwargs["fsl"])
        ax0.legend(handles=[p, ma], labels=["isosbestic", "moving average"], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title("Smoothed Isosbestic and Calcium signals with respective baselines", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        ax1 = plt.subplot(212, sharex=ax0)
        b, = ax1.plot(x, calcium, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"])
        ma, = ax1.plot(x, function_calcium, alpha=0.8, c="orange", lw=2)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax1.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(calcium, 0.1)
        ax1.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax1.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax1.set_ylim(y_min, y_max)
        ax1.set_ylabel("mV", fontsize=kwargs["fsl"])
        ax1.legend(handles=[b, ma], labels=["calcium", "moving average"], loc=2, fontsize=kwargs["fsl"])
        ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
        ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["photometry_pp"]["multicursor"] :
            
            multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "Baseline_Determination.{0}".format(kwargs["extension"])), dpi=200.)
        
    return x, isosbestic, calcium, function_isosbestic, function_calcium

def baseline_correction(x, isosbestic, calcium, isosbestic_fc, calcium_fc, **kwargs) :
    
    """Function that performs the basleine correction of the smoothed and cropped 
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The cropped isosbestic signal
                calcium (arr) = The cropped calcium signal
                function_isosbestic (arr) = The baseline for the isosbestic signal
                function_calcium (arr) = The baseline for the calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   isosbestic_corrected (arr) = The baseline corrected isosbestic signal
                calcium_corrected (arr) = The baseline corrected calcium signal
    """
    
    print("\nStarting baseline correcion for Isosbestic and Calcium signals !")
    
    isosbestic_corrected = ( isosbestic - isosbestic_fc ) / isosbestic_fc #baseline correction for isosbestic
    calcium_corrected = ( calcium - calcium_fc ) / calcium_fc #baseline correction for calcium
    x_max = x[-1]
    
    if kwargs["photometry_pp"]["plots_to_display"][3] :
        
        xticks, xticklabels, unit = ut.generate_xticks_and_labels(x_max)
                
        fig = plt.figure(figsize=(10, 5), dpi=200.)
        ax0 = plt.subplot(211)
        p, = ax0.plot(x, isosbestic_corrected, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"])
        ax0.plot((0, x[-1]), (0, 0), "--", color="black", lw=kwargs["lw"]) #Creates a horizontal dashed line at y = 0 to signal the baseline
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(isosbestic_corrected, 0.1)
        ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.legend(handles=[p], labels=["isosbestic"], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title("Baseline Correction of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        ax1 = plt.subplot(212, sharex=ax0)
        b, = ax1.plot(x, calcium_corrected, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"])
        ax1.plot((0, x[-1]), (0, 0), "--", color="black", lw=kwargs["lw"]) #Creates a horizontal dashed line at y = 0 to signal the baseline
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax1.set_xlim(0, x_max)
        y_min, y_max, round_factor = ut.generate_yticks(calcium_corrected, 0.1)
        ax1.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax1.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*1000], fontsize=kwargs["fsl"])
        ax1.set_ylim(y_min, y_max)
        ax1.legend(handles=[b], labels=["calcium"], loc=2, fontsize=kwargs["fsl"])
        ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
        ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["photometry_pp"]["multicursor"] :
            
            multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "Baseline_Correction.{0}".format(kwargs["extension"])), dpi=200.)
        
    return isosbestic_corrected, calcium_corrected

def standardization(x, isosbestic, calcium, **kwargs) :
    
    """Function that performs the standardization of the corrected 
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The baseline corrected isosbestic signal
                calcium (arr) = The baseline corrected calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   isosbestic_standardized (arr) = The standardized isosbestic signal
                calcium_standardized (arr) = The standardized calcium signal
    """
    
    if kwargs["photometry_pp"]["standardize"] : 
        
        print("\nStarting standardization for Isosbestic and Calcium signals !")
    
        isosbestic_standardized = ( isosbestic - np.median(isosbestic) ) / np.std(isosbestic) #standardization correction for isosbestic
        calcium_standardized = ( calcium - np.median(calcium) ) / np.std(calcium) #standardization for calcium
        
        x_max = x[-1]
    
        if kwargs["photometry_pp"]["plots_to_display"][4] :
            
            xticks, xticklabels, unit = ut.generate_xticks_and_labels(x_max)
                    
            fig = plt.figure(figsize=(10, 5), dpi=200.)
            ax0 = plt.subplot(211)
            p, = ax0.plot(x, isosbestic_standardized, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"])
            ax0.plot((0, x[-1]), (0, 0), "--", color="black", lw=kwargs["lw"]) #Creates a horizontal dashed line at y = 0 to signal the baseline
            ax0.set_xticks(xticks)
            ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
            ax0.set_xlim(0, x_max)
            y_min, y_max, round_factor = ut.generate_yticks(isosbestic_standardized, 0.1)
            ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
            ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
            ax0.set_ylim(y_min, y_max)
            ax0.set_ylabel("z-score", fontsize=kwargs["fsl"])
            ax0.legend(handles=[p], labels=["isosbestic"], loc=2, fontsize=kwargs["fsl"])
            ax0.set_title("Standardization of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
            ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            
            ax1 = plt.subplot(212, sharex=ax0)
            b, = ax1.plot(x, calcium_standardized, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"])
            ax1.plot((0, x[-1]), (0, 0), "--", color="black", lw=kwargs["lw"]) #Creates a horizontal dashed line at y = 0 to signal the baseline
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
            ax1.set_xlim(0, x_max)
            y_min, y_max, round_factor = ut.generate_yticks(calcium_standardized, 0.1)
            ax1.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
            ax1.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
            ax1.set_ylim(y_min, y_max)
            ax1.set_ylabel("z-score", fontsize=kwargs["fsl"])
            ax1.legend(handles=[b], labels=["calcium"], loc=2, fontsize=kwargs["fsl"])
            ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
            ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            
            if kwargs["photometry_pp"]["multicursor"] :
                
                multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])
            
            if kwargs["save"] :
                
                plt.savefig(os.path.join(kwargs["save_dir"], "Standardization.{0}".format(kwargs["extension"])), dpi=200.)
        
    else :
        
        ut.print_in_color("\nThe standardization step in skipped. Parameter plot_args['photometry_pp']['standardize'] == False", "RED")
        isosbestic_standardized = isosbestic #standardization skipped
        calcium_standardized = calcium #standardization skipped
        
    return isosbestic_standardized, calcium_standardized

def interchannel_regression(isosbestic, calcium, **kwargs) :
    
    """Function that performs the inter-channel regression of the two 
    signals and displays it in a plot.
    
    Args :      isosbestic (arr) = The standardized (or not) isosbestic signal
                calcium (arr) = The standardized (or not) calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   isosbestic_fitted (arr) = The fitted isosbestic signal
    """

    print("\nStarting interchannel regression and alignement for Isosbestic and Calcium signals !")
    
    if kwargs["photometry_pp"]["regression"] == "Lasso" :
    
        reg = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=False, random_state=9999, selection='random')
        
    else :
        
        reg = LinearRegression()
        
    n = len(calcium)
    reg.fit(isosbestic.reshape(n,1), calcium.reshape(n,1))
    isosbestic_fitted = reg.predict(isosbestic.reshape(n,1)).reshape(n,)
    
    if kwargs["photometry_pp"]["plots_to_display"][5] :
                
        fig = plt.figure(figsize=(5, 5), dpi=200.)
        ax0 = plt.subplot(111)
        ax0.scatter(isosbestic, calcium, color="blue", s=0.5, alpha=1)
        ax0.plot(isosbestic, isosbestic_fitted, 'r-', linewidth=1)
        ax0.set_xlabel("Isosbestic", fontsize=kwargs["fsl"])
        ax0.set_ylabel("Calcium", fontsize=kwargs["fsl"])
        ax0.set_title("Inter-channel regression of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "Interchannel_Regression.{0}".format(kwargs["extension"])), dpi=200.)
        
    return isosbestic_fitted

def align_channels(x, isosbestic, calcium, **kwargs) :
    
    """Function that performs the alignement of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The fitted isosbestic signal
                calcium (arr) = The standardized (or not) calcium signal
                kwargs (dict) = Dictionnary with the parameters
    """
    
    x_max = x[-1]
    
    if kwargs["photometry_pp"]["plots_to_display"][6] :
                
        fig = plt.figure(figsize=(10, 3), dpi=200.)
        ax0 = plt.subplot(111)
        b, = ax0.plot(x, calcium, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"], zorder=0)
        p, = ax0.plot(x, isosbestic, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"], zorder=1)
        ax0.plot((0, x[-1]), (0, 0), "--", color="black", lw=kwargs["lw"]) #Creates a horizontal dashed line at y = 0 to signal the baseline
        
        xticks, xticklabels, unit = ut.generate_xticks_and_labels(x_max)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, x_max)
        ax0.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
        
        if kwargs["photometry_pp"]["standardize"] :
            
            y_min, y_max, round_factor = ut.generate_yticks(np.concatenate((isosbestic,  calcium)), 0.1)
            ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
            ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
            ax0.set_ylim(y_min, y_max)
            ax0.set_ylabel("z-score", fontsize=kwargs["fsl"])
            
        else :
            
            y_min, y_max, round_factor = ut.generate_yticks(np.concatenate((isosbestic,  calcium)), 0.1)
            ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
            ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*100], fontsize=kwargs["fsl"])
            ax0.set_ylim(y_min, y_max)
            ax0.set_ylabel("Change in signal (%)", fontsize=kwargs["fsl"])

        ax0.legend(handles=[p, b], labels=["isosbestic", "calcium"], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title("Alignement of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "Alignement.{0}".format(kwargs["extension"])), dpi=200.)
            
def dFF(x, isosbestic, calcium, **kwargs) :
    
    """Function that computes the dF/F of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The fitted isosbestic signal
                calcium (arr) = The standardized (or not) calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   dFF (arr) = Relative changes of fluorescence over time 
    """
    
    print("\nStarting the computation of dF/F !")
    
    dFF = calcium - isosbestic #computing dF/F
    
    max_x = x[-1]
#    print("Data length : {0}".format(ut.h_m_s(max_x, add_tags=True)))
    
    if kwargs["photometry_pp"]["plots_to_display"][7] :
        
        xticks, xticklabels, unit = ut.generate_xticks_and_labels(max_x)
        
        fig = plt.figure(figsize=(10, 3), dpi=200.)
        ax0 = plt.subplot(111)
        g, = ax0.plot(x, dFF, alpha=0.8, c="green", lw=kwargs["lw"])
        ax0.plot((0, x[-1]), (0, 0), "--", color="black", lw=kwargs["lw"]) #Creates a horizontal dashed line at y = 0 to signal the baseline
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])
        
        if kwargs["photometry_pp"]["standardize"] :
            
            y_min, y_max, round_factor = ut.generate_yticks(dFF, 0.1)
            ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
            ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
            ax0.set_ylim(y_min, y_max)
            ax0.set_ylabel(r"z-score $\Delta$F/F", fontsize=kwargs["fsl"])
            ax0.legend(handles=[g], labels=[r"z-score $\Delta$F/F"], loc=2, fontsize=kwargs["fsl"])
            ax0.set_title(r"z-score $\Delta$F/F", fontsize=kwargs["fst"])
            
        else :
            
            y_min, y_max, round_factor = ut.generate_yticks(dFF, 0.1)
            ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
            ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)*100], fontsize=kwargs["fsl"])
            ax0.set_ylim(y_min, y_max)
            ax0.set_ylabel(r"$\Delta$F/F", fontsize=kwargs["fsl"])
            ax0.legend(handles=[g], labels=[r"$\Delta$F/F"], loc=2, fontsize=kwargs["fsl"])
            ax0.set_title(r"$\Delta$F/F", fontsize=kwargs["fst"])
            
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        
        if kwargs["save"] :
            
            plt.savefig(os.path.join(kwargs["save_dir"], "dFF.{0}".format(kwargs["extension"])), dpi=200.)
            
    return dFF
    
def load_photometry_data(file, **kwargs) :
    
    """Function that runs all the pre-processing steps on the raw isosbestic and
    calcium data and displays it in plot(s).
    
    Args :      file (arr) = The input photometry file for analysis
                kwargs (dict) = Dictionnary with the parameters

    Returns :   data (dict) = A dictionnary holding all the results from subsequent steps
                OR
                dFF (arr) = Relative changes of fluorescence over time 
    """
    
    x0, isosbestic, calcium = extract_raw_data(file, **kwargs)
    
    x1, isosbestic_smoothed, calcium_smoothed = smooth(x0, isosbestic, calcium, **kwargs)
    
    x2, isosbestic_cropped, calcium_cropped, function_isosbestic, function_calcium = find_baseline(x1, isosbestic_smoothed, calcium_smoothed, **kwargs)
    
    isosbestic_corrected, calcium_corrected = baseline_correction(x2, isosbestic_cropped, calcium_cropped, function_isosbestic, function_calcium, **kwargs)

    isosbestic_standardized, calcium_standardized = standardization(x2, isosbestic_corrected, calcium_corrected, **kwargs)

    isosbestic_fitted = interchannel_regression(isosbestic_standardized, calcium_standardized, **kwargs)
        
    align_channels(x2, isosbestic_fitted, calcium_standardized, **kwargs)
    
    dF = dFF(x2, isosbestic_fitted, calcium_standardized, **kwargs)
    
    time_lost = (len(x0) - len(x2))/kwargs["recording_sampling_rate"]
        
    data = {"raw" : {"x" : x0,
                     "isosbestic" : isosbestic,
                     "calcium" : calcium,
                    },
            "smoothed" : {"x" : x1,
                          "isosbestic" : isosbestic_smoothed,
                          "calcium" : calcium_smoothed,
                          },
            "cropped" : {"x" : x2,
                         "isosbestic" : isosbestic_cropped,
                         "calcium" : calcium_cropped,
                         "isosbestic_fc" : function_isosbestic,
                         "calcium_fc" : function_calcium,
                          },
            "baseline_correction" : {"x" : x2,
                                     "isosbestic" : isosbestic_corrected,
                                     "calcium" : calcium_corrected,
                                    },
            "standardization" : {"x" : x2,
                                 "isosbestic" : isosbestic_standardized,
                                 "calcium" : calcium_standardized,
                                },
            "regression" : {"fit" : isosbestic_fitted,
                            },
            "dFF" : {"x" : x2,
                     "dFF" : dF,
                    },
            "time_lost" : time_lost,
            }
            
    return data