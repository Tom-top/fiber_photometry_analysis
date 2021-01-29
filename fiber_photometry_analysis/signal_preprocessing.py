#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:08:31 2020

@author: thomas.topilko
"""

import os

import numpy as np

from scipy.interpolate import interp1d

from sklearn.linear_model import Lasso, LinearRegression

import matplotlib.pyplot as plt

from fiber_photometry_analysis.generic_signal_processing import down_sample_signal, smooth_signal, crop_signal
from fiber_photometry_analysis.generic_signal_processing import baseline_asymmetric_least_squares_smoothing
from fiber_photometry_analysis.plot import plot_data_pair, plot_cropped_data, plot_ca_iso_regression, add_line_at_zero

plt.style.use("default")

from fiber_photometry_analysis import utilities as utils


def extract_raw_data(file, **kwargs):
    """Function that extracts the raw data from the csv file and displays it
    in a plot.

    Args :      file (str) = The input photometry file for analysis
                kwargs (dict) = Dictionnary with the parameters

    Returns :   x (arr) = The time data in X
                isosbestic_adjusted (arr) = The adjusted isosbestic signal (time fitted to the video)
                calcium_adjusted (arr) = The adjusted calcium signal (time fitted to the video)
    """

    print("\nExtracting raw data for Isosbestic and Calcium recordings !")

    photometry_data = np.load(file)  # Load the NPY file

    x = photometry_data[0]  # time to be extracted
    isosbestic_adjusted = photometry_data[1]  # data compressed in time
    calcium_adjusted = photometry_data[2]  # data compressed in time
    plot_data_pair(calcium_adjusted, isosbestic_adjusted, 'raw', kwargs, x, units='mV', to_kilo=True)

    return x, isosbestic_adjusted, calcium_adjusted


def adjust_signal_to_video_time(time_video, time_final, source):
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


def down_sample(x, isosbestic, calcium, factor):
    """Downsample the two channels using a certain factor.

    Args :  x (arr) = The time data in X
            isosbestic (arr) = The adjusted isosbestic signal (time fitted to the video)
            calcium (arr) = The adjusted calcium signal (time fitted to the video)
            factor (int) = The factor of down sampling

    Returns : sink = The downsampled signal
    """

    delta = len(x) % factor

    x = x[:-delta][::factor]
    isosbestic = down_sample_signal(isosbestic[:-delta], factor)
    calcium = down_sample_signal(calcium[:-delta], factor)

    return x, isosbestic, calcium


def smooth(x, isosbestic, calcium, **kwargs):
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
    
    isosbestic_smoothed = smooth_signal(isosbestic, window_len=kwargs["smoothing_window"])[:-kwargs["smoothing_window"] + 1]
    calcium_smoothed = smooth_signal(calcium, window_len=kwargs["smoothing_window"])[:-kwargs["smoothing_window"] + 1]

    plot_data_pair(calcium_smoothed, isosbestic_smoothed, 'smoothed', kwargs, x, to_kilo=True)
        
    return x, isosbestic_smoothed, calcium_smoothed


def find_baseline_and_crop(x, isosbestic, calcium, **kwargs):
    """Function that estimates the baseline of the smoothed signals and 
    displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The smoothed isosbestic signal (time fitted to the video)
                calcium (arr) = The smoothed calcium signal (time fitted to the video)
                kwargs (dict) = Dictionnary with the parameters

    Returns :   x (arr) = The time data in X
                isosbestic (arr) = The cropped isosbestic signal
                calcium (arr) = The cropped calcium signal
                isosbestic_fc (arr) = The baseline for the isosbestic signal
                calcium_fc (arr) = The baseline for the calcium signal
    """
    
    print("\nStarting baseline computation for Isosbestic and Calcium signals !")

    x = crop_signal(x, int(kwargs["cropping_window"]))
    x = x - x[0]
    isosbestic = crop_signal(isosbestic, int(kwargs["cropping_window"]))
    calcium = crop_signal(calcium, int(kwargs["cropping_window"]))
    isosbestic_fc = baseline_asymmetric_least_squares_smoothing(isosbestic, kwargs["lambda"], kwargs["p"])
    calcium_fc = baseline_asymmetric_least_squares_smoothing(calcium, kwargs["lambda"], kwargs["p"])

    plot_cropped_data(calcium, calcium_fc, isosbestic, isosbestic_fc, kwargs, x)
        
    return x, isosbestic, calcium, isosbestic_fc, calcium_fc


def baseline_correction(x, isosbestic, calcium, isosbestic_fc, calcium_fc, **kwargs):
    """Function that performs the basleine correction of the smoothed and cropped 
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The cropped isosbestic signal
                calcium (arr) = The cropped calcium signal
                isosbestic_fc (arr) = The baseline for the isosbestic signal
                calcium_fc (arr) = The baseline for the calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   isosbestic_corrected (arr) = The baseline corrected isosbestic signal
                calcium_corrected (arr) = The baseline corrected calcium signal
    """
    
    print("\nStarting baseline correcion for Isosbestic and Calcium signals !")
    
    isosbestic_corrected = (isosbestic - isosbestic_fc) / isosbestic_fc  # baseline correction for isosbestic
    calcium_corrected = (calcium - calcium_fc) / calcium_fc  # baseline correction for calcium

    plot_data_pair(calcium_corrected, isosbestic_corrected, 'baseline_corrected', kwargs, x,
                   add_zero_line=True, to_kilo=True)
        
    return isosbestic_corrected, calcium_corrected


def standardization(x, isosbestic, calcium, **kwargs):
    """Function that performs the standardization of the corrected 
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The baseline corrected isosbestic signal
                calcium (arr) = The baseline corrected calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   isosbestic_standardized (arr) = The standardized isosbestic signal
                calcium_standardized (arr) = The standardized calcium signal
    """
    
    if kwargs["photometry_pp"]["standardize"]:
        print("\nStarting standardization for Isosbestic and Calcium signals !")
    
        isosbestic_standardized = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)  # standardization correction for isosbestic
        calcium_standardized = (calcium - np.median(calcium)) / np.std(calcium)  # standardization for calcium

        plot_data_pair(calcium_standardized, isosbestic_standardized, 'Standardized', kwargs, x,
                       add_zero_line=True, units='z-score')
    else:
        utils.print_in_color("\nThe standardization step in skipped."
                             " Parameter plot_args['photometry_pp']['standardize'] == False", "RED")
        isosbestic_standardized = isosbestic  # standardization skipped
        calcium_standardized = calcium  # standardization skipped
        
    return isosbestic_standardized, calcium_standardized


def interchannel_regression(isosbestic, calcium, **kwargs):
    """Function that performs the inter-channel regression of the two 
    signals and displays it in a plot.
    
    Args :      isosbestic (arr) = The standardized (or not) isosbestic signal
                calcium (arr) = The standardized (or not) calcium signal
                kwargs (dict) = Dictionnary with the parameters

    Returns :   isosbestic_fitted (arr) = The fitted isosbestic signal
    """

    print("\nStarting interchannel regression and alignement for Isosbestic and Calcium signals !")
    
    if kwargs["photometry_pp"]["regression"] == "Lasso":
        reg = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')
    else:
        reg = LinearRegression()
        
    n = len(calcium)
    reg.fit(isosbestic.reshape(n, 1), calcium.reshape(n, 1))
    # isosbestic_fitted = (abs(reg.coef_[0])*isosbestic.reshape(n,1)+reg.intercept_[0]).reshape(n,)
    isosbestic_fitted = reg.predict(isosbestic.reshape(n, 1)).reshape(n,)

    plot_ca_iso_regression(calcium, isosbestic, isosbestic_fitted, kwargs)
        
    return isosbestic_fitted


def align_channels(x, isosbestic, calcium, **kwargs):
    """Function that performs the alignement of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The fitted isosbestic signal
                calcium (arr) = The standardized (or not) calcium signal
                kwargs (dict) = Dictionnary with the parameters
    """
    if kwargs["photometry_pp"]["plots_to_display"]["channel_alignement"]:
        max_x = x[-1]

        plt.figure(figsize=(10, 3), dpi=200.)
        ax0 = plt.subplot(111)
        b, = ax0.plot(x, calcium, alpha=0.8, c=kwargs["photometry_pp"]["blue_laser"], lw=kwargs["lw"], zorder=0)
        p, = ax0.plot(x, isosbestic, alpha=0.8, c=kwargs["photometry_pp"]["purple_laser"], lw=kwargs["lw"], zorder=1)
        add_line_at_zero(ax0, x, kwargs["lw"])

        xticks, xticklabels, unit = utils.generate_xticks_and_labels(max_x)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

        y_min, y_max, round_factor = utils.generate_yticks(np.concatenate((isosbestic, calcium)), 0.1)
        ax0.set_yticks(np.arange(y_min, y_max + round_factor, round_factor))
        if kwargs["photometry_pp"]["standardize"]:
            multiplication_factor = 1
            title = "z-score"
        else:
            multiplication_factor = 100
            title = "Change in signal (%)"

        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor) * multiplication_factor],
                            fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel(title, fontsize=kwargs["fsl"])
        ax0.legend(handles=[p, b], labels=["isosbestic", "calcium"], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title("Alignement of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        plt.tight_layout()

        if kwargs["save"]:
            plt.savefig(os.path.join(kwargs["save_dir"], "Alignement.{0}".format(kwargs["extension"])), dpi=200.)


def dFF(x, isosbestic, calcium, **kwargs):
    """Function that computes the dF/F of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.
    
    Args :      x (arr) = The time data in X
                isosbestic (arr) = The fitted isosbestic signal
                calcium (arr) = The standardized (or not) calcium signal
                kwargs (dict) = Dictionary with the parameters

    Returns :   dFF (arr) = Relative changes of fluorescence over time 
    """
    
    print("\nStarting the computation of dF/F !")
    
    df_f = calcium - isosbestic  # computing dF/F
    
    if kwargs["photometry_pp"]["plots_to_display"]["dFF"]:
        max_x = x[-1]

        plt.figure(figsize=(10, 3), dpi=200.)
        ax0 = plt.subplot(111)
        g, = ax0.plot(x, df_f, alpha=0.8, c="green", lw=kwargs["lw"])
        add_line_at_zero(ax0, x, kwargs["lw"])

        xticks, xticklabels, unit = utils.generate_xticks_and_labels(max_x)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

        y_min, y_max, round_factor = utils.generate_yticks(df_f, 0.1)
        ax0.set_yticks(np.arange(y_min, y_max + round_factor, round_factor))
        if kwargs["photometry_pp"]["standardize"]:
            multiplication_factor = 1
            title = r"z-score $\Delta$F/F"
        else:
            multiplication_factor = 100
            title = r"$\Delta$F/F"

        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor) * multiplication_factor],
                            fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel(title, fontsize=kwargs["fsl"])
        ax0.legend(handles=[g], labels=[title], loc=2, fontsize=kwargs["fsl"])
        ax0.set_title(title, fontsize=kwargs["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
        plt.tight_layout()

        if kwargs["save"]:
            plt.savefig(os.path.join(kwargs["save_dir"], "dFF.{0}".format(kwargs["extension"])), dpi=200.)
            
    return df_f


def load_photometry_data(file, **kwargs):
    """Function that runs all the pre-processing steps on the raw isosbestic and
    calcium data and displays it in plot(s).
    
    Args :      file (arr) = The input photometry file for analysis
                kwargs (dict) = Dictionary with the parameters

    Returns :   data (dict) = A dictionary holding all the results from subsequent steps
                OR
                dFF (arr) = Relative changes of fluorescence over time 
    """
    
    x0, isosbestic, calcium = extract_raw_data(file, **kwargs)

    x1, isosbestic_smoothed, calcium_smoothed = smooth(x0, isosbestic, calcium, **kwargs)
    
    x2, isosbestic_cropped, calcium_cropped, function_isosbestic, function_calcium = find_baseline_and_crop(x1,
                                                                                                            isosbestic_smoothed,
                                                                                                            calcium_smoothed,
                                                                                                            **kwargs)
    
    isosbestic_corrected, calcium_corrected = baseline_correction(x2, isosbestic_cropped, calcium_cropped, function_isosbestic, function_calcium, **kwargs)

    isosbestic_standardized, calcium_standardized = standardization(x2, isosbestic_corrected, calcium_corrected, **kwargs)

    isosbestic_fitted = interchannel_regression(isosbestic_standardized, calcium_standardized, **kwargs)
        
    align_channels(x2, isosbestic_fitted, calcium_standardized, **kwargs)
    
    dF = dFF(x2, isosbestic_fitted, calcium_standardized, **kwargs)
    
    time_lost = (len(x0) - len(x2))/kwargs["recording_sampling_rate"]
        
    data = {
        "raw": {"x": x0,
                "isosbestic": isosbestic,
                "calcium": calcium,
                },
        "smoothed": {"x": x1,
                     "isosbestic": isosbestic_smoothed,
                     "calcium": calcium_smoothed,
                     },
        "cropped": {"x": x2,
                    "isosbestic": isosbestic_cropped,
                    "calcium": calcium_cropped,
                    "isosbestic_fc": function_isosbestic,
                    "calcium_fc": function_calcium,
                    },
        "baseline_correction": {"x": x2,
                                "isosbestic": isosbestic_corrected,
                                "calcium": calcium_corrected,
                                },
        "standardization": {"x": x2,
                            "isosbestic": isosbestic_standardized,
                            "calcium": calcium_standardized,
                            },
        "regression": {"fit": isosbestic_fitted},
        "dFF": {"x": x2,
                "dFF": dF,
                },
        "time_lost": time_lost
    }
            
    return data
