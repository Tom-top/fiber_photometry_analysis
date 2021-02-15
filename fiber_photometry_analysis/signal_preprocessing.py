#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:08:31 2020

@author: thomas.topilko
"""

import os

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from sklearn.linear_model import Lasso, LinearRegression

import matplotlib.pyplot as plt

from fiber_photometry_analysis.generic_signal_processing import down_sample_signal, smooth_signal, crop_signal
from fiber_photometry_analysis.generic_signal_processing import baseline_asymmetric_least_squares_smoothing
from fiber_photometry_analysis.plot import plot_data_pair, plot_cropped_data, plot_ca_iso_regression, add_line_at_zero

plt.style.use("default")

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import photometry_io as io


def extract_raw_data(file_path, params):
    """
    Function that extracts the raw data from the csv file and displays it
    in a plot.

    :param str file_path: The input photometry file for analysis
    :param dict params: Dictionary with the parameters
    :return: x : The time data in X
            isosbestic_adjusted : The adjusted isosbestic signal (time fitted to the video)
            calcium_adjusted : The adjusted calcium signal (time fitted to the video)
    :rtype: (np.array, np.array, np.array)
    """
    print("\nExtracting raw data for Isosbestic and Calcium recordings !")
    photometry_data = pd.read_feather(file_path)
    x = photometry_data["Time(s)"]
    isosbestic = photometry_data["AIn-1 - Dem (AOut-{})".format(kwargs["isosbestic_channel"])]
    calcium = photometry_data["AIn-1 - Dem (AOut-{})".format(kwargs["calcium_channel"])]
    plot_data_pair(calcium, isosbestic, 'raw', kwargs, x, units='mV', to_kilo=True)

    return x, isosbestic, calcium


def adjust_signal_to_video_time(time_video, time_final, source):
    """
    Function that adjusts the time stamps of the source signal to the time
    stamps of the video. It then extracts the signal with the same sampling rate
    as the video to simplify the analysis with videos.


    :param np.array time_video:  The time of the video (photometry sampling rate)
    :param np.array time_final:   The time of the video (real fps recording)
    :param np.array source: The signal to be adjusted
    :return: The adjusted signal
    :rtype: np.array
    """
    spl = interp1d(time_video, source)
    return spl(time_final)


def down_sample(x, isosbestic, calcium, factor):
    """
    Downsample the two channels using a certain factor.

    :param np.array x: The time data in X
    :param np.array isosbestic: The adjusted isosbestic signal (time fitted to the video)
    :param np.array calcium: The adjusted calcium signal (time fitted to the video)
    :param int factor: The factor of down sampling
    :return: The downsampled signal
    """
    delta = len(x) % factor

    x = x[:-delta][::factor]  # FIXME: inplace
    isosbestic = down_sample_signal(isosbestic[:-delta], factor)
    calcium = down_sample_signal(calcium[:-delta], factor)

    return x, isosbestic, calcium


def smooth(x, isosbestic, calcium, win_len=1):
    """
    Function that smooths the raw data and displays it in a plot.

    :param np.array x:  The time data in X
    :param np.array isosbestic: The adjusted isosbestic signal (time fitted to the video)
    :param np.array calcium: The adjusted calcium signal (time fitted to the video)
    :param int win_len: The size of the window to smooth
    :returns: x : The new time data in X
              isosbestic_smoothed : The smoothed isosbestic signal
                calcium_smoothed : The smoothed calcium signal
    :rtype: (np.array, np.array, np.array)
    """
    print("\nStarting smoothing for Isosbestic and Calcium signals !")
    isosbestic_smoothed = smooth_signal(isosbestic, window_len=win_len)[: - win_len + 1]
    calcium_smoothed = smooth_signal(calcium, window_len=win_len)[: - win_len + 1]
        
    return x, isosbestic_smoothed, calcium_smoothed


def find_baseline_and_crop(x, isosbestic, calcium, params):  # WARNING: does 2 things + docstring does not match
    """
    Function that estimates the baseline of the smoothed signals.

    :param np.array x: The time data in X
    :param np.array isosbestic: The smoothed isosbestic signal (time fitted to the video)
    :param np.array calcium: The smoothed calcium signal (time fitted to the video)
    :param dict params: Dictionary with the parameters
    :return: x : The time data in X
            isosbestic : The cropped isosbestic signal
            calcium : The cropped calcium signal
            isosbestic_fc : The baseline for the isosbestic signal
            calcium_fc : The baseline for the calcium signal
    :rtype: (np.array, np.array, np.array, np.array, np.array)
    """
    print("\nStarting baseline computation for Isosbestic and Calcium signals !")

    x = crop_signal(x,
                    params["recording_sampling_rate"],
                    crop_start=params["crop_start"],
                    crop_end=params["crop_end"])
    x = x - x.iloc[0]
    isosbestic = crop_signal(isosbestic,
                             params["recording_sampling_rate"],
                             crop_start=params["crop_start"],
                             crop_end=params["crop_end"])
    calcium = crop_signal(calcium,
                          params["recording_sampling_rate"],
                          crop_start=params["crop_start"],
                          crop_end=params["crop_end"])

    isosbestic_fc = baseline_asymmetric_least_squares_smoothing(isosbestic, params["lambda"], params["p"])
    calcium_fc = baseline_asymmetric_least_squares_smoothing(calcium, params["lambda"], params["p"])
        
    return x, isosbestic, calcium, isosbestic_fc, calcium_fc


def baseline_correction(isosbestic, calcium, isosbestic_fc, calcium_fc):
    """
    Function that performs the baseline correction of the smoothed and cropped
    signals and displays it in a plot.

d isosbestic signal
    :param np.array calcium: The cropped calcium signal
    :param np.array isosbestic_fc: The baseline for the isosbestic signal
    :param np.array calcium_fc: The baseline for the calcium signal
    :return: isosbestic_corrected : The baseline corrected isosbestic signal
             calcium_corrected : The baseline corrected calcium signal
    :rtype: (np.array, np.array)
    """
    print("\nStarting baseline correction for Isosbestic and Calcium signals !")
    isosbestic_corrected = (isosbestic - isosbestic_fc) / isosbestic_fc  # baseline correction for isosbestic
    calcium_corrected = (calcium - calcium_fc) / calcium_fc  # baseline correction for calcium
    return isosbestic_corrected, calcium_corrected


def standardization(isosbestic, calcium, standardise):
    """
    Function that performs the standardization of the corrected
    signals and displays it in a plot.

    :param np.array isosbestic: The corrected isosbestic signal
    :param np.array calcium: The baseline corrected calcium signal
    :param bool standardise: Whether to actually do the standardisation
    :return: isosbestic_standardized : The standardized isosbestic signal
             calcium_standardized : The standardized calcium signal
    :rtype: (np.array, np.array)
    """
    if standardise:
        print("\nStarting standardization for Isosbestic and Calcium signals !")
    
        isosbestic_standardized = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)  # standardization correction for isosbestic
        calcium_standardized = (calcium - np.median(calcium)) / np.std(calcium)  # standardization for calcium
    else:
        utils.print_in_color("\nThe standardization step in skipped."
                             " Parameter plot_args['photometry_pp']['standardize'] == False", "RED")
        isosbestic_standardized = isosbestic  # standardization skipped
        calcium_standardized = calcium  # standardization skipped
        
    return isosbestic_standardized, calcium_standardized


def interchannel_regression(isosbestic, calcium, regression_type):
    """
    Function that performs the inter-channel regression of the two
    signals and displays it in a plot.

    :param np.array isosbestic: The standardized (or not) isosbestic signal
    :param np.array calcium: The standardized (or not) calcium signal
    :param str regression_type:  The type of regression (one of ('Lasso', 'Linear'))
    :return: isosbestic_fitted : The fitted isosbestic signal
    :rtype: np.array
    """
    print("\nStarting inter-channel regression and alignment for Isosbestic and Calcium signals !")
    
    if regression_type.lower() == "lasso":
        reg = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')  # WARNING: magic numbers
    else:
        reg = LinearRegression()
        
    n = len(calcium)
    reg.fit(isosbestic.reshape(n, 1), calcium.reshape(n, 1))
    # isosbestic_fitted = (abs(reg.coef_[0])*isosbestic.reshape(n,1)+reg.intercept_[0]).reshape(n,)
    isosbestic_fitted = reg.predict(isosbestic.reshape(n, 1)).reshape(n,)
        
    return isosbestic_fitted


def align_channels(x, isosbestic, calcium, params):  # FIXME: plot only. Rename
    """
    Function that performs the alignment of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.

    :param np.array x: The time data in X
    :param np.array isosbestic: The fitted isosbestic signal
    :param np.array calcium: The standardized (or not) calcium signal
    :param dict params: Dictionary with the parameters
    """
    if params["photometry_pp"]["plots_to_display"]["channel_alignement"]:
        max_x = x.iloc[-1]

        plt.figure(figsize=(10, 3), dpi=200.)
        ax0 = plt.subplot(111)
        b, = ax0.plot(x, calcium, alpha=0.8, c=params["photometry_pp"]["blue_laser"], lw=params["lw"], zorder=0)
        p, = ax0.plot(x, isosbestic, alpha=0.8, c=params["photometry_pp"]["purple_laser"], lw=params["lw"], zorder=1)
        add_line_at_zero(ax0, x, params["lw"])

        xticks, xticklabels, unit = utils.generate_xticks_and_labels(max_x)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=params["fsl"])
        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time ({0})".format(unit), fontsize=params["fsl"])

        y_min, y_max, round_factor = utils.generate_yticks(np.concatenate((isosbestic, calcium)), 0.1)
        ax0.set_yticks(np.arange(y_min, y_max + round_factor, round_factor))
        if params["photometry_pp"]["standardize"]:
            multiplication_factor = 1
            title = "z-score"
        else:
            multiplication_factor = 100
            title = "Change in signal (%)"

        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor) * multiplication_factor],
                            fontsize=params["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel(title, fontsize=params["fsl"])
        ax0.legend(handles=[p, b], labels=["isosbestic", "calcium"], loc=2, fontsize=params["fsl"])
        ax0.set_title("Alignement of Isosbestic and Calcium signals", fontsize=params["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=params["fsl"])
        plt.tight_layout()

        if params["save"]:
            plt.savefig(os.path.join(params["save_dir"], "Alignement.{0}".format(params["extension"])), dpi=200.)


def compute_delta_f(x, isosbestic, calcium, params):
    """
    Function that computes the dF/F of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.


    :param np.array x: The time data in X
    :param np.array isosbestic: The fitted isosbestic signal
    :param np.array calcium: The standardized (or not) calcium signal
    :param dict params: Dictionary with the parameters
    :return: dFF : Relative changes of fluorescence over time
    :rtype: np.array
    """
    print("\nStarting the computation of dF/F !")
    
    df_f = calcium - isosbestic  # computing dF/F
    
    if params["photometry_pp"]["plots_to_display"]["dFF"]:
        max_x = x.iloc[-1]

        plt.figure(figsize=(10, 3), dpi=200.)
        ax0 = plt.subplot(111)
        g, = ax0.plot(x, df_f, alpha=0.8, c="green", lw=params["lw"])
        add_line_at_zero(ax0, x, params["lw"])

        xticks, xticklabels, unit = utils.generate_xticks_and_labels(max_x)
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticklabels, fontsize=params["fsl"])
        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time ({0})".format(unit), fontsize=params["fsl"])

        y_min, y_max, round_factor = utils.generate_yticks(df_f, 0.1)
        ax0.set_yticks(np.arange(y_min, y_max + round_factor, round_factor))
        if params["photometry_pp"]["standardize"]:
            multiplication_factor = 1
            title = r"z-score $\Delta$F/F"
        else:
            multiplication_factor = 100
            title = r"$\Delta$F/F"

        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor) * multiplication_factor],
                            fontsize=params["fsl"])
        ax0.set_ylim(y_min, y_max)
        ax0.set_ylabel(title, fontsize=params["fsl"])
        ax0.legend(handles=[g], labels=[title], loc=2, fontsize=params["fsl"])
        ax0.set_title(title, fontsize=params["fst"])
        ax0.tick_params(axis='both', which='major', labelsize=params["fsl"])
        plt.tight_layout()

        if params["save"]:
            plt.savefig(os.path.join(params["save_dir"], "dFF.{0}".format(params["extension"])), dpi=200.)
            
    return df_f


def load_photometry_data(photometry_data_file_path, params):
    """
    Function that runs all the pre-processing steps on the raw isosbestic and
    calcium data and displays it in plot(s).
    
    :param str photometry_data_file_path: The input photometry file for analysis
    :param dict params: Dictionary with the parameters
    :return: data (dict) = A dictionary holding all the results from subsequent steps
             OR
             dFF (arr) = Relative changes of fluorescence over time
    """
    x0, isosbestic, calcium = extract_raw_data(photometry_data_file_path, params)
    plot_data_pair(calcium, isosbestic, 'raw', params, x0, units='mV', to_kilo=True)

    x1, isosbestic_smoothed, calcium_smoothed = smooth(x0, isosbestic, calcium, win_len=params["smoothing_window"])
    plot_data_pair(calcium_smoothed, isosbestic_smoothed, 'smoothed', params, x1, to_kilo=True)
    
    x2, isosbestic_cropped, calcium_cropped, function_isosbestic, function_calcium = find_baseline_and_crop(x1,
                                                                                                            isosbestic_smoothed,
                                                                                                            calcium_smoothed,
                                                                                                            params)
    plot_cropped_data(calcium, function_calcium, isosbestic, function_isosbestic, params, x2)
    
    isosbestic_corrected, calcium_corrected = baseline_correction(isosbestic_cropped, calcium_cropped,
                                                                  function_isosbestic, function_calcium)
    plot_data_pair(calcium_corrected, isosbestic_corrected, 'baseline_corrected',
                   params, x2, add_zero_line=True, to_kilo=True)

    isosbestic_standardized, calcium_standardized = standardization(isosbestic_corrected, calcium_corrected,
                                                                    params["photometry_pp"]["standardize"])
    plot_data_pair(calcium_standardized, isosbestic_standardized, 'Standardized',
                   params, x2, add_zero_line=True, units='z-score')

    isosbestic_fitted = interchannel_regression(isosbestic_standardized, calcium_standardized,
                                                regression_type=params["photometry_pp"]["regression"])
    plot_ca_iso_regression(isosbestic_standardized, calcium_standardized, isosbestic_fitted, params)
        
    align_channels(x2, isosbestic_fitted, calcium_standardized, **params)
    
    delta_f = compute_delta_f(x2, isosbestic_fitted, calcium_standardized, **params)
    
    sampling_rate = params["recording_sampling_rate"]
    time_lost = (len(x0) - len(x2))/sampling_rate
    raw_isosbestic_cropped, raw_calcium_cropped =\
    [crop_signal(i, sampling_rate, crop_start=params["crop_start"], crop_end=params["crop_end"])
     for i in [isosbestic, calcium]]

    df = pd.DataFrame({"time": x2,
                    "raw_isosbestic_cropped": raw_isosbestic_cropped,
                    "isosbestic_cropped" : isosbestic_cropped,
                    "isosbestic_corrected" : isosbestic_corrected,
                    "isosbestic_standardized" : isosbestic_standardized,
                    "isosbestic_fitted" : isosbestic_fitted,
                    "raw_calcium_cropped": raw_isosbestic_cropped,
                    "calcium_cropped": isosbestic_cropped,
                    "calcium_cropped": isosbestic_corrected,
                    "calcium_standardized": isosbestic_standardized,
                    "dF": dF,
                    })

    df = io.reset_dataframe_index(df)
    df.to_feather(os.path.join(kwargs["save_dir"], "photometry_data.feather"))
        
    data = {
        "raw": {
            "x": x0,
            "isosbestic": isosbestic,
            "calcium": calcium
        },
        "smoothed": {
            "x": x1,
            "isosbestic": isosbestic_smoothed,
            "calcium": calcium_smoothed
        },
        "cropped": {
            "x": x2,
            "isosbestic": isosbestic_cropped,
            "calcium": calcium_cropped,
            "isosbestic_fc": function_isosbestic,
            "calcium_fc": function_calcium
        },
        "baseline_correction": {
            "x": x2,
            "isosbestic": isosbestic_corrected,
            "calcium": calcium_corrected
        },
        "standardization": {
            "x": x2,
            "isosbestic": isosbestic_standardized,
            "calcium": calcium_standardized
        },
        "regression": {"fit": isosbestic_fitted},
        "dFF": {
            "x": x2,
            "dFF": delta_f
        },
        "time_lost": time_lost
    }
            
    return data
