#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:08:31 2020

@author: thomas.topilko
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd
import peakutils

from scipy.interpolate import interp1d

from sklearn.linear_model import Lasso, LinearRegression

import matplotlib.pyplot as plt

from fiber_photometry_analysis.generic_signal_processing import down_sample_signal, smooth_signal, crop_signal
from fiber_photometry_analysis.generic_signal_processing import baseline_asymmetric_least_squares_smoothing, airPLS
from fiber_photometry_analysis.plot import plot_data_pair, plot_cropped_data, plot_ca_iso_regression, \
    plot_delta_f, plot_aligned_channels

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
    logging.info("\nExtracting raw data for Isosbestic and Calcium recordings !")
    photometry_data = pd.read_feather(file_path)
    x = photometry_data["Time(s)"]
    isosbestic = photometry_data["AIn-1 - Dem (AOut-{})".format(params["general"]["photometry"]["isosbestic_channel"])]
    calcium = photometry_data["AIn-1 - Dem (AOut-{})".format(params["general"]["photometry"]["calcium_channel"])]

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
    logging.info("\nStarting smoothing for Isosbestic and Calcium signals !")
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
    logging.info("\nStarting baseline computation for Isosbestic and Calcium signals !")

    x = crop_signal(x,
                    params["general"]["recording_sampling_rate"],
                    crop_start=params["signal_pp"]["general"]["crop_start"],
                    crop_end=params["signal_pp"]["general"]["crop_end"])
    x = x - x.iloc[0]
    isosbestic = crop_signal(isosbestic,
                             params["general"]["recording_sampling_rate"],
                             crop_start=params["signal_pp"]["general"]["crop_start"],
                             crop_end=params["signal_pp"]["general"]["crop_end"])
    calcium = crop_signal(calcium,
                          params["general"]["recording_sampling_rate"],
                          crop_start=params["signal_pp"]["general"]["crop_start"],
                          crop_end=params["signal_pp"]["general"]["crop_end"])

    if params["signal_pp"]["baseline_determination"]["method"] == "als":
        isosbestic_fc = baseline_asymmetric_least_squares_smoothing(isosbestic,
                                                                    params["signal_pp"]["baseline_determination"]["als_lambda"],
                                                                    params["signal_pp"]["baseline_determination"]["als_pval"])
        calcium_fc = baseline_asymmetric_least_squares_smoothing(calcium,
                                                                 params["signal_pp"]["baseline_determination"]["als_lambda"],
                                                                 params["signal_pp"]["baseline_determination"]["als_pval"])
    else:
        isosbestic_fc = peakutils.baseline(isosbestic, deg=4)
        calcium_fc = peakutils.baseline(calcium, deg=4)
        
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
    logging.info("\nStarting baseline correction for Isosbestic and Calcium signals !")
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
        logging.info("\nStarting standardization for Isosbestic and Calcium signals !")
    
        isosbestic_standardized = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)  # standardization correction for isosbestic
        calcium_standardized = (calcium - np.median(calcium)) / np.std(calcium)  # standardization for calcium
    else:
        logging.error("\nThe standardization step in skipped."
                      " Parameter plot_args['photometry_pp']['standardize'] == False")
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
    logging.info("\nStarting inter-channel regression and alignment for Isosbestic and Calcium signals !")
    
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


def compute_delta_f(isosbestic, calcium):
    """
    Function that computes the dF/F of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.


    :param np.array isosbestic:  The standardized (or not) isosbestic signal
    :param np.array calcium: The standardized (or not) calcium signal
    :rtype: np.array
    """
    logging.info("\nStarting the computation of dF/F !")
    delta_f = calcium - isosbestic
    return delta_f


def load_photometry_data(photometry_data_file_path, params, recompute=True):
    """
    Function that runs all the pre-processing steps on the raw isosbestic and
    calcium data and displays it in plot(s).
    
    :param recompute:
    :param str photometry_data_file_path: The input photometry file for analysis
    :param dict params: Dictionary with the parameters
    :return: data (dict) = A dictionary holding all the results from subsequent steps
             OR
             dFF (arr) = Relative changes of fluorescence over time
    """
    dir_name = os.path.dirname(photometry_data_file_path)
    file_name = os.path.splitext(os.path.basename(photometry_data_file_path))[0]
    pickle_path = os.path.join(dir_name, "processed_photometry_data.pickle".format(file_name))
    if recompute:
        x0, isosbestic, calcium = extract_raw_data(photometry_data_file_path, params)
        if params["signal_pp"]["raw_data"]["plot"]:
            plot_data_pair(calcium, isosbestic, x0, 'raw', params, units='mV', to_milli=True,
                           save=params["signal_pp"]["raw_data"]["save"])

        x1, isosbestic_smoothed, calcium_smoothed = smooth(x0, isosbestic, calcium,
                                                           win_len=params["signal_pp"]["smoothing"]["smoothing_window"])
        if params["signal_pp"]["smoothing"]["plot"]:
            plot_data_pair(calcium_smoothed, isosbestic_smoothed, x1, 'smoothed', params, to_milli=True,
                           save=params["signal_pp"]["smoothing"]["save"])

        x2, isosbestic_cropped, calcium_cropped, function_isosbestic, function_calcium = find_baseline_and_crop(x1,
                                                                                                                isosbestic_smoothed,
                                                                                                                calcium_smoothed,
                                                                                                                params)
        if params["signal_pp"]["baseline_determination"]["plot"]:
            plot_cropped_data(calcium_cropped, function_calcium, isosbestic_cropped, function_isosbestic, x2, params)

        isosbestic_corrected, calcium_corrected = baseline_correction(isosbestic_cropped, calcium_cropped,
                                                                      function_isosbestic, function_calcium)
        if params["signal_pp"]["baseline_correction"]["plot"]:
            plot_data_pair(calcium_corrected, isosbestic_corrected, x2, 'baseline_corrected', params,
                           add_zero_line=True, to_milli=True, save=params["signal_pp"]["baseline_correction"]["save"])

        isosbestic_standardized, calcium_standardized = standardization(isosbestic_corrected, calcium_corrected,
                                                                        params["signal_pp"]["standardization"]["standardize"])
        if params["signal_pp"]["standardization"]["plot"]:
            plot_data_pair(calcium_standardized, isosbestic_standardized, x2, 'Standardized', params,
                           add_zero_line=True, units='z-score', save=params["signal_pp"]["standardization"]["save"])

        isosbestic_fitted = interchannel_regression(isosbestic_standardized, calcium_standardized,
                                                    regression_type=params["signal_pp"]["inter_channel_regression"]["method"])
        if params["signal_pp"]["inter_channel_regression"]["plot"]:
            plot_ca_iso_regression(isosbestic_standardized, calcium_standardized, isosbestic_fitted, params,
                                   save=params["signal_pp"]["inter_channel_regression"]["save"])

        if params["signal_pp"]["channel_alignement"]["plot"]:
            plot_aligned_channels(x2, isosbestic_fitted, calcium_standardized, params,
                                  save=params["signal_pp"]["channel_alignement"]["save"])

        delta_f = compute_delta_f(isosbestic_fitted, calcium_standardized)
        if params["signal_pp"]["delta_F"]["plot"]:
            plot_delta_f(calcium_standardized, isosbestic_fitted, x2, params,
                         save=params["signal_pp"]["delta_F"]["save"])

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
            # "time_lost": time_lost
        }

        with open(pickle_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return data
    else:
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as handle:
                data = pickle.load(handle)
            return data
        else:
            data = load_photometry_data(photometry_data_file_path, params, recompute=True)
            return data