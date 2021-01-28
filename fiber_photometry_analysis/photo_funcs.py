#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:40:32 2019

@author: thomas.topilko
"""

import os
import datetime

import numpy as np
import pandas as pd
import scipy.optimize as optimization
from scipy.interpolate import interp1d
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import MultiCursor


def hours_minutes_seconds(time):  # FIXME: seems duplicate + extract
    delta = time % 3600
    h = (time - delta) / 3600
    m = (delta - (delta % 60)) / 60
    s = (delta % 60)
    
    return h, m, s


def seconds(h, m, s):
    return h * 3600 + m * 60 + s


def convert_photometry_data(csv_file_path):
    npy_file_path = os.path.join(os.path.dirname(csv_file_path), "{0}.npy".format(os.path.basename(csv_file_path).split(".")[0]))
    
    if not os.path.exists(csv_file_path):
        raise RuntimeError("{0} file doesn't exist !".format(csv_file_path))  # FIXME: specific exceptions
    
    if not os.path.exists(npy_file_path):
        print("\n")
        print("Converting CSV photometry data into NPY")
    
        photometry_sheet = pd.read_csv(csv_file_path, header=1, usecols=np.arange(0, 3))  # Load the data
        non_nan_mask_ch1 = photometry_sheet["AIn-1 - Dem (AOut-1)"] > 0.  # Filtering NaN values (missing values)
        non_nan_mask_ch2 = photometry_sheet["AIn-2 - Dem (AOut-2)"] > 0.  # Filtering NaN values (missing values)
        non_nan_mask = np.logical_or(non_nan_mask_ch1, non_nan_mask_ch2)
        
        filtered_photometry_sheet = photometry_sheet[non_nan_mask]  # Filter the data
        photometry_data_numpy = filtered_photometry_sheet.to_numpy()  # Convert to numpy for speed

        np.save(npy_file_path, photometry_data_numpy)  # Save the data as numpy file
        print("Filetered : {0} points".format(len(photometry_sheet)-len(filtered_photometry_sheet)))
        return npy_file_path
    else:
        print("[WARNING] NPY file already existed, bypassing conversion step !")
        return npy_file_path


def check_time_shift(npy, SRD, plot=False, label=None):
    SRD = int(SRD)
    photometry_data = np.load(npy)  # Load the NPY file
    
    # Cropping the data to the last second
    remaining = len(photometry_data) % SRD
    crop_length = int(len(photometry_data) - remaining)
    
    cropped = photometry_data[0:crop_length+1]
    clone = cropped.copy()  # FIXME: unused
    
    exp_time = []
    fixed_time = []  # FIXME: unused
    # step = int(SRD / 2)
    
    for n in np.arange(SRD, len(cropped), SRD-1):
        exp_time.append(1 - (cropped[:, 0][n-1] - cropped[:, 0][n-SRD]))
        # dt = (1 - (cropped[:,0][n-1] - cropped[:,0][n-SRD])) / SRD
        # clone[:,0][n-SRD : n-1] = clone[:,0][n-SRD : n-1]+dt
        # fixed_time.append(clone[:,0][n-1] - clone[:,0][n-SRD])
    
    exp_time = np.array(exp_time)
    # fixed_time = np.array(fixedTime);
    theo_time = np.arange(0, len(exp_time), 1)
    
    if plot:
        fig = plt.figure(figsize=(5, 3), dpi=200.)  # FIXME: unused
        ax0 = plt.subplot(1, 1, 1)
        
        ax0.scatter(theo_time, exp_time, s=0.1)
#        ax0.plot(np.cumsum(expTime))
        print(np.cumsum(exp_time))
        
        minimum = min(exp_time)
        maximum = max(exp_time)
        r = maximum-minimum
        print(minimum, maximum)
        
        ax0.set_ylim(minimum-r*0.1, maximum+r*0.1)
        
    ax0.set_title(label)


def moving_average(arr, n=1):
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    new_arr = ret[n - 1:] / n
    new_arr = np.insert(new_arr, len(new_arr), np.full(len(arr)-len(new_arr), new_arr[-1]))
    
    return new_arr


def trailing_moving_average(arr, window=1):
    cumsum = np.cumsum(arr)
    result = (cumsum[window:] - cumsum[:-window]) / float(window)
    return result


def centered_moving_average(arr, window=1):
    cumsum = np.cumsum(arr)
    result = (cumsum[window:] - cumsum[:-window]) / float(window)
    result = np.insert(result, 0, result[:int(window/2)])
    result = np.insert(result, len(result), result[len(result)-int(window/2):])
    return result


def lowpass_filter(data, SRD, freq, order):
    sampling_rate = SRD
    nyq = sampling_rate / 2
    cutoff = freq
    normalized_cutoff = cutoff / nyq
    
    z, p, k = signal.butter(order, normalized_cutoff, output="zpk")
    lesos = signal.zpk2sos(z, p, k)
    filtered = signal.sosfilt(lesos, data)
    filtered = np.array(filtered)
    
    return filtered


def interpolated_and_extract(compressed_x, final_x, source):
    spl = interp1d(compressed_x, source)
    sink = spl(final_x)
    
    return sink


def plot_intermediate_photometry_plots(data, **kwargs):
    purple_laser = "#8200c8"  # hex color of the calcium laser
    blue_laser = "#0092ff"  # hex color of the isosbestic laser
    
    if kwargs["function"] is None:
        sink = data
    elif kwargs["function"] == "low_pass":
        sink = []
        for n, d in enumerate(data):
            if n == 10:
                sink.append(lowpass_filter(d, kwargs["SRD"], kwargs["freq"], kwargs["order"]))
            else:
                sink.append(d)
    elif kwargs["function"] == "moving_avg":
        sink = []
        for n, d in enumerate(data):
            if n == 10:
                sink.append(moving_average(d, n=kwargs["roll_size"]))
            else:
                sink.append(d)
    plt.ion()
    
    for n, bol in enumerate(kwargs["which"]):
        if n == 0 and bol:
            fig = plt.figure(figsize=(15, 7), dpi=200.)  # FIXME: unused
            ax0 = plt.subplot(211)
            b, = ax0.plot(sink[0], sink[1], alpha=0.8, c=purple_laser, lw=kwargs["lw"])
            g, = ax0.plot(sink[0], sink[3], alpha=0.8, c="orange", lw=kwargs["lw"])
            ax0.legend(handles=[b, g], labels=["isosbestic", "moving average"], loc=2, fontsize=kwargs["fsl"])
            ax1 = plt.subplot(212)
            b, = ax1.plot(sink[0], sink[2], alpha=0.8, c=blue_laser, lw=kwargs["lw"])
            g, = ax1.plot(sink[0], sink[5], alpha=0.8, c="orange", lw=kwargs["lw"])
            ax1.legend(handles=[b, g], labels=["calcium", "moving average"], loc=2, fontsize=kwargs["fsl"])
            ax0.set_title("Raw Isosbestic and Calcium signals", fontsize=kwargs["fst"])
            ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"])
            ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            
            if kwargs["save"]:
                plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Raw.svg"), dpi=200.)
        elif n == 1 and bol:
            fig = plt.figure(figsize=(15, 7), dpi=200.)
            ax0 = plt.subplot(211)
            isos, = ax0.plot(sink[0], sink[4], alpha=0.8, c=purple_laser, lw=kwargs["lw"])
            ax1 = plt.subplot(212)
            calc, = ax1.plot(sink[0], sink[6], alpha=0.8, c=blue_laser, lw=kwargs["lw"])
            ax0.legend(handles=[isos, calc], labels=["405 signal", "465 signal"], loc=2, fontsize=kwargs["fsl"])
            ax0.set_title("Baseline Correction", fontsize=kwargs["fst"])
            ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"])
            ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            
            if kwargs["save"]:
                plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Corrected.svg"), dpi=200.)
        elif n == 2 and bol:
            fig = plt.figure(figsize=(15, 7), dpi=200.)
            ax0 = plt.subplot(211)
            isos, = ax0.plot(sink[0], sink[7], alpha=0.8, c=purple_laser, lw=kwargs["lw"])
            ax1 = plt.subplot(212)
            calc, = ax1.plot(sink[0], sink[8], alpha=0.8, c=blue_laser, lw=kwargs["lw"])
            ax0.legend(handles=[isos, calc], labels=["405 signal", "465 signal"], loc=2, fontsize=kwargs["fsl"])
            ax0.set_title("Standardization", fontsize=kwargs["fst"])
            ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"])
            ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            if kwargs["save"]:
                plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Standardized.svg"), dpi=200.)
        elif n == 3 and bol:
            fig = plt.figure(figsize=(15, 7), dpi=200.)  # FIXME: unused
            ax = plt.subplot(111)
            ax.scatter(moving_average(standardizedIsosbestic, n=resample),
                       moving_average(standardizedCalcium, n=resample), alpha=0.5, s=1.)
            ax.plot([min(moving_average(standardizedIsosbestic, n=resample)),
                     max(moving_average(standardizedCalcium, n=resample))],
                    line, '--', c="red")
            ax.set_xlabel("410 signal", fontsize=6.)
            ax.set_ylabel("465 signal", fontsize=6.)
            ax.set_title("Linear regression fit", fontsize=8.)
            ax.tick_params(axis='both', which='major', labelsize=6)
            if save:
                plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Fit.svg"), dpi=200.)
        elif n == 4 and bol:
            fig = plt.figure(figsize=(15, 7), dpi=200.)
            ax = plt.subplot(111)
            calc, = ax.plot(sink[0], sink[8], alpha=1., lw=kwargs["lw"],c=blue_laser)
            isos, = ax.plot(sink[0], sink[9], alpha=1., lw=kwargs["lw"],c=purple_laser)
            ax.plot([sink[0][0], sink[0][-1]], [0, 0], "--", color="red")
            ax.legend(handles=[isos, calc], labels=["fitted 405 signal", "465 signal"], loc=2, fontsize=kwargs["fsl"])
            ax.set_xlabel("Time (s)", fontsize=kwargs["fsl"])
            ax.set_title("Alignement of signals", fontsize=kwargs["fst"])
            ax.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            if kwargs["save"]:
                plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_Aligned.svg"), dpi=200.)
        elif n == 5 and bol:
            # plt.style.use("dark_background")
            fig = plt.figure(figsize=(15, 7), dpi=200.)
            ax = plt.subplot(211)
            df, = ax.plot(sink[0], sink[10], alpha=1., lw=kwargs["lw"], c="green")
            ax.plot([sink[0][0], sink[0][-1]], [0, 0], "--", color="red")
            ax.legend(handles=[df], labels=["dF/F"], loc=2, fontsize=kwargs["fsl"])
            ax.set_xlabel("Time (s)", fontsize=kwargs["fsl"])
            ax.set_title("dF/F", fontsize=kwargs["fst"])
            ax.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            # ax0 = plt.subplot(313, sharex=ax)
            # b, = ax0.plot(sink[0],sink[1],alpha=0.8,c=purpleLaser,lw=kwargs["lw"])
            # g, = ax0.plot(sink[0],sink[3],alpha=0.8,c="orange",lw=kwargs["lw"])
            # ax0.legend(handles=[b, g], labels=["isosbestic", "moving average"], loc=2, fontsize=kwargs["fsl"])
            # ax0.set_title("Raw Isosbestic and Calcium signals", fontsize=kwargs["fst"])
            # ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            ax1 = plt.subplot(212, sharex=ax)
            b, = ax1.plot(sink[0], sink[2], alpha=0.8, c=blue_laser, lw=kwargs["lw"])
            g, = ax1.plot(sink[0], sink[5], alpha=0.8, c="orange", lw=kwargs["lw"])
            ax1.legend(handles=[b, g], labels=["calcium", "moving average"], loc=2, fontsize=kwargs["fsl"])
            ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"])
            ax1.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])
            multi = MultiCursor(fig.canvas, (ax, ax1), color='r', lw=1, horizOn=True, vertOn=True)
            plt.tight_layout()
            plt.show()
            
            if kwargs["save"]:
                plt.savefig(os.path.join(os.path.expanduser("~")+"/Desktop", "Isosbestic_Calcium_dF.svg"), dpi=200.)


def load_photometry_data(npy, SRD, start, end, video_length, roll_avg, opt_plots=False, recompute=True, plot_args=None):
    SRD = int(SRD)  # sampling rate of the Doric system
    plot_args["SRD"] = SRD
    
    photometry_data = np.load(npy)  # Load the NPY file
    
    hi, mi, si = hours_minutes_seconds(len(photometry_data) / SRD)  # FIXME: unused
    print("\n")
    print("Theoretical length of measurement : " + str(datetime.timedelta(seconds=len(photometry_data)/SRD)) + " h:m:s")
    print("Real length of measurement : " + str(datetime.timedelta(seconds=photometry_data[:, 0][-1])) + " h:m:s")
    
    if photometry_data[:, 0][-1] - len(photometry_data)/SRD != 0:
        print("[WARNING] Shift in length of measurement : " + str(datetime.timedelta(seconds=(float(len(photometry_data) / SRD) - float(photometry_data[:, 0][-1])))) + " h:m:s")
    # print( (len(photometry_data)/SRD)/60, (len(photometry_data)/SRD)%60)

    # Cropping the data to the last second
    remaining = len(photometry_data) % SRD
    crop_length = int(len(photometry_data) - remaining)
    # print( (len(photometry_data[:crop_length])/SRD)/60, (len(photometry_data[:crop_length])/SRD)%60)
    
    transposed_photometry_data = np.transpose(photometry_data[:crop_length])  # Transposed data
    
    print("\n")
    print("Photometry recording length : " + str(datetime.timedelta(seconds=transposed_photometry_data.shape[1]/SRD)) + " h:m:s")
    
    raw_x = np.array(transposed_photometry_data[0])  # time data
    compressed_x = np.linspace(0, video_length, len(raw_x))  # compressed time data to fit the "real" time
    final_x = np.arange(start, end, 1/SRD)  # time to be extracted
    
    raw_isosbestic = np.array(transposed_photometry_data[1])  # isosbestic data
    final_raw_isosbestic = interpolated_and_extract(compressed_x, final_x, raw_isosbestic)  # data compressed in time
    
    raw_calcium = np.array(transposed_photometry_data[2])  # calcium data
    final_raw_calcium = interpolated_and_extract(compressed_x, final_x, raw_calcium)  # data compressed in time

    #######################################################################
                            #Baseline correction#
    #######################################################################
    print("\n")
    print("Computing moving average for Isosbestic signal !")

    # func_isosbestic = moving_average(final_raw_isosbestic, n=SRD*rollAvg); #moving average for isosbestic data
    func_isosbestic = centered_moving_average(final_raw_isosbestic, window=SRD * roll_avg)  # moving average for isosbestic data
    corrected_isosbestic = (final_raw_isosbestic - func_isosbestic) / func_isosbestic  # baseline correction for isosbestic
    
    print("\n")
    print("Computing moving average for Calcium signal !")
    # func_calcium = moving_average(final_raw_calcium, n=SRD*rollAvg)  # moving average for calcium data
    func_calcium = centered_moving_average(final_raw_calcium, window=SRD * roll_avg)  # moving average for calcium data
    corrected_calcium = (final_raw_calcium - func_calcium) / func_calcium  # baseline correction for calcium
        
    #######################################################################
                            #Standardization#
    #######################################################################
    print("\n")
    print("Starting standardization for Isosbestic signal !")
    # (x-moy)/var
    
    standardized_isosbestic = (corrected_isosbestic - np.median(corrected_isosbestic)) / np.std(corrected_isosbestic)  # standardization for isosbestic
    standardized_calcium = (corrected_calcium - np.median(corrected_calcium)) / np.std(corrected_calcium)  # standardization for calcium
        
    #######################################################################
                        #Inter-channel regression#
    #######################################################################
    print("\n")
    print("Starting interchannel regression !")
    
    x1 = np.array([0.0, 0.0])  # init value
    
    def func2(x, a, b):
        return a + b * x  # polynomial
    
    regr_fit = optimization.curve_fit(func2, standardized_isosbestic, standardized_calcium, x1)  # regression
    line = regr_fit[0][0] + regr_fit[0][1]*np.array([min(standardized_isosbestic), max(standardized_isosbestic)])  # regression line  # FIXME: unused
         
    #######################################################################
                        #Signal alignement#
    #######################################################################
    print("\n")
    print("Starting signal alignement !")
    
    fitted_isosbestic = regr_fit[0][0] + regr_fit[0][1]*standardized_isosbestic  # alignement of signals
        
    #######################################################################
                        #dF/F computation#
    #######################################################################
    print("\n")
    print("Computing dF/F(s) !")

    delta_f = standardized_calcium - fitted_isosbestic  # computing dF/F
    
    #######################################################################
                        #For display#
    #######################################################################
    source = [final_x, final_raw_isosbestic, final_raw_calcium, func_isosbestic, corrected_isosbestic,
              func_calcium, corrected_calcium, standardized_isosbestic, standardized_calcium,
              fitted_isosbestic, delta_f]
    
    if opt_plots:
        plot_intermediate_photometry_plots(source, **plot_args)
    
    return final_x, fitted_isosbestic, standardized_calcium, delta_f


def load_tracking_data(npy, fps, start, end):
    raw_cotton = np.load(npy)

    cotton = np.array([np.mean([float(i) for i in raw_cotton[int(n): int(n + fps)]])
                       for n in np.arange(0, len(raw_cotton), fps)])
    
    raw_cotton = raw_cotton[int(start*fps):int(end*fps)]
    cotton = cotton[start:end]  # Match the array length
    
    return raw_cotton, cotton


def detect_raw_peaks(data, PAT, PTPD):
    detected_peaks = []
    pos_detected_peaks = []
    
    # Raw peack detection
    print("\n")
    print("Running peak detection algorithm")
    
    for i, dp in enumerate(data):  # For every data point
        if i <= (len(data) - 1) - PTPD:  # If the point has enough points available in front of it
            
            if data[i]+PAT <= data[i+PTPD] or data[i]-PAT >= data[i+PTPD]:
                detected_peaks.append(True)
                pos_detected_peaks.append(i)
            else:
                detected_peaks.append(False)
        else:
            detected_peaks.append(False)
                
    print("Detected {0} peaks!".format(len(pos_detected_peaks)))
    return detected_peaks, pos_detected_peaks


def merge_close_peaks(peaks, pos_peaks, PMD):
    print("\n")
    print("Running peak merging algorithm")
    
    detected_peaks_merged = []
    detected_peaks_merged_positions = []
    
    for pos, peak in enumerate(peaks):
        d = pos
        if peak:
            ind = pos_peaks.index(pos)
            if ind < len(pos_peaks) - 1:
                if pos_peaks[ind + 1] - pos_peaks[ind] < PMD:
                    for i in np.arange(0, pos_peaks[ind + 1] - pos_peaks[ind]):
                        detected_peaks_merged.append(True)
                        detected_peaks_merged_positions.append(d)
                        d += 1
                else:
                    detected_peaks_merged.append(True)
                    detected_peaks_merged_positions.append(d)
                    d += 1
            else:
                detected_peaks_merged.append(True)
                detected_peaks_merged_positions.append(d)
                d += 1
        else:
            if len(detected_peaks_merged) <= pos:
                detected_peaks_merged.append(False)
                
    return detected_peaks_merged, detected_peaks_merged_positions


def detect_major_bouts(events, DBE, BL, GD):
    major_events_positions = []
    major_events_lengths = []
    seeds_positions = []
    
    new_bout_detected = False
    new_major_bout_detected = False
    
    pos_potential_event = 0
    
    distance_to_next_event = 0
    cumulative_events = 0
    
    for pos, event in enumerate(events):
        if event:
            cumulative_events += 1
            if not new_bout_detected:
                new_bout_detected = True
                pos_potential_event = pos
                seeds_positions.append(pos)
            elif new_bout_detected:
                distance_to_next_event = 0
                if cumulative_events >= BL:
                    if pos_potential_event - GD > 0 and pos_potential_event + GD < len(events):
                        if not new_major_bout_detected:
                            new_major_bout_detected = True
                            major_events_positions.append(pos_potential_event)
        else:
            if new_bout_detected:
                if distance_to_next_event < DBE:
                    distance_to_next_event += 1
                elif distance_to_next_event >= DBE:
                    if new_major_bout_detected:
                        major_events_lengths.append(cumulative_events)
                    
                    distance_to_next_event = 0
                    new_bout_detected = False
                    new_major_bout_detected = False
                    cumulative_events = 0
                    
    major_events = [True if n in major_events_positions else False for n, i in enumerate(events)]
    seeds_events = [True if n in seeds_positions else False for n, i in enumerate(events)]
    
    return major_events, major_events_positions, major_events_lengths, seeds_events, seeds_positions


def extract_calcium_data_when_behaving(peaks, calcium, GD, SRD, resample=1):
    data = []
    for p in peaks:
        segment = calcium[int((p*SRD)-(GD*SRD)): int((p*SRD)+((GD+1)*SRD))]
        data.append(segment)
    return data


def peri_event_plot(data, length, res_graph, res_heatmap, GD, GDf, GDb, SRD, save=False,
                    show_std=False, file_name_label="", cmap="viridis", lowpass=(2, 2), norm=False):
    res_graph_data = []
    res_heatmap_data = []
    res_graph_factor = SRD / res_graph
    res_heatmap_factor = SRD / res_heatmap
    
    for d in data:
        smoothed_graph = d
        # smoothed_graph = lowpass_filter(d, SRD, lowpass[0], lowpass[1]);
        
        resampled_graph = np.array([np.mean(smoothed_graph[int(i) : int(i)+int(res_graph_factor)])
                                    for i in np.arange(0, len(smoothed_graph), int(res_graph_factor))])
        resampled_heatmap = np.array([np.mean(smoothed_graph[int(i) : int(i)+int(res_heatmap_factor)])
                                      for i in np.arange(0, len(smoothed_graph), int(res_heatmap_factor))])
        res_graph_data.append(resampled_graph)
        res_heatmap_data.append(resampled_heatmap)
        
    res_graph_data = np.array(res_graph_data)
    res_heatmap_data = np.array(res_heatmap_data)

    mean_data_around_peaks = np.mean(res_graph_data, axis=0)
    std_data_around_peaks = np.std(res_graph_data, axis=0)
    
    fig = plt.figure(figsize=(20, 10))
    
    # First Plot
    ax0 = plt.subplot(2, 1, 1)
    
    if show_std:
        avg = ax0.plot(np.arange(0, len(mean_data_around_peaks)), mean_data_around_peaks, color="blue", alpha=0.5, lw=0.5)  # FIXME: unused
    
        ax0.plot(np.arange(0, len(mean_data_around_peaks)), mean_data_around_peaks + std_data_around_peaks, color="blue", alpha=0.2, lw=0.5)
        ax0.plot(np.arange(0, len(mean_data_around_peaks)), mean_data_around_peaks - std_data_around_peaks, color="blue", alpha=0.2, lw=0.5)
        
        ax0.fill_between(np.arange(0, len(mean_data_around_peaks)), mean_data_around_peaks,
                         mean_data_around_peaks + std_data_around_peaks,
                         color="blue", alpha=0.1)
        ax0.fill_between(np.arange(0, len(mean_data_around_peaks)), mean_data_around_peaks,
                         mean_data_around_peaks - std_data_around_peaks,
                         color="blue", alpha=0.1)
                         
        ax0.set_ylim(min(mean_data_around_peaks - std_data_around_peaks) + 0.1 * min(mean_data_around_peaks - std_data_around_peaks),
                     max(mean_data_around_peaks + std_data_around_peaks) + 0.1 * max(mean_data_around_peaks + std_data_around_peaks))
                     
        patch = patches.Rectangle((res_graph * GD, min(mean_data_around_peaks - std_data_around_peaks) - 0.1 * min(mean_data_around_peaks - std_data_around_peaks)),
                                  width=np.mean(length) * res_graph,
                                  height=(max(mean_data_around_peaks + std_data_around_peaks)-min(mean_data_around_peaks - std_data_around_peaks))*0.1,
                                  color='gray', lw=1, alpha=0.5)
        ax0.add_patch(patch)
    else:
        avg = ax0.plot(np.arange(0,len(mean_data_around_peaks)), mean_data_around_peaks, color="blue", alpha=0.5, lw=2)  # FIXME: unused
        
        ax0.set_ylim(min(mean_data_around_peaks)+0.1*min(mean_data_around_peaks),
                     max(mean_data_around_peaks)+0.1*max(mean_data_around_peaks))
                     
        patch = patches.Rectangle((res_graph * GD, min(mean_data_around_peaks) - 0.1 * min(mean_data_around_peaks)),
                                  width=np.mean(length) * res_graph,
                                  height=(max(mean_data_around_peaks)-min(mean_data_around_peaks)) * 0.1,
                                  color='gray', lw=1, alpha=0.5)
        ax0.add_patch(patch)
    
    xRange = np.arange(0,len(mean_data_around_peaks))
    ax0.plot((xRange[0], xRange[-1]), (0, 0), "--", color="blue", lw=1)

    ax0.set_xticks(np.arange((res_graph * GD) - (res_graph * GDb), (res_graph * GD) + (res_graph * GDf) + 5 * res_heatmap, 5 * res_heatmap))
    ax0.set_xticklabels(np.arange(-GDb, GDf+5, 5))
    ax0.set_xlim((res_graph * GD) - (res_graph * GDb), (res_graph * GD) + (res_graph * GDf))
    vline = ax0.axvline(x=res_graph * GD, color='red', linestyle='--', lw=1)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("dF/F")
    ax0.set_title("dF/F in function of time before & after nest-building initiation")
    
    ax0.legend(handles=[vline, patch], labels=["Begining of bout","Average behavioral bout length"], loc=2)
    
    # Second Plot
    ax1 = plt.subplot(2, 1, 2)

    if norm:
        ax1.imshow(res_heatmap_data, cmap=cmap, aspect="auto", norm=matplotlib.colors.LogNorm())
    else:
        ax1.imshow(res_heatmap_data, cmap=cmap, aspect="auto")

    ax1.set_ylim(-0.5, len(res_heatmap_data)-0.5)
    # ax1.set_yticks(np.arange(0, len(res_heatmap_data), 1));
        
    ax1.set_yticks([])
        
    n = 0
    for l in length:
        left_plot = (res_graph * GD) - (res_graph * GDb)
        ax1.text(left_plot-left_plot*0.01, n, l, ha="center", va="center")
        n += 1
    
    ax1.set_xticks(np.arange((res_graph * GD) - (res_graph * GDb), (res_graph * GD) + (res_graph * GDf) + 5 * res_heatmap, 5 * res_heatmap))
    ax1.set_xticklabels(np.arange(-GDb, GDf+5, 5))
    ax1.set_xlim((res_graph * GD) - (res_graph * GDb), (res_graph * GD) + (res_graph * GDf))
    ax1.axvline(x=res_heatmap * GD, color='red', linestyle='--', lw=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("dF/F for each individual event in function of time before & after nest-building initiation")
    # cbar = plt.colorbar(heatmap, orientation="vertical");
    
    plt.tight_layout()
    if save:
        plt.savefig("/home/thomas.topilko/Desktop/Photometry{0}.svg".format(file_name_label), dpi=200.)


def peak_plot(peaks, merged_peaks, major_peaks, seed_peaks, multi_plot=False, time_frame=[], save=False):
    fig = plt.figure(figsize=(20, 10))  # FIXME: unused
    fsl = 15.
    fst = 20.
    
    if time_frame:
        peaks = np.array(peaks)[(time_frame[0] < np.array(peaks)) & (np.array(peaks) < time_frame[1])]
        merged_peaks = np.array(merged_peaks)[(time_frame[0] < np.array(merged_peaks)) & (np.array(merged_peaks) < time_frame[1])]
        seed_peaks = np.array(seed_peaks)[(time_frame[0] < np.array(seed_peaks)) & (np.array(seed_peaks) < time_frame[1])]
        major_peaks = np.array(major_peaks)[(time_frame[0] < np.array(major_peaks)) & (np.array(major_peaks) < time_frame[1])]
    
    if not multi_plot:
        ax0 = plt.subplot(1, 1, 1)
        ax0.plot(peaks, color="blue")
        ax0.plot(merged_peaks, color="blue")
        ax0.plot(seed_peaks, color="orange")
        ax0.plot(major_peaks, color="red")
    else:
        ax0 = plt.subplot(4, 1, 1)
        ax0.eventplot(peaks, color="blue", lineoffsets=0, linelengths=1)
        ax0.set_ylim(-0.5, 0.5)
        ax0.set_yticks([])
        ax0.set_ylabel("Raw behavior bouts", rotation=0, va="center", ha="right", fontsize=fsl)
        
        ax0.set_title("Detection of major nesting events", fontsize=fst)
        ax0.tick_params(axis='both', which='major', labelsize=fsl)
        
        ax1 = plt.subplot(4, 1, 2, sharex=ax0, sharey=ax0)
        ax1.eventplot(merged_peaks, color="blue", lineoffsets=0, linelengths=1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        ax1.set_ylabel("Merged behavior bouts", rotation=0, va="center", ha="right", fontsize=fsl)
        ax1.tick_params(axis='both', which='major', labelsize=fsl)
        
        ax2 = plt.subplot(4, 1, 3, sharex=ax0, sharey=ax0)
        ax2.eventplot(seed_peaks, color="blue", lineoffsets=0, linelengths=1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.set_ylabel("Start of each bout", rotation=0, va="center", ha="right", fontsize=fsl)
        ax2.tick_params(axis='both', which='major', labelsize=fsl)
        
        ax3 = plt.subplot(4, 1, 4, sharex=ax0, sharey=ax0)
        ax3.eventplot(major_peaks, color="blue", lineoffsets=0, linelengths=1)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_yticks([])
        ax3.set_ylabel("Major bouts", rotation=0, va="center", ha="right", fontsize=fsl)
        ax3.tick_params(axis='both', which='major', labelsize=fsl)
        
    if save:
        plt.savefig("/home/thomas.topilko/Desktop/PeakDetection.svg", dpi=200.)


def save_peri_event_data(data, length, pos, sink_dir, mouse):
    sink = np.array([data, length, pos])
    
    np.save(os.path.join(sink_dir, "peData_{0}.npy".format(mouse)), sink)


def reorder_by_bout_size(data, length, pos):
    order = np.argsort(length)
    
    sink_length = np.array(length)[order]
    sink_data = np.array(data)[order]
    sink_pos = np.array(pos)[order]
    
    return sink_data, sink_length, sink_pos


def filter_short_bouts(data, length, pos, thresh):
    sink_data = []
    sink_length = []
    sink_pos = []
    for d, l, p in zip(data, length, pos):
        if l >= thresh:
            sink_data.append(d)
            sink_length.append(l)
            sink_pos.append(p)
            
    return sink_data, sink_length, sink_pos


def extract_manual_behavior(raw, file):
    peaks = np.full_like(raw, False)
    f = pd.read_excel(file, header=None)
    for s, e in zip(f.iloc[0][1:], f.iloc[1][1:]):
        for i in np.arange(s, e, 1):
            peaks[i] = True
    return peaks


def extract_manual_bouts(manual_file_path, end_vid, beh):
    peaks = []
    pos_peaks = []
    
    f = pd.read_excel(manual_file_path, header=None)
    
    start_pos = np.where(f[0] == "tStart{0}".format(beh))[0]
    end_pos = np.where(f[0] == "tEnd{0}".format(beh))[0]
    
    print(f.iloc[start_pos[0]][1:][f.iloc[start_pos[0]][1:] > 0])
    print(f.iloc[end_pos[0]][1:][f.iloc[end_pos[0]][1:] > 0])
    
    for s, e in zip(f.iloc[start_pos[0]][1:][f.iloc[start_pos[0]][1:] > 0], f.iloc[end_pos[0]][1:][f.iloc[end_pos[0]][1:] > 0]):
        for i in np.arange(len(peaks), s, 1):
            peaks.append(False)
        
        for i in np.arange(s, e, 1):
            peaks.append(True)
            pos_peaks.append(i)
            
    for i in np.arange(e, end_vid, 1):
        peaks.append(False)
    
    print(hours_minutes_seconds(len(peaks)))
    
    return peaks, pos_peaks
