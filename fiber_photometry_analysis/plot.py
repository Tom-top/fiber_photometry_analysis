#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:19:37 2020

@author: thomas.topilko
"""

import os

import numpy as np

import matplotlib.colors as col
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.widgets import MultiCursor

plt.style.use("default")

import scipy.stats as stats

import seaborn as sns
from sklearn.metrics import auc

from fiber_photometry_analysis import utilities as utils


def add_line_at_zero(ax, x, line_width, color='black'):
    """
    Creates a horizontal dashed line at y = 0 to signal the baseline
    """
    ax.plot((0, x[-1]), (0, 0), "--", color=color, lw=line_width)


def plot_bouts(ax, colors, length_bouts, position_bouts, y_min, y_max):
    handles = []
    for n, P, L in enumerate(zip(position_bouts, length_bouts)):
        patch = patches.Rectangle((0, 0), 1, 1, alpha=0.3, color=colors[n], lw=0, edgecolor=None)
        handles.append(patch)
        for p, l in zip(P, L):
            patch = patches.Rectangle((p[0], y_min + y_min * 0.1), l, (y_max + y_max * 0.1) - (y_min + y_min * 0.1),
                                      alpha=0.3, color=colors[n], lw=0, edgecolor=None)
            ax.add_patch(patch)
    return handles


def check_delta_f_with_behavior(position_bouts, length_bouts, colors=("blue"), fig_name="dF_&_behavioral_overlay", **params):
    """Function that plots the pre-processed photometry data and overlays the
    behavior data.
    
    Args :  position_bouts (arr) = list of start and end of each behavioral bout
            length_bouts (list) = list of the length of each behavioral bout
            color (list) = color(s) of the behavioral overlay(s)
            kwargs (dict) = dictionary with additional parameters
    """

    data = params["photometry_data"]["dFF"]["dFF"]
    x = params["photometry_data"]["dFF"]["x"]
    line_width = params['lw']
    font_size = params['fsl']
    standardize = params["photometry_pp"]["standardize"]
    multiplication_factor = 1 if standardize else 100

    labels = [params["behavior_to_segment"]]
    if len(position_bouts) != 1:
        labels.append("Excluded")

    plt.figure(figsize=(10, 3), dpi=200.)
    ax = plt.subplot(111)

    ax.plot(x, data, color="green", lw=line_width)
    add_line_at_zero(ax, (params["video_end"] - params["photometry_data"]["time_lost"],),
                     line_width, 'blue')  # TODO: check if could use X

    x_max = x[-1]
    xticks, xticklabels, unit = utils.generate_xticks_and_labels(x_max)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=font_size)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Time ({0})".format(unit), fontsize=font_size)

    y_min, y_max, round_factor = utils.generate_yticks(data, 0.1)
    int_ticks = np.arange(y_min, y_max + round_factor, round_factor)
    ax.set_yticks(int_ticks)
    int_ticks *= multiplication_factor
    ticks = ["{:.0f}".format(i) for i in int_ticks]
    ax.set_yticklabels(ticks, fontsize=font_size)
    ax.set_ylim(y_min, y_max)

    plot_handles = plot_bouts(ax, colors, length_bouts, position_bouts, y_min, y_max)

    ax.legend(handles=plot_handles, labels=labels, loc=2, fontsize=font_size)
    plt.tight_layout()
    if params["save"]:
        plt.savefig(os.path.join(params["save_dir"], "{}.{}".format(fig_name, params["extension"])), dpi=200.)


def peri_event_plot(data_around_major_bouts, length_major_bouts, cmap="inferno", **kwargs):
    """Function that plots the peri-event photometry data in two different formats.
    First a line plot showing the average signal before and after initiation of the behavior,
    Second a heatmap showing each individual signal traces before and after initiation of the behavior.
    
    Args :  data_around_major_bouts (arr) = list of the pre-processed photometry data
            before and after initiation of the behavior
            length_major_bouts (list) = list of the length of each behavioral bout
            cmap (str) = colormap used for the heatmap
            kwargs (dict) = dictionary with additional parameters
    """
    
    mean_data_around_bouts = np.mean(data_around_major_bouts, axis=0)  # Computes the mean for the peri-event photometry data
    std_data_around_bouts = np.std(data_around_major_bouts, axis=0)  # Computes the standard deviation for the peri-event photometry data

    fig = plt.figure(figsize=(5, 5), dpi=200.)  # Creates a figure
    
    # =============================================================================
    #     First Plot
    # =============================================================================
    
    ax0 = plt.subplot(2, 1, 1)  # Creates a subplot for the first figure
    
    x_range = np.linspace(0, len(mean_data_around_bouts), len(mean_data_around_bouts))
    ax0.plot(x_range, mean_data_around_bouts, color="green", alpha=1., lw=kwargs["lw"]+1, zorder=1)  # Plots the average signal
    
    if kwargs["peri_event"]["style"] == "individual":  # If individual traces are to be displayed in the line plot
        for l in data_around_major_bouts:
            if kwargs["peri_event"]["individual_color"]:
                ax0.plot(l, alpha=0.5, lw=kwargs["lw"], zorder=0)  # Plots individual traces
            else:
                ax0.plot(l, color="gray", alpha=0.5, lw=kwargs["lw"], zorder=0)  # Plots individual traces
            
        ax0.plot(x_range, mean_data_around_bouts + std_data_around_bouts,
                 color="green", alpha=0.5, lw=kwargs["lw"], zorder=1)
        ax0.plot(x_range, mean_data_around_bouts - std_data_around_bouts,
                 color="green", alpha=0.5, lw=kwargs["lw"], zorder=1)
        
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts + std_data_around_bouts),
                         color="green", alpha=0.1)
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts - std_data_around_bouts),
                         color="green", alpha=0.1)
        
        y_min, y_max, round_factor = utils.generate_yticks(data_around_major_bouts.flatten(), 0.2)
    elif kwargs["peri_event"]["style"] == "average":
        ax0.plot(x_range, mean_data_around_bouts + std_data_around_bouts,
                 color="green", alpha=0.3, lw=kwargs["lw"])
        ax0.plot(x_range, mean_data_around_bouts - std_data_around_bouts,
                 color="green", alpha=0.3, lw=kwargs["lw"])
        
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts + std_data_around_bouts),
                         color="green", alpha=0.1)
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts - std_data_around_bouts),
                         color="green", alpha=0.1)
        y_min, y_max, round_factor = utils.generate_yticks(np.concatenate((mean_data_around_bouts + std_data_around_bouts, mean_data_around_bouts - std_data_around_bouts)), 0.2)
    else:
        raise NotImplementedError('Unrecognised style keyword "{}"'.format(kwargs["peri_event"]["style"]))
    y_range = y_max - y_min
    ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
    ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
    ax0.set_ylim(y_min, y_max)
    
    # Creates a gray square on the line plot that represents the average length of a behavioral bout
    patch = patches.Rectangle(((kwargs["peri_event"]["graph_distance_pre"]+0.5)*kwargs["recording_sampling_rate"],
                               - y_range*0.1),
                              width=np.mean(length_major_bouts)*kwargs["recording_sampling_rate"],
                              height=y_range*0.1,
                              color="gray",
                              lw=0,
                              alpha=0.7)
    
    ax0.add_patch(patch)  # Adds the patch to the plot
    ax0.plot((0, len(mean_data_around_bouts)), (0, 0), "--", color="black", lw=kwargs["lw"])  # Creates a horizontal dashed line at y = 0 to signal the baseline
    vline = ax0.axvline(x=(kwargs["peri_event"]["graph_distance_pre"]+0.5) * kwargs["recording_sampling_rate"],
                        color='red', linestyle='--', lw=kwargs["lw"])  # Creates a vertical dashed line at x = 0 to signal the begining of the behavioral bout
    
    ax0.set_xticks(np.linspace(0, ((kwargs["peri_event"]["graph_distance_pre"] + kwargs["peri_event"]["graph_distance_post"]+1) * kwargs["recording_sampling_rate"]), 5)) #Generates ticks for the x axis
    ax0.set_xticklabels(np.linspace(-kwargs["peri_event"]["graph_distance_pre"], kwargs["peri_event"]["graph_distance_post"], 5), fontsize=kwargs["fsl"])  # Generates labels for the x axis
    ax0.set_xlim(0, len(mean_data_around_bouts))  # Sets the limits for the x axis
#    ax0.set_xlabel("Time (s)", fontsize=kwargs["fsl"]) #Displays the label for the x axis

    ax0.legend(handles=[vline, patch], labels=["Begining of bout", "Average behavioral bout length"],
               loc=2, fontsize=kwargs["fsl"])  # Displays the legend of the first figure
    
    # =============================================================================
    #     Second Plot
    # =============================================================================
    
    ax1 = plt.subplot(2, 1, 2)  # Creates a subplot for the second figure
    
    if not kwargs["peri_event"]["normalize_heatmap"]:
        heatmap = ax1.imshow(data_around_major_bouts, cmap=cmap, aspect="auto",
                             interpolation="none")
    else:
        heatmap = ax1.imshow(data_around_major_bouts, cmap=cmap, aspect="auto",
                             norm=col.LogNorm(), interpolation="none")  # norm=matplotlib.colors.LogNorm()
        
    ax1.axvline(x=(kwargs["peri_event"]["graph_distance_pre"]+0.5)*kwargs["recording_sampling_rate"],
                color='red', linestyle='--', lw=kwargs["lw"])
        
    ax1.set_xticks(np.linspace(0, ((kwargs["peri_event"]["graph_distance_pre"] + kwargs["peri_event"]["graph_distance_post"]+1) * kwargs["recording_sampling_rate"]), 5)) #Generates ticks for the x axis
    ax1.set_xticklabels(np.linspace(-kwargs["peri_event"]["graph_distance_pre"], kwargs["peri_event"]["graph_distance_post"], 5), fontsize=kwargs["fsl"]) #Generates labels for the x axis
    ax1.set_xlim(0, len(mean_data_around_bouts))  # Sets the limits for the x axis
    ax1.set_xlabel("Time (s)", fontsize=kwargs["fsl"])  # Displays the label for the x axis
    
    ax1.set_yticks([])
    ax1.set_ylim(-0.5, len(data_around_major_bouts) - 0.5)
    ax1.set_ylabel("Duration of individual bouts (s)", fontsize=kwargs["fsl"], labelpad=20)  # Displays the label for the x axis

    for n, l in enumerate(length_major_bouts):
        ax1.text(-len(mean_data_around_bouts) * 0.01, n, l, ha="right", va="center", fontsize=kwargs["fsl"])
    
    plt.gca().invert_yaxis()
    cax = fig.add_axes([1, 0.1, 0.02, 0.35])
    cb = plt.colorbar(heatmap, ax=ax1, cax=cax)
    cb.ax.tick_params(labelsize=kwargs["fsl"])
    cb.ax.get_yaxis().labelpad = 8
    
    if kwargs["photometry_pp"]["standardize"]:
        ax0.set_ylabel(r"$\Delta$F/F (Z-Score)", fontsize=kwargs["fsl"]) #Displays the label for the y axis
        ax0.set_title(r"$\Delta$F/F (Z-Score) in function of time before & after {0} initiation".format(kwargs["behavior_to_segment"]), fontsize=kwargs["fst"]) #Displays the titel for the first figure
        ax1.set_title(r"Individual $\Delta$F/F (Z-Score) in function of time before & after {0} initiation".format(kwargs["behavior_to_segment"]), fontsize=kwargs["fst"])
        cb.ax.set_ylabel(r"$\Delta$F/F (Z-Score)", rotation=270, fontsize=kwargs["fsl"])
    else:
        ax0.set_ylabel(r"$\Delta$F/F (%)", fontsize=kwargs["fsl"]) #Displays the label for the y axis
        ax0.set_title(r"$\Delta$F/F (%) in function of time before & after {0} initiation".format(kwargs["behavior_to_segment"]), fontsize=kwargs["fst"]) #Displays the titel for the first figure
        ax1.set_title(r"Individual $\Delta$F/F (%) in function of time before & after {0} initiation".format(kwargs["behavior_to_segment"]), fontsize=kwargs["fst"])
        cb.ax.set_ylabel(r"$\Delta$F/F (%)", rotation=270, fontsize=kwargs["fsl"])
        
    plt.tight_layout()
    
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "Peri_Event_Plot.{0}".format(kwargs["extension"])), dpi=200., bbox_inches='tight')


def peri_event_bar_plot(data_around_major_bouts, **kwargs):
    """Function that compares the area under the curve (AUC) before and after the intitation of the behavior.
    The results are summarized in a bar plot showing the AUC before and after initiation of the behavior.
    
    Args :  data_around_major_bouts (arr) = list of the pre-processed photometry data
            kwargs (dict) = dictionary with additional parameters
    """
    
    time_before = np.linspace(0, kwargs["peri_event"]["graph_auc_pre"],
                              kwargs["peri_event"]["graph_auc_pre"] * kwargs["recording_sampling_rate"])
    data_before = data_around_major_bouts[:, 0: kwargs["peri_event"]["graph_auc_pre"] * kwargs["recording_sampling_rate"]]
    
    time_after = np.linspace(0, kwargs["peri_event"]["graph_auc_post"],
                             kwargs["peri_event"]["graph_auc_post"] * kwargs["recording_sampling_rate"])
    data_after = data_around_major_bouts[:, (kwargs["peri_event"]["graph_auc_pre"]+1) * kwargs["recording_sampling_rate"]: (kwargs["peri_event"]["graph_auc_pre"]+1+kwargs["peri_event"]["graph_auc_post"])*kwargs["recording_sampling_rate"]]
    
    all_auc1 = [auc(time_before, i) for i in data_before]
    auc1_mean = auc(time_before, np.mean(data_before, axis=0))
    auc1_std = auc(time_before, np.std(data_before, axis=0))
    
    all_auc2 = [auc(time_after, i) for i in data_after]
    auc2_mean = auc(time_after, np.mean(data_after, axis=0))
    auc2_std = auc(time_after, np.std(data_after, axis=0))

    auc_mean = [auc1_mean, auc2_mean]
    auc_std = [auc1_std, auc2_std]

    stat, pval = stats.ttest_ind(all_auc1, all_auc2, equal_var=False)
    
    fig = plt.figure(figsize=(5, 5), dpi=200.)
    ax0 = plt.subplot(1, 1, 1)
    palette = ("#81dafc", "#ff6961"),
    plot_params = {
        'edgecolor': "black",
        'linewidth': 1.,
        'alpha': 0.8
    }
    
    ax0.bar(np.arange(len(auc_mean)),
            auc_mean,
            yerr=auc_std,
            error_kw={"elinewidth": 1,
                      "solid_capstyle": "projecting",
                      "capsize": 5,
                      "capthick": 1
                      },
            color=palette,
            **plot_params
            # zorder=1
            )
    
    sns.swarmplot(np.concatenate([np.full_like(all_auc1, 0), np.full_like(all_auc2, 1)]),
                  np.concatenate([all_auc1, all_auc2]),
                  ax=ax0,
                  s=10,
                  palette=palette,
                  **plot_params
                  # zorder = 0,
                  )
    
    y_max = max([max(all_auc1), max(all_auc2)]) + max([max(all_auc1), max(all_auc2)])*0.1
    y_min = min([min(all_auc1), min(all_auc2)]) + min([min(all_auc1), min(all_auc2)])*0.1
    ax0.plot((0, 0), (y_max+y_max*0.05, y_max+y_max*0.1), lw=1, color="black")
    ax0.plot((1, 1), (y_max+y_max*0.05, y_max+y_max*0.1), lw=1, color="black")
    ax0.plot((0, 1), (y_max+y_max*0.1, y_max+y_max*0.1), lw=1, color="black")
    
    utils.print_in_color("The comparison of AUC for before and after the behavioral initiation is : {0}"
                         .format(pval), "GREEN")

    if 0.01 < pval <= 0.05:
        stars = '*'
    elif 0.001 < pval <= 0.01:
        stars = '**'
    elif 0.0001 < pval <= 0.001:
        stars = '***'
    elif pval <= 0.0001:
        stars = '****'
    else:  # if pval > 0.05
        stars = 'n.s'
    ax0.text(0.5, y_max+y_max*0.1, stars)

    int_ticks = np.arange(round(y_min), round(y_max + y_max * 0.3) + 2, 1)
    ax0.set_yticks(int_ticks)
    ax0.set_yticklabels(int_ticks, fontsize=kwargs["fsl"])
    ax0.plot((-0.5, 1.5), (0, 0), lw=1, color="black")
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Before initiating {0}".format(kwargs["behavior_to_segment"]),
                         "After initiating {0}".format(kwargs["behavior_to_segment"])],
                        fontsize=kwargs["fsl"])
    
    ax0.set_ylim(y_min+y_min*0.5, y_max+y_max*0.5)
    ax0.set_ylabel("Area under the curve (AUC)", fontsize=kwargs["fsl"])
    ax0.set_title("Before vs After initiation of {0} Response Changes".format(kwargs["behavior_to_segment"]),
                  fontsize=kwargs["fst"])
    
    fig.tight_layout()
    
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "AUC.{0}".format(kwargs["extension"])),
                    dpi=200., bbox_inches='tight')


def plot_data_pair(calcium_data, isosbestic_data, title, kwargs, x, add_zero_line=False, units='', to_kilo=False):
    multiplication_factor = 1000 if to_kilo else 1

    fig = plt.figure(figsize=(10, 5), dpi=200.)

    ax0 = plt.subplot(211)
    sub_plot(data=isosbestic_data, x=x, ax=ax0, laser='purple_laser', label='isosbestic', params=kwargs,
             add_zero_line=add_zero_line, units=units, multiplication_factor=multiplication_factor)
    ax0.set_title("{} Isosbestic and Calcium signals".format(title.title()), fontsize=kwargs["fst"])

    ax1 = plt.subplot(212, sharex=ax0)
    sub_plot(data=calcium_data, x=x, ax=ax1, laser='blue_laser', label='calcium', params=kwargs,
             add_zero_line=add_zero_line, units=units, multiplication_factor=multiplication_factor)
    ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

    plt.tight_layout()

    if kwargs["photometry_pp"]["multicursor"]:
        multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])  # FIXME: unused
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"],
                                 "{}_Signals.{}".format(title.title(), kwargs["extension"])),
                    dpi=200.)


def sub_plot(data, x, ax, laser, label, params, add_zero_line, units, multiplication_factor, baseline=None):
    line_width = params['lw']
    font_size = params['fsl']

    plot_handle, = ax.plot(x, data, alpha=0.8, c=params["photometry_pp"][laser],
                           lw=line_width)
    plot_handles = [plot_handle]
    labels = [label]
    if baseline is not None:
        plot_handle_2, = ax.plot(x, baseline, alpha=0.8, c="orange", lw=2)
        plot_handles.append(plot_handle_2)
        labels.append('baseline')
    if add_zero_line:
        add_line_at_zero(ax, x, line_width)

    x_max = x[-1]
    xticks, xticklabels, unit = utils.generate_xticks_and_labels(x_max)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=font_size)
    ax.set_xlim(0, x_max)

    y_min, y_max, round_factor = utils.generate_yticks(data, 0.1)
    int_ticks = np.arange(y_min, y_max + round_factor, round_factor)
    ax.set_yticks(int_ticks)
    int_ticks *= multiplication_factor
    ticks = ["{:.0f}".format(i) for i in int_ticks]
    ax.set_yticklabels(ticks, fontsize=font_size)
    ax.set_ylim(y_min, y_max)
    if units:
        ax.set_ylabel(units, fontsize=font_size)
    ax.legend(handles=plot_handles, labels=labels, loc=2, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)


def plot_cropped_data(calcium, calcium_fc, isosbestic, isosbestic_fc, kwargs, x):
    fig = plt.figure(figsize=(10, 5), dpi=200.)

    ax0 = plt.subplot(211)
    sub_plot(data=isosbestic, x=x, ax=ax0, laser='purple_laser', label='isosbestic', params=kwargs, add_zero_line=False,
             units='mV', multiplication_factor=1000, baseline=isosbestic_fc)
    ax0.set_title("Smoothed Isosbestic and Calcium signals with respective baselines", fontsize=kwargs["fst"])

    ax1 = plt.subplot(212, sharex=ax0)
    sub_plot(data=calcium, x=x, ax=ax1, laser='blue_laser', label='calcium', params=kwargs, add_zero_line=False,
             units='mV', multiplication_factor=1000, baseline=calcium_fc)
    ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

    plt.tight_layout()

    if kwargs["photometry_pp"]["multicursor"]:
        multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])  # FIXME: unused
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"],
                                 "Baseline_Determination.{0}".format(kwargs["extension"])),
                    dpi=200.)


def plot_ca_iso_regression(calcium, isosbestic, isosbestic_fitted, kwargs):
    plt.figure(figsize=(5, 5), dpi=200.)
    ax = plt.subplot(111)
    ax.scatter(isosbestic, calcium, color="blue", s=0.5, alpha=0.05)
    ax.plot(isosbestic, isosbestic_fitted, 'r-', linewidth=1)
    ax.set_xlabel("Isosbestic", fontsize=kwargs["fsl"])
    ax.set_ylabel("Calcium", fontsize=kwargs["fsl"])
    ax.set_title("Inter-channel regression of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
    ax.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])

    plt.tight_layout()

    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"],
                                 "Interchannel_Regression.{0}".format(kwargs["extension"])),
                    dpi=200.)
