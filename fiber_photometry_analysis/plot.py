#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:19:37 2020

@author: thomas.topilko
"""

import os

import numpy as np

import matplotlib.colors as col
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.widgets import MultiCursor

plt.style.use("default")

import scipy.stats as stats

import seaborn as sns
from sklearn.metrics import auc

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import behavior_preprocessing as behav_preproc


def check_delta_f_with_behavior(bool_map, zorder=[0,1],  name="dF_&_behavioral_overlay", extra_legend_label="", **kwargs):
    """Function that plots the pre-processed photometry data and overlays the
    behavior data.
    
    Args :  position_bouts (arr) = list of start and end of each behavioral bout
            length_bouts (list) = list of the length of each behavioral bout
            color (list) = color(s) of the behavioral overlay(s)
            kwargs (dict) = dictionary with additional parameters
    """
    
    plt.figure(figsize=(10, 3), dpi=200.)
    ax0 = plt.subplot(111)

    x_delta_f = kwargs["photometry_data"]["dFF"]["x"]
    y_delta_f = kwargs["photometry_data"]["dFF"]["dFF"]

    xticks, xticklabels, unit = utils.generate_xticks_and_labels(x_delta_f.iloc[-1])
    y_min, y_max, round_factor = utils.generate_yticks(y_delta_f, 0.1)
    colors = ["red", "orange"]

    event_plot_args = {"linelengths": y_max - y_min,
                       "lineoffsets": (y_max + y_min) / 2,
                       "alpha": 1,
                       }
    handles = [None, None]

    if isinstance(bool_map, list) and len(bool_map) == 2:
        labels = [kwargs["behavior_to_segment"], "{}".format(extra_legend_label)]
        raster_data_raw = behav_preproc.get_position_from_bool_map(bool_map[0], kwargs["resolution_data"])
        raster_data_processed = behav_preproc.get_position_from_bool_map(bool_map[1], kwargs["resolution_data"])

        b1, = ax0.eventplot(raster_data_processed,
                            color=colors[zorder[0]],
                            zorder=zorder[0],
                            **event_plot_args)
        handles[zorder[1]] = b1

    elif isinstance(bool_map, np.ndarray):
        labels = [kwargs["behavior_to_segment"]]
        raster_data_raw = behav_preproc.get_position_from_bool_map(bool_map, kwargs["resolution_data"])

    b2, = ax0.eventplot(raster_data_raw,
                        color=colors[zorder[1]],
                        zorder=zorder[1],
                        **event_plot_args)
    handles[zorder[0]] = b2

    ax0.plot(x_delta_f,
             y_delta_f,
             zorder=2,
             color="green",
             lw=kwargs["lw"],
             )

    ax0.plot([0, x_delta_f.iloc[-1]],
             [0]*2,
             "--",
             color="blue",
             lw=kwargs["lw"],
             )

    ax0.set_xlim(0, x_delta_f.iloc[-1])
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels, fontsize=kwargs["fsl"])
    ax0.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

    ax0.set_ylim(y_min, y_max)
    ax0.set_yticks(np.arange(y_min, y_max + round_factor, round_factor))
    ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
        
    ax0.legend(handles=handles, labels=labels, loc=2, fontsize=kwargs["fsl"])
    plt.tight_layout()
    
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "{0}.{1}".format(name, kwargs["extension"])), dpi=200.)


def peri_event_plot(data_around_major_bouts, length_major_bouts, interpolated_sampling_rate,
                    style="individual", individual_colors=False, cmap="inferno", **kwargs):
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
    
    if style == "individual":  # If individual traces are to be displayed in the line plot
        for l in data_around_major_bouts:
            if individual_colors:
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
        y_range = y_max - y_min
        ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
    elif style == "average":
        ax0.plot(x_range, mean_data_around_bouts + std_data_around_bouts,
                 color="green", alpha=0.3, lw=kwargs["lw"])
        ax0.plot(x_range, mean_data_around_bouts - std_data_around_bouts,
                 color="green", alpha=0.3, lw=kwargs["lw"])
        
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts + std_data_around_bouts),
                         color="green", alpha=0.1)
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts - std_data_around_bouts),
                         color="green", alpha=0.1)
        
        y_min, y_max, round_factor = utils.generate_yticks(np.concatenate((mean_data_around_bouts + std_data_around_bouts, mean_data_around_bouts - std_data_around_bouts)), 0.2)
        y_range = y_max - y_min
        ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
        ax0.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=kwargs["fsl"])
        ax0.set_ylim(y_min, y_max)
    
    # Creates a gray square on the line plot that represents the average length of a behavioral bout
    patch = patches.Rectangle(((kwargs["peri_event"]["graph_distance_pre"]+0.5)*interpolated_sampling_rate, - y_range*0.1),
                              width=np.mean(length_major_bouts)*interpolated_sampling_rate,
                              height=y_range*0.1,
                              color="gray",
                              lw=0,
                              alpha=0.7)
    
    ax0.add_patch(patch)  # Adds the patch to the plot
    ax0.plot((0, len(mean_data_around_bouts)), (0, 0), "--", color="black", lw=kwargs["lw"])  # Creates a horizontal dashed line at y = 0 to signal the baseline
    vline = ax0.axvline(x=(kwargs["peri_event"]["graph_distance_pre"]+0.5) * interpolated_sampling_rate,
                        color='red', linestyle='--', lw=kwargs["lw"])  # Creates a vertical dashed line at x = 0 to signal the begining of the behavioral bout

    print(np.linspace(0, ((kwargs["peri_event"]["graph_distance_pre"] + kwargs["peri_event"]["graph_distance_post"]) * interpolated_sampling_rate), 5))
    ax0.set_xticks(np.linspace(0, ((kwargs["peri_event"]["graph_distance_pre"] + kwargs["peri_event"]["graph_distance_post"] + 1) * interpolated_sampling_rate), 5)) #Generates ticks for the x axis
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
        
    ax1.axvline(x=(kwargs["peri_event"]["graph_distance_pre"]+0.5)*interpolated_sampling_rate,
                color='red', linestyle='--', lw=kwargs["lw"])
        
    ax1.set_xticks(np.linspace(0, ((kwargs["peri_event"]["graph_distance_pre"] + kwargs["peri_event"]["graph_distance_post"]+1) * interpolated_sampling_rate), 5)) #Generates ticks for the x axis
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


def peri_event_bar_plot(data_around_major_bouts, interpolated_sampling_rate, duration_pre=2, duration_post=2, **kwargs):
    """Function that compares the area under the curve (AUC) before and after the intitation of the behavior.
    The results are summarized in a bar plot showing the AUC before and after initiation of the behavior.
    
    Args :  data_around_major_bouts (arr) = list of the pre-processed photometry data
            kwargs (dict) = dictionary with additional parameters
    """

    center = int((kwargs["peri_event"]["graph_distance_pre"] + kwargs["peri_event"]["graph_distance_post"] + 1)/2)
    time_before = np.linspace(0, duration_pre, duration_pre * interpolated_sampling_rate)
    data_before = data_around_major_bouts[:, (center - duration_pre) * interpolated_sampling_rate :
                                             center * interpolated_sampling_rate]
    
    time_after = np.linspace(0, duration_post, duration_post * interpolated_sampling_rate)
    data_after = data_around_major_bouts[:, center * interpolated_sampling_rate :
                                            (center + duration_post) * interpolated_sampling_rate]
    
    all_AUC1 = [auc(time_before, i) for i in data_before]
    AUC1_mean = auc(time_before, np.mean(data_before, axis=0))
    AUC1_std = auc(time_before, np.std(data_before, axis=0))
    
    all_AUC2 = [auc(time_after, i) for i in data_after]
    AUC2_mean = auc(time_after, np.mean(data_after, axis=0))
    AUC2_std = auc(time_after, np.std(data_after, axis=0))

    AUC_mean = [AUC1_mean, AUC2_mean]
    AUC_std = [AUC1_std, AUC2_std]

    stat, pval = stats.ttest_ind(all_AUC1, all_AUC2, equal_var=False)
    
    fig = plt.figure(figsize=(5, 5), dpi=200.)
    ax0 = plt.subplot(1, 1, 1)
    
    plot = ax0.bar(np.arange(len(AUC_mean)),
                   AUC_mean,
                   yerr=AUC_std,
                   error_kw={"elinewidth": 1,
                             "solid_capstyle": "projecting",
                             "capsize": 5,
                             "capthick": 1,
                             },
                   color=("#81dafc", "#ff6961"),
                   edgecolor="black",
                   linewidth=1.,
                   alpha=0.8,
                   # zorder=1,
                   )
    
    scat = sns.swarmplot(np.concatenate([np.full_like(all_AUC1, 0), np.full_like(all_AUC2, 1)]),
                         np.concatenate([all_AUC1, all_AUC2]),
                         ax=ax0,
                         s=10,
                         palette=("#81dafc", "#ff6961"),
                         edgecolor="black",
                         linewidth=1.,
                         alpha=0.8,
                         # zorder = 0,
                         )
    
    y_max = max([max(all_AUC1), max(all_AUC2)]) + max([max(all_AUC1), max(all_AUC2)])*0.1
    y_min = min([min(all_AUC1), min(all_AUC2)]) + min([min(all_AUC1), min(all_AUC2)])*0.1
    ax0.plot((0, 0), (y_max+y_max*0.05, y_max+y_max*0.1), lw=1, color="black")
    ax0.plot((1, 1), (y_max+y_max*0.05, y_max+y_max*0.1), lw=1, color="black")
    ax0.plot((0, 1), (y_max+y_max*0.1, y_max+y_max*0.1), lw=1, color="black")
    
    utils.print_in_color("The comparison of AUC for before and after the behavioral initiation is : {0}"
                         .format(pval), "GREEN")
    
    if pval > 0.05 and pval >= 0.01:
        ax0.text(0.5, y_max+y_max*0.1, "n.s")
    elif 0.05 >= pval > 0.01:
        ax0.text(0.5, y_max+y_max*0.1, "*")
    elif 0.01 >= pval > 0.001:
        ax0.text(0.5, y_max+y_max*0.1, "**")
    elif 0.001 >= pval > 0.0001:
        ax0.text(0.5, y_max+y_max*0.1, "***")
    elif pval <= 0.0001:
        ax0.text(0.5, y_max+y_max*0.1, "****")
    
    ax0.set_yticks(np.arange(round(y_min), round(y_max+y_max*0.3)+2, 1))
    ax0.set_yticklabels(np.arange(round(y_min), round(y_max+y_max*0.3)+2, 1),  fontsize=kwargs["fsl"])
    ax0.plot((-0.5, 1.5), (0, 0), lw=1, color="black")
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Before initiating {0}".format(kwargs["behavior_to_segment"]),"After initiating {0}"
                        .format(kwargs["behavior_to_segment"])], fontsize=kwargs["fsl"])
    
    ax0.set_ylim(y_min+y_min*0.5, y_max+y_max*0.5)
    ax0.set_ylabel("Area under the curve (AUC)", fontsize=kwargs["fsl"])
    ax0.set_title("Before vs After initiation of {0} Response Changes".format(kwargs["behavior_to_segment"]), fontsize=kwargs["fst"])
    
    fig.tight_layout()
    
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "AUC.{0}".format(kwargs["extension"])),
                    dpi=200., bbox_inches='tight')


def add_line_at_zero(ax, x, line_width):
    """
    Creates a horizontal dashed line at y = 0 to signal the baseline
    """
    ax.plot((0, x.iloc[-1]), (0, 0), "--", color="black", lw=line_width)


def plot_data_pair(calcium_data, isosbestic_data, title, kwargs, x, add_zero_line=False, units='', to_kilo=False):
    x_max = x.iloc[-1]
    multiplication_factor = 1000 if to_kilo else 1
    xticks, xticklabels, unit = utils.generate_xticks_and_labels(x_max)

    fig = plt.figure(figsize=(10, 5), dpi=200.)

    ax0 = plt.subplot(211)
    sub_plot(data=isosbestic_data, x=x, xticks=xticks, xticklabels=xticklabels, ax=ax0, laser='purple_laser',
             label='isosbestic', kwargs=kwargs, add_zero_line=add_zero_line, units=units,
             multiplication_factor=multiplication_factor)
    ax0.set_title("{} Isosbestic and Calcium signals".format(title.title()), fontsize=kwargs["fst"])

    ax1 = plt.subplot(212, sharex=ax0)
    sub_plot(data=calcium_data, x=x, xticks=xticks, xticklabels=xticklabels, ax=ax1, laser='blue_laser',
             label='calcium', kwargs=kwargs, add_zero_line=add_zero_line, units=units,
             multiplication_factor=multiplication_factor)
    ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

    plt.tight_layout()

    if kwargs["photometry_pp"]["multicursor"]:
        multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])  # FIXME: unused
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "{}_Signals.{}".format(title.title(), kwargs["extension"])), dpi=200.)


# FIXME: simplify signature
def sub_plot(data, x, xticks, xticklabels, ax, laser, label, kwargs, add_zero_line, units, multiplication_factor, baseline=None):
    x_max = x.iloc[-1]
    # TODO: check if compute xticks locally
    line_width = kwargs['lw']
    font_size = kwargs["fsl"]

    plot_handle, = ax.plot(x, data, alpha=0.8, c=kwargs["photometry_pp"][laser], lw=line_width)
    plot_handles = [plot_handle]
    labels = [label]
    if baseline is not None:
        plot_handle_2, = ax.plot(x, baseline, alpha=0.8, c="orange", lw=2)
        plot_handles.append(plot_handle_2)
        labels.append('baseline')
    if add_zero_line:
        add_line_at_zero(ax, x, line_width)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=font_size)
    ax.set_xlim(0, x_max)
    y_min, y_max, round_factor = utils.generate_yticks(data, 0.1)
    ax.set_yticks(np.arange(y_min, y_max + round_factor, round_factor))
    ax.set_yticklabels(
        ["{:.0f}".format(i) for i in np.arange(y_min, y_max + round_factor, round_factor) * multiplication_factor],
        fontsize=font_size)
    ax.set_ylim(y_min, y_max)
    if units:
        ax.set_ylabel(units, fontsize=font_size)
    ax.legend(handles=plot_handles, labels=labels, loc=2, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)


def plot_cropped_data(calcium, calcium_fc, isosbestic, isosbestic_fc, kwargs, x):
    x_max = x.iloc[-1]
    xticks, xticklabels, unit = utils.generate_xticks_and_labels(x_max)

    fig = plt.figure(figsize=(10, 5), dpi=200.)

    ax0 = plt.subplot(211)
    sub_plot(data=isosbestic, x=x, xticks=xticks, xticklabels=xticklabels, ax=ax0, laser='purple_laser',
             label='isosbestic', kwargs=kwargs, add_zero_line=False, units='mV',
             multiplication_factor=1000, baseline=isosbestic_fc)
    ax0.set_title("Smoothed Isosbestic and Calcium signals with respective baselines", fontsize=kwargs["fst"])

    ax1 = plt.subplot(212, sharex=ax0)
    sub_plot(data=calcium, x=x, xticks=xticks, xticklabels=xticklabels, ax=ax1, laser='blue_laser',
             label='calcium', kwargs=kwargs, add_zero_line=False, units='mV',
             multiplication_factor=1000, baseline=calcium_fc)
    ax1.set_xlabel("Time ({0})".format(unit), fontsize=kwargs["fsl"])

    plt.tight_layout()

    if kwargs["photometry_pp"]["multicursor"]:
        multi = MultiCursor(fig.canvas, [ax0, ax1], color='r', lw=1, vertOn=[ax0, ax1])  # FIXME: unused
    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "Baseline_Determination.{0}".format(kwargs["extension"])), dpi=200.)


def plot_ca_iso_regression(calcium, isosbestic, isosbestic_fitted, kwargs):
    plt.figure(figsize=(5, 5), dpi=200.)
    ax0 = plt.subplot(111)
    ax0.scatter(isosbestic, calcium, color="blue", s=0.5, alpha=0.05)
    ax0.plot(isosbestic, isosbestic_fitted, 'r-', linewidth=1)
    ax0.set_xlabel("Isosbestic", fontsize=kwargs["fsl"])
    ax0.set_ylabel("Calcium", fontsize=kwargs["fsl"])
    ax0.set_title("Inter-channel regression of Isosbestic and Calcium signals", fontsize=kwargs["fst"])
    ax0.tick_params(axis='both', which='major', labelsize=kwargs["fsl"])

    plt.tight_layout()

    if kwargs["save"]:
        plt.savefig(os.path.join(kwargs["save_dir"], "Interchannel_Regression.{0}".format(kwargs["extension"])),
                    dpi=200.)
