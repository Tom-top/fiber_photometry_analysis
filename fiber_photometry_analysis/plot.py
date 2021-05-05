#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:19:37 2020

@author: thomas.topilko
"""
import logging
import os

import numpy as np

import matplotlib.colors as col
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.widgets import MultiCursor

plt.style.use("default")
plt.rcParams['svg.fonttype'] = 'none'

import scipy.stats as stats

import seaborn as sns
from sklearn.metrics import auc

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import behavior_preprocessing as behav_preproc
from fiber_photometry_analysis import mask_operations as mask_op


def add_line_at_zero(ax, x, line_width, color='black'):
    """
    Creates a horizontal dashed line at y = 0 to signal the baseline
    """
    ax.plot((0, x.iloc[-1]), (0, 0), "--", color=color, lw=line_width)


def save_fig(fig_name, save_dir, params, save=False, **save_options):
    plt.tight_layout()
    if save:
        fig_path = os.path.join(save_dir,
                                "{}.{}".format(fig_name, params["saving"]["general"]["extension"]))
        plt.savefig(fig_path, dpi=200., **save_options)


def set_axis(ax, data, x, add_zero_line, fig_title, label_x, multiplication_factor, y_units, params,
             video_time=False):
    """
    Sets the X and Y axis

    :param matplotlib.axes.Axes ax: The axis to use to plot
    :param np.array data: The Y data
    :param np.array x: The X data
    :param bool add_zero_line: Whether to draw an horizontal line to indicate the baseline
    :param str fig_title: The name to title the figure
    :param bool label_x: Whether to label the time axis
    :param int multiplication_factor: The amount to scale the Y ticks by (1 for none)
    :param str y_units: The units of the Y axis
    :param dict params: The usual params dictionary containing all sorts of plotting parameters
    :return:
    """
    line_width = params["plotting"]["general"]["lw"]
    font_size = params["plotting"]["general"]["fsl"]

    if add_zero_line:
        add_line_at_zero(ax, x, line_width)
    x_max = x.iloc[-1]
    xticks, xticklabels, unit = utils.generate_xticks_and_labels(x_max, params, video_time=video_time)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=font_size)
    ax.set_xlim(0, x_max)
    if label_x:
        ax.set_xlabel("Time ({0})".format(unit), fontsize=font_size)
    y_min, y_max, round_factor = utils.generate_yticks(data, 0.1)
    int_ticks = np.arange(y_min, y_max + round_factor, round_factor)
    ax.set_yticks(int_ticks)
    int_ticks = int_ticks * multiplication_factor
    ticks = ["{:.1f}".format(i) for i in int_ticks]
    ax.set_yticklabels(ticks, fontsize=font_size)
    ax.set_ylim(y_min, y_max)
    if y_units:
        ax.set_ylabel(y_units, fontsize=font_size)
    ax.legend(loc=2, fontsize=font_size)
    if fig_title:
        ax.set_title(fig_title, fontsize=params["plotting"]["general"]["fst"])
    ax.tick_params(axis='both', which='major', labelsize=font_size)


def check_delta_f_with_behavior(bool_map, params, zorder=(0, 1), behavior_tag="", fig_name="behavior_overlay",
                                extra_legend_label="", video_time=False, save=False, plot=False):
    """
    Function that plots the pre-processed photometry data and overlays the
    behavior data.

    :param Union(np.array, list(np.array, np.array) bool_map: The behaviour data as a binary array
    :param dict params: The plotting parameters
    :param (int, int) zorder: Tuple of int to indicate the z plot order of each subplot
    :param str fig_name: The file_name of the figure without extension
    :param str extra_legend_label: The label of the second plot if bool_map is a list of arrays
    """

    data = params["photometry_data"]["dFF"]["dFF"]
    x = params["photometry_data"]["dFF"]["x"]
    line_width = params["plotting"]["general"]["lw"]
    multiplication_factor = 1 if params["signal_pp"]["standardization"]["standardize"] else 100
    fig_title = fig_name

    colors = ["red", "orange"]
    y_min, y_max, round_factor = utils.generate_yticks(data, 0.1)
    event_plot_args = {
        "linelengths": y_max - y_min,
        "lineoffsets": (y_max + y_min) / 2,
        "linewidths" : 1.,
        "alpha": 1,
    }

    plt.ioff()
    fig = plt.figure(figsize=(10, 3), dpi=200.)
    ax = plt.subplot(111)

    if isinstance(bool_map, np.ndarray):
        raster_data_raw = mask_op.get_up_times_bool_map(bool_map) / params["signal_pp"]["general"]["resolution_data"]
        # raster_data_raw = behav_preproc.get_position_from_bool_map(bool_map, params["resolution_data"])
    elif isinstance(bool_map, list) and len(bool_map) == 2:
        raster_data_raw = mask_op.get_up_times_bool_map(bool_map[0]) / params["signal_pp"]["general"]["resolution_data"]
        raster_data_processed = mask_op.get_up_times_bool_map(bool_map[1]) / params["signal_pp"]["general"]["resolution_data"]
        # raster_data_raw = behav_preproc.get_position_from_bool_map(bool_map[0], params["resolution_data"])
        # raster_data_processed = behav_preproc.get_position_from_bool_map(bool_map[1], params["resolution_data"])

        # ax.eventplot(raster_data_processed,
        #              color=colors[zorder[0]],
        #              zorder=zorder[0],
        #              label=extra_legend_label,
        #              **event_plot_args)

        label_patch = patches.Rectangle((0, 0),
                                        0,
                                        0,
                                        facecolor=colors[zorder[0]],
                                        linewidth=0,
                                        label=extra_legend_label,)
        ax.add_patch(label_patch)

        for s, e in raster_data_processed:
            patch = patches.Rectangle((s, y_min),
                                      e-s,
                                      y_max - y_min,
                                      facecolor=colors[zorder[0]],
                                      linewidth=0,
                                      zorder=0)
            ax.add_patch(patch)
    else:
        raise NotImplementedError('bool_map must be a list of arrays or a numpy array. got {} instead'
                                  .format(type(bool_map)))

    label_patch = patches.Rectangle((0, 0),
                                    0,
                                    0,
                                    facecolor=colors[zorder[1]],
                                    linewidth=0,
                                    label=behavior_tag, )
    ax.add_patch(label_patch)

    for s, e in raster_data_raw:
        patch = patches.Rectangle((s, y_min),
                                  e - s,
                                  y_max - y_min,
                                  facecolor=colors[zorder[1]],
                                  linewidth=0,
                                  zorder=1)
        ax.add_patch(patch)
    # ax.eventplot(raster_data_raw,
    #              color=colors[zorder[1]],
    #              zorder=zorder[1],
    #              label=params["behavior_tag"],
    #              **event_plot_args)

    ax.plot(x, data, zorder=2, color="green", lw=line_width)

    set_axis(ax, data, x, True, fig_title, True, multiplication_factor, '', params, video_time=video_time)

    save_fig(fig_name, params["general"]["behavior_saving_directory"], params, save=save)
    if plot:
        plt.show()
    else:
        plt.close(fig)


def peri_event_plot(data_around_major_bouts, length_major_bouts, interpolated_sampling_rate, params, behavior_tag="", sem=True,
                    style="individual", individual_colors=False, cmap="inferno", save_options={"bbox_inches":'tight'}):
    """
    Function that plots the peri-event photometry data in two different formats.
    First a line plot showing the average signal before and after initiation of the behavior,
    Second a heatmap showing each individual signal traces before and after initiation of the behavior.

    :param np.array data_around_major_bouts: list of the pre-processed photometry data
            before and after initiation of the behavior
    :param list length_major_bouts: list of the length of each behavioral bout
    :param line_colors:
    :param str cmap:  colormap used for the heatmap
    :param dict params:dictionary with additional parameters
    """

    crop_pre = (params["plotting"]["peri_event"]["time_before_event"] -
                params["plotting"]["peri_event"]["extract_time_before_event"])*interpolated_sampling_rate
    crop_post = -(params["plotting"]["peri_event"]["time_after_event"] -
                  params["plotting"]["peri_event"]["extract_time_after_event"]) * interpolated_sampling_rate
    if crop_post == 0:
        crop_post = None

    fig = plt.figure(figsize=(5, 5), dpi=200.)  # Creates a figure
    ax0 = plt.subplot(2, 1, 1)  # Creates a subplot for the first figure

    data_around_major_bouts = data_around_major_bouts[:, crop_pre:crop_post]

    mean_data_around_bouts = np.mean(data_around_major_bouts, axis=0)  # Computes the mean for the peri-event photometry data
    if not sem or len(data_around_major_bouts) == 1:
        std_data_around_bouts = np.std(data_around_major_bouts, axis=0)  # Computes the standard deviation for the peri-event photometry data
    else:
        std_data_around_bouts = stats.sem(data_around_major_bouts, axis=0)
    # =============================================================================
    #     First Plot
    # =============================================================================

    x_range = np.linspace(0, len(mean_data_around_bouts), len(mean_data_around_bouts))
    ax0.plot(x_range, mean_data_around_bouts, color="green", alpha=1., lw=params["plotting"]["general"]["lw"] + 1, zorder=1)  # Plots the average signal
    line_color = "green"

    if style == "individual":  # If individual traces are to be displayed in the line plot
        for line in data_around_major_bouts:
            if individual_colors:
                ax0.plot(line, alpha=0.5, lw=params["plotting"]["general"]["lw"], zorder=0)  # Plots individual traces
            else:
                ax0.plot(line, color=line_color, alpha=0.5, lw=params["plotting"]["general"]["lw"], zorder=0)  # Plots individual traces

        ax0.plot(x_range, mean_data_around_bouts + std_data_around_bouts,
                 color=line_color, alpha=0.5, lw=params["plotting"]["general"]["lw"], zorder=1)
        ax0.plot(x_range, mean_data_around_bouts - std_data_around_bouts,
                 color=line_color, alpha=0.5, lw=params["plotting"]["general"]["lw"], zorder=1)

        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts + std_data_around_bouts),
                         color=line_color, alpha=0.1)
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts - std_data_around_bouts),
                         color=line_color, alpha=0.1)

        y_min, y_max, round_factor = utils.generate_yticks(data_around_major_bouts.flatten(), 0.2)
    elif style == "average":
        ax0.plot(x_range, mean_data_around_bouts + std_data_around_bouts,
                 color=line_color, alpha=0.3, lw=params["plotting"]["general"]["lw"])
        ax0.plot(x_range, mean_data_around_bouts - std_data_around_bouts,
                 color=line_color, alpha=0.3, lw=params["plotting"]["general"]["lw"])

        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts + std_data_around_bouts),
                         color=line_color, alpha=0.1)
        ax0.fill_between(x_range, mean_data_around_bouts, (mean_data_around_bouts - std_data_around_bouts),
                         color=line_color, alpha=0.1)
        y_min, y_max, round_factor = utils.generate_yticks(np.concatenate((mean_data_around_bouts + std_data_around_bouts,
                                                                           mean_data_around_bouts - std_data_around_bouts)), 0.2)
    else:
        raise NotImplementedError('Unrecognised style keyword "{}"'.format(params["plotting"]["peri_event"]["style"]))
    #y_min=-2
    #y_max=4
    y_min = -0.1
    y_max = 0.5
    round_factor = 0.1
    y_range = y_max - y_min
    ax0.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
    if params["signal_pp"]["standardization"]["standardize"]:
        ax0.set_yticklabels(["{:.2f}".format(i) for i in np.arange(y_min, y_max + round_factor, round_factor)],
                            fontsize=params["plotting"]["general"]["fsl"])
    else:
        ax0.set_yticklabels(np.arange(y_min * 100, (y_max + round_factor) * 100, round_factor * 100),
                            fontsize=params["plotting"]["general"]["fsl"])
    ax0.set_ylim(y_min, y_max)

    # Creates a gray square on the line plot that represents the average length of a behavioral bout
    patch = patches.Rectangle(((params["plotting"]["peri_event"]["time_before_event"]+0.5)*interpolated_sampling_rate, - y_range*0.1),
                              width=np.mean(length_major_bouts)*interpolated_sampling_rate,
                              height=y_range*0.1,
                              color="gray",
                              lw=0,
                              alpha=0.7)

    ax0.add_patch(patch)  # Adds the patch to the plot
    ax0.plot((0, len(mean_data_around_bouts)), (0, 0), "--", color="black", lw=params["plotting"]["general"]["lw"])  # Creates a horizontal dashed line at y = 0 to signal the baseline
    vline = ax0.axvline(x=(params["plotting"]["peri_event"]["time_before_event"]+0.5) * interpolated_sampling_rate,
                        color='red', linestyle='--', lw=params["plotting"]["general"]["lw"])  # Creates a vertical dashed line at x = 0 to signal the begining of the behavioral bout

    ax0.set_xticks(np.linspace(0, ((params["plotting"]["peri_event"]["time_before_event"] +
                                    params["plotting"]["peri_event"]["time_after_event"] + 1)
                                   * interpolated_sampling_rate), 5))  # Generates ticks for the x axis
    ax0.set_xticklabels(np.linspace(-params["plotting"]["peri_event"]["time_before_event"],
                                     params["plotting"]["peri_event"]["time_after_event"],
                                     5),
                        fontsize=params["plotting"]["general"]["fsl"],
                        )  # Generates labels for the x axis
    ax0.set_xlim(0, len(mean_data_around_bouts))  # Sets the limits for the x axis

    ax0.legend(handles=[vline, patch], labels=["Begining of bout", "Average behavioral bout length"],
               loc=2, fontsize=params["plotting"]["general"]["fsl"])  # Displays the legend of the first figure

    # =============================================================================
    #     Second Plot
    # =============================================================================

    ax1 = plt.subplot(2, 1, 2)  # Creates a subplot for the second figure

    if not params["plotting"]["peri_event"]["normalize_heatmap"]:
        if params["signal_pp"]["standardization"]["standardize"]:
            heatmap = ax1.imshow(data_around_major_bouts, cmap=cmap, aspect="auto",
                                 interpolation="none", vmin=-1, vmax=3)
        else:
            heatmap = ax1.imshow(data_around_major_bouts, cmap=cmap, aspect="auto",
                                 interpolation="none", vmin=-0.1, vmax=0.5)
    else:
        heatmap = ax1.imshow(data_around_major_bouts, cmap=cmap, aspect="auto",
                             norm=col.LogNorm(), interpolation="none")  # norm=matplotlib.colors.LogNorm()

    ax1.axvline(x=(params["plotting"]["peri_event"]["time_before_event"]+0.5)*interpolated_sampling_rate,
                color='red', linestyle='--', lw=params["plotting"]["general"]["lw"])

    ax1.set_xticks(np.linspace(0, ((params["plotting"]["peri_event"]["time_before_event"] +
                                    params["plotting"]["peri_event"]["time_after_event"]+1)
                                   * interpolated_sampling_rate), 5)) #Generates ticks for the x axis
    ax1.set_xticklabels(np.linspace(-params["plotting"]["peri_event"]["time_before_event"],
                                    params["plotting"]["peri_event"]["time_after_event"], 5),
                        fontsize=params["plotting"]["general"]["fsl"]) #Generates labels for the x axis
    ax1.set_xlim(0, len(mean_data_around_bouts))  # Sets the limits for the x axis
    ax1.set_xlabel("Time (s)", fontsize=params["plotting"]["general"]["fsl"])  # Displays the label for the x axis

    ax1.set_yticks([])
    ax1.set_ylim(-0.5, len(data_around_major_bouts) - 0.5)
    ax1.set_ylabel("Duration of individual bouts (s)", fontsize=params["plotting"]["general"]["fsl"], labelpad=20)  # Displays the label for the x axis

    for n, line in enumerate(length_major_bouts):
        if len(length_major_bouts) > 20:
            if n % 5 == 0:
                ax1.text(-len(mean_data_around_bouts) * 0.01, n, round(line), ha="right", va="center", fontsize=6)
        else:
            ax1.text(-len(mean_data_around_bouts) * 0.01, n, round(line), ha="right", va="center", fontsize=6)
    
    plt.gca().invert_yaxis()
    cax = fig.add_axes([0.98, 0.1, 0.02, 0.35])
    cb = plt.colorbar(heatmap, ax=ax1, cax=cax)
    cb.ax.tick_params(labelsize=params["plotting"]["general"]["fsl"])
    cb.ax.get_yaxis().labelpad = 8
    
    if params["signal_pp"]["standardization"]["standardize"]:
        ax0.set_ylabel(r"$\Delta$F/F (Z-Score)", fontsize=params["plotting"]["general"]["fsl"]) #Displays the label for the y axis
        ax0.set_title(r"$\Delta$F/F (Z-Score) in function of time before & after {0} initiation"
                      .format(behavior_tag),
                      fontsize=params["plotting"]["general"]["fst"]) #Displays the titel for the first figure
        ax1.set_title(r"Individual $\Delta$F/F (Z-Score) in function of time before & after {0} initiation"
                      .format(behavior_tag),
                      fontsize=params["plotting"]["general"]["fst"])
        cb.ax.set_ylabel(r"$\Delta$F/F (Z-Score)",
                         rotation=270,
                         fontsize=params["plotting"]["general"]["fsl"])
    else:
        ax0.set_ylabel(r"$\Delta$F/F (%)", fontsize=params["plotting"]["general"]["fsl"]) #Displays the label for the y axis
        ax0.set_title(r"$\Delta$F/F (%) in function of time before & after {0} initiation"
                      .format(behavior_tag),
                      fontsize=params["plotting"]["general"]["fst"]) #Displays the titel for the first figure
        ax1.set_title(r"Individual $\Delta$F/F (%) in function of time before & after {0} initiation"
                      .format(behavior_tag), fontsize=params["plotting"]["general"]["fst"])
        cb.ax.set_ylabel(r"$\Delta$F/F (%)", rotation=270, fontsize=params["plotting"]["general"]["fsl"])

    save_fig('Peri_Event_Plot', params["general"]["behavior_saving_directory"], params,
             params["plotting"]["peri_event"]["save"], **save_options)
    if params["plotting"]["peri_event"]["plot"]:
        plt.show()
    else:
        plt.close(fig)

    perievent_metadata = {"data": data_around_major_bouts,
                          "data_bout_length": length_major_bouts,
                          "data_sampling_rate": interpolated_sampling_rate,
                          "duration_pre": params["plotting"]["peri_event"]["time_before_event"],
                          "duration_post": params["plotting"]["peri_event"]["time_after_event"],
                          }

    return perievent_metadata


def peri_event_bar_plot(data_around_major_bouts, interpolated_sampling_rate, params, duration_pre=2,
                         duration_post=2, save_options=None):
    """
    Function that compares the area under the curve (AUC) before and after the intitation of the behavior.
    The results are summarized in a bar plot showing the AUC before and after initiation of the behavior.

    :param np.array data_around_major_bouts: list of the pre-processed photometry data
    :param dict params: dictionary with additional parameters
    """

    center = int((params["plotting"]["peri_event"]["time_before_event"]
                  + params["plotting"]["peri_event"]["time_after_event"] + 1)/2)
    time_before = np.linspace(0, duration_pre, duration_pre * interpolated_sampling_rate)
    data_before = data_around_major_bouts[:, (center - duration_pre) * interpolated_sampling_rate :
                                             center * interpolated_sampling_rate]
    
    time_after = np.linspace(0, duration_post, duration_post * interpolated_sampling_rate)
    data_after = data_around_major_bouts[:, center * interpolated_sampling_rate :
                                            (center + duration_post) * interpolated_sampling_rate]
    
    all_auc1 = [auc(time_before, i) for i in data_before]  # OPTIMISE: map or apply
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
    palette = ["#81dafc", "#ff6961"]
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
                  s=5,
                  palette=palette,
                  **plot_params
                  # zorder = 0,
                  )
    
    y_max = max([max(all_auc1), max(all_auc2)]) + max([max(all_auc1), max(all_auc2)])*0.1
    y_min = min([min(all_auc1), min(all_auc2)]) + min([min(all_auc1), min(all_auc2)])*0.1
    ax0.plot((0, 0), (y_max+y_max*0.05, y_max+y_max*0.1), lw=1, color="black")
    ax0.plot((1, 1), (y_max+y_max*0.05, y_max+y_max*0.1), lw=1, color="black")
    ax0.plot((0, 1), (y_max+y_max*0.1, y_max+y_max*0.1), lw=1, color="black")
    
    logging.info("The comparison of AUC for before and after the behavioral initiation is : {}".format(pval))

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
    ax0.text(0.5, y_max+y_max*0.1, stars, ha="center")

    int_ticks = np.arange(round(y_min), round(y_max + y_max * 0.3) + 2, 5)
    ax0.set_yticks(int_ticks)
    ax0.set_yticklabels(int_ticks, fontsize=params["plotting"]["general"]["fsl"])
    ax0.plot((-0.5, 1.5), (0, 0), lw=1, color="black")  # FIXME: add_zero_line ??
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Before initiating {0}".format(params["general"]["general"]["behavior_tag"]),
                         "After initiating {0}".format(params["general"]["general"]["behavior_tag"])],
                        fontsize=params["plotting"]["general"]["fsl"])
    
    ax0.set_ylim(y_min+y_min*0.5, y_max+y_max*0.5)
    ax0.set_ylabel("Area under the curve (AUC)", fontsize=params["plotting"]["general"]["fsl"])
    ax0.set_title("Before vs After initiation of {0} Response Changes".format(params["general"]["general"]["behavior_tag"]),
                  fontsize=params["plotting"]["general"]["fst"])

    ax0.text(-0.2, y_max+y_max*0.3, r'$\int_{i=-5}^0 f(x)dx$', ha="center")
    ax0.text(1.2, y_max + y_max * 0.3, r'$\int_{i=0}^5 f(x)dx$', ha="center")

    save_fig('AUC', params["general"]["behavior_saving_directory"], params, save_options=save_options)
    plt.show()


def peri_event_comparison_bar_plot(data_around_major_bouts, interpolated_sampling_rate, params, duration_pre=2,
                        duration_post=2):
    """
    Function that compares the area under the curve (AUC) before and after the intitation of the behavior.
    The results are summarized in a bar plot showing the AUC before and after initiation of the behavior.

    :param np.array data_around_major_bouts: list of the pre-processed photometry data
    :param dict params: dictionary with additional parameters
    """

    swarplot_data = []
    auc_mean = []
    auc_std = []

    for i in data_around_major_bouts:
        center = int((params["peri_event"]["graph_distance_pre"] + params["peri_event"]["graph_distance_post"] + 1) / 2)
        time_before = np.linspace(0, duration_pre, duration_pre * interpolated_sampling_rate)
        data_before = i[:, (center - duration_pre) * interpolated_sampling_rate:
                                                 center * interpolated_sampling_rate]

        time_after = np.linspace(0, duration_post, duration_post * interpolated_sampling_rate)
        data_after = i[:, center * interpolated_sampling_rate:
                                                (center + duration_post) * interpolated_sampling_rate]

        all_auc1 = [auc(time_before, i) for i in data_before]  # OPTIMISE: map or apply
        all_auc2 = [auc(time_after, i) for i in data_after]
        diff = np.array([all_auc2]) - np.array([all_auc1])
        swarplot_data.append(diff[0])
        mean0 = auc(time_before, np.mean(data_before, axis=0))
        std0 = auc(time_before, np.std(data_before, axis=0))
        mean1 = auc(time_after, np.mean(data_after, axis=0))
        std1 = auc(time_after, np.std(data_after, axis=0))
        auc_mean.append(mean1 - mean0)
        auc_std.append(std1 - std0)

    stat, pval = stats.ttest_ind(swarplot_data[0], swarplot_data[1], equal_var=False)

    fig = plt.figure(figsize=(5, 5), dpi=200.)
    ax0 = plt.subplot(1, 1, 1)
    palette = ["#81dafc", "#ff6961"]
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

    sns.swarmplot(np.concatenate([np.full_like(swarplot_data[0], 0), np.full_like(swarplot_data[1], 1)]),
                  np.concatenate([swarplot_data[0], swarplot_data[1]]),
                  ax=ax0,
                  s=5,
                  palette=palette,
                  **plot_params
                  # zorder = 0,
                  )

    y_max = max([max(swarplot_data[0]), max(swarplot_data[1])]) + max([max(swarplot_data[0]), max(swarplot_data[1])]) * 0.1
    y_min = min([min(swarplot_data[0]), min(swarplot_data[1])]) + min([min(swarplot_data[0]), min(swarplot_data[1])]) * 0.1
    ax0.plot((0, 0), (y_max + y_max * 0.05, y_max + y_max * 0.1), lw=1, color="black")
    ax0.plot((1, 1), (y_max + y_max * 0.05, y_max + y_max * 0.1), lw=1, color="black")
    ax0.plot((0, 1), (y_max + y_max * 0.1, y_max + y_max * 0.1), lw=1, color="black")

    logging.info("The comparison of AUC for before and after the behavioral initiation is : {}".format(pval))

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
    ax0.text(0.5, y_max + y_max * 0.1, stars, ha="center")

    int_ticks = np.arange(round(y_min), round(y_max + y_max * 0.3) + 2, 5)
    ax0.set_yticks(int_ticks)
    ax0.set_yticklabels(int_ticks, fontsize=params["fsl"])
    ax0.plot((-0.5, 1.5), (0, 0), lw=1, color="black")  # FIXME: add_zero_line ??
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Before initiating {0}".format(params["behavior_tag"]),
                         "After initiating {0}".format(params["behavior_tag"])],
                        fontsize=params["fsl"])

    ax0.set_ylim(y_min + y_min * 0.5, y_max + y_max * 0.5)
    ax0.set_ylabel("Area under the curve (AUC)", fontsize=params["fsl"])
    ax0.set_title("Before vs After initiation of {0} Response Changes".format(params["behavior_tag"]),
                  fontsize=params["fst"])

    ax0.text(-0.2, y_max + y_max * 0.3, r'$\int_{i=-5}^0 f(x)dx$', ha="center")
    ax0.text(1.2, y_max + y_max * 0.3, r'$\int_{i=0}^5 f(x)dx$', ha="center")

    save_fig('AUC', params["save_dir_behavior"], params, save_options={'bbox_inches': 'tight'})
    plt.show()
    if params["clear_and_close_figures"]:
        fig.clf()
        plt.close()


def plot_data_pair(calcium_data, isosbestic_data, x, title, params, add_zero_line=False, units='', to_milli=False,
                   isosbestic_baseline=None, calcium_baseline=None, fig_name=None, fig_title=None, plot=False, save=False):
    """
    Plot 2 suplots one above the other with calcium data and then isosbestic data.

    :param save:
    :param np.array calcium_data: The physiological data channel
    :param np.array isosbestic_data: The morphological optical channel
    :param np.array x: The time data for the two data types above
    :param str title: The base title used to compute the figure name and title (if not supplied)
    :param dict params: Dictionary of additional parameters
    :param bool add_zero_line: Whether to add an horizontal line at 0
    :param str units: The name of the Y units
    :param bool to_milli: Whether to convert Y units to milli units (i.e. *1000)
    :param np.array isosbestic_baseline: The baseline for the morphological channel
    :param np.array calcium_baseline: The baseline for the physiological channel
    :param str fig_name: The path basename of the figure (to not compute using title)
    :param str fig_title: The title of the figure (to not compute using title)
    :return:
    """
    if fig_name is None:
        fig_name = "{}_Signals".format(title.title())
    if fig_title is None:
        fig_title = "{} Isosbestic and Calcium signals".format(title.title())
    multiplication_factor = 1000 if to_milli else 1
    fig = plt.figure(figsize=(10, 5), dpi=200.)

    ax0 = plt.subplot(211)
    line_width = params["plotting"]["general"]["lw"]
    ax0.plot(x, isosbestic_data, alpha=0.8, c=params["general"]["photometry"]["purple_laser"], lw=line_width, label='isosbestic')
    if isosbestic_baseline is not None:
        ax0.plot(x, isosbestic_baseline, alpha=0.8, c="orange", lw=0.5, label='baseline')
    set_axis(ax0, isosbestic_data, x, add_zero_line, '', False, multiplication_factor, units, params)
    ax0.set_title(fig_title, fontsize=params["plotting"]["general"]["fst"])  # REFACTOR: move inside sub_plot

    ax1 = plt.subplot(212, sharex=ax0)
    ax1.plot(x, calcium_data, alpha=0.8, c=params["general"]["photometry"]['blue_laser'], lw=line_width, label='calcium')
    if calcium_baseline is not None:
        ax1.plot(x, calcium_baseline, alpha=0.8, c="orange", lw=0.5, label='baseline')
    set_axis(ax1, calcium_data, x, add_zero_line, '', True, multiplication_factor, units, params)
    save_fig(fig_name, params["general"]["plot_directory"], params, save=save)
    if plot:
        plt.show()
    else:
        plt.close(fig)

def plot_cropped_data(calcium_data, calcium_fc, isosbestic_data, isosbestic_fc, x, params, plot=False, save=False):
    """
    Plot the pre and post crop/detrend data for calcium and isosbestic

    :param save:
    :param np.array calcium_data:
    :param np.array calcium_fc:
    :param np.array isosbestic_data:
    :param np.array isosbestic_fc:
    :param dict params:
    :param x:
    :return:
    """
    fig_name = 'Baseline_Determination'
    fig_title = "Smoothed Isosbestic and Calcium signals with respective baselines"
    plot_data_pair(calcium_data, isosbestic_data, x, '', params, units='mV', to_milli=True,
                   isosbestic_baseline=isosbestic_fc, calcium_baseline=calcium_fc, fig_name=fig_name,
                   fig_title=fig_title, plot=plot, save=save)


def plot_ca_iso_regression(calcium, isosbestic, isosbestic_fitted, params, plot=False, save=False):
    """
    Scatter plot and regression of the calcium vs isosbestic signals

    :param save:
    :param np.array calcium:
    :param np.array isosbestic:
    :param np.array isosbestic_fitted:
    :param dict params:
    :return:
    """
    fig = plt.figure(figsize=(5, 5), dpi=200.)
    ax = plt.subplot(111)
    ax.scatter(isosbestic, calcium, color="blue", s=0.5, alpha=0.05)
    # ax.plot(isosbestic, isosbestic_fitted, 'r-', linewidth=1)
    ax.set_xlabel("Isosbestic", fontsize=params["plotting"]["general"]["fsl"])
    ax.set_ylabel("Calcium", fontsize=params["plotting"]["general"]["fsl"])
    ax.set_title("Inter-channel regression of Isosbestic and Calcium signals", fontsize=params["plotting"]["general"]["fst"])
    ax.tick_params(axis='both', which='major', labelsize=params["plotting"]["general"]["fsl"])

    save_fig("Interchannel_Regression", params["general"]["plot_directory"], params, save=save)
    if plot:
        plt.show()
    else:
        plt.close(fig)


def plot_delta_f(calcium, isosbestic, x, params, plot=False, save=False):
    """
    Plot the delta_F/F

    :param save:
    :param np.array calcium:
    :param np.array isosbestic:
    :param np.array x:
    :param dict params:
    :return:
    """
    delta_f = calcium - isosbestic
    standardize = params["signal_pp"]["standardization"]["standardize"]
    multiplication_factor = 1 if standardize else 100
    y_units = r"z-score $\Delta$F/F" if standardize else r"$\Delta$F/F"
    fig_title = y_units

    fig = plt.figure(figsize=(10, 3), dpi=200.)
    ax = plt.subplot(111)

    ax.plot(x, delta_f, alpha=0.8, c="green", lw=params["plotting"]["general"]['lw'], label=y_units)
    set_axis(ax, np.concatenate((isosbestic, calcium)), x, True, fig_title,
             True, multiplication_factor, y_units, params)

    save_fig("Delta_F", params["general"]["plot_directory"], params, save=save)
    if plot:
        plt.show()
    else:
        plt.close(fig)


def plot_aligned_channels(x, isosbestic, calcium, params, plot=False, save=False):  # FIXME: plot only. Rename
    """
    Function that performs the alignment of the fitted isosbestic and standardized (or not) calcium
    signals and displays it in a plot.

    :param save:
    :param np.array x: The time data in X
    :param np.array isosbestic: The fitted isosbestic signal
    :param np.array calcium: The standardized (or not) calcium signal
    :param dict params: Dictionary with the parameters
    """
    standardize = params["signal_pp"]["standardization"]["standardize"]
    multiplication_factor = 1 if standardize else 100
    y_units = "z-score" if standardize else "Change in signal (%)"
    fig_title = "Alignement of Isosbestic and Calcium signals"
    fig_name = 'Alignement'
    line_width = params["plotting"]["general"]['lw']

    fig = plt.figure(figsize=(10, 3), dpi=200.)
    ax = plt.subplot(111)

    ax.plot(x, calcium, alpha=0.8, c=params["general"]["photometry"]["blue_laser"],
            lw=line_width, zorder=0, label='calcium')
    ax.plot(x, isosbestic, alpha=0.8, c=params["general"]["photometry"]["purple_laser"],
            lw=line_width, zorder=1, label='isosbestic')
    set_axis(ax, np.concatenate((isosbestic, calcium)), x, True, fig_title,
             True, multiplication_factor, y_units, params)

    save_fig(fig_name, params["general"]["plot_directory"], params, save=save)
    if plot:
        plt.show()
    else:
        plt.close(fig)
