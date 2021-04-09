#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:35:52 2020

@author: thomas.topilko
"""
import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt

import moviepy.editor as mpy
import numpy as np

from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import photometry_io
from fiber_photometry_analysis import generic_signal_processing as gen_preproc
from fiber_photometry_analysis import signal_preprocessing as preproc
from fiber_photometry_analysis import behavior_preprocessing as behav_preproc
from fiber_photometry_analysis import mask_operations as mask_op
from fiber_photometry_analysis import parameters
from fiber_photometry_analysis import plot
from fiber_photometry_analysis import video_plot as v_plot
from fiber_photometry_analysis.behavior_preprocessing import set_ranges_high
from fiber_photometry_analysis.logger import init_logging
from fiber_photometry_analysis.logger import config as logger_config


@dataclass
class Coordinates:
    x1: int
    x2: int
    y1: int
    y2: int


# Setting the working directory and the related files
folder_tag = "210408_353_Photo"
experiment_tag = "210408"
# mice = ["130_{}".format(str(i).zfill(4)) for i in np.arange(0, 11, 1)]
# mice = ["164_0029"]
# tags = [2, 4, 19, 20, 21, 23, 25, 27, 28, 30, 31]
# mice = ["193_{}".format(str(i).zfill(4)) for i in tags]
tags = [0]
mice = ["353_{}".format(str(i).zfill(4)) for i in tags]
# mice = ["193_0019"]

# base_directory = "/Users/tomtop/Desktop/Photometry"
# working_directory = os.path.join(base_directory, "{}".format(folder_tag))
# utils.replace_photometry_files_into_folders(working_directory)
# utils.rename_video_names(working_directory)

base_directory = "/Users/tomtop/Desktop/Photometry"

for m in mice:
    invert = True
    mouse = m
    working_directory = os.path.join(base_directory, "{}/{}".format(folder_tag, mouse))
    config = parameters.load_config(working_directory, experiment_tag, mouse, overwrite=True)

    photometry_data_test = photometry_io.save_dataframe_to_feather(config["general"]["photometry_file"], working_directory)
    photometry_data = preproc.load_photometry_data(photometry_data_test, config, recompute=True)
    params["photometry_data"] = photometry_data  # Adds a new parameter containing all the photometry data

    if invert:
        params["save_dir_behavior"] = os.path.join(params["save_dir"],
                                                   "Inverted_" + "_".join(params["behavior_to_segment"]))
    else:
        params["save_dir_behavior"] = os.path.join(params["save_dir"], "_".join(params["behavior_to_segment"]))
    if not os.path.exists(params["save_dir_behavior"]):
        os.mkdir(params["save_dir_behavior"])
    behavior_bouts = behav_preproc.extract_behavior_data(behavior_manual_file,
                                                         params["behavior_to_segment"])  # REFACTOR: replace by ranges

    # params["resolution_data"] = behav_preproc.estimate_behavior_time_resolution(starts, ends)
    params["resolution_data"] = params["recording_sampling_rate"]
    n_points = behav_preproc.estimate_length_bool_map(params["recording_duration"], params["resolution_data"])
    behavior_bouts_matched = behav_preproc.match_time_behavior_to_photometry(behavior_bouts,
                                                                             params["recording_duration"],
                                                                             params["video_duration"],
                                                                             params["resolution_data"])
    bool_map = set_ranges_high(np.zeros(n_points), behavior_bouts_matched)
    break
    trimmed_bool_map = behav_preproc.trim_behavioral_data(bool_map,
                                                          params["crop_start"],
                                                          params["crop_end"],
                                                          params["resolution_data"])

    # INVERSION TO LOOK FOR SMALL WAKE PERIODS
    if invert:
        trimmed_bool_map = 1 - trimmed_bool_map

    # filtered_bool_map = mask_op.filter_large_events(trimmed_bool_map, 100*241)
    filtered_bool_map = trimmed_bool_map

    plot.check_delta_f_with_behavior(filtered_bool_map, params, fig_name="behavior_overlay_raw",
                                     extra_legend_label=None, video_time=True)

    merged_bool_map = mask_op.merge_neighboring_events(filtered_bool_map, max_bout_gap)

    plot.check_delta_f_with_behavior([filtered_bool_map, merged_bool_map], params, fig_name="behavior_overlay_merged",
                                     zorder=[0, 1], extra_legend_label="Merged", video_time=True)

    large_events_bool_map = mask_op.filter_small_events(merged_bool_map, minimal_bout_length)

    plot.check_delta_f_with_behavior([large_events_bool_map, merged_bool_map], params,
                                     fig_name="behavior_overlay_large_events",
                                     zorder=[0, 1], extra_legend_label="Small bouts", video_time=True)

    params["peri_event"]["graph_distance_pre"] = 15
    params["peri_event"]["graph_distance_post"] = 15
    edge_filtered_bool_map = mask_op.filter_edge_events(large_events_bool_map, params["resolution_data"],
                                                        params["peri_event"]["graph_distance_pre"],
                                                        params["peri_event"]["graph_distance_post"])

    plot.check_delta_f_with_behavior([edge_filtered_bool_map, large_events_bool_map], params,
                                     fig_name="behavior_overlay_filtered_edge_events",
                                     zorder=[0, 1], extra_legend_label="Filtered edge events", video_time=True)

    trimmed_duration = len(trimmed_bool_map) / params["resolution_data"]
    new_sr = gen_preproc.round_to_closest_ten(params["recording_sampling_rate"])
    # new_sr = 241
    new_x = gen_preproc.generate_new_x(new_sr, trimmed_duration)
    interpolated_delta_f = gen_preproc.interpolate_signal(params["photometry_data"]["dFF"]["x"],
                                                          params["photometry_data"]["dFF"]["dFF"],
                                                          new_x)

    filtered_bool_map_new = edge_filtered_bool_map[:len(interpolated_delta_f)]
    # print(len(interpolated_delta_f))
    # print(len(filtered_bool_map_new))
    # import pandas as pd
    # df_feather = pd.DataFrame({"time(s)" : new_x,
    #                           "delta_f" : interpolated_delta_f,
    #                           "behavior_map" : filtered_bool_map_new,
    #                           })
    # df_feather.to_feather(os.path.join("/Users/tomtop/Desktop", "data_{}_{}.feather".format(experiment, mouse)))
    # plt.plot(new_x, interpolated_delta_f)
    # plt.plot(new_x, filtered_bool_map_new)

    start_behavioral_bouts = mask_op.find_start_points_events(edge_filtered_bool_map)
    length_behavioral_bouts = mask_op.get_length_events(edge_filtered_bool_map, params["resolution_data"])

    delta_f_around_bouts = behav_preproc.extract_peri_event_photometry_data(interpolated_delta_f, new_sr,
                                                                            start_behavioral_bouts, params)
    delta_f_around_bouts_ordered, length_bouts_ordered = behav_preproc.reorder_by_bout_size(delta_f_around_bouts,
                                                                                            length_behavioral_bouts,
                                                                                            rising=False)
    params["peri_event"]["graph_display_pre"] = 15
    params["peri_event"]["graph_display_post"] = 15
    perievent_metadata = plot.peri_event_plot(delta_f_around_bouts_ordered,
                                              length_bouts_ordered,
                                              new_sr,
                                              params,
                                              cmap="inferno",  # cividis, viridis, inferno
                                              style="individual",  # individual, average
                                              individual_colors=False,
                                              )

    photometry_io.save_perievent_data_to_pickle(perievent_metadata, experiment_tag, mouse, params)

plot.peri_event_bar_plot(delta_f_around_bouts_ordered, new_sr, params, duration_pre=3,
                         duration_post=3)

# [OPTIONAL] Creates a video with aligned photometry and behavior
video_clip = mpy.VideoFileClip(video_file).subclip(t_start=params["video_start"], t_end=params["video_end"])
plt.imshow(video_clip.get_frame(0))
plt.show()

cage_coordinates = Coordinates(x1=150, y1=120, x2=650, y2=270)
video_clip_cropped = video_clip.crop(x1=cage_coordinates.x1, y1=cage_coordinates.y1,
                                     x2=cage_coordinates.x2, y2=cage_coordinates.y2)
plt.imshow(video_clip_cropped.get_frame(0))
plt.show()

video_clip_trimmed = video_clip_cropped.subclip(t_start=params["crop_start"],
                                                t_end=params["video_duration"] - params["crop_end"])
y_data = [filtered_bool_map[:len(params["photometry_data"]["dFF"]["dFF"])],
          params["photometry_data"]["dFF"]["dFF"]]
params["video_photometry"]["resize_video"] = 3.
params["video_photometry"]["display_threshold"] = int(20 * params["video_sampling_rate"])
v_plot.live_video_plot(video_clip_trimmed, params["photometry_data"]["dFF"]["x"], y_data, **params)
