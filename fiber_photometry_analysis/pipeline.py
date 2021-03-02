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
experiment = "210121"
mice = ["207_0012", "207_0016", "207_0018", "207_0022", "207_0024", "207_0025", "207_0026", "207_0027"]
for m in mice:
    mouse = m
    base_directory = "/Users/tomtop/Desktop/Photometry_Analysis"
    working_directory = os.path.join(base_directory, "{}/{}".format(experiment, mouse))
    init_logging(working_directory, experiment_name=experiment, config=logger_config)  # TODO put in main cfg

    files = utils.set_file_paths(working_directory, experiment, mouse)
    photometry_file_csv, video_file, behavior_automatic_file, behavior_manual_file, saving_directory = files

    params = parameters.set_parameters(files, allow_downsampling=True)
    params["behavior_to_segment"] = ["Rest"] #["Nesting", "InteractNest"]
    params["behavior_tag"] = "Rest" #"Nesting"
    max_bout_gap = 4  # TODO: inline
    minimal_bout_length = 2  # TODO: inline
    params["peak_merging_distance"] = max_bout_gap  # TODO remove
    params["minimal_bout_length"] = minimal_bout_length
    params["peri_event"]["graph_distance_pre"] = 10
    params["peri_event"]["graph_distance_post"] = 10
    params["peri_event"]["style"] = "average"
    params["peri_event"]["individual_color"] = False
    params["peri_event"]["graph_auc_pre"] = 2
    params["peri_event"]["graph_auc_post"] = 2
    params["video_photometry"]["resize_video"] = 1.
    params["video_photometry"]["global_acceleration"] = 20

    # photometry_file_npy = io.convert_to_npy(photometry_file_csv, **params)
    photometry_data_test = photometry_io.save_dataframe_to_feather(photometry_file_csv, working_directory,
                                                                   "photometry_{}_{}".format(experiment, mouse))
    photometry_data = preproc.load_photometry_data(photometry_data_test, params)  # dict of different preprocessed data
    # FIXME: remove from metadata
    params["photometry_data"] = photometry_data  # Adds a new parameter containing all the photometry data


    starts, ends = behav_preproc.extract_behavior_data(behavior_manual_file, params["behavior_to_segment"])  # REFACTOR: replace by ranges
    # params["resolution_data"] = behav_preproc.estimate_behavior_time_resolution(starts, ends)
    params["resolution_data"] = params["recording_sampling_rate"]
    n_points = behav_preproc.estimate_length_bool_map(params["recording_duration"], params["resolution_data"])
    starts_matched, ends_matched = behav_preproc.match_time_behavior_to_photometry(starts, ends,
                                                                                   params["recording_duration"],
                                                                                   params["video_duration"],
                                                                                   params["resolution_data"])
    bool_map = set_ranges_high(np.zeros(n_points), np.column_stack((starts_matched, ends_matched)))
    trimmed_bool_map = behav_preproc.trim_behavioral_data(bool_map,
                                                          params["crop_start"],
                                                          params["crop_end"],
                                                          params["resolution_data"])

    #INVERSION TO LOOK FOR SMALL WAKE PERIODS
    inverted_bool_map = 1 - trimmed_bool_map
    filtered_bool_map = mask_op.filter_large_events(inverted_bool_map, 100*241)

    plot.check_delta_f_with_behavior(filtered_bool_map, params, extra_legend_label=None, video_time=True)

    merged_bool_map = mask_op.merge_neighboring_events(filtered_bool_map, max_bout_gap)

    plot.check_delta_f_with_behavior([filtered_bool_map, merged_bool_map], params, zorder=[0, 1],
                                     extra_legend_label="Merged")

    large_events_bool_map = mask_op.filter_small_events(merged_bool_map, minimal_bout_length)

    plot.check_delta_f_with_behavior([merged_bool_map, large_events_bool_map], params, zorder=[1, 0],
                                     extra_legend_label="Large bouts")

    params["peri_event"]["graph_distance_pre"] = 10
    params["peri_event"]["graph_distance_post"] = 10
    filtered_bool_map = mask_op.filter_edge_events(large_events_bool_map, params["resolution_data"],
                                                   params["peri_event"]["graph_distance_pre"],
                                                   params["peri_event"]["graph_distance_post"])

    plot.check_delta_f_with_behavior([large_events_bool_map, filtered_bool_map], params, zorder=[1, 0],
                                     extra_legend_label="Filtered edge events")

    trimmed_duration = len(trimmed_bool_map)/params["resolution_data"]
    new_sr = gen_preproc.round_to_closest_ten(params["recording_sampling_rate"])
    # new_sr = 241
    new_x = gen_preproc.generate_new_x(new_sr, trimmed_duration)
    interpolated_delta_f = gen_preproc.interpolate_signal(params["photometry_data"]["dFF"]["x"],
                                                          params["photometry_data"]["dFF"]["dFF"],
                                                          new_x)

    filtered_bool_map_new = filtered_bool_map[:len(interpolated_delta_f)]
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

    start_behavioral_bouts = mask_op.find_start_points_events(filtered_bool_map)
    length_behavioral_bouts = mask_op.get_length_events(filtered_bool_map, params["resolution_data"])

    delta_f_around_bouts = behav_preproc.extract_peri_event_photometry_data(interpolated_delta_f, new_sr,
                                                                            start_behavioral_bouts, params)
    delta_f_around_bouts_ordered, length_bouts_ordered = behav_preproc.reorder_by_bout_size(delta_f_around_bouts,
                                                                                            length_behavioral_bouts,
                                                                                            rising=False)
    params["peri_event"]["graph_display_pre"] = 2
    params["peri_event"]["graph_display_post"] = 10
    perievent_metadata = plot.peri_event_plot(delta_f_around_bouts_ordered,
                                              length_bouts_ordered,
                                              new_sr,
                                              params,
                                              cmap="inferno", # cividis, viridis, inferno
                                              style="individual", # individual, average
                                              individual_colors=False,
                                              )

    photometry_io.save_perievent_data_to_pickle(perievent_metadata, experiment, mouse, params)

plot.peri_event_bar_plot(delta_f_around_bouts_ordered, new_sr, params, duration_pre=3,
                         duration_post=3)


# [OPTIONAL] Creates a video with aligned photometry and behavior
video_clip = mpy.VideoFileClip(video_file).subclip(t_start=params["video_start"], t_end=params["video_end"])
plt.imshow(video_clip.get_frame(0))
plt.show()

cage_coordinates = Coordinates(x1=823, y1=433, x2=1307, y2=591)
video_clip_cropped = video_clip.crop(x1=cage_coordinates.x1, y1=cage_coordinates.y1,
                                     x2=cage_coordinates.x2, y2=cage_coordinates.y2)
plt.imshow(video_clip_cropped.get_frame(0))
plt.show()

video_clip_trimmed = video_clip_cropped.subclip(t_start=params["crop_start"],
                                                t_end=params["video_duration"] - params["crop_end"])
y_data = [filtered_bool_map[:len(params["photometry_data"]["dFF"]["dFF"])],
          params["photometry_data"]["dFF"]["dFF"]]
params["video_photometry"]["resize_video"] = 3.
v_plot.live_video_plot(video_clip_trimmed, params["photometry_data"]["dFF"]["x"], y_data, **params)
