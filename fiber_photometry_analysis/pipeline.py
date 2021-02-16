#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:35:52 2020

@author: thomas.topilko
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt

import moviepy.editor as mpy
import numpy as np

import fiber_photometry_analysis.mask_operations
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


@dataclass
class Coordinates:
    x1: int
    x2: int
    y1: int
    y2: int


# Setting the working directory and the related files
experiment = "yymmdd"
mouse = "test"
base_directory = "/Users/tomtop/Documents/Github/Fiber_Photometry_Analysis"
working_directory = os.path.join(base_directory, "{}/{}".format(experiment, mouse))

files = utils.set_file_paths(working_directory, experiment, mouse)
photometry_file_csv, video_file, behavior_automatic_file, behavior_manual_file, saving_directory = files

params = parameters.set_parameters(files, allow_downsampling=True)
cage_coordinates = Coordinates(x1=80, y1=62, x2=844, y2=402)
params["behavior_to_segment"] = "Behavior1"
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
params["resolution_data"] = behav_preproc.estimate_behavior_time_resolution(starts, ends)
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

plot.check_delta_f_with_behavior(trimmed_bool_map, name="dF_&_behavioral_overlay",
                                 extra_legend_label="Merged", **params)

merged_bool_map = mask_op.merge_neighboring_events(trimmed_bool_map, max_bout_gap)

plot.check_delta_f_with_behavior([trimmed_bool_map, merged_bool_map], zorder=[0, 1], name="dF_&_behavioral_overlay",
                                 extra_legend_label="Merged", **params)

filtered_bool_map = mask_op.filter_small_events(merged_bool_map, minimal_bout_length)

plot.check_delta_f_with_behavior([merged_bool_map, filtered_bool_map], zorder=[1, 0], name="dF_&_behavioral_overlay",
                                 extra_legend_label="Filtered out", **params)

new_sr = len(trimmed_bool_map)/params["resolution_data"]
new_x = gen_preproc.generate_new_x(params["recording_sampling_rate"], new_sr)
interpolated_delta_f = gen_preproc.interpolate_signal(params["photometry_data"]["dFF"]["x"],
                                                      params["photometry_data"]["dFF"]["dFF"],
                                                      new_x)

start_behavioral_bouts = mask_op.find_start_points_events(filtered_bool_map)
length_behavioral_bouts = mask_op.get_length_events(filtered_bool_map, params["resolution_data"])
delta_f_around_bouts = behav_preproc.extract_peri_event_photometry_data(interpolated_delta_f, new_sr,
                                                                        start_behavioral_bouts, params)
delta_f_around_bouts_ordered, length_bouts_ordered = behav_preproc.reorder_by_bout_size(delta_f_around_bouts,
                                                                                        length_behavioral_bouts,
                                                                                        rising=False)

plot.peri_event_plot(delta_f_around_bouts_ordered,
                     length_bouts_ordered,
                     new_sr,
                     cmap="inferno", # cividis, viridis, inferno
                     style="average", # individual, average
                     individual_colors=False,
                     **params)

plot.peri_event_bar_plot(delta_f_around_bouts_ordered, int(new_sr), duration_pre=3,
                         duration_post=3, **params)

# [OPTIONAL] Creates a video with aligned photometry and behavior
behavior_data_x = behav_preproc.create_bool_map(position_major_bouts,
                                                params["video_duration"],
                                                1 / params["recording_sampling_rate"])
utils.print_in_color("\nBehavioral data extracted. Behavior = {0}".format(params["behavior_to_segment"]), "GREEN")
i = int(round((params["video_duration"] - int(params["photometry_data"]["time_lost"])) * params["recording_sampling_rate"]))  # FIXME: simplify
behavior_data_x_trimmed = behavior_data_x[0: i]

y_data = [behavior_data_x_trimmed, photometry_data["dFF"]["dFF"]]

half_time_lost = int(params["photometry_data"]["time_lost"] / 2)  # TODO: make asymetric (split start and end vars)
video_clip = mpy.VideoFileClip(video_file).subclip(t_start=params["video_start"], t_end=params["video_end"])
plt.imshow(video_clip.get_frame(0))
video_clip_cropped = video_clip.crop(x1=cage_coordinates.x1, y1=cage_coordinates.y1,
                                     x2=cage_coordinates.x2, y2=cage_coordinates.y2)
plt.imshow(video_clip_cropped.get_frame(0))
video_clip_trimmed = video_clip_cropped.subclip(t_start=half_time_lost,
                                                t_end=params["video_duration"] - half_time_lost)
v_plot.live_video_plot(video_clip_trimmed, params["photometry_data"]["dFF"]["x"], y_data, **params)
