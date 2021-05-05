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

folder_tag = "210414_353_Opto_5mW_30s_Cntrl"
experiment_tag = "210414"
# mice = ["130_{}".format(str(i).zfill(4)) for i in np.arange(0, 11, 1)]
# mice = ["164_0029"]
# tags = [2, 4, 19, 20, 21, 23, 25, 27, 28, 30, 31]
# mice = ["193_{}".format(str(i).zfill(4)) for i in tags]
tags = [0, 1, 2, 3, 4, 5]
mice = ["353_{}".format(str(i).zfill(4)) for i in tags]

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

folder_tag = "210414_353_Opto_5mW_30s_Cntrl"
experiment_tag = "210414"
# mice = ["130_{}".format(str(i).zfill(4)) for i in np.arange(0, 11, 1)]
# mice = ["164_0029"]
# tags = [2, 4, 19, 20, 21, 23, 25, 27, 28, 30, 31]
# mice = ["193_{}".format(str(i).zfill(4)) for i in tags]
tags = [0, 1, 2, 3, 4, 5]
mice = ["353_{}".format(str(i).zfill(4)) for i in tags]

# base_directory = "/Users/tomtop/Desktop/Photometry"
# working_directory = os.path.join(base_directory, "{}".format(folder_tag))
# utils.replace_photometry_files_into_folders(working_directory)
# utils.rename_video_names(working_directory)

base_directory = "/Users/tomtop/Desktop/Photometry"

for m in mice:
    mouse = m
    working_directory = os.path.join(base_directory, "{}/{}".format(folder_tag, mouse))
    ff_config = parameters.load_config(working_directory, experiment_tag, mouse, overwrite=True)

    photometry_data_test = photometry_io.save_dataframe_to_feather(ff_config["general"]["photometry_file"], working_directory)
    photometry_data = preproc.load_photometry_data(photometry_data_test, ff_config, recompute=True)
    ff_config["photometry_data"] = photometry_data  # Adds a new parameter containing all the photometry data

    for bts, bt, ib in zip(ff_config["general"]["general"]["behavior_to_segment"],
                           ff_config["general"]["general"]["behavior_tag"],
                           ff_config["general"]["general"]["invert_behavior"],):
        print("Processing behavior: {} for animal {}".format(bts, m))

        if ib:
            ff_config["general"]["behavior_saving_directory"] = os.path.join(ff_config["general"]["plot_directory"],
                                                                             "Inverted_{}_pmd{}_mbl{}"
                                                                             .format(bts,
                                                                                     ff_config["behavior_pp"]["merge_close_peaks"]["peak_merging_distance"],
                                                                                     ff_config["behavior_pp"]["filter_small_events"]["minimal_bout_length"]))
        else:
            ff_config["general"]["behavior_saving_directory"] = os.path.join(ff_config["general"]["plot_directory"],
                                                                             "{}_pmd{}_mbl{}"
                                                                             .format(bts,
                                                                                     ff_config["behavior_pp"]["merge_close_peaks"]["peak_merging_distance"],
                                                                                     ff_config["behavior_pp"]["filter_small_events"]["minimal_bout_length"]))
        if not os.path.exists(ff_config["general"]["behavior_saving_directory"]):
            os.mkdir(ff_config["general"]["behavior_saving_directory"])
        behavior_bouts = behav_preproc.extract_behavior_data(ff_config["general"]["behavior_manual"], bts)

        for beh in list(behavior_bouts.keys()):
            if behavior_bouts[beh].size > 0:

                ff_config["signal_pp"]["general"]["resolution_data"] = ff_config["general"]["recording_sampling_rate"]
                # params["resolution_data"] = behav_preproc.estimate_behavior_time_resolution(starts, ends)
                n_points = behav_preproc.estimate_length_bool_map(ff_config["general"]["recording_duration"],
                                                                  ff_config["signal_pp"]["general"]["resolution_data"])
                behavior_bouts_matched = behav_preproc.match_time_behavior_to_photometry(behavior_bouts,
                                                                                         ff_config["general"]["recording_duration"],
                                                                                         ff_config["general"]["video_duration"],
                                                                                         ff_config["signal_pp"]["general"]["resolution_data"])
                bool_map = set_ranges_high(np.zeros(n_points), behavior_bouts_matched)
                trimmed_bool_map = behav_preproc.trim_behavioral_data(bool_map,
                                                                      ff_config["signal_pp"]["general"]["crop_start"],
                                                                      ff_config["signal_pp"]["general"]["crop_end"],
                                                                      ff_config["signal_pp"]["general"]["resolution_data"])

                # INVERSION TO LOOK FOR SMALL WAKE PERIODS
                if ib:
                    trimmed_bool_map = 1 - trimmed_bool_map

                # filtered_bool_map = mask_op.filter_large_events(trimmed_bool_map, 100*241)
                filtered_bool_map = trimmed_bool_map

                plot.check_delta_f_with_behavior(filtered_bool_map, ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_raw",
                                                 extra_legend_label=None,
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["general"]["plot"],
                                                 save=ff_config["behavior_pp"]["general"]["save"])

                merged_bool_map = mask_op.merge_neighboring_events(filtered_bool_map,
                                                                   ff_config["behavior_pp"]["merge_close_peaks"]["peak_merging_distance"]*
                                                                    ff_config["signal_pp"]["general"]["resolution_data"])

                plot.check_delta_f_with_behavior([filtered_bool_map, merged_bool_map], ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_merged",
                                                 zorder=[0, 1],
                                                 extra_legend_label="Merged",
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["merge_close_peaks"]["plot"],
                                                 save=ff_config["behavior_pp"]["merge_close_peaks"]["save"])

                large_events_bool_map = mask_op.filter_small_events(merged_bool_map,
                                                                    ff_config["behavior_pp"]["filter_small_events"]["minimal_bout_length"]*
                                                                    ff_config["signal_pp"]["general"]["resolution_data"])

                plot.check_delta_f_with_behavior([large_events_bool_map, merged_bool_map], ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_large_events",
                                                 zorder=[0, 1], extra_legend_label="Small bouts",
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["filter_small_events"]["plot"],
                                                 save=ff_config["behavior_pp"]["filter_small_events"]["save"])

                edge_filtered_bool_map = mask_op.filter_edge_events(large_events_bool_map, ff_config["signal_pp"]["general"]["resolution_data"],
                                                                    ff_config["plotting"]["peri_event"]["extract_time_before_event"],
                                                                    ff_config["plotting"]["peri_event"]["extract_time_after_event"])

                plot.check_delta_f_with_behavior([edge_filtered_bool_map, large_events_bool_map], ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_filtered_edge_events",
                                                 zorder=[0, 1], extra_legend_label="Filtered edge events",
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["general"]["plot"],
                                                 save=ff_config["behavior_pp"]["general"]["save"])

                trimmed_duration = len(trimmed_bool_map) / ff_config["signal_pp"]["general"]["resolution_data"]
                new_sr = gen_preproc.round_to_closest_ten(ff_config["general"]["recording_sampling_rate"])
                # new_sr = 241
                new_x = gen_preproc.generate_new_x(new_sr, trimmed_duration)
                interpolated_delta_f = gen_preproc.interpolate_signal(ff_config["photometry_data"]["dFF"]["x"],
                                                                      ff_config["photometry_data"]["dFF"]["dFF"],
                                                                      new_x)

                filtered_bool_map_new = edge_filtered_bool_map[:len(interpolated_delta_f)]

                start_behavioral_bouts = mask_op.find_start_points_events(edge_filtered_bool_map)
                length_behavioral_bouts = mask_op.get_length_events(edge_filtered_bool_map, ff_config["signal_pp"]["general"]["resolution_data"])

                delta_f_around_bouts = behav_preproc.extract_peri_event_photometry_data(interpolated_delta_f, new_sr,
                                                                                        start_behavioral_bouts, ff_config)
                delta_f_around_bouts_ordered, length_bouts_ordered = behav_preproc.reorder_by_bout_size(delta_f_around_bouts,
                                                                                                        length_behavioral_bouts,
                                                                                                        rising=False)

                if not delta_f_around_bouts_ordered.size == 0:
                    perievent_metadata = plot.peri_event_plot(delta_f_around_bouts_ordered,
                                                              length_bouts_ordered,
                                                              new_sr,
                                                              ff_config,
                                                              behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                              cmap="inferno",  # cividis, viridis, inferno
                                                              style="average",  # individual, average
                                                              individual_colors=False,
                                                              )

                    photometry_io.save_perievent_data_to_pickle(perievent_metadata, experiment_tag, mouse, ff_config)

            else:
                print("Bypassing behavior: {} for animal {}. Reason: Empty".format(bts, m))

                
# base_directory = "/Users/tomtop/Desktop/Photometry"
# working_directory = os.path.join(base_directory, "{}".format(folder_tag))
# utils.replace_photometry_files_into_folders(working_directory)
# utils.rename_video_names(working_directory)

base_directory = "/Users/tomtop/Desktop/Photometry"

for m in mice:
    mouse = m
    working_directory = os.path.join(base_directory, "{}/{}".format(folder_tag, mouse))
    ff_config = parameters.load_config(working_directory, experiment_tag, mouse, overwrite=True)

    photometry_data_test = photometry_io.save_dataframe_to_feather(ff_config["general"]["photometry_file"], working_directory)
    photometry_data = preproc.load_photometry_data(photometry_data_test, ff_config, recompute=True)
    ff_config["photometry_data"] = photometry_data  # Adds a new parameter containing all the photometry data

    for bts, bt, ib in zip(ff_config["general"]["general"]["behavior_to_segment"],
                           ff_config["general"]["general"]["behavior_tag"],
                           ff_config["general"]["general"]["invert_behavior"],):
        print("Processing behavior: {} for animal {}".format(bts, m))

        if ib:
            ff_config["general"]["behavior_saving_directory"] = os.path.join(ff_config["general"]["plot_directory"],
                                                                             "Inverted_{}_pmd{}_mbl{}"
                                                                             .format(bts,
                                                                                     ff_config["behavior_pp"]["merge_close_peaks"]["peak_merging_distance"],
                                                                                     ff_config["behavior_pp"]["filter_small_events"]["minimal_bout_length"]))
        else:
            ff_config["general"]["behavior_saving_directory"] = os.path.join(ff_config["general"]["plot_directory"],
                                                                             "{}_pmd{}_mbl{}"
                                                                             .format(bts,
                                                                                     ff_config["behavior_pp"]["merge_close_peaks"]["peak_merging_distance"],
                                                                                     ff_config["behavior_pp"]["filter_small_events"]["minimal_bout_length"]))
        if not os.path.exists(ff_config["general"]["behavior_saving_directory"]):
            os.mkdir(ff_config["general"]["behavior_saving_directory"])
        behavior_bouts = behav_preproc.extract_behavior_data(ff_config["general"]["behavior_manual"], bts)

        for beh in list(behavior_bouts.keys()):
            if behavior_bouts[beh].size > 0:

                ff_config["signal_pp"]["general"]["resolution_data"] = ff_config["general"]["recording_sampling_rate"]
                # params["resolution_data"] = behav_preproc.estimate_behavior_time_resolution(starts, ends)
                n_points = behav_preproc.estimate_length_bool_map(ff_config["general"]["recording_duration"],
                                                                  ff_config["signal_pp"]["general"]["resolution_data"])
                behavior_bouts_matched = behav_preproc.match_time_behavior_to_photometry(behavior_bouts,
                                                                                         ff_config["general"]["recording_duration"],
                                                                                         ff_config["general"]["video_duration"],
                                                                                         ff_config["signal_pp"]["general"]["resolution_data"])
                bool_map = set_ranges_high(np.zeros(n_points), behavior_bouts_matched)
                trimmed_bool_map = behav_preproc.trim_behavioral_data(bool_map,
                                                                      ff_config["signal_pp"]["general"]["crop_start"],
                                                                      ff_config["signal_pp"]["general"]["crop_end"],
                                                                      ff_config["signal_pp"]["general"]["resolution_data"])

                # INVERSION TO LOOK FOR SMALL WAKE PERIODS
                if ib:
                    trimmed_bool_map = 1 - trimmed_bool_map

                # filtered_bool_map = mask_op.filter_large_events(trimmed_bool_map, 100*241)
                filtered_bool_map = trimmed_bool_map

                plot.check_delta_f_with_behavior(filtered_bool_map, ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_raw",
                                                 extra_legend_label=None,
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["general"]["plot"],
                                                 save=ff_config["behavior_pp"]["general"]["save"])

                merged_bool_map = mask_op.merge_neighboring_events(filtered_bool_map,
                                                                   ff_config["behavior_pp"]["merge_close_peaks"]["peak_merging_distance"]*
                                                                    ff_config["signal_pp"]["general"]["resolution_data"])

                plot.check_delta_f_with_behavior([filtered_bool_map, merged_bool_map], ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_merged",
                                                 zorder=[0, 1],
                                                 extra_legend_label="Merged",
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["merge_close_peaks"]["plot"],
                                                 save=ff_config["behavior_pp"]["merge_close_peaks"]["save"])

                large_events_bool_map = mask_op.filter_small_events(merged_bool_map,
                                                                    ff_config["behavior_pp"]["filter_small_events"]["minimal_bout_length"]*
                                                                    ff_config["signal_pp"]["general"]["resolution_data"])

                plot.check_delta_f_with_behavior([large_events_bool_map, merged_bool_map], ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_large_events",
                                                 zorder=[0, 1], extra_legend_label="Small bouts",
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["filter_small_events"]["plot"],
                                                 save=ff_config["behavior_pp"]["filter_small_events"]["save"])

                edge_filtered_bool_map = mask_op.filter_edge_events(large_events_bool_map, ff_config["signal_pp"]["general"]["resolution_data"],
                                                                    ff_config["plotting"]["peri_event"]["extract_time_before_event"],
                                                                    ff_config["plotting"]["peri_event"]["extract_time_after_event"])

                plot.check_delta_f_with_behavior([edge_filtered_bool_map, large_events_bool_map], ff_config,
                                                 behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                 fig_name="behavior_overlay_filtered_edge_events",
                                                 zorder=[0, 1], extra_legend_label="Filtered edge events",
                                                 video_time=True,
                                                 plot=ff_config["behavior_pp"]["general"]["plot"],
                                                 save=ff_config["behavior_pp"]["general"]["save"])

                trimmed_duration = len(trimmed_bool_map) / ff_config["signal_pp"]["general"]["resolution_data"]
                new_sr = gen_preproc.round_to_closest_ten(ff_config["general"]["recording_sampling_rate"])
                # new_sr = 241
                new_x = gen_preproc.generate_new_x(new_sr, trimmed_duration)
                interpolated_delta_f = gen_preproc.interpolate_signal(ff_config["photometry_data"]["dFF"]["x"],
                                                                      ff_config["photometry_data"]["dFF"]["dFF"],
                                                                      new_x)

                filtered_bool_map_new = edge_filtered_bool_map[:len(interpolated_delta_f)]

                start_behavioral_bouts = mask_op.find_start_points_events(edge_filtered_bool_map)
                length_behavioral_bouts = mask_op.get_length_events(edge_filtered_bool_map, ff_config["signal_pp"]["general"]["resolution_data"])

                delta_f_around_bouts = behav_preproc.extract_peri_event_photometry_data(interpolated_delta_f, new_sr,
                                                                                        start_behavioral_bouts, ff_config)
                delta_f_around_bouts_ordered, length_bouts_ordered = behav_preproc.reorder_by_bout_size(delta_f_around_bouts,
                                                                                                        length_behavioral_bouts,
                                                                                                        rising=False)

                if not delta_f_around_bouts_ordered.size == 0:
                    perievent_metadata = plot.peri_event_plot(delta_f_around_bouts_ordered,
                                                              length_bouts_ordered,
                                                              new_sr,
                                                              ff_config,
                                                              behavior_tag=ff_config["general"]["general"]["behavior_tag"],
                                                              cmap="inferno",  # cividis, viridis, inferno
                                                              style="average",  # individual, average
                                                              individual_colors=False,
                                                              )

                    photometry_io.save_perievent_data_to_pickle(perievent_metadata, experiment_tag, mouse, ff_config)

            else:
                print("Bypassing behavior: {} for animal {}. Reason: Empty".format(bts, m))

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
