#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:35:52 2020

@author: thomas.topilko
"""

import os
import numpy as np

import matplotlib.pyplot as plt

import moviepy.editor as mpy
    
from fiber_photometry_analysis import utilities as utils
from fiber_photometry_analysis import signal_preprocessing as preproc
from fiber_photometry_analysis import behavior_preprocessing as behav_preproc
from fiber_photometry_analysis import parameters as params
from fiber_photometry_analysis import plot
from fiber_photometry_analysis import video_plot as v_plot


#%% ===========================================================================
# Setting the working directory and the related files
# =============================================================================

experiment = "201109"
mouse = "112"
analysis_folder = os.path.normpath('~/photometry_analysis')  # TODO: change

working_directory = os.path.join(analysis_folder, "Test/{0}/{1}".format(experiment, mouse))

files = utils.set_file_paths(working_directory, experiment, mouse)
photometry_file_csv, video_file, behavior_automatic_file, behavior_manual_file, saving_directory = files

#%% ===========================================================================
# Loading all the parameters
# =============================================================================

args = params.set_parameters(files, allow_downsampling=True) #Loads all the parameters for the analysis

#%% ===========================================================================
# Conversion of the photometry data into a numpy format
# =============================================================================

photometry_file_npy = IO.convert_to_npy(photometry_file_csv, **args) #Convert CSV file to NPY if needed

#%% ===========================================================================
# Preprocessing of the photometry data
# =============================================================================

photometry_data = preproc.load_photometry_data(photometry_file_npy, **args) #Preprocesses the photometry data at once
args["photometry_data"] = photometry_data #Adds a new parameter containing all the photometry data

#%% ===========================================================================
# Reading the Video data
# =============================================================================

video_clip = mpy.VideoFileClip(video_file).subclip(t_start=args["video_start"], t_end=args["video_end"]) #Loads the video
plt.imshow(video_clip.get_frame(0)) #[OPTIONAL] Displays the first frame of the video

#%% ===========================================================================
# Cropping the video field of view (for future display)
# =============================================================================

video_clip_cropped = video_clip.crop(x1=80, y1=62, x2=844, y2=402) #Crops the video file to the cage only
plt.imshow(video_clip_cropped.get_frame(0)) #[OPTIONAL] Displays the first frame of the cropped video

#%% ===========================================================================
# Importing behavioral data
# =============================================================================

args["behavior_to_segment"] = "Behavior1"

start, end = behav_preproc.extract_behavior_data(behavior_manual_file, **args)
args["resolution_data"] = behav_preproc.estimate_minimal_resolution(start, end)
start_trimmed, end_trimmed = behav_preproc.trim_behavioral_data(start, end, **args) #Trims the behavioral data to fit the pre-processed photometry data
position_bouts, length_bouts = behav_preproc.extract_manual_bouts(start_trimmed, end_trimmed, **args) #Extracts the raw behavioral events

plot.check_dF_with_behavior([position_bouts], [length_bouts], color="blue", name="dF_&_raw_behavior", **args)

#%% ===========================================================================
# Merge behavioral bouts that are close together
# =============================================================================

args["peak_merging_distance"] = 2
position_bouts_merged, length_bouts_merged = behav_preproc.merge_neighboring_bouts(position_bouts, **args)

plot.check_dF_with_behavior([position_bouts_merged], [length_bouts_merged], color="blue", name="dF_&_merged_behavior", **args)

#%% ===========================================================================
# Detect major behavioral bouts based on size
# =============================================================================

args["minimal_bout_length"] = 1
position_major_bouts, length_major_bouts, position_seed_bouts, length_seed_bouts = behav_preproc.detect_major_bouts(position_bouts_merged, **args) #Extracts major bouts from the data

plot.check_dF_with_behavior([position_major_bouts, position_seed_bouts], [length_major_bouts, length_seed_bouts], color=["blue", "red"], name="dF_&_filtered_behavior", **args)

#%% ===========================================================================
# Extract photometry data around major bouts of behavior
# =============================================================================

args["peri_event"]["graph_distance_pre"] = 10
args["peri_event"]["graph_distance_post"] = 10
dF_around_bouts = behav_preproc.extract_peri_event_photmetry_data(position_major_bouts, **args)
dF_around_bouts_ordered, length_bouts_ordered = behav_preproc.reorder_by_bout_size(dF_around_bouts, length_major_bouts)

args["peri_event"]["style"] = "average"
args["peri_event"]["individual_color"] = False
plot.peri_event_plot(dF_around_bouts_ordered, length_bouts_ordered, cmap="inferno", **args) #cividis, viridis

#%% ===========================================================================
# Generate bar plot for the peri-event data
# =============================================================================

args["peri_event"]["graph_auc_pre"] = 2
args["peri_event"]["graph_auc_post"] = 2
plot.peri_event_bar_plot(dF_around_bouts_ordered, **args)

#%% ===========================================================================
# Creates a video with aligned photometry and behavior
# =============================================================================

args["video_photometry"]["resize_video"] = 1.
args["video_photometry"]["global_acceleration"] = 20
video_clip_trimmed = video_clip_cropped.subclip(t_start=int(args["photometry_data"]["time_lost"]/2), t_end=args["video_duration"]-int(args["photometry_data"]["time_lost"]/2))

behavior_data_x = behav_preproc.create_bool_map(np.array(position_major_bouts).T[0], np.array(position_major_bouts).T[1], **args)
behavior_data_x_trimmed = behavior_data_x[0 : int(round((args["video_duration"]-int(args["photometry_data"]["time_lost"]))*args["recording_sampling_rate"]))]

y_data = [behavior_data_x_trimmed, photometry_data["dFF"]["dFF"]]

v_plot.live_video_plot(video_clip_trimmed, args["photometry_data"]["dFF"]["x"], y_data, **args)
