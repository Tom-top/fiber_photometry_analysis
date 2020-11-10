#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:35:52 2020

@author: thomas.topilko
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import MultiCursor

import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoFileClip, VideoClip, clips_array,ImageSequenceClip

photometry_lib = "/home/thomas.topilko/Documents/Fiber_Photometry_Analysis"

if os.getcwd() != photometry_lib :
    os.chdir(photometry_lib)
    
import IO as io
import Utilities as ut
import Signal_Preprocessing as spp
import Behavior_Preprocessing as bpp
import Parameters as p
import Plot as plot
import Video_Plot as v_plot

#%% ===========================================================================
# Setting the working directory and the related files
# =============================================================================

experiment = "201109"
mouse = "112"

working_directory = os.path.join(photometry_lib, "Test/{0}/{1}".format(experiment, mouse))

files = ut.set_file_paths(working_directory, experiment, mouse)
photometry_file_csv, video_file, behavior_automatic_file, behavior_manual_file, saving_directory = files

#%% ===========================================================================
# Loading all the parameters
# =============================================================================

args = p.set_parameters(files, allow_downsampling=True) #Loads all the parameters for the analysis

#%% ===========================================================================
# Conversion of the photometry data into a numpy format
# =============================================================================

photometry_file_npy = io.convert_to_npy(photometry_file_csv, **args) #Convert CSV file to NPY if needed

#%% ===========================================================================
# Preprocessing of the photometry data
# =============================================================================

photometry_data = spp.load_photometry_data(photometry_file_npy, **args) #Preprocesses the photometry data at once
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

start, end = bpp.extract_behavior_data(behavior_manual_file, **args)
args["resolution_data"] = bpp.estimate_minimal_resolution(start, end)
start_trimmed, end_trimmed = bpp.trim_behavioral_data(start, end, **args) #Trims the behavioral data to fit the pre-processed photometry data
position_bouts, length_bouts = bpp.extract_manual_bouts(start_trimmed, end_trimmed, **args) #Extracts the raw behavioral events

plot.check_dF_with_behavior([position_bouts], [length_bouts], color="blue", name="dF_&_raw_behavior", **args)

#%% ===========================================================================
# Merge behavioral bouts that are close together
# =============================================================================

args["peak_merging_distance"] = 1
position_bouts_merged, length_bouts_merged = bpp.merge_neighboring_bouts(position_bouts, **args)

plot.check_dF_with_behavior([position_bouts_merged], [length_bouts_merged], color="blue", name="dF_&_merged_behavior", **args)

#%% ===========================================================================
# Detect major behavioral bouts based on size
# =============================================================================

args["minimal_bout_length"] = 2
position_major_bouts, length_major_bouts, position_seed_bouts, length_seed_bouts = bpp.detect_major_bouts(position_bouts_merged, **args) #Extracts major bouts from the data

plot.check_dF_with_behavior([position_major_bouts, position_seed_bouts], [length_major_bouts, length_seed_bouts], color=["blue", "red"], name="dF_&_filtered_behavior", **args)

#%% ===========================================================================
# Extract photometry data around major bouts of behavior
# =============================================================================

args["peri_event"]["graph_distance_pre"] = 2
args["peri_event"]["graph_distance_post"] = 10
dF_around_bouts = bpp.extract_peri_event_photmetry_data(position_major_bouts, **args)
dF_around_bouts_ordered, length_bouts_ordered = bpp.reorder_by_bout_size(dF_around_bouts, length_major_bouts)

args["peri_event"]["style"] = "individual"
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

behavior_data_x = bpp.create_bool_map(np.array(position_major_bouts).T[0], np.array(position_major_bouts).T[1], **args)
behavior_data_x_trimmed = behavior_data_x[0 : int(round((args["video_duration"]-int(args["photometry_data"]["time_lost"]))*args["recording_sampling_rate"]))]

y_data = [behavior_data_x_trimmed, photometry_data["dFF"]["dFF"]]

v_plot.live_video_plot(video_clip_trimmed, args["photometry_data"]["dFF"]["x"], y_data, **args)
















