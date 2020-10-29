#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:41:56 2020

@author: thomas.topilko
"""

import numpy as np

import IO as io
import Utilities as ut

def set_parameters(files) :
    
    photometry_file_csv, video_file, behavior_automatic_file, behavior_manual_file, saving_directory = files #Assigning new variables for each file
    
    recording_duration, recording_sampling_rate = io.get_recording_duration_and_sampling_rate(photometry_file_csv) #Get metadata from the photometry file
    
    general_args = {"recording_sampling_rate" : recording_sampling_rate, #Sampling rate of the photometry system
                    "recording_duration" : recording_duration, #The time of recording according to the photometry dataset (s)
                    "smoothing_window" : int(recording_sampling_rate), #The window used to smooth the raw signal
                    "moving_average_window" : int(recording_sampling_rate)*60, #The window used to estimate the baseline of each signal
                   }
    
    if video_file != None :
        
        general_args["video"] = True #Bool. True if a video is to be analyzed, otherwise False
        video_duration, video_sampling_rate = io.get_video_duration_and_framerate(video_file) #Get metadata from the video file
        general_args["video_duration"] = video_duration #The duration of the video (s)
        general_args["video_start"] = 0 #The start of the video (s). Change this value if you would like to skip some time at the beginning 
        general_args["video_end"] = video_duration #The end of the video (s). Change this value if you would like to skip some time at the end 
        general_args["video_sampling_rate"] = video_sampling_rate #The framerate of the video
        
    else :
        
        general_args["video"] = False #If no video
        
    plot_args = {"photometry_pp" : {"plots_to_display" : np.array([True, #raw_data
                                                                   True, #smoothed
                                                                   True, #compute baseline
                                                                   True, #baseline correction
                                                                   True, #standardization
                                                                   True, #inter-channel regression
                                                                   True, #channel alignement
                                                                   True, #dFF
                                                                   ]),
                                    "regression" : "Lasso", #The algorithm to use for the inter-channel regression step
                                    "standardize" : True, #If True = Standardizes the signals, otherwise skips this step
                                    "multicursor" : True, #If True = Displays multicursor on plots, otherwise not (Doesn't work properly)
                                    "purple_laser" : "#8200c8", #Hex color of the 405nm laser trace
                                    "blue_laser" : "#0092ff", #Hex color of the 465nm laser trace
                                    },
                "peri_event" : {"normalize_heatmap" : False, #If True, normalizes the heatmap of the peri-event plot
                                "graph_distance_pre" : 10., #Graph distance (distance before the event started (s))
                                "graph_distance_post" : 10., #Graph distance (distance after the event started (s))
                                "resample_graph" : recording_sampling_rate,
                                "resample_heatmap" : recording_sampling_rate/100,
                                "style" : "individual",
                                "individual_color" : False,
                                },
                "video_photometry" : {"display_threshold" : int(5*general_args["video_sampling_rate"]), 
                                      "plot_acceleration" : general_args["recording_sampling_rate"],
                                      "global_acceleration" : 5,
                                      "resize_video" : 1.5,
                                      "live_plot_fps" : 10.,
                                      },
                "lw" : 1., #linewidth of in the plots
                "fsl" : 6., #fontsize of labels in the plot
                "fst" : 8., #fontsize of titles in the plot
                "save" : True,
                "save_dir" : saving_directory, #Direcotry were the plots will be saved
                "extension" : "svg", #Extension of the saved plots   
                }
    
    behavioral_segmentation_args = {"peak_merging_distance" : 7., #Peak Merging Distance
                                    "minimal_bout_length" : 0, #Minimal bout length (s)
                                    }
    
    args = {**general_args, **behavioral_segmentation_args, **plot_args}
    
    print("\n")
    
    ut.print_in_color("Parameters loaded successfully", "GREEN")
    
    print("\n")
    
    ut.print_in_color("If you like to change some of the parameters you can directly modify them in\n\
the 'Parameters.py' file or change them by calling : 'args['arg'] = new_value' with\n\
arg corresponding to the argument you desire to change, and new_value the new value\n\
of the argument in question", "GREEN")
    
    return args