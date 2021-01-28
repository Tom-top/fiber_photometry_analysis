#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:27:25 2020

@author: thomas.topilko
"""

import os

import numpy as np

import matplotlib.pyplot as plt

import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import clips_array

from fiber_photometry_analysis import utilities as utils


def live_video_plot(video_clip, x_data, y_data, **kwargs):
    
    def make_frame_mpl(t):
        i = int(t)
        if i < kwargs["video_photometry"]["display_threshold"]*kwargs["video_photometry"]["plot_acceleration"]:
            try:
                behavior_plot.set_data(x_data[0:i], y_data[0][0:i])
                df_plot.set_data(x_data[0:i], y_data[1][0:i])
            except :
                print("Oups a problem occured")
    
            last_frame = mplfig_to_npimage(live_figure)
            return last_frame
        else:
            delta = (i/kwargs["video_photometry"]["plot_acceleration"]) - kwargs["video_photometry"]["display_threshold"]
        
            live_ax0.set_xlim(x_data[0]+delta, x_data[0]+(i/kwargs["video_photometry"]["plot_acceleration"]))
            live_ax1.set_xlim(x_data[0]+delta, x_data[0]+(i/kwargs["video_photometry"]["plot_acceleration"]))
            
            try:
                behavior_plot.set_data(x_data[0:i], y_data[0][0:i])
                df_plot.set_data(x_data[0:i], y_data[1][0:i])
            except:
                print("Oups a problem occured")

            last_frame = mplfig_to_npimage(live_figure)
            return last_frame
    
    number_subplots = len([x for x in y_data if x != []])
    
    plt.style.use("dark_background")
    live_figure = plt.figure(figsize=(10, 6), facecolor='black')
    gs = live_figure.add_gridspec(nrows=number_subplots, ncols=1)
    
    # =============================================================================
    #     First live axis
    # =============================================================================
    live_ax0 = live_figure.add_subplot(gs[0, :])
    
    live_ax0.set_title("Behavior bouts : {0}".format(kwargs["behavior_to_segment"]), fontsize=10.)
    live_ax0.set_xlim([x_data[0], x_data[0]+kwargs["video_photometry"]["display_threshold"]])
    live_ax0.set_yticks([0, 1])
    live_ax0.set_yticklabels(["Not behaving", "Behaving"], fontsize=10.)
    live_ax0.set_ylim(-0.1, 1.1)
    
    behavior_plot, = live_ax0.plot(x_data[0], y_data[0][0], '-', color="red", alpha=0.8, ms=1., lw=3.)
    
    # =============================================================================
    #     Second live axis
    # =============================================================================
    live_ax1 = live_figure.add_subplot(gs[1, :])
    
    live_ax1.set_title(r"$\Delta$F/F", fontsize=kwargs["fst"])
    live_ax1.set_xlim([x_data[0], x_data[0]+kwargs["video_photometry"]["display_threshold"]])
    live_ax1.set_xlabel("Time (s)", fontsize=10.)
    y_min, y_max, round_factor = utils.generate_yticks(y_data[1], 0.1)
    live_ax1.set_yticks(np.arange(y_min, y_max+round_factor, round_factor))
    live_ax1.set_yticklabels(["{:.0f}".format(i) for i in np.arange(y_min, y_max+round_factor, round_factor)], fontsize=10.)
    live_ax1.set_ylim(y_min, y_max)
    
    df_plot, = live_ax1.plot(x_data[0], y_data[1][0], '-', color="green", alpha=1., ms=1., lw=3.)
    
    plt.tight_layout()
    
    animation = mpy.VideoClip(make_frame_mpl, duration=(kwargs["video_duration"] * kwargs["video_photometry"]["plot_acceleration"]))
    
    clips = [clip.margin(2, color=[0, 0, 0]) for clip in [(video_clip.resize(kwargs["video_photometry"]["resize_video"]).speedx(kwargs["video_photometry"]["global_acceleration"])),\
             animation.speedx(kwargs["video_photometry"]["global_acceleration"]).speedx(kwargs["video_photometry"]["plot_acceleration"])]]
    
    final_clip = clips_array([[clips[0]],
                             [clips[1]]],
                             bg_color=[0, 0, 0])
    
    final_clip.write_videofile(os.path.join(kwargs["save_dir"],"Live_Video_Plot.mp4"), fps=kwargs["video_photometry"]["live_plot_fps"])
    
    # final_clip = final_clip.subclip(t_start=0, t_end=final_clip.duration-5)
    # final_clip.write_gif(os.path.join(kwargs["save_dir"],"Live_Video_Plot.gif"), fps=kwargs["video_photometry"]["live_plot_fps"])
    plt.style.use("default")
