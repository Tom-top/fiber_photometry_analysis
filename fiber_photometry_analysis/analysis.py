#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:30:37 2019

@author: thomas.topilko
"""

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import clips_array

mouse_tracker_dir = "/home/thomas.topilko/Documents/TopMouseTracker"
if not os.getcwd() == mouse_tracker_dir:
    os.chdir(mouse_tracker_dir)
    
from fiber_photometry_analysis import photo_funcs as photo_funcs
from fiber_photometry_analysis import utilities as utils

experiment = 201020
mouse = "301"

workDir = "/home/thomas.topilko/Desktop/Alba_Analysis/{0}/{1}".format(experiment, mouse)

photometryFileCSV = os.path.join(workDir, "{0}_{1}_0.csv".format(experiment, mouse))
videoFile = os.path.join(workDir, "RGB_mouse_301.avi")
#cottonFile = os.path.join(workDir, "{0}/Data_{0}_CottonPixelIntensities.npy".format(mouse))
manualFile = os.path.join(workDir, "{0}_{1}_manual.xlsx".format(experiment, mouse))

#%%
###############################################################################
#Reading the Photometry data
###############################################################################
print("\n")
print("Reading Photometry data")

""" File structure :
    -----------------------------------
    | Time(s) | Dem Out-1 | Dem Out-2 |
    -----------------------------------
"""

DF = 50.  # Decimation factor for data downsampling
SRD = 12000./DF  # Sampling rate of the Doric system
rollAvg = 100  # Rolling Average Parameter (s)
startVideo = 0 * 3600 + 0 * 60 + 0  # When to start the video
endVideo = 0 * 3600 + 6 * 60 + 53  # When to end the video
videoLength = 0 * 3600 + 6 * 60 + 53  # The length of the video ;; FOR TIME COMPRESSION

f = photometryFileCSV  # photometryFileCSV
photometry_npy_file_path = photo_funcs.convert_photometry_data(f)  # Convert CSV file to NPY if needed photometryFileCSV
photometry_npy_file_path = "/home/thomas.topilko/Desktop/Alba_Analysis/201020/301/201020-301_0.npy"

# fc.checkTimeShift(photometryFileNPY, SRD, plot=True, label="Decimation = 1 ; Video");

args = {
    "lw": 1.,  # linewidth in plot
    "fsl": 12.,  # fontsize for labels
    "fst": 16.,  # fontsize for titles
    "save": False,  # saving intermediate plots
    "function": "moving_avg",  # low_pass moving_avg None ---- To filter the dF signal
    "freq": 2,  # frequency for low pass filter
    "order": 10,  # order of the low pass filter
    "roll_size": 200,  # rolling average parameter
    "which": [False, False, False, False, False, True]  # Raw, Corrected, Standardized, Fit, Aligned, dF
}

rawX, rawIsosbestic, rawCalcium, dF = photo_funcs.load_photometry_data(photometry_npy_file_path, SRD,
                                                                       startVideo, endVideo, videoLength, rollAvg,
                                                                       opt_plots=True,
                                                                       recompute=True, plot_args=args)  # Loads the NPY data
                                                       
#%%
###############################################################################
#Reading the Video data
###############################################################################
print("\n")
print("Reading Video data")

videoClip = mpy.VideoFileClip(videoFile).subclip(t_start=startVideo, t_end=endVideo)
plt.imshow(videoClip.get_frame(0))

videoClipCropped = videoClip.crop(x1=77, y1=79, x2=807, y2=373)
plt.imshow(videoClipCropped.get_frame(0))

fps = videoClipCropped.fps

#%%
###############################################################################
#Reading the Cotton data from automatic segmentation
###############################################################################
print("\n")
print("Reading Cotton data")

rawCotton, Cotton = photo_funcs.load_tracking_data(cottonFile, fps, startVideo, endVideo)

"""Peak Detection"""
PAT = 0.7  # Peak Amplitude Threshold
PTPD = 1  # Peak to Peak Distance
PMD = 7.  # Peak Merging Distance
RS = 1./fps  # Raster Spread

peaks, posPeaks = photo_funcs.detect_raw_peaks(Cotton, PAT, PTPD)
peaksMerged, peaksPosMerged = photo_funcs.merge_close_peaks(peaks, posPeaks, PMD)

"""Major peak detection"""
DBE = 2  # Distance Between Events 2
BL = 0  # Behavioral Bout Length
GD = 40.  # Graph distance (distance before+after the event started)

mPeaks, mPosPeaks, mLengthPeaks, seeds, posSeeds = photo_funcs.detect_major_bouts(peaksMerged, DBE, BL, GD)

# fc.peak_plot(posPeaks, peaksPosMerged, mPosPeaks, posSeeds, multiPlot=True, timeFrame=[], save=False);

#%%
###############################################################################
#Reading the Cotton data from manual segmentation
###############################################################################

"""Peak Detection"""
PMD = 7.  # Peak Merging Distance

peaks, posPeaks = photo_funcs.extract_manual_bouts(manualFile, endVideo, "Touching")
half_time_lost = time_lost / 2
peaks_c = peaks[int(half_time_lost): -int(half_time_lost)]
posPeaks_c = np.array(posPeaks)+(int(half_time_lost) / sampling_rate)
# peaksMerged = fc.extract_manual_behavior(Cotton, manualFile)
peaksMerged, peaksPosMerged = photo_funcs.merge_close_peaks(peaks, posPeaks, PMD)

"""Major peak detection"""
DBE = 0  # Distance Between Events 2
BL = 0  # Behavioral Bout Length
GD = 10.  # Graph distance (distance before+after the event started)

# mPeaks, mPosPeaks, mLengthPeaks, seeds, posSeeds = fc.detect_major_bouts(peaksMerged, DBE, BL, GD)
mPeaks, mPosPeaks, mLengthPeaks, seeds, posSeeds = photo_funcs.detect_major_bouts(peaks_c, DBE, BL, GD)

fig = plt.figure()
ax0 = plt.subplot(2, 1, 1)
ax0.plot(x2, dF, color="blue")
ax1 = plt.subplot(2, 1, 2)
ax1.plot(peaks_c)
ax1.plot(mPeaks)

photo_funcs.peak_plot(posPeaks, peaksPosMerged, mPosPeaks, posSeeds, multi_plot=True, time_frame=[], save=False)

#%%
###############################################################################
#Reading the Cotton data from manual segmentation
###############################################################################

"""Extracting Calcium data from the major peak detection zone"""

# fc.save_peri_event_data(dataAroundPeaks, mLengthPeaks, mPosPeaks, workDir, mouse);

boutThresh = 0  # Behavioral Bout Length Threshold for display

# dataAroundPeaks = fc.extract_calcium_data_when_behaving(mPosPeaks, dF, GD, SRD, resample=1)
dataAroundPeaks = photo_funcs.extract_calcium_data_when_behaving(mPosPeaks, dF, GD, SRD, resample=1)
dataAroundPeaksFiltered, mLengthPeaksFiltered, mPosPeaksFiltered = photo_funcs.filter_short_bouts(dataAroundPeaks, mLengthPeaks, mPosPeaks, boutThresh)
dataAroundPeaksOrdered, mLengthPeaksOrdered, mPosPeaksOrdered = photo_funcs.reorder_by_bout_size(dataAroundPeaksFiltered, mLengthPeaksFiltered, mPosPeaksFiltered)
GDb, GDf = GD, GD

# cmap = gnuplot, twilight, inferno, viridis

fig = plt.figure(figsize=(15, 3), dpi=200.)
ax0 = plt.subplot(111)
ax0.plot(dF)
Min = min(dF)
Max = max(dF)
for i, j in zip(np.array(mPosPeaks)*SRD, np.array(mLengthPeaks)*SRD) :
    patch = patches.Rectangle((i, Min), j, (Max+Max*0.1)-(Min), alpha=0.1, color="red")
    ax0.add_patch(patch)
    ax0.plot([i, i], [Min+Min*0.1, Max+Max*0.1], color="red")
ax0.set_ylim(Min+Min*0.1, Max+Max*0.1)
# plt.savefig(os.path.join("/home/thomas.topilko/Desktop", "Photometry_With_Peaks_Walking.png"), dpi=300.)
# plt.savefig(os.path.join("/home/thomas.topilko/Desktop", "Photometry_With_Peaks_Walking.svg"), dpi=300.)

photo_funcs.peri_event_plot(dataAroundPeaksOrdered, mLengthPeaksOrdered, SRD, SRD, GD, GDf, GDb, SRD, save=False,
                            show_std=True, file_name_label="_All", cmap="inferno", lowpass=(0, 0))

#%%


def live_photometry_track(vidClip, x, yData, thresh, globalAcceleration=1,
                          plotAcceleration=SRD, showHeight=True, showTrace=True):
    
    n_subplots = len([x for x in yData if x != []])
    print(n_subplots)
    
    def make_frame_mpl(t):
        i = int(t)
        if i < thresh*plotAcceleration:
            try:
                cotton_graph.set_data(x[0:i], yData[0][0:i])
                # cotton_graph.fill_between(x[0:i], 0, y0[0:i], color="blue", alpha=0.8)
                # eventGraph.set_data(x[0:i], y1)
                # calciumGraph.set_data(x[0:i], y2[0:i])
                df_graph.set_data(x[0:i], yData[3][0:i])
            except:
                print("Oups a problem occured")
    
            last_frame = mplfig_to_npimage(live_fig)
            return last_frame
        else:
            delta = (i/plotAcceleration) - thresh
        
            liveAx0.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration))
            # liveAx1.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration))
            # liveAx2.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration))
            live_ax3.set_xlim(x[0]+delta, x[0]+(i/plotAcceleration))
            try:
                cotton_graph.set_data(x[0:i], yData[0][0:i])
                # cotton_graph.fill_between(x[0:i], 0, y0[0:i], color="blue", alpha=0.8)
                # eventGraph.set_data(x[0:i], y1[0:i])
                # calciumGraph.set_data(x[0:i], y2[0:i])
                df_graph.set_data(x[0:i], yData[3][0:i])
            except:
                print("Oups a problem occured")
    
            last_frame = mplfig_to_npimage(live_fig)
            return last_frame
    
    _FrameRate = vidClip.fps
    _Duration = vidClip.duration
    
    live_fig = plt.figure(figsize=(10,6), facecolor='white')
    
    gs = live_fig.add_gridspec(nrows=n_subplots, ncols=1)
    
    # First live axis
    liveAx0 = live_fig.add_subplot(gs[0, :])
    
    liveAx0.set_title("Touching object")
    liveAx0.set_xlim([x[0], x[0]+thresh])
    liveAx0.set_ylim([min(yData[0])-(min(yData[0])*0.05), max(yData[0])+(max(yData[0])*0.05)])
    
    cotton_graph, = liveAx0.plot(x[0], yData[0][0], '-', color="blue", alpha=0.8, ms=1.)
    
    #    # Second live axis
    #    liveAx1 = live_fig.add_subplot(gs[1, :]);
    #    
    #    liveAx1.set_title("Raster plot");
    #    liveAx1.set_xlim([x[0], x[0]+thresh]);
    #    liveAx1.set_ylim([-1, 1]);
    #    
    #    eventGraph, = liveAx1.eventplot(y1,colors="blue",alpha=0.8);
    #    
    # Third live axis
    #    liveAx2 = live_fig.add_subplot(gs[0, :]);
    #    
    #    liveAx2.set_title("Calcium dependant trace (465nm) (mV)");
    #    liveAx2.set_xlim([x[0], x[0]+thresh]);
    #    liveAx2.set_ylim([min(y2)-(min(y2)*0.1), max(y2)+(max(y2)*0.1)]);
    #    
    #    calciumGraph, = liveAx2.plot(x[0], y2[0], '-', color="cyan", alpha=0.8, ms=1., lw=0.5);
    
    # Fourth live axis
    live_ax3 = live_fig.add_subplot(gs[1, :])
    
    live_ax3.set_title("dF/F")
    live_ax3.set_xlim([x[0], x[0]+thresh])
    live_ax3.set_ylim([min(yData[3]) - (min(yData[3]) * 0.05), max(yData[3]) + (max(yData[3]) * 0.05)])
    
    df_graph, = live_ax3.plot(x[0], yData[3][0], '-', color="green", alpha=0.8, ms=1., lw=0.5)
    
    plt.tight_layout()
    
    anim = mpy.VideoClip(make_frame_mpl, duration=(_Duration*plotAcceleration))
    
    _clips = [clip.margin(2, color=[255, 255, 255]) for clip in [(vidClip.resize(2.).speedx(globalAcceleration)),
                                                                 anim.speedx(globalAcceleration).speedx(plotAcceleration)]]
    
    final_clip = clips_array([[_clips[0]],
                             [_clips[1]]],
                             bg_color=[255, 255, 255])
    
    final_clip.write_videofile(os.path.join("/home/thomas.topilko/Desktop", 'PhotoMetry_Tracking_test.mp4'), fps=10)
    

# raw_x_zeroed = rawX - rawX[0]
newCotton = [np.full(int(SRD), i) for i in peaks_c]
newCotton = [x for sublist in newCotton for x in sublist]
# new_cotton.append(newCotton[-1])
newCotton = np.array(newCotton)
newCotton = newCotton[:len(x2)]

yData = [newCotton, [], [], dF]

# for y in yData:
#    print(len(y))

video_clip_cropped_in_time = videoClipCropped.subclip(t_start=int(half_time_lost), t_end=videoClipCropped.duration - int(half_time_lost))

print(utils.h_m_s(videoClipCropped.duration))
print(utils.h_m_s(len(x2)/sampling_rate))
print(utils.h_m_s(len(yData[0])/sampling_rate))
print(utils.h_m_s(len(yData[3])/sampling_rate))
print(utils.h_m_s(video_clip_cropped_in_time.duration))

displayThreshold = int(5*videoClipCropped.fps)  # How much of the graph will be displayed in live (x scale)

live_photometry_track(video_clip_cropped_in_time, x2, yData, displayThreshold,
                      globalAcceleration=30, plotAcceleration=SRD);

#%%
###############################################################################
#Main Peri Event plot
###############################################################################
peFolder = "/raid/Tomek/Photometry/peData"

boutThresh = 10
GDb, GDf = 5, 40

peData = []
leData = []
posData = []

for f in os.listdir(peFolder):
    temp = np.load(os.path.join(peFolder, f))
    peData.append(temp[0])
    leData.append(temp[1])
    posData.append(temp[2])
    
peData = [x for sublist in peData for x in sublist]
leData = [x for sublist in leData for x in sublist]
posData = [x for sublist in posData for x in sublist]

peData, leData, posData = photo_funcs.filter_short_bouts(peData, leData, posData, boutThresh)
peData, leData, posData = photo_funcs.reorder_by_bout_size(peData, leData, posData)
    
photo_funcs.peri_event_plot(peData, leData, SRD, SRD, GD, GDf, GDb, SRD, save=False,
                            show_std=True, file_name_label="_All", cmap="inferno",
                            lowpass=(2, 2), norm=True)
