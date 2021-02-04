#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:16:29 2020

@author: thomas.topilko
"""

import errno
import os
import sys

from fiber_photometry_analysis.utilities import set_file_paths

def main(working_directory):
    # FIXME: use set_file_paths
    video_file_path = os.path.join(working_directory, "video.ext")
    video_metadata_file_path = os.path.join(working_directory, "video_metadata.ext")
    segmentation_file_path = os.path.join(working_directory, "segmentation.ext")
    photometry_file_path = os.path.join(working_directory, "photometry.ext")

    if os.path.exists(video_file_path):
        video=True
    else:
        video=False

    if os.path.exists(video_metadata_file_path):
        video_duration = load_video_metadata(video_metadata_file_path)
    else:
        if video:
            video_duration = get_video_duration(video_file_path)
            save_video_metadata(video_duration)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), video_metadata_file_path)

    segmentation = load_segmentation_data(segmentation_file_path)

    photometry = load_photometry_data(photometry_file_path)
    photometry_duration = get_photometry_duration(photometry)

    segmentation = adjust_segmentation_to_photometry(segmentation, video_duration, photometry_duration)

    photometry_smoothed = smooth_photometry_data(photometry)
    photometry_corrected = baseline_correction_photometry_data(photometry_smoothed)
    photometry_cropped = crop_data(photometry_corrected)

    # segmentation_cropped = crop_data(segmentation)

    photometry_standardized = standardize_photometry_data(photometry_cropped)
    photometry_aligned = align_photometry_data(photometry_standardized)
    photometry_delta_f = compute_delta_f_photometry_data(photometry_aligned)

    segmentation_cropped = crop_data(segmentation)
    segmentation_merged_close_events = merge_close_events(segmentation_cropped)
    segmentation_filtered_large_events = filter_large_events(segmentation_merged_close_events)

    perievent_data = extract_perievent_data(photometry_delta_f, segmentation_filtered_large_events)

    perievent_plot()


if __name__ == '__main__':
    main(sys.argv[1])
