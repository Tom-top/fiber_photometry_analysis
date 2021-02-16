import numpy as np


def find_range_starts(src_mask):
    """
    :param np.array src_mask: The boolean mask to be extract the rising edges from
    :return:

    For a binary mask of the form:
    (0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1)
    returns:
    (0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1)
    """
    tmp_mask = np.logical_and(src_mask[1:], np.diff(src_mask))
    output_mask = np.hstack(([src_mask[0]], tmp_mask))  # reintroduce first element
    return output_mask


def find_range_ends(src_mask):
    """
    :param np.array src_mask: The boolean mask to be extract the falling edges from
    :return:

    For a binary mask of the form:
    (0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1)
    returns:
    (0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1)
    """
    inverted_mask = 1 - src_mask
    tmp_mask = np.logical_and(inverted_mask[1:], np.diff(inverted_mask))
    output_mask = np.hstack(([inverted_mask[0]], tmp_mask))  # reintroduce first element
    return output_mask


def find_event_starts(mask):
    """
    Find the index of the event starts (i.e. rising edges after running find_range_starts)

    :param np.array mask: A 1D boolean array where events are True
    :return:
    """
    event_starts = find_range_starts(mask)
    return np.where(event_starts)[0]


def find_event_ends(mask):
    """
    Find the index of the event starts (i.e. rising edges after running find_range_ends)

    :param np.array mask: A 1D boolean array where events are True
    :return:
    """
    event_ends = find_range_ends(mask)
    return np.where(event_ends)[0]


def find_events_boundaries(mask):
    """
    Convenient combination of find_event_starts and find_event_ends
    Tha returns a 2D array where the 1st dimension is start/end and the second the different events

    :param np.array mask: A 1D boolean array where events are True
    :return:
    """
    return find_event_starts(mask), find_event_ends(mask)


def get_down_times_bool_map(bool_map):
    zeros_pos = np.concatenate(([0], np.equal(bool_map, 0).view(np.int8), [0]))
    absolute_diff = np.abs(np.diff(zeros_pos))
    downtimes = np.where(absolute_diff == 1)[0].reshape(-1, 2)
    return downtimes


def get_up_times_bool_map(bool_map):
    ones_pos = np.concatenate(([0], np.equal(bool_map, 1).view(np.int8), [0]))
    absolute_diff = np.abs(np.diff(ones_pos))
    uptimes = np.where(absolute_diff == 1)[0].reshape(-1, 2)
    print(uptimes)
    return uptimes


def merge_neighboring_events(bool_map, merging_gap):
    """
    Algorithm that merges behavioral bouts that are close together.

    :param np.array position_bouts: list of start and end of each behavioral bout
    :param int max_bout_gap: Maximum number of points between 2 bouts unless they get fused
    :param int total_length: The size of the source array
    :return: position_bouts_merged (list) = list of merged start and end of each behavioral bout
             length_bouts_merged (list) = list of the length of each merged behavioral bout
    """
    downtimes = get_down_times_bool_map(bool_map)
    length_downtimes = downtimes[:, 1] - downtimes[:, 0]
    small_downtimes_mask = length_downtimes <= merging_gap
    small_downtimes_pos = np.where(small_downtimes_mask)[0]
    merged_bool_map = bool_map.copy()
    for s, e in downtimes[small_downtimes_pos]: # OPTIMISE: numpy
        merged_bool_map[s:e] = 1
    return merged_bool_map


def filter_small_events(bool_map, event_size):
    uptimes = get_up_times_bool_map(bool_map)
    length_uptimes = uptimes[:, 1] - uptimes[:, 0]
    small_uptimes_mask = length_uptimes <= event_size
    small_uptimes_pos = np.where(small_uptimes_mask)[0]
    filtered_bool_map = bool_map.copy()
    for s, e in uptimes[small_uptimes_pos]: # OPTIMISE: numpy
        filtered_bool_map[s:e] = 0
    return filtered_bool_map

def find_start_points_events(bool_map):
    ones_pos = np.concatenate(([0], np.equal(bool_map, 1).view(np.int8), [0]))
    absolute_diff = np.diff(ones_pos)
    start_points = np.where(absolute_diff == 1)[0]
    return start_points

def get_length_events(bool_map, res):
    uptimes = get_up_times_bool_map(bool_map)
    length_uptimes = uptimes[:, 1] - uptimes[:, 0]
    length_uptimes = length_uptimes/res
    return length_uptimes