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