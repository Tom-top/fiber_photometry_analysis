import os
from natsort import natsorted
import numpy as np
import pandas as pd
from fiber_photometry_analysis import behavior_preprocessing as behav_preproc
from fiber_photometry_analysis import plot

cre_folder_path = "207_pkl"

files = natsorted([os.path.join(cre_folder_path, x) for x in os.listdir(cre_folder_path) if x.split(".")[-1] == "pkl"])

def load_data(file_paths):
    data = []
    for f in file_paths:
        data.append(pd.read_pickle(f))
    return data

def flatten(list):
    return [x for sublist in list for x in sublist]

data = load_data(files)
peri_event_data = flatten(np.array([i["data"] for i in data]))
length = flatten([i["data_bout_length"] for i in data])

delta_f_ordered, length_ordered = behav_preproc.reorder_by_bout_size(peri_event_data,
                                                                    length,
                                                                    rising=False)

plot_args = {"lw": 0.5,
             "fsl": 6.,
             "fst": 8.,
             "behavior_tag": "nesting",
             "save": True,
             "save_dir": "/Users/tomtop/Desktop",
             "extension": "png",
             "photometry_pp":{"standardize" : True},
             "peri_event":{"graph_distance_pre": 10,
                           "graph_distance_post": 10,
                           "graph_display_pre": 5,
                           "graph_display_post": 10,
                           "style": "individual",
                           "normalize_heatmap": False,
                           },
             }

perievent_metadata = plot.peri_event_plot(delta_f_ordered,
                                          length_ordered,
                                          240,
                                          plot_args,
                                          cmap="inferno", # cividis, viridis, inferno
                                          style="average", # individual, average
                                          individual_colors=False,
                                          )

