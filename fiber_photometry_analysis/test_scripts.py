import os
from natsort import natsorted
import numpy as np
import pandas as pd
from fiber_photometry_analysis import parameters
from fiber_photometry_analysis import behavior_preprocessing as behav_preproc
from fiber_photometry_analysis import plot

def load_data(file_paths):
    data = []
    for f in file_paths:
        data.append(pd.read_pickle(f))
    return data

def flatten(list):
    return [x for sublist in list for x in sublist]

working_directory = "/Users/tomtop/Desktop/Photometry/210414_353_Opto_5mW_30s_Cntrl"
experiment_tag = "210414"
mouse = "353"

pmd = 1
mbl = 1
behaviors = ["Dig", "Groom", "InteractFood", "InteractNest", "Nesting", "Rest", "Inverted_Rest"]
behavior_tags = ["{}_pmd{}_mbl{}".format(beh, pmd, mbl) for beh in behaviors]

for behavior in behavior_tags:

    behavior_tag = behavior
    saving_directory = os.path.join(working_directory, "{}".format(behavior_tag))
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    files = []

    for d in natsorted(os.listdir(working_directory)):
        if d.startswith("{}".format(mouse)):
            path_to_directory = os.path.join(working_directory, d)
            if os.path.isdir(path_to_directory):
                pickle_file_path = os.path.join(path_to_directory,
                                          "results/plots/{}/perievent_data_{}_{}.pickle"
                                          .format(behavior_tag, experiment_tag, d))
                if os.path.exists(pickle_file_path):
                    files.append(os.path.join(pickle_file_path))
                else:
                    print("FILE MISSING: {}. Behavior: {}".format(os.path.basename(pickle_file_path), behavior))

    data = load_data(files)
    peri_event_data = flatten(np.array([i["data"] for i in data]))
    length = flatten([i["data_bout_length"] for i in data])

    delta_f_ordered, length_ordered = behav_preproc.reorder_by_bout_size(peri_event_data,
                                                                        length,
                                                                        rising=False)

    plot_args = {"general":{"behavior_saving_directory": saving_directory,
                            "general":{"behavior_tag":behavior_tag,
                                       },
                            },
                 "plotting":{"general":{"lw": 0.5,
                                        "fsl": 6.,
                                        "fst": 8.,},
                             "peri_event":{"extract_time_before_event":30,
                                           "extract_time_after_event":240,
                                           "time_before_event": 30,
                                           "time_after_event": 240,
                                           "style": "individual",
                                           "normalize_heatmap": False,
                                           "plot":False,
                                           "save": True,},
                             },
                 "signal_pp":{"standardization":{"standardize":False,
                                                 },
                              },
                 "saving":{"general":{"extension":"png",
                                      },
                           },
                 }

    perievent_metadata = plot.peri_event_plot(delta_f_ordered,
                                              length_ordered,
                                              240,
                                              plot_args,
                                              sem=True,
                                              cmap="inferno", # cividis, viridis, inferno
                                              style="average", # individual, average
                                              individual_colors=False,
                                              save_options={"bbox_inches":'tight'}
                                              )

plot.peri_event_bar_plot(delta_f_ordered, 240, plot_args, duration_pre=30,
                         duration_post=30, save_options={"bbox_inches":'tight'})

