import os

import numpy as np

folder_tag = "210413_359_Opto_5mW_30s"
experiment_tag = "210413"
tags = np.arange(0, 6, 1)
mice = ["359_{}".format(str(i).zfill(4)) for i in tags]
base_directory = "/Users/tomtop/Desktop/Photometry"

for m in mice:
    pass