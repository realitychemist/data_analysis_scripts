from data_analysis_scripts.utils import tk_popover
import hyperspy.api as hs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

#%% Load & preview
infile = hs.load(tk_popover(), lazy=False)
for i, f in enumerate(infile):
    if "HAADF" in f.metadata.General.title:
        haadf_idx = i

# Throws an error if infile has no HAADF image
# noinspection PyUnboundLocalVariable
infile[haadf_idx].plot()  # Make sure title is "HAADF Signal"

#%%
haadf = infile[haadf_idx].data


