import mne
from os.path import isdir
import numpy as np
import re

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

proc_dir = root_dir+"/proc/"

subjs = ["1001", "1002"]
conds = ["Stim", "Sham"]

l_freq = 0.3
h_freq = 200
n_jobs = "cuda"

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw("{}EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         preload=True)
        raw.filter(l_freq=0.3, h_freq=200, n_jobs=n_jobs)
        raw.notch_filter(np.arange(50,h_freq,50), n_jobs=n_jobs)
        raw.save("{}f_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                 overwrite=True)
