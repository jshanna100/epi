import mne
from os.path import isdir
import numpy as np
from anoar import BadChannelFind

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

proc_dir = root_dir+"/proc/"

subjs = ["1001"]
conds = ["Stim"]

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw("{}caof_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         preload=True)
        picks = mne.pick_types(raw.info, eeg=True)
        bcf = BadChannelFind(picks, thresh=0.5)
        bad_chans = bcf.recommend(raw)
        print(bad_chans)
        raw.info["bads"].extend(bad_chans)
        raw.save("{}bcaof_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                 overwrite=True)
