import mne
from os.path import join
import numpy as np
from anoar import BadChannelFind

"""
Finds bad channels. Because of the enormous size of these files, the old ones
are simply overwritten, making it impossible to know if this has been run
or not from the filename alone. If you wish to change this, change the
raw.save parameters in the final line.
"""

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

subjs = ["1001", "1002"]
conds = ["Sham", "Stim"]

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw(join(proc_dir, f"f_EPI_{subj}_{cond}-raw.fif"),
                         preload=True)
        picks = mne.pick_types(raw.info, eeg=True)
        bcf = BadChannelFind(picks, thresh=0.5)
        bad_chans = bcf.recommend(raw)
        print(bad_chans)
        raw.info["bads"].extend(bad_chans)
        raw.save(join(proc_dir, f"f_EPI_{subj}_{cond}-raw.fif"),
                 overwrite=True)
