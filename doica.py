import mne
from os.path import isdir
import numpy as np
from mne.preprocessing import ICA

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
        raw = mne.io.Raw("{}bcaof_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         preload=True)
        ica = ICA(method="picard",max_iter=500)
        ica.fit(raw, picks="eeg")
        ica.save("{}bcof_EPI_{}_{}-ica.fif".format(proc_dir,subj,cond))
        eog_inds, scores = ica.find_bads_eog(raw)
        ecg_inds, scores = ica.find_bads_ecg(raw)
        bad_inds = eog_inds + ecg_inds
        new_raw = ica.apply(raw, exclude=bad_inds)
        new_raw.save("{}ibcaof_EPI_{}_{}-raw.fif".format(proc_dir,subj,cond),
                     overwrite=True)
