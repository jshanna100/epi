import mne
from os.path import isdir
from os import listdir
import numpy as np
import re
from mne.preprocessing import ICA

def chan_sub(chans):
    chans[0,] = chans[0,] - chans[1,]
    return chans

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

proc_dir = root_dir+"proc/"
proclist = listdir(proc_dir)

l_freq = 0.3
h_freq = 200
n_jobs = "cuda"
max_thresh = 2e-3

for filename in proclist:
    this_match = re.match("c_EPI_(.*)-raw.fif", filename)
    if not this_match:
        continue
    file_id = this_match.groups(1)[0]
    raw = mne.io.Raw(proc_dir+filename, preload=True)
    raw_sel = raw.copy().pick_channels(["Cz", "Fz", "F3", "F4", "C3", "C4",
                                        "Pz", "Fp1", "Fp2", "O1", "O2",
                                        "VEOG", "HEOG", "ECG1+"])
    raw_sel.filter(l_freq=0.3, h_freq=30, n_jobs=4)
    raw_sel.resample(100, n_jobs="cuda")
    ica = ICA(method="picard")
    ica.fit(raw_sel)
    ica.save("{}sel_c_EPI_{}-ica.fif".format(proc_dir, file_id))

    bads_eog, scores = ica.find_bads_eog(raw_sel, ch_name="VEOG", threshold=2.3)
    bads_ecg ,scores = ica.find_bads_ecg(raw_sel, ch_name="ECG1+")
    bad_comps = bads_eog + bads_ecg
    raw_sel = ica.apply(raw_sel, exclude=bad_comps)

    raw_sel.save("{}csel_{}".format(proc_dir, filename), overwrite=True)
