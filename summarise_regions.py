import mne
from os.path import isdir
import numpy as np
import re

def pca(data):
    summary = np.zeros((data.shape[0], 1, data.shape[2]))
    for epo_idx in range(data.shape[0]):
        U, s, V = np.linalg.svd(data[epo_idx,], full_matrices=False)
        # use average power in label for scaling
        scale = np.linalg.norm(s) / np.sqrt(len(data))
        summary[epo_idx,] = scale * V[0]
    return summary

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

proc_dir = root_dir+"/proc/"

subjs = ["1001", "1002"]
subjs = ["1001"]
conds = ["Stim"]
n_jobs = 8

regions = ["HHL", "HCL", "HTL", "HHR", "HCR", "HTR"]

for subj in subjs:
    for cond in conds:
        epo = mne.read_epochs("{}epi_{}_{}-epo.fif".format(proc_dir,subj,cond))
        epo_chs = []
        for reg in regions:
            chs = []
            for ch in epo.ch_names:
                match = re.match(reg,ch)
                if match:
                    chs.append(ch)
            if len(chs):
                data = epo.get_data(chs)
                summary = pca(data)
                epo_ch = mne.EpochsArray(summary,mne.create_info([reg],
                                                                 epo.info["sfreq"],
                                                                 ch_types="ecog"))
                epo_chs.append(epo_ch)
        epo.add_channels(epo_chs, force_update_info=True)
        epo.save("{}s_epi_{}_{}-epo.fif".format(proc_dir,subj,cond))
