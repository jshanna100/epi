import mne
from os.path import isdir, join
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
method = "adjacent"

white_dict = {}

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw(join(proc_dir, f"f_EPI_{subj}_{cond}-raw.fif"), preload=True)
        for reg in regions:
            chs = [ch for ch in raw.ch_names if re.match(f"{reg}\d", ch)]
            inds = [int(ch[-1]) for ch in chs]
            chs = [chs[idx-1] for idx in inds] # be 100% sure channels are ordered
            if method == "adjacent":
                for idx in range(0, len(chs)-1):
                    raw = mne.set_bipolar_reference(raw, chs[idx], chs[idx+1],
                                                    ch_name=f"{chs[idx]}-{chs[idx+1]}",
                                                    drop_refs=False)
                raw.drop_channels(chs)
            elif method == "white":
                for ch in chs:
                    if ch != white_dict[subj]:
                        raw = mne.set_bipolar_reference(raw, ch, white_dict[subj],
                                                        ch_name=f"{ch}-white",
                                                        drop_refs=False)


        raw.save(join(proc_dir, f"ref_f_EPI_{subj}_{cond}-raw.fif"))
            
        
