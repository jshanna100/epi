import mne
import numpy as np
import re
from os import listdir

def pca(data):
    U, s, V = np.linalg.svd(data, full_matrices=False)
    # use average power in label for scaling
    scale = np.linalg.norm(s) / np.sqrt(len(data))
    summary = scale * V[0]
    return summary.T

# different directories for home and office computers; not generally relevant
# for other users

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"proc/"
proclist = listdir(proc_dir)

file_sections = {"3002_20201014-raw.fif":(11700,14000),
                 "3002_20201015-raw.fif":(10653,11653),
                 "3001_20200708-raw.fif":(4640,6700),
                 "3001_20200710-raw.fif":(2520,4050)}

regions = ["HHL", "HBL", "HTL", "HHR", "HBR", "HTR"]

for file_id, section in file_sections.items():
    raw = mne.io.Raw("{}c_EPI_{}".format(proc_dir, file_id), preload=True)
    raw.crop(tmin=section[0], tmax=section[1])
    raw.pick_types(ecog=True)
    breakpoint()
    new_chs = []
    for reg in regions:
        chs = []
        for ch in raw.ch_names:
            match = re.match(reg,ch)
            if match and ch not in raw.info["bads"]:
                chs.append(ch)
                print("Included {}".format(ch))
        if len(chs):
            data = raw.get_data(chs)
            summary = np.expand_dims(pca(data), 0)
            new_ch = mne.io.RawArray(summary,mne.create_info([reg],
                                  raw.info["sfreq"], ch_types="ecog"))
            new_chs.append(new_ch)
    raw.add_channels(new_chs, force_update_info=True)
    raw.pick_channels(regions)

    raw.save("{}ecog_EPI_{}".format(proc_dir, file_id))
