import mne 
import numpy as np
from os import listdir
from os.path import join
import re

def get_ecog_channels(reg, chans):
    chs = []
    for ch in raw.ch_names:
        match = re.match(reg, ch)
        if match:
            chs.append(ch)
    return chs


root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc") # where the files are
proc_files = listdir(proc_dir)

subjs = ["1001", "1002", "1003", "1005", "2001", "3001", "3002"]
overwrite = False

for subj in subjs:
    infile = f"HT_f_EPI_{subj}_Stim-raw.fif"
    outfile = f"stim_EPI_{subj}_Stim-annot.fif"
    # check if it already exists
    if outfile in proc_files and not overwrite:
        print(f"{outfile} already exists. Skipping...")
        continue
    raw = mne.io.Raw(join(proc_dir, infile), preload=True)
    raw.plot(duration=400, block=True)
    onsets, durations, descrs = [], [], []
    # properly rename what the user marked
    stim_idx = 0
    for annot in raw.annotations:
        if annot["duration"] > 30:
            onsets.append(annot["onset"])
            durations.append(annot["duration"])
            descrs.append(f"BAD_Stimulation {stim_idx}")
            stim_idx += 1
    annots = mne.Annotations(onsets, durations, descrs)
    annots.save(join(proc_dir, outfile))
    

