import mne
from os.path import isdir
import numpy as np
import re
from os.path import join
from os import listdir

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc") # where the files are
proc_files = listdir(proc_dir)

subjs = ["1001", "1002"]
conds = ["Stim", "Sham"]

l_freq = 0.3
h_freq = 150
n_jobs = "cuda" # if you don't have cuda, change this 1 or something higher
overwrite = True

for subj in subjs:
    for filename in proc_files:
        # correctly identify subject raw files
        ma = re.match(f"EPI_{subj}_(.*)-raw.fif", filename)
        if not ma:
            continue
        infile = f"EPI_{subj}_{ma.groups(0)[0]}-raw.fif"
        # check if it already exists
        outfile = f"f_{infile}"
        if outfile in proc_files and not overwrite:
            print(f"{outfile} already exists. Skipping...")
            continue
        # filter and save
        raw = mne.io.Raw(join(proc_dir, infile), preload=True)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
        raw.notch_filter(np.arange(50, h_freq, 50), n_jobs=n_jobs)

        # reference to nose
        try:
            raw = mne.set_eeg_reference(raw, ref_channels=["Nase"])[0]
        except:
            print("No Nase channel found")

        raw.save(join(proc_dir, outfile), overwrite=overwrite)
