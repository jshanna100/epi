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

subjs = ["1001", "1002", "1003", "1004", "1005", "2001", "3001", "3002"]
conds = ["Stim", "Sham"]
# search in these regions, in this order, for valid channels
reg_priority = ["HT", "HC", "HH", "H", "AMY"] 
hemis = ["L", "R"]
magd_hemis = ["1.", "2."]

overwrite = False

for subj in subjs:
    for cond in conds:
        infile = f"f_EPI_{subj}_{cond}-raw.fif"
        outfile = "HT_" + infile
        # check if it already exists
        if outfile in proc_files and not overwrite:
            print(f"{outfile} already exists. Skipping...")
            continue
        try:
            raw = mne.io.Raw(join(proc_dir, infile), preload=True)
        except:
            continue
        hem_orig = magd_hemis if "200" in subj else hemis
        for hemi_idx in range(2):
            for rp in reg_priority:
                chs = get_ecog_channels(f"{rp}{hem_orig[hemi_idx]}", raw.ch_names)
                if len(chs):
                    break
            if not len(chs):
                print("Could not find a valid region")
                continue
            this_raw = raw.copy().pick_channels(chs)
            actual_bads = this_raw.info["bads"].copy()
            this_raw.plot(block=True)
            hc_picks = list(set(this_raw.info["bads"]) - set(actual_bads))
            raw = mne.set_bipolar_reference(raw, *hc_picks, ch_name=f"H{hemis[hemi_idx]}")
            raw.set_channel_types({f"H{hemis[hemi_idx]}":"misc"})
        raw.pick_types(eeg=True, misc=True, eog=True, emg=True, ecog=False)
        raw.set_channel_types({f"H{hemi}":"ecog" for hemi in hemis if f"H{hemi}" in raw.ch_names})
        raw.save(join(proc_dir, outfile))

