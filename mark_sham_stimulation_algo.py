import mne
from os import listdir
from os.path import isdir, join
import re
import numpy as np
from gssc.infer import EEGInfer
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

"""
Algorithmically determines and marks stimulations on sham data
"""

def annotate(raw, infer_chans, max_stims=5):
    chans_to_use = [c for c in infer_chans if c not in raw.info["bads"]]
    stages, times = ei.mne_infer(raw, eeg=chans_to_use, eog=["HEOG"])
    hypno_annots = mne.Annotations(times, 30., stages.astype("str"))
    ## mark the sham stimulation and pre/post periods
    stim_annots = mne.Annotations([], [], [])
    # calculate some arrays we'll need
    n2_inds = stages==2
    n2_min_idx = int(np.round(min_n2 / 0.5)) # assumes min_n2 is expressed in minutes; 0.5 is 30s
    n2_min_inter_idx = int(np.round(min_n2_inter / 0.5)) # assumes min_n2_inter is expressed in minutes; 0.5 is 30s
    n2_min = np.array([n2_inds[x-n2_min_idx:x].sum() for x in range(n2_min_idx, len(n2_inds))])
    n2_min_inter = np.array([n2_inds[x-n2_min_inter_idx:x].sum() for x in range(n2_min_inter_idx, len(n2_inds))])
    stim_len_idx = int(np.round((stim_duration+analy_duration)/30))
    # find the first stimulation point
    if not sum(n2_min) or (n2_min.max() < n2_min_idx):
        print(f"Subject {subj} does not appear to sleep. Skipping...")
        return None, 0, None
    # set current idx to first place with n2_min_idx of consecutive N2 sleep
    cur_idx = np.where(n2_min==n2_min_idx)[0][0] + n2_min_idx # add this because n2_min starts n2_min_idx ahead (see above)
    stim_annots.append(times[cur_idx], stim_duration, "BAD_Stimulation 0")
    stim_annots.append(times[cur_idx]+stim_duration, analy_duration, "Post_Stimulation 0")
    # do subsequent stimuli
    nrem_inds = (stages==2) | (stages==3)
    stim_idx = 1
    last_idx = cur_idx
    cur_idx = last_idx + stim_len_idx + np.random.randint(*gap_idx_range)
    # keep going until 15 stimulations or end of recording
    while stim_idx <= max_stims and cur_idx < (len(stages)-6):
        # check if Wake happened in the meantime
        if sum(~nrem_inds[last_idx:cur_idx]):
            # there was a wake or REM stage; find the next min_n2_inter stage
            next_n2_idx = np.where(n2_min_inter[cur_idx-n2_min_inter_idx:]==n2_min_inter_idx)[0]
            if not len(next_n2_idx):
                # not another N2; we're done here
                break
            cur_idx = next_n2_idx[0] + cur_idx-n2_min_inter_idx
        stim_annots.append(times[cur_idx], stim_duration, f"BAD_Stimulation {stim_idx}")
        stim_annots.append(times[cur_idx]+stim_duration, analy_duration, 
                           f"Post_Stimulation {stim_idx}")
        last_idx = cur_idx
        cur_idx = last_idx + stim_len_idx + np.random.randint(*gap_idx_range)
        stim_idx += 1
    return stim_annots, stim_idx, hypno_annots


root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

stim_duration = 90
analy_duration = 60
gap_idx_range = [1, 3]
min_stims = 3
max_stims = 5
min_n2 = 4 # minimum minutes of N2 sleep for beginning
min_n2_inter = 1 # same but for interval between stimulations
overwrite = False
infer_chans = ["C3", "C4"]

df_dict = {"Subject":[], "Stimulations":[]}
ei = EEGInfer()
filenames = listdir(proc_dir)
for filename in filenames:
    match = re.match("HT_f_EPI_(\d{4})_Sham-raw.fif", filename)
    if match:
        subj = match.groups()[0]
    else:
        continue
    outfile = f"stim_EPI_{subj}_Sham-annot.fif"
    if outfile in filenames and not overwrite:
        print(f"{outfile} already exists. Skipping...")
        continue

    raw = mne.io.Raw(join(proc_dir, filename), preload=True)
    raw.filter(l_freq=0.3, h_freq=30)
    stim_annots, stim_idx, hypno_annots = annotate(raw, infer_chans, max_stims=max_stims)
    # if stim_idx < min_stims:
    #     print(f"\n\nFewer than {min_stims} stimulations could be marked. Trying again...\n\n")
    #     stim_annots, stim_idx, hypno_annots = annotate(raw, backup_chans, max_stims=max_stims)
    if stim_idx < min_stims:
        print(f"\n\nFewer than {min_stims} stimulations could be marked. Failed.\n\n")
    else:
        print(f"\n\n{stim_idx} stimulations marked\n\n")
        stim_annots.save(join(proc_dir, outfile), overwrite=overwrite)
    df_dict["Subject"].append(subj)
    df_dict["Stimulations"].append(stim_idx)

df = pd.DataFrame.from_dict(df_dict)
df = df.sort_values(["Subject"])
df.to_csv(join(proc_dir, "Stim_Ns.csv"))