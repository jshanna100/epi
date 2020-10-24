import mne
from os.path import isdir
import numpy as np
import re

def chan_sub(chans):
    chans[0,] = chans[0,] - chans[1,]
    return chans

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

proc_dir = root_dir+"/proc/"

subjs = ["1001", "1002"]
conds = ["Stim", "Sham"]

l_freq = 0.3
h_freq = 200
n_jobs = "cuda"

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw("{}f_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         preload=True)
        # rename the EEG channels
        eeg_chans = [ch for ch in raw.ch_names if "EEG" in ch]
        eeg_chans = {ch:ch[4:-4] for ch in eeg_chans}
        raw.rename_channels(eeg_chans)
        # Nase
        raw.set_channel_types({"POL Nase":"eeg"})
        raw.rename_channels({"POL Nase":"Nase"})
        raw.set_eeg_reference(ref_channels=["Nase"])
        raw.set_channel_types({"Nase":"misc"})
        # EOG
        eog_chans = [ch for ch in raw.ch_names if "EOG" in ch]
        eog_chans = {ch:ch[4:] for ch in eog_chans}
        raw.rename_channels(eog_chans)
        eog_chans = {v:"eog" for v in eog_chans.values()}
        raw.set_channel_types(eog_chans)
        raw.apply_function(chan_sub, ["EOG O", "EOG U"])
        raw.apply_function(chan_sub, ["EOG L", "EOG R"])
        raw.rename_channels({"EOG O":"VEOG", "EOG L":"HEOG"})
        raw.drop_channels(["EOG U", "EOG R"])
        # EMG
        emg_chans = [ch for ch in raw.ch_names if "EMG" in ch]
        emg_chans = {ch:ch[4:] for ch in emg_chans}
        raw.rename_channels(emg_chans)
        emg_chans = {v:"emg" for v in emg_chans.values()}
        raw.set_channel_types(emg_chans)
        raw.apply_function(chan_sub, ["EMG L", "EMG R"])
        raw.rename_channels({"EMG L":"EMG"})
        raw.drop_channels(["EMG R"])
        # ECOG
        ecog_chans = [ch for ch in raw.ch_names if "AM" in ch or "HL" in ch or "HR" in ch or "HC" in ch or "HT" in ch or "PVH" in ch]
        ecog_chans = {ch:ch[4:] for ch in ecog_chans}
        raw.rename_channels(ecog_chans)
        ecog_chans = {v:"ecog" for v in ecog_chans.values()}
        raw.set_channel_types(ecog_chans)
        # EKG
        ekg_chans = [ch for ch in raw.ch_names if "EKG" in ch]
        ekg_chans = {ch:ch[4:] for ch in ekg_chans}
        raw.rename_channels(ekg_chans)
        ekg_chans = {v:"ecg" for v in ekg_chans.values()}
        raw.set_channel_types(ekg_chans)
        # dump everything else
        pol_chans = [ch for ch in raw.ch_names if "POL" in ch]
        raw.drop_channels(pol_chans)

        raw.save("{}of_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond), overwrite=True)
