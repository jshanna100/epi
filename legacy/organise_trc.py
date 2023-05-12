import mne
from os import listdir
import numpy as np
import re

def chan_sub(chans):
    chans[0,] = chans[0,] - chans[1,]
    return chans

# different directories for home and office computers; not generally relevant
# for other users
root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"/proc/"
proclist = listdir(proc_dir)

subjs = ["3001", "3002"]
l_freq = 0.3
h_freq = 200
n_jobs = "cuda"
overwrite = True

for subj in subjs:
    for filename in proclist:
        this_match = re.match("EPI_{}_(.*)-raw.fif".format(subj), filename)
        if not this_match:
            continue
        if "c_{}".format(filename) in proclist and not overwrite:
            print("Skipping.")
            continue
        raw = mne.io.Raw(proc_dir+filename, preload=True)
        if "VO1" in raw.ch_names:
            raw.rename_channels({"VO1":"VO"})
        if "Vu1" in raw.ch_names:
            raw.rename_channels({"Vu1":"VU"})
        if "RE" in raw.ch_names:
            raw.rename_channels({"RE":"Re"})
        if "RE1" in raw.ch_names:
            raw.rename_channels({"RE1":"Re"})
        if "Li1" in raw.ch_names:
            raw.rename_channels({"Li1":"Li"})

        # EOG

        raw.apply_function(chan_sub, ["VO", "VU"])
        raw.apply_function(chan_sub, ["Li", "Re"])
        raw.rename_channels({"VO":"VEOG", "Li":"HEOG"})
        raw.drop_channels(["VU", "Re"])
        eog_chans = {c:"eog" for c in ["VEOG", "HEOG"]}
        raw.set_channel_types(eog_chans)

        # ECOG
        ecog_chans = [ch for ch in raw.ch_names if "AM" in ch or "HB" in ch or "HH" in ch or "HT" in ch]
        ecog_chans = {v:"ecog" for v in ecog_chans}
        raw.set_channel_types(ecog_chans)
        # EKG
        ekg_chans = [ch for ch in raw.ch_names if "ECG" in ch]
        ekg_chans = {v:"ecg" for v in ekg_chans}
        raw.set_channel_types(ekg_chans)
        # dump everything else
        dump_chans = [ch for ch in raw.ch_names if "el" in ch]
        dump_chans.extend([ch for ch in raw.ch_names if re.match("\d", ch)])
        for misc_chan in ['thor+', 'abdo+', 'xyz+', 'PULS+', 'BEAT+', 'cn']:
            if misc_chan in raw.ch_names:
                dump_chans.append(misc_chan)
        raw.drop_channels(dump_chans)
        for ch in raw.ch_names:
            if "MKR" in ch or "Mo" in ch or "REF" in ch or "T4" in ch or "T3" in ch or "A1" in ch or "A2" in ch:
                raw.drop_channels([ch])
        raw.set_montage("standard_1005", on_missing="ignore")
        raw.notch_filter([50,100,150], n_jobs="cuda")
        raw.save("{}c_{}".format(proc_dir, filename), overwrite=overwrite)
        del raw
