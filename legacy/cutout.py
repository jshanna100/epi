import mne
import numpy as np

file_sections = {"csel_c_EPI_3002_20201014-raw.fif":(11700,14000),
                 "csel_c_EPI_3002_20201015-raw.fif":(10653,11653),
                 "csel_c_EPI_3001_20200708-raw.fif":(4640,6700),
                 "csel_c_EPI_3001_20200710-raw.fif":(2520,4050)}

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"proc/"

chans = ["Fz","F3","F4", "Cz","C3","C4"]

for filename, section in file_sections.items():
    raw = mne.io.Raw("{}{}".format(proc_dir, filename), preload=True)
    raw.crop(tmin=section[0], tmax=section[1])
    breakpoint()
    these_chans = [ch for ch in chans if ch not in raw.info["bads"]]
    raw_pick = raw.copy().pick_channels(these_chans)
    print(len(raw_pick.ch_names))
    avg_signal = raw_pick.get_data().mean(axis=0, keepdims=True)
    avg_info = mne.create_info(["central"], raw.info["sfreq"], ch_types="eeg")
    avg_raw = mne.io.RawArray(avg_signal, avg_info)
    raw.add_channels([avg_raw], force_update_info=True)
    raw.save("{}cut_{}".format(proc_dir, filename), overwrite=True)
