import mne
from os import listdir
from os.path import isdir
import re
from read_trc import raw_from_neo
import numpy as np
from os.path import join
import datetime


"""
Conversion function for EPI files recorded at Frankfurt (trc to mne-python).
This has been set-up for the 3001 and 3002, and may not necessarily work for
later subjects if they are organised differently. The read_trc function
is imported from raw_from_neo.py, which must be in the same directory
as this script.
"""

subjs = ["3001", "3002"] # define subjects
subjs = ["3001"]

root_dir = "/home/jev/hdd/epi/" # root directory
raw_dir = join(root_dir, "raw") # get raw files from here
proc_dir = join(root_dir, "proc") # save the processed files here
proclist = listdir(proc_dir) # get list of files already processed
overwrite = True # if False, skip if file already there

downsamp = 300 # downsample to 300Hz, could probably go even lower
n_jobs = 12

cond_dict = {"3001":{0:"Stim", 3:"Sham"},
             "3002":{2:"Stim", 0:"Sham"}
             }

for this_subj in subjs:
    this_dir = join(raw_dir, f"EPI_{this_subj}", "EEG")
    filelist = listdir(this_dir) # get list of all files in raw directory
    subs, days, ids = [], [], []

    # prepare lists of all days, and IDs for this subject
    for filename in filelist: # cycle through all files in raw directory
        this_match = re.search("(.*)_(.*)_(.*).((TRC)|(trc))", filename)
        # do something if the file fits the raw file pattern
        if not this_match:
            continue # doesn't match, skip
        days.append(this_match.group(2))
        ids.append(this_match.group(3))

    # get rid of redundant
    days = list(set(days))
    days.sort()
    ids = list(set(ids))

    for k, v in cond_dict[this_subj].items():
        raw_names = []
        datetimes = []
        if f"EPI_{this_subj}_{days[k]}-raw.fif" in proclist and not overwrite:
            print("Already exist; Skipping...")
            continue
        for id in ids:
            # this loop makes temporary conversions to MNE in the next
            # step these conversions are combined into a single file per
            # session
            filename = "{}_{}_{}.TRC".format(this_subj, days[k], id)
            if filename not in filelist:
                continue
            raw = raw_from_neo(join(this_dir, filename)) # convert
            raw.resample(downsamp, n_jobs=n_jobs)
            temp_path = join(proc_dir, "temp", f"{id}-raw.fif")
            raw.save(temp_path, overwrite=True)
            raw_names.append(temp_path)
            datetimes.append(raw.info["meas_date"])
            del raw

        # now combine the raw files into single session
        print("\nCombining files\n")
        file_order = np.argsort(datetimes)[::-1] # make sure we append files in correct order
        raw_names = [raw_names[idx] for idx in np.nditer(file_order)]
        raw = mne.io.Raw(raw_names[0])
        expected_time = raw.info["meas_date"] + datetime.timedelta(0, raw.times[-1])
        for rn_idx, rn in enumerate(raw_names[1:]):
            next_raw = mne.io.Raw(rn)
            print(f"\nExpected {expected_time}\nFound {next_raw.info['meas_date']}\n")
            expected_time = next_raw.info["meas_date"] + datetime.timedelta(0, next_raw.times[-1])
            raw.append(next_raw)
        raw.meas_date = datetimes[0]
        del next_raw
        raw.load_data()

        # channel organisation
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
        raw = mne.set_bipolar_reference(raw, "VO", "VU", ch_name="VEOG")
        raw = mne.set_bipolar_reference(raw, "Li", "Re", ch_name="HEOG")
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

        raw.save(join(proc_dir, f"EPI_{this_subj}_{v}-raw.fif"),
                 overwrite=overwrite)
        breakpoint()
        del raw
