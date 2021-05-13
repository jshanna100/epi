import mne
from os import listdir
from os.path import isdir
import re
from read_trc import raw_from_neo
import numpy as np

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

subjs = ["3001", "3002"]

raw_dir = root_dir+"raw/" # get raw files from here
proc_dir = root_dir+"proc/" # save the processed files here
proclist = listdir(proc_dir) # and in proc directory
overwrite = True # skip
downsamp = 500

for this_subj in subjs:
    this_dir = "{}EPI_{}/EEG/".format(raw_dir, this_subj)
    filelist = listdir(this_dir) # get list of all files in raw directory
    subs, days, ids = [], [], []
    for filename in filelist: # cycle through all files in raw directory
        this_match = re.search("(.*)_(.*)_(.*).((TRC)|(trc))", filename)
        # do something if the file fits the raw file pattern
        if this_match:
            subs.append(this_match.group(1))
            days.append(this_match.group(2))
            ids.append(this_match.group(3))
    subs = list(np.unique(np.array(subs)))
    days = list(np.unique(np.array(days)))
    ids = list(np.unique(np.array(ids)))

    for sub in subs:
        for day in days:
            raw_names = []
            datetimes = []
            if "EPI_{}_{}-raw.fif".format(sub, day) in proclist and not overwrite:
                print("Skipping.")
                continue
            for id in ids:
                filename = "{}_{}_{}.TRC".format(sub, day, id)
                if filename not in filelist:
                    continue
                raw = raw_from_neo(this_dir+filename) # convert
                raw.resample(400, n_jobs="cuda")
                raw.save("{}temp/{}-raw.fif".format(proc_dir,id), overwrite=True)
                raw_names.append("{}temp/{}-raw.fif".format(proc_dir,id))
                datetimes.append(raw.info["meas_date"])
                del raw

            file_order = np.argsort(datetimes)
            raw_names = [raw_names[idx] for idx in np.nditer(file_order)]
            raw = mne.io.Raw(raw_names[0])
            for rn in raw_names[1:]:
                next_raw = mne.io.Raw(rn)
                raw.append(next_raw)
                del next_raw
            raw.meas_date = datetimes[0]
            raw.save("{}EPI_{}_{}-raw.fif".format(proc_dir, sub, day),
                     overwrite=overwrite)
            del raw
