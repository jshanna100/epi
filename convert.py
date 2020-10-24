import mne
from os import listdir
from os.path import isdir
import re

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

subjs = ["1001", "1002"]
conds = ["Stim", "Sham"]

raw_dir = root_dir+"raw/" # get raw files from here
proc_dir = root_dir+"proc/" # save the processed files here
proclist = listdir(proc_dir) # and in proc directory
overwrite = True # skip

dones = []
for subj in subjs:
    this_dir = "{}EPI_{}/EEG/".format(raw_dir,subj)
    filelist = listdir(this_dir) # get list of all files in raw directory
    for filename in filelist: # cycle through all files in raw directory
        this_match = re.search("EPI.*_(.*)Nacht(_\d{2})?.edf", filename)
        # do something if the file fits the raw file pattern
        if this_match:
            # pull subject and tag out of the filename and assign to variables
            cond, file_idx = this_match.group(1), this_match.group(2)
            if (("EPI_{}_{}-raw.fif".format(subj, cond) in proclist and
                not overwrite) or "{}_{}".format(subj,cond) in dones):
                print("Already exists. Skipping.")
                continue
            if file_idx:
                files = [filename[:-5]+"1.edf", filename[:-5]+"2.edf"]
            else:
                files = [filename]
            raws = []
            for file in files:
                raws.append(mne.io.read_raw_edf(this_dir+filename)) # convert
            if len(raws) > 1:
                raws[0].append(raws[1:])
            raws[0].save("{}EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         overwrite=overwrite) # save
            dones.append("{}_{}".format(subj, cond))
