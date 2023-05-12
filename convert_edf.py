import mne
from os import listdir
from os.path import join
import re

"""
Conversion function for EPI files recorded at Greifswald (edf to mne-python).
This has been set-up for the 1001 and 1002, and may not necessarily work for
later subjects if they are organised differently. Recordings which are split
into two files are combined into one file.
"""

subjs = ["1001", "1002"] # define subjects
conds = ["Stim", "Sham"] # define conditions

root_dir = "/home/jev/hdd/epi/" # root directory
raw_dir = join(root_dir, "raw") # get raw files from here
proc_dir = join(root_dir, "proc") # save the processed files here
proclist = listdir(proc_dir) # get list of files already processed
overwrite = True # if False, skip if file already there

done = []
for subj in subjs:
    this_dir = join(raw_dir, f"EPI_{subj}", "EEG/")
    filelist = listdir(this_dir) # get list of all files in raw directory
    for filename in filelist: # cycle through all files in raw directory
        # grabs only Stim/ShamNacht files, potentially allowing Nacht to be
        # followed by _xx, where x is a number.
        this_match = re.search("EPI.*_(.*)Nacht(_\d{2})?.edf", filename)
        # do something if the file fits the raw file pattern
        if not this_match:
            continue # doesn't match; go to next file
        # pull subject and tag out of the filename and assign to variables
        cond, file_idx = this_match.group(1), this_match.group(2)
        # if statement is complicated because some recordings are split
        # into multiple files
        if (("EPI_{}_{}-raw.fif".format(subj, cond) in proclist and
            not overwrite) or "{}_{}".format(subj, cond) in done):
            print("Already exists. Skipping.")
            continue
        if file_idx:
            # if split into two
            files = [filename[:-5]+"1.edf", filename[:-5]+"2.edf"]
        else:
            files = [filename]
        raws = []
        # combine into one file
        for file in files:
            raws.append(mne.io.read_raw_edf(this_dir+filename, preload=True))
        if len(raws) > 1:
            raws[0].append(raws[1:])

        raw = raws[0]
        del raws

        ## organise and mark the channels
        # rename the EEG channels
        eeg_chans = [ch for ch in raw.ch_names if "EEG" in ch]
        eeg_chans = {ch:ch[4:-4] for ch in eeg_chans}
        raw.rename_channels(eeg_chans)
        # Nase reference
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
        raw = mne.set_bipolar_reference(raw, "EOG O", "EOG U", ch_name="VEOG")
        raw = mne.set_bipolar_reference(raw, "EOG L", "EOG R", ch_name="HEOG")
        # EMG
        emg_chans = [ch for ch in raw.ch_names if "EMG" in ch]
        emg_chans = {ch:ch[4:] for ch in emg_chans}
        raw.rename_channels(emg_chans)
        emg_chans = {v:"emg" for v in emg_chans.values()}
        raw.set_channel_types(emg_chans)
        raw = mne.set_bipolar_reference(raw, "EMG L", "EMG R", ch_name="EMG")
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
        raw.save(join(proc_dir, f"EPI_{subj}_{cond}-raw.fif"),
                 overwrite=overwrite) # save
        # keep track of this for use with the complex if statement above
        done.append("{}_{}".format(subj, cond))
