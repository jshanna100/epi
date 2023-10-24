import mne
from os import listdir
from os.path import join
import re

"""
Conversion function for EPI files recorded at Greifswald and Magdeburg
"""

def reorg(raw):
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

    return raw


subjs = ["1001", "1002"] # define subjects
conds = ["Stim", "Sham"] # define conditions
subjs = ["2001"]

root_dir = "/home/jev/hdd/epi/" # root directory
raw_dir = join(root_dir, "raw") # get raw files from here
proc_dir = join(root_dir, "proc") # save the processed files here
proclist = listdir(proc_dir) # get list of files already processed
overwrite = True # if False, skip if file already there

for subj in subjs:
    this_dir = join(raw_dir, f"EPI_{subj}", "EEG/")
    if subj == "1001" or subj == "1002":
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
            if file_idx:
                # if split into two
                files = [filename[:-5]+"1.edf", filename[:-5]+"2.edf"]
            else:
                files = [filename]
            raws = []
            # combine into one file
            for file in files:
                raws.append(mne.io.read_raw_edf(join(this_dir, filename), preload=True))
            if len(raws) > 1:
                raws[0].append(raws[1:])

            raw = raws[0]
            del raws

            raw = reorg(raw)
            raw.save(join(proc_dir, f"EPI_{subj}_{cond}-raw.fif"),
                 overwrite=overwrite) # save

    elif subj == "1003":
        files = ["2022_07_22_Nacht_Studie_2327-0025_SHAM.edf",
                 "2022_07_20_STIM_0030-0140.edf"]
        conds = ["Sham", "Stim"]
        for cond, file in zip(conds, files):
            raw = mne.io.read_raw_edf(join(this_dir, file), preload=True)
            raw.rename_channels({"POL NASE":"POL Nase"})
            raw = reorg(raw)
            raw.save(join(proc_dir, f"EPI_{subj}_{cond}-raw.fif"),
                 overwrite=overwrite) # save
    
    elif subj == "1004":
        dir_1004 = join(this_dir, "T1")
        files = ["DA0380RW_1.edf",
                 "DA0380RX_2.edf",
                 "DA0380RY_3.edf",
                 "DA0380RZ_4.edf",
                 "DA0380S0_5.edf"]
        raws = [mne.io.read_raw_edf(join(dir_1004, file), preload=True) for file in files]
        raws[0].append(raws[1:])
        raw = raws[0]
        del raws
        raw.rename_channels({"POL NASE":"POL Nase"})
        raw = reorg(raw)
        raw.save(join(proc_dir, f"EPI_{subj}_Sham-raw.fif"),
                 overwrite=overwrite) # save
        
    elif subj == "1005":
        conds = ["Stim", "Sham"]
        Ts = ["T2", "T3"]
        files = {"T2":["VA0369HF_1.edf", "VA0369HG_2.edf", "VA0369HH_3.edf", "VA0369HI_4.edf", "VA0369HJ_5.edf"],
                 "T3":["VA0369I3_1.edf", "VA0369I4_2.edf", "VA0369I5_3.edf", "VA0369I6_4.edf", "VA0369I7_5.edf"]}
        for T, cond in zip(Ts, conds):
            dir_1005 = join(this_dir, T)
            raws = [mne.io.read_raw_edf(join(dir_1005, file), preload=True) for file in files[T]]
            raws[0].append(raws[1:])
            raw = raws[0]
            del raws
            raw = reorg(raw)
            raw.save(join(proc_dir, f"EPI_{subj}_{cond}-raw.fif"),
                    overwrite=overwrite) # save
            
    elif subj == "2001":
        conds = ["Stim", "Sham"]
        Ts = ["T2", "T1"]
        files = {"T2":["FA2004LX.edf", "FA2004LY.edf", "FA2004LZ.edf", "FA2004M0.edf", "FA2004M1.edf", "FA2004M2.edf", "FA2004M3.edf"],
                 "T1":["FA2004KX.edf", "FA2004KY.edf", "FA2004KZ.edf", "FA2004L0.edf", "FA2004L1.edf", "FA2004L2.edf", "FA2004L3.edf"]}
        for T, cond in zip(Ts, conds):
            dir_1005 = join(this_dir, T)
            raws = [mne.io.read_raw_edf(join(dir_1005, file), preload=True, encoding="latin1") for file in files[T]]
            raws = [r.resample(500, n_jobs="cuda") for r in raws] # run out of ram otherwise
            raws[0].append(raws[1:])
            raw = raws[0]
            del raws
            raw.rename_channels({"POL LOC":"EOG EOG L", "POL ROC":"EOG EOG R", "POL Vo":"EOG EOG O", "POL Vu":"EOG EOG U",
                                 "POL Kinn1":"EMG EMG L", "POL Kinn2":"EMG EMG R"})
            raw = reorg(raw)
            raw.save(join(proc_dir, f"EPI_{subj}_{cond}-raw.fif"),
                    overwrite=overwrite) # save



        
        
