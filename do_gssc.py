import mne
from gssc.infer import EEGInfer
from os import listdir
from os.path import join
import numpy as np

"""
Run the GSSC.
"""

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

filelist = listdir(proc_dir)
ei = EEGInfer()

subjs = ["1001", "1002", "3001", "3002"]
conds = ["Stim", "Sham"] # no point in doing this to sham conditions
for subj in subjs:
    for cond in conds:
        try:
            raw = mne.io.Raw(join(proc_dir, f"af_EPI_{subj}_{cond}-raw.fif"),
                             preload=True)
        except:
            continue
        raw.filter(l_freq=0.3, h_freq=30., n_jobs=8)
        if cond == "Stim":
            # make two raws, one before and one after stimulation
            stim_annots = [annot for annot in raw.annotations
                           if "BAD_Stimulation" in annot["description"]]
            raw_before = raw.copy().crop(0, stim_annots[0]["onset"])
            after_onset = stim_annots[-1]["onset"] + stim_annots[-1]["duration"]
            raw_after = raw.copy().crop(after_onset, raw.times[-1])
            # infer
            b4_out_infs, b4_times = ei.mne_infer(raw_before, eeg=["C3", "C4"],
                                                 eog=["HEOG"])
            out_infs, times = ei.mne_infer(raw_after, eeg=["C3", "C4"],
                                           eog=["HEOG"])
            # have to adjust the times because of GSSC doesn't handle cut
            # raw objects so well yet
            times = np.arange(after_onset, after_onset+len(out_infs)*30, 30)

            out_infs = np.hstack((b4_out_infs, out_infs))
            times = np.hstack((b4_times, times))
        else:
            out_infs, times = ei.mne_infer(raw, eeg=["C3", "C4"], eog=["HEOG"])

        annot = mne.Annotations(times, 30., out_infs.astype("str"),
                                orig_time=raw.annotations.orig_time)
        annot += raw.annotations
        breakpoint()
        annot.save(join(proc_dir, f"af_EPI_{subj}_{cond}-annot.fif"),
                   overwrite=True)
