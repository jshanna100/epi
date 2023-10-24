import mne
from gssc.infer import EEGInfer
from os import listdir
from os.path import join
import numpy as np
from utils import mark_osc, mark_spindle, mark_ied

"""
Do sleep staging and oscillation detection
"""

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

filelist = listdir(proc_dir)
ei = EEGInfer()

subjs = ["1001", "1002", "1003", "1004", "1005", "2001", "3001", "3002"]
conds = ["Sham"]

chans_a = ["FC1", "FC2"]
chans_b = ["F3", "Fz", "F4"]

for subj in subjs:
    chans = chans_a if "100" in subj else chans_b
    for cond in conds:
        try:
            ur_raw = mne.io.Raw(join(proc_dir, f"HT_f_EPI_{subj}_{cond}-raw.fif"),
                             preload=True)
        except:
            continue
        # sleep staging
        print("Sleep staging")
        raw = ur_raw.copy()
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
        annot.save(join(proc_dir, f"hypno_EPI_{subj}_{cond}-annot.fif"),
                overwrite=True)
        del raw

        # slow osc
        print("Slow oscillations")
        so_annots = mark_osc("SO", ur_raw, chans, (0.16, 1.25), (0.8, 2))
        so_annots.save(join(proc_dir, f"SO_EPI_{subj}_{cond}-annot.fif"),
                   overwrite=True)

        # spindles
        print("Spindles")
        spindle_band = [12, 16]
        spindle_lens = [0.5, 3]
        annot_len = [-2.5, 2.5]

        spindle_annots =  mark_spindle(ur_raw, chans, spindle_band, spindle_lens, annot_len)
        spindle_annots.save(join(proc_dir, f"spindle_EPI_{subj}_{cond}-annot.fif"),
                            overwrite=True)

        # IED detection
        print("IED detection")
        hl_ied_annots = mark_ied(ur_raw, "HL", [25, 80], [0.02, 0.1])
        hl_ied_annots.save(join(proc_dir, f"HLIED_EPI_{subj}_{cond}-annot.fif"),
                           overwrite=True)
        hr_ied_annots = mark_ied(ur_raw, "HR", [25, 80], [0.02, 0.1])
        hr_ied_annots.save(join(proc_dir, f"HRIED_EPI_{subj}_{cond}-annot.fif"),
                           overwrite=True)
