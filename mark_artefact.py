import mne
from gssc.infer import EEGInfer
from os import listdir
from os.path import join
import numpy as np
from scipy.stats import iqr

def chunk_mask(mask):
    flip_inds = np.where(mask[1:]!=mask[:-1])[0]
    last_idx = 0
    chunks, ids = [], []
    id = mask[0]
    for flip_idx in flip_inds:
        chunks.append(np.arange(last_idx, flip_idx))
        last_idx = flip_idx+1
        ids.append(id)
        id = ~id
    return chunks, ids

"""
Figures out and marks where stimulation occurred.
"""

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")
n_jobs = 1

filelist = listdir(proc_dir)
ei = EEGInfer()

chans = ['C3', 'C4', 'FC1', 'FC2', 'CP1', 'CP2']
chans = ["Fz", "Cz"]

subjs = ["1001", "1002"]
conds = ["Stim", "Sham"] # no point in doing this to sham conditions
for subj in subjs:
    for cond in conds:
        try:
            ur_raw = mne.io.Raw(join(proc_dir, f"f_EPI_{subj}_{cond}-raw.fif"),
                                preload=True)
        except:
            print(f"Could not load af_EPI_{subj}_{cond}-raw.fif")
            continue

        # default to Cz, use Fz if Cz is bad
        this_chan = chans[0] if chans[0] not in ur_raw.info["bads"] else chans[1]
        ur_raw.pick_channels([this_chan])

        raws = []
        if cond == "Stim":
            annots = mne.read_annotations(join(proc_dir,
                                               f"stim_EPI_{subj}_Stim-annot.fif"))
            ur_raw.set_annotations(annots)
            # make two raws, one before and one after stimulation
            stim_annots = [ann for ann in ur_raw.annotations
                             if "Stimulation" in ann["description"]]
            raws.append(ur_raw.copy().crop(0, stim_annots[0]["onset"]))
            after_onset = stim_annots[-1]["onset"] + stim_annots[-1]["duration"]
            raws.append(ur_raw.copy().crop(after_onset, ur_raw.times[-1]))
        else:
            raws = [ur_raw]
            stim_annots = mne.Annotations([], [], [],
                                          orig_time=ur_raw.annotations.orig_time)

        annots = []
        for raw in raws:
            # prepare a mask for all artefactual time points
            art_mask = np.zeros(len(raw), dtype=bool)

            # # all amp over 750uv
            this_raw = raw.copy().filter(l_freq=0.3, h_freq=150.,
                                         n_jobs=n_jobs)
            raw_arr = this_raw.get_data()
            over_inds = np.where(np.any(abs(raw_arr)>750e-6, axis=0))[0]
            art_mask[over_inds] = 1

            # gradients
            grad_arr = raw_arr[:, 1:] - raw_arr[:, :-1]
            meds = np.median(grad_arr, axis=1)
            iqrs = iqr(grad_arr, axis=1)
            thresh = (meds + iqrs*6)[:, None]
            over_inds = np.where(np.any(abs(grad_arr)>thresh, axis=0))[0]
            art_mask[over_inds] = 1

            # +150Hz Noise
            win_len = .1
            this_raw = raw.copy().filter(l_freq=150., h_freq=None)
            raw_arr = this_raw.get_data()
            raw_sq = raw_arr ** 2
            win = np.ones(raw.time_as_index(win_len))
            ms = []
            for r_sq in raw_sq:
                ms.append(np.convolve(r_sq, win, mode="same"))
            ms = np.array(ms)
            rms = np.sqrt(ms)
            meds = np.median(rms, axis=1)
            iqrs = iqr(rms, axis=1)
            thresh = (meds + iqrs*4)[:, None]
            over_inds = np.where(np.any(rms>thresh, axis=0))[0]
            art_mask[over_inds] = 1

            # pad
            pad_idx = raw.time_as_index(.25)[0]
            for idx in np.where(art_mask)[0]:
                if idx < pad_idx:
                    left_pad = 0
                else:
                    left_pad = idx-pad_idx
                if len(art_mask)-idx < pad_idx:
                    right_pad = len(art_mask)
                else:
                    right_pad = idx+pad_idx
                art_mask[left_pad:right_pad] = 1


            # fill in small gaps between artefacts as artefacts
            gap_idx = raw.time_as_index(3)[0]
            chunks, ids = chunk_mask(art_mask)
            for chunk, id in zip(chunks, ids):
                if ~id and len(chunk) < gap_idx:
                    art_mask[chunk] = True

            # convert to annotations
            chunks, ids = chunk_mask(art_mask)
            onsets, durations = [], []
            first_time = ur_raw.times[raw.first_samp]
            for chunk, id in zip(chunks, ids):
                if id:
                    onsets.append(raw.times[chunk[0]])
                    durations.append(len(chunk)/this_raw.info["sfreq"])
            onsets += first_time
            annot = mne.Annotations(onsets, durations, ["BAD"]*len(onsets),
                                    orig_time=ur_raw.annotations.orig_time)
            annots.append(annot)

        annot = annots[0]
        for ann in annots[1:]:
            annot += ann
        annot += ur_raw.annotations
        ur_raw.set_annotations(annot)
        annot.save(join(proc_dir, f"art_EPI_{subj}_{cond}-annot.fif"),
                   overwrite=True)
