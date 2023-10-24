import mne
from os.path import join
from os import listdir
import numpy as np
import re

# taken from github @alimanfoo
def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

chan_groups = {"frontal":["Fz", "FC1","FC2"]}
spindle_band = [12, 16]
spindle_std = [1.5, np.inf]
spindle_lens = [0.5, 3]
annot_len = [-2.5, 2.5]

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

filelist = listdir(proc_dir)
for filename in filelist:
    for ROI, chans in chan_groups.items():
        this_match = re.match("c_EPI_(\d{4})_(.*)-raw.fif", filename)
        if not this_match:
            continue
        (subj, cond) = this_match.groups()

        # if subj != "1005" or cond != "Sham":
        #     continue

        raw = mne.io.Raw(join(proc_dir, filename), preload=True)        
        raw_work = raw.copy()
        raw_work.filter(l_freq=spindle_band[0], h_freq=spindle_band[1],
                        verbose="warning")
        signal = raw_work.pick_channels(chans).get_data().mean(axis=0)
        # moving average of 200ms
        idx_200ms = raw.time_as_index(0.2)[0]
        i_100 = idx_200ms // 2
        times = raw.times[i_100:-i_100]

        # normalise around the mean
        sig_rms = np.zeros(len(signal) - idx_200ms)
        for ss_idx, s_idx in enumerate(range(i_100, len(signal) - i_100)):
            sig_rms[ss_idx] = np.sqrt(np.mean(signal[s_idx-i_100:s_idx+i_100]**2))
        spindle_thresh = [sig_rms.mean() + spindle_std[0] * sig_rms.std(),
                          sig_rms.mean() + spindle_std[1] * sig_rms.std()]

        # all segments between the spindle std range
        hits = ((sig_rms > spindle_thresh[0]) &
                (sig_rms < spindle_thresh[1]))
        # contiguous hits
        run_vals, run_starts, run_lengths = find_runs(hits)
        run_starts = run_starts[run_vals==True]
        run_lengths = run_lengths[run_vals==True]
        run_secs = run_lengths / raw_work.info["sfreq"]
        # all segments with the right time length
        spindle_inds = np.where((run_secs > spindle_lens[0]) &
                                (run_secs < spindle_lens[1]))[0]
        # calculate average, normalised power per spindle, peaks
        spindle_pows = []
        spindle_peaks = []
        for sp_idx in spindle_inds:
            seg = sig_rms[run_starts[sp_idx]:
                            run_starts[sp_idx] + run_lengths[sp_idx]]
            spindle_pows.append(((seg - sig_rms.mean()) / sig_rms.std()).mean())
            spindle_peaks.append(run_starts[sp_idx] + np.argmax(seg))
        # translate indices to times and mark in the annotations
        first_time = raw.first_samp / raw.info["sfreq"]
        spindle_starts = times[run_starts[spindle_inds]] + first_time
        spindle_peaks = times[spindle_peaks] + first_time

        peak_annots = mne.Annotations(spindle_peaks,
                                      np.zeros(len(spindle_peaks)),
                                      ["Spindle Peak" for x in spindle_peaks],
                                      orig_time=raw.annotations.orig_time)
        spind_annots = mne.Annotations(spindle_peaks + annot_len[0],
                                      np.ones(len(spindle_peaks))*annot_len[1]-annot_len[0],
                                      ["Spindle" for x in spindle_peaks],
                                      orig_time=raw.annotations.orig_time)
        

        annots = peak_annots + spind_annots
        annots.save(join(proc_dir, f"spindle_EPI_{subj}_{cond}-annot.fif"), overwrite=True)
