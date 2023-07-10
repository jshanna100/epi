import mne
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from os.path import join
plt.ion()

class OscEvents():
    def __init__(self):
        self.start_times = []
        self.end_times = []
        self.peak_times = []
        self.peak_amps = []
        self.trough_times = []
        self.trough_amps = []
        self.event_ids = []
    def append(self, time0, time1, peak_time, peak_amp, trough_time, trough_amp):
        self.start_times.append(time0)
        self.end_times.append(time1)
        self.peak_times.append(peak_time)
        self.peak_amps.append(peak_amp)
        self.trough_times.append(trough_time)
        self.trough_amps.append(trough_amp)

def mark_osc(osc_events, amp_thresh, mm_time):
    osc_idx = 0
    for (trough_time, peak_time, end_time, start_time,
         peak_amp, trough_amp) in zip(osc_events.trough_times, osc_events.peak_times,
                                      osc_events.end_times, osc_events.start_times,
                                      osc_events.peak_amps, osc_events.trough_amps):
        pt_time_diff = trough_time - peak_time
        time_diff = end_time - start_time
        pt_amp_diff = peak_amp - trough_amp
        if (pt_amp_diff > amp_thresh) and (mm_time[0] < time_diff < mm_time[1]):
            osc_events.event_ids.append(f"SO {osc_idx}")
            osc_idx += 1
        else:
            osc_events.event_ids.append(None)

def check_down_annot(desc):
    event_idx = 0
    if "Down" in desc:
        event_idx = 10
    else:
        event_idx = None
    return event_idx

def is_range_in_annot(t_range, in_annot):
    for in_an in in_annot:
        a_t_range = np.array([in_an["onset"], in_an["onset"]+in_an["duration"]])
        if t_range[0] <= a_t_range[1] and t_range[1] >= a_t_range[0]:
            return True
    return False

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

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

amp_percentile = 75
min_samples = 10
minmax_freq = (0.16, 1.25)
minmax_time = (0.8, 2)
osc_type = ["SO"]
chans = ["Fz", "Cz"]

subjs = ["1001", "1002"]
conds = ["Stim", "Sham"] # no point in doing this to sham conditions

for subj in subjs:
    for cond in conds:
        art_annots = mne.read_annotations(join(proc_dir,
                                               f"art_EPI_{subj}_{cond}-annot.fif"))
        if cond == "Stim":
            stim_annots = mne.read_annotations(join(proc_dir,
                                                   f"stim_EPI_{subj}_{cond}-annot.fif"))
        raw = mne.io.Raw(join(proc_dir, f"f_EPI_{subj}_{cond}-raw.fif"),
                         preload=True)

        this_chan = chans[0] if chans[0] not in raw.info["bads"] else chans[1]
        raw_work = raw.copy().pick_channels([this_chan])
        raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1])
        first_time = raw_work.first_samp / raw_work.info["sfreq"]

        print("Identifying slow oscillations")
        # zero crossings
        pick_ind = mne.pick_channels(raw_work.ch_names, include=[this_chan])
        if cond == "Stim":
            raw_work.set_annotations(stim_annots)
        signal = raw_work.get_data(reject_by_annotation="Nan")[pick_ind,]
        signal = signal.squeeze()
        signal[np.isnan(signal)] = 0

        # need to add infinitesimals to zeros to prevent weird x-crossing bugs
        for null_idx in list(np.where(signal==0)[0]):
            if null_idx:
                signal[null_idx] = 1e-16*np.sign(signal[null_idx-1])
            else:
                signal[null_idx] = 1e-16*np.sign(signal[null_idx+1])

        zero_x_inds = (np.where((signal[:-1] * signal[1:]) < 0)[0]) + 1
        # cycle through negative crossings
        neg_x0_ind = 1 if signal[0] < 0 else 2
        osc_events = OscEvents()
        first_time = raw_work.first_time
        times = raw_work.times
        for zx_ind in range(neg_x0_ind, len(zero_x_inds)-2, 2):
            idx0 = zero_x_inds[zx_ind]
            idx1 = zero_x_inds[zx_ind+1]
            idx2 = zero_x_inds[zx_ind+2]
            if (idx1 - idx0) < min_samples or (idx2 - idx1) < min_samples:
                continue
            time0 = first_time + times[idx0]
            time1 = first_time + times[idx2]
            peak_time_idx = np.argmax(signal[idx1:idx2]) + idx1
            trough_time_idx = np.argmin(signal[idx0:idx1]) + idx0
            peak_amp, trough_amp = signal[peak_time_idx], signal[trough_time_idx]
            peak_time = first_time + times[peak_time_idx]
            trough_time = first_time + times[trough_time_idx]
            osc_events.append(time0, time1, peak_time, peak_amp, trough_time,
                              trough_amp)
        # get percentiles of peaks and troughs
        peaks, troughs = np.array(osc_events.peak_amps), np.array(osc_events.trough_amps)
        amps = peaks - troughs
        amp_thresh = np.percentile(amps, amp_percentile)
        so_annots = mne.Annotations([], [], [],
                                     orig_time=raw.annotations.orig_time)
        for (start_time, end_time,
             peak_time, trough_time,
             event_id) in zip(osc_events.start_times, osc_events.end_times,
                               osc_events.peak_times, osc_events.trough_times,
                               osc_events.event_ids):
            if not event_id:
                continue
            # check if this overlaps with an artefact
            t_range = np.array([start_time, end_time])
            if is_range_in_annot(t_range, art_annots):
                continue
            so_annots.append(start_time, end_time-start_time,
                              f"{event_id}")
            so_annots.append(trough_time, 0,
                              f"Down_Spitz {event_id}")
            so_annots.append(peak_time, 0,
                              f"Up_Spitz {event_id}")


        so_annots.save(join(proc_dir,
                            f"osc_{subj}_{cond}-annot.fif"),
                             overwrite=True)

        # spindle detection
        print("Spindles")
        spindle_band = [12, 16]
        spindle_std = [1.5, np.inf]
        spindle_lens = [0.4, 3]

        raw_work = raw.copy()
        raw_work.pick_channels([this_chan])
        raw_work.filter(l_freq=spindle_band[0], h_freq=spindle_band[1],
                        verbose="warning")
        if cond == "Stim":
            raw_work.set_annotations(stim_annots)
        signal = raw_work.get_data(reject_by_annotation="Nan")[0]
        signal[np.isnan(signal)] = 0
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
        spindle_lens = run_secs[spindle_inds]
        spindle_peaks = times[spindle_peaks] + first_time

        spind_annots = mne.Annotations(spindle_starts, spindle_lens,
                                       ["Spindle" for x in spindle_starts],
                                       orig_time=raw.annotations.orig_time)
        peak_annots = mne.Annotations(spindle_peaks,
                                      np.zeros(len(spindle_peaks)),
                                      ["Spindle Peak" for x in spindle_peaks],
                                       orig_time=raw.annotations.orig_time)
        spind_annots += peak_annots
        breakpoint()



        # raw.set_annotations(so_annots)
        #
        # events = mne.events_from_annotations(raw, check_down_annot)
        # epo = mne.Epochs(raw, events[0], tmin=-2.5, tmax=2.5,
        #                      baseline=None, metadata=df,
        #                      preload=True)
        # epo.save(join(proc_dir, f"{subj}_{cond}_osc-epo.fif"),
        #          overwrite=True)
