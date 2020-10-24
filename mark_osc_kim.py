import mne
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from os.path import isdir
plt.ion()

class OscEvent():
    def __init__(self, start_time, end_time, peak_time, peak_amp, trough_time,
                 trough_amp):
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_amp = peak_amp
        self.trough_time = trough_time
        self.trough_amp = trough_amp
        self.event_id = None
        self.event_annot = None

def get_annotation(annotations, time):
    period = None
    for annot in annotations:
        if "Pre" not in annot["description"] and "Post" not in annot["description"]:
            continue
        begin = annot["onset"]
        end = begin + annot["duration"]
        if time > begin and time < end:
            period = annot["description"]
    return period

def check_trough_annot(desc):
    event = None
    if "Trough" in desc:
        event = 0
        if "deltO" in desc:
            event += 100
        if "Post" in desc:
            event += 50
        event += int(desc[-1])
    return event

def osc_peaktroughs(osc_events):
    peaks = []
    troughs = []
    for oe in osc_events:
        peaks.append(oe.peak_amp)
        troughs.append(oe.trough_amp)
    peaks, troughs = np.array(peaks), np.array(troughs)
    return peaks, troughs

def mark_osc(osc_events, peak_thresh, trough_thresh, sig_name, raw_inst=None):
    SO_idx = 0
    deltO_idx = 0
    for oe in osc_events:
        if raw_inst is not None:
            event_annot = get_annotation(raw_inst.annotations,
                                         oe.start_time)
            if event_annot is None:
                continue
        else:
            event_annot = None
        pt_time_diff = oe.trough_time - oe.peak_time
        if (oe.trough_amp < trough_thresh and oe.peak_amp > peak_thresh and
            0.15 < pt_time_diff < 0.5): # is a slow oscillation
            oe.event_id = "{} SO {}".format(sig_name, SO_idx)
            oe.event_annot = event_annot
            SO_idx += 1
        elif (oe.trough_amp < trough_thresh and oe.peak_amp < peak_thresh and
              pt_time_diff < 0.5): # is a delta oscillation
            oe.event_id = "{} deltO {}".format(sig_name, deltO_idx)
            oe.event_annot = event_annot
            deltO_idx += 1

def mark_osc_amp(osc_events, amp_thresh, trough_thresh, sig_name, raw_inst=None):
    SO_idx = 0
    deltO_idx = 0
    for oe in osc_events:
        if raw_inst is not None:
            event_annot = get_annotation(raw_inst.annotations,
                                         oe.start_time)
            if event_annot is None:
                continue
        else:
            event_annot = None
        pt_time_diff = oe.trough_time - oe.peak_time
        pt_amp_diff = oe.peak_amp - oe.trough_amp
        if (pt_amp_diff > amp_thresh and oe.trough_amp < trough_thresh and
            0.15 < pt_time_diff < 0.5): # is a slow oscillation
            oe.event_id = "{} SO {}".format(sig_name, SO_idx)
            oe.event_annot = event_annot
            SO_idx += 1
        elif (pt_amp_diff < amp_thresh and oe.trough_amp < trough_thresh and
              pt_time_diff < 0.5): # is a delta oscillation
            oe.event_id = "{} deltO {}".format(sig_name, deltO_idx)
            oe.event_annot = event_annot
            deltO_idx += 1

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
chan_groups = {"central":["Fz","FC1","FC2","Cz","CP1","CP2","Pz"]}
peak_percentile = 80
amp_percentile = 80
trough_percentile = 40
min_samples = 30
l_freq, h_freq = 0.3, 3

subjs = ["1001"]
conds = ["Stim"]

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw("{}ibcaof_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         preload=True)
        # produce channel-ROI averages
        for k,v in chan_groups.items():
            raw_pick = raw.copy().pick_channels(v)
            avg_signal = raw_pick.get_data().mean(axis=0, keepdims=True)
            avg_info = mne.create_info([k], raw.info["sfreq"], ch_types="eeg")
            avg_raw = mne.io.RawArray(avg_signal, avg_info)
            raw.add_channels([avg_raw], force_update_info=True)
        raw_work = raw.copy()
        raw_work.filter(l_freq=l_freq, h_freq=h_freq)
        first_time = raw_work.first_samp / raw_work.info["sfreq"]

        # zero crossings
        zero_xs = {}
        pick_annots = []
        for k in chan_groups.keys():
            pick_ind = mne.pick_channels(raw_work.ch_names, include=[k])
            signal = raw_work.get_data()[pick_ind,].squeeze()

            # need to add infinitesimals to zeros to prevent weird x-crossing bugs
            for null_idx in list(np.where(signal==0)[0]):
                if null_idx:
                    signal[null_idx] = 1e-16*np.sign(signal[null_idx-1])
                else:
                    signal[null_idx] = 1e-16*np.sign(signal[null_idx+1])

            zero_x_inds = (np.where((signal[:-1] * signal[1:]) < 0)[0]) + 1
            # cycle through negative crossings
            neg_x0_ind = 1 if signal[0] < 0 else 2
            osc_events = []
            for zx_ind in range(neg_x0_ind, len(zero_x_inds)-1, 2):
                idx0 = zero_x_inds[zx_ind-1]
                idx1 = zero_x_inds[zx_ind]
                idx2 = zero_x_inds[zx_ind+1]
                if (idx1 - idx0) < min_samples or (idx2 - idx1) < min_samples:
                    continue
                time0 = raw_work.first_time + raw_work.times[idx0]
                time1 = raw_work.first_time + raw_work.times[idx2]
                peak_time_idx = np.max(find_peaks(signal[idx0:idx1])[0]) + idx0
                trough_time_idx = np.argmin(signal[idx1:idx2]) + idx1
                peak_amp, trough_amp = signal[peak_time_idx], signal[trough_time_idx]
                peak_time = raw_work.first_time + raw_work.times[peak_time_idx]
                trough_time = raw_work.first_time + raw_work.times[trough_time_idx]
                osc_events.append(OscEvent(time0, time1, peak_time,
                                           peak_amp, trough_time, trough_amp))
            # get percentiles of peaks and troughs
            peaks, troughs = osc_peaktroughs(osc_events)
            amps = peaks - troughs
            amp_thresh = np.percentile(amps, amp_percentile)
            peak_thresh = np.percentile(peaks, peak_percentile)
            trough_thresh = np.percentile(troughs, trough_percentile)

            mark_osc_amp(osc_events, amp_thresh, trough_thresh, k, raw_inst=raw_work)
            marked_oe = [oe for oe in osc_events if oe.event_id is not None]
            for moe_idx, moe in enumerate(marked_oe):
                if moe_idx == 0:
                    new_annots = mne.Annotations(moe.start_time,
                                                 moe.end_time-moe.start_time,
                                                 "{} {}".format(moe.event_id, moe.event_annot),
                                                 orig_time=raw_work.annotations.orig_time)
                else:
                    new_annots.append(moe.start_time, moe.end_time-moe.start_time,
                                      "{} {}".format(moe.event_id, moe.event_annot))
                new_annots.append(moe.trough_time, 0,
                                  "Trough {} {}".format(moe.event_id, moe.event_annot))
                new_annots.append(moe.peak_time, 0,
                                  "Peak {} {}".format(moe.event_id, moe.event_annot))

            pick_annots.append(new_annots)
        all_annots = pick_annots[0]
        for annot in pick_annots[1:]:
            all_annots += annot
        raw.set_annotations(all_annots)
        raw.save("{}aibcaof_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                 overwrite=True)
        events = mne.events_from_annotations(raw, check_trough_annot)
        df_dict = {"Subj":[],"Cond":[],"Ort":[],"OscType":[],"Index":[],
                   "PrePost":[]}
        for event in np.nditer(events[0][:,-1]):
            eve = event.copy()
            if eve >= 200:
                df_dict["Ort"].append("posterior")
                eve -= 200
            else:
                df_dict["Ort"].append("central")
            if eve >= 100:
                df_dict["OscType"].append("deltO")
                eve -= 100
            else:
                df_dict["OscType"].append("SO")
            if eve >= 50:
                df_dict["PrePost"].append("Post")
                eve -= 50
            else:
                df_dict["PrePost"].append("Pre")
            df_dict["Index"].append(int(eve))
            df_dict["Subj"].append(subj)
            df_dict["Cond"].append(cond)

        print("Trough threshold: {}, Amplitude threshold: {}".format(amp_thresh, trough_thresh))
        df = pd.DataFrame.from_dict(df_dict)
        epo = mne.Epochs(raw, events[0], tmin=-1.25, tmax=1.25, detrend=None,
                         baseline=(-1.25,-0.75), metadata=df, event_repeated="drop").load_data()
        epo.save("{}epi_{}_{}-epo.fif".format(proc_dir,subj,cond), overwrite=True)
