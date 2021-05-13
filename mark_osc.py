import mne
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
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

def check_trough_annot(desc):
    event_idx = 0
    if "Trough" in desc:
        event_idx = 10
    else:
        event_idx = None
    return event_idx


def osc_peaktroughs(osc_events):
    peaks = []
    troughs = []
    for oe in osc_events:
        peaks.append(oe.peak_amp)
        troughs.append(oe.trough_amp)
    peaks, troughs = np.array(peaks), np.array(troughs)
    return peaks, troughs

def mark_osc_amp(osc_events, amp_thresh, mm_times, osc_type,
                 raw_inst=None):
    osc_idx = 0
    for oe in osc_events:
        pt_time_diff = oe.trough_time - oe.peak_time
        time_diff = oe.end_time - oe.start_time
        pt_amp_diff = oe.peak_amp - oe.trough_amp
        if pt_amp_diff > amp_thresh and mm_times[0] < time_diff < mm_times[1]:
            oe.event_id = "{} {}".format(osc_type, osc_idx)
            osc_idx += 1

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

amp_percentile = 75
min_samples = 10
minmax_freqs = [(0.16, 1.25)]
minmax_times = [(0.8, 2)]
osc_types = ["SO"]
amp_thresh_dict = {"Subj":[], "OscType":[], "Thresh":[]}

eeg_epos = []
ecog_epos = []
for filename in filelist:
    this_match = re.match("cut_csel_c_EPI_(.*)_(.*)-raw.fif", filename)
    if this_match:
        df_dict = {"Subj":[], "ID":[], "OscType":[]}
        subj, file_id = this_match.group(1), this_match.group(2)
        raw = mne.io.Raw(proc_dir+filename,preload=True)

        for minmax_freq, minmax_time, osc_type in zip(minmax_freqs,
                                                      minmax_times,
                                                      osc_types):
            raw_work = raw.copy()
            raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1])
            first_time = raw_work.first_samp / raw_work.info["sfreq"]


            # zero crossings
            pick_ind = mne.pick_channels(raw_work.ch_names, include=["central"])
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
            for zx_ind in range(neg_x0_ind, len(zero_x_inds)-2, 2):
                idx0 = zero_x_inds[zx_ind]
                idx1 = zero_x_inds[zx_ind+1]
                idx2 = zero_x_inds[zx_ind+2]
                if (idx1 - idx0) < min_samples or (idx2 - idx1) < min_samples:
                    continue
                time0 = raw_work.first_time + raw_work.times[idx0]
                time1 = raw_work.first_time + raw_work.times[idx2]
                peak_time_idx = np.min(find_peaks(signal[idx1:idx2])[0]) + idx1
                trough_time_idx = np.argmin(signal[idx0:idx1]) + idx0
                peak_amp, trough_amp = signal[peak_time_idx], signal[trough_time_idx]
                peak_time = raw_work.first_time + raw_work.times[peak_time_idx]
                trough_time = raw_work.first_time + raw_work.times[trough_time_idx]
                osc_events.append(OscEvent(time0, time1, peak_time,
                                           peak_amp, trough_time, trough_amp))
            # get percentiles of peaks and troughs
            osc_events = [oe for oe in osc_events if (oe.end_time-oe.start_time)>minmax_time[0] and (oe.end_time-oe.start_time)<minmax_time[1]]
            peaks, troughs = osc_peaktroughs(osc_events)
            amps = peaks - troughs
            amp_thresh = np.percentile(amps, amp_percentile)
            amp_thresh_dict["Subj"].append(subj)
            amp_thresh_dict["OscType"].append(osc_type)
            amp_thresh_dict["Thresh"].append(amp_thresh)
            mark_osc_amp(osc_events, amp_thresh, minmax_time, osc_type,
                         raw_inst=raw_work)
            marked_oe = [oe for oe in osc_events if oe.event_id is not None]
            if len(marked_oe):
                for moe_idx, moe in enumerate(marked_oe):
                    if moe_idx == 0:
                        new_annots = mne.Annotations(moe.start_time,
                                                     moe.end_time-moe.start_time,
                                                     "{} {}".format(moe.event_id, moe.event_annot),
                                                     orig_time=raw.annotations.orig_time)
                    else:
                        new_annots.append(moe.start_time, moe.end_time-moe.start_time,
                                          "{} {}".format(moe.event_id, moe.event_annot))
                    new_annots.append(moe.trough_time, 0,
                                      "Trough {} {}".format(moe.event_id, moe.event_annot))
                    new_annots.append(moe.peak_time, 0,
                                      "Peak {} {}".format(moe.event_id, moe.event_annot))

                    df_dict["Subj"].append(subj)
                    df_dict["ID"].append(file_id)
                    df_dict["OscType"].append(osc_type)

                new_annots.save("{}osc_{}_{}-annot.fif".format(proc_dir, subj, file_id))
                raw.set_annotations(new_annots)
            else:
                print("\nNo oscillations found. Skipping.\n")
                continue

            raw.save("{}osc_{}_{}-raw.fif".format(proc_dir,subj, file_id), overwrite=True)
            df = pd.DataFrame.from_dict(df_dict)
            events, event_ids = mne.events_from_annotations(raw, check_trough_annot)
            epo = mne.Epochs(raw, events, tmin=-2, tmax=2, baseline=None,
                             metadata=df, preload=True)
            epo.pick_channels(["central"])
            epo.save("{}osc_{}_{}-epo.fif".format(proc_dir,subj, file_id), overwrite=True)
            eeg_epos.append(epo)

            raw_ecog = mne.io.Raw("{}ecog_EPI_{}_{}-raw.fif".format(proc_dir,
                                                                    subj,
                                                                    file_id))
            raw_ecog.set_annotations(new_annots)
            events, event_ids = mne.events_from_annotations(raw_ecog, check_trough_annot)
            ecog_epo = mne.Epochs(raw_ecog, events, tmin=-2, tmax=2, baseline=None,
                                  metadata=df, preload=True)
            ecog_epo.save("{}osc_{}_{}-epo.fif".format(proc_dir,subj, file_id),
                          overwrite=True)
            ecog_epos.append(ecog_epo)

grand_eeg = mne.concatenate_epochs(eeg_epos)
grand_ecog = mne.concatenate_epochs(ecog_epos)

grand_eeg.save("{}grand_eeg-epo.fif".format(proc_dir), overwrite=True)
grand_ecog.save("{}grand_ecog-epo.fif".format(proc_dir), overwrite=True)
