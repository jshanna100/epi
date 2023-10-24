import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.stats import pearsonr
import mne
import pandas as pd

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

def check_trough_annot(desc):
    # helper function for marking troughs of oscillations
    event = None
    if "trough" in desc:
        event = int(desc[-1])
    return event


def mark_osc_amp(osc_events, amp_thresh, chan_name, mm_times, osc_type):
    osc_idx = 0
    for oe in osc_events:
        pt_time_diff = oe.trough_time - oe.peak_time
        time_diff = oe.end_time - oe.start_time
        pt_amp_diff = oe.peak_amp - oe.trough_amp
        if pt_amp_diff > amp_thresh and mm_times[0] < time_diff < mm_times[1]:
            oe.event_id = "{} {} {}".format(chan_name, osc_type, osc_idx)
            oe.event_annot = f"{osc_type} {osc_idx}"
            osc_idx += 1

def mark_osc(desc, raw, chans, minmax_freq, minmax_time, min_samples=10, amp_percentile=65):
    # find and mark slow or delta oscillations
    raw_work = raw.copy()
    raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1])
    raw_work.get_data(picks=chans)
    signal = raw_work.get_data().mean(axis=0)

    # zero crossings
    df_dict = {"Subj":[],"Cond":[],"Index":[], "ROI":[],
                "OscType":[], "OscLen":[], "OscFreq":[]}
    
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
    zx_len = len(zero_x_inds)-2
    zx_range = range(neg_x0_ind, zx_len, 2)
    time0s = np.zeros(len(zx_range))
    time1s = np.zeros(len(zx_range)) 
    peak_times = np.zeros(len(zx_range)) 
    peak_amps = np.zeros(len(zx_range)) 
    trough_times = np.zeros(len(zx_range)) 
    trough_amps = np.zeros(len(zx_range))
    for idx, zx_ind in enumerate(zx_range):
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
        time0s[idx] = time0
        time1s[idx] = time1
        peak_times[idx] = peak_time
        peak_amps[idx] = peak_amp
        trough_times[idx] = trough_time
        trough_amps[idx] = trough_amp
    # get percentiles of peaks and troughs
    times = np.array(time1s) - np.array(time0s)
    amps = np.array(peak_amps) - np.array(trough_amps)
    amp_thresh = np.percentile(amps, amp_percentile)
    valid_inds = ((minmax_time[0] < times) & (times < minmax_time[1]) & (amps > amp_thresh))
    valid_inds = np.where(valid_inds)[0]
    new_annots = mne.Annotations(time0s[valid_inds[0]], times[valid_inds[0]], f"{desc} 0",
                                 orig_time=raw_work.annotations.orig_time)
    new_annots.append(peak_times[valid_inds[0]], 0, f"{desc} peak 0")
    new_annots.append(trough_times[valid_inds[0]], 0, f"{desc} trough 0")
    for abs_idx, idx in enumerate(valid_inds[1:]):
        new_annots.append(time0s[idx], times[idx], f"{desc} {abs_idx+1}")
        new_annots.append(peak_times[idx], 0, f"{desc} peak {abs_idx+1}")
        new_annots.append(trough_times[idx], 0, f"{desc} trough {abs_idx+1}")

    return new_annots
            
def mark_spindle(raw, chans, spindle_band, spindle_lens, annot_len, spindle_percentile=75,
                 moving_average=0.2):
    # detect and mark spindles
    raw_work = raw.copy()
    raw_work.filter(l_freq=spindle_band[0], h_freq=spindle_band[1],
                    verbose="warning")
    signal = raw_work.pick_channels(chans).get_data().mean(axis=0)
    

    envelope = abs(hilbert(signal))
    # moving average
    idx_200ms = raw.time_as_index(moving_average)[0]
    filt = np.ones(idx_200ms) / idx_200ms
    env_filter = np.convolve(envelope, filt)
    thresh = np.percentile(env_filter, spindle_percentile)


    # all segments between the spindle std range
    hits = env_filter > thresh
    # contiguous hits
    run_vals, run_starts, run_lengths = find_runs(hits)
    run_starts = run_starts[run_vals==True]
    run_lengths = run_lengths[run_vals==True]
    run_secs = run_lengths / raw_work.info["sfreq"]
    # all segments with the right time length
    spindle_inds = np.where((run_secs > spindle_lens[0]) &
                            (run_secs < spindle_lens[1]))[0]
    # calculate average, normalised power per spindle, peaks
    spindle_peaks = []
    for sp_idx in spindle_inds:
        seg = env_filter[run_starts[sp_idx]:
                         run_starts[sp_idx] + run_lengths[sp_idx]]
        spindle_peaks.append(run_starts[sp_idx] + np.argmax(seg))
    # translate indices to times and mark in the annotations
    first_time = raw.first_samp / raw.info["sfreq"]
    spindle_peaks = raw_work.times[spindle_peaks] + first_time

    peak_annots = mne.Annotations(spindle_peaks,
                                  np.zeros(len(spindle_peaks)),
                                  [f"Spindle Peak {idx}" for idx, x in enumerate(spindle_peaks)],
                                  orig_time=raw.annotations.orig_time)
    spind_annots = mne.Annotations(spindle_peaks + annot_len[0],
                                   np.ones(len(spindle_peaks))*annot_len[1]-annot_len[0],
                                   [f"Spindle {idx}" for idx, x in enumerate(spindle_peaks)],
                                    orig_time=raw.annotations.orig_time)
    

    annots = peak_annots + spind_annots
    return annots

def mark_ied(raw, chan, ied_band, ied_lens, ied_std=3.):
    # detect and mark IED artefacts
    raw_work = raw.copy()
    raw_work.filter(l_freq=ied_band[0], h_freq=ied_band[1])
    signal = raw_work.get_data(picks=[chan])[0,]
    envelope = abs(hilbert(signal))
    env_norm = (envelope - envelope.mean()) / envelope.std()

    # all segments between the spindle std range
    hits = env_norm > ied_std
    # contiguous hits
    run_vals, run_starts, run_lengths = find_runs(hits)
    run_starts = run_starts[run_vals==True]
    run_lengths = run_lengths[run_vals==True]
    run_secs = run_lengths / raw_work.info["sfreq"]
    # all segments with the right time length
    art_mask = (run_secs > ied_lens[0]) & (run_secs < ied_lens[1])

    ied_annots = mne.Annotations(raw.times[run_starts[art_mask]], run_secs[art_mask], 
                                 ["BAD_IED" for x in range(art_mask.sum())],
                                 orig_time=raw.annotations.orig_time)
    
    return ied_annots

def annot_within(annots, within_annots):
    # check if annotations are within another set of annotations
    annot_times = np.array([(a["onset"], a["onset"]+a["duration"]) for a in annots])
    within_annot_times = np.array([(a["onset"], a["onset"]+a["duration"]) for a in within_annots])
    match_annots, containing_annots = [], []
    for a_t_idx, a_t in enumerate(annot_times):
        hits = (a_t[0] > within_annot_times[:, 0]) & (a_t[0] < within_annot_times[:, 1])
        if sum(hits):
            hit_idx = np.where(hits)[0][0]
            match_annots.append(annots[a_t_idx])
            containing_annots.append(within_annots[hit_idx])
            

    return match_annots, containing_annots

def annot_overlap(annots_a, annots_b):
    # check if an annotation has any overlap with another
    annot_a_times = np.array([(a["onset"], a["onset"]+a["duration"]) for a in annots_a])
    annot_b_times = np.array([(a["onset"], a["onset"]+a["duration"]) for a in annots_b])
    overlap_inds = []
    for a_t_idx, a_t in enumerate(annot_a_times):
        hits_1 = (a_t[0] > annot_b_times[:, 0]) & (a_t[0] < annot_b_times[:, 1])
        hits_2 = (a_t[1] > annot_b_times[:, 0]) & (a_t[1] < annot_b_times[:, 1])
        hits = hits_1 | hits_2
        if sum(hits):
            overlap_inds.append(a_t_idx)
    return np.array(overlap_inds)

def hfb_power(raw, chan, bands):
    # calculate hfb power according to Hefrich
    band_powers = []
    raw_chan = raw.copy().pick_channels([chan])
    for band in bands:
        this_raw = raw_chan.copy().filter(l_freq=band[0], h_freq=band[1])
        signal = this_raw.get_data().mean(axis=0)
        envelope = abs(hilbert(signal))
        band_powers.append(envelope)
    amp = np.mean(band_powers, axis=0)
    return amp

def circ_linear_corr(rads, lins):
    # circular linear correlation
    rsin, _ = pearsonr(lins, np.sin(rads))
    rcos, _ = pearsonr(lins, np.cos(rads))
    rsc, _ = pearsonr(np.sin(rads), np.cos(rads))
    rho = np.sqrt((rcos**2 + rsin**2 - 2*rcos*rsin*rsc) / (1-rsc**2))
    return rho

def reassemble_annots(annot_list, orig_time=None):
    # put a list of annotations back into proper annotation form
    onsets, durations, descs = [], [], []
    for al in annot_list:
        onsets.append(al["onset"])
        durations.append(al["duration"])
        descs.append(al["description"])
    annotations = mne.Annotations(onsets, durations, descs, orig_time=orig_time)
    return annotations

def output_annot_csv(annotations, outpath):
    # output annotation info as a csv
    df_dict = {"Start":[], "End":[], "Description":[]}
    for annot in annotations:
        df_dict["Start"].append(annot["onset"])
        df_dict["End"].append(annot["onset"]+annot["duration"])
        df_dict["Description"].append(annot["description"])
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(outpath)