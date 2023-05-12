import mne
import numpy as np
import pickle
from os import listdir
import re
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
plt.ion()

def dict_to_array(in_dict, factors):
    X = []
    for factor in factors:
        if factor[0] == "freqs":
            for band in factor[2]:
                X.append(np.squeeze(in_dict["freqs"][factor[1]][band]))
        else:
            X.append(in_dict["amps"][factor[1]])
    X = np.array(X).T
    return X

def sub_channels(raw, picks, name, ch_type="eeg"):
    if len(picks) != 2:
        raise ValueError("Can only take two channels for subtraction")
    to_sub = []
    for pick in picks:
        if type(pick) is list:
            sub0 = raw.copy().pick_channels(pick).get_data()
            to_sub.append(np.mean(sub0, axis=0))
        else:
            to_sub.append(raw.copy().pick_channels([pick]).get_data())
    new_signal = to_sub[0] - to_sub[1]
    new_info = mne.create_info([name], raw.info["sfreq"], ch_types=ch_type)
    new_raw = mne.io.RawArray(new_signal, new_info)
    raw.add_channels([new_raw], force_update_info=True)


def power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    eeg_picks = mne.pick_types(epo.info, eeg=True)
    eeg_ch_names = [epo.ch_names[idx] for idx in np.nditer(eeg_picks)]
    psds, freqs = psd_welch(epochs, picks=eeg_picks, fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = {ch_name:{band_name:[] for band_name in FREQ_BANDS.keys()}
         for ch_name in eeg_ch_names}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        freq_inds = (freqs >= fmin) & (freqs < fmax)
        for ch_idx, ch_name in enumerate(eeg_ch_names):
            psds_band = psds[:, ch_idx, freq_inds].mean(axis=-1)
            X[ch_name][band_name] = psds_band.reshape(len(psds), -1)
    return X

def epo_amps(epo):
    chans = ["HEOG"]
    picks = mne.pick_channels(epo.ch_names, chans)
    X = {}
    data = epo.load_data().get_data()
    for ch, pick in zip(chans, np.nditer(picks)):
        X[ch] = np.sqrt(np.sum(data[:,pick,]**2, axis=-1))
    return X

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"/proc/"
filelist = listdir(proc_dir)
bands = ["delta", "theta", "alpha", "sigma", "beta"]
factors = [["freqs","Fpz - Cz",bands], ["freqs","Pz - Oz", bands],
           ["amps",'HEOG']]

with open("{}sleep_stage_classifier.pickle".format(proc_dir), "rb") as f:
    xifer = pickle.load(f)

for filename in filelist:
    this_match = re.match("csel_c_EPI_(.*)-raw.fif",filename)
    if this_match:
        file_id = this_match.group(1)
        raw = mne.io.Raw(proc_dir+filename, preload=True)
        sub_channels(raw, [["Fp1", "Fp2"], "Cz"], "Fpz - Cz")
        sub_channels(raw, ["Pz", ["O2", "O1"]], "Pz - Oz")
        raw.pick_channels(["Fpz - Cz", "Pz - Oz", "HEOG"])
        events = mne.make_fixed_length_events(raw, duration=30)
        epo = mne.Epochs(raw,events, tmin=0, tmax=30, baseline=None).load_data()
        events = epo.events
        freqs = power_band(epo)
        amps = epo_amps(epo)
        this_mat = {"freqs":freqs, "amps":amps}
        features = dict_to_array(this_mat, factors)
        event_ids = xifer.predict(features)


        current_stage = None
        begin = 0
        for ev_idx in range(len(events)):
            if event_ids[ev_idx] == current_stage:
                continue
            else:
                current_stage = event_ids[ev_idx]
                end = events[ev_idx,0] - 1
        events[:,-1] = event_ids
        mne.viz.plot_events(events, sfreq=raw.info["sfreq"],
                            first_samp=events[0,0])
        plt.title(filename)
