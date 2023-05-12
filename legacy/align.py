import mne
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 28}
matplotlib.rc('font', **font)

def _shift_signals(sig, n_shifts, fill_with=0):
        """Shift an array of signals according to an array of delays.

        Parameters
        ----------
        sig : array_like
            Array of signals of shape (n_freq, n_trials, n_times)
        n_shifts : array_like
            Array of delays to apply to each trial of shape (n_trials,)
        fill_with : int
            Value to prepend / append to each shifted time-series

        Returns
        -------
        sig_shifted : array_like
            Array of shifted signals with the same shape as the input
        """
        # prepare the needed variables
        n_freqs, n_trials, n_pts = sig.shape
        sig_shifted = np.zeros_like(sig)
        # shift each trial
        for tr in range(n_trials):
            # select the data of a specific trial
            st_shift = n_shifts[tr]
            st_sig = sig[:, tr, :]
            fill = np.full((n_freqs, abs(st_shift)), fill_with,
                           dtype=st_sig.dtype)
            # shift this specific trial
            if st_shift > 0:   # move forward = prepend zeros
                sig_shifted[:, tr, :] = np.c_[fill, st_sig][:, 0:-st_shift]
            elif st_shift < 0:  # move backward = append zeros
                sig_shifted[:, tr, :] = np.c_[st_sig, fill][:, abs(st_shift):]

        return sig_shifted

def _peak_detection(pha, cue):
        """Single trial closest to a cue peak detection.

        Parameters
        ----------
        pha : array_like
            Array of single trial phases of shape (n_trials, n_times)
        cue : int
            Cue to use as a reference (in sample unit)

        Returns
        -------
        peaks : array_like
            Array of length (n_trials,) describing each delay to apply
            to each trial in order to realign the phases. In detail :

                * Positive delays means that zeros should be prepend
                * Negative delays means that zeros should be append
        """
        n_trials, n_times = pha.shape
        peaks = []
        for tr in range(n_trials):
            # select the single trial phase
            st_pha = pha[tr, :]
            # detect all peaks across time points
            st_peaks = []
            for t in range(n_times - 1):
                if (st_pha[t - 1] < st_pha[t]) and (st_pha[t] > st_pha[t + 1]):
                    st_peaks += [t]
            # detect the minimum peak
            min_peak = st_peaks[np.abs(np.array(st_peaks) - cue).argmin()]
            peaks += [cue - min_peak]

        return np.array(peaks)

def match_times(times, times_dest):
    inds = np.zeros_like(times_dest, dtype=int)
    for nt_idx, nt in enumerate(np.nditer(times_dest)):
        inds[nt_idx] = np.abs(times - nt).argmin()
    return inds

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"proc/"
spindle_cue = -.5
spindle_freqs = (12, 16)
ripple_freqs = (50, 80)
SO_freqs = (0.5, 1.25)
modality = "ecog"
bl_mode = "mean"

if modality == "ecog":
    chan = "H"
    filename = "{}ecog_grand_{}-epo.fif".format(proc_dir, chan)
elif modality == "eeg":
    chan = "central"
    filename = "{}eeg_grand_{}-epo.fif".format(proc_dir, chan)

# load
grand = mne.read_epochs(filename)

# filter data and extract arrays
SO_filt = grand.copy().filter(l_freq=SO_freqs[0], h_freq=SO_freqs[1],
                              n_jobs=4)
SO_filt.crop(tmin=-1.5, tmax=1.5)
data_SO = SO_filt.get_data()[:,0,] * 1e+6
spindle_filt = grand.copy().filter(l_freq=spindle_freqs[0],
                                   h_freq=spindle_freqs[1], n_jobs=4)
spindle_filt.crop(tmin=-1.5, tmax=1.5)
data_spindle = spindle_filt.get_data()[:,0,] * 1e+6

# get delays that will align phases in spindle abnd
complex_tfr = tfr_morlet(grand,
                         np.linspace(spindle_freqs[0], spindle_freqs[1], 50),
                         5, return_itc=False, output="complex", average=False,
                         n_jobs=4)
phase_data = np.squeeze(np.angle(complex_tfr.data).mean(axis=2))
cue_idx = np.abs(grand.times - spindle_cue).argmin() - 1
peaks = _peak_detection(phase_data, cue_idx)

shift_SO = _shift_signals(np.expand_dims(data_SO, 0), peaks)[0,]
shift_spindle = _shift_signals(np.expand_dims(data_spindle, 0), peaks)[0,]


#tfr on data and shifted data
g_data = grand.get_data().copy()
for ch_idx in range(g_data.shape[1]):
    shifted = _shift_signals(np.expand_dims(g_data[:,ch_idx,],0), peaks)
    g_data[:,ch_idx,] = shifted
grand_shifted = mne.EpochsArray(g_data, grand.info, tmin=grand.times[0])
ripple_tfr = tfr_morlet(grand, np.arange(ripple_freqs[0],ripple_freqs[1]),
                        n_cycles=12, average=False, return_itc=False)
ripple_shifted_tfr = tfr_morlet(grand_shifted,
                                np.arange(ripple_freqs[0],ripple_freqs[1]),
                                n_cycles=7, average=False, return_itc=False)

# average, baseline, crop
bl = (-2.4, -1.8)
crop = (-1.5, 1.5)
ripple_tfr_avg = ripple_tfr.average()
ripple_shifted_tfr_avg = ripple_shifted_tfr.average()
ripple_tfr_avg.apply_baseline(bl, mode=bl_mode)
ripple_shifted_tfr_avg.apply_baseline(bl, mode=bl_mode)
ripple_tfr_avg.crop(tmin=crop[0], tmax=crop[1])
ripple_shifted_tfr_avg.crop(tmin=crop[0], tmax=crop[1])


## plot
vmin, vmax = -4, 4
vmin, vmax = None, None
fig, axes = plt.subplots(2,2, figsize=(21.6, 21.6))
titler_loc = (0, 0.93)
titler_fontsize = 48

# Spindle TFR
tfr_freqs = (10,20)
spindle_tfr = tfr_morlet(grand, np.linspace(tfr_freqs[0], tfr_freqs[1], 50),
                         9, return_itc=False, n_jobs=4)
spindle_tfr.apply_baseline((-2.4, -1.5), mode=bl_mode)
spindle_tfr.crop(tmin=-1.5, tmax=1.5)
spindle_tfr.plot(axes=axes[0][0], colorbar=False)

# SO overlay on spindle TFR
SO_overlay = data_SO.mean(axis=0)
s_min, s_max = SO_overlay.min(), SO_overlay.max()
SO_overlay = (SO_overlay - s_min) / (s_max - s_min) - 0.5
band_breit = tfr_freqs[1] - tfr_freqs[0]
SO_overlay = (SO_overlay * 5 + tfr_freqs[0] + band_breit/2)
axes[0][0].plot(spindle_tfr.times, SO_overlay, alpha=0.5, color="black",
                linewidth=3)
axes[0][0].set_xlabel("")
axes[0][0].set_title("Original, non-aligned data (A-B)\n\nSpindle band")
axes[0][0].text(titler_loc[0], titler_loc[1], "A",
                transform=axes[0][0].transAxes, fontsize=titler_fontsize)

# ripple TFR
ripple_tfr_avg.plot(picks=chan, axes=axes[1][0], colorbar=False,
                    vmin=vmin, vmax=vmax, combine="mean")
# SO overlay on ripple TFR
SO_overlay = data_SO.mean(axis=0)
s_min, s_max = SO_overlay.min(), SO_overlay.max()
SO_overlay = (SO_overlay - s_min) / (s_max - s_min) - 0.5
band_breit = ripple_freqs[1] - ripple_freqs[0]
SO_overlay = (SO_overlay * 15 + ripple_freqs[0] + band_breit/2)
axes[1][0].plot(ripple_tfr_avg.times, SO_overlay, alpha=0.5, color="black",
                linewidth=3)
axes[1][0].set_title("Ripple band")
axes[1][0].text(titler_loc[0], titler_loc[1], "B",
                transform=axes[1][0].transAxes, fontsize=titler_fontsize)

# aligned SO waveform
axes[0][1].plot(SO_filt.times, shift_SO.mean(axis=0)+shift_spindle.mean(axis=0),
                color="black")
rect = matplotlib.patches.Rectangle((-.8, -70), .6, 60, facecolor="none",
                                    edgecolor="black")
axes[0][1].add_patch(rect)
axes[0][1].text(-.8, -20, "D")
axes[0][1].set_yticklabels([])
axes[0][1].set_title("Aligned to spindle band phase (C-D)\n\nAligned Slow Oscillation")
axes[0][1].text(titler_loc[0], titler_loc[1], "C",
                transform=axes[0][1].transAxes, fontsize=titler_fontsize)

# aligned ripple tfr
ripple_shifted_tfr_avg.crop(tmin=-.8, tmax=-.2)
ripple_shifted_tfr_avg.plot(picks=chan, axes=axes[1][1], colorbar=False,
                            vmin=vmin, vmax=vmax, combine="mean")

spindle_inds = match_times(spindle_filt.times, ripple_shifted_tfr_avg.times)
spindle_overlay = shift_spindle.mean(axis=0)[spindle_inds]
spindle_times = spindle_filt.times[spindle_inds]
s_min, s_max = spindle_overlay.min(), spindle_overlay.max()
spindle_overlay = (spindle_overlay - s_min) / (s_max - s_min) - 0.5
band_breit = ripple_freqs[1]-ripple_freqs[0]
spindle_overlay = (spindle_overlay * 25 + ripple_freqs[0] +
                   band_breit/2)

axes[1][1].plot(spindle_times, spindle_overlay, alpha=0.5, color="black",
                linewidth=3)
axes[1][1].set_title("Ripple band at spindle maximum")
axes[1][1].text(titler_loc[0], titler_loc[1], "D",
                transform=axes[1][1].transAxes, fontsize=titler_fontsize)

plt.suptitle("Hippocampal Electrocorticogram")

fig.savefig("../aligned_tfr.png")
fig.savefig("../aligned_tfr.eps")
