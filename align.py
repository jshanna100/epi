import mne
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import tensorpac
plt.ion()

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

root_dir = "/home/jev/hdd/epi/"
proc_dir = root_dir+"proc/"
spindle_cue = -0.45
spindle_cue = 0.45

grand_eeg = mne.read_epochs("{}grand_eeg-epo.fif".format(proc_dir))
grand_ecog = mne.read_epochs("{}grand_ecog-epo.fif".format(proc_dir))
grand_eeg.resample(grand_ecog.info["sfreq"])

SO_filt = grand_eeg.copy().filter(l_freq=0.16, h_freq=1.25)
data_SO = SO_filt.get_data()[:,0,] * 1e+6

spindle_filt = grand_eeg.copy().filter(l_freq=14, h_freq=18, n_jobs=4)
data_spindle = spindle_filt.get_data()[:,0,] * 1e+6

cue_idx = np.abs(grand_eeg.times - spindle_cue).argmin() - 1
peaks = _peak_detection(data_spindle, cue_idx)

shift_SO = _shift_signals(np.expand_dims(data_SO, 0), peaks)[0,]
shift_spindle = _shift_signals(np.expand_dims(data_spindle, 0), peaks)[0,]
plt.plot(grand_eeg.times, shift_SO.mean(axis=0)+shift_spindle.mean(axis=0))

# ripple_tfr = tfr_morlet(grand_ecog, np.arange(80,140), n_cycles=9, return_itc=False)
# ripple_tfr.crop(tmin=-.4, tmax=.4)
data_ripple = grand_ecog.get_data() * 1e+6
pac = tensorpac.Pac(f_amp=np.arange(80,160))

amps = []
for ch_idx in range(data_ripple.shape[1]):
    amp = pac.filter(grand_ecog.info["sfreq"], data_ripple[:,ch_idx,],
                     ftype="amplitude", n_jobs=4)
    amp = _shift_signals(amp, peaks)
    amps.append(amp)
amps = np.stack(amps)
