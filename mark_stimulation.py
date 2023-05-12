import mne
from os import listdir
from os.path import join
import numpy as np
from mne.time_frequency import tfr_morlet

"""
Figures out and marks where stimulation occurred.
"""

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

tfr_thresh_range = list(np.linspace(0.001, 0.008, 50))
tfr_lower_thresh = 1e-6
pre_stim_buffer = 5 # in case of residual stimulation effects
post_stim_buffer = 10 # in case of residual stimulation effects
analy_duration = 60 # how much of a time period before and after stimulation
# if you want a different time period for the intermediate stimulations,
# otherwise leave the same as analy_duration
between_duration = 60
filelist = listdir(proc_dir)
epolen = 10 # length of equal length epochs to break the raw file into
min_bad = 25
picks = ["Fz","AFz","Fp1","Fp2","FC1","FC2","Cz"]
n_jobs = 8

subjs = ["1001", "1002", "3001", "3002"]
conds = ["Stim"] # no point in doing this to sham conditions

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw(join(proc_dir, f"f_EPI_{subj}_{cond}-raw.fif"),
                         preload=True)
        # identify what frequency was the stimulation
        psd = raw.compute_psd(fmax=2, picks=picks, n_jobs=n_jobs,
                              method="welch", n_fft=16384)
        freqs, psd = psd.freqs, psd.get_data().mean(axis=0)
        fmax = freqs[np.argmax(psd)]
        epo = mne.make_fixed_length_epochs(raw, duration=epolen)
        # get a TFR at the stimulation frequency
        power = tfr_morlet(epo, [fmax], n_cycles=3, picks=picks,
                           average=False, return_itc=False, n_jobs=n_jobs)

        # average the channels, put back into single dimensional form
        tfr_aschan= power.data.mean(axis=1)[:, 0].reshape(-1)

        # identify stimulation periods. do not remember how any of this works
        winner_std = np.inf
        for tfr_upper_thresh in tfr_thresh_range:
            these_annotations = raw.annotations.copy()
            tfr_over_thresh = (tfr_aschan > tfr_upper_thresh).astype(float) - 0.5
            tfr_over_cross = tfr_over_thresh[:-1] * tfr_over_thresh[1:]
            tfr_over_cross = np.concatenate((np.zeros(1),tfr_over_cross))
            tfr_under_thresh = (tfr_aschan < tfr_lower_thresh).astype(float) - 0.5
            tfr_under_cross = tfr_under_thresh[:-1] * tfr_under_thresh[1:]
            tfr_under_cross = np.concatenate((np.zeros(1),tfr_under_cross))
            tfr_under_cross_inds = np.where(tfr_under_cross < 0)[0]

            if (len(np.where(tfr_over_cross < 0)[0]) == 0 or
                len(np.where(tfr_over_cross < 0)[0]) == 0):
                continue

            earliest_idx = 0
            stim_idx = 0
            for cross in np.nditer(np.where(tfr_over_cross < 0)[0]):
                if cross < earliest_idx:
                    continue
                min_bad_idx = cross + int(np.round(min_bad * raw.info["sfreq"]))
                if min_bad_idx > len(tfr_under_thresh):
                    min_bad_idx = len(tfr_under_thresh) - 1
                if tfr_under_thresh[min_bad_idx] > 0: # false alarm; do not mark
                    earliest_idx = min_bad_idx
                    continue

                begin = raw.times[cross] - pre_stim_buffer
                idx = tfr_under_cross_inds[tfr_under_cross_inds > min_bad_idx][0]
                end = raw.times[idx] + post_stim_buffer
                duration = end - begin
                if stim_idx == 0:
                    pre_dur = analy_duration
                    post_dur = between_duration
                else:
                    pre_dur = between_duration
                    post_dur = between_duration
                these_annotations.append(begin, duration,
                                         "BAD_Stimulation {}".format(stim_idx))
                these_annotations.append(begin - pre_dur, pre_dur,
                                         "Pre_Stimulation {}".format(stim_idx))
                these_annotations.append(begin + duration, post_dur,
                                         "Post_Stimulation {}".format(stim_idx))
                earliest_idx = idx
                stim_idx += 1

            # assess uniformity
            durations = []
            for annot in these_annotations:
                if "BAD" in annot["description"]:
                    durations.append(annot["duration"])
            dur_std = np.array(durations).std()
            if dur_std < winner_std and dur_std != 0.:
                winner_annot = these_annotations.copy()
                winner_std =  dur_std
                winner_id = tfr_upper_thresh
                winner_durations = durations.copy()

        # last post-stimulation period should be longer
        last_annot = winner_annot[-1].copy()
        winner_annot.delete(-1)
        winner_annot.append(last_annot["onset"], analy_duration,
                            last_annot["description"])

        raw.set_annotations(winner_annot)
        print("\nThreshold of {} was optimal.\nDurations:".format(winner_id))
        print(winner_durations)
        print("\nStd:{}\n".format(winner_std))

        avg_dur = np.array(winner_durations).mean()

    raw.save(join(proc_dir, f"af_EPI_{subj}_{cond}-raw.fif"), overwrite=True)
