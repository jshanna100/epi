import mne
from gssc.infer import EEGInfer
from os import listdir
from os.path import join
import numpy as np
from utils import annot_within, annot_overlap, hfb_power, circ_linear_corr, reassemble_annots, output_annot_csv
from scipy.signal import hilbert
from scipy.signal.windows import boxcar
import matplotlib.pyplot as plt

"""
triple coupling
"""

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")
fig_dir = join(root_dir, "figs")

filelist = listdir(proc_dir)
ei = EEGInfer()

subjs = ["1001", "1002", "1003", "1004", "1005", "2001", "3001", "3002"]
conds = ["Sham"]
chans_a = ["FC1", "FC2"]
chans_b = ["F3", "Fz", "F4"]

phase_bins = np.linspace(-np.pi, np.pi, 24)

for subj in subjs:
    chans = chans_a if "100" in subj else chans_b
    for cond in conds:
        for hemi in ["L", "R"]:
            ur_raw = mne.io.Raw(join(proc_dir, f"HT_f_EPI_{subj}_{cond}-raw.fif"), preload=True)
            so_annots = mne.read_annotations(join(proc_dir, f"SO_EPI_{subj}_{cond}-annot.fif"))
            # which SOs do not have an IED within them
            ied_annots = mne.read_annotations(join(proc_dir, f"H{hemi}IED_EPI_{subj}_{cond}-annot.fif"))
            exclude_inds = annot_overlap(so_annots, ied_annots)
            if len(exclude_inds):
                so_annots.delete(exclude_inds)
            # which spindle peaks are embedded within an SO
            spindle_annots = mne.read_annotations(join(proc_dir, f"spindle_EPI_{subj}_{cond}-annot.fif"))
            non_spindle_inds = [idx for idx, annot in enumerate(spindle_annots) 
                                if "Peak" not in annot["description"]]
            spindle_annots.delete(non_spindle_inds)
            match_annots, containing_annots = annot_within(spindle_annots, so_annots)
            # which SO-embedded spindle peaks are also in NREM sleep
            hypno_annots = mne.read_annotations(join(proc_dir, f"hypno_EPI_{subj}_{cond}-annot.fif"))
            hypno_annots = [annot for annot in hypno_annots if 0 < int(annot["description"]) < 4]
            match_annots, _ = annot_within(match_annots, hypno_annots)

            # make spreadsheets of event times
            embedded_peaks = reassemble_annots(match_annots, orig_time=spindle_annots.orig_time)
            output_annot_csv(so_annots, join(proc_dir, f"{subj}_{cond}_H{hemi}_SO.csv"))
            output_annot_csv(embedded_peaks, join(proc_dir, f"{subj}_{cond}_H{hemi}_spindpeaks_emb.csv"))
            output_annot_csv(spindle_annots, join(proc_dir, f"{subj}_{cond}_H{hemi}_spindpeaks.csv"))
            output_annot_csv(ied_annots, join(proc_dir, f"{subj}_{cond}_H{hemi}_ied.csv"))

            ## now HFB as a function of spindle peak phase
            # instantaneous SO phase
            raw = ur_raw.copy().pick_channels(chans).filter(l_freq=0.5, h_freq=1.25)
            signal = raw.get_data(picks=chans).mean(axis=0)
            phase = np.angle(hilbert(signal))
            
            # SO phase at spindle peaks
            peak_inds = np.array([ur_raw.time_as_index(annot["onset"])[0] for annot in match_annots])
            if len(peak_inds) <  100:
                # too few events
                print(f"Only {len(peak_inds)} events found. Skipping.")
                continue
            # bin all the phases
            all_bins = np.digitize(phase, phase_bins)
            # get the bin, bin indices around each peak
            bin_inds, spind_bins = [], []
            for p_idx in peak_inds:
                this_bin = all_bins[p_idx]
                search_win_len = int(ur_raw.info["sfreq"]*0.05)
                search_window = all_bins[p_idx-search_win_len:p_idx+search_win_len]
                local_inds = np.where(search_window == this_bin)[0]
                inds = local_inds + p_idx-search_win_len
                bin_inds.append(inds)
                spind_bins.append(this_bin)
            
            # calculate HFB power
            hfb_amp = hfb_power(ur_raw, f"H{hemi}",
                                [[70, 80], [80, 90], [90, 100], [100, 110], 
                                 [110, 120], [120, 130], [130, 140], [140, 149.99]])
            hfb_amp_z = (hfb_amp - hfb_amp.mean()) / hfb_amp.std()


            # bin the HFB power
            power_bins = [[] for pb in phase_bins]
            for this_bin, inds in zip(spind_bins, bin_inds):
                power_bins[this_bin].append(hfb_amp_z[inds].mean())
            power_bins = power_bins[1:]
            power_bin_means = np.array([np.mean(b) for b in power_bins])

            # smoothing
            smooth_bins = np.convolve(power_bin_means, np.ones(5)/5, mode="same")

            rho = circ_linear_corr(phase_bins[1:], smooth_bins)
            max_bin_idx = smooth_bins.argmax()
            max_degree = np.rad2deg(np.mean([phase_bins[max_bin_idx], phase_bins[max_bin_idx-1]]))

            plt.figure()
            ax = plt.subplot(111)
            ax.bar(np.arange(23), smooth_bins)
            ax.set_xticks([0, 22], labels=[f"-{chr(928)}", chr(928)])
            ax.set_title(f"{subj} {cond} ({len(peak_inds)})")
            
            # now calculate surrogate dist
            print("Calculating surrogate distro")
            surr_rhos = []
            for idx in range(5000):
                np.random.shuffle(power_bin_means)

                # smoothing
                s_smooth_bins = np.convolve(power_bin_means, np.ones(5)/5, mode="same")
                surr_rho = circ_linear_corr(phase_bins[1:], s_smooth_bins)
                surr_rhos.append(surr_rho)
            surr_rhos = np.array(surr_rhos)
            z_rho = (rho - surr_rhos.mean()) / surr_rhos.std()

            # print summary on the graph
            ax.text(0.1, 0.8, 
                    f"Rho: {rho:.2f}\nZ: {z_rho:.2f}\nSO ph. at max HFB: {np.round(max_degree):.0f}", 
                    transform = ax.transAxes)
            plt.savefig(join(fig_dir, f"{subj}_{cond}_H{hemi}_3coupling.png"))




                
            




        

    