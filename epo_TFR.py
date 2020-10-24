import mne
from os.path import isdir
import numpy as np
from mne.time_frequency import tfr_morlet

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"

proc_dir = root_dir+"/proc/"

subjs = ["1001", "1002"]
subjs = ["1001"]
conds = ["Stim"]
n_jobs = 8

for subj in subjs:
    for cond in conds:
        epo = mne.read_epochs("{}s_epi_{}_{}-epo.fif".format(proc_dir,subj,cond))
        e = epo["OscType=='SO'"]
        if len(e):
            tfr_eeg = tfr_morlet(e, np.arange(10,20), 5, n_jobs=n_jobs,
                             picks="central", return_itc=False)
            tfr_eeg.apply_baseline((-1.25,-0.75), mode="zscore")
            tfr_ecog = tfr_morlet(e, np.arange(50,200), 9, n_jobs=n_jobs,
                             picks=["HHL", "HCL", "HHR", "HCR"],
                             return_itc=False)
            tfr_ecog.apply_baseline((-0.25, 0.1), mode="zscore")
