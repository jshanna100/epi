import mne
from os import listdir
import re
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/epi/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/epi/"
proc_dir = root_dir+"proc/"
subjs = ["1001"]
conds = ["Stim"]

for subj in subjs:
    for cond in conds:
        raw = mne.io.Raw("{}aof_EPI_{}_{}-raw.fif".format(proc_dir, subj, cond),
                         preload=True)
        raws = []
        for annot in raw.annotations:
            match = re.match("(.*)_Stimulation (\d)", annot["description"])
            if match:
                stim_pos, stim_idx = match.group(1), match.group(2)
                if stim_pos == "BAD":
                    continue
                begin, duration = annot["onset"], annot["duration"]
                end = begin + duration
                if end > raw.times[-1]:
                    end = raw.times[-1]
                raws.append(raw.copy().crop(begin,end))
        if len(raws) == 0:
            continue
        raw_cut = raws[0]
        raw_cut.append(raws[1:])
        raw_cut.save("{}caof_EPI_{}_{}-raw.fif".format(proc_dir,subj,cond),
                     overwrite=True)
