import mne
import numpy as np
from os.path import join
from os import listdir
import re


root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")

filelist = listdir(proc_dir)
for filename in filelist:
    this_match = re.match("spindle_EPI_(\d{4})_(.*)-annot.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()
    spindle_annots = mne.read_annotations(join(proc_dir, filename))
    so_filename = f"osc_EPI_{subj}_{cond}_frontal_SO-annot.fif"
    so_annots = mne.read_annotations(join(proc_dir, so_filename))
    raw_filename = f"c_EPI_{subj}_{cond}-raw.fif"
    raw = mne.io.Raw(join(proc_dir, raw_filename))

    onsets, durations, descs = [], [], [] 
    for annot in spindle_annots:
        found_embed = False
        for so_annot in so_annots:
            so_onset, so_end = so_annot["onset"], so_annot["onset"] + so_annot["duration"]
            onset, end = annot["onset"], annot["onset"] + annot["duration"]
            if (onset >= so_onset <= end) or (onset >= so_end <= end):
                onsets.append(onset)
                durations.append(annot["duration"])
                descs.append("Spindle SO-embedded")
                found_embed = True
                break
        if not found_embed:
            onsets.append(onset)
            durations.append(annot["duration"])
            descs.append("Spindle")
    new_annots = mne.Annotations(onsets, durations, descs, orig_time=spindle_annots.orig_time)
    breakpoint()



