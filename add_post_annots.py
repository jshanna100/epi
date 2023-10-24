import mne
from os import listdir
from os.path import join
import re

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc") # where the files are
proc_files = listdir(proc_dir)

overwrite = False
subjs = ["1001", "1002", "1003", "1005", "3001", "3002"]
analy_duration = 60

for subj in subjs:
    infile = f"stim_EPI_{subj}_Stim-annot.fif"
    outfile = f"pstim_EPI_{subj}_Stim-annot.fif"
    # check if it already exists
    if outfile in proc_files and not overwrite:
        print(f"{outfile} already exists. Skipping...")
        continue
    annots = mne.read_annotations(join(proc_dir, infile))
    annots_c = annots.copy()
    for annot in annots_c:
        ma = re.match("BAD_Stimulation (\d)", annot["description"])
        if not ma:
            continue
        stim_idx = int(ma.groups()[0])
        annots.append(annot["onset"]+annot["duration"], analy_duration,
                      f"Post_Stimulation {stim_idx}")
    annots.save(join(proc_dir, outfile))