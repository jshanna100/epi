import mne
from os import listdir
import re
from os.path import join

root_dir = "/home/jev/hdd/epi/"
proc_dir = join(root_dir, "proc")
proclist = listdir(proc_dir)
subjs = ["1001", "1002", "1003", "1004", "1005", "2001", "3001", "3002"]
conds = ["Stim", "Sham"]
overwrite = False

for subj in subjs:
    for cond in conds:
        try:
            raw = mne.io.Raw(join(proc_dir, f"HT_f_EPI_{subj}_{cond}-raw.fif"),
                            preload=True)
            stim_str = "pstim" if cond == "Stim" else "stim"
            annots = mne.read_annotations(join(proc_dir, f"{stim_str}_EPI_{subj}_{cond}-annot.fif"))
        except:
            print(f"{subj} {cond} does not exist. Skipping.")
            continue
        outfile = f"c_EPI_{subj}_{cond}-raw.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists. Skipping...")
            continue
        raws = []
        for annot in annots:
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
        raw_cut.save(join(proc_dir, outfile))
