# Run order for scripts

## Convert
At the beginning of the scripts, specify the directories where the files should be loaded from/saved to
### Greifswald/Magdeburg (1000s/2000s)
convert_edf.py
### Frankfurt (3000s)
convert_trc.py

## Filter
do_filter.py. 
Runs a bandpass and notch filter on the data and resamples them.

## Select hippocampal channels
hand_mark_HT.py
This is a helper script for selecting the hippocampal channels. For every recording/hemisphere, the potential hippocampal channels will be displayed. Select two and then close the window. The highest number channel you selected will be the active hippocampal channel and the 2nd you selected will be the reference. The active hippocampal channels will then be re-referenced and renamed, and the other intracranial channels will be thrown away to save space.

# Mark stimulation

hand_mask_stim.py
Helper script for marking stimulation. Every stim recording will be loaded one by one. Go into annotation mode (hit "a") and then select the areas where stimulation occurred.

add_post_annots.py
This adds the post-stimulation periods to the ones you selected in the previous script.

mark_sham_stimulation_algo.py
This algorithmically determines where to add sham stimulations in the sham recordings.

# Oscillation detection
get_osc.py
Does sleep staging, detects slow oscillations, spindle events, IED artefacts. Saves each one as a separate MNE Python annotation file.

# Oscillation embedding
Detects spindle peaks that are 1) embedded in slow oscillations 2) within NREM sleep, 3) not coocurring with IED artefacts. The script then produces a triple coupling analysis of ripple band hippocampal power against SO embedded spindle peak phase.
