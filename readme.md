# Run order for scripts

## Convert
At the beginning of the scripts, specify the directories where the files should be loaded from/saved to
### Greifswald (1000s)
Run convert_edf.py
### Frankfurt (3000s)
Run convert_trc.py

## Filter
Run do_filter.py. Runs a bandpass and notch filter on the data, saves them with
a f_ prefix.

## Mark bad channels
Run mark_badchans.py. Unless changed by the user, this will save with the
original, input filenames (prefixed with f_), to save space.

## Mark stimulation periods
Run mark_stimulation.py. On recordings where stimulation was applied, this will
mark the points where stimulation occurred, as well as pre- and post-stimulation
periods (for analysis).
