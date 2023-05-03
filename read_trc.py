import os
import neo
import numpy as np
import mne
import re
import datetime

def raw_from_neo(fname):
    seg_micromed = neo.MicromedIO(filename=fname).read_segment()

    ch_strs = [sig.name for sig in seg_micromed.analogsignals]
    ch_names = []
    for cs in ch_strs:
        this_match = re.search("Channels: \((.*)\)", cs)
        if this_match:
            ch_list = this_match.groups(0)[0].split(" ")
        else:
            ch_list = ["cn"]
        ch_names.extend(ch_list)

    sfreq = seg_micromed.analogsignals[0].sampling_rate
    data = []
    for sig in seg_micromed.analogsignals:
        data.append(np.array(sig))
    data = np.hstack(data).T
    data *= 1e-6  # put data from microvolts to volts
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    raw = mne.io.RawArray(data, info)

    orig_datetime = seg_micromed.rec_datetime
    utc_offset = datetime.datetime.utcnow() - orig_datetime.now()
    utc_dt = orig_datetime + utc_offset
    utc_datetime = datetime.datetime(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour,
                          utc_dt.minute, utc_dt.second,
                          tzinfo=datetime.timezone.utc)
    raw.set_meas_date(utc_datetime)

    return raw
