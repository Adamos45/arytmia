import wfdb as wf


def read_records(record_choice, dataset_choice):
    ds1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    ds2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    if record_choice > len(ds1):
        raise IndexError
    if dataset_choice == 1:
        record_nr = str(ds1[record_choice])
    else:
        record_nr = str(ds2[record_choice])
    sampfrom = 0
    sampto = 5000
    channel_nr = 0
    if record_nr == 114:
        channel_nr = 1
    record = wf.rdrecord('mitdb/' + record_nr, channels=[channel_nr])
    annotation = wf.rdann('mitdb/' + record_nr, 'atr', shift_samps=True)

    # wf.plot_wfdb(record, annotation=annotation, title='Record ' + record_nr + ' from MIT-BIH Arrhythmia Database',
    #              figsize=(10, 4), time_units="seconds", plot_sym=True)  # plot loaded singal

    return record, annotation
