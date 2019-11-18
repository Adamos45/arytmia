from scipy import signal
import matplotlib.pyplot as plt


def preprocessing(record, for_gui=False):
    if for_gui:
        baseline = signal.medfilt(record, 71)
    else:
        baseline = signal.medfilt(record.p_signal[:, 0], 71)
        plt.figure(figsize=(10, 4))
        plt.plot(baseline)
        plt.show()
    baseline = signal.medfilt(baseline, 215)
    # plt.figure(figsize=(10, 4))
    # plt.plot(baseline)
    # plt.show()

    if for_gui:
        for i in range(len(record)):
            baseline[i] = record[i] - baseline[i]
    else:
        for i in range(len(record.p_signal)):
            baseline[i] = record.p_signal[i] - baseline[i]
    # plt.figure(figsize=(10, 4))
    # plt.plot(baseline)
    # plt.show()

    fc = 60  # Cut-off frequency of the filter
    w = fc / (360 / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low', output='ba')
    baseline = signal.filtfilt(b, a, baseline)
    # plt.figure(figsize=(10, 4))
    # plt.plot(baseline)
    # plt.show()

    max_samp = max(baseline)
    min_samp = min(baseline)
    baseline = [(2 * (n - min_samp) / (max_samp - min_samp) - 1) for n in baseline]

    # plt.figure(figsize=(10, 4))
    # plt.plot(baseline)
    # plt.show()
    return baseline
