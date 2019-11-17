from symbols_enums import standarize_annotation
import scipy.stats as stats
import scipy.signal as signal
import numpy as np
from symbols_enums import FeatureNames
import pywt


def features_calculation(baseline, annotation, feature):
    x = annotation.sample
    y = annotation.symbol

    if feature == FeatureNames.diffs:
        diffs = []
        previous_class = 0
        for i in range(len(x) - 1):
            if standarize_annotation(y[i]) == "unrecognized":
                continue
            if len(diffs) == 0:
                left = 0
                right = x[i + 1] - x[i]
            else:
                previous_class = standarize_annotation(y[i - 1])
                if previous_class == "unrecognized":
                    continue
                left = diffs[len(diffs) - 1][1]
                if i == len(x) - 1 or x[i] + 90 >= len(baseline):
                    right = len(baseline) - x[i]
                else:
                    right = x[i + 1] - x[i]
            diffs.append([left, right, previous_class, standarize_annotation(y[i])])

        diffs.pop(0)
        for i in range(0, len(diffs)):
            num = 0
            avg_val = 0
            for j in range(-9, 1):
                if j+i >= 0:
                    avg_val = avg_val + diffs[i+j][0]
                    num = num + 1
            diffs[i].insert(len(diffs[i])-2, avg_val/num)
        for i in range(0, len(diffs)):
            num = 0
            avg_val = 0
            for j in range(-1199, 1):
                if j + i >= 0:
                    avg_val = avg_val + diffs[i + j][0]
                    num = num + 1
            diffs[i].insert(len(diffs[i]) - 2, avg_val / num)
        global_avg_val_left = 0
        global_avg_val_right = 0
        for i in range(0, len(diffs)):
            global_avg_val_left += diffs[i][0]
            global_avg_val_right += diffs[i][1]
        global_avg_val_left = global_avg_val_left/len(diffs)
        global_avg_val_right = global_avg_val_right/len(diffs)
        for beat in diffs:
            beat.insert(len(beat) - 2, beat[0]/global_avg_val_left)
            beat.insert(len(beat) - 2, beat[1]/global_avg_val_right)
            beat.insert(len(beat) - 2, beat[2]/global_avg_val_left)
            beat.insert(len(beat) - 2, beat[3]/global_avg_val_left)
        return diffs

    else:
        segmented_beats = []
        for i in range(len(x) - 1):
            if standarize_annotation(y[i]) == "unrecognized":
                continue
            if len(segmented_beats) > 0 and standarize_annotation(y[i-1]) == "unrecognized":
                continue
            class_nr = standarize_annotation(y[i])
            if i == len(x) - 1 or x[i] + 90 >= len(baseline):
                segmented_beats.append([baseline[x[i] - 90:len(baseline) - 1], class_nr])
            else:
                segmented_beats.append([baseline[x[i] - 90:x[i] + 90], class_nr])
        segmented_beats.pop(0)

        if feature == FeatureNames.db5 or feature == FeatureNames.db6:
            wavs = []
            for seg in segmented_beats:
                c = pywt.wavedec(seg[0], feature.value, level=3)
                features = np.append(c[0], seg[1])
                wavs.append(features)
            return wavs

        if feature == FeatureNames.mvsk or feature == FeatureNames.vsk:
            hos = []
            for seg in segmented_beats:
                seg = [signal.savgol_filter(seg[0], 7, 2), seg[1]]
                features = [np.mean(seg[0]), stats.variation(seg[0]), stats.skew(seg[0]),
                            stats.kurtosis(seg[0])]
                features.append(seg[1])
                hos.append(features)
            return hos

        if feature == FeatureNames.raw90 or feature == FeatureNames.raw45:
            if feature == FeatureNames.raw90:
                divider = 2
            else:
                divider = 4
            raw = []
            for seg in segmented_beats:
                features = []
                for i, s in enumerate(seg[0]):
                    if i % divider == 0:
                        features.append(s)
                features.append(seg[1])
                raw.append(features)
            return raw

        if feature == FeatureNames.spect:
            spect = []
            for i, seg in enumerate(segmented_beats):
                _, sxx = signal.welch(np.array(seg[0]), 60, nperseg=180)
                features = np.append(sxx, seg[1])
                spect.append(features)
            return spect
