import enum


def standarize_annotation(symbol):
    switcher = {
        "N": 0,
        "L": 0,
        "R": 0,
        "e": 1,
        "j": 1,
        "A": 1,
        "a": 1,
        "J": 1,
        "S": 1,
        "V": 2,
        "E": 2,
        "F": 3
    }
    return switcher.get(symbol, "unrecognized")


class FeatureNames(enum.Enum):
    mvsk = "hos_mvsk"
    vsk = "hos_vsk"
    db5 = "db5"
    db6 = "db6"
    raw90 = "raw90"
    raw45 = "raw45"
    diffs = "diffs"
    spect = "spect"
