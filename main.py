from symbols_enums import FeatureNames
from feature_extraction import prepare_data, read_data
from train_models import train_models


def main():
    files = [FeatureNames.mvsk, FeatureNames.db5,
             FeatureNames.db6, FeatureNames.diffs]

    prepare_data(files)
    data = read_data(files)
    train_models(data, files)


if __name__ == "__main__":
    main()
