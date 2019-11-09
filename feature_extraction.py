from read_records import read_records
from preprocessing import preprocessing
from features_calculation import features_calculation
import pandas


def feature_extraction(feature, dataset):
    features_package = []

    for i in range(22):
        print(str(i+1)+"/22")
        if dataset == 1:
            record, annotation = read_records(i, 1)
        else:
            record, annotation = read_records(i, 2)
        baseline = preprocessing(record)
        feature_sample = features_calculation(baseline, annotation, feature)
        features_package.extend(feature_sample)

    pandas.DataFrame(features_package).to_csv("./features/" + feature + str(dataset) + ".csv", index=False, header=False)
