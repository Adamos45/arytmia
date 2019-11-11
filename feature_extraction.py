from read_records import read_records
from preprocessing import preprocessing
from features_calculation import features_calculation
import pandas
from logger import logger
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


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

    pandas.DataFrame(features_package).to_csv("./features/" + feature.value + "_" + str(dataset)
                                              + ".csv", index=False, header=False)


def read_data(files):
    store = []

    for file in files:
        logger.info("Reading" + str(file))
        train_features_vector = []
        train_label = []
        test_features_vector = []
        test_label = []
        scaler = StandardScaler()
        for i in range(1, 3):
            data = pandas.read_csv("./features/" + file.value + "_" + str(i) + ".csv")
            data = data.values
            if i == 1:
                test_features_vector = scaler.fit_transform(data[:, 0:(len(data[0]) - 1)].astype(np.float64))
                test_label = data[:, len(data[0]) - 1].astype(np.float).astype(np.int)
            else:
                train_features_vector = scaler.fit_transform(data[:, 0:(len(data[0]) - 1)].astype(np.float64))
                train_label = data[:, len(data[0]) - 1].astype(np.float).astype(np.int)

        store.append([train_features_vector, train_label, test_features_vector, test_label])

    return store


def prepare_data(files_to_read):
    if "features" not in os.listdir("."):
        os.mkdir("features/")
    files_in_directory = os.listdir("./features/")
    for i in range(1, 3):
        for file in files_to_read:
            file_name = file.value + "_" + str(i) + ".csv"
            if file_name not in files_in_directory:
                logger.debug("Calculating file " + file_name + "...")
                feature_extraction(file, i)
                logger.debug("finished.")
            else:
                logger.debug("Found file " + file_name + ".")