import os
import pandas

import numpy as np
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from feature_extraction import feature_extraction
from scoring import scorer
from symbols_enums import FeatureNames
import joblib


def prepare_data(files_to_read):
    if "features" not in os.listdir("."):
        os.mkdir("features/")
    files_in_directory = os.listdir("./features/")
    for i in range(1, 3):
        for file in files_to_read:
            file_name = file.value + "_" + str(i) + ".csv"
            if file_name not in files_in_directory:
                print("Calculating file " + file_name + "...")
                feature_extraction(file, i)
                print("finished.")
            else:
                print("Found file " + file_name + ".")


def feature_selection(x, y):
    n_number = 32
    selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_classif, k=n_number)
    fit = selector.fit(x, y)
    fit_x = selector.transform(x)
    scores = []
    for j in range(len(fit.scores_)):
        scores.append([j, fit.scores_[j]])
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    return fit_x, scores[0:n_number]


def main():
    files = [FeatureNames.mvsk, FeatureNames.vsk, FeatureNames.db5, FeatureNames.db6,
             FeatureNames.raw90, FeatureNames.raw45, FeatureNames.diffs, FeatureNames.spect]
    prepare_data(files)
    print("Reading data")
    data = read_data(files)
    print("Finished.")


    for file_number, samples in enumerate(data):
        X_train = samples[0]
        y_train = samples[1]
        X_test = samples[2]
        y_test = samples[3]

        # model = svm.SVC(C=0.8, kernel='rbf', degree=4, gamma='auto',
        #                 coef0=0.0, shrinking=True, probability=False, tol=0.001,
        #                 cache_size=200, class_weight='balanced', verbose=False,
        #                 max_iter=-1, decision_function_shape='ovo', random_state=1)
        # model = DecisionTreeClassifier(class_weight='balanced')

        if file_number in [1, 5, 7]:
            model = KNeighborsClassifier(n_neighbors=7, algorithm='auto')
        else:
            model = MLPClassifier(hidden_layer_sizes=(20, 10, 10), max_iter=200, alpha=0.001,
                                  nesterovs_momentum=True, solver='sgd', verbose=False, tol=0.0001,
                                  learning_rate='adaptive')

        # print(np.mean(cross_val_score(model, X_train, y_train, scoring=scorer, cv=10)))

        def train_model(name):
            print("Training " + files[file_number].value)
            model.fit(X_train, y_train)
            joblib.dump(model, open("./trained_models/" + name, 'wb'))

        model_name = files[file_number].value+".sav"
        if "trained_models" not in os.listdir("."):
            os.mkdir("./trained_models/")
            train_model(model_name)
        elif model_name not in os.listdir("./trained_models"):
            train_model(model_name)
        else:
            print("Found " + model_name)
            model = joblib.load("./trained_models/" + model_name)

        scorer(model, X_test, y_test)


def read_data(files):
    store = []

    for file in files:
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


if __name__ == "__main__":
    main()
