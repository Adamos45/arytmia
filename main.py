import os
import pandas

import numpy as np
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from feature_extraction import feature_extraction
from scoring import scorer
from symbols_enums import FeatureNames
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight


def prepare_data(files_to_read):
    if "features" not in os.listdir("."):
        os.mkdir("features/")
    files_in_directory = os.listdir("./features/")
    for i in range(1, 3):
        for file in files_to_read:
            if file + str(i) + ".csv" not in files_in_directory:
                print("Calculating file " + file + str(i) + ".csv...")
                feature_extraction(file, i)
                print("finished.")
            else:
                print("Found file " + file + str(i) + ".csv.")


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
    files = [FeatureNames.hos.value, FeatureNames.wavlt.value, FeatureNames.raw.value,
             FeatureNames.diffs.value, FeatureNames.spect.value]
    prepare_data(files)
    print("Reading data")
    data = read_data(files)
    print("Finished.")


    for file_number, samples in enumerate(data):
        X_train = samples[0]
        y_train = samples[1]
        X_test = samples[2]
        y_test = samples[3]
        cw = class_weight.compute_class_weight('balanced', [0, 1, 2, 3], y_train)
        def preprocess(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.int64)

            return x, y

        def create_dataset(xs, ys, n_classes=4):
            ys = tf.one_hot(ys, depth=n_classes)
            return tf.data.Dataset.from_tensor_slices((xs, ys)) \
                .map(preprocess) \
                .shuffle(len(ys)) \
                .batch(128)

        train_dataset = create_dataset(X_train, y_train)
        val_dataset = create_dataset(X_test, y_test)

        model = keras.Sequential([
            keras.layers.Dense(units=10, activation='relu'),
            keras.layers.Dense(units=20, activation='relu'),
            keras.layers.Dense(units=10, activation='relu'),
            keras.layers.Dense(units=4, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(
            train_dataset.repeat(),
            epochs=150,
            steps_per_epoch=5000,
            validation_data=val_dataset.repeat(),
            validation_steps=10,
            class_weight=cw
        )
        # train_x, scores = feature_selection(train_x, train_y)
        # indices = []
        # for i in scores:
        #     indices.append(i[0])
        # test_x = test_x[:, np.array(indices)]

        # svm_model = svm.SVC(C=0.8, kernel='rbf', degree=4, gamma='auto',
        #                     coef0=0.0, shrinking=True, probability=False, tol=0.001,
        #                     cache_size=200, class_weight='balanced', verbose=False,
        #                     max_iter=-1, decision_function_shape='ovo', random_state=1)
        # mlp = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=1000, alpha=0.0001, momentum=0.9,
        #                     nesterovs_momentum=True, solver='adam', verbose=True, tol=0.0001, random_state=1)

        # print(np.mean(cross_val_score(svm_model, rr_train[0], rr_train[1], scoring=scorer, cv=10)))
        # mlp.fit(X_train, y_train)
        print(files[file_number])
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
            data = pandas.read_csv("./features/" + file + str(i) + ".csv")
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
