import os

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scoring import scorer
from logger import logger
from symbols_enums import FeatureNames


def train_models(data, files):
    predictions = []

    for file_number, samples in enumerate(data):
        X_train = samples[0]
        y_train = samples[1]
        X_test = samples[2]
        y_test = samples[3]

        if files[file_number] in [FeatureNames.hos_vsk, FeatureNames.raw45, FeatureNames.spect]:
            model = KNeighborsClassifier(n_neighbors=7, algorithm='auto')
        else:
            model = MLPClassifier(hidden_layer_sizes=(20, 10, 10), max_iter=200, alpha=0.001,
                                  nesterovs_momentum=True, solver='sgd', verbose=False, tol=0.0001,
                                  learning_rate='adaptive', random_state=1)

        def train_model(name):
            logger.debug("Training " + files[file_number].value)
            model.fit(X_train, y_train)
            joblib.dump(model, open("./trained_models/" + name, 'wb'))

        model_name = files[file_number].value + ".sav"
        if "trained_models" not in os.listdir("."):
            os.mkdir("./trained_models/")
            train_model(model_name)
        elif model_name not in os.listdir("./trained_models"):
            train_model(model_name)
        else:
            logger.debug("Found " + model_name)
            model = joblib.load("./trained_models/" + model_name)
        logger.info(files[file_number])
        score_results = scorer(X_test, estimator=model)
        predictions.append(score_results)
    ensembled_prediction = []
    for i in range(len(predictions[0])):
        voting = np.array(np.zeros(shape=(len(files), 4)))
        for j in range(len(predictions)):
            voting[j] = predictions[j][i]
        voting = voting.T
        products = np.array(np.zeros(shape=4))
        for p in range(len(products)):
            products[p] = np.product(voting[p])
        ensembled_prediction.append(np.argmax(products))
    scorer(Y=data[0][3], ensembled_predction=ensembled_prediction, diffs=True)
