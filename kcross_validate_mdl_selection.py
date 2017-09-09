#!/usr/bin/env python3

import argparse
import sklearn.utils
import numpy as np

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.metrics import classification_report, accuracy_score

from gating_models import *


def load_dataset(filename, number_of_detectors, to_ignore):
    n = number_of_detectors
    X_rows = []
    performances_rows = []
    row_to_dataset = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split(",")
            dataset = parts[0]
            row_to_dataset.append(dataset)
            names = parts[1:1+n]

            # performances
            performances = np.array(list(map(float, parts[1+n:1+2*n])))
            performances = performances.reshape((1, performances.shape[0]))
            performances_rows.append(performances)

            # features
            features = np.array(list(map(float, parts[1+2*n:])))
            features = features.reshape((1, features.shape[0]))
            X_rows.append(features)

    # features
    X = np.concatenate(X_rows, axis=0)
    imputer = Imputer(missing_values=np.nan)
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # performances
    performances = np.concatenate(performances_rows, axis=0)

    # remove the detectors to ignore
    to_ignore = set([name.lower() for name in to_ignore])
    to_keep = [i for i, name in enumerate(names) if name.lower() not in to_ignore]
    agreement_indexing = agreement_feature_indexing(number_of_detectors)
    complexity_indexing = complexity_feature_indexing(number_of_detectors, X.shape[1])
    new_order = []
    for i in range(len(to_keep)):
        for j in range(i+1, len(to_keep)):
            new_order.append(agreement_indexing[(to_keep[i], to_keep[j])])
    for col in to_keep:
        new_order += complexity_indexing[col]
    X = X[:, new_order]
    performances = performances[:, to_keep]
    names = [names[i] for i in to_keep]

    # shuffling is necessary for some training algorithms and the kcross
    X, row_to_dataset, performances = \
        sklearn.utils.shuffle(X, row_to_dataset, performances)

    return X, row_to_dataset, performances, names


if __name__ == "__main__":
    # options
    parser = argparse.ArgumentParser(description='Outlier Detection Benchmarks')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs=1, help="ensemble learning dataset")
    parser.add_argument('-n', metavar='num detectors', type=int, nargs='?', default=4, help="number of outlier detectors")
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help="names of the detectors to exclude from the model selection")
    parser.add_argument('--features', metavar='feature index', type=int, nargs='*', default=[0, 1], help="indices of the active features")

    args = vars(parser.parse_args())
    dataset = args['dataset'][0]
    n = args['n']
    to_ignore = args['exclude']
    active_features = args['features']

    X, row_to_dataset, performances, names = load_dataset(dataset, n, to_ignore)
    n = performances.shape[1]  # because 'n' can be changed by the exclude option

    # labels
    labels = performances.argmax(axis=1)

    # shuffling is necessary for some models
    X, row_to_dataset, performances, labels = \
        sklearn.utils.shuffle(X, row_to_dataset, performances, labels)

    # kcross
    y_true = []
    y_pred = []
    avg_performance = 0
    all_datasets = set(row_to_dataset)
    for dataset in all_datasets:
        training_indices = [i for i, d in enumerate(row_to_dataset)
                            if d != dataset and np.max(performances[i]) > 0.1]
        testing_indices = [i for i, d in enumerate(row_to_dataset)
                           if d == dataset]

        for _ in range(50):
            per_detector_pred = []
            sub_indices = np.random.choice(training_indices, 3*len(training_indices)//4, replace=False)
            for m in range(n):
                #model = MLfreeModel()
                #model = RelativeperformanceModel()
                #model = IndirectperformanceModel()
                model = AbsolutePrecisionModel(active_features)
                #model = DeltaPrecisionNarrowModel()
                #model = DeltaPrecisionModel()
                #model = DifferentialModel(active_features)
                model.set_detector_index(m, n)
                model.fit(X[sub_indices], performances[sub_indices])
                predictions = model.predict(X[testing_indices])
                per_detector_pred.append(predictions.reshape((predictions.shape[0], 1)))

            estimations = np.concatenate(per_detector_pred, axis=1)
            predicted_labels = estimations.argmax(axis=1)
            y_pred += predicted_labels.tolist()
            y_true += labels[testing_indices].tolist()

            local_performance = performances[testing_indices, predicted_labels]
            avg_performance += np.sum(local_performance)

    avg_performance /= len(y_pred)

    # classification reporting
    report = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("accuracy %f%%" % (100*accuracy))
    print(report)

    # performance reporting
    print("AVG performance(ensemble) = %0.2f%%" % (100*avg_performance))
    for i, name in enumerate(names):
        p = np.mean(performances[:, i])
        print("AVG performance(%s) = %0.2f%%" % (name, 100*p))

    print("AVG MAX performance = %0.2f%%" % (100*np.mean(np.max(performances, axis=1))))

    # plots
    for m in range(n):
        model = AbsolutePrecisionModel(active_features)
        model.set_detector_index(m, n)
        model.fit(X, performances)
        model.plot(X, performances, names[m])
