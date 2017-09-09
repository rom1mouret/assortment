#!/usr/bin/env python3

import argparse
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn.utils

from scipy.stats import rankdata
import scipy.io.arff
import numpy as np

from features import session_features
from random_knn_wrapper import RandomKNNWrapper
from auto_encoder import AutoEncoder
from one_class_svm import OneClassSVM
from lof import LOF
from quasi_abod import QuasiABOD
from evaluation import recall_at_k


class DecisionRerversing:
    def __init__(self, base_estimator):
        self.base_estimator_ = base_estimator

    def fit(self, X, y=None, sample_weight=None):
        self.base_estimator_.fit(X, y, sample_weight)

    def decision_function(self, X):
        return -self.base_estimator_.decision_function(X)


def load_data(inp):
    print("loading", inp)
    if ".arff" in inp:
        arff_content, meta = scipy.io.arff.loadarff(inp)
        is_categorical = [t.lower() not in ("real", "numeric") for t in meta.types()]
        label_index = meta.names().index('outlier')
        try:
            id_index = meta.names().index('id')
        except:
            id_index = -1
        npcol = 0
        arffcol_to_npcol = []
        for i in range(len(meta.names())):
            if i not in (label_index, id_index):
                arffcol_to_npcol.append(npcol)
                npcol += 1
            else:
                arffcol_to_npcol.append(None)
        categorical = [arffcol_to_npcol[i] for i, c in enumerate(is_categorical)
                       if c and arffcol_to_npcol[i] is not None]
        print("total number of columns:", npcol, ", categorical:", len(categorical))

        # convert to numpy matrix
        print("converting to numpy matrix")
        cat_to_int = {}
        X = np.empty((len(arff_content), npcol))
        for i, row in enumerate(arff_content):
            for j, v in enumerate(row):
                dest = arffcol_to_npcol[j]
                if dest is not None:
                    if is_categorical[j]:
                        cat_int = cat_to_int.get(v)
                        if cat_int is None:
                            cat_int = len(cat_to_int)
                            cat_to_int[v] = cat_int
                        X[i, dest] = cat_int
                        #X[i, dest] = int(abs(v.__hash__()))  # collisions are very unlikely
                    else:
                        X[i, dest] = v

        # get labels
        print("getting labels")
        label_to_int = {b"'yes'": 1, b"'no'": 0}
        ground_truth = np.array([label_to_int[row[label_index]] for row in arff_content])
    else:
        # assuming no categorical columns in the CSV
        categorical = []

        alldata = np.loadtxt(inp, delimiter=',')
        X = alldata[:, :-1]
        ground_truth = alldata[:, -1]

    for i in categorical:
        print("number of categories for column", i, ":", len(set(X[:, i])))

    print("number of outliers", np.sum(ground_truth))
    print("ratio of outliers", np.mean(ground_truth))

    # shuffle data
    print("shuffling rows")
    X, ground_truth = sklearn.utils.shuffle(X, ground_truth)

    return X, ground_truth, categorical


def min_rank_agreement(scores):
    # ranking
    min_rank = 9*scores[0].shape[0]//10   # lower ranks are irrelevant
    ranks = [np.clip(rankdata(s), min_rank, np.inf) for s in scores]

    # pairwise comparison of ranks
    max_agreement = np.sum(np.clip(np.arange(scores[0].shape[0]), min_rank, np.inf))
    agreements = []
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            agreement = np.sum(np.minimum(ranks[i], ranks[j]))/max_agreement
            agreements.append(agreement)

    return agreements


def dot_rank_agreement(scores):
    # ranking
    min_rank = 9*scores[0].shape[0]//10   # lower ranks are irrelevant
    ranks = [np.clip(rankdata(s), min_rank, np.inf) for s in scores]

    # pairwise comparison of ranks
    max_agreement = np.dot(np.arange(scores[0].shape[0]), np.arange(scores[0].shape[0]))
    agreements = []
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            agreement = np.dot(ranks[i], ranks[j])/max_agreement
            agreements.append(agreement)

    return agreements


def intersection_agreement(scores):
    # ranking
    tail = scores[0].shape[0]//10   # lower ranks are irrelevant
    ranks = [set(s.argsort()[::-1][:tail]) for s in scores]

    # pairwise comparison of ranks
    max_agreement = len(ranks[0])
    agreements = []
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            agreement = len(ranks[i] & ranks[j])/max_agreement
            agreements.append(agreement)

    return agreements


def one_round(X, ground_truth, categorical):
    pipelines = [

        Pipeline(steps=[("encoding", OneHotEncoder(categorical_features=categorical, sparse=False)),
                        ("scaling", StandardScaler()),
                        ("knn", RandomKNNWrapper())]),

        Pipeline(steps=[("encoding", OneHotEncoder(categorical_features=categorical, sparse=False)),
                        ("iforest", DecisionRerversing(IsolationForest(n_jobs=-1)))]),

        Pipeline(steps=[("encoding", OneHotEncoder(categorical_features=categorical, sparse=False)),
                        ("scaling", StandardScaler()),
                        ("abod", QuasiABOD())]),

        Pipeline(steps=[("encoding", OneHotEncoder(categorical_features=categorical, sparse=False)),
                        ("scaling", StandardScaler()),
                        ("lof", LOF())]),

        Pipeline(steps=[("encoding", OneHotEncoder(categorical_features=categorical, sparse=False)),
                        ("scaling", StandardScaler()),
                        ("svm", OneClassSVM())]),


        Pipeline(steps=[("encoding", OneHotEncoder(categorical_features=categorical, sparse=False)),
                        ("scaling", StandardScaler()),
                        ("autoencoder", AutoEncoder())])
    ]

    # scoring
    scores = []
    for pipeline in pipelines:
        name = pipeline.steps[-1][0]
        print("fit", name)
        pipeline.fit(X)
        print("predict with", name)
        s = pipeline.decision_function(X)
        assert s.shape[0] == X.shape[0] and len(s.shape) == 1
        scores.append(s)

    # agreements
    agreements = intersection_agreement(scores)

    # absolute evaluation
    precisions = [recall_at_k(ground_truth, s) for s in scores]
    print("precision or recall", precisions)

    # complexity metrics
    complexity = []
    for s in scores:
        complexity += session_features(X, s)

    # push example
    names = [p.steps[-1][0] for p in pipelines]

    return (names, precisions, agreements, complexity)


def get_training_examples(input_file):
    X, ground_truth, categorical = load_data(input_file)
    return [one_round(X, ground_truth, categorical) for _ in range(3)]


if __name__ == "__main__":
    # options
    parser = argparse.ArgumentParser(description='Outlier Detection')
    parser.add_argument('dataset', metavar='dataset-path', type=str, nargs='+', help="path of the ARFF/CSV files to run the outlier detetion on")
    parser.add_argument('-s', default="1", metavar="suffix", type=str, nargs=1, help="output dataset suffix")

    args = vars(parser.parse_args())
    inp = args['dataset']
    suffix = args['s']

    # build output dataset
    dataset = []
    for arff_file in inp:
        for example in get_training_examples(arff_file):
            dataset.append((arff_file, example))

    # write on disk
    with open("mdl_selection_dataset_%s.csv" % suffix, "w") as f:
        for arff_file, (names, precisions, features, complexity) in dataset:
            x = [arff_file] + names + precisions + features + complexity
            f.write("%s\n" % ",".join(map(str, x)))
