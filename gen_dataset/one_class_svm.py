import sklearn.svm
import numpy as np
import sklearn.utils
from sklearn.base import BaseEstimator


class OneClassSVM(BaseEstimator):

    def fit(self, X, y=None, sample_weight=None):
        max_samples = 8000
        if X.shape[0] > max_samples:
            # we are training two models to prevent penalizing data points
            # that are not in the training set
            training_size = min(max_samples, X.shape[0]//2)
            X = sklearn.utils.shuffle(X)

            model1 = sklearn.svm.OneClassSVM()
            model1.fit(X[:training_size])

            model2 = sklearn.svm.OneClassSVM()
            model2.fit(X[training_size:2*training_size])

            self._models = [model1, model2]
        else:
            model = sklearn.svm.OneClassSVM()
            model.fit(X)
            self._models = [model]

    def decision_function(self, X):
        scores = [-model.decision_function(X) for model in self._models]
        res = np.max(scores, axis=0)
        assert res.shape[0] == X.shape[0]
        return res.squeeze()
