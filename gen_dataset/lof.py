from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import sklearn.utils
from sklearn.base import BaseEstimator


class LOF(BaseEstimator):

    def fit(self, X, y=None, sample_weight=None):
        max_samples = 4000
        if X.shape[0] > max_samples:
            # we are training two models to prevent penalizing data points
            # that are not in the training set
            training_size = min(max_samples, X.shape[0]//2)
            X = sklearn.utils.shuffle(X)

            model1 = LocalOutlierFactor()
            model1.fit(X[:training_size])

            model2 = LocalOutlierFactor()
            model2.fit(X[training_size:2*training_size])

            self._models = [model1, model2]
        else:
            model = LocalOutlierFactor()
            model.fit(X)
            self._models = [model]

    def decision_function(self, X):
        scores = [-model._decision_function(X) for model in self._models]
        return np.max(scores, axis=0)
