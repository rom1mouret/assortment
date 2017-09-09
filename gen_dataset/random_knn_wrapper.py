from random_knn import RandomKNN
import numpy as np
from sklearn.base import BaseEstimator


class RandomKNNWrapper(BaseEstimator):
    """ distances will be set to zero for points that are
        randomly chosen as centroids by RandomKNN. So, in
        order to prevent this kind of overfitting, we are taking
        the maximum distance over a bunch of RandomKNN, typically 2 """

    def fit(self, X, y=None, sample_weight=None):
        self._models = []
        for _ in range(2):
            model = RandomKNN()
            model.fit(X)
            self._models.append(model)

    def decision_function(self, X):
        scores = [model.decision_function(X) for model in self._models]
        return np.max(scores, axis=0)
