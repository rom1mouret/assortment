import numpy as np
from sklearn.base import BaseEstimator
from math import log


class RandomKNN(BaseEstimator):
    def get_params():
        pass

    def set_params():
        pass

    def fit(self, X, y=None, sample_weight=None):
        # Sugiyama's calculations to get the optimal sampling size:
        # s = log_β(log α/(log α + log β)
        anomaly_ratio = 0.05
        min_cluster_size = 0.2
        alpha = 1 - anomaly_ratio
        beta = 1 - min_cluster_size
        sampling = int(log(log(alpha)/(log(alpha)+log(beta)))/log(beta))
        print("KNN number of points", sampling)

        indices = np.random.choice(np.arange(X.shape[0]), sampling, replace=False)

        selection = X[indices]
        matrices = []
        for i in range(len(indices)):
            rolled = np.roll(selection, i, axis=0)
            matrices.append(rolled.reshape(rolled.shape[0], rolled.shape[1], 1))

        self._points = np.concatenate(matrices, axis=2)

    def decision_function(self, X):
        # padding
        misalignment = self._points.shape[0] - X.shape[0] % self._points.shape[0]
        if misalignment != 0:
            extra = np.empty((misalignment, X.shape[1]))
        X = np.concatenate([X, extra], axis=0)

        # split in section that aligns with training set
        n = X.shape[0]//self._points.shape[0]
        batches = np.split(np.arange(X.shape[0]), n)

        # compute distances
        dist = []
        for batch in batches:
            b = X[batch]
            b = b.reshape(b.shape[0], b.shape[1], 1)
            diff = np.sum(np.square(self._points - b), axis=1, keepdims=True)
            min_dist = np.min(diff, axis=2)
            dist.append(min_dist.squeeze())
        dist = np.concatenate(dist)

        # remove padding
        if misalignment > 0:
            dist = dist[:-misalignment]

        return dist
