from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
import numpy as np


class QuasiABOD(BaseEstimator):
    """ Similar to ABOD except:
    - training set is heavily downsampled
    - distance to neighbors is taken into account
    """

    def __init__(self, samples: int=200):
        self._samples = samples

    def fit(self, X, y=None, sample_weight=None):
        # randomly sample N/2 points
        # alternatively, the user can provide points already sampled
        if X.shape[0] < self._samples:
            X = shuffle(X)
        else:
            size = self._samples - self._samples % 2
            random_indices = np.random.choice(np.arange(X.shape[0]), size, replace=False)
            X = X[random_indices]

        cut = X.shape[0]//2
        points1 = X[:cut]
        points2 = X[-cut:]

        # determine a reasonable batch size based on the dimensionality
        self._batch_size = 4096
        n_points = points1.shape[0]
        dim = points1.shape[1]
        while dim*self._batch_size*n_points > 5000000 and self._batch_size > 1:
            self._batch_size //= 2

        # prepare the decision function
        self._repeated_p1 = np.tile(points1, (self._batch_size, 1))
        self._repeated_p2 = np.tile(points2, (self._batch_size, 1))

        self._points1 = points1

    def decision_function(self, X):
        # add some empty rows to match batch_size
        extra = self._batch_size - X.shape[0] % self._batch_size
        if extra > 0:
            padding = np.zeros((extra, X.shape[1])) #np.ones would prevent division-by-zero though
            padded = np.append(X, padding, axis=0)
        else:
            padded = X

        n_points = self._points1.shape[0]
        dim = self._points1.shape[1]

        res = []
        for batch_idx in range(padded.shape[0] // self._batch_size):
            batch = padded[batch_idx*self._batch_size:(batch_idx+1)*self._batch_size]
            tiled = np.tile(batch, (1, n_points)).reshape((self._batch_size*n_points, dim))

            # vectors between which angles will be computed
            vec1 = self._repeated_p1 - tiled
            vec2 = self._repeated_p2 - tiled
            norm = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1) + 0.0000001
            dot_product = np.sum(vec1*vec2, axis=1)  # sum of element-wise multiplications
            dot_product /= norm
            angles = np.arccos(dot_product)

            # well-shaped angles
            angles = np.reshape(angles, (self._batch_size, n_points))

            # simple variance, as in the original ABOD
            scores = -np.var(angles, axis=1)
            res.append(scores)

        alltogether = np.concatenate(res, axis=0)

        # remove padding
        alltogether = alltogether[:X.shape[0]]

        return alltogether
