import numpy as np

from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import PolynomialFeatures


def agreement_feature_indexing(dim):
    index = {}
    for i in range(dim):
        for j in range(i+1, dim):
            index[(i, j)] = len(index)
    return index


def complexity_feature_indexing(dim, total_features):
    first_index = max(agreement_feature_indexing(dim).values())
    feat_per_detector = (total_features - first_index)//dim
    index = []
    for i in range(dim):
        start = first_index + i*feat_per_detector
        index.append(list(range(start, start+feat_per_detector)))

    return index


class MLfreeModel:
    def set_precisions(self, precisions):
        pass

    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def fit(self, X, y):
        pass

    def predict(self, X):
        total_agreement = np.zeros(X.shape[0])
        index = agreement_feature_indexing(self.dim_)
        for i in range(self.dim_):
            for j in range(i+1, self.dim_):
                if self.current_ in (i, j):
                    total_agreement += X[:, index[(i, j)]]

        return -total_agreement


class RelativePrecisionModel:
    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def fit(self, X, precisions):
        avg_precisions = precisions.mean(axis=1, keepdims=True)
        delta_precisions = precisions - avg_precisions
        my_delta = delta_precisions[:, self.current_]

        self.indices_ = complexity_feature_indexing(self.dim_, X.shape[1])[self.current_]
        X = X[:, self.indices_]

        self.model_ = SVR(kernel="rbf")
        self.model_.fit(X, my_delta)

    def predict(self, X):
        X = X[:, self.indices_]
        return self.model_.predict(X)


class DifferentialModel:
    def __init__(self, active_features):
        self.active_ = active_features

    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def feat_matrix_(self, X):
        avg = np.mean([X[:, o] for o in self.other_indices_], axis=0)
        X = X[:, self.my_indices_] - avg
        return X[:, self.active_]

    def fit(self, X, precisions):
        avg_precisions = precisions.mean(axis=1, keepdims=True)
        delta_precisions = precisions - avg_precisions
        my_delta = delta_precisions[:, self.current_]

        all_indices = complexity_feature_indexing(self.dim_, X.shape[1])
        self.my_indices_ = all_indices[self.current_]
        self.other_indices_ = [all_indices[i] for i in range(self.dim_) if i != self.current_]

        X = self.feat_matrix_(X)

        self.model_ = SVR(kernel="linear")
        #self.model_ = RandomForestRegressor()
        self.model_.fit(X, my_delta)

    def predict(self, X):
        X = self.feat_matrix_(X)
        return self.model_.predict(X)


class DeltaPrecisionModel:
    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def fit(self, X, precisions):
        max_precisions = precisions.max(axis=1, keepdims=True)
        delta_precisions = max_precisions - precisions
        my_delta = delta_precisions[:, self.current_]

        self.indices_ = list(agreement_feature_indexing(self.dim_).values())
        X = X[:, self.indices_]

        self.model_ = SVR(kernel="rbf")
        self.model_.fit(X, my_delta)

    def predict(self, X):
        X = X[:, self.indices_]
        return -self.model_.predict(X)


class DeltaPrecisionNarrowModel:
    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def fit(self, X, precisions):
        max_precisions = precisions.max(axis=1, keepdims=True)
        delta_precisions = max_precisions - precisions
        my_delta = delta_precisions[:, self.current_]

        agreement_i = agreement_feature_indexing(self.dim_)
        self.indices_ = []
        for key, index in agreement_i.items():
            if self.current_ in key:
                self.indices_.append(index)

        X = X[:, self.indices_]

        self.model_ = SVR(kernel="rbf")
        self.model_.fit(X, my_delta)

    def predict(self, X):
        X = X[:, self.indices_]
        return -self.model_.predict(X)


class AbsolutePrecisionModel:
    def __init__(self, active_features):
        self.active_ = active_features

    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def fit(self, X, precisions):
        my_precision = precisions[:, self.current_]

        self.indices_ = complexity_feature_indexing(self.dim_, X.shape[1])[self.current_]
        self.indices_ = np.array(self.indices_)
        X = X[:, self.indices_]
        X = X[:, self.active_]

        #SVR(kernel='linear', C=4.0, cache_size=7000)
        self.model_ = LinearSVR(C=4.0)
        self.model_.fit(X, my_precision)

    def predict(self, X):
        X = X[:, self.indices_]
        X = X[:, self.active_]

        return self.model_.predict(X)

    def plot(self, X, precisions, name):
        my_precision = precisions[:, self.current_]

        from matplotlib import pyplot as plt
        plt.scatter(self.predict(X), my_precision, marker='o', s=3)
        plt.xlabel("predicted performance")
        plt.ylabel("actual performance")
        plt.legend(loc=2)
        plt.title(name)
        plt.show()



class IndirectPrecisionModel:
    def set_detector_index(self, detector_index, out_of):
        self.current_ = detector_index
        self.dim_ = out_of

    def fit(self, X, precisions):
        precisions[:, self.current_] = -1  # only interested in other detetors' precision
        max_precisions = precisions.max(axis=1, keepdims=True)
        best = precisions.argmax(axis=1)

        filtering = np.where(max_precisions.squeeze() > 0.5)
        X = X[filtering]
        best = best[filtering]
        if X.shape[0] < 5:
            print("IndirectPrecisionModel: not enough data to learn from")
            return None

        # target values
        left = np.minimum(best, self.current_)
        right = np.maximum(best, self.current_)
        indexing = agreement_feature_indexing(self.dim_)
        agreement = [X[i, indexing[(left[i], right[i])]] for i in range(len(left))]

        # features
        self.indices_ = complexity_feature_indexing(self.dim_, X.shape[1])[self.current_]
        X = X[:, self.indices_]

        self.model_ = SVR()
        #self.model_ = RandomForestRegressor()
        self.model_.fit(X, agreement)

    def predict(self, X):
        X = X[:, self.indices_]
        return self.model_.predict(X)
