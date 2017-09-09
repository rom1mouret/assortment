from sklearn.preprocessing import robust_scale
from sklearn.linear_model import Ridge
from scipy.stats import shapiro
from scipy.stats import rankdata
import numpy as np


def session_features(X, scores):
    """ assuming X has been scaled properly """

    subsample = np.random.choice(np.arange(scores.shape[0]), size=3000)
    X_small = X[subsample]

    # average Gaussianity of each feature
    g = 0
    n = 0
    for j in range(X.shape[1]):
        col = X_small[:, j]
        if np.min(col) != np.max(col):
            g += shapiro(col)[0]
            n += 1
    if n == 0:
        m0 = np.nan
    else:
        m0 = g/n


    # from statsmodels.stats.outliers_influence import variance_inflation_factor
    #print("computing VIF")
    #vif = [variance_inflation_factor(X_small, col) for col in range(X.shape[1])]
    #m1 = np.mean(vif)
    #print("VIF avg", m1)

    # percentiles of scores
    m2 = np.percentile(scores, 70)
    m3 = np.percentile(scores, 90)
    m4 = np.percentile(scores, 95)

    # histogram of scores
    hist, edges = np.histogram(scores, bins=5)
    hist += 1
    hist = np.log(hist)
    hist /= np.sum(hist)

    # do scores explain well the data?
    m5 = score_explanation(X, scores)

    # TODO: accuracy of KDE or GMM

    return [m0, m2, m3, m4, m5]+hist.tolist()


def score_explanation(X, scores):
    def regression_error(features, y):
        #model = RandomForestRegressor()# DecisionTreeRegressor() #Ridge()
        model = Ridge()
        model.fit(features, y)
        return np.mean(np.square(y - model.predict(features)))

    # resample to speed up the regression, while keeping the alleged anomalies
    ranks = rankdata(scores)
    probas = ranks / np.sum(ranks)
    score_selection = np.random.choice(np.arange(len(scores)), size=7000, p=probas)
    X = X[score_selection]
    scores = scores[score_selection]
    scores = robust_scale(scores.reshape(scores.shape[0], 1)).squeeze()

    # make room for storing score cubes and score log in the feature matrix
    scores_cube = np.power(scores, 3)
    scores_cube_idx = X.shape[1]
    scores_log = np.log(1 + scores - np.min(scores))
    scores_log_idx = scores_cube_idx + 1
    scores_tan = np.tanh(scores) + 1
    scores_tan_idx = scores_log_idx + 1
    extra = ((scores_cube_idx, scores_cube), (scores_log_idx, scores_log), (scores_tan_idx, scores_tan))
    X = np.concatenate([X, np.empty((X.shape[0], len(extra)))], axis=1)

    predictability = 0
    for j in range(X.shape[1]):
        col = X[:, j].copy()
        X[:, j] = 0  # or np.random.randn(X.shape[0])?
        for idx, _ in extra:
            X[:, idx] = 0
        no_score = regression_error(X, col)
        X[:, j] = scores
        for idx, data in extra:
            X[:, idx] = data
        with_score = regression_error(X, col)
        predictability += np.log(1 + max(0, no_score - with_score))
        X[:, j] = col  # cancel changes

    predictability /= X.shape[1]

    return predictability
