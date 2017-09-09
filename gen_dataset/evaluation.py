import numpy as np


def precision_at_k(ground_truth, scores, adjusted=True):
    # expected number of outliers
    k = np.sum(ground_truth)

    # ranking
    top_ranked = scores.argsort()[::-1]

    # precision
    precision = np.mean(ground_truth[top_ranked[:k]])

    # adjustment for chance
    if adjusted:
        chance = k/len(ground_truth)
        precision = (precision - chance)/(1 - chance)

    return precision


def recall_at_k(ground_truth, scores, ratio=0.05):
    # expected number of outliers
    to_be_found = np.sum(ground_truth)

    # actually found
    k = int(ratio*len(ground_truth))
    top_ranked = scores.argsort()[::-1]
    found = np.sum(ground_truth[top_ranked[:k]])

    # recall
    recall = found/to_be_found

    # adjusment for chance
    chance = ratio  # (ratio*to_be_found)/to_be_found
    recall = (recall-ratio)/(1-chance)

    return recall

def precision_at_mk(ground_truth, scores, m=4, adjusted=True):
    # expected number of outliers
    k = np.sum(ground_truth)

    # ranking
    top_ranked = scores.argsort()[::-1]

    # precision
    precision = m*np.mean(ground_truth[top_ranked[:m*k]])

    # adjustment for chance
    if adjusted:
        chance = m*k/len(ground_truth)
        precision = (precision - chance)/(1 - chance)

    return precision


def avg_precision(ground_truth, scores, adjusted=True):
    # expected number of outliers
    k = np.sum(ground_truth)

    # ranking
    top_ranked = scores.argsort()[::-1]

    # average precision
    avg = 0
    for n in range(1, k+1):
        avg += np.mean(ground_truth[top_ranked[:n]])
    avg /= k

    # adjustment for chance
    if adjusted:
        chance = k/len(ground_truth)
        precision = (avg - chance)/(1 - chance)

    return precision
