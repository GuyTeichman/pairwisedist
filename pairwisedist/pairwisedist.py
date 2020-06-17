import numpy as np
from scipy.stats.mstats import rankdata


def ys1_distance(data: np.ndarray, omega1: float = 0.5, omega2: float = 0.25, omega3: float = 0.25, rowvar: bool = True,
                 similarity: bool = False):
    """
    Calculates the pairwise YS1 distance matrix for a given array of n samples by p features, ranging between 0 and 1. \
    The YS1 dissimilarity is a metric that takes into account the Spearman rank correlation between the samples \
    (S* i,j), the positon of the minimal and maximal values of each sample (M i,j), \
    and the agreement of their slopes (A i,j). \
    The final score (Ys1 i,j) is a weighted average of these three paremeters: \
    YS1 i,j = omega1 * (S* i,j) + omega2 * (A i,j) + omega3 * (M i,j)

    :param data: an n-by-p numpy array of n samples by p features, to calculate pairwise distance on.
    :type data: np.ndarray
    :param omega1: Relative weight of the correlation (S* i,j) component of the YS1 distance. \
    All three relative weights (omega1-3) must add up to exactly 1.0.
    :type omega1: float between 0 and 1
    :param omega2: Relative weight of the slope concordance (A i,j) component of the YS1 distance. \
    All three relative weights (omega1-3) must add up to exactly 1.0.
    :type omega2: float between 0 and 1
    :param omega3: Relative weight of the minimum-maximum similarity (M i,j) component of the YS1 distance. \
    All three relative weights (omega1-3) must add up to exactly 1.0.
    :type omega3: float between 0 and 1
    :param rowvar: If True, calculates the pairwise distance between the rows of 'data'. \
    If False, calculate the pairwise distance between the columns of 'data'.
    :type rowvar: bool (default True)
    :param similarity: If False, returns a pairwise distance matrix (0 means closest, 1 means furthest). \
    If True, returns a pairwise similarity matrix (1 means most similar, 0 means most different).
    :type similarity: bool (default False)
    :return: an n-by-n numpy array of pairwise YS1 dissimilarity scores.
    :rtype: np.ndarray
    """
    assert (omega1 + omega2 + omega3) == 1, \
        f"All three omega values must sum to 1. Instead they sum to {omega1 + omega2 + omega3}"
    if not rowvar:
        data = data.T
    similarity_mat = omega1 * _correlation_star(data, 'spearman') + omega2 * _slope_concordance_similarity(
        data) + omega3 * _minmax_match_similarity(data)
    if similarity:
        return similarity_mat
    return _similarity_to_distance(similarity_mat)


def yr1_distance(data, omega1: float = 0.5, omega2: float = 0.25, omega3: float = 0.25, rowvar: bool = True,
                 similarity: bool = False):
    """
    Calculates the pairwise YR1 distance matrix for a given array of n samples by p features, ranging between 0 and 1. \
    The YS1 dissimilarity is a metric that takes into account the Pearson linear correlation between the samples \
    (R* i,j), the positon of the minimal and maximal values of each sample (M i,j), \
    and the agreement of their slopes (A i,j). \
    The final score (Ys1 i,j) is a weighted average of these three paremeters: \
    YS1 i,j = omega1 * (R* i,j) + omega2 * (A i,j) + omega3 * (M i,j)

    :param data: an n-by-p numpy array of n samples by p features, to calculate pairwise distance on.
    :type data: np.ndarray
    :param omega1: Relative weight of the correlation (R* i,j) component of the YR1 distance. \
    All three relative weights (omega1-3) must add up to exactly 1.0.
    :type omega1: float between 0 and 1
    :param omega2: Relative weight of the slope concordance (A i,j) component of the YR1 distance. \
    All three relative weights (omega1-3) must add up to exactly 1.0.
    :type omega2: float between 0 and 1
    :param omega3: Relative weight of the minimum-maximum similarity (M i,j) component of the YR1 distance. \
    All three relative weights (omega1-3) must add up to exactly 1.0.
    :type omega3: float between 0 and 1
    :param rowvar: If True, calculates the pairwise distance between the rows of 'data'. \
    If False, calculate the pairwise distance between the columns of 'data'.
    :type rowvar: bool (default True)
    :param similarity: If False, returns a pairwise distance matrix (0 means closest, 1 means furthest). \
    If True, returns a pairwise similarity matrix (1 means most similar, 0 means most different).
    :type similarity: bool (default False)
    :return: an n-by-n numpy array of pairwise YR1 dissimilarity scores.
    :rtype: np.ndarray
    """

    assert isinstance(data, np.ndarray), f"'data' must be a numpy array. Instead got {type(data)}."
    assert (omega1 + omega2 + omega3) == 1, \
        f"All three omega values must sum to 1. Instead they sum to {omega1 + omega2 + omega3}"
    if not rowvar:
        data = data.T
    similarity_mat = omega1 * _correlation_star(data, 'pearson') + omega2 * _slope_concordance_similarity(
        data) + omega3 * _minmax_match_similarity(data)
    if similarity:
        return similarity_mat
    return _similarity_to_distance(similarity_mat)


def _minmax_match_similarity(data: np.ndarray):
    return (np.argmax(data, axis=1)[:, None] == np.argmax(data, axis=1)[None, :]) * 0.5 + (
            np.argmin(data, axis=1)[:, None] == np.argmin(data, axis=1)[None, :]) * 0.5


def _correlation_star(data: np.ndarray, method: str):
    assert isinstance(method, str), f"'method' must be a string. Instead got {type(method)}."
    if method == 'spearman':
        return (np.corrcoef(rankdata(data, axis=1)) + 1) / 2
    return (np.corrcoef(data) + 1) / 2


def _slope_concordance_similarity(data: np.ndarray):
    incline_array = (1 * (data[:, 1:] > data[:, :-1]) - 1 * (data[:, 1:] < data[:, :-1]))
    return np.mean((incline_array[None, :, :] == incline_array[:, None, :]), axis=2)


def _similarity_to_distance(similarity_matrix, max_val: int = 1):
    return max_val - similarity_matrix
