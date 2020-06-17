import numpy as np
from scipy.stats.mstats import rankdata


def ys1_distance(data, omega1: float = 0.5, omega2: float = 0.25, omega3: float = 0.25):
    assert (omega1 + omega2 + omega3) == 1, \
        f"All three omega values must sum to 1. Instead they sum to {omega1 + omega2 + omega3}"
    similarity = omega1 * _correlation_star(data, 'spearman') + omega2 * _concordance_index_similarity(
        data) + omega3 * _minmax_match_similarity(data)
    return _similarity_to_distance(similarity)


def _minmax_match_similarity(data: np.ndarray):
    return (np.argmax(data, axis=1)[:, None] == np.argmax(data, axis=1)[None, :]) * 0.5 + (
            np.argmin(data, axis=1)[:, None] == np.argmin(data, axis=1)[None, :]) * 0.5


def _correlation_star(data: np.ndarray, method: str):
    assert isinstance(method, str), f"'method' must be a string. Instead got {type(method)}."
    if method == 'spearman':
        return (np.corrcoef(rankdata(data, axis=1)) + 1) / 2
    return (np.corrcoef(data) + 1) / 2


def _concordance_index_similarity(data: np.ndarray):
    incline_array = (1 * (data[:, 1:] > data[:, :-1]) - 1 * (data[:, 1:] < data[:, :-1]))
    return np.mean((incline_array[None, :, :] == incline_array[:, None, :]), axis=2)


def _similarity_to_distance(similarity_matrix, max_val: int = 1):
    return max_val - similarity_matrix
