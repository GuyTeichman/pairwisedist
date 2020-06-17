import pytest
from pairwisedist import pairwisedist
import numpy as np


def test_ys1_distance():
    assert False


def test_minmax_match_similarity():
    inp = np.array([[1, 2, 3, 4],
                    [5, 6.5, 5, 9],
                    [-8, -9, 0, -3],
                    [5, 5, 5, 5],
                    [1, 2, 4, 3]])
    truth = np.array([[1, 1, 0, 0.5, 0.5],
                      [1, 1, 0, 0.5, 0.5],
                      [0, 0, 1, 0, 0.5],
                      [0.5, 0.5, 0, 1, 0.5],
                      [0.5, 0.5, 0.5, 0.5, 1]])
    assert np.all(truth == pairwisedist._minmax_match_similarity(inp))


def test_correlation_star_pearson():
    assert False


def test_correlation_star_spearman():
    assert False


def test_correlation_start_bad_input():
    with pytest.raises(AssertionError):
        pairwisedist._correlation_star(np.zeros(5), 'parson')


def test_slope_concordance_similarity():
    assert False


def test_similarity_to_distance():
    assert False
