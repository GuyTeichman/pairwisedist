import pytest
from pairwisedist import pairwisedist
import numpy as np

inp = np.array([[1, 2, 3, 4],
                [5, 6.5, 5, 9],
                [-8, -9, 0, -3],
                [5, 5, 5, 5],
                [1, 2, 4, 3]])


def test_ys1_distance():
    truth_default = np.array([[0., 0.17521945, 0.51666667, np.nan, 0.25833333],
                              [0.17521945, 0., 0.80270463, np.nan, 0.51531435],
                              [0.51666667, 0.80270463, 0., np.nan, 0.25833333],
                              [np.nan, np.nan, np.nan, np.nan, np.nan],
                              [0.25833333, 0.51531435, 0.25833333, np.nan, 0.]])
    truth_thirds = np.array([[0., 0.17236852, 0.62222222, np.nan, 0.31111111],
                             [0.17236852, 0., 0.86846975, np.nan, 0.53798735],
                             [0.62222222, 0.86846975, 0., np.nan, 0.31111111],
                             [np.nan, np.nan, np.nan, np.nan, np.nan],
                             [0.31111111, 0.53798735, 0.31111111, np.nan, 0.]])
    omega = 1 / 3

    assert np.isclose(truth_default, pairwisedist.ys1_distance(inp), equal_nan=True).all()
    assert np.isclose(truth_thirds, pairwisedist.ys1_distance(inp, omega, omega, omega), equal_nan=True).all()
    assert np.isclose(1 - truth_default, pairwisedist.ys1_distance(inp, similarity=True), equal_nan=True).all()
    assert np.isclose(pairwisedist.ys1_distance(inp.T), pairwisedist.ys1_distance(inp, rowvar=False),
                      equal_nan=True).all()


def test_yr1_distance():
    truth_default = np.array([[0., 0.15378712, 0.48409248, np.nan, 0.25833333],
                              [0.15378712, 0., 0.72918698, np.nan, 0.49891757],
                              [0.48409248, 0.72918698, 0., np.nan, 0.2301156],
                              [np.nan, np.nan, np.nan, np.nan, np.nan],
                              [0.25833333, 0.49891757, 0.2301156, np.nan, 0.]])
    truth_thirds = np.array([[0., 0.1580803, 0.6005061, np.nan, 0.31111111],
                             [0.1580803, 0., 0.81945799, np.nan, 0.52705616],
                             [0.6005061, 0.81945799, 0., np.nan, 0.29229929],
                             [np.nan, np.nan, np.nan, np.nan, np.nan],
                             [0.31111111, 0.52705616, 0.29229929, np.nan, 0.]])
    omega = 1 / 3

    assert np.isclose(truth_default, pairwisedist.yr1_distance(inp), equal_nan=True).all()
    assert np.isclose(truth_thirds, pairwisedist.yr1_distance(inp, omega, omega, omega), equal_nan=True).all()
    assert np.isclose(1 - truth_default, pairwisedist.yr1_distance(inp, similarity=True), equal_nan=True).all()
    assert np.isclose(pairwisedist.yr1_distance(inp.T), pairwisedist.yr1_distance(inp, rowvar=False),
                      equal_nan=True).all()


def test_minmax_match_similarity():
    truth = np.array([[1, 1, 0, 0.5, 0.5],
                      [1, 1, 0, 0.5, 0.5],
                      [0, 0, 1, 0, 0.5],
                      [0.5, 0.5, 0, 1, 0.5],
                      [0.5, 0.5, 0.5, 0.5, 1]])

    assert np.all(truth == pairwisedist._minmax_match_similarity(inp))


def test_correlation_star_pearson():
    truth = (1 + np.array([[1., 0.71818485, 0.73029674, np.nan, 0.8],
                           [0.71818485, 1., 0.08325207, np.nan, 0.17099639],
                           [0.73029674, 0.08325207, 1., np.nan, 0.91287093],
                           [np.nan, np.nan, np.nan, np.nan, np.nan],
                           [0.8, 0.17099639, 0.91287093, np.nan, 1.]])) / 2
    assert np.isclose(truth, pairwisedist._correlation_star(inp, 'Pearson'), equal_nan=True).all()


def test_correlation_star_spearman():
    truth = (1 + np.array([[1., 0.63245553, 0.6, np.nan, 0.8],
                           [0.63245553, 1., -0.21081851, np.nan, 0.10540926],
                           [0.6, -0.21081851, 1., np.nan, 0.8],
                           [np.nan, np.nan, np.nan, np.nan, np.nan],
                           [0.8, 0.10540926, 0.8, np.nan, 1.]])) / 2
    assert np.isclose(truth, pairwisedist._correlation_star(inp, 'Spearman'), equal_nan=True).all()


def test_correlation_start_bad_input():
    with pytest.raises(AssertionError):
        pairwisedist._correlation_star(np.zeros(5), 'parson')


def test_slope_concordance_similarity():
    truth = np.array([[1, 2 / 3, 1 / 3, 0, 2 / 3],
                      [2 / 3, 1, 0, 0, 1 / 3],
                      [1 / 3, 0, 1, 0, 2 / 3],
                      [0, 0, 0, 1, 0],
                      [2 / 3, 1 / 3, 2 / 3, 0, 1]])

    assert np.all(pairwisedist._slope_concordance_similarity(inp) == truth)


def test_similarity_to_distance():
    res = np.array([[1, 2 / 3, 1 / 3, 0, 2 / 3],
                    [-1, 1, 0, 0, 1 / 3],
                    [1 / 3, 0, 1, 0, 2 / 3],
                    [0, 0, 0, 1, 0],
                    [2 / 3, 1 / 3, 2 / 3, 0, 1]])
    truth = np.array([[0, 1 / 3, 2 / 3, 1, 1 / 3],
                      [2, 0, 1, 1, 2 / 3],
                      [2 / 3, 1, 0, 1, 1 / 3],
                      [1, 1, 1, 0, 1],
                      [1 / 3, 2 / 3, 1 / 3, 1, 0]])

    assert np.isclose(pairwisedist._similarity_to_distance(res), truth).all()
