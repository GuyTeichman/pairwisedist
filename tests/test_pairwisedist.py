import pytest
from pairwisedist import pairwisedist
import numpy as np

inp = np.array([[1, 2, 3, 4],
                [5, 6.5, 5, 9],
                [-8, -9, 0, -3],
                [5, 5, 5, 5],
                [1, 2, 4, 3]])


def test_sharpened_cosine_distance():
    truth = truth = np.array([[1, 0.33560609, 0.50002998, 0.42575679, 0.33708719],
                              [0.33560609, 1, 0.50574587, 0.27300692, 0.43677134],
                              [0.50002998, 0.50574587, 1, 0.51185475, 0.50000804],
                              [0.42575679, 0.27300692, 0.51185475, 1, 0.42575679],
                              [0.33708719, 0.43677134, 0.50000804, 0.42575679, 1]])
    print(pairwisedist.sharpened_cosine_distance(inp))
    assert np.isclose(pairwisedist.sharpened_cosine_distance(inp), truth, equal_nan=True).all()
    assert np.isclose(pairwisedist.sharpened_cosine_distance(inp.T),
                      pairwisedist.sharpened_cosine_distance(inp, rowvar=False), equal_nan=True).all()


def test_spearman_distance():
    truth_corr = np.corrcoef(pairwisedist.rankdata(inp, axis=1))
    truth_sim = (truth_corr + 1) / 2
    truth_dist = pairwisedist._similarity_to_distance(truth_sim)

    assert np.isclose(pairwisedist.spearman_distance(inp), truth_dist, equal_nan=True).all()
    assert np.isclose(pairwisedist.spearman_distance(inp, similarity=True), truth_sim, equal_nan=True).all()
    assert np.isclose(pairwisedist.spearman_distance(inp.T), pairwisedist.spearman_distance(inp, rowvar=False),
                      equal_nan=True).all()


def test_pearson_distance():
    truth_corr = np.corrcoef(inp)
    truth_sim = (truth_corr + 1) / 2
    truth_dist = pairwisedist._similarity_to_distance(truth_sim)

    assert np.isclose(pairwisedist.pearson_distance(inp), truth_dist, equal_nan=True).all()
    assert np.isclose(pairwisedist.pearson_distance(inp, similarity=True), truth_sim, equal_nan=True).all()
    assert np.isclose(pairwisedist.pearson_distance(inp.T), pairwisedist.pearson_distance(inp, rowvar=False),
                      equal_nan=True).all()


def test_rowvar():
    truth = inp
    truth_t = inp.transpose()
    assert np.all(truth == pairwisedist._rowvar(inp, True))
    assert np.all(truth_t == pairwisedist._rowvar(inp, False))


def test_similarity():
    truth = np.array([[1., 0., 0.4, np.nan, 0.4465],
                      [0., -1., -0.58520574, 0.2, -0.3714],
                      [-0.87, -0.5120574, 1., 1, 0.7777],
                      [0, 0, 0, 0.432, -0.89],
                      [-0.5, 0.37, 0.77377, np.nan, 1.]])

    assert np.isclose(truth, pairwisedist._similarity(truth, True), equal_nan=True).all()
    assert np.isclose(1 - truth, pairwisedist._similarity(truth, False), equal_nan=True).all()


def test_jackknife_distance():
    truth_corr = np.array([[1., 0., 0.65465367, np.nan, 0.5],
                           [0., 1., -0.58520574, np.nan, -0.37115374],
                           [0.65465367, -0.58520574, 1., np.nan, 0.77771377],
                           [np.nan, np.nan, np.nan, np.nan, np.nan],
                           [0.5, -0.37115374, 0.77771377, np.nan, 1.]])
    truth_sim = (truth_corr + 1) / 2
    truth_dist = pairwisedist._similarity_to_distance(truth_sim)

    assert np.isclose(pairwisedist.jackknife_distance(inp), truth_dist, equal_nan=True).all()
    assert np.isclose(pairwisedist.jackknife_distance(inp, similarity=True), truth_sim, equal_nan=True).all()
    assert np.isclose(pairwisedist.jackknife_distance(inp.T), pairwisedist.jackknife_distance(inp, rowvar=False),
                      equal_nan=True).all()


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


def test_jackknife():
    truth = np.array([2, 5.5, -6 - (2 / 3), 5, 2])
    res = pairwisedist._jackknife(inp, np.mean, axis=1)
    print("truth: ")
    print(truth)
    print(res)
    assert np.isclose(truth, res).all()
