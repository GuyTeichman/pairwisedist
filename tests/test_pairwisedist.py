import pytest
from pairwisedist import pairwisedist
import numpy as np


def test_ys1_distance():
    assert False


def test_minmax_match_similarity():
    assert False


def test_correlation_star_pearson():
    assert False


def test_correlation_star_spearman():
    assert False


def test_correlation_start_bad_input():
    with pytest.raises(AssertionError):
        pairwisedist._correlation_star(np.zeros(5), 'parson')


def test_concordance_index_similarity():
    assert False


def test_similarity_to_distance():
    assert False
