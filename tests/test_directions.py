"""Tests for refusal-direction extraction utilities."""

import numpy as np
import pytest

from refusal_repro.directions import compute_difference_in_means


def test_compute_difference_in_means_returns_expected_direction() -> None:
    """Difference in means should match the analytically expected result."""
    positive = np.array(
        [
            [3.0, 5.0],
            [5.0, 7.0],
        ]
    )
    negative = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    direction = compute_difference_in_means(positive, negative)

    expected = np.array([2.0, 3.0])
    np.testing.assert_allclose(direction, expected)  #we dont use direction == expected because of floating point precision issues


def test_compute_difference_in_means_handles_larger_matrices() -> None:
    """Difference in means should work for larger activation matrices."""
    positive = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [3.0, 6.0, 9.0, 12.0, 15.0],
            [4.0, 8.0, 12.0, 16.0, 20.0],
        ]
    )

    negative = np.array(
        [
            [0.0, 1.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 2.0, 3.0, 5.0],
            [1.0, 2.0, 3.0, 5.0, 8.0],
            [2.0, 3.0, 5.0, 8.0, 13.0],
        ]
    )

    direction = compute_difference_in_means(positive, negative)

    expected = np.array([1.5, 3.25, 4.75, 5.5, 5.25])
    np.testing.assert_allclose(direction, expected)


def test_compute_difference_in_means_rejects_non_2d_input() -> None:
    """Inputs must be two-dimensional activation matrices."""
    positive = np.array([1.0, 2.0, 3.0])
    negative = np.zeros((2, 3))

    with pytest.raises(ValueError, match="two-dimensional"):
        compute_difference_in_means(positive, negative)


def test_compute_difference_in_means_rejects_mismatched_dimensions() -> None:
    """Inputs with different hidden dimensions should raise an error."""
    positive = np.zeros((2, 3))
    negative = np.zeros((2, 4))

    with pytest.raises(ValueError, match="same hidden dimension"):
        compute_difference_in_means(positive, negative)


@pytest.mark.parametrize( #last test should be checked with 2 cases: one where positive is empty and one where negative is empty
    ("positive", "negative"),
    [
        (np.empty((0, 3)), np.zeros((2, 3))),
        (np.zeros((2, 3)), np.empty((0, 3))),
    ],
)
def test_compute_difference_in_means_rejects_empty_inputs(
    positive: np.ndarray,
    negative: np.ndarray,
) -> None:
    """Neither activation set may be empty."""
    with pytest.raises(ValueError, match="at least one example"):
        compute_difference_in_means(positive, negative)