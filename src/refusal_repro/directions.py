"""Utilities for extracting refusal directions from model activations."""

import numpy as np


def compute_difference_in_means(positive_activations: np.ndarray, negative_activations: np.ndarray,) -> np.ndarray:
    """Compute a difference-in-means direction between two activation sets.

    Args:
        positive_activations: Activation matrix for the positive class with
            shape ``(n_positive, hidden_dim)``.
        negative_activations: Activation matrix for the negative class with
            shape ``(n_negative, hidden_dim)``.

    Returns:
        The difference between the positive-class mean activation and the
        negative-class mean activation, with shape ``(hidden_dim,)``.

    Raises:
        ValueError: If either input is not two-dimensional.
        ValueError: If the activation dimensions do not match.
        ValueError: If either activation set is empty.
    """
    if positive_activations.ndim != 2 or negative_activations.ndim != 2:
        raise ValueError("Activation arrays must be two-dimensional.")

    if positive_activations.shape[1] != negative_activations.shape[1]:
        raise ValueError("Activation arrays must have the same hidden dimension.")

    if positive_activations.shape[0] == 0 or negative_activations.shape[0] == 0:
        raise ValueError("Activation arrays must contain at least one example.")

    positive_mean = positive_activations.mean(axis=0)  #axis=0 computes the mean across the rows
    negative_mean = negative_activations.mean(axis=0)

    return positive_mean - negative_mean