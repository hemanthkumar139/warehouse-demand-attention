import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.attention import (
    stable_softmax,
    scaled_dot_product_attention
)


def test_softmax_rows_sum_to_one():
    x = np.random.randn(4, 5)

    result = stable_softmax(x)

    assert np.allclose(
        result.sum(axis=1),
        1.0
    )


def test_attention_shapes():
    X = np.random.randn(6, 8)

    W_Q = np.random.randn(8, 4)
    W_K = np.random.randn(8, 4)
    W_V = np.random.randn(8, 4)

    Y, A, Q, K, V, scores = scaled_dot_product_attention(
        X, W_Q, W_K, W_V
    )

    assert Q.shape == (6, 4)
    assert K.shape == (6, 4)
    assert V.shape == (6, 4)
    assert scores.shape == (6, 6)
    assert A.shape == (6, 6)
    assert Y.shape == (6, 4)


def test_attention_probabilities():
    X = np.random.randn(5, 3)

    W_Q = np.random.randn(3, 2)
    W_K = np.random.randn(3, 2)
    W_V = np.random.randn(3, 2)

    _, A, _, _, _, _ = scaled_dot_product_attention(
        X, W_Q, W_K, W_V
    )

    assert np.all(A >= 0)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_deterministic_random_values():
    np.random.seed(42)

    a = np.random.randn(10)

    np.random.seed(42)

    b = np.random.randn(10)

    assert np.array_equal(a, b)
