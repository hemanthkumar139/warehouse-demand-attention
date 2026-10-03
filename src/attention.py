import numpy as np


def stable_softmax(x, axis=-1):
    """
    Numerically stable softmax.
    Subtracting the maximum prevents large exponentials.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(X, W_Q, W_K, W_V, scale=True):
    """
    First-principles scaled dot-product self-attention.

    Equations:
        Q = X W_Q
        K = X W_K
        V = X W_V
        S = Q K^T
        S' = S / sqrt(d_k)
        A = softmax(S')
        Y = A V
    """

    # 1. Linear projections
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # 2. Similarity scores
    scores = Q @ K.T

    # 3. Scaling
    if scale:
        d_k = Q.shape[-1]
        scores = scores / np.sqrt(d_k)

    # 4. Softmax
    attention_weights = stable_softmax(scores, axis=-1)

    # 5. Weighted aggregation
    Y = attention_weights @ V

    return Y, attention_weights, Q, K, V, scores


if __name__ == "__main__":

    np.random.seed(42)

    # Sequence length = 4
    # Input dimension = 8
    # Key/query/value dimension = 4
    X = np.random.randn(4, 8)

    W_Q = np.random.randn(8, 4)
    W_K = np.random.randn(8, 4)
    W_V = np.random.randn(8, 4)

    Y, A, Q, K, V, scores = scaled_dot_product_attention(
        X, W_Q, W_K, W_V
    )

    print("Input X shape:", X.shape)
    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("Scores shape:", scores.shape)
    print("Attention weights shape:", A.shape)
    print("Output Y shape:", Y.shape)

    print("\nAttention weights:")
    print(A)

    print("\nRow sums:")
    print(A.sum(axis=1))

    print("\nAll rows sum to 1:",
          np.allclose(A.sum(axis=1), 1.0))