import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.attention import scaled_dot_product_attention


np.random.seed(42)

# Small problem for finite-difference verification
X = np.random.randn(3, 4)
W_Q = np.random.randn(4, 3)
W_K = np.random.randn(4, 3)
W_V = np.random.randn(4, 3)


def compute_loss(W_V):
    """
    L = 1/2 * sum(Y^2)
    """
    Y, _, _, _, _, _ = scaled_dot_product_attention(
        X, W_Q, W_K, W_V
    )

    return 0.5 * np.sum(Y ** 2)


# --------------------------------------------------
# Analytical gradient with respect to W_V
# --------------------------------------------------

Y, A, Q, K, V, scores = scaled_dot_product_attention(
    X, W_Q, W_K, W_V
)

# dL/dY = Y because L = 1/2 * sum(Y^2)
dY = Y

# Y = A @ V
dV = A.T @ dY

# V = X @ W_V
dW_V_analytical = X.T @ dV


# --------------------------------------------------
# Numerical gradient using central finite difference
# --------------------------------------------------

epsilon = 1e-5
dW_V_numerical = np.zeros_like(W_V)

for i in range(W_V.shape[0]):
    for j in range(W_V.shape[1]):

        W_plus = W_V.copy()
        W_minus = W_V.copy()

        W_plus[i, j] += epsilon
        W_minus[i, j] -= epsilon

        loss_plus = compute_loss(W_plus)
        loss_minus = compute_loss(W_minus)

        dW_V_numerical[i, j] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)


# --------------------------------------------------
# Compare gradients
# --------------------------------------------------

absolute_difference = np.max(
    np.abs(dW_V_analytical - dW_V_numerical)
)

relative_difference = absolute_difference / (
    np.max(np.abs(dW_V_analytical))
    + np.max(np.abs(dW_V_numerical))
    + 1e-12
)


print("Gradient Verification")
print("---------------------")

print("\nAnalytical gradient:")
print(dW_V_analytical)

print("\nNumerical gradient:")
print(dW_V_numerical)

print("\nMaximum absolute difference:",
      absolute_difference)

print("Relative difference:",
      relative_difference)

print(
    "\nGradient check passed:",
    np.allclose(
        dW_V_analytical,
        dW_V_numerical,
        rtol=1e-5,
        atol=1e-7
    )
)
