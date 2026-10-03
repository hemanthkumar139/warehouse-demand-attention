import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.attention import scaled_dot_product_attention


np.random.seed(42)


def run_experiment(d_k, scale):

    X = np.random.randn(8, 16)

    W_Q = np.random.randn(16, d_k)
    W_K = np.random.randn(16, d_k)
    W_V = np.random.randn(16, d_k)

    Y, A, Q, K, V, scores = scaled_dot_product_attention(
        X, W_Q, W_K, W_V, scale=scale
    )

    # Entropy measures how concentrated the attention is
    entropy = -np.sum(
        A * np.log(A + 1e-12),
        axis=1
    )

    return {
        "mean_score_abs": np.mean(np.abs(scores)),
        "max_score": np.max(scores),
        "mean_attention_entropy": np.mean(entropy),
        "min_attention": np.min(A),
        "max_attention": np.max(A),
    }


print("Scaled vs Unscaled Attention Ablation")
print("======================================")

for d_k in [4, 16, 64, 128]:

    scaled = run_experiment(d_k, True)
    unscaled = run_experiment(d_k, False)

    print(f"\nd_k = {d_k}")

    print("\nScaled:")
    print("  Mean |score|:",
          scaled["mean_score_abs"])
    print("  Max score:",
          scaled["max_score"])
    print("  Attention entropy:",
          scaled["mean_attention_entropy"])
    print("  Max attention:",
          scaled["max_attention"])

    print("\nUnscaled:")
    print("  Mean |score|:",
          unscaled["mean_score_abs"])
    print("  Max score:",
          unscaled["max_score"])
    print("  Attention entropy:",
          unscaled["mean_attention_entropy"])
    print("  Max attention:",
          unscaled["max_attention"])

