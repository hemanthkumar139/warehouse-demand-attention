import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.attention import scaled_dot_product_attention


np.random.seed(42)

# --------------------------------------------------
# 1. Create toy sequence dataset
# --------------------------------------------------

n_samples = 300
seq_len = 6
input_dim = 1
d_k = 4
d_v = 4

X_data = np.random.randn(n_samples, seq_len, input_dim)

# Target = average of last two values
y_data = (
    X_data[:, -1, 0] +
    X_data[:, -2, 0]
) / 2


# --------------------------------------------------
# 2. Train / validation split
# --------------------------------------------------

split = 240

X_train = X_data[:split]
y_train = y_data[:split]

X_val = X_data[split:]
y_val = y_data[split:]


# --------------------------------------------------
# 3. Initialize attention parameters
# --------------------------------------------------

W_Q = np.random.randn(input_dim, d_k) * 0.3
W_K = np.random.randn(input_dim, d_k) * 0.3
W_V = np.random.randn(input_dim, d_v) * 0.3

# Prediction head
W_out = np.random.randn(d_v) * 0.3
b_out = 0.0


# --------------------------------------------------
# 4. Training
# --------------------------------------------------

learning_rate = 0.05
epochs = 100


def get_prediction(X):
    predictions = []

    for sequence in X:

        Y, A, Q, K, V, scores = scaled_dot_product_attention(
            sequence,
            W_Q,
            W_K,
            W_V
        )

        # Mean attention representation
        context = np.mean(Y, axis=0)

        prediction = context @ W_out + b_out

        predictions.append(prediction)

    return np.array(predictions)


print("Toy Attention Learning Experiment")
print("=================================")

for epoch in range(epochs):

    # Forward pass
    predictions = get_prediction(X_train)

    error = predictions - y_train

    loss = np.mean(error ** 2)

    # --------------------------------------------------
    # Gradient for output layer
    # --------------------------------------------------

    dW_out = np.zeros_like(W_out)
    db_out = 0.0

    for i, sequence in enumerate(X_train):

        Y, A, Q, K, V, scores = scaled_dot_product_attention(
            sequence,
            W_Q,
            W_K,
            W_V
        )

        context = np.mean(Y, axis=0)

        dW_out += error[i] * context
        db_out += error[i]

    dW_out /= len(X_train)
    db_out /= len(X_train)

    # Update output layer
    W_out -= learning_rate * dW_out
    b_out -= learning_rate * db_out

    if (epoch + 1) % 10 == 0:

        val_predictions = get_prediction(X_val)

        val_mse = np.mean(
            (val_predictions - y_val) ** 2
        )

        val_mae = np.mean(
            np.abs(val_predictions - y_val)
        )

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train MSE: {loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )


# --------------------------------------------------
# 5. Final evaluation
# --------------------------------------------------

val_predictions = get_prediction(X_val)

final_mae = np.mean(
    np.abs(val_predictions - y_val)
)

final_rmse = np.sqrt(
    np.mean((val_predictions - y_val) ** 2)
)

print("\nFinal Results")
print("-------------")
print("Validation MAE:", final_mae)
print("Validation RMSE:", final_rmse)
