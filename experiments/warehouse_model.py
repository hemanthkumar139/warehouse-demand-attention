import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.attention import scaled_dot_product_attention


np.random.seed(42)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv("results/warehouse_data.csv")

demand = df["demand"].values.astype(float)


# --------------------------------------------------
# Chronological split
# --------------------------------------------------

n = len(demand)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_demand = demand[:train_end]
val_demand = demand[train_end:val_end]
test_demand = demand[val_end:]


# Normalize using training statistics only
mean = train_demand.mean()
std = train_demand.std()

normalized = (demand - mean) / std


# --------------------------------------------------
# Create 24-hour sequences
# --------------------------------------------------

window = 24


def make_sequences(values, start, end):

    X = []
    y = []

    for t in range(start, end):

        if t < window:
            continue

        X.append(values[t-window:t])
        y.append(values[t])

    return np.array(X), np.array(y)


X_train, y_train = make_sequences(
    normalized,
    window,
    train_end
)

X_val, y_val = make_sequences(
    normalized,
    train_end,
    val_end
)

X_test, y_test = make_sequences(
    normalized,
    val_end,
    n
)


# Convert to (samples, sequence, features)
X_train = X_train[:, :, None]
X_val = X_val[:, :, None]
X_test = X_test[:, :, None]


print("Warehouse Attention Model")
print("=========================")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# --------------------------------------------------
# Attention parameters
# --------------------------------------------------

d_k = 8
d_v = 8

W_Q = np.random.randn(1, d_k) * 0.2
W_K = np.random.randn(1, d_k) * 0.2
W_V = np.random.randn(1, d_v) * 0.2

W_out = np.random.randn(d_v) * 0.2
b_out = 0.0


# --------------------------------------------------
# Forward prediction
# --------------------------------------------------

def predict(X):

    predictions = []

    for sequence in X:

        Y, A, Q, K, V, scores = scaled_dot_product_attention(
            sequence,
            W_Q,
            W_K,
            W_V
        )

        # Aggregate sequence representation
        context = np.mean(Y, axis=0)

        prediction = context @ W_out + b_out

        predictions.append(prediction)

    return np.array(predictions)


# --------------------------------------------------
# Training
# --------------------------------------------------

learning_rate = 0.01
epochs = 60

for epoch in range(epochs):

    predictions = predict(X_train)

    errors = predictions - y_train

    train_mse = np.mean(errors ** 2)

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

        dW_out += errors[i] * context
        db_out += errors[i]

    dW_out /= len(X_train)
    db_out /= len(X_train)

    W_out -= learning_rate * dW_out
    b_out -= learning_rate * db_out

    if (epoch + 1) % 10 == 0:

        val_predictions = predict(X_val)

        val_mse = np.mean(
            (val_predictions - y_val) ** 2
        )

        print(
            f"Epoch {epoch+1:3d} | "
            f"Train MSE: {train_mse:.5f} | "
            f"Val MSE: {val_mse:.5f}"
        )


# --------------------------------------------------
# Attention model evaluation
# --------------------------------------------------

attention_pred = predict(X_test)

attention_pred_original = (
    attention_pred * std + mean
)

y_test_original = (
    y_test * std + mean
)


attention_mae = np.mean(
    np.abs(attention_pred_original - y_test_original)
)

attention_rmse = np.sqrt(
    np.mean(
        (attention_pred_original - y_test_original) ** 2
    )
)


# --------------------------------------------------
# Last-value baseline
# --------------------------------------------------

baseline_pred = X_test[:, -1, 0] * std + mean

baseline_mae = np.mean(
    np.abs(baseline_pred - y_test_original)
)

baseline_rmse = np.sqrt(
    np.mean(
        (baseline_pred - y_test_original) ** 2
    )
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nTest Results")
print("============")

print("\nAttention model:")
print("MAE :", attention_mae)
print("RMSE:", attention_rmse)

print("\nLast-value baseline:")
print("MAE :", baseline_mae)
print("RMSE:", baseline_rmse)

print("\nMAE difference:")
print("Attention - Baseline:",
      attention_mae - baseline_mae)
