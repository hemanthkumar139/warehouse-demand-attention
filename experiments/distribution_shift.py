import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.warehouse_data import generate_warehouse_data


def evaluate_last_value(df):

    demand = df["demand"].values

    # Predict next hour using current hour
    predictions = demand[:-1]
    actual = demand[1:]

    mae = np.mean(np.abs(predictions - actual))

    rmse = np.sqrt(
        np.mean((predictions - actual) ** 2)
    )

    return mae, rmse


# --------------------------------------------------
# Original distribution
# --------------------------------------------------

normal_data = generate_warehouse_data(
    days=365,
    seed=42,
    noise_std=5.0,
    spike_probability=0.01,
    spike_scale=50.0
)


# --------------------------------------------------
# Shifted distribution
# --------------------------------------------------

shifted_data = generate_warehouse_data(
    days=365,
    seed=43,
    noise_std=15.0,
    spike_probability=0.03,
    spike_scale=100.0
)


normal_mae, normal_rmse = evaluate_last_value(normal_data)

shifted_mae, shifted_rmse = evaluate_last_value(shifted_data)


print("Distribution Shift Experiment")
print("=============================")

print("\nOriginal distribution:")
print("Noise std: 5")
print("Spike probability: 0.01")
print("Spike scale: 50")

print("\nShifted distribution:")
print("Noise std: 15")
print("Spike probability: 0.03")
print("Spike scale: 100")

print("\nLast-value baseline:")

print("\nOriginal:")
print("MAE :", normal_mae)
print("RMSE:", normal_rmse)

print("\nShifted:")
print("MAE :", shifted_mae)
print("RMSE:", shifted_rmse)

print("\nChange:")
print("MAE increase:",
      shifted_mae - normal_mae)

print("RMSE increase:",
      shifted_rmse - normal_rmse)
