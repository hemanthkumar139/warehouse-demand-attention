import numpy as np
import pandas as pd


def generate_warehouse_data(
    days=365,
    seed=42,
    noise_std=5.0,
    spike_probability=0.01,
    spike_scale=50.0,
):
    rng = np.random.default_rng(seed)

    hours = days * 24

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=hours,
        freq="h"
    )

    t = np.arange(hours)

    # Base demand
    base = 100.0

    # Daily seasonality
    daily = 25 * np.sin(2 * np.pi * t / 24)

    # Weekly seasonality
    weekly = 15 * np.sin(2 * np.pi * t / (24 * 7))

    # Small trend
    trend = 0.005 * t

    # Random noise
    noise = rng.normal(0, noise_std, size=hours)

    # Rare positive and negative events
    event = np.zeros(hours)

    event_mask = rng.random(hours) < spike_probability
    positive_events = rng.random(hours) < 0.75

    event[event_mask & positive_events] = spike_scale
    event[event_mask & ~positive_events] = -spike_scale

    demand = base + daily + weekly + trend + event + noise

    # Demand cannot be negative
    demand = np.maximum(demand, 0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand": demand,
    })

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df


if __name__ == "__main__":
    df = generate_warehouse_data()

    print(df.head())
    print()
    print(df.describe())

    df.to_csv("results/warehouse_data.csv", index=False)
    print("\nSaved dataset to results/warehouse_data.csv")