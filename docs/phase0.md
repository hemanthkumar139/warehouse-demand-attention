# Phase 0 — Design Before Code

## 1. Problem Formulation

The objective is to predict the next hour of warehouse order demand from the previous 24 hourly observations.

For each time step \(t\), the model receives a sequence of the previous 24 demand observations:

\[
X_t = [D_{t-23}, D_{t-22}, ..., D_{t-1}, D_t]
\]

and predicts:

\[
\hat{D}_{t+1}
\]

The project consists of two connected components:

1. A first-principles implementation of scaled dot-product self-attention.
2. A warehouse demand prediction model that uses the implemented attention mechanism.

The attention mechanism will be implemented using raw tensor/matrix operations rather than a high-level Transformer or attention API.

---

## 2. Mathematical Attention Formulation

Given an input sequence \(X\), query, key, and value representations are calculated as:

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

The similarity between sequence positions is calculated using:

\[
S = QK^T
\]

The similarity scores are scaled by the square root of the key dimension:

\[
S' = \frac{QK^T}{\sqrt{d_k}}
\]

Attention weights are then calculated using softmax:

\[
A = softmax(S')
\]

Finally, the attention output is:

\[
Y = AV
\]

The scaling operation is expected to prevent excessively large dot-product values from causing overly concentrated softmax distributions and undesirable gradient behavior.

---

## 3. Synthetic Warehouse Environment

A synthetic warehouse environment will be generated rather than using real warehouse data.

The demand-generating process will contain meaningful temporal structure.

The demand at time \(t\) will be modeled conceptually as:

\[
D_t =
B +
S_{daily}(t) +
S_{weekly}(t) +
T(t) +
E(t) +
\epsilon_t
\]

where:

- \(B\) is the base demand.
- \(S_{daily}(t)\) represents daily/hourly seasonality.
- \(S_{weekly}(t)\) represents weekly seasonality.
- \(T(t)\) represents a small trend component.
- \(E(t)\) represents occasional demand events such as spikes or drops.
- \(\epsilon_t\) represents random noise.

The generated data will contain hourly observations.

The generator will use a fixed random seed when reproducibility is required.

---

## 4. Statistical Assumptions

### Temporal structure

Warehouse demand is assumed to contain:

- hourly/daily patterns;
- weekly patterns;
- a small trend component;
- dependence between nearby observations.

### Noise

Random noise will be added to demand observations.

The initial experiment will use approximately constant noise variance.

### Unusual events

Occasional demand spikes and drops will be introduced to represent unusual warehouse conditions.

These events will occur relatively infrequently compared with normal demand observations.

### Historical information

Recent observations are expected to be useful for predicting the next hour.

Observations corresponding to similar times of day and days of the week may also contain useful information because of the synthetic seasonal structure.

---

## 5. Hypotheses

### Hypothesis 1 — Attention versus simple baseline

Because warehouse demand contains temporal and seasonal structure, the attention-based model is expected to capture useful relationships among historical observations and achieve lower test MAE than a simple last-observation baseline.

### Hypothesis 2 — Scaled versus unscaled attention

Scaled dot-product attention is expected to produce more stable learning behavior than unscaled attention, particularly when the query/key dimension is increased.

The scaling factor \(1/\sqrt{d_k}\) is expected to reduce the magnitude of attention logits and therefore reduce excessive softmax concentration.

---

## 6. Expected Baseline Behavior

The primary baseline will be the last-observation baseline:

\[
\hat{D}_{t+1} = D_t
\]

This baseline assumes that the next hour will have demand approximately equal to the current hour.

Because warehouse demand contains temporal patterns, the attention model is expected to improve upon this baseline, although this will be determined experimentally rather than assumed to be true.

---

## 7. Evaluation Methodology

The data will be divided chronologically into:

- Training set
- Validation set
- Test set

Random splitting will not be used because future observations should not leak into the training data.

The primary forecasting metrics will be:

### Mean Absolute Error

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

### Root Mean Squared Error

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

MAE provides an interpretable measure of average prediction error, while RMSE gives greater weight to large prediction errors.

---

## 8. Planned Ablation Experiment

The primary ablation will compare:

### Scaled attention

\[
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\]

against:

### Unscaled attention

\[
softmax(QK^T)
\]

The comparison will examine training loss and learning behavior.

The hypothesis is that scaled attention will behave more stably because the scaling factor controls the magnitude of the attention logits.

---

## 9. Planned Distribution-Shift Experiment

The model will first be trained using a normal synthetic warehouse distribution containing moderate noise and moderate demand events.

The generalization experiment will modify the test distribution by increasing the magnitude of noise and/or demand spikes.

### Training distribution

- Moderate noise
- Regular seasonality
- Moderate demand events

### Shifted test distribution

- Higher noise
- Larger demand spikes

The expected result is that prediction error may increase under the shifted distribution because the model is evaluated on conditions that differ from those observed during training.

The actual effect will be measured experimentally.

---

## 10. Planned Failure Modes

### Failure Mode 1 — Demand spike

The model may underestimate an unusual demand spike because such events are rare in the training data.

### Failure Mode 2 — Distribution shift

Prediction error may increase when noise or event magnitude is substantially different from the training distribution.

### Failure Mode 3 — Attention saturation

Large attention logits may produce highly concentrated softmax weights. This may result in small gradients for some positions and affect training behavior.

---

## 11. Planned Failure Investigation

At least one concrete failure will be selected after the experiments.

The investigation will document:

1. The scenario.
2. The input that produced the failure.
3. The expected prediction.
4. The actual prediction.
5. A possible explanation.
6. Evidence supporting the explanation.
7. A potential improvement.

The purpose is not necessarily to completely fix the failure, but to understand why it occurred.

---

## 12. Reproducibility Plan

The project will record:

- Python version.
- Framework and library versions.
- Random seed.
- Dataset-generation parameters.
- Model configuration.
- Training configuration.
- Evaluation commands.

The synthetic dataset generator will support deterministic generation using a fixed seed.

---

## 13. Success Criteria

The project will be considered complete when it demonstrates:

1. A mathematically defined synthetic warehouse environment.
2. A first-principles scaled dot-product attention implementation.
3. Mathematical verification of important operations.
4. Gradient verification.
5. A small toy learning experiment.
6. Training-dynamics analysis.
7. A scaled-versus-unscaled ablation.
8. Next-hour warehouse demand prediction.
9. Comparison with a simple baseline.
10. Quantitative evaluation.
11. Distribution-shift evaluation.
12. Investigation of at least one genuine failure.
13. Reproducible experiments.
14. Basic tests.
15. Clear documentation and reflection.