# Failure Analysis

## 1. Failure: Attention model underperformed the simple baseline

### Scenario

The warehouse task predicts next-hour demand from the previous 24
hourly observations.

The attention model uses first-principles scaled dot-product attention
followed by a prediction head.

The baseline predicts the next hour using the most recent observation.

### Expected behavior

The initial hypothesis was that attention could exploit multiple
historical observations and temporal structure, producing lower MAE than
the last-value baseline.

### Actual result

| Model | MAE | RMSE |
|---|---:|---:|
| Attention model | 27.87 | 33.78 |
| Last-value baseline | 7.85 | 11.44 |

The attention model was substantially worse than the simple baseline.

The MAE difference was:

Attention MAE - Baseline MAE = 20.02

### Investigation

The warehouse model used the same first-principles attention mechanism
implemented for the mathematical experiment.

However, the attention projection matrices W_Q, W_K and W_V were
initialized randomly and kept fixed during the warehouse training
experiment. Only the final prediction head was optimized.

Therefore, the attention representation was not learned end-to-end.

The baseline, in contrast, directly uses the most recent demand value.
The synthetic demand process has strong short-term temporal dependence,
so the latest observation is highly informative for the next observation.

This creates a difficult comparison for a randomly initialized,
partially trained attention representation.

### Evidence

The attention model achieved:

- MAE = 27.87
- RMSE = 33.78

The last-value baseline achieved:

- MAE = 7.85
- RMSE = 11.44

The training loss of the attention model decreased gradually, but the
model did not approach the performance of the simple baseline.

### Interpretation

This failure does not show that attention is inherently unsuitable for
warehouse demand prediction.

Instead, it provides evidence that the particular experimental setup
was inadequate.

The attention mechanism requires useful learned projections or another
appropriate parameterization to transform the input sequence into a
predictive representation.

A random fixed attention representation followed by a trained linear
head can discard useful temporal information.

### Potential improvements

1. Train W_Q, W_K and W_V jointly with the prediction head.
2. Add explicit time-of-day features.
3. Compare 24-hour and 48-hour input windows.
4. Include a residual connection from the latest demand value.
5. Train for more epochs and tune the learning rate.
6. Compare learned attention with a simple linear autoregressive model.

### Lesson

The experiment reinforced the importance of comparing a complex model
against a simple baseline.

A more sophisticated mathematical mechanism does not automatically
produce better predictive performance.
