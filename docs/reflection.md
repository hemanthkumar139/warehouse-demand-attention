# Reflection

## Initial Expectation

I expected the attention-based warehouse model to outperform the
last-value baseline because attention can combine information from
multiple previous observations.

I also expected scaled attention to behave more stably than unscaled
attention as the key/query dimension increased.

## What Happened

The scaled attention implementation and numerical gradient verification
worked as expected.

The toy learning experiment showed gradual improvement in validation
loss.

The scaled versus unscaled experiment showed that unscaled attention
can produce much larger logits and highly concentrated softmax
distributions, particularly at d_k = 16 and d_k = 64.

However, the warehouse attention model performed substantially worse
than the simple last-value baseline.

## Hypothesis That Was Supported

The scaling hypothesis received supporting evidence.

For example, at d_k = 64:

Scaled attention entropy = 0.1491

Unscaled attention entropy = 0.0012

The unscaled case also produced substantially larger attention scores.

This is consistent with the mathematical motivation for dividing by
sqrt(d_k).

## Hypothesis That Was Not Supported

The expectation that the attention model would outperform the
last-value baseline was not supported.

Attention model:

MAE = 27.87
RMSE = 33.78

Last-value baseline:

MAE = 7.85
RMSE = 11.44

## Most Important Surprise

The simple last-value baseline performed much better than the
experimental attention model.

This demonstrated that model complexity does not automatically result
in better predictive performance.

## Hardest Part

The most challenging part was connecting the mathematical attention
operations with a complete experimental pipeline and verifying that the
implementation was correct.

Numerical gradient verification was particularly useful because it
provided independent evidence for the analytical derivative.

## Main Failure

The warehouse attention model used randomly initialized fixed
attention projection matrices while only the output prediction head was
trained.

This meant that the attention representation was not optimized
end-to-end.

## What I Learned

I learned that:

1. Tensor shapes are central to implementing attention correctly.
2. Scaling affects the magnitude of attention logits.
3. Softmax can become highly concentrated when logits are large.
4. Numerical gradient checking is useful for validating analytical
   derivatives.
5. Simple baselines are essential for evaluating whether a complex model
   actually provides value.
6. Distribution shift can significantly increase prediction error.
7. Experimental failures can provide useful evidence when they are
   investigated rather than hidden.

## Remaining Uncertainty

The current warehouse experiment does not establish whether a fully
trainable attention model would outperform the baseline.

The attention projections were not trained jointly with the prediction
head, so this remains an open question.

## What I Would Investigate With One More Day

With more time I would:

- Train W_Q, W_K and W_V jointly with the prediction head.
- Add time-of-day features.
- Compare 24-hour and 48-hour windows.
- Add a residual connection from the most recent demand.
- Tune learning rate and initialization.
- Compare against a linear autoregressive baseline.
