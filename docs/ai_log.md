# AI Assistance Log

## Tool Used

ChatGPT

## Purpose of AI Assistance

AI assistance was used as a development and learning aid during the
implementation of the mathematical attention experiment and warehouse
demand experiment.

## Tasks Requested

The assistance included:

- Explaining the mathematical formulation of scaled dot-product
  attention.
- Helping structure the project into source code, experiments, tests,
  results, and documentation.
- Providing NumPy implementation patterns for Q, K, V, attention
  scores, scaling, softmax, and weighted aggregation.
- Helping design numerical gradient verification.
- Helping interpret experimental results.
- Helping identify failure modes and formulate failure analysis.
- Helping structure the README and experiment documentation.

## Generated Output

AI-generated code and explanations were used as starting points for:

- First-principles attention implementation.
- Numerical gradient verification.
- Toy learning experiment.
- Scaled versus unscaled attention ablation.
- Warehouse demand experiment.
- Distribution-shift experiment.
- Basic tests.

## Candidate Modifications

The generated approach was modified to fit the available environment.

PyTorch installation was attempted initially, but the installation was
not completed successfully because of package/network constraints.

The implementation was therefore changed to NumPy.

The warehouse model was also deliberately kept lightweight. The
attention projection matrices were fixed in that experiment and only
the output prediction head was trained.

## Verification

The candidate executed the code locally and checked:

- Attention tensor shapes.
- Softmax row normalization.
- Numerical gradient agreement.
- Toy learning loss behavior.
- Attention versus baseline metrics.
- Distribution-shift metrics.
- Basic automated tests.

The gradient verification produced:

Maximum absolute difference:
2.4714008617365835e-11

Relative difference:
4.805629992623409e-12

Gradient check:
True

## Resulting Understanding

The implementation helped demonstrate that scaled dot-product attention
consists of projections into Q, K and V spaces, dot-product similarity,
scaling, row-wise softmax, and weighted aggregation.

The experiments also showed that a more complex attention-based model
does not automatically outperform a simple baseline.

In the warehouse experiment, the attention model performed worse than
the last-value baseline because the attention projections were random
and fixed while only the prediction head was trained.

The AI-generated suggestions were therefore treated as development
starting points rather than unquestioned results. Experimental results
were generated and verified locally.
