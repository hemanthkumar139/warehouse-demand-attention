# From First Principles: Warehouse Demand Attention Model

## Overview

This project implements scaled dot-product self-attention from first
principles using NumPy and investigates its behavior on synthetic
warehouse demand data.

The project follows the workflow:

Mathematical formulation
→ First-principles implementation
→ Verification
→ Toy learning
→ Ablation
→ Warehouse forecasting
→ Baseline comparison
→ Distribution shift
→ Failure analysis

## Project Structure

```text
warehouse-demand-attention/
│
├── src/
│   ├── attention.py
│   └── warehouse_data.py
│
├── experiments/
│   ├── gradient_verification.py
│   ├── toy_learning.py
│   ├── ablation.py
│   ├── warehouse_model.py
│   └── distribution_shift.py
│
├── tests/
│   └── test_attention.py
│
├── docs/
│   ├── phase0.md
│   ├── failure_analysis.md
│   ├── ai_log.md
│   └── reflection.md
│
├── results/
│   └── tables/
│
├── README.md
└── requirements.txt
