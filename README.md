Markdown
# Japan Stock Backtester (`japan-stock-backtester`)

A lightweight, high-performance quantitative backtesting engine tailored for Japanese equities, designed to simulate screening signals with strict elimination of common financial modeling biases.

---

## Key Features

- **Lookahead Bias Mitigation**: Signals confirmed at market Close are executed at the next trading day's Open (`Open` of `t+1`).
- **Flexible Multi-Mode Exit Engine**:
  - `fixed`: Time-based exit after a specified number of holding days.
  - `tp_sl`: Take-profit and stop-loss execution with conservative tie-breaking (stop-loss takes priority if both thresholds are breached within the same bar).
  - `trailing`: Trailing stop triggered by percentage pullbacks from peak prices, with support for activation thresholds and initial hard stop-loss levels.
- **Survivorship & Delisting Handling**: Tracks positions through historical delistings, forcing settlement at the final available closing price to prevent survivorship distortion.
- **Execution Friction**: Deducts customizable round-trip transaction costs (`cost_bps`) from each trade return.
- **Optimized Array Lookups**: Avoids slow DataFrame iterations by caching price series into NumPy arrays with $O(1)$ datetime index lookups (`_CodeSeries`).

---

## Project Structure

```text
japan-stock-backtester/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── backtest.py       # Core execution engine and performance summarizer
└── tests/
    └── test_backtest.py  # Unit tests for edge cases and exit strategies
