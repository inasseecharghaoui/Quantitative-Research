# Commodity Trading Signals

Systematic BUY / HOLD / SELL signal generation on a universe of 150 commodities,
developed during a Quantitative Research internship (Cross-Asset desk, Jul.–Aug. 2026).

## Models

| Model | Type | Idea |
|---|---|---|
| RSI | Rule-based | Mean reversion: BUY if RSI < 30, SELL if RSI > 70 |
| Multi-Factor | Rule-based | Sum of 4 signals (RSI, Momentum, MACD, Carry) in {-1, 0, 1}; BUY if score ≥ 3, SELL if ≤ -3 |
| Linear Regression | ML | Predicts next-day direction from 10 market features |
| Ridge Regression | ML | Same, with standardization and L2 regularization |
| Neural Network | ML | Keras MLP (64-32-3, ReLU, softmax), 3-class classification |

## Methodology

- **Target**: next-day return, labelled BUY (> +0.2%), SELL (< -0.2%), HOLD otherwise.
- **Features**: price, volume, 30d volatility, 12d momentum, RSI-14, MACD, carry, roll yield, term spread, 20d z-score.
- **Validation**: chronological 80/20 train/test split to avoid look-ahead bias.
- **Metric**: classification accuracy on the test period.

## Usage

```bash
pip install -r requirements.txt
python trading_models.py
```

The dataset is not included (internship data). The script expects
`data/commodities.xlsx` with a `Data` sheet containing the columns listed above
plus `Date`, `Commodity`, `MACD_Signal`.

Output: `Results_BUY_HOLD_SELL.xlsx` with every model's signal for each commodity and date.
