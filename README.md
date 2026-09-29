# FX Martingale Portfolio Backtest

GitHub Actions project for a five-account FX averaging/martingale portfolio.

## Portfolio
- Total capital: JPY 2,000,000
- Deployed: JPY 1,500,000 across five accounts (allocation optimized)
- Reserve/redeposit pool: JPY 500,000
- Leverage: 1:1000
- Stop-out model: 20% margin level; negative balance floor at zero
- Monthly profit sweep above each account's assigned base capital when flat

## Research design
- Primary timeframe: M15, up to 10 years where available
- Secondary stress/frequency run: M5
- Train/OOS split: 70/30
- Conservative same-bar ordering: adverse grid/stop event before favorable TP
- Spread/slippage are configurable execution assumptions; historical HFM execution cannot be reconstructed exactly from Dukascopy market data.
- Swap is reported as a sensitivity/configuration item rather than claimed as historical HFM swap data.

## Run
Use Actions -> `FX Martingale Backtest` -> Run workflow. Start with `m15`.

Outputs are uploaded as an artifact under `results/`.

This is research software, not a guarantee of future performance.
