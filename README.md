# Ontario Electricity Demand Forecasting  
Probabilistic Forecasting, Uncertainty Quantification, and Risk Simulation

## Overview
This project develops a **probabilistic forecasting system** for Ontario’s hourly electricity demand, with a focus on **uncertainty modeling and peak-load risk analysis** rather than point prediction alone.

The pipeline produces:
- Interpretable demand forecasts
- Calibrated prediction intervals
- Monte Carlo demand scenarios
- Quantitative risk metrics for extreme load events

This mirrors real-world forecasting and risk workflows used in energy systems and quantitative modeling.

---

## Data
- **Frequency:** Hourly
- **Time span:** ~2003–2023
- **Target:** Electricity demand (kWh)
- **Auxiliary variable:** Hourly average price

Data integrity checks ensure:
- No missing timestamps
- No missing target values after preprocessing

---

## Feature Engineering

### Temporal Structure
- Hour of day (sin / cos encoding)
- Day of week
- Month
- Day of year
- Weekend indicator

### Holiday Effects (Ontario-specific)
- Fixed-date holidays
- Variable holidays (e.g., Good Friday, Labour Day, Thanksgiving)

### Lagged & Rolling Features
- Demand lags: 1h, 24h, 168h
- Rolling demand statistics: mean and volatility
- Price lags and rolling statistics

---

## Modeling

### Baseline
- Naive 24-hour lag forecast  
Used as a strong seasonal benchmark.

### Linear Regression
- Deterministic reference model
- High interpretability
- Establishes baseline error levels

### Quantile Regression (Core Model)
- Quantiles: 0.1, 0.5, 0.9
- Produces **prediction intervals**
- Enables uncertainty-aware downstream analysis

Evaluation metrics:
- MAE
- RMSE
- Empirical interval coverage

---

## Uncertainty & Risk Analysis

### Monte Carlo Simulation
- Horizon: 7 days (168 hours)
- Simulations: 1000
- Quantile-derived uncertainty
- Generates full demand trajectories

### Risk Metrics
- Peak demand distribution
- Probability of exceeding system thresholds
- Worst-case (top 1%) stress events
- Daily peak load statistics

All visual outputs are saved to the `/outputs` directory.

---

## Results (High-Level)
- Quantile model achieves ~80% empirical coverage for the 80% interval
- Demand variability is driven primarily by:
  - Hour-of-day seasonality
  - Weekend effects
  - Holiday structure
- Simulated peak demand distributions reveal **material upside risk**, even when median forecasts appear conservative

---

## Limitations
- No weather or temperature inputs
- Linear conditional quantile assumption
- Gaussian error assumption in simulation
- No regime-switching or volatility clustering

These limitations are **explicit modeling tradeoffs**, not oversights.

---

## Future Extensions
- Weather-driven demand modeling
- Nonlinear quantile models (GBDT / trees)
- Regime-dependent volatility
- Scenario-conditioned stress testing

---

## How to Run
```bash
pip install -r requirements.txt
python OntarioEnergyPrediction.py
