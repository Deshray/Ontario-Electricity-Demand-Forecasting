import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD AND ALIGN DATA
# =========================
df = pd.read_csv(r"ontario_electricity_demand.csv")

df["date"] = pd.to_datetime(df["date"])
df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit="h")

df = df.sort_values("timestamp").set_index("timestamp")

df = df[["hourly_demand", "hourly_average_price"]]
df = df.rename(
    columns={
        "hourly_demand": "demand_kwh",
        "hourly_average_price": "price_cents"
    }
)

# =========================
# 2. MISSING DATA DIAGNOSTICS
# =========================
print("Missing rate per column:")
print(df.isna().mean())

expected_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
missing_timestamps = expected_index.difference(df.index)
print(f"Missing timestamps: {len(missing_timestamps)}")

# Reindex to full hourly grid and interpolate
df = df.reindex(expected_index)
df = df.interpolate(method="time")

# Final integrity check
assert df.isna().sum().sum() == 0, "Missing values remain after interpolation"

# =========================
# 3. EDA — SANITY CHECKS
# =========================
plt.figure(figsize=(12,4))
df["demand_kwh"].plot()
plt.title("Ontario Hourly Electricity Demand (Full Period)")
plt.show()

plt.figure(figsize=(12,4))
df.loc["2022-01":"2022-03", "demand_kwh"].plot()
plt.title("Ontario Hourly Electricity Demand (Winter 2022)")
plt.show()

plt.figure(figsize=(12,4))
df.loc["2022-01-10":"2022-01-17", ["demand_kwh", "price_cents"]].plot()
plt.title("Demand vs Price (Jan 10–17, 2022)")
plt.show()

# =========================
# 4. FEATURE ENGINEERING
# =========================
df["hour"] = df.index.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df["dayofyear"] = df.index.dayofyear

# =========================
# 5. HOLIDAY FEATURES (ONTARIO)
# =========================
start_year = df.index.min().year
end_year = df.index.max().year
holidays = []

for year in range(start_year, end_year + 1):
    holidays.extend([
        pd.Timestamp(f"{year}-01-01"),
        pd.Timestamp(f"{year}-07-01"),
        pd.Timestamp(f"{year}-12-25"),
        pd.Timestamp(f"{year}-12-26"),
    ])

    feb_mondays = pd.date_range(f"{year}-02-01", f"{year}-02-28", freq="W-MON")
    if len(feb_mondays) >= 3:
        holidays.append(feb_mondays[2])

    easter = {
        2020: "04-10", 2021: "04-02", 2022: "04-15",
        2023: "04-07", 2024: "03-29", 2025: "04-18"
    }
    if year in easter:
        holidays.append(pd.Timestamp(f"{year}-{easter[year]}"))

    may_24 = pd.Timestamp(f"{year}-05-24")
    holidays.append(may_24 - pd.Timedelta(days=(may_24.dayofweek + 1) % 7))

    aug = pd.date_range(f"{year}-08-01", f"{year}-08-07", freq="W-MON")
    if len(aug) >= 1:
        holidays.append(aug[0])

    sep = pd.date_range(f"{year}-09-01", f"{year}-09-07", freq="W-MON")
    if len(sep) >= 1:
        holidays.append(sep[0])

    octo = pd.date_range(f"{year}-10-01", f"{year}-10-31", freq="W-MON")
    if len(octo) >= 2:
        holidays.append(octo[1])

holidays = list(set(h.normalize() for h in holidays))
df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)

print(f"Holidays marked (hours): {df['is_holiday'].sum()}")
print(f"Unique holiday dates: {len(holidays)}")

# =========================
# 6. LAG + ROLLING FEATURES
# =========================
for lag in [1, 24, 168]:
    df[f"demand_lag_{lag}"] = df["demand_kwh"].shift(lag)

df["rolling_mean_24"] = df["demand_kwh"].rolling(24).mean()
df["rolling_std_24"] = df["demand_kwh"].rolling(24).std()
df["rolling_mean_168"] = df["demand_kwh"].rolling(168).mean()
df["rolling_std_168"] = df["demand_kwh"].rolling(168).std()

df["price_lag_24"] = df["price_cents"].shift(24)
df["price_lag_168"] = df["price_cents"].shift(168)
df["price_rolling_mean_168"] = df["price_cents"].rolling(168).mean()
df["price_rolling_std_168"] = df["price_cents"].rolling(168).std()

# Target: 24h ahead demand
df["target_demand_24h"] = df["demand_kwh"].shift(-24)

df_model = df.dropna()

os.makedirs("data/processed", exist_ok=True)
df_model.to_csv("data/processed/energy_features.csv")

# =========================
# 7. TRAIN / TEST SPLIT
# =========================
features = df_model.drop(
    columns=["target_demand_24h", "demand_kwh", "price_cents", "hour"]
)
target = df_model["target_demand_24h"]

split_date = "2019-01-01"
X_train = features.loc[features.index < split_date]
X_test  = features.loc[features.index >= split_date]
y_train = target.loc[target.index < split_date]
y_test  = target.loc[target.index >= split_date]

# =========================
# 8. BASELINE + LINEAR MODEL
# =========================
y_pred_naive = df_model["demand_lag_24"].loc[y_test.index]
print("Naive MAE:", mean_absolute_error(y_test, y_pred_naive))
print("Naive RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_naive)))

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("LR MAE:", mean_absolute_error(y_test, y_pred_lr))
print("LR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# =========================
# 9. QUANTILE REGRESSION (PROBABILISTIC)
# =========================
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index,
    columns=X_test.columns
)

qr_low  = QuantileRegressor(quantile=0.1, alpha=0.1)
qr_med  = QuantileRegressor(quantile=0.5, alpha=0.1)
qr_high = QuantileRegressor(quantile=0.9, alpha=0.1)

qr_med_reg = QuantileRegressor(quantile=0.5, alpha=0.1)
qr_med_reg.fit(X_train_scaled, y_train)
y_med_reg = qr_med_reg.predict(X_test_scaled)

qr_low.fit(X_train_scaled, y_train)
qr_med.fit(X_train_scaled, y_train)
qr_high.fit(X_train_scaled, y_train)

y_low = qr_low.predict(X_test_scaled)
y_med = qr_med.predict(X_test_scaled)
y_high = qr_high.predict(X_test_scaled)

y_low = pd.Series(y_low, index=y_test.index)
y_med = pd.Series(y_med, index=y_test.index)
y_high = pd.Series(y_high, index=y_test.index)

print("QR (Median) MAE:", mean_absolute_error(y_test, y_med))
print("QR (Median) RMSE:", np.sqrt(mean_squared_error(y_test, y_med)))

# =========================
# 10. UNCERTAINTY VISUALIZATION
# =========================
n = 500
plt.figure(figsize=(12,4))
plt.plot(y_test.index[:n], y_test[:n], label="Actual", color="black")
plt.plot(y_test.index[:n], y_med[:n], label="Median Forecast", color="blue")
plt.fill_between(
    y_test.index[:n], y_low[:n], y_high[:n],
    alpha=0.3, label="80% Prediction Interval"
)
plt.legend()
plt.title("24h Ahead Probabilistic Demand Forecast")
plt.show()

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "linear_coef": lr.coef_,
    "qr_median_coef": qr_med.coef_
})

coef_df["abs_linear"] = coef_df["linear_coef"].abs()
coef_df = coef_df.sort_values("abs_linear", ascending=False)

print(coef_df.head(15))

plt.figure(figsize=(10,8))
coef_df.head(15).plot(x='feature', y='abs_linear', kind='barh')
plt.title("Top 15 Feature Importance (Linear Regression)")
plt.xlabel("Absolute Coefficient Value")
plt.show()

# =========================
# 11. RESIDUAL DIAGNOSTICS
# =========================
res_lr = y_test - y_pred_lr
res_qr = y_test - y_med

plt.figure(figsize=(10,6))
plt.hist(res_lr, bins=50, alpha=0.6, label="Linear Regression")
plt.hist(res_qr, bins=50, alpha=0.6, label="Quantile (Median)")
plt.legend()
plt.title("Residual Distribution Comparison")
plt.show()

coverage = np.mean((y_test >= y_low) & (y_test <= y_high))
print(f"80% interval empirical coverage: {coverage:.3f}")

os.makedirs("outputs", exist_ok=True)

# ============================================================================
# STEP 4 — MONTE CARLO SIMULATION
# ============================================================================
# Configuration
n_horizon = 168  # 7 days ahead
n_simulations = 1000

# Extract predictions for simulation horizon
y_low_horizon = y_low[-n_horizon:]
y_med_horizon = y_med[-n_horizon:]
y_high_horizon = y_high[-n_horizon:]
y_test_horizon = y_test[-n_horizon:]

# Compute uncertainty (standard deviation from quantile spread)
# For normal distribution: q90 - q10 ≈ 2.56σ
sigma = (y_high_horizon - y_low_horizon) / 2.56

# Generate scenarios
np.random.seed(42)
scenarios = np.zeros((n_horizon, n_simulations))

for t in range(n_horizon):
    scenarios[t, :] = np.random.normal(
        loc=y_med_horizon.iloc[t],
        scale=sigma.iloc[t],
        size=n_simulations
    )

scenarios = np.clip(scenarios, a_min=0, a_max=None)

# ============================================================================
# VISUALIZATION 1: FAN CHART
# ============================================================================
percentiles = [5, 25, 50, 75, 95]
scenario_percentiles = np.percentile(scenarios, percentiles, axis=1)

plt.figure(figsize=(14, 6))
time_index = y_test_horizon.index

plt.plot(time_index, y_test_horizon, 
         label='Actual Demand', color='black', linewidth=2, zorder=5)
plt.plot(time_index, y_med_horizon, 
         label='Median Forecast', color='blue', linewidth=2, linestyle='--', zorder=4)

plt.fill_between(time_index, 
                 scenario_percentiles[0], scenario_percentiles[4],
                 alpha=0.2, label='5th-95th percentile', color='red', zorder=1)
plt.fill_between(time_index, 
                 scenario_percentiles[1], scenario_percentiles[3],
                 alpha=0.3, label='25th-75th percentile', color='blue', zorder=2)

plt.title('7-Day Demand Forecast with Uncertainty (Monte Carlo Fan Chart)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Demand (kWh)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fan_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VISUALIZATION 2: PEAK DEMAND DISTRIBUTION
# ============================================================================
peak_demands = scenarios.max(axis=0)

plt.figure(figsize=(10, 6))
plt.hist(peak_demands, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
plt.axvline(peak_demands.mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Mean: {peak_demands.mean():.0f}')
plt.axvline(np.percentile(peak_demands, 95), color='red', linestyle='--',
            linewidth=2, label=f'95th %ile: {np.percentile(peak_demands, 95):.0f}')
plt.axvline(y_test_horizon.max(), color='black', linestyle='-',
            linewidth=2, label=f'Actual: {y_test_horizon.max():.0f}')

plt.title('Distribution of Peak Demand Over 7-Day Horizon', 
          fontsize=14, fontweight='bold')
plt.xlabel('Peak Demand (kWh)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/peak_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n Peak Demand Statistics:")
print(f"  Mean: {peak_demands.mean():.0f} kWh")
print(f"  Std:  {peak_demands.std():.0f} kWh")
print(f"  95th: {np.percentile(peak_demands, 95):.0f} kWh")
print(f"  99th: {np.percentile(peak_demands, 99):.0f} kWh")

# ============================================================================
# STEP 5 — STRESS TESTING & RISK METRICS
# ============================================================================
# METRIC 1: Peak Demand Risk P(Demand > X)
thresholds = [18_000_000, 19_000_000, 20_000_000, 21_000_000, 22_000_000]

risk_table = []

for threshold in thresholds:
    exceedance_count = np.sum(np.any(scenarios > threshold, axis=0))
    probability = exceedance_count / n_simulations
    
    risk_table.append({
        'Threshold (kWh)': threshold,
        'P(Peak > Threshold)': f"{probability:.3f}",
        'Scenarios': exceedance_count
    })

risk_df = pd.DataFrame(risk_table)
print(risk_df.to_string(index=False))

# METRIC 2: Extreme Event Analysis
for p in [90, 95, 99]:
    value = np.percentile(scenarios.flatten(), p)
    print(f"  {p}th percentile: {value:.0f} kWh")

# METRIC 3: System Stress (Worst 1%)
worst_1pct = np.percentile(scenarios.flatten(), 99)
print(f"3. WORST 1% THRESHOLD: {worst_1pct:.0f} kWh")

# Visualize frequency of extreme events by hour
worst_hours_mask = scenarios > worst_1pct
worst_hours_count = worst_hours_mask.sum(axis=1)

plt.figure(figsize=(12, 5))
plt.bar(range(n_horizon), worst_hours_count, alpha=0.7, color='crimson')
plt.axhline(10, color='black', linestyle='--', linewidth=1.5, 
            label='1% threshold (10 scenarios)')
plt.title('Frequency of Extreme Demand Events by Hour', 
          fontsize=14, fontweight='bold')
plt.xlabel('Hour in Forecast Horizon', fontsize=12)
plt.ylabel('Count (out of 1000 scenarios)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/extreme_events_by_hour.png', dpi=300, bbox_inches='tight')
plt.show()

# METRIC 4: Daily Peak Load Distribution
n_days = 7
hours_per_day = 24
daily_peaks = np.zeros((n_days, n_simulations))

for day in range(n_days):
    start_idx = day * hours_per_day
    end_idx = start_idx + hours_per_day
    daily_peaks[day, :] = scenarios[start_idx:end_idx, :].max(axis=0)

plt.figure(figsize=(12, 6))
box_plot = plt.boxplot(
    [daily_peaks[i, :] for i in range(n_days)],
    tick_labels=[f'Day {i+1}' for i in range(n_days)],
    patch_artist=True
)


for patch in box_plot['boxes']:
    patch.set_facecolor('lightblue')

plt.title('Daily Peak Demand Distribution (1000 Scenarios)', 
          fontsize=14, fontweight='bold')
plt.ylabel('Peak Demand (kWh)', fontsize=12)
plt.xlabel('Day in Forecast Horizon', fontsize=12)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/daily_peaks.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics table
summary_stats = []
for day in range(n_days):
    summary_stats.append({
        'Day': day + 1,
        'Mean': f"{daily_peaks[day, :].mean():.0f}",
        'Std': f"{daily_peaks[day, :].std():.0f}",
        '95th': f"{np.percentile(daily_peaks[day, :], 95):.0f}",
        '99th': f"{np.percentile(daily_peaks[day, :], 99):.0f}"
    })

summary_df = pd.DataFrame(summary_stats)
print("\n4. DAILY PEAK STATISTICS")
print(summary_df.to_string(index=False))

# ============================================================================
# COMPREHENSIVE SUMMARY
# ============================================================================
print("COMPREHENSIVE RISK SUMMARY")
print(f"Forecast horizon: {n_horizon} hours ({n_days} days)")
print(f"Simulations: {n_simulations}")
print(f"\nActual peak demand: {y_test_horizon.max():.0f} kWh")
print(f"Forecasted median peak: {y_med_horizon.max():.0f} kWh")
print(f"Expected simulated peak: {peak_demands.mean():.0f} kWh")
print(f"\nP(Peak > 20,000): {np.mean(peak_demands > 20000):.3f}")
print(f"P(Peak > 21,000): {np.mean(peak_demands > 21000):.3f}")
print(f"\n95% CI for peak: [{np.percentile(peak_demands, 2.5):.0f}, "

      f"{np.percentile(peak_demands, 97.5):.0f}] kWh")
