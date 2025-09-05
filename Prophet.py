import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from datetime import timedelta
from darts import concatenate
from darts.utils.statistics import check_seasonality
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.utils.statistics import plot_acf
from darts.models import Prophet
from darts.metrics import mae, rmse
import os
import csv
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_edf,
)

# -----------------------------
# Daten einlesen und vorbereiten
# -----------------------------
df = pd.read_csv("./data/processed_combined_df.csv", index_col="timestamp", parse_dates=True)
df.index = df.index.tz_convert(None)

target_col = "CH_price [Euro/MWh]"
forecast_cols = [col for col in df.columns if "forecast" in col and col != target_col]
other_cols = [col for col in df.columns if col != target_col and col not in forecast_cols]
df[forecast_cols] = df[forecast_cols].fillna(0)

series = TimeSeries.from_dataframe(df, value_cols=target_col)
future_covariates = TimeSeries.from_dataframe(df, value_cols=forecast_cols)
#past_covariates = TimeSeries.from_dataframe(df, value_cols=other_cols)

filler = MissingValuesFiller()
series = filler.transform(series)
#past_covariates = filler.transform(past_covariates)¬
future_covariates = filler.transform(future_covariates)

# -----------------------------
# Splits definieren
# -----------------------------
train_start = pd.Timestamp("2020-04-08 10:00:00")
val_start = pd.Timestamp("2023-04-08 10:00:00")
test_start = pd.Timestamp("2024-04-08 10:00:00")
test_end = pd.Timestamp("2025-04-08 10:00:00")

train, valtest = series.split_before(val_start)
val, test = valtest.split_before(test_start)
test = test.slice(test_start, test_end)

_, fut_valtest = future_covariates.split_before(val_start)
future_val, future_test = fut_valtest.split_before(test_start)
future_val = future_val.slice(val_start, test_start-timedelta(hours=1))
future_test = future_test.slice(test_start, test_end)
print("future_val von ", future_val.start_time(), "bis", future_val.end_time())

future_train = future_covariates.slice(future_covariates.start_time(), val_start-timedelta(hours=1))
train = train.slice(train.start_time(), val_start)

forecast_horizon = 38
history_horizon = 48
stride = 24  # für 1 Forecast pro Tag (24h) bei 15-min Daten


# ============================================================================
# Visualisierung vorbereiten (nur bis Test-Start)
# ============================================================================
df_plot = df[df.index < train_start]

# Spaltenlogik basierend auf der neuen Gruppierung
target_col = "CH_price [Euro/MWh]"
forecast_cols = [col for col in df.columns if "forecast" in col and col != target_col]
past_cols = [col for col in df.columns if col not in forecast_cols and col != target_col]

# ============================================================================
# Preisplot: Nur Spalten mit "Euro" oder "Euro/MWh" aus allen 3 Gruppen
# ============================================================================
price_cols = [
    col for col in [target_col] + forecast_cols + past_cols
    if "Euro/MWh" in col
]

df_price = df_plot[price_cols]
df_price_norm = (df_price - df_price.mean()) / df_price.std()
df_price_melted = df_price_norm.melt(var_name="Column", value_name="Normalized")

plt.figure(figsize=(14, 6))
sns.violinplot(x="Column", y="Normalized", data=df_price_melted, inner="box")
plt.xticks(rotation=45)
plt.title("Verteilung der Preisvariablen (normalisiert)")
plt.tight_layout()
plt.show()

# ============================================================================
# Saisonale Muster prüfen
# ============================================================================
print("Seasonality check:")
print("Daily:", check_seasonality(train, m=24, max_lag=7*24))        # 1 Tag, 7 Tage
print("Weekly:", check_seasonality(train, m=24*7, max_lag=24*30))    # 1 Woche, 1 Monat
print("Monthly:", check_seasonality(train, m=24*30, max_lag=24*90))  # 1 Monat, 3 Monate
print("Yearly:", check_seasonality(train, m=24*365, max_lag=24*730)) # 1 Jahr, 2 Jahre

# ============================================================================
# Autokorrelationsanalyse (ACF) für geglättete Zeitreihen
# ============================================================================
# Stundenmittelwerte
hourly_series = train.to_series()
plot_acf(TimeSeries.from_series(hourly_series), m=24, alpha=0.05, max_lag=48)
plt.title("ACF der Stundenwerte über 2 Tage")
plt.tight_layout()
plt.show()

# Tagesmittelwerte
daily_series = train.to_series().resample("1D").mean()
plot_acf(TimeSeries.from_series(daily_series), m=7, alpha=0.05, max_lag=30)
plt.title("ACF der Tagesmittelwerte einen 1 Monat")
plt.tight_layout()
plt.show()

# Monatsmittelwerte
monthly_series = train.to_series().resample("1M").mean()
plot_acf(TimeSeries.from_series(monthly_series), m=12, alpha=0.05, max_lag=12)
plt.title("ACF der Monatsmittelwerte über 1 Jahr")
plt.tight_layout()
plt.show()

# ============================================================================
# Gleitender Mittelwert im Originalsignal
# ============================================================================
train_df = train.to_dataframe()
train_df["rolling_mean"] = train_df[target_col].rolling(window=24*7).mean()  # 7 Tage Fenster
train_df.plot(figsize=(12, 5), title="Train-Zeitreihe mit 7-Tage gleitendem Mittelwert")
plt.tight_layout()
plt.show()

# ============================================================================
# Prophet-Modell konfigurieren
# ============================================================================

def objective(trial):
    changepoint_prior_scale = trial.suggest_categorical("changepoint_prior_scale", [0.01, 0.03, 0.05, 0.1])
    seasonality_mode = trial.suggest_categorical("seasonality_mode", ["multiplicative", "additive"])
    weekly_seasonality = trial.suggest_categorical("weekly_seasonality", [7, 10, 12])
    yearly_seasonality = trial.suggest_categorical("yearly_seasonality", [False, 5, 10])

    model = Prophet(
        growth="linear",
        seasonality_mode=seasonality_mode,
        daily_seasonality=24,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        country_holidays="CH",
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=0.90,
        n_changepoints=80,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=5.0,
        mcmc_samples=0,
        add_encoders={"cyclic": {"future": ["hour", "day", "dayofweek"]}},
    )

    model.fit(train, future_covariates=future_train)
    
    val_forecast = model.predict(n=len(val), future_covariates=future_val)
    val_target = val.slice_intersect(val_forecast)

    mae_val = mae(val_target, val_forecast)
    rmse_val = rmse(val_target, val_forecast)

    os.makedirs("./results", exist_ok=True)

    metrics_path = f"./results/Prophet_metrics_validationset_V{trial.number}.csv"
    df_result_path = f"./results/Prophet_forecast_vs_truth_validationset_V{trial.number}.csv"
    plot_path = f"./results/Prophet_forecast_plot_validationset_V{trial.number}.png"
    model_path = f"./results/Prophet_model_sliding_V{trial.number}.pkl"

    with open(metrics_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["MAE", "RMSE"])
        writer.writerow([mae_val, rmse_val])

    df_result = pd.DataFrame({
        "timestamp": val_forecast.time_index,
        "forecast": val_forecast.values().flatten(),
        "truth": val_target.values().flatten(),
    })
    df_result.to_csv(df_result_path, index=False)

    plt.figure(figsize=(10, 4))
    val_target.plot(label="CH_price [Euro/MWh]", lw=2, color="black")
    val_forecast.plot(label="Prophet Forecast [Euro/MWh]", lw=2, color="blue")
    plt.title("Forecast mit Optuna-getuntem Prophet-Modell")
    plt.xlabel("Zeit")
    plt.ylabel("Strompreis CH [Euro/MWh]")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Modell speichern
    model.save(model_path)
    print("Prophet-Modell gespeichert in:", model_path)

    return rmse(val_target, val_forecast)

# ============================================================================
# Evaluation & Visualisierung
# ============================================================================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Beste Parameter:", study.best_params)

#Verlauf der besten Objective-Werte (pro Trial)
plot_optimization_history(study)
plt.show()

# Wichtige Hyperparameter (fANOVA)
plot_param_importances(study)
plt.show()

# Slices pro Parameter
plot_slice(study)
plt.show()

# Empirical Distribution Function (EDF)
plot_edf(study)
plt.show()