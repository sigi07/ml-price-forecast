import pandas as pd
from datetime import timedelta
import optuna
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.models import RegressionModel
from xgboost import XGBRegressor
from darts.metrics import mae, rmse
import matplotlib.pyplot as plt
import os
import csv
from functools import reduce
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
past_covariates = TimeSeries.from_dataframe(df, value_cols=other_cols)

filler = MissingValuesFiller()
series = filler.transform(series)
past_covariates = filler.transform(past_covariates)
future_covariates = filler.transform(future_covariates)

# -----------------------------
# Splits definieren
# -----------------------------
val_start = pd.Timestamp("2023-04-08 23:00:00")
test_start = pd.Timestamp("2024-04-08 23:00:00")
test_end = pd.Timestamp("2025-04-08 23:00:00")

train, valtest = series.split_before(val_start)
val, test = valtest.split_before(test_start)
test = test.slice(test_start, test_end)

_, fut_valtest = future_covariates.split_before(val_start)
future_val, future_test = fut_valtest.split_before(test_start)
future_val = future_val.slice(val_start, test_start)
future_test = future_test.slice(test_start, test_end)

_, past_valtest = past_covariates.split_before(val_start)
past_val, past_test = past_valtest.split_before(test_start)
past_val = past_val.slice(val_start, test_start)
past_test = past_test.slice(test_start, test_end)

future_train = future_covariates.slice(future_covariates.start_time(), val_start)
past_train = past_covariates.slice(past_covariates.start_time(), val_start)
train = train.slice(train.start_time(), val_start)

forecast_horizon = 38 # 38 Stunden Vorhersagehorizont

# -----------------------------
# Rolling Forecast Funktion
# -----------------------------
def rolling_forecast_eval(model, series, past_covariates, future_covariates,
                          start_time, end_time, lags, horizon):
    time_index = series.time_index
    start_idx = time_index.get_loc(start_time)
    forecasts = []

    while True:
        forecast_start = time_index[start_idx]
        forecast_end = forecast_start + timedelta(hours=1 * horizon)

        if forecast_end > end_time:
            break

        input_series = series.slice(
            forecast_start - timedelta(hours=1 * lags),
            forecast_start - timedelta(hours=1)
        )
        input_past = past_covariates.slice(
            forecast_start - timedelta(hours=1 * lags),
            forecast_start - timedelta(hours=1)
        )
        input_future = future_covariates.slice(
            forecast_start,
            forecast_end - timedelta(hours=1)
        )
        # In DataFrame umwandeln
        df_input_future = input_future.to_dataframe()
        # CSV speichern
        df_input_future.to_csv("./results/input_future.csv")

        forecast = model.predict(
            n=horizon,
            series=input_series,
            past_covariates=input_past,
            future_covariates=input_future,
        )

        last_24h_forecast = forecast[-24:]
        forecasts.append(last_24h_forecast)
        start_idx += 24 # 24 Stunden weiter

    if len(forecasts) == 1:
        return forecasts[0]
    else:
        return reduce(lambda a, b: a.append(b), forecasts)

# -----------------------------
# Optuna Ziel-Funktion
# -----------------------------

def objective(trial):
 # Hyperparameter
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    lags = trial.suggest_int("lags", 24, 48)

 # Modell konfigurieren
    model = RegressionModel(
        model=XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        ),
        lags=lags,
        lags_past_covariates=[-lags + i for i in range(lags)],
        lags_future_covariates=[i for i in range(forecast_horizon)],
        output_chunk_length=forecast_horizon,
    )

    model.fit(series=train, past_covariates=past_train, future_covariates=future_train)

    val_forecast = rolling_forecast_eval(
        model, val, past_val, future_val,
        start_time=pd.Timestamp("2023-04-10 23:00:00"),
        end_time=pd.Timestamp("2024-04-07 09:00:00"),
        lags=lags,
        horizon=forecast_horizon
    )
    val_target = val.slice_intersect(val_forecast)

    mae_val = mae(val_target, val_forecast)
    rmse_val = rmse(val_target, val_forecast)

    os.makedirs("./results", exist_ok=True)

    metrics_path = f"./results/XGB_metrics_validationset_V{trial.number}.csv"
    df_result_path = f"./results/XGB_forecast_vs_truth_validationset_V{trial.number}.csv"
    plot_path = f"./results/XGB_forecast_plot_validationset_V{trial.number}.png"
    model_path = f"./results/XGB_model_sliding_V{trial.number}.pkl"

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
    val_forecast.plot(label="XGB Forecast [Euro/MWh]", lw=2, color="blue")
    plt.title("Rolling Forecast mit Optuna-getuntem XGBoost-Modell")
    plt.xlabel("Zeit")
    plt.ylabel("Strompreis CH [Euro/MWh]")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Modell speichern
    model.save(model_path)
    print("XGBoost-Modell gespeichert in:", model_path)

    return rmse(val_target, val_forecast)

# -----------------------------
# Optuna-Studie starten
# -----------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, timeout=21600)  # 10 trials, 12h timeout

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

