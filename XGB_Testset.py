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

# Pfad zum gespeicherten Modell
model_path = "./results/XGB_model_sliding_V27.pkl"

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
# Beste Hyperparameter und Model auf Testset evaluieren
# -----------------------------

# Beste Hyperparameter aus Training (manuell eingefügt)
best_params = {
    "n_estimators": 230,
    "max_depth": 7,
    "learning_rate": 0.03043378771705656,
    "subsample": 0.7016182280906526,
    "colsample_bytree": 0.5660027556373397,
    "lags": 45
}

# Modell laden
best_model_loaded = RegressionModel.load(model_path)

forecast_test = rolling_forecast_eval(
    best_model_loaded, test, past_test, future_test,
    start_time=pd.Timestamp("2024-04-10 23:00:00"),
    end_time=pd.Timestamp("2025-04-07 09:00:00"),
    lags=best_params["lags"],
    horizon=forecast_horizon
)

test_truth = test.slice_intersect(forecast_test)

mae_test = mae(test_truth, forecast_test)
rmse_test = rmse(test_truth, forecast_test)

metrics_path = f"./results/XGB_metrics_testset.csv"
df_result_path = f"./results/XGB_forecast_vs_truth_testset.csv"
plot_path = f"./results/XGB_forecast_plot_testset.png"
model_path = f"./results/XGB_model_sliding_testset.pkl"

os.makedirs("./results", exist_ok=True)

with open(metrics_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["MAE", "RMSE"])
    writer.writerow([mae_test, rmse_test])

df_result = pd.DataFrame({
    "timestamp": forecast_test.time_index,
    "forecast": forecast_test.values().flatten(),
    "truth": test_truth.values().flatten(),
})
df_result.to_csv(df_result_path, index=False)

plt.figure(figsize=(10, 4))
test_truth.plot(label="CH_price [Euro/MWh]", lw=2, color="black")
forecast_test.plot(label="XGB Forecast [Euro]", lw=2, color="blue")
plt.title("Rolling Forecast mit Optuna-getuntem XGBoost-Modell")
plt.xlabel("Zeit")
plt.ylabel("Strompreis CH [Euro/MWh]")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

# Modell speichern
best_model_loaded.save(os.path.join(model_path))
print("XGBoost-Modell gespeichert in:", os.path.join(model_path))