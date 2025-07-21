import pandas as pd
import numpy as np
import optuna
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mae, mape, rmse
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler

# Daten laden
df = pd.read_csv("./data/processed_combined_df.csv", index_col="Unnamed: 0", parse_dates=True)
df.index = df.index.tz_convert(None)
target_col = "CH_price [Euro]"
forecast_cols = [col for col in df.columns if "forecast" in col and col != target_col]
other_cols = [col for col in df.columns if col != target_col and col not in forecast_cols]
df[forecast_cols] = df[forecast_cols].fillna(0)
df[other_cols] = df[other_cols].fillna(0)

# Zielserie
series = TimeSeries.from_dataframe(df, value_cols=target_col)

# -----------------------------
# 1. Zeitbasierte Covariates erstellen
# -----------------------------
hour_series = datetime_attribute_timeseries(series, attribute="hour", one_hot=True)
day_series = datetime_attribute_timeseries(series, attribute="day", one_hot=True)
weekday_series = datetime_attribute_timeseries(series, attribute="weekday", one_hot=True)
month_series = datetime_attribute_timeseries(series, attribute="month", one_hot=True)
year_series = datetime_attribute_timeseries(series, attribute="year", one_hot=True)

# Alle stacken zu einer multivariaten TimeSeries
datetime_covariates = hour_series \
    .stack(day_series) \
    .stack(weekday_series) \
    .stack(month_series) \
    .stack(year_series)


# Skalierung (fit nur auf Training)
val_start = pd.Timestamp("2023-04-10 00:00:00")
cov_train_dt, _ = datetime_covariates.split_before(val_start)
scaler_dt = Scaler()
scaler_dt.fit(cov_train_dt)
datetime_covariates_scaled = scaler_dt.transform(datetime_covariates)

# -----------------------------
# 2. Forecast-Covariates laden & mit datetime-Covariates kombinieren
# -----------------------------
future_covariates_raw = TimeSeries.from_dataframe(df, value_cols=forecast_cols) if forecast_cols else None
if future_covariates_raw:
    future_covariates = future_covariates_raw.stack(datetime_covariates_scaled)
else:
    future_covariates = datetime_covariates_scaled

# -----------------------------
# 3. Past-Covariates laden
# -----------------------------
past_covariates = TimeSeries.from_dataframe(df, value_cols=other_cols) if other_cols else None

# -----------------------------
# 4. Zeitliche Aufteilung
# -----------------------------
val_end = pd.Timestamp("2024-04-09 23:45:00")
test_start = pd.Timestamp("2024-04-10 00:00:00")
test_end = pd.Timestamp("2025-04-09 23:45:00")

train, valtest = series.split_before(val_start)
val, test = valtest.split_before(test_start)
val = val.slice(val_start, val_end)
test = test.slice(test_start, test_end)

future_train, future_valtest = future_covariates.split_before(val_start)
future_val, future_test = future_valtest.split_before(test_start)
future_val = future_val.slice(val_start, val_end)
future_test = future_test.slice(test_start, test_end)

past_train, past_valtest = past_covariates.split_before(val_start)
past_val, past_test = past_valtest.split_before(test_start)
past_val = past_val.slice(val_start, val_end)
past_test = past_test.slice(test_start, test_end)

# Sicherstellen, dass past_val früh genug startet (für input_chunk)
min_lag = 96
past_val_start = val_start - pd.Timedelta(minutes=15 * min_lag)
past_val = past_covariates.slice(past_val_start, val_end)

# Forecast-Länge
forecast_horizon = 152

# -----------------------------
# 5. Optuna-Zielfunktion mit TFT
# -----------------------------
def objective(trial):
    input_chunk_length = trial.suggest_int("input_chunk_length", 96, 192)
    hidden_size = trial.suggest_int("hidden_size", 8, 64)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 50, 150)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    early_stopper = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        random_state=42,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={"callbacks": [early_stopper]},
    )

    model.fit(
        series=train,
        past_covariates=past_train,
        future_covariates=future_train,
        val_series=val,
        val_past_covariates=past_val,
        val_future_covariates=future_val,
        verbose=False,
    )

    forecast = model.predict(
        n=len(val),
        past_covariates=past_val,
        future_covariates=future_val,
    )

    rmse_val = rmse(val, forecast)
    mae_val = mae(val, forecast)
    mape_val = mape(val, forecast)

    print(f"Trial {trial.number} – RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    return rmse_val
