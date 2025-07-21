import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from datetime import timedelta
from darts import concatenate
from darts.utils.statistics import check_seasonality
from darts.utils.statistics import plot_acf
from darts.models import Prophet
from darts.metrics import mae, rmse, mape

# ============================================================================
# Daten laden & vorbereiten
# ============================================================================
df = pd.read_csv("./data/processed_combined_df.csv", index_col="Unnamed: 0", parse_dates=True)

# Zeitzone entfernen (macht Zeitstempel tz-naiv)
df.index = df.index.tz_convert(None)

# Zielvariable
target_col = "CH_price [Euro]"

# Spalten identifizieren
forecast_cols = [col for col in df.columns if "forecast" in col and col != target_col]
other_cols = [col for col in df.columns if col != target_col and col not in forecast_cols]

# NaN-Werte ersetzen
df[forecast_cols] = df[forecast_cols].fillna(0)

# Zeitreihen erstellen
series = TimeSeries.from_dataframe(df, value_cols=target_col)
future_covariates = TimeSeries.from_dataframe(df, value_cols=forecast_cols) if forecast_cols else None
past_covariates = TimeSeries.from_dataframe(df, value_cols=other_cols) if other_cols else None

# ============================================================================
# Zeitliche Aufteilung: Neues kombiniertes Train-Set + Test-Set
# ============================================================================
val_start = pd.Timestamp("2023-04-10 10:00:00")  # bleibt erhalten
test_start = pd.Timestamp("2024-04-10 10:00:00")
test_end   = pd.Timestamp("2025-04-10 00:00:00")

# Neues kombiniertes Train-Set: alles vor test_start
train = series.slice(series.start_time(), test_start - pd.Timedelta("15min"))
test = series.slice(test_start, test_end)

# past_covariates kombinieren (falls vorhanden)
if past_covariates:
    past_train = past_covariates.slice(series.start_time(), test_start - pd.Timedelta("15min"))
    past_test  = past_covariates.slice(test_start, test_end)

# future_covariates: für Training reicht bis kurz vor Testbeginn, Test muss darüber hinaus gehen
if future_covariates:
    forecast_horizon = pd.Timedelta("1 days") + pd.Timedelta("14 hours")  # wie zuvor: 39h
    future_train = future_covariates.slice(series.start_time(), test_start - pd.Timedelta("15min"))
    future_test  = future_covariates.slice(test_start, test_end + forecast_horizon)


# ============================================================================
# Visualisierung vorbereiten (nur bis Test-Start)
# ============================================================================
df_plot = df[df.index < val_start]

# Spaltenlogik basierend auf der neuen Gruppierung
target_col = "CH_price [Euro]"
forecast_cols = [col for col in df.columns if "forecast" in col and col != target_col]
past_cols = [col for col in df.columns if col not in forecast_cols and col != target_col]

# ============================================================================
# Preisplot: Nur Spalten mit "Euro" oder "Euro/MWh" aus allen 3 Gruppen
# ============================================================================
price_cols = [
    col for col in [target_col] + forecast_cols + past_cols
    if "Euro" in col or "Euro/MWh" in col
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
# MWh-Plot: Nur CH-bezogene MWh-Spalten aus allen Gruppen
# ============================================================================
mwh_cols = [
    col for col in [target_col] + forecast_cols + past_cols
    if "MWh" in col and "CH" in col
]

df_mwh = df_plot[mwh_cols]
df_mwh_daily = df_mwh.resample("D").mean()

# Optional: Falls zu viele Spalten vorhanden sind, auf max. 10 beschränken
subset_cols = df_mwh_daily.columns[:10]
df_mwh_melted = df_mwh_daily[subset_cols].melt(var_name="Column", value_name="Value")

plt.figure(figsize=(14, 6))
sns.violinplot(x="Column", y="Value", data=df_mwh_melted, inner="box")
plt.xticks(rotation=45)
plt.title("Verteilung der täglichen Mittelwerte (CH MWh)")
plt.tight_layout()
plt.show()

# ============================================================================
# Saisonale Muster prüfen
# ============================================================================
print("Seasonality check:")
print("Daily:", check_seasonality(train, m=96, max_lag=7*96))        # 1 Tag, 7 Tage
print("Weekly:", check_seasonality(train, m=96*7, max_lag=96*30))    # 1 Woche, 1 Monat
print("Monthly:", check_seasonality(train, m=96*30, max_lag=96*90))  # 1 Monat, 3 Monate
print("Yearly:", check_seasonality(train, m=96*365, max_lag=96*730)) # 1 Jahr, 2 Jahre

# ============================================================================
# Autokorrelationsanalyse (ACF) für geglättete Zeitreihen
# ============================================================================
# Stundenmittelwerte
hourly_series = train.pd_series().resample("1H").mean()
plot_acf(TimeSeries.from_series(hourly_series), m=24, alpha=0.05, max_lag=48)
plt.title("ACF der Stundenmittelwerte (2 Tage)")
plt.tight_layout()
plt.show()

# Tagesmittelwerte
daily_series = train.pd_series().resample("1D").mean()
plot_acf(TimeSeries.from_series(daily_series), m=7, alpha=0.05, max_lag=30)
plt.title("ACF der Tagesmittelwerte (1 Monat)")
plt.tight_layout()
plt.show()

# Monatsmittelwerte
monthly_series = train.pd_series().resample("1M").mean()
plot_acf(TimeSeries.from_series(monthly_series), m=12, alpha=0.05, max_lag=12)
plt.title("ACF der Monatsmittelwerte (1 Jahr)")
plt.tight_layout()
plt.show()

# ============================================================================
# Gleitender Mittelwert im Originalsignal
# ============================================================================
train_df = train.pd_dataframe()
train_df["rolling_mean"] = train_df[target_col].rolling(window=96*7).mean()  # 7 Tage Fenster
train_df.plot(figsize=(12, 5), title="Train-Zeitreihe mit 7-Tage gleitendem Mittelwert")
plt.tight_layout()
plt.show()

# ============================================================================
# Prophet-Modell konfigurieren
# ============================================================================
# Manuelle Changepoints (alle 4 Wochen, Ukraine-Zeitraum ausgeschlossen)
#train_index = train.time_index
#changepoints = pd.date_range(start=train_index[0], end=train_index[-1], freq="4W")
#changepoints = changepoints[(changepoints < "2022-02-01") | (changepoints > "2023-01-15")]

# Modell initialisieren
model = Prophet(
    growth="linear",
    changepoint_prior_scale=0.5,
    country_holidays="CH",
    seasonality_mode="multiplicative",
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    add_encoders={"cyclic": {"future": ["hour", "day", "dayofweek"]}},
)

# ============================================================================
# Modell trainieren
# ============================================================================
#model.fit(
    #series=train,
    #future_covariates=future_train,
    #past_covariates=past_train
#)

# ============================================================================
# Modell fitten (einmalig)
# ============================================================================
model.fit(
    series=train,
    future_covariates=future_train,
    #past_covariates=past_train  # optional, Prophet nutzt sie nicht
)

# ============================================================================
# Einmalige Vorhersage für den gesamten Val-Bereich
# ============================================================================
forecast = model.predict(
    n=len(test),  # oder len(val), wenn du gegen val_start-end evaluieren willst
    future_covariates=future_test,
)


# ============================================================================
# Evaluation & Visualisierung
# ============================================================================
# Forecast-Zeitreihen zusammenführen
forecast_concat = forecast_val[0]
for f in forecast_val[1:]:
    forecast_concat = forecast_concat.append(f)

# Ground Truth entsprechend beschneiden
val_actual = series.slice_intersect(forecast_concat)

# Metriken berechnen
mae_val = mae(val_actual, forecast_concat)
rmse_val = rmse(val_actual, forecast_concat)
mape_val = mape(val_actual, forecast_concat)

print(f"MAE (Sliding Window): {mae_val:.3f}")
print(f"RMSE (Sliding Window): {rmse_val:.3f}")
print(f"MAPE (Sliding Window): {mape_val:.3f}%")

# Plot
plt.figure(figsize=(12, 5))
val_actual.plot(label="Validation (True)")
forecast_concat.plot(label="Sliding Window Forecast (Prophet)")
plt.legend()
plt.title("Prophet Sliding Window Forecast vs. Validation")
plt.tight_layout()
plt.show()