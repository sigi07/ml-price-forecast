from darts.dataprocessing.transformers import MissingValuesFiller
import pandas as pd
import optuna
from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import rmse, mae, mape
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

# Füge den MissingValuesFiller hinzu
filler = MissingValuesFiller()

# Daten einlesen
df = pd.read_csv("./data/processed_combined_df.csv", index_col="Unnamed: 0", parse_dates=True)
df.index = df.index.tz_convert(None)
print(f"Anzahl Spalten in df: {df.shape[1]}")

target_col = "CH_price [Euro]"
forecast_cols = [col for col in df.columns if "forecast" in col and col != target_col]
other_cols = [col for col in df.columns if col != target_col and col not in forecast_cols and "CH" in col]  # 🔧 Nur CH
df[forecast_cols] = df[forecast_cols].fillna(0)

series = TimeSeries.from_dataframe(df, value_cols=target_col)
future_covariates = TimeSeries.from_dataframe(df, value_cols=forecast_cols) if forecast_cols else None
past_covariates = TimeSeries.from_dataframe(df, value_cols=other_cols) if other_cols else None

print(f"Anzahl Werte past_covariates: {past_covariates.values().size}")
print(f"Anzahl Spalten in past_covariates: {past_covariates.width}")

# Wende ihn auf alle relevanten TimeSeries an
series = filler.transform(series)
past_covariates = filler.transform(past_covariates)
future_covariates = filler.transform(future_covariates)

# Zeit-Splits (nicht verändern!)
val_start = pd.Timestamp("2023-04-10 00:00:00")
val_end = pd.Timestamp("2024-04-10 00:00:00")
test_start = pd.Timestamp("2024-04-10 00:00:00")
test_end = pd.Timestamp("2025-04-10 10:00:00")

train, valtest = series.split_before(val_start)
val, test = valtest.split_before(test_start)
val_full = val.slice(val_start, val_end)
test = test.slice(test_start, test_end)

future_train, future_valtest = future_covariates.split_before(val_start)
future_val, future_test = future_valtest.split_before(test_start)
future_val_full = future_val.slice(val_start, val_end)
future_test = future_test.slice(test_start, test_end)

past_train, past_valtest = past_covariates.split_before(val_start)
past_val, past_test = past_valtest.split_before(test_start)
past_val_full = past_val.slice(val_start, val_end)
past_test = past_test.slice(test_start, test_end)

# Forecast-Horizont
forecast_horizon = 152 # 15-Minuten-Schritte

# ---------------------------------------------
# Optuna-Ziel-Funktion
# ---------------------------------------------
def objective(trial):
    print(f"Starte Trial {trial.number} mit Parametern: {trial.params}")

    # Hyperparameter
    n_estimators = trial.suggest_int("n_estimators", 10, 50)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    lags = trial.suggest_int("lags", 96, 192)


    # Zeitfenster für Val dynamisch anpassen
    past_val_dyn = filler.transform(
        past_covariates.slice(
            val_start + pd.Timedelta(minutes=15 * (lags)), 
            val_end - pd.Timedelta(minutes=15 * (forecast_horizon + 1))
        )
    )

    val_dyn = filler.transform(
        val_full.slice(
            val_start + pd.Timedelta(minutes=15 * (lags)),
            val_end - pd.Timedelta(minutes=15 * (forecast_horizon + 1))
        )
    )
    
    future_val_dyn = filler.transform(
        future_val_full.slice(
            val_start + (pd.Timedelta(minutes=15 * 2 * (lags))),
            val_end
        )
    )

    # Start- und Endzeitpunkt ausgeben
    print("Start von past_val_dyn:", past_val_dyn.start_time())
    print("Ende von past_val_dyn:", past_val_dyn.end_time())
    # Start- und Endzeitpunkte ausgeben
    print("val_dyn Start:", val_dyn.start_time())
    print("val_dyn Ende:", val_dyn.end_time())
    print("future_val_dyn Start:", future_val_dyn.start_time())
    print("future_val_dyn Ende:", future_val_dyn.end_time())

    # Modell konfigurieren
    model = RegressionModel(
        model=RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
        ),
        lags=lags,
        lags_past_covariates=[-lags + i for i in range(lags)],
        lags_future_covariates=[i for i in range(forecast_horizon)],
        output_chunk_length=forecast_horizon,
    )

    # Modell fitten
    model.fit(
        series=filler.transform(train),
        past_covariates=filler.transform(past_train),
        future_covariates=filler.transform(future_train),
    )

    # Vorhersage
    forecast = model.predict(
        n=forecast_horizon,
        series=val_dyn,
        past_covariates=past_val_dyn,
        future_covariates=future_val_dyn,
    )

    # Evaluation
    val_target = val.slice_intersect(forecast)
    rmse_val = rmse(val_target, forecast)
    mae_val = mae(val_target, forecast)
    mape_val = mape(val_target, forecast)

    print(f"Trial {trial.number} – RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")
    return rmse_val

# ---------------------------------------------
# Optuna-Studie starten
# ---------------------------------------------
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1, timeout=180)

print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_trial.params}")

# ---------------------------------------------
# Bestes Modell mit den optimalen Parametern aus der Optuna-Studie erstellen
# ---------------------------------------------

best_params = study.best_params
best_model = RegressionModel(
    model= RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        max_features=best_params["max_features"],
        random_state=42,
    ),

    lags=best_params["lags"],
    lags_past_covariates=[-best_params["lags"] + i for i in range(best_params["lags"])],
    lags_future_covariates=[i for i in range(forecast_horizon)],
    output_chunk_length=forecast_horizon,
)

lags=best_params["lags"]

# Zeitfenster für Val dynamisch anpassen
past_val_dyn = filler.transform(
    past_covariates.slice(
        val_start + pd.Timedelta(minutes=15 * (lags)), 
        val_end - pd.Timedelta(minutes=15 * (forecast_horizon + 1))
    )
)

val_dyn = filler.transform(
    val_full.slice(
        val_start + pd.Timedelta(minutes=15 * (lags)),
        val_end - pd.Timedelta(minutes=15 * (forecast_horizon + 1))
    )
)
    
future_val_dyn = filler.transform(
    future_val_full.slice(
        val_start + (pd.Timedelta(minutes=15 * 2 * (lags))),
        val_end
    )
)

# Modell mit besten Parametern erneut trainieren
best_model.fit(
    series=train,
    past_covariates=past_train,
    future_covariates=future_train,
)

# Vorhersage auf dem Validierungsset
forecast_val = model.predict(
    n=forecast_horizon,
    series=val_dyn,
    past_covariates=past_val_dyn,
    future_covariates=future_val_dyn,
)

# Fehlermaße berechnen
val_target = val.slice_intersect(forecast_val)
mae_val = mae(val, forecast_val)
mape_val = mape(val, forecast_val)
rmse_val = rmse(val, forecast_val)

# Ordner erstellen (falls noch nicht vorhanden)
os.makedirs("./results", exist_ok=True)

# -----------------------
# 1. Fehlermaße speichern
# -----------------------
with open("./results/RF_metrics.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["MAE", "MAPE", "RMSE"])
    writer.writerow([mae_val, mape_val, rmse_val])

print("Fehlermetriken gespeichert in results/RF_metrics.csv")

# ----------------------------------------
# 2. Forecast vs. Wahrheit als CSV speichern
# ----------------------------------------

# Slice der echten Werte passend zum Forecast
val_truth = val.slice_intersect(forecast_val)

# DataFrame erstellen
df_result = pd.DataFrame({
    "timestamp": forecast_val.time_index,
    "forecast": forecast_val.values().flatten(),
    "truth": val_truth.values().flatten(),
})

# CSV schreiben
df_result.to_csv("./results/RF_forecast_vs_truth.csv", index=False)

print("Forecast-Vergleich gespeichert in results/RF_forecast_vs_truth.csv")

# --------------------------
# 3. Plot im Stil von Darts speichern
# --------------------------

plt.figure(figsize=(10, 4))

# Echte Werte in Schwarz
val.plot(label="CH_price [Euro]", lw=2, color="black")

# Falls probabilistisch: Konfidenzintervall anzeigen
if forecast_val.n_samples > 1:
    forecast_val.plot(
        label="forecast",
        lw=2,
        color="blue",
        low_quantile=0.1,
        high_quantile=0.9,
        display_confidence_interval=True,
        alpha=0.8
    )
else:
    # Wenn deterministisch (z. B. RandomForest/XGBoost), nur Linie
    forecast_val.plot(label="RF Forecast", lw=2, color="blue")

# Styling wie im Screenshot
plt.title("Random Forest Forecast von Schweizer Energiepreisen")  # Kein Titel
plt.xlabel("Zeit")  # Keine Achsenbeschriftungen
plt.ylabel("Strompreis CH in Euro")
plt.grid(False)  # Kein Gitter
plt.legend()
plt.tight_layout()
plt.savefig("./results/RF_forecast_plot.png", dpi=300)
plt.close()

print("Plot gespeichert in results/RF_forecast_plot.png")
