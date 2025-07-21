import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import mae, rmse, mape
from xgboost import XGBRegressor

# -----------------------
# Daten laden
# -----------------------
df = pd.read_csv("./data/processed_combined_df.csv", index_col="Unnamed: 0", parse_dates=True)

# Zielspalte
target_col = "CH_price [Euro]"

# Forecast-Features automatisch erkennen
forecast_features = [col for col in df.columns if "forecast" in col.lower()]

# Weitere exogene Features
other_exog_features = [
    "FR_price [Euro]",
    "IT_NORD_price [Euro]",
    "AT_price [Euro]",
    "AT_generation_Solar [MWh]",
    "AT_generation_Wind_Onshore [MWh]"
]

# Alle exogenen Features kombinieren
relevant_features = forecast_features + other_exog_features

# -----------------------
# Zeitreihen erstellen (ohne Skalierung)
# -----------------------
series = TimeSeries.from_dataframe(df, value_cols=target_col)
exog_series = TimeSeries.from_dataframe(df, value_cols=relevant_features)

# -----------------------
# Zeit-Splits
# -----------------------
val_start = pd.Timestamp("2023-04-10 00:00:00")
val_end = pd.Timestamp("2024-04-09 23:45:00")
test_start = pd.Timestamp("2024-04-10 00:00:00")
test_end = pd.Timestamp("2025-04-09 23:45:00")

train, valtest = series.split_before(val_start)
val, test = valtest.split_before(test_start)
val = val.slice(val_start, val_end)
test = test.slice(test_start, test_end)

exog_train, exog_valtest = exog_series.split_before(val_start)
exog_val, exog_test = exog_valtest.split_before(test_start)
exog_val = exog_val.slice(val_start, val_end)
exog_test = exog_test.slice(test_start, test_end)

# -----------------------
# Erstes Modell zur Feature Selection trainieren
# -----------------------
model_initial = RegressionModel(
    lags=48,
    lags_future_covariates=[0],
    model=XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )
)
model_initial.fit(series=train, future_covariates=exog_train)

# -----------------------
# Feature Importance analysieren
# -----------------------
xgb_model = model_initial.model
importances = xgb_model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": exog_series.components,
    "Importance": importances[:len(exog_series.components)]
}).sort_values(by="Importance", ascending=False)

print("Top Features:")
print(feature_df.head(10))

# -----------------------
# Nur Top 5 Features behalten (ohne Skalierung)
# -----------------------
top_features = feature_df.head(5)["Feature"].tolist()
exog_series_top = exog_series[top_features]

# Neue Splits mit Top-Features
exog_train_top, exog_valtest_top = exog_series_top.split_before(val_start)
exog_val_top, exog_test_top = exog_valtest_top.split_before(test_start)
exog_val_top = exog_val_top.slice(val_start, val_end)
exog_test_top = exog_test_top.slice(test_start, test_end)

# -----------------------
# Finales Modell mit Top-Features trainieren
# -----------------------
model_final = RegressionModel(
    lags=48,
    lags_future_covariates=[0],
    model=XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )
)
model_final.fit(series=train, future_covariates=exog_train_top)

# -----------------------
# Prognose und Bewertung
# -----------------------
forecast_val = model_final.predict(len(val), future_covariates=exog_val_top)

mae_val = mae(val, forecast_val)
rmse_val = rmse(val, forecast_val)
mape_val = mape(val, forecast_val)

print(f"\nFinale Validierungsmetriken (mit Top-Features):")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"MAPE: {mape_val:.2f}%")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(12, 5))
val.plot(label="Validation (True)")
forecast_val.plot(label="XGBoost Forecast")
plt.legend()
plt.title("XGBoost Prognose mit Top-Feature-Selektion (ohne Skalierung)")
plt.grid()
plt.tight_layout()
plt.show()
