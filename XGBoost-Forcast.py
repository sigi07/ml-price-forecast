from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.compose import ForecastingPipeline, make_reduction
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.date import DateTimeFeatures
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Daten einlesen und vorbereiten
df = pd.read_csv(
    "/Users/andrinsiegenthaler/Documents/MAS/Thesis/Code/master-thesis/data/processed_combined_df.csv",
    parse_dates=True
)

# Zeitindex verarbeiten
df['timestamp'] = pd.to_datetime(df['Unnamed: 0'])
df = df.drop(columns=['Unnamed: 0'])
df = df.set_index('timestamp')
df = df.sort_index()


df["CH_price [Euro]"].plot(style='.', figsize=(15,5), title='Strompreis in der Schweiz')
plt.ylabel("Euro pro MWh")
plt.xlabel("Zeit")
plt.show()

# Zielvariable und Feature-Auswahl
target_col = "CH_price [Euro]"
y = df[target_col].copy()
X = df.drop(columns=[target_col])

# Forecasting-Horizont: 38 Zeitschritte = 9.5 Stunden (15min-Intervalle)
fh = list(range(1, 38 + 1))  # 1-based index for forecasting horizon

# Feature Engineering Pipeline
pipe = ForecastingPipeline(steps=[
    ("lag", Lag([1, 4, 24, 96])),  # Lags: 15min, 1h, 6h, 1 Tag
    ("date", DateTimeFeatures(keep_original_columns=True)),
    ("regressor", make_reduction(
        XGBRegressor(n_estimators=100, max_depth=4, random_state=42),
        strategy="recursive",
        window_length=96,
        scitype="infer",
    ))
])

# Expanding Window Splitter: täglicher Schritt = 96 Schritte (96*15min = 24h)
cv = ExpandingWindowSplitter(
    initial_window=96 * 30,  # 30 Tage als Trainingsbeginn
    step_length=96,          # täglich erweitern
    fh=fh
)

# Ergebnisse sammeln
y_preds = []
y_trues = []
cutoffs = []

for train_idx, test_idx in cv.split(y):
    y_train = y.iloc[train_idx]
    X_train = X.iloc[train_idx]
    y_test = y.iloc[test_idx]

    pipe.fit(y_train, X=X_train)
    y_pred = pipe.predict(fh=ForecastingHorizon(y_test.index, is_relative=False), X=X.iloc[test_idx])

    y_preds.append(y_pred)
    y_trues.append(y_test)
    cutoffs.append(y_train.index[-1])

# Evaluation: MAE berechnen
mae_scores = [mean_absolute_error(y_t, y_p) for y_t, y_p in zip(y_trues, y_preds)]

# Mittelwert MAE
avg_mae = np.mean(mae_scores)

# Plot letzter Forecast
plt.figure(figsize=(12, 4))
plt.plot(y_trues[-1].index, y_trues[-1], label="True")
plt.plot(y_preds[-1].index, y_preds[-1], label="Forecast")
plt.title(f"Letzter Forecast @ {cutoffs[-1]} | MAE: {mae_scores[-1]:.2f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

