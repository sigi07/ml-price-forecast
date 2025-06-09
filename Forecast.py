import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MAE
from pytorch_lightning import Trainer
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pytorch_forecasting.data import GroupNormalizer

# === Daten einlesen und vorbereiten ===
df = pd.read_csv(
    "/Users/andrinsiegenthaler/Documents/MAS/Thesis/Code/master-thesis/data/processed_combined_df.csv",
    parse_dates=True
)

# Zeitindex verarbeiten
df['timestamp'] = pd.to_datetime(df['Unnamed: 0'])
df = df.drop(columns=['Unnamed: 0'])
df = df.set_index('timestamp')
df = df.sort_index()

# Zusätzliche Spalten generieren
df["time_idx"] = np.arange(len(df))
df["group"] = "CH"
df["hour"] = df.index.hour
df["weekday"] = df.index.weekday
df["month"] = df.index.month

# === Visualisierung ===
plt.subplot(4, 1, 1)
plt.plot(df.index, df["CH_price [Euro]"], label="CH_price [Euro]")
plt.ylabel("Euro pro MWh")
plt.title("Schweizer Day-Ahead Strompreis von 2020 bis 2025")

plt.subplot(4, 1, 2)
plt.plot(df.index, df["CH_load [MWh]"], label="CH_load [MWh]", color="orange")
plt.ylabel("MWh")
plt.title("Last in der Schweiz")

plt.subplot(4, 1, 3)
plt.plot(df.index, df["CH_generation_Total [MWh]"], label="CH_generation_Total [MWh]", color="green")
plt.ylabel("MWh")

plt.subplot(4, 1, 4)
plt.plot(df.index, df["CH_hydro_storage [MWh]"], label="CH_hydro_storage [MWh]", color="purple")
plt.ylabel("MWh")
plt.title("Schweizer Wasserspeicher-Kapazität")

plt.xlabel("Zeit")
plt.tight_layout()
plt.show()

# === Split-Index definieren ===
split_index = int(len(df) * 0.8)

# Vor dem Dataset-Split aufteilen
df_train = df[df["time_idx"] < split_index]
df_val = df[df["time_idx"] >= split_index]

# === TimeSeriesDataSet für Training & Validation ===
training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target="CH_price [Euro]",
    group_ids=["group"],
    max_encoder_length=96,
    max_prediction_length=96,
    time_varying_known_reals=["hour", "weekday", "month"],
    time_varying_unknown_reals=["CH_price [Euro]"],
    target_normalizer=GroupNormalizer(groups=["group"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization=True)

train_dataloader = training.to_dataloader(train=True, batch_size=64)
val_dataloader = validation.to_dataloader(train=False, batch_size=64)

tft_model = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    learning_rate=1e-3,
    log_interval=10,
    log_val_interval=1
)

trainer = Trainer(
    max_epochs=30,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # für schnelles Testen
)

# Beispiel: LSTM trainieren
# trainer.fit(lstm_model, train_dataloader, val_dataloader)

# === Klassischer XGBoost-Vergleich ===
df_train = df.iloc[:split_index]
df_val = df.iloc[split_index:]

features = ["hour", "weekday", "month"]
target = "CH_price [Euro]"

X_train = df_train[features]
y_train = df_train[target]
X_val = df_val[features]
y_val = df_val[target]

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=3)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f"XGBoost RMSE: {rmse:.2f}")

# === Vorhersage mit dem TFT-Modell ===
raw_predictions = tft_model.predict(val_dataloader, mode="raw", return_x=True)

# Zugriff auf Vorhersagen und Inputdaten
predictions = raw_predictions[0]  # shape: (batch_size, prediction_length, quantiles)
x = raw_predictions[1]

# Mittelwert der quantilen Vorhersagen nehmen
tft_preds = predictions["prediction"].detach().cpu().numpy()
tft_preds_mean = tft_preds.mean(axis=2)  # (batch_size x prediction_length)

# Wahre Werte aus dem Validation-Set extrahieren
true_vals = x["decoder_target"].detach().cpu().numpy()

# Für Visualisierung: nur erstes Beispiel anzeigen
tft_y_true = true_vals[0]
tft_y_pred = tft_preds_mean[0]


# === XGBoost Vergleich (bereits berechnet)
# y_val, y_pred sind schon definiert – nutze z. B. die letzten 96 Punkte
xgb_y_true = y_val[-96:].values
xgb_y_pred = y_pred[-96:]


plt.figure(figsize=(12, 6))
plt.plot(range(96), tft_y_true, label="TFT True", linestyle="--")
plt.plot(range(96), tft_y_pred, label="TFT Prediction")
plt.plot(range(96), xgb_y_true, label="XGBoost True", linestyle="--", alpha=0.7)
plt.plot(range(96), xgb_y_pred, label="XGBoost Prediction", alpha=0.7)
plt.xlabel("Zeitschritt (15-Min-Takt)")
plt.ylabel("CH_price [Euro]")
plt.title("Vergleich Vorhersagen: TFT vs. XGBoost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

