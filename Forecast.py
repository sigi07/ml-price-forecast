import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL

# Weiter mit Saisonlalität evaluieren
# Sinvolle train,test, val Aufteilung

# Datei laden
df_raw = pd.read_csv("/Users/andrinsiegenthaler/Desktop/Thesis/Code/master-thesis/trading-algorithm/data/processed_combined_df.csv")
df_raw = df_raw.rename(columns={"Unnamed: 0": "datetime"})
df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], utc=True)
df = df_raw.set_index("datetime").sort_index().asfreq("15min")

# Ausgabe-Spalte
print(df.columns)

# DataFrame-Statistiken anzeigen
print(df.describe().transpose())

# Zielreihe vorbereiten
y = df["CH_price [Euro]"].copy().interpolate(method="time")

#Visualisierung der Zeitreihe

df["CH_price [Euro]"].plot(style='.', figsize=(15,5), title='Strompreis in der Schweiz')
plt.ylabel("Euro pro MWh")
plt.xlabel("Zeit")
plt.show()

# STL-Zerlegung (z.B. Tageszyklus bei 15min-Takt: period=96)
stl = STL(y, period=96)
result = stl.fit()

# Saisonale Komponente als neue Spalte speichern
df["seasonal_96"] = result.seasonal

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

# 1. Halbjahr 2022
df.loc["2022":"2022-06"]["seasonal_96"].plot(ax=axes[0], color='tab:blue')
axes[0].set_title("Tages-Saisonalität im 1. Halbjahr 2022")
axes[0].set_ylabel("CH_price [Euro]")
axes[0].grid(True)

# 2. Halbjahr 2022
df.loc["2022-07":"2022-12"]["seasonal_96"].plot(ax=axes[1], color='tab:orange')
axes[1].set_title("Tages-Saisonalität im 2. Halbjahr 2022")
axes[1].set_ylabel("CH_price [Euro]")
axes[1].grid(True)

# 1. Halbjahr 2023
df.loc["2023":"2023-06"]["seasonal_96"].plot(ax=axes[2], color='tab:green')
axes[2].set_title("Tages-Saisonalität im 1. Halbjahr 2023")
axes[2].set_ylabel("CH_price [Euro]")
axes[2].set_xlabel("Zeit")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Training- und Testdaten aufteilen
df = df.sort_index()  # oder df.sort_values("timestamp_column")

# Annahme: df.index ist ein DatetimeIndex

# Testdaten: genau ein volles Jahr (Mai 2024 bis Mai 2025)
test_start = pd.Timestamp("2024-05-01", tz="UTC")
test_end = pd.Timestamp("2025-05-01", tz="UTC")  # exklusiv

df_test = df[(df.index >= test_start) & (df.index < test_end)]

# Trainingsdaten: alles vor dem Testzeitraum
df_train = df[df.index < test_start]

# Kontrollausgabe
print(f"Train: {len(df_train)} Zeilen")
print(f"Test: {len(df_test)} Zeilen")

# Spalte für Plot
column = "CH_price [Euro]"

# Plot
plt.figure(figsize=(14, 5))
plt.plot(df_train.index, df_train[column], label="Train", alpha=0.8)
plt.plot(df_test.index, df_test[column], label="Test", alpha=0.8)
plt.title("Zeitreihe: CH_price [Euro]")
plt.xlabel("Datum")
plt.ylabel("Preis [Euro]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
