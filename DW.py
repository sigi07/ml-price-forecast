# test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import os
import seaborn as sns
import io
from contextlib import redirect_stdout
import missingno as msno
from statsmodels.graphics.tsaplots import plot_acf


Version = 1

# Paths to the uploaded files
file_paths = [
    "./data/combined_crossborder_flows_all.csv",
    "./data/combined_aggregate_water_reservoirs_and_hydro_storage_all_countries.csv",
    "./data/combined_forecast_all_countries.csv",
    "./data/combined_load_all_countries.csv",
    "./data/combined_generation_all_countries.csv",
    "./data_price/combined_day_ahead_prices_all_countries.csv"
]

# Lade jede CSV-Datei in einen DataFrame
dfs = [pd.read_csv(os.path.join(file_path.lstrip("/"))) for file_path in file_paths]

# Stelle sicher, dass alle DataFrames eine 'timestamp'-Spalte haben und setze sie als Index
for i, df in enumerate(dfs):
    if "timestamp" not in df.columns:
        raise ValueError(f"DataFrame {i + 1} hat keine 'timestamp'-Spalte.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # In datetime umwandeln
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='first')]  # Doppelte Zeitstempel entfernen
    dfs[i] = df  # Aktualisiere den DataFrame in der Liste

# Kombiniere alle DataFrames zu einem gemeinsamen DataFrame, ausgerichtet auf den Zeitstempel-Index
combined_df = pd.concat(dfs, axis=1)

# Display the combined DataFrame
print("Kombinierter DataFrame mit aktualisierten Spaltennamen:")
print(combined_df.head())

# Überprüfung des DataFrames
print("\nForm des kombinierten DataFrames:")
print(combined_df.shape)

def process_time_series(dataframe, categorical_columns=None, negative_values_columns=None, report_path=None):
    """
    Überprüft, analysiert und dokumentiert die Qualität und Struktur eines Zeitreihen-DataFrames.

    Diese Funktion dient der umfassenden Datenprüfung und -analyse im Rahmen der Zeitreihen-Vorverarbeitung.
    Sie gibt einen detaillierten Überblick über die Datenqualität und potentielle Probleme, die vor der Modellierung
    behoben werden sollten.

    Folgende Prüfungen und Analysen werden durchgeführt:
    - Prüfung auf fehlende Werte (NaN) pro Spalte, inklusive Anzahl und prozentualem Anteil.
    - Überprüfung auf inkonsistente Kategorien in angegebenen kategorischen Spalten.
    - Überprüfung auf negative Werte in Spalten, die normalerweise nur positive Werte enthalten sollten.
    - Analyse von Blöcken von aufeinanderfolgenden fehlenden Werten (>24) pro Zeitreihe.
    - Überprüfung, ob die Zeitstempel im Index regelmässig sind (1 Stunde).
    - Zählen und Ausgeben der Anzahl vollständig duplizierter Zeilen.

    Args:
        dataframe (pd.DataFrame): Der zu analysierende Zeitreihen-DataFrame.
        categorical_columns (list, optional): Liste der Spaltennamen, die als kategorisch gelten und auf inkonsistente Werte geprüft werden.
        negative_values_columns (list, optional): Liste der Spaltennamen, die auf negative Werte geprüft werden sollen.

    Returns:
        pd.DataFrame: Der unveränderte, aber analysierte DataFrame (zur weiteren Verarbeitung).
    """
    output_buffer = io.StringIO()

    def write(msg):
        print(msg)
        output_buffer.write(str(msg) + "\n")

    if dataframe.empty:
        write("Der übergebene DataFrame ist leer. Verarbeitung wird übersprungen.")
        if report_path:
            with open(report_path, "w") as f:
                f.write(output_buffer.getvalue())
        return dataframe

    # Überprüfung auf verbleibende fehlende Werte
    categorical_columns = categorical_columns or []
    negative_values_columns = negative_values_columns or []

    # Überprüfung auf inkonsistente Kategorien in kategorischen Spalten
    if categorical_columns:
        write("\nÜberprüfung auf inkonsistente Kategorien in ausgewählten kategorischen Spalten:")
        for column in categorical_columns:
            write(f"Unique values in '{column}': {dataframe[column].unique()}")

    # Überprüfung auf negative Werte in normalerweise positiven Spalten
    if negative_values_columns:
        negative_values_summary = {column: (dataframe[column] < 0).sum() for column in negative_values_columns}
        write("\nÜberprüfung auf negative Werte in normalerweise positiven Spalten:")
        write(negative_values_summary)

    # Überprüfung auf fehlende Daten
    write("\nÜberprüfung auf fehlende Daten:")
    missing_data = dataframe.isnull().sum()
    missing_data_percentage = (missing_data / len(dataframe)) * 100
    missing_data_summary = pd.DataFrame({'Missing Count': missing_data, 'Percentage': missing_data_percentage})
    write(missing_data_summary.sort_values(by='Missing Count', ascending=False))

    # Überprüfung auf regelmässige Zeitstempel
    write("\nÜberprüfung auf regelmässige Zeitstempel:")
    time_differences = dataframe.index.to_series().diff().dropna()
    unique_differences = time_differences.unique()
    write(f"Einzigartige Zeitdifferenzen: {unique_differences}")

    if len(unique_differences) == 1:
        write(f"Die Zeitstempel sind regelmässig mit einem Intervall von {unique_differences[0]}.")
    else:
        write("Die Zeitstempel sind nicht regelmässig. Es gibt unterschiedliche Intervalle:")
        for diff in unique_differences:
            write(f"- {diff}: {sum(time_differences == diff)} Vorkommen")

    # Überprüfung auf doppelte Zeilen
    duplicate_rows = dataframe.duplicated().sum()
    write(f"\nAnzahl der vollständigen Duplikate: {duplicate_rows}")

    # Analyse: Blöcke von aufeinanderfolgenden fehlenden Werten (>24)
    write("\nBlöcke von mehr als 24 aufeinanderfolgenden fehlenden Werten pro Zeitreihe:")
    for col in dataframe.columns:
        missing_mask = dataframe[col].isnull()
        # Finde Start und Länge jedes NaN-Blocks
        block_lengths = []
        count = 0
        for is_missing in missing_mask:
            if is_missing:
                count += 1
            else:
                if count > 0:
                    block_lengths.append(count)
                count = 0
        # Falls der letzte Block am Ende ist
        if count > 0:
            block_lengths.append(count)
        # Liste alle Blöcke mit mehr als 24 fehlenden Werten
        long_blocks = [l for l in block_lengths if l > 24]
        if long_blocks:
            write(f"Spalte '{col}': {len(long_blocks)} Block(e) mit mehr als 24 fehlenden Werten (Längen: {long_blocks})")

    # Schreibe den Bericht in die Datei, falls Pfad angegeben
    if report_path:
        with open(report_path, "w") as f:
            f.write(output_buffer.getvalue())

    return dataframe

def ride_report(dataframe, Version):
    """Führt die Datenprüfung durch und speichert den Bericht in einer Textdatei."""
    # Erstelle einen String-Buffer für die Ausgaben
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        # Setze die Anzeigeoption nur für den Report
        pd.set_option('display.max_rows', None)
        report_path = f"./data/process_time_series_report V{Version}.txt"
        dataframe = process_time_series(dataframe, report_path=report_path)
        pd.reset_option('display.max_rows')  #Setze nach dem Report wieder zurück

    # Schreibe die Ausgaben in eine Textdatei
    with open("./data/process_time_series_report.txt", "w") as f:
        f.write(output_buffer.getvalue())
    print(f"Report der Datenprüfung wurde gespeichert unter: ./data/process_time_series_report V{Version}.txt")
    
    Version += 1

    return dataframe, Version

def to_hourly_mean(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Bringt DataFrame mit ['timestamp','price'] auf exakte Stundentaktung (HH:00).
    Substündliche Punkte → stündliches Mittel, danach Rundung auf 2 Nachkommastellen.
    """
    if df_ts.empty:
        return df_ts

    df = df_ts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # TZ-Handhabung → konsistent Europe/Brussels
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Brussels")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Brussels")

    df = df.sort_values("timestamp").set_index("timestamp")

    all_full_hours = (df.index.minute == 0).all() and (df.index.second == 0).all()

    if all_full_hours:
        # Nur doppelte Stunden (DST etc.) mitteln & auf H-Frequenz setzen
        df_hourly = df.groupby(df.index).mean(numeric_only=True).asfreq("H")
    else:
        # Substündlich → Stundenmittel
        df_hourly = df.resample("H").mean(numeric_only=True)
        # Falls irgendwo doppelte Stunden entstanden sind, nochmal mitteln
        df_hourly = df_hourly.groupby(df_hourly.index).mean(numeric_only=True)

    # Auf 2 Nachkommastellen runden
    df_hourly = df_hourly.round(2)

    return df_hourly.reset_index().rename(columns={"index": "timestamp"})


#--------------------------------------------------------------------------------------------------------------------
# Erste Überprüfung des kombinierten DataFrames
#--------------------------------------------------------------------------------------------------------------------

combined_df, Version = ride_report(combined_df, Version)

# Entferne alle Zeilen bis einschliesslich 2019-12-31 22:59:59+00:00
cutoff = pd.Timestamp("2019-12-31 22:59:59+00:00")
processed_df = combined_df[combined_df.index > cutoff]

# Entferne alle eilen nach dem 2025-04-08 23:59:59
cutoff_end = pd.Timestamp("2025-04-08 23:59:59", tz="UTC")
processed_df = processed_df[processed_df.index <= cutoff_end]

# Überprüfung des DataFrames nach dem Entfernen der Zeilen
print("DataFrame nach dem Entfernen der ersten Zeilen:")
print(processed_df.head())

processed_df, Version = ride_report(processed_df, Version)

#--------------------------------------------------------------------------------------------------------------------
# Swissgrid-Daten einlesen und verarbeiten
#--------------------------------------------------------------------------------------------------------------------
# Liste der Jahresdateien (passe ggf. die Jahre und Pfade an)
years = range(2020, 2026)
dfs_swissgrid = []

for year in years:
    csv_path = os.path.join("./data", f"EnergieUebersichtCH-{year}_SG.csv")
    try:
        df = pd.read_csv(csv_path, sep=";", parse_dates=[0], dayfirst=True, usecols=[0, 1], header=0)
        df.columns = ["Zeitstempel", "Summe produzierte Energie Regelblock Schweiz in kWh"]
        df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True, utc=True)
        df.set_index("Zeitstempel", inplace=True)
        # Umrechnen von kWh → MWh und Faktor anwenden
        df["CH_generation_Total [MWh]"] = df["Summe produzierte Energie Regelblock Schweiz in kWh"] / 1000
        df["CH_generation_Total [MWh]"] = df["CH_generation_Total [MWh]"].round(1)
        dfs_swissgrid.append(df[["CH_generation_Total [MWh]"]])
        print(f"{year}: {df.shape[0]} Zeilen eingelesen.")
    except FileNotFoundError:
        print(f"Datei für {year} nicht gefunden, überspringe.")

# Alle Jahre zu einer fortlaufenden Zeitreihe verbinden
if dfs_swissgrid:
    swissgrid_full = pd.concat(dfs_swissgrid).sort_index()
    print("\nVorschau der fortlaufenden Zeitreihe CH_generation_Total [MWh]:")
    print(swissgrid_full.head())
    print(swissgrid_full.tail())
else:
    print("Keine Swissgrid-Dateien gefunden.")


# Index zurück zur Spalte machen, damit to_hourly_mean funktioniert
swissgrid_full = swissgrid_full.reset_index()

# Spalte 'Zeitstempel' in 'timestamp' umbenennen, damit to_hourly_mean funktioniert
swissgrid_full = swissgrid_full.rename(columns={"Zeitstempel": "timestamp"})

# Wandle Swissgrid-Daten auf stündliche Auflösung um
swissgrid_full = to_hourly_mean(swissgrid_full)

# Setze den Zeitstempel als Index
swissgrid_full.set_index("timestamp", inplace=True)

swissgrid_full = swissgrid_full[swissgrid_full.index > cutoff]
swissgrid_full = swissgrid_full[swissgrid_full.index <= cutoff_end]

print("Swissgrid-Daten (stündlich):")
print(swissgrid_full.head())
print("Größe von processed_df:", processed_df["CH_generation_Total [MWh]"].shape)
print("Größe von swissgrid_full:", swissgrid_full["CH_generation_Total [MWh]"].shape)

cutoff_2025 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
processed_df_cut = processed_df[processed_df.index <= cutoff_2025]
swissgrid_full_cut = swissgrid_full[swissgrid_full.index <= cutoff_2025]

# Setze Zeitzone der Swissgrid-Daten explizit auf UTC
if swissgrid_full_cut.index.tz is None:
    swissgrid_full_cut.index = swissgrid_full_cut.index.tz_localize("UTC")
else:
    swissgrid_full_cut.index = swissgrid_full_cut.index.tz_convert("UTC")

# Wähle alle Spalten, die mit "CH_generation" beginnen
ch_gen_cols = [col for col in processed_df_cut.columns if col.startswith("CH_generation")]

# Passe die Spaltennamen an: Entferne [MWh] und füge ENTSO-E hinzu
rename_dict = {col: col.replace("CH_", "").replace("[MWh]", "").strip() + " ENTSO-E" for col in ch_gen_cols}
corr_df = processed_df_cut[ch_gen_cols].rename(columns=rename_dict)

# Füge die Swissgrid-Zeitreihe als neue Spalte hinzu (Index-Abgleich erfolgt automatisch)
corr_df["Swissgrid generation data"] = swissgrid_full_cut["CH_generation_Total [MWh]"]

# Berechne die Korrelationsmatrix
corr_matrix = corr_df.corr()

# Plot als Heatmap
plt.figure(figsize=(len(corr_matrix.columns), len(corr_matrix.index)))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korrelationsmatrix: ENTSO-E & Swissgrid generation Data")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------
# Bereinigen der Generation-Spalten
#--------------------------------------------------------------------------------------------------------------------
# 
pattern = r"^(CH|IT|FR|AT|DE_LU)_.*generation.*(?!forecast)"
cols_generation_CH = [col for col in processed_df.columns
                  if (col.startswith(("CH_"))
                      and "generation" in col
                      and "forecast" not in col
                      and "Total " not in col)]

plt.figure(figsize=(18, 8))
for col in cols_generation_CH:
    plt.plot(processed_df.index, processed_df[col], label=col, alpha=0.7)

plt.title("Zeitlicher Verlauf der Produktionsdaten in der Schweiz")
plt.xlabel("Zeitstempel")
plt.ylabel("MWh")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 8))
total_cols = [
    "CH_generation_Total [MWh]",
    "FR_generation_Total [MWh]",
    "IT_generation_Total [MWh]"
]

for col in total_cols:
    if col in processed_df.columns:
        plt.plot(processed_df.index, processed_df[col], label=col, alpha=0.8)
    else:
        print(f"Spalte '{col}' nicht im DataFrame vorhanden!")

plt.title("Zeitlicher Verlauf der Gesamtproduktion für CH, FR und IT")
plt.xlabel("Zeitstempel")
plt.ylabel("MWh")
plt.legend()
plt.tight_layout()
plt.show()

# Liste der relevanten Spalten
total_cols = [
    "CH_generation_Total [MWh]",
    "AT_generation_Total [MWh]",
    "DE_LU_generation_Total [MWh]",
    "FR_generation_Total [MWh]",
    "IT_generation_Total [MWh]"
]

# Jahr wählen
year = 2023

# Liste der Länder
countries = ["CH", "AT", "DE_LU", "FR", "IT"]

# Speicher-Spalten definieren (werden ausgeschlossen)
storage_keywords = ["Pumped_Storage", "hydro_storage", "Water_Reservoir", "Energy_storage"]

for year in years:
    country_sums = {}
    for country in countries:
        # Wähle alle Produktionsspalten des Landes, die "generation" enthalten,
        # aber KEIN Speicher, KEIN "forecast" und KEIN "generation_Total"
        prod_cols = [col for col in processed_df.columns
                     if col.startswith(f"{country}_generation")
                     and not any(storage in col for storage in storage_keywords)
                     and "forecast" not in col
                     and "generation_Total" not in col]
        # Filter auf das gewünschte Jahr
        df_year = processed_df[processed_df.index.year == year]
        # Summe über alle Produktionsspalten (ohne Speicher und ohne generation_Total)
        total_sum = df_year[prod_cols].sum().sum()
        country_sums[country] = total_sum

    print(f"\nGesamtproduktion (ohne Speicher und ohne generation_Total) für das Jahr {year}:")
    for country, value in country_sums.items():
        print(f"{country}: {value:.2f} MWh")

# Liste der relevanten Spalten
total_cols = [
    "CH_generation_Total [MWh]",
    "AT_generation_Total [MWh]",
    "DE_LU_generation_Total [MWh]",
    "FR_generation_Total [MWh]",
    "IT_generation_Total [MWh]"
]

# Filter für 2023
df_2023 = processed_df[processed_df.index.year == 2023]
sum_2023 = df_2023[total_cols].sum()

# Filter für 2024
df_2024 = processed_df[processed_df.index.year == 2024]
sum_2024 = df_2024[total_cols].sum()

print("Summe der Gesamtproduktion für 2023:")
print(sum_2023)
print("\nSumme der Gesamtproduktion für 2024:")
print(sum_2024)

# Entferne alle Spalten, die mit CH_ beginnen, "generation" enthalten, aber NICHT "forecast", "Solar" oder "Wind"
pattern = r"^(CH)_.*generation.*(?!forecast)"
cols_to_remove = [col for col in processed_df.columns
                  if (col.startswith(("CH_"))
                      and "generation" in col
                      and "forecast" not in col
                      and "Solar" not in col
                      and "Wind" not in col)]

processed_df = processed_df.drop(columns=cols_to_remove)

# Entferne alle Spalten, die mit IT_, FR_, AT_ oder DE_LU_ beginnen, "generation" enthalten, aber NICHT "forecast"
pattern = r"^(IT|FR|AT|DE_LU)_.*generation.*(?!forecast)"
cols_to_remove = [col for col in processed_df.columns
                  if (col.startswith(("IT_", "FR_", "AT_", "DE_LU_"))
                      and "generation" in col
                      and "forecast" not in col)]

processed_df = processed_df.drop(columns=cols_to_remove)

print(f"Folgende Spalten wurden entfernt:\n{cols_to_remove}")

if "timestamp" in swissgrid_full.columns:
    swissgrid_full.set_index("timestamp", inplace=True)

# Füge die Swissgrid-Zeitreihe passend zu den Zeitstempeln von processed_df hinzu
processed_df["CH_generation_Swissgrid [MWh]"] = swissgrid_full["CH_generation_Total [MWh]"].reindex(processed_df.index)

# Entferne alle Nullstellen aus der Swissgrid-Zeitreihe
processed_df["CH_generation_Total [MWh]"] = processed_df[
    ["CH_generation_Swissgrid [MWh]", "CH_generation_Solar [MWh]", "CH_generation_Wind_Onshore [MWh]"]
].fillna(0).sum(axis=1)
print("Swissgrid-Daten wurden hinzugefügt")

total_cols = [
    "CH_generation_Swissgrid [MWh]"
]

# Filter für 2023
df_2023 = processed_df[processed_df.index.year == 2023]
sum_2023 = df_2023[total_cols].sum()

# Filter für 2024
df_2024 = processed_df[processed_df.index.year == 2024]
sum_2024 = df_2024[total_cols].sum()

print("Summe der Gesamtproduktion für 2023:")
print(sum_2023)
print("\nSumme der Gesamtproduktion für 2024:")
print(sum_2024)

#--------------------------------------------------------------------------------------------------------------------
# Lücken kleiner als 24 Stunden füllen
#--------------------------------------------------------------------------------------------------------------------

processed_df, Version = ride_report(processed_df, Version)

# Liste der gewünschten Zeitreihen
selected_columns = [
    "CH_generation_forecast [MWh]",
    "FR_wind_and_solar_forecast [MWh]",
    "IT_generation_forecast [MWh]",
    "IT_wind_and_solar_forecast [MWh]",
    "FR_load [MWh]"
]

# Fülle die Lücken aller Zeitreihen, die NICHT in selected_columns sind, mit linearer Interpolation
other_columns = [col for col in processed_df.columns if col not in selected_columns]

# Interpolation anwenden
processed_df[other_columns] = processed_df[other_columns].interpolate(method="linear", limit_direction="both")

print(f"Folgende Zeitreihen wurden per linearer Interpolation aufgefüllt:\n{other_columns}")

# Filtere den DataFrame auf die gewünschten Spalten
plot_df = processed_df[selected_columns]

plt.figure(figsize=(16, 6))
msno.matrix(plot_df, sparkline=False)
plt.title("Visualisierung der grösseren Lücken im zeitlichen Verlauf")
plt.tight_layout()
plt.yticks(fontsize=10)
plt.show()

for col in selected_columns:
    if col not in processed_df.columns:
        print(f"Spalte '{col}' nicht im DataFrame vorhanden!")
        continue

    missing_mask = processed_df[col].isnull()
    gap_start = None
    gap_length = 0
    gaps = []

    for idx, is_missing in zip(processed_df.index, missing_mask):
        if is_missing:
            if gap_start is None:
                gap_start = idx
            gap_length += 1
        else:
            if gap_length > 24:
                gap_end = idx
                gaps.append((gap_start, gap_end, gap_length))
            gap_start = None
            gap_length = 0
    # Falls ein Gap am Ende ist
    if gap_length > 24:
        gap_end = processed_df.index[-1]
        gaps.append((gap_start, gap_end, gap_length))

    if gaps:
        print(f"\nLücken >24h in '{col}':")
        for start, end, length in gaps:
            print(f"  Von {start} bis {end} ({length} Stunden)")
    else:
        print(f"\nKeine Lücken >24h in '{col}'.")

# Entferne alle Nullstellen aus der Swissgrid-Zeitreihe
processed_df = processed_df.drop(columns=["CH_generation_Total [MWh]"])
processed_df["CH_generation_Total [MWh]"] = processed_df[
    ["CH_generation_Swissgrid [MWh]", "CH_generation_Solar [MWh]", "CH_generation_Wind_Onshore [MWh]"]
].fillna(0).sum(axis=1)

processed_df, Version = ride_report(processed_df, Version)

#--------------------------------------------------------------------------------------------------------------------
# Lücken kleiner als 24 Stunden füllen
#--------------------------------------------------------------------------------------------------------------------

cutoff_2025 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
processed_df_cut = processed_df[processed_df.index <= cutoff_2025]

# Wähle die beiden relevanten Zeitreihen aus
compare_cols = [
    "CH_generation_forecast [MWh]",
    "IT_generation_forecast [MWh]",
    "FR_generation_forecast [MWh]",
    "AT_generation_forecast [MWh]",
    "DE_LU_generation_forecast [MWh]",
    "CH_generation_Total [MWh]"
]
 
# Erstelle einen DataFrame mit diesen beiden Spalten
compare_df = processed_df_cut[compare_cols].copy()

# Berechne die Korrelationsmatrix
corr_matrix = compare_df.corr()

# Plot als Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korrelation: Forecast ENTSO-E vs. Swissgrid Daten der Produktion")
plt.tight_layout()
plt.show()

# Entferne alle Spalten, der Spalte "CH_generation_forecast"
pattern = r"^(CH)_.*generation.*(?!forecast)"
cols_to_remove = [col for col in processed_df.columns
                  if (col.startswith(("CH_"))
                      and "generation" in col
                      and "forecast" in col)]

processed_df = processed_df.drop(columns=cols_to_remove)

#--------------------------------------------------------------------------------------------------------------------
# FR Load prüfen
#--------------------------------------------------------------------------------------------------------------------
# Wähle die beiden relevanten Zeitreihen aus
compare_cols = [
    "FR_load [MWh]",
    "IT_load [MWh]",
    "DE_LU_load [MWh]",
    "CH_load [MWh]",
    "AT_load [MWh]",
    "FR_load_forecast [MWh]"
]
 
# Erstelle einen DataFrame mit diesen beiden Spalten
compare_df = processed_df_cut[compare_cols].copy()

# Berechne die Korrelationsmatrix
corr_matrix = compare_df.corr()

# Plot als Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korrelation: FR_load vs. Load anderer Länder")
plt.tight_layout()
plt.show()

# Entferne alle Spalten, der Spalte "FR_Load"
cols_to_remove = [col for col in processed_df.columns
                  if (col.startswith(("FR_"))
                      and "load" in col
                      and "forecast" not in col)]

processed_df = processed_df.drop(columns=cols_to_remove)

processed_df, Version = ride_report(processed_df, Version)
#--------------------------------------------------------------------------------------------------------------------
# Forecast erneuerbare Produktion prüfen
#--------------------------------------------------------------------------------------------------------------------
# Wähle die beiden relevanten Zeitreihen aus
compare_cols = [
    "FR_wind_and_solar_forecast [MWh]",
    "IT_wind_and_solar_forecast [MWh]",
    "DE_LU_wind_and_solar_forecast [MWh]",
    "CH_wind_and_solar_forecast [MWh]",
]
 
# Erstelle einen DataFrame mit diesen beiden Spalten
compare_df = processed_df_cut[compare_cols].copy()

# Berechne die Korrelationsmatrix
corr_matrix = compare_df.corr()

# Plot als Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korrelation: Forecast erneuerbare Produktion ENTSO-E")
plt.tight_layout()
plt.show()

processed_df["FR_wind_and_solar_forecast [MWh]"] = processed_df["FR_wind_and_solar_forecast [MWh]"].fillna(0)
processed_df["IT_wind_and_solar_forecast [MWh]"] = processed_df["IT_wind_and_solar_forecast [MWh]"].fillna(0)
processed_df["IT_generation_forecast [MWh]"] = processed_df["IT_generation_forecast [MWh]"].fillna(0)

processed_df, Version = ride_report(processed_df, Version)

#--------------------------------------------------------------------------------------------------------------------
# Verlauf der Zeitreihen Prüfen
#--------------------------------------------------------------------------------------------------------------------
pattern = r"^(CH|IT|FR|AT|DE_LU)_.*price.*"
cols_price = [col for col in processed_df.columns
                  if (col.startswith(("CH","IT_", "FR_", "AT_", "DE_LU_"))
                      and "price" in col
                  )]

# Erstelle einen DataFrame mit Preis-Zeitreihen
price_df = processed_df[cols_price].copy()

plt.figure(figsize=(18, 8))
for col in price_df.columns:
    plt.plot(price_df.index, price_df[col], label=col, alpha=0.8)
    
plt.title("Zeitlicher Verlauf aller Preis-Zeitreihen")
plt.xlabel("Zeitstempel")
plt.ylabel("Wert (Euro/MWh)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.violinplot([processed_df[col].dropna() for col in cols_price], showmeans=True)
plt.xticks(range(1, len(cols_price) + 1), cols_price, rotation=45, ha='right')
plt.title("Verteilung der Day-Ahead-Preise in den untersuchten Ländern")
plt.ylabel("Wert (EURO/MWh)")
plt.tight_layout()
plt.show()

compare_cols = [
    "FR_price [Euro/MWh]",
    "IT_price [Euro/MWh]",
    "DE_LU_price [Euro/MWh]",
    "CH_price [Euro/MWh]",
    "AT_price [Euro/MWh]"
]
 
# Plot als Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korrelation: Preise in den untersuchten Ländern")
plt.tight_layout()
plt.show()

pattern = r"^(CH|IT|FR|AT|DE_LU)_.*crossborder.*"
cols_crossborder_flows = [col for col in processed_df.columns
                  if ("crossborder" in col
                  )]

# Erstelle einen DataFrame mit crossborder-Zeitreihen
plot_df = processed_df[cols_crossborder_flows].copy()

plt.figure(figsize=(18, 8))
if not plot_df.empty and len(plot_df.columns) > 0:
    for col in plot_df.columns:
        plt.plot(plot_df.index, plot_df[col], label=col, alpha=0.8)
    plt.legend(loc="upper right", fontsize=9)
else:
    print("Keine Zeitreihen zum Plotten vorhanden!")
    
plt.title("Zeitlicher Verlauf aller Hydro Storage-Zeitreihen")
plt.xlabel("Zeitstempel")
plt.ylabel("Wert (MWh)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.violinplot([processed_df[col].dropna() for col in cols_crossborder_flows], showmeans=True)
plt.xticks(range(1, len(cols_crossborder_flows) + 1), cols_crossborder_flows, rotation=45, ha='right')
plt.title("Verteilung der grenzüberschreitenden Lastfülusse im Vergleich")
plt.ylabel("Wert (MWh)")
plt.tight_layout()
plt.show()

pattern = r"^(CH|IT|FR|AT|DE_LU)_.*forecast.*"
cols_forecast = [col for col in processed_df.columns
                  if (col.startswith(("CH","IT_", "FR_", "AT_", "DE_LU_"))
                      and "forecast" in col
                  )]

# Erstelle einen DataFrame mit Forecast-Zeitreihen
plot_df = processed_df[cols_forecast].copy()

plt.figure(figsize=(18, 8))
for col in plot_df.columns:
    plt.plot(plot_df.index, plot_df[col], label=col, alpha=0.8)
    
plt.title("VZeitlicher Verlauf aller Forecast-Zeitreihen")
plt.xlabel("Zeitstempel")
plt.ylabel("Wert (MWh)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.violinplot([processed_df[col].dropna() for col in cols_forecast], showmeans=True)
plt.xticks(range(1, len(cols_forecast) + 1), cols_forecast, rotation=45, ha='right')
plt.title("Verteilung der unterschiedlichen Forecast im Vergleich")
plt.ylabel("Wert (MWh)")
plt.tight_layout()
plt.show()

pattern = r"^(CH|IT|FR|AT|DE_LU)_.*load.*"
cols_load = [col for col in processed_df.columns
                  if (col.startswith(("CH","IT_", "FR_", "AT_", "DE_LU_"))
                      and "load" in col
                      and "forecast" not in col
                  )]

# Erstelle einen DataFrame mit Last-Zeitreihen
plot_df = processed_df[cols_load].copy()

plt.figure(figsize=(18, 8))
for col in plot_df.columns:
    plt.plot(plot_df.index, plot_df[col], label=col, alpha=0.8)
    
plt.title("Zeitlicher Verlauf aller Last-Zeitreihen")
plt.xlabel("Zeitstempel")
plt.ylabel("Wert (MWh)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.violinplot([processed_df[col].dropna() for col in cols_load], showmeans=True)
plt.xticks(range(1, len(cols_load) + 1), cols_load, rotation=45, ha='right')
plt.title("Verteilung der Last in den untersuchten Ländern")
plt.ylabel("Wert (MWh)")
plt.tight_layout()
plt.show()

pattern = r"^(CH|IT|FR|AT|DE_LU)_.*hydro_storage.*"
cols_hydro_storage = [col for col in processed_df.columns
                  if (col.startswith(("CH","IT_", "FR_", "AT_", "DE_LU_"))
                      and "hydro_storage" in col
                  )]

# Erstelle einen DataFrame mit Last-Zeitreihen
plot_df = processed_df[cols_hydro_storage].copy()

plt.figure(figsize=(18, 8))
for col in plot_df.columns:
    plt.plot(plot_df.index, plot_df[col], label=col, alpha=0.8)
    
plt.title("Zeitlicher Verlauf aller Hydro Storage-Zeitreihen")
plt.xlabel("Zeitstempel")
plt.ylabel("Wert (MWh)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.violinplot([processed_df[col].dropna() for col in cols_hydro_storage], showmeans=True)
plt.xticks(range(1, len(cols_hydro_storage) + 1), cols_hydro_storage, rotation=45, ha='right')
plt.title("Verteilung des Wasserspeicher in den untersuchten Ländernnn")
plt.ylabel("Wert (MWh)")
plt.tight_layout()
plt.show()

pattern = r"^(CH|IT|FR|AT|DE_LU)_.*generation.*"
cols_generation = [col for col in processed_df.columns
                  if (col.startswith(("CH","IT_", "FR_", "AT_", "DE_LU_"))
                      and "generation" in col
                      and "forecast" not in col
                  )]

# Erstelle einen DataFrame mit Last-Zeitreihen
plot_df = processed_df[cols_generation].copy()

plt.figure(figsize=(18, 8))
for col in plot_df.columns:
    plt.plot(plot_df.index, plot_df[col], label=col, alpha=0.8)
    
plt.title("Zeitlicher Verlauf aller Produktion-Zeitreihen")
plt.xlabel("Zeitstempel")
plt.ylabel("Wert (MWh)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.violinplot([processed_df[col].dropna() for col in cols_generation], showmeans=True)
plt.xticks(range(1, len(cols_generation) + 1), cols_generation, rotation=45, ha='right')
plt.title("Verteilung des Prouktion in der Schweiz")
plt.ylabel("Wert (MWh)")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------
# Speichern der DataFrames
#--------------------------------------------------------------------------------------------------------------------

# Speichere den unbearbeiteten kombinierten DataFrame
unprocessed_csv_path = "./data/unprocessed_combined_df.csv"
combined_df.to_csv(unprocessed_csv_path)
print(f"Unbearbeiteter DataFrame wurde gespeichert unter: {unprocessed_csv_path}")

# Speichere den bearbeiteten kombinierten DataFrame
processed_csv_path = "./data/processed_combined_df.csv"
processed_df.to_csv(processed_csv_path)
print(f"Unbearbeiteter DataFrame wurde gespeichert unter: {processed_csv_path}")

