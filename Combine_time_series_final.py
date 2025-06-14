import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import os

# Paths to the uploaded files

folder_path = "/Users/andrinsiegenthaler/Desktop/Thesis/Code/master-thesis/trading-algorithm"

file_paths = [
    "/data/combined_crossborder_final.csv",
    "/data/combined_aggregate_water_reservoirs_and_hydro_storage_all_countries_2020_2025.csv",
    "/data/combined_forecast_all_countries.csv",
    "/data/combined_load_all_countries.csv",
    "/data/combined_generation_all_countries.csv"

]

# Add the additional file for price data
price_file_path = os.path.join(folder_path, "data_price/combined_day_ahead_prices_all_countries_2020_2025.csv")

# Load each CSV file into a DataFrame
dfs = [pd.read_csv(os.path.join(folder_path, file_path.lstrip("/"))) for file_path in file_paths]

# Load the price data into a separate DataFrame
price_df = pd.read_csv(price_file_path)

# Ensure all DataFrames have a 'timestamp' column and set it as the index
for i, df in enumerate(dfs):
    if "timestamp" not in df.columns:
        raise ValueError(f"DataFrame {i + 1} hat keine 'timestamp'-Spalte.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert to datetime
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='first')]  # Entferne doppelte Zeitstempel
    dfs[i] = df  # Aktualisiere den DataFrame in der Liste

# Process the price DataFrame
if "timestamp" not in price_df.columns:
    raise ValueError("Die Preis-CSV-Datei hat keine 'timestamp'-Spalte.")
price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])  # Convert to datetime
price_df.set_index("timestamp", inplace=True)
price_df = price_df[~price_df.index.duplicated(keep='first')]  # Entferne doppelte Zeitstempel

# Add the price DataFrame to the list of DataFrames
dfs.append(price_df)

# Combine all DataFrames into one, aligning on the timestamp index
combined_df = pd.concat(dfs, axis=1)

# Display the combined DataFrame
print("Kombinierter DataFrame mit aktualisierten Spaltennamen:")
print(combined_df.head())

# Überprüfung des DataFrames
print("\nForm des kombinierten DataFrames:")
print(combined_df.shape)

def process_time_series(dataframe, categorical_columns=None, negative_values_columns=None):
    """
    Überprüft und analysiert Zeitreihen-Daten.

    Args:
        dataframe (pd.DataFrame): Der DataFrame mit den Zeitreihen-Daten.
        categorical_columns (list, optional): Liste der kategorischen Spalten zur Überprüfung auf inkonsistente Werte.
        negative_values_columns (list, optional): Liste der Spalten zur Überprüfung auf negative Werte.

    Returns:
        pd.DataFrame: Der bearbeitete DataFrame.
    """
    if dataframe.empty:
        print("Der übergebene DataFrame ist leer. Verarbeitung wird übersprungen.")
        return dataframe

    # Überprüfung auf fehlende Daten
    print("\nÜberprüfung auf fehlende Daten:")
    missing_data = dataframe.isnull().sum()
    missing_data_percentage = (missing_data / len(dataframe)) * 100
    missing_data_summary = pd.DataFrame({'Missing Count': missing_data, 'Percentage': missing_data_percentage})
    print(missing_data_summary[missing_data_summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

    # Visualisierung der fehlenden Daten mit einer Heatmap
    msno.heatmap(dataframe, figsize=(20, 5))  # Visualisiert die Korrelationen fehlender Werte

    # Überprüfung auf regelmäßige Intervalle der fehlenden Werte
    print("\nÜberprüfung auf die Regelmäßigkeit der fehlenden Werte pro Spalte:")
    for column in dataframe.columns:
        missing_timestamps = dataframe[dataframe[column].isnull()].index  # Zeitstempel der fehlenden Werte
        if len(missing_timestamps) > 1:
            missing_intervals = missing_timestamps.to_series().diff().dropna()  # Berechne die Intervalle zwischen fehlenden Werten
            unique_intervals = missing_intervals.unique()  # Finde alle einzigartigen Intervalle
            print(f"\nSpalte '{column}':")
            print(f"Einzigartige Intervalle der fehlenden Werte: {unique_intervals}")
            if len(unique_intervals) == 1:
                print(f"Die fehlenden Werte treten regelmäßig mit einem Intervall von {unique_intervals[0]} auf.")
            else:
                print("Die fehlenden Werte treten unregelmäßig auf. Unterschiedliche Intervalle:")
                for interval in unique_intervals:
                    print(f"- {interval}: {sum(missing_intervals == interval)} Vorkommen")
        else:
            print(f"\nSpalte '{column}': Keine oder nur ein fehlender Wert, daher keine Analyse der Intervalle möglich.")

    # Überprüfung auf regelmäßige Zeitstempel
    print("\nÜberprüfung auf regelmäßige Zeitstempel:")
    time_differences = dataframe.index.to_series().diff().dropna()  # Berechne die Differenzen zwischen den Zeitstempeln
    unique_differences = time_differences.unique()  # Finde alle einzigartigen Zeitdifferenzen
    print("Einzigartige Zeitdifferenzen:", unique_differences)

    if len(unique_differences) == 1:
        print(f"Die Zeitstempel sind regelmäßig mit einem Intervall von {unique_differences[0]}.")
    else:
        print("Die Zeitstempel sind nicht regelmäßig. Es gibt unterschiedliche Intervalle:")
        for diff in unique_differences:
            print(f"- {diff}: {sum(time_differences == diff)} Vorkommen")

    # Überprüfung auf kategorische Spalten
    if categorical_columns:
        print("\nÜberprüfung auf inkonsistente Kategorien in ausgewählten kategorischen Spalten:")
        for column in categorical_columns:
            print(f"Unique values in '{column}':", dataframe[column].unique())

    # Überprüfung auf negative Werte
    if negative_values_columns:
        negative_values_summary = {column: (dataframe[column] < 0).sum() for column in negative_values_columns}
        print("\nÜberprüfung auf negative Werte in normalerweise positiven Spalten:")
        print(negative_values_summary)

    # Überprüfung auf doppelte Zeilen
    duplicate_rows = dataframe.duplicated().sum()
    print(f"\nAnzahl der vollständigen Duplikate: {duplicate_rows}")

    # Rückgabe des bearbeiteten DataFrames
    return dataframe

# Beispiel für kategorische Spalten (falls vorhanden)
categorical_columns = []  # Fügen Sie hier die Namen der kategorischen Spalten ein
# Überprüfung auf inkonsistente Kategorien in kategorischen Spalten
if categorical_columns:
    print("\nÜberprüfung auf inkonsistente Kategorien in ausgewählten kategorischen Spalten:")
    for column in categorical_columns:
        print(f"Unique values in '{column}':", combined_df[column].unique())

# Überprüfung auf negative Werte in normalerweise positiven Spalten
negative_values_columns = []  # Fügen Sie hier die Namen der Spalten ein, die normalerweise positive Werte haben
# Überprüfung auf negative Werte in den angegebenen Spalten
if negative_values_columns:
    negative_values_summary = {column: (combined_df[column] < 0).sum() for column in negative_values_columns}
    print("\nÜberprüfung auf negative Werte in normalerweise positiven Spalten:")
    print(negative_values_summary)

# Überprüfung auf doppelte Zeilen
duplicate_rows = combined_df.duplicated().sum()
print(f"\nAnzahl der vollständigen Duplikate: {duplicate_rows}")

# Überprüfung auf verbleibende fehlende Werte
print("\nÜberprüfung auf verbleibende fehlende Werte (NaN):")
remaining_missing_data = combined_df.isnull().sum()
remaining_missing_data_summary = pd.DataFrame({
    'Missing Count': remaining_missing_data,
    'Percentage': (remaining_missing_data / len(combined_df)) * 100
})
print(remaining_missing_data_summary[remaining_missing_data_summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

if remaining_missing_data_summary['Missing Count'].sum() == 0:
    print("Es gibt keine fehlenden Werte mehr im DataFrame.")
else:
    print("Es gibt weiterhin fehlende Werte im DataFrame.")

# Display the combined DataFrame
print("Kombinierter DataFrame mit aktualisierten Spaltennamen:")
print(remaining_missing_data.head())

# Liste der Spalten, in denen fehlende Werte mit dem letzten bekannten Wert für 672 Zeitstempel aufgefüllt werden sollen
columns_to_fill = ["CH_hydro_storage [MWh]", "AT_hydro_storage [MWh]", "FR_hydro_storage [MWh]", "IT_hydro_storage [MWh]"]

print("\nFülle fehlende Werte in den folgenden Spalten mit dem letzten bekannten Wert für maximal 672 Zeitstempel auf:")
for column in columns_to_fill:
    if column in combined_df.columns:
        print(f"- {column}")
        
        # Erstelle eine Kopie der Spalte, um die Werte zu bearbeiten
        filled_column = combined_df[column].copy()
        
        # Iteriere über die Zeitstempel und fülle die Werte
        last_value = None
        count = 0
        for idx in filled_column.index:
            if pd.notna(filled_column[idx]):  # Wenn ein Wert vorhanden ist
                last_value = filled_column[idx]
                count = 0  # Zähler zurücksetzen
            elif last_value is not None and count < 672:  # Wenn kein Wert vorhanden ist und innerhalb von 672 Zeitstempeln
                filled_column[idx] = last_value
                count += 1
            else:  # Wenn kein Wert vorhanden ist und die 672 Zeitstempel überschritten wurden
                last_value = None  # Setze den letzten Wert zurück
        
        # Aktualisiere die Spalte im DataFrame
        combined_df[column] = filled_column
    else:
        print(f"- {column} nicht in den Spalten des DataFrames gefunden.")

# Überprüfung auf verbleibende fehlende Werte in den angegebenen Spalten
print("\nÜberprüfung auf verbleibende fehlende Werte in den angegebenen Spalten:")
remaining_missing_data = combined_df[columns_to_fill].isnull().sum()
print(remaining_missing_data)

if remaining_missing_data.sum() == 0:
    print("Es gibt keine fehlenden Werte mehr in den angegebenen Spalten.")
else:
    print("Es gibt weiterhin fehlende Werte in den angegebenen Spalten.")

# Liste der Spalten, die nur stündliche Werte enthalten und auf Viertelstunden erweitert werden sollen
hourly_columns = [
    "CH_Total",
    "CH_generation_Hydro_Pumped_Storage [MWh]",
    "CH_generation_Hydro_Run_of_river_and_poundage [MWh]",
    "CH_generation_Hydro_Water_Reservoir [MWh]",
    "CH_generation_Nuclear [MWh]",
    "CH_generation_Solar [MWh]",
    "CH_generation_Wind_Onshore [MWh]",
    "CH_generation_Total [MWh]",
    "AT_generation_Biomass [MWh]",
    "AT_generation_Fossil_Gas [MWh]",
    "AT_generation_Fossil_Hard_coal [MWh]",
    "AT_generation_Fossil_Oil [MWh]",
    "AT_generation_Geothermal [MWh]",
    "AT_generation_Hydro_Pumped_Storage [MWh]",
    "AT_generation_Hydro_Run_of_river_and_poundage [MWh]",
    "AT_generation_Hydro_Water_Reservoir [MWh]",
    "AT_generation_Other_renewablev [MWh]",
    "AT_generation_Other_renewablev [MWh]",
    "AT_generation_Solar [MWh]",
    "AT_generation_Waste [MWh]",
    "AT_generation_Wind_Onshore [MWh]",
    "AT_generation_Other [MWh]",
    "AT_generation_Total  [MWh]",
    "FR_generation_Biomass [MWh]",
    "FR_generation_Fossil_Gas [MWh]",
    "FR_generation_Fossil_Hard_coal [MWh]",
    "FR_generation_Fossil_Oil [MWh]",
    "FR_generation_Hydro_Pumped_Storage [MWh]",
    "FR_generation_Hydro_Run_of_river_and_poundage [MWh]",
    "FR_generation_Hydro_Water_Reservoir [MWh]",
    "FR_generation_Nuclear [MWh]",
    "FR_generation_Solar [MWh]",
    "FR_generation_Waste [MWh]",
    "FR_generation_Wind_Offshore [MWh]",
    "FR_generation_Wind_Onshore [MWh]",
    "FR_generation_Energy_storage [MWh]",
    "FR_generation_Energy_storage [MWh]",
    "FR_generation_Total [MWh]",
    "DE_LU_generation_Biomass [MWh]",
    "DE_LU_generation_Fossil_Brown_coal_Lignite [MWh]",
    "DE_LU_generation_Fossil_Coal_derived_gas [MWh]",
    "DE_LU_generation_Fossil_Gas [MWh]",
    "DE_LU_generation_Fossil_Hard_coal [MWh]",
    "DE_LU_generation_Fossil_Oil [MWh]",
    "DE_LU_generation_Geothermal [MWh]",
    "DE_LU_generation_Hydro_Pumped_Storage [MWh]",
    "DE_LU_generation_Hydro_Run_of_river_and_poundage [MWh]",
    "DE_LU_generation_Hydro_Water_Reservoir [MWh]",
    "DE_LU_generation_Nuclear [MWh]",
    "DE_LU_generation_Other_renewablev [MWh]",
    "DE_LU_generation_Solar [MWh]",
    "DE_LU_generation_Waste [MWh]",
    "DE_LU_generation_Wind_Offshore [MWh]",
    "DE_LU_generation_Other [MWh]",
    "DE_LU_generation_Total [MWh]",
	"IT_generation_Biomass [MWh]",
	"IT_generation_Fossil_Coal_derived_gas [MWh]",
	"IT_generation_Fossil_Gas [MWh]",
	"IT_generation_Fossil_Hard_coal [MWh]",
	"IT_generation_Fossil_Oil [MWh]",
	"IT_generation_Geothermal [MWh]",
	"IT_generation_Hydro_Pumped_Storage [MWh]",
	"IT_generation_Hydro_Run_of_river_and_poundage [MWh]",
	"IT_generation_Hydro_Water_Reservoir [MWh]",
	"IT_generation_Hydro_Water_Reservoir [MWh]",
	"IT_generation_Solar [MWh]",
	"IT_generation_Waste [MWh]",
	"IT_generation_Wind_Offshore [MWh]",
	"IT_generation_Wind_Onshore [MWh]",
	"IT_generation_Other [MWh]",
	"IT_generation_Total  [MWh]",
    "CH_load [MWh]",
    "IT_load [MWh]",
    "FR_load [MWh]",
    "DE_LU_load [MWh]",
    "AT_load [MWh]",
    "CH_generation_forecast [MWh]",
    "IT_generation_forecast [MWh]",
    "FR_generation_forecast [MWh]",
    "DE_LU_generation_forecast [MWh]",
    "AT_generation_forecast [MWh]",
    "CH_wind_and_solar_forecast [MWh]",
    "IT_wind_and_solar_forecast [MWh]",
    "FR_wind_and_solar_forecast [MWh]",
    "DE_LU_wind_and_solar_forecast [MWh]",
    "AT_wind_and_solar_forecast [MWh]",
    "CH_load_forecast [MWh]",
    "IT_load_forecast [MWh]",
    "FR_load_forecast [MWh]",
    "DE_LU_load_forecast [MWh]",
    "AT_load_forecast [MWh]",
    "CH_intraday_wind_and_solar_forecast [MWh]",
    "IT_intraday_wind_and_solar_forecast [MWh]",
    "FR_intraday_wind_and_solar_forecast [MWh]",
    "DE_LU_intraday_wind_and_solar_forecast [MWh]",
    "AT_intraday_wind_and_solar_forecast [MWh]",
    "CH_price [Euro]",
    "IT_NORD_price [Euro]",
    "FR_price [Euro]",
]

print("\nErweitere stündliche Daten auf Viertelstundenintervalle:")

# Erstelle einen neuen Index mit Viertelstundenintervallen
quarter_hourly_index = pd.date_range(
    start=combined_df.index.min(),
    end=combined_df.index.max(),
    freq="15min"  # Ändern Sie '15T' zu '15min'
)

# Reindizieren des DataFrames auf den neuen Index
combined_df = combined_df.reindex(quarter_hourly_index)

# Fülle die Werte in den angegebenen Spalten auf
for column in hourly_columns:
    if column in combined_df.columns:
        print(f"- {column}")
        combined_df[column] = combined_df[column].fillna(method='ffill')  # Fülle fehlende Werte mit dem letzten bekannten Wert
    else:
        print(f"- {column} nicht in den Spalten des DataFrames gefunden.")

# Überprüfung auf verbleibende fehlende Werte in den angegebenen Spalten
print("\nÜberprüfung auf verbleibende fehlende Werte in den angegebenen Spalten:")

# Filtere nur die Spalten, die tatsächlich im DataFrame vorhanden sind
existing_columns = [column for column in hourly_columns if column in combined_df.columns]

if not existing_columns:
    print("Keine der angegebenen Spalten ist im DataFrame vorhanden.")
else:
    remaining_missing_data = combined_df[existing_columns].isnull().sum()
    print(remaining_missing_data)

    if remaining_missing_data.sum() == 0:
        print("Es gibt keine fehlenden Werte mehr in den angegebenen Spalten.")
    else:
        print("Es gibt weiterhin fehlende Werte in den angegebenen Spalten.")

# Entferne die ersten 196 Zeilen aus dem DataFrame
processed_df = combined_df.iloc[196:]

# Überprüfung des DataFrames nach dem Entfernen der Zeilen
print("DataFrame nach dem Entfernen der ersten 192 Zeilen:")
print(processed_df.head())

processed_df = process_time_series(processed_df, categorical_columns, negative_values_columns)

# Prüfen, wie viele Zeilen komplett nur NaNs enthalten
fully_empty_rows = processed_df[processed_df.isna().all(axis=1)]

# Anzahl dieser Zeilen
print(f"Anzahl der Zeilen mit nur NaN-Werten in allen Spalten: {len(fully_empty_rows)}")

# Optional: Zeitstempel dieser Zeilen anzeigen
print("\nZeitstempel der vollständig leeren Zeilen:")
print(fully_empty_rows.index)

# Spalten, für die Interpolation sinnvoll ist
columns_to_interpolate = [
    "AT_price [Euro]",
    "DE_LU_price [Euro]",
    "CH_hydro_storage [MWh]",
    "AT_hydro_storage [MWh]",
    "FR_hydro_storage [MWh]",
    "IT_hydro_storage [MWh]",
    "FR_intraday_wind_and_solar_forecast [MWh]"
]

# Interpolation anwenden (nur auf die angegebenen Spalten)
processed_df.loc[:, columns_to_interpolate] = processed_df[columns_to_interpolate].interpolate(method='time')

# Überprüfen, wie viele fehlende Werte noch vorhanden sind
missing_data = processed_df.isnull().sum()
missing_data_percentage = (missing_data / len(processed_df)) * 100

# Ausgabe der fehlenden Werte pro Spalte und deren Prozentsatz
print("\nAnzahl der fehlenden Werte pro Spalte und deren Prozentsatz:")
missing_data_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_data_percentage
})
print(missing_data_summary[missing_data_summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

# Definiere die Spaltenpaare erneut
correlation_pairs = [
    ("IT_wind_and_solar_forecast [MWh]", "IT_intraday_wind_and_solar_forecast [MWh]"),
    ("AT_wind_and_solar_forecast [MWh]", "AT_intraday_wind_and_solar_forecast [MWh]"),
    ("FR_wind_and_solar_forecast [MWh]", "FR_intraday_wind_and_solar_forecast [MWh]"),
    ("DE_LU_wind_and_solar_forecast [MWh]", "DE_LU_intraday_wind_and_solar_forecast [MWh]")
]

# Berechne die Korrelationen erneut
correlations = {}
for col1, col2 in correlation_pairs:
    if col1 in processed_df.columns and col2 in processed_df.columns:
        correlation = processed_df[[col1, col2]].dropna().corr().iloc[0, 1]
        correlations[f"{col1} & {col2}"] = correlation
    else:
        correlations[f"{col1} & {col2}"] = "Eine oder beide Spalten fehlen"

correlations
# Ausgabe der Korrelationen
print("\nKorrelationen zwischen den Spaltenpaaren:")
for pair, correlation in correlations.items():
    if isinstance(correlation, str):
        print(f"{pair}: {correlation}")
    else:
        print(f"{pair}: {correlation:.2f}")

# Fülle nur NaNs in 'AT_wind_and_solar_forecast [MWh]' mit den Werten aus 'AT_intraday_wind_and_solar_forecast [MWh]'
processed_df["AT_wind_and_solar_forecast [MWh]"] = processed_df["AT_wind_and_solar_forecast [MWh]"].combine_first(
    processed_df["AT_intraday_wind_and_solar_forecast [MWh]"]
)
# Fülle nur NaNs in 'AT_wind_and_solar_forecast [MWh]' mit den Werten aus 'AT_intraday_wind_and_solar_forecast [MWh]'
processed_df["IT_intraday_wind_and_solar_forecast [MWh]"] = processed_df["IT_intraday_wind_and_solar_forecast [MWh]"].combine_first(
    processed_df["IT_wind_and_solar_forecast [MWh]"]
)

# Überprüfen, wie viele fehlende Werte noch vorhanden sind
missing_data = processed_df.isnull().sum()
missing_data_percentage = (missing_data / len(processed_df)) * 100

# Ausgabe der fehlenden Werte pro Spalte und deren Prozentsatz
print("\nAnzahl der fehlenden Werte pro Spalte und deren Prozentsatz:")
missing_data_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_data_percentage
})
print(missing_data_summary[missing_data_summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

# Zeige alle vollständig duplizierten Zeilen
duplicate_rows = processed_df[processed_df.duplicated()]
print(f"Anzahl Duplikate: {len(duplicate_rows)}")

duplicate_timestamps = processed_df.index[processed_df.index.duplicated()]
# Zeige alle vollständigen Duplikate mit unterschiedlichen Zeitstempeln
print(f"{len(duplicate_rows)} von {len(processed_df)} Zeilen sind Duplikate ({len(duplicate_rows)/len(processed_df)*100:.2f} %)")

# Alle Duplikate finden (ohne Zeitstempelvergleich, da dieser eindeutig ist)
value_duplicates = processed_df.duplicated()

# Nur die Duplikate extrahieren
duplicate_rows = processed_df[value_duplicates]

# Verteilung der Duplikate nach zeitlichem Abstand
duplicate_distribution = duplicate_rows.index.to_series().diff().value_counts().sort_index()

# Vorbereiten der Ausgabe
duplicate_info = {
    "Anzahl Duplikate": value_duplicates.sum(),
    "Anteil Duplikate (%)": round((value_duplicates.sum() / len(df)) * 100, 2),
    "Verteilung Zeitabstände (erste 10)": duplicate_distribution.head(10)
}

# Schritt 1: Finde die Indizes der Duplikate
duplicate_indices = processed_df[processed_df.duplicated()].index

# Schritt 2: Prüfe, ob diese Indizes direkt aufeinander folgen
are_duplicates_consecutive = all((b - a == 1) for a, b in zip(duplicate_indices[:-1], duplicate_indices[1:]))

# Ausgabe
if are_duplicates_consecutive:
    print("Alle Duplikate liegen direkt hintereinander.")
else:
    print("Die Duplikate sind nicht vollständig aufeinanderfolgend.")

# Suche nach vollständig doppelten Zeilen
duplicate_rows = df[df.duplicated(keep=False)]

# Extrahiere die Zeitstempel dieser doppelten Zeilen
duplicate_timestamps = duplicate_rows.index
print(f"Zeitstempel der doppelten Zeilen: {duplicate_timestamps}")

#======================================================================================================================

# Liste der Jahresdateien (passe ggf. die Jahre und Pfade an)
years = range(2020, 2026)
dfs_swissgrid = []

for year in years:
    csv_path = os.path.join(folder_path, "data", f"EnergieUebersichtCH-{year}_SG.csv")
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

print("processed_df generation CH in MWh:", processed_df["CH_generation_Total [MWh]"].head())
print("swissgrid_full generation CH in MWh:", swissgrid_full.head())
print("Größe von processed_df:", processed_df["CH_generation_Total [MWh]"].shape)
print("Größe von swissgrid_full:", swissgrid_full["CH_generation_Total [MWh]"].shape)
print("Letzte Zeitstempel processed_df:", processed_df["CH_generation_Total [MWh]"].tail(5))
print("Letzte Zeitstempel swissgrid_full:", swissgrid_full["CH_generation_Total [MWh]"].tail(5))

# Entferne alle Zeilen nach dem 01.01.2025
cutoff = pd.Timestamp("2025-01-01 00:00:00+00:00   ", tz=processed_df.index.tz)
processed_df_cut = processed_df[processed_df.index < cutoff]
swissgrid_full_cut = swissgrid_full[swissgrid_full.index < cutoff]

print("Größe von processed_df:", processed_df_cut["CH_generation_Total [MWh]"].shape)
print("Größe von swissgrid_full:", swissgrid_full_cut["CH_generation_Total [MWh]"].shape)
print("Letzte Zeitstempel processed_df:", processed_df_cut["CH_generation_Total [MWh]"].tail(5))
print("Letzte Zeitstempel swissgrid_full:", swissgrid_full_cut["CH_generation_Total [MWh]"].tail(5))

# Setze Zeitzone der Swissgrid-Daten explizit auf UTC
if swissgrid_full_cut.index.tz is None:
    swissgrid_full_cut.index = swissgrid_full_cut.index.tz_localize("UTC")
else:
    swissgrid_full_cut.index = swissgrid_full_cut.index.tz_convert("UTC")

# Korrelation zwischen processed_df und swissgrid_full für CH_generation_Total [MWh]
if "CH_generation_Total [MWh]" in processed_df_cut.columns and "CH_generation_Total [MWh]" in swissgrid_full_cut.columns:
    # Gemeinsame Zeitstempel finden
    common_idx = processed_df_cut.index.intersection(swissgrid_full_cut.index)
    # Werte extrahieren und auf gemeinsame Zeitstempel beschränken
    series1 = processed_df_cut.loc[common_idx, "CH_generation_Total [MWh]"]
    series2 = swissgrid_full_cut.loc[common_idx, "CH_generation_Total [MWh]"]
    # Zu DataFrame zusammenfassen und nur Zeilen ohne NaN vergleichen
    df_corr = pd.DataFrame({
        "processed": series1,
        "swissgrid": series2
    }).dropna()
    corr = df_corr["processed"].corr(df_corr["swissgrid"])
    print(f"\nKorrelation zwischen processed_df und swissgrid_full_cut für 'CH_generation_Total [MWh]': {corr:.4f}")
else:
    print("Spalte 'CH_generation_Total [MWh]' fehlt in einem der DataFrames.")


plt.figure(figsize=(8, 6))
plt.scatter(df_corr["processed"], df_corr["swissgrid"], alpha=0.3)
plt.xlabel("processed_df CH_generation_Total [MWh]")
plt.ylabel("swissgrid_full CH_generation_Total [MWh]")
plt.title(f"Scatterplot (Korrelation: {corr:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()

window = 96*7  # z.B. eine Woche bei 15min-Daten
rolling_corr = df_corr["processed"].rolling(window).corr(df_corr["swissgrid"])

plt.figure(figsize=(14, 4))
plt.plot(df_corr.index, rolling_corr)
plt.title("Rollende Korrelation (Fenster: 1 Woche)")
plt.xlabel("Zeit")
plt.ylabel("Korrelation")
plt.grid(True)
plt.tight_layout()
plt.show()

# Zeitreihen normalisieren (z-Standardisierung)
df_corr["processed_norm"] = (df_corr["processed"] - df_corr["processed"].mean()) / df_corr["processed"].std()
df_corr["swissgrid_norm"] = (df_corr["swissgrid"] - df_corr["swissgrid"].mean()) / df_corr["swissgrid"].std()

plt.figure(figsize=(14, 5))
plt.plot(df_corr.index, df_corr["processed_norm"], label="processed_df (normalisiert)")
plt.plot(df_corr.index, df_corr["swissgrid_norm"], label="swissgrid_full (normalisiert)", alpha=0.7)
plt.title("Normalisierte Zeitreihen: CH_generation_Total [MWh]")
plt.xlabel("Zeit")
plt.ylabel("z-standardisierter Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

processed_df = processed_df.drop(columns=["CH_generation_Total [MWh]"])
print("Spalte 'CH_generation_Total [MWh]' wurde entfernt.")

# Stelle sicher, dass beide DataFrames eindeutige Indizes haben
processed_df = processed_df[~processed_df.index.duplicated(keep='first')]
swissgrid_full = swissgrid_full[~swissgrid_full.index.duplicated(keep='first')]

# Füge die Zeitreihe als neue Spalte hinzu (Index-Abgleich erfolgt automatisch)
processed_df["CH_generation_Total [MWh]"] = swissgrid_full["CH_generation_Total [MWh]"]

print("Spalte 'CH_generation_Total [MWh]' aus swissgrid_full wurde als neue Spalte in processed_df eingefügt.")

print("\n=== Ausreißer-Analyse für alle numerischen Zeitreihen (IQR-Methode) ===")
for col in processed_df.select_dtypes(include=[np.number]).columns:
    data = processed_df[col].dropna()
    if data.empty:
        print(f"\nSpalte '{col}': Keine Daten vorhanden.")
        continue

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    ausreisser = data[(data < lower_bound) | (data > upper_bound)]
    print(f"\nSpalte '{col}':")
    print(f"  Anzahl Ausreißer: {len(ausreisser)}")
    if len(ausreisser) > 0:
        print(f"  Zeitstempel der Ausreißer (max. 10): {ausreisser.index[:10].tolist()}")

    # Optional: Visualisierung (nur für Spalten mit Ausreißern)
    if len(ausreisser) > 0:
        plt.figure(figsize=(12, 3))
        plt.plot(data.index, data.values, label=col)
        plt.scatter(ausreisser.index, ausreisser.values, color='red', label='Ausreißer')
        plt.title(f"Ausreißer in {col}")
        plt.xlabel("Zeit")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.show()

# --- Manuelle Ausreisserbehandlung für definierte Schwellwerte ---

def replace_outlier_with_neighbors_mean(series, threshold):
    s = series.copy()
    outlier_idx = s[s > threshold].index
    for idx in outlier_idx:
        pos = s.index.get_loc(idx)
        # Nur ersetzen, wenn Nachbarn existieren
        if pos > 0 and pos < len(s) - 1:
            prev_val = s.iloc[pos - 1]
            next_val = s.iloc[pos + 1]
            # Nur ersetzen, wenn beide Nachbarn keine NaN sind
            if not pd.isna(prev_val) and not pd.isna(next_val):
                s.iloc[pos] = (prev_val + next_val) / 2
    return s

# AT_wind_and_solar_forecast [MWh] > 6000
col_at = "AT_wind_and_solar_forecast [MWh]"
if col_at in processed_df.columns:
    processed_df[col_at] = replace_outlier_with_neighbors_mean(processed_df[col_at], 6000)

# CH_load [MWh] > 15000
col_ch = "CH_load [MWh]"
if col_ch in processed_df.columns:
    processed_df[col_ch] = replace_outlier_with_neighbors_mean(processed_df[col_ch], 15000)

# Entferne alle Zeilen nach dem 09.04.2025 (einschließlich)
cutoff = pd.Timestamp("2025-04-09 00:00:00", tz=processed_df.index.tz)
processed_df = processed_df[processed_df.index < cutoff]

# Speichere den unbearbeiteten kombinierten DataFrame
unprocessed_csv_path = "/data/unprocessed_combined_df.csv"
combined_df.to_csv(unprocessed_csv_path)
print(f"Unbearbeiteter DataFrame wurde gespeichert unter: {unprocessed_csv_path}")

# Speichere den bearbeiteten kombinierten DataFrame
processed_csv_path = "/data/processed_combined_df.csv"
processed_df.to_csv(processed_csv_path)
print(f"Bearbeiteter DataFrame wurde gespeichert unter: {processed_csv_path}")

