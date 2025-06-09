import os
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET

def parse_file(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        timestamps = []
        values = []

        for time_series in root.findall(".//{*}TimeSeries"):
            for period in time_series.findall(".//{*}Period"):
                time_interval = period.find(".//{*}timeInterval")
                start_time = pd.to_datetime(time_interval.find(".//{*}start").text)
                resolution = period.find(".//{*}resolution").text

                for point in period.findall(".//{*}Point"):
                    position = int(point.find(".//{*}position").text)
                    quantity = float(point.find(".//{*}quantity").text)

                    if resolution == "PT60M":
                        delta = pd.to_timedelta(position - 1, unit="h")
                    elif resolution == "PT30M":
                        delta = pd.to_timedelta((position - 1) * 30, unit="min")
                    elif resolution == "PT15M":
                        delta = pd.to_timedelta((position - 1) * 15, unit="min")
                    else:
                        print(f"Unbekannte Aufl\u00f6sung '{resolution}' in {filepath}, Punkt {position}")
                        continue

                    timestamp = start_time + delta
                    timestamps.append(timestamp)
                    values.append(quantity)

        data = pd.DataFrame({"timestamp": timestamps, "value": values})
        return data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()

def combine_forecast_files(filepaths):
    combined_data = pd.DataFrame()

    for filepath in filepaths:
        print(f"Verarbeite {filepath}...")
        data = parse_file(filepath)
        if not data.empty:
            combined_data = pd.concat([combined_data, data], ignore_index=True)

    if not combined_data.empty:
        combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)
        combined_data = combined_data.drop_duplicates(subset="timestamp")
        full_time_index = pd.date_range(start=combined_data["timestamp"].min(),
                                        end=combined_data["timestamp"].max(),
                                        freq="h")
        combined_data = combined_data.set_index("timestamp").reindex(full_time_index).reset_index()
        combined_data.columns = ["timestamp", "value"]

    return combined_data

def plot_time_series(data, title="Zeitreihe"):
    if data.empty:
        print("Keine Daten zum Plotten verf\u00fcgbar.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data["timestamp"], data["value"], label="Werte", marker='o', markersize=2, color="blue")
    plt.xlabel("Zeit")
    plt.ylabel("Wert")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def combine_all_countries_to_single_file(forecast_filepaths_by_country_and_type, output_path):
    """
    Kombiniert die Daten aller Länder in eine einzige Datei, wobei jedes Land und jeder Typ eine eigene Spalte erhält.

    Args:
        forecast_filepaths_by_country_and_type (dict): Dictionary mit Ländern als Schlüssel und Dateitypen als Unter-Schlüssel.
        output_path (str): Pfad zur Ausgabedatei.

    Returns:
        None
    """
    combined_all_countries = pd.DataFrame()

    for country, filepaths_by_type in forecast_filepaths_by_country_and_type.items():
        for forecast_type, filepaths in filepaths_by_type.items():
            print(f"Kombiniere Dateien für Land: {country}, Typ: {forecast_type}")
            combined_data = combine_forecast_files(filepaths)

            if not combined_data.empty:
                # Benenne die Spalte nach Land, Typ und füge die Einheit [MWh] hinzu
                column_name = f"{country}_{forecast_type} [MWh]"
                combined_data = combined_data.set_index("timestamp").rename(columns={"value": column_name})

                # Kombiniere die Daten mit dem Haupt-DataFrame
                if combined_all_countries.empty:
                    combined_all_countries = combined_data
                else:
                    combined_all_countries = combined_all_countries.join(combined_data, how="outer")

    # Sortiere die Daten nach Zeitstempel
    combined_all_countries = combined_all_countries.sort_index()

    # Speichere die kombinierten Daten in eine CSV-Datei
    combined_all_countries.to_csv(output_path)
    print(f"Alle kombinierten Daten wurden in {output_path} gespeichert.")

if __name__ == "__main__":
    forecast_filepaths_by_country_and_type = {
        "CH": {
            "generation_forecast": [
                "./data/generation_forecast_CH_2020.xml",
                "./data/generation_forecast_CH_2021.xml",
                "./data/generation_forecast_CH_2022.xml",
                "./data/generation_forecast_CH_2023.xml",
                "./data/generation_forecast_CH_2024.xml",
                "./data/generation_forecast_CH_2025.xml"
            ],
            "wind_and_solar_forecast": [
                "./data/wind_and_solar_forecast_CH_2020.xml",
                "./data/wind_and_solar_forecast_CH_2021.xml",
                "./data/wind_and_solar_forecast_CH_2022.xml",
                "./data/wind_and_solar_forecast_CH_2023.xml",
                "./data/wind_and_solar_forecast_CH_2024.xml",
                "./data/wind_and_solar_forecast_CH_2025.xml"
            ],
            "intraday_wind_and_solar_forecast": [
                "./data/intraday_wind_and_solar_forecast_CH_2020.xml",
                "./data/intraday_wind_and_solar_forecast_CH_2021.xml",
                "./data/intraday_wind_and_solar_forecast_CH_2022.xml",
                "./data/intraday_wind_and_solar_forecast_CH_2023.xml",
                "./data/intraday_wind_and_solar_forecast_CH_2024.xml",
                "./data/intraday_wind_and_solar_forecast_CH_2025.xml"
            ],
            "load_forecast": [
                "./data/load_forecast_CH_2020.xml",
                "./data/load_forecast_CH_2021.xml",
                "./data/load_forecast_CH_2022.xml",
                "./data/load_forecast_CH_2023.xml",
                "./data/load_forecast_CH_2024.xml",
                "./data/load_forecast_CH_2025.xml"
            ]
        },
        "AT": {
            "generation_forecast": [
                "./data/generation_forecast_AT_2020.xml",
                "./data/generation_forecast_AT_2021.xml",
                "./data/generation_forecast_AT_2022.xml",
                "./data/generation_forecast_AT_2023.xml",
                "./data/generation_forecast_AT_2024.xml",
                "./data/generation_forecast_AT_2025.xml"
            ],
            "wind_and_solar_forecast": [
                "./data/wind_and_solar_forecast_AT_2020.xml",
                "./data/wind_and_solar_forecast_AT_2021.xml",
                "./data/wind_and_solar_forecast_AT_2022.xml",
                "./data/wind_and_solar_forecast_AT_2023.xml",
                "./data/wind_and_solar_forecast_AT_2024.xml",
                "./data/wind_and_solar_forecast_AT_2025.xml"
            ],
            "intraday_wind_and_solar_forecast": [
                "./data/intraday_wind_and_solar_forecast_AT_2020.xml",
                "./data/intraday_wind_and_solar_forecast_AT_2021.xml",
                "./data/intraday_wind_and_solar_forecast_AT_2022.xml",
                "./data/intraday_wind_and_solar_forecast_AT_2023.xml",
                "./data/intraday_wind_and_solar_forecast_AT_2024.xml",
                "./data/intraday_wind_and_solar_forecast_AT_2025.xml"
            ],
            "load_forecast": [
                "./data/load_forecast_AT_2020.xml",
                "./data/load_forecast_AT_2021.xml",
                "./data/load_forecast_AT_2022.xml",
                "./data/load_forecast_AT_2023.xml",
                "./data/load_forecast_AT_2024.xml",
                "./data/load_forecast_AT_2025.xml"
            ]
                
        },
        "DE_LU": {
            "generation_forecast": [
                "./data/generation_forecast_DE_LU_2020.xml",
                "./data/generation_forecast_DE_LU_2021.xml",
                "./data/generation_forecast_DE_LU_2022.xml",
                "./data/generation_forecast_DE_LU_2023.xml",
                "./data/generation_forecast_DE_LU_2024.xml",
                "./data/generation_forecast_DE_LU_2025.xml"
            ],
            "wind_and_solar_forecast": [
                "./data/wind_and_solar_forecast_DE_LU_2020.xml",
                "./data/wind_and_solar_forecast_DE_LU_2021.xml",
                "./data/wind_and_solar_forecast_DE_LU_2022.xml",
                "./data/wind_and_solar_forecast_DE_LU_2023.xml",
                "./data/wind_and_solar_forecast_DE_LU_2024.xml",
                "./data/wind_and_solar_forecast_DE_LU_2025.xml"
            ],
            "intraday_wind_and_solar_forecast": [
                "./data/intraday_wind_and_solar_forecast_DE_LU_2020.xml",
                "./data/intraday_wind_and_solar_forecast_DE_LU_2021.xml",
                "./data/intraday_wind_and_solar_forecast_DE_LU_2022.xml",
                "./data/intraday_wind_and_solar_forecast_DE_LU_2023.xml",
                "./data/intraday_wind_and_solar_forecast_DE_LU_2024.xml",
                "./data/intraday_wind_and_solar_forecast_DE_LU_2025.xml"
            ],
            "load_forecast": [
                "./data/load_forecast_DE_LU_2020.xml",
                "./data/load_forecast_DE_LU_2021.xml",
                "./data/load_forecast_DE_LU_2022.xml",
                "./data/load_forecast_DE_LU_2023.xml",
                "./data/load_forecast_DE_LU_2024.xml",
                "./data/load_forecast_DE_LU_2025.xml"
            ]
        },
        "FR": {
            "generation_forecast": [
                "./data/generation_forecast_FR_2020.xml",
                "./data/generation_forecast_FR_2021.xml",
                "./data/generation_forecast_FR_2022.xml",
                "./data/generation_forecast_FR_2023.xml",
                "./data/generation_forecast_FR_2024.xml",
                "./data/generation_forecast_FR_2025.xml"
            ],
            "wind_and_solar_forecast": [
                "./data/wind_and_solar_forecast_FR_2020.xml",
                "./data/wind_and_solar_forecast_FR_2021.xml",
                "./data/wind_and_solar_forecast_FR_2022.xml",
                "./data/wind_and_solar_forecast_FR_2023.xml",
                "./data/wind_and_solar_forecast_FR_2024.xml",
                "./data/wind_and_solar_forecast_FR_2025.xml"
            ],
            "intraday_wind_and_solar_forecast": [
                "./data/intraday_wind_and_solar_forecast_FR_2020.xml",
                "./data/intraday_wind_and_solar_forecast_FR_2021.xml",
                "./data/intraday_wind_and_solar_forecast_FR_2022.xml",
                "./data/intraday_wind_and_solar_forecast_FR_2023.xml",
                "./data/intraday_wind_and_solar_forecast_FR_2024.xml",
                "./data/intraday_wind_and_solar_forecast_FR_2025.xml"
            ],
            "load_forecast": [
                "./data/load_forecast_FR_2020.xml",
                "./data/load_forecast_FR_2021.xml",
                "./data/load_forecast_FR_2022.xml",
                "./data/load_forecast_FR_2023.xml",
                "./data/load_forecast_FR_2024.xml",
                "./data/load_forecast_FR_2025.xml"
            ]
        },
        "IT": {
            "generation_forecast": [
                "./data/generation_forecast_IT_2020.xml",
                "./data/generation_forecast_IT_2021.xml",
                "./data/generation_forecast_IT_2022.xml",
                "./data/generation_forecast_IT_2023.xml",
                "./data/generation_forecast_IT_2024.xml",
                "./data/generation_forecast_IT_2025.xml"
            ],
            "wind_and_solar_forecast": [
                "./data/wind_and_solar_forecast_IT_2020.xml",
                "./data/wind_and_solar_forecast_IT_2021.xml",
                "./data/wind_and_solar_forecast_IT_2022.xml",
                "./data/wind_and_solar_forecast_IT_2023.xml",
                "./data/wind_and_solar_forecast_IT_2024.xml",
                "./data/wind_and_solar_forecast_IT_2025.xml"
            ],
            "intraday_wind_and_solar_forecast": [
                "./data/intraday_wind_and_solar_forecast_IT_2020.xml",
                "./data/intraday_wind_and_solar_forecast_IT_2021.xml",
                "./data/intraday_wind_and_solar_forecast_IT_2022.xml",
                "./data/intraday_wind_and_solar_forecast_IT_2023.xml",
                "./data/intraday_wind_and_solar_forecast_IT_2024.xml",
                "./data/intraday_wind_and_solar_forecast_IT_2025.xml"
            ],
            "load_forecast": [
                "./data/load_forecast_IT_2020.xml",
                "./data/load_forecast_IT_2021.xml",
                "./data/load_forecast_IT_2022.xml",
                "./data/load_forecast_IT_2023.xml",
                "./data/load_forecast_IT_2024.xml",
                "./data/load_forecast_IT_2025.xml"
            ]
        }
    }

    for country, filepaths_by_type in forecast_filepaths_by_country_and_type.items():
        for forecast_type, filepaths in filepaths_by_type.items():
            print(f"Kombiniere Dateien f\u00fcr Land: {country}, Typ: {forecast_type}")
            combined_data = combine_forecast_files(filepaths)

            if not combined_data.empty:
                print(f"Daten f\u00fcr Land '{country}', Typ '{forecast_type}' erfolgreich kombiniert:")
                print(combined_data)
                output_directory = os.path.dirname(filepaths[0])
                output_csv_path = os.path.join(output_directory, f"data_combined_{country}_{forecast_type}.csv")
                combined_data.to_csv(output_csv_path, index=False)
                print(f"Kombinierte Daten wurden in {output_csv_path} gespeichert.")
                plot_time_series(combined_data, title=f"Kombinierte Zeitreihe: {country} - {forecast_type}")
            else:
                print(f"Keine Daten f\u00fcr Land '{country}', Typ '{forecast_type}' geladen.")

    # Kombiniere alle Länder in eine einzige Datei
    output_csv_path = "./data/combined_forecast_all_countries.csv"
    combine_all_countries_to_single_file(forecast_filepaths_by_country_and_type, output_csv_path)


file_path = "./data/combined_forecast_all_countries.csv"
# CSV einlesen und Zeitstempel parsen
df = pd.read_csv(file_path, parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Duplikate aggregieren (Mittelwert je Zeitstempel)
df = df.groupby(df.index).mean()

# Auf Viertelstundenauflösung interpolieren
df_15min = df.resample("15min").interpolate(method="linear")

# Vorschau
print(df_15min.head())
print(f"Größe nach Resampling: {df_15min.shape} (Zeilen, Spalten)")

#Speichere die interpolierten Daten in eine CSV-Datei
df_15min.to_csv(output_csv_path.replace(".csv", "_15min.csv"), index=True)