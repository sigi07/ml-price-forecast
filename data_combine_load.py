import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_xml_file(filepath):
    """
    Parses an XML file and extracts time series data.

    Args:
        filepath (str): Path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data with columns 'timestamp' and 'value'.
    """
    try:
        # Parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Namespace handling
        namespace = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

        # Extract the start time from <timeInterval>
        time_interval = root.find(".//ns:time_Period.timeInterval", namespace)
        start_time = time_interval.find("ns:start", namespace).text
        start_time = pd.to_datetime(start_time)

        timestamps = []
        values = []

        for time_series in root.findall(".//ns:TimeSeries", namespace):
            for period in time_series.findall(".//ns:Period", namespace):
                resolution = period.find("ns:resolution", namespace).text
                for point in period.findall(".//ns:Point", namespace):
                    position = int(point.find("ns:position", namespace).text)
                    quantity = float(point.find("ns:quantity", namespace).text)

                    # Unterstützt PT15M, PT30M, PT60M
                    if resolution == "PT60M":
                        delta = pd.to_timedelta(position - 1, unit='h')
                    elif resolution == "PT30M":
                        delta = pd.to_timedelta((position - 1) * 30, unit='min')
                    elif resolution == "PT15M":
                        delta = pd.to_timedelta((position - 1) * 15, unit='min')
                    else:
                        print(f"Unbekannte Auflösung '{resolution}' in Datei {filepath}, Punkt {position}")
                        continue

                    timestamp = start_time + delta
                    timestamps.append(timestamp)
                    values.append(quantity)

        if timestamps and values:
            data = pd.DataFrame({"timestamp": timestamps, "value": values})
            data = data.drop_duplicates(subset="timestamp")
            return data
        else:
            print(f"Keine Daten in {filepath} gefunden.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()


def combine_files_by_country_and_type(filepaths_by_country_and_type):
    """
    Combines multiple XML files for each country and type into a single time series DataFrame.

    Args:
        filepaths_by_country_and_type (dict): Dictionary where keys are countries and values are dictionaries
                                              with types (e.g., "load", "generation") as keys and lists of file paths as values.

    Returns:
        dict: A dictionary where keys are countries and values are dictionaries with types as keys and combined DataFrames as values.
    """
    combined_data_by_country_and_type = {}

    for country, filepaths_by_type in filepaths_by_country_and_type.items():
        print(f"Verarbeite Daten für Land: {country}")
        combined_data_by_type = {}

        for data_type, filepaths in filepaths_by_type.items():
            print(f"Kombiniere Dateien für Typ: {data_type}")
            combined_data = pd.DataFrame()

            for filepath in filepaths:
                print(f"Verarbeite {filepath}...")
                data = parse_xml_file(filepath)
                if not data.empty:
                    combined_data = pd.concat([combined_data, data], ignore_index=True)

            # Sort by timestamp and remove duplicates
            if not combined_data.empty:
                combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)
                combined_data = combined_data.drop_duplicates(subset="timestamp")

                # Speichere die kombinierte Tabelle in den gleichen Ordner wie die Rohdaten
                output_directory = os.path.dirname(filepaths[0])  # Verzeichnis der ersten Datei
                output_csv_path = os.path.join(output_directory, f"data_combined_{country}_{data_type}.csv")
                combined_data.to_csv(output_csv_path, index=False)
                print(f"Kombinierte Daten für {country} - Typ '{data_type}' wurden in {output_csv_path} gespeichert.")

            combined_data_by_type[data_type] = combined_data

        combined_data_by_country_and_type[country] = combined_data_by_type

    return combined_data_by_country_and_type

def combine_files_from_FR_and_type(data_folder="./data"):
    """
    Liest alle ENTSO-E Last-XML-Dateien für Frankreich aus dem angegebenen Ordner ein
    und gibt einen kombinierten DataFrame mit Zeitstempel-Index zurück.
    """
    xml_files = sorted(Path(data_folder).glob("load_FR_20*.xml"))

    def parse_entsoe_load_file(filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
        data = []
        for ts in root.findall("ns:TimeSeries", ns):
            for period in ts.findall("ns:Period", ns):
                start = pd.to_datetime(period.find("ns:timeInterval/ns:start", ns).text)
                resolution = period.find("ns:resolution", ns).text
                res_minutes = {"PT60M": 60, "PT15M": 15}.get(resolution, 60)
                for point in period.findall("ns:Point", ns):
                    pos = int(point.find("ns:position", ns).text) - 1
                    quantity = float(point.find("ns:quantity", ns).text)
                    timestamp = start + pd.Timedelta(minutes=pos * res_minutes)
                    data.append((timestamp, quantity))
        return pd.DataFrame(data, columns=["timestamp", "load_MW"])

    all_dfs = [parse_entsoe_load_file(file) for file in xml_files]
    combined_df = pd.concat(all_dfs).drop_duplicates().sort_values("timestamp").set_index("timestamp")

    # Speichere die kombinierte Zeitreihe als CSV
    output_csv_path = os.path.join(data_folder, "data_combined_FR_load.csv")
    combined_df.to_csv(output_csv_path)
    print(f"Kombinierte Frankreich-Lastdaten wurden in {output_csv_path} gespeichert.")

    return combined_df

def combine_all_countries_to_single_file(combined_data_by_country_and_type, output_path, df_fr=None):
    """
    Kombiniert die Daten aller Länder in eine einzige Datei, wobei jedes Land und jeder Typ eine eigene Spalte erhält.
    Optional kann ein zusätzlicher DataFrame für Frankreich (df_fr) mit aufgenommen werden.
    """
    combined_all_countries = pd.DataFrame()

    # Alle Länder und Typen wie bisher
    for country, combined_data_by_type in combined_data_by_country_and_type.items():
        for data_type, data in combined_data_by_type.items():
            if not data.empty:
                column_name = f"{country}_{data_type} [MWh]"
                data = data.set_index("timestamp").rename(columns={"value": column_name})
                if combined_all_countries.empty:
                    combined_all_countries = data
                else:
                    combined_all_countries = combined_all_countries.join(data, how="outer")

    # Frankreich-DataFrame (df_fr) hinzufügen, falls vorhanden
    if df_fr is not None and not df_fr.empty:
        df_fr = df_fr.rename(columns={"load_MW": "FR_load [MW]"})
        if combined_all_countries.empty:
            combined_all_countries = df_fr
        else:
            combined_all_countries = combined_all_countries.join(df_fr, how="outer")

    # Sortiere die Daten nach Zeitstempel
    combined_all_countries = combined_all_countries.sort_index()

    # Speichere die kombinierten Daten in eine CSV-Datei
    combined_all_countries.to_csv(output_path)
    print(f"Alle kombinierten Daten wurden in {output_path} gespeichert.")

def plot_combined_data_by_country_and_type(combined_data_by_country_and_type):
    """
    Plots the combined time series data for each country and type.

    Args:
        combined_data_by_country_and_type (dict): Dictionary where keys are countries and values are dictionaries
                                                  with types as keys and combined DataFrames as values.
    """
    for country, combined_data_by_type in combined_data_by_country_and_type.items():
        for data_type, data in combined_data_by_type.items():
            if not data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(data["timestamp"], data["value"], label=f"{country} - {data_type}", marker='o', markersize=2)
                plt.xlabel("Zeit")
                plt.ylabel("Wert")
                plt.title(f"Kombinierte Zeitreihe: {country} - {data_type}")
                plt.legend()
                plt.grid()
                plt.show()

if __name__ == "__main__":
    # Define file paths for each country and type of data
    filepaths_by_country_and_type = {
        "CH": {
            "load": [
                "./data/load_CH_2020.xml",
                "./data/load_CH_2021.xml",
                "./data/load_CH_2022.xml",
                "./data/load_CH_2023.xml",
                "./data/load_CH_2024.xml",
                "./data/load_CH_2025.xml"
            ],
        },
        "AT": {
            "load": [
                "./data/load_AT_2020.xml",
                "./data/load_AT_2021.xml",
                "./data/load_AT_2022.xml",
                "./data/load_AT_2023.xml",
                "./data/load_AT_2024.xml",
                "./data/load_AT_2025.xml"
            ],

        },
        "DE_LU": {
            "load": [
                "./data/load_DE_LU_2020.xml",
                "./data/load_DE_LU_2021.xml",
                "./data/load_DE_LU_2022.xml",
                "./data/load_DE_LU_2023.xml",
                "./data/load_DE_LU_2024.xml",
                "./data/load_DE_LU_2025.xml"
            ],
        },
        "IT": {
            "load": [
                "./data/load_IT_2020.xml",
                "./data/load_IT_2021.xml",
                "./data/load_IT_2022.xml",
                "./data/load_IT_2023.xml",
                "./data/load_IT_2024.xml",
                "./data/load_IT_2025.xml"
            ],
        },
    }

    # Combine files by country and type
    combined_data_by_country_and_type = combine_files_by_country_and_type(filepaths_by_country_and_type)
    print("Kombinierte Daten für alle Länder und Typen:")
    for country, data_by_type in combined_data_by_country_and_type.items():
        print(f"{country}:")
        for data_type, df in data_by_type.items():
            print(f"  {data_type}: {len(df)} Einträge")
    
    df_fr = combine_files_from_FR_and_type()
    print(f"Geladene Daten für Frankreich: {len(df_fr)} Einträge")
    print(df_fr.head())

    if not df_fr.empty:
        combined_data_by_country_and_type["FR"] = {
            "load": df_fr.reset_index().rename(columns={"timestamp": "timestamp", "load_MW": "value"})
        }

    # Speichere alle Länder in einer einzigen Datei
    output_csv_path = "./data/combined_load_all_countries.csv"
    combine_all_countries_to_single_file(combined_data_by_country_and_type, output_csv_path, df_fr=df_fr)

    # Plot the combined data
    plot_combined_data_by_country_and_type(combined_data_by_country_and_type)
