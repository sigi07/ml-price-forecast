import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def parse_generation_load_document(filepath):
    """
    Parses a GL_MarketDocument XML file and extracts time series data.

    Args:
        filepath (str): Path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data with columns 'timestamp', 'psrType', 'position', and 'quantity'.
    """
    try:
        # Parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Namespace handling
        namespace = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

        data = []

        # Iterate over each TimeSeries element
        for time_series in root.findall("ns:TimeSeries", namespace):
            psr_type = time_series.find("ns:MktPSRType/ns:psrType", namespace).text

            for period in time_series.findall("ns:Period", namespace):
                start_time = period.find("ns:timeInterval/ns:start", namespace).text
                resolution = period.find("ns:resolution", namespace).text

                for point in period.findall("ns:Point", namespace):
                    position = int(point.find("ns:position", namespace).text)
                    quantity = float(point.find("ns:quantity", namespace).text)

                    # Calculate the timestamp based on the start time and position
                    if resolution == "P7D":  # Weekly resolution
                        timestamp = pd.to_datetime(start_time) + pd.to_timedelta(position - 1, unit='W')
                    elif resolution == "P1Y":  # Yearly resolution
                        timestamp = pd.to_datetime(start_time) + pd.DateOffset(years=position - 1)
                    elif resolution == "PT60M":  # Hourly resolution
                        timestamp = pd.to_datetime(start_time) + pd.to_timedelta(position - 1, unit='h')
                    else:
                        raise ValueError(f"Unsupported resolution: {resolution}")

                    data.append({"timestamp": timestamp, "psrType": psr_type, "quantity": quantity})

        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()

def parse_installed_generation_capacity_file(filepath):
    """
    Parses the installed_generation_capacity_CH_2020.xml file and extracts time series data.

    Args:
        filepath (str): Path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data with columns 'timestamp', 'value', and 'psrType'.
    """
    try:
        # Parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()

        timestamps = []
        values = []
        psr_types = []

        # Extract data from <TimeSeries>
        for time_series in root.findall(".//{*}TimeSeries"):
            psr_type = time_series.find(".//{*}MktPSRType/{*}psrType").text  # Extract psrType
            for period in time_series.findall(".//{*}Period"):
                # Extract the start time and resolution
                time_interval = period.find(".//{*}timeInterval")
                start_time = time_interval.find(".//{*}start").text
                start_time = pd.to_datetime(start_time)  # Convert to datetime
                resolution = period.find(".//{*}resolution").text

                # Extract data from <Point>
                for point in period.findall(".//{*}Point"):
                    position = int(point.find(".//{*}position").text)  # Offset from start_time
                    quantity = float(point.find(".//{*}quantity").text)  # Value for the position

                    # Calculate the timestamp for the current position
                    if resolution == "P1Y":  # Yearly resolution
                        timestamp = start_time + pd.DateOffset(years=position - 1)
                    elif resolution == "PT60M":  # Hourly resolution
                        timestamp = start_time + pd.to_timedelta(position - 1, unit='h')
                    else:
                        raise ValueError(f"Unsupported resolution: {resolution}")

                    timestamps.append(timestamp)
                    values.append(quantity)
                    psr_types.append(psr_type)

        # Convert to DataFrame
        data = pd.DataFrame({"timestamp": timestamps, "value": values, "psrType": psr_types})
        return data

    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()

def parse_aggregate_water_reservoirs_file(filepath):
    """
    Parses an aggregate_water_reservoirs_and_hydro_storage_CH XML file and extracts time series data.

    Args:
        filepath (str): Path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data with columns 'timestamp', 'Storage (MWh)', and 'Week'.
    """
    try:
        # Parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Namespace handling
        namespace = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

        data = []

        # Iterate over each TimeSeries element
        for time_series in root.findall("ns:TimeSeries", namespace):
            week = time_series.find("ns:mRID", namespace).text  # Extract mRID (renamed to Week)

            for period in time_series.findall("ns:Period", namespace):
                start_time = period.find("ns:timeInterval/ns:start", namespace).text
                resolution = period.find("ns:resolution", namespace).text

                for point in period.findall("ns:Point", namespace):
                    position = int(point.find("ns:position", namespace).text)
                    storage = float(point.find("ns:quantity", namespace).text)  # Renamed to Storage (MWh)

                    # Calculate the timestamp based on the start time and position
                    if resolution == "P7D":  # Weekly resolution
                        timestamp = pd.to_datetime(start_time) + pd.to_timedelta(position - 1, unit='W')
                    else:
                        raise ValueError(f"Unsupported resolution: {resolution}")

                    data.append({"timestamp": timestamp, "Storage (MWh)": storage, "Week": week})

        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()

def combine_files(filepaths):
    """
    Combines multiple XML files into a single DataFrame.

    Args:
        filepaths (list): List of file paths to the XML files.

    Returns:
        pd.DataFrame: A combined DataFrame with a continuous time series.
    """
    combined_data = pd.DataFrame()

    for filepath in filepaths:
        print(f"Verarbeite Datei: {filepath}")
        data = parse_aggregate_water_reservoirs_file(filepath)
        if not data.empty:
            combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Sort the combined data by timestamp
    combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)
    return combined_data

def combine_all_countries_to_single_file(countries, output_path):
    """
    Kombiniert die Daten aller Länder in eine einzige Datei.

    Args:
        countries (dict): Dictionary mit Ländern als Schlüssel und Dateipfaden als Werte.
        output_path (str): Pfad zur Ausgabedatei.

    Returns:
        None
    """
    combined_all_countries = pd.DataFrame()

    for country, filepaths in countries.items():
        print(f"Kombiniere Dateien für Land: {country}")
        combined_data = combine_files(filepaths)

        if not combined_data.empty:
            # Entferne die Spalte 'Week', da sie redundant ist
            combined_data = combined_data.drop(columns=["Week"], errors="ignore")

            # Benenne die Spalte nach dem Land mit "_hydro_storage [MWh]"
            combined_data = combined_data.set_index("timestamp").rename(columns={"Storage (MWh)": f"{country}_hydro_storage [MWh]"})

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

def plot_time_series(data, title, output_path):
    """
    Plots the time series data and saves the plot as a PNG file.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data with columns 'timestamp', 'Storage (MWh)', and 'Week'.
        title (str): Title of the plot.
        output_path (str): Path to save the plot as a PNG file.
    """
    if data.empty:
        print("Keine Daten zum Plotten verfügbar.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data["timestamp"], data["Storage (MWh)"], label="Storage (MWh)", marker='o', markersize=4)
    plt.xlabel("Zeit")
    plt.ylabel("Storage (MWh)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.show()

def plot_time_series_by_psr_type(data, title="Zeitreihe nach Typ"):
    """
    Plots the time series data for each psrType.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data with columns 'timestamp', 'psrType', and 'quantity'.
        title (str): Title of the plot.
    """
    if data.empty:
        print("Keine Daten zum Plotten verfügbar.")
        return

    plt.figure(figsize=(12, 6))
    for psr_type in data["psrType"].unique():
        subset = data[data["psrType"] == psr_type]
        plt.plot(subset["timestamp"], subset["quantity"], label=f"psrType: {psr_type}", marker='o', markersize=4)

    plt.xlabel("Zeit")
    plt.ylabel("Menge (MWh)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    # Länder und zugehörige Dateien
    countries = {
        "CH": [
            "./data/aggregate_water_reservoirs_and_hydro_storage_CH_2020.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_CH_2021.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_CH_2022.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_CH_2023.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_CH_2024.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_CH_2025.xml",
        ],
        "AT": [
            "./data/aggregate_water_reservoirs_and_hydro_storage_AT_2020.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_AT_2021.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_AT_2022.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_AT_2023.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_AT_2024.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_AT_2025.xml",
        ],
        "FR": [
            "./data/aggregate_water_reservoirs_and_hydro_storage_FR_2020.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_FR_2021.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_FR_2022.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_FR_2023.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_FR_2024.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_FR_2025.xml",
        ],
        "IT": [
            "./data/aggregate_water_reservoirs_and_hydro_storage_IT_2020.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_IT_2021.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_IT_2022.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_IT_2023.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_IT_2024.xml",
            "./data/aggregate_water_reservoirs_and_hydro_storage_IT_2025.xml",
        ],
    }

    # Verarbeite die Daten für jedes Land
    for country, filepaths in countries.items():
        print(f"Verarbeite Daten für {country}...")

        # Dateien kombinieren
        combined_data = combine_files(filepaths)

        if not combined_data.empty:
            print(f"Daten für {country} erfolgreich kombiniert:")
            print(combined_data)

            # Speichere die kombinierte DataFrame in eine CSV-Datei
            output_csv_path = f"./data/combined_aggregate_water_reservoirs_and_hydro_storage_{country}_2020_2025.csv"
            combined_data.to_csv(output_csv_path, index=False)
            print(f"Kombinierte Daten für {country} wurden in {output_csv_path} gespeichert.")

            # Plot der kombinierten Zeitreihe
            output_plot_path = f"./data/combined_aggregate_water_reservoirs_and_hydro_storage_{country}_2020_2025.png"
            plot_time_series(combined_data, title=f"Aggregate Water Reservoirs and Hydro Storage ({country} 2020-2025)", output_path=output_plot_path)
            print(f"Plot für {country} wurde in {output_plot_path} gespeichert.")
        else:
            print(f"Keine Daten für {country} geladen.")

    # Kombiniere alle Länder in eine einzige Datei
    output_csv_path = "./data/combined_aggregate_water_reservoirs_and_hydro_storage_all_countries_2020_2025.csv"
    combine_all_countries_to_single_file(countries, output_csv_path)


# CSV einlesen und Zeitstempel parsen
df = pd.read_csv(output_csv_path, parse_dates=["timestamp"])
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
