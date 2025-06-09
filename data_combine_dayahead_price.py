import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def parse_weekly_file(filepath):
    """
    Liest eine einzelne XML-Datei ein und extrahiert die Zeitreihendaten.

    Args:
        filepath (str): Pfad zur XML-Datei.

    Returns:
        pd.DataFrame: Ein DataFrame mit den Spalten 'timestamp' und 'price'.
    """
    try:
        # Parse die XML-Datei
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Namespace für die XML-Datei
        namespace = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

        data = []

        # Iteriere über alle TimeSeries-Elemente
        for time_series in root.findall("ns:TimeSeries", namespace):
            for period in time_series.findall("ns:Period", namespace):
                start_time = period.find("ns:timeInterval/ns:start", namespace).text
                resolution = period.find("ns:resolution", namespace).text

                for point in period.findall("ns:Point", namespace):
                    position = int(point.find("ns:position", namespace).text)
                    price = float(point.find("ns:price.amount", namespace).text)

                    # Berechne den Timestamp basierend auf der Startzeit und der Position
                    if resolution == "PT60M":  # Stündliche Auflösung
                        timestamp = pd.to_datetime(start_time) + pd.to_timedelta(position - 1, unit='h')
                    elif resolution == "PT15M":  # Viertelstündliche Auflösung
                        timestamp = pd.to_datetime(start_time) + pd.to_timedelta((position - 1) * 15, unit='m')
                    else:
                        raise ValueError(f"Nicht unterstützte Auflösung: {resolution}")

                    data.append({"timestamp": timestamp, "price": price})

        # Konvertiere die Daten in ein DataFrame
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {filepath}: {e}")
        return pd.DataFrame()

def combine_weekly_files_to_dataframe(directory, country_code):
    """
    Kombiniert alle XML-Dateien für ein bestimmtes Land in ein einziges DataFrame.

    Args:
        directory (str): Verzeichnis, das die XML-Dateien enthält.
        country_code (str): Ländercode (z. B. 'CH').

    Returns:
        pd.DataFrame: Ein DataFrame mit den kombinierten Daten.
    """
    combined_data = pd.DataFrame()

    # Iteriere über alle Dateien im Verzeichnis
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(f"day_ahead_prices_{country_code}_") and filename.endswith(".xml"):
            filepath = os.path.join(directory, filename)
            print(f"Lese Datei {filepath} ein...")
            weekly_data = parse_weekly_file(filepath)

            # Überprüfen, ob die Spalten 'timestamp' und 'price' vorhanden sind
            if not weekly_data.empty and "timestamp" in weekly_data.columns and "price" in weekly_data.columns:
                combined_data = pd.concat([combined_data, weekly_data], ignore_index=True)
            else:
                print(f"Warnung: Datei {filepath} enthält keine gültigen Daten und wird übersprungen.")

    # Überprüfen, ob das kombinierte DataFrame nicht leer ist
    if not combined_data.empty:
        # Sortiere die Daten nach Timestamp
        combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)
    else:
        print(f"Keine gültigen Daten für {country_code} gefunden.")

    return combined_data

def combine_all_countries_to_dataframe(directory, countries):
    """
    Kombiniert die Daten aller Länder in ein einziges DataFrame, wobei jedes Land eine eigene Spalte erhält.

    Args:
        directory (str): Verzeichnis, das die XML-Dateien enthält.
        countries (list): Liste der Ländercodes (z. B. ['CH', 'IT_NORD', 'FR', 'DE_LU']).

    Returns:
        pd.DataFrame: Ein DataFrame mit Zeitstempeln als Index und den Ländern als Spalten.
    """
    combined_data = pd.DataFrame()

    for country_code in countries:
        country_data = combine_weekly_files_to_dataframe(directory, country_code)

        if not country_data.empty:
            # Setze den Zeitstempel als Index und benenne die Spalte nach dem Ländercode mit "_price [Euro]"
            country_data = country_data.set_index("timestamp").rename(columns={"price": f"{country_code}_price [Euro]"})

            # Kombiniere die Daten mit dem Haupt-DataFrame
            if combined_data.empty:
                combined_data = country_data
            else:
                combined_data = combined_data.join(country_data, how="outer")

    # Sortiere die Daten nach Zeitstempel
    combined_data = combined_data.sort_index()

    return combined_data

def plot_combined_time_series(data_dict, title, output_path):
    """
    Plottet die Zeitreihendaten für mehrere Länder und speichert den Plot als PNG-Datei.

    Args:
        data_dict (dict): Ein Dictionary mit Ländern als Schlüssel und DataFrames als Werte.
        title (str): Titel des Plots.
        output_path (str): Pfad, um den Plot als PNG-Datei zu speichern.
    """
    plt.figure(figsize=(14, 8))

    for country, data in data_dict.items():
        if not data.empty:
            plt.plot(data["timestamp"], data["price"], label=f"{country} (EUR)", linewidth=1)

    plt.xlabel("Zeit")
    plt.ylabel("Preis (EUR)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    # Verzeichnis mit den XML-Dateien
    data_directory = "./data_price"
    countries = ["CH", "IT_NORD", "FR", "DE_LU", "AT"]  # Länder, die verarbeitet werden sollen

    # Dictionary, um die kombinierten Daten für jedes Land zu speichern
    combined_data_dict = {}

    for country_code in countries:
        # Kombiniere alle Wochen-Dateien in ein DataFrame
        combined_data = combine_weekly_files_to_dataframe(data_directory, country_code)

        if not combined_data.empty:
            print(f"Daten für {country_code} erfolgreich kombiniert:")
            print(combined_data.head())

            # Speichere die kombinierten Daten in eine CSV-Datei
            output_csv_path = f"./data_price/combined_day_ahead_prices_{country_code}_2020_2025.csv"
            combined_data.to_csv(output_csv_path, index=False)
            print(f"Kombinierte Daten für {country_code} wurden in {output_csv_path} gespeichert.")

            # Füge die Daten dem Dictionary hinzu
            combined_data_dict[country_code] = combined_data
        else:
            print(f"Keine Daten für {country_code} gefunden.")
            combined_data_dict[country_code] = pd.DataFrame()

    # Kombiniere die Daten aller Länder in ein DataFrame
    combined_data = combine_all_countries_to_dataframe(data_directory, countries)

    if not combined_data.empty:
        print("Daten für alle Länder erfolgreich kombiniert:")
        print(combined_data.head())

        # Speichere die kombinierten Daten in eine CSV-Datei
        output_csv_path = "./data_price/combined_day_ahead_prices_all_countries_2020_2025.csv"
        combined_data.to_csv(output_csv_path)
        print(f"Kombinierte Daten für alle Länder wurden in {output_csv_path} gespeichert.")
    else:
        print("Keine Daten für die angegebenen Länder gefunden.")

    # Plot der Zeitreihen für alle Länder
    output_plot_path = "./data_price/combined_day_ahead_prices_all_countries_2020_2025.png"
    plot_combined_time_series(combined_data_dict, title="Day Ahead Prices (2020-2025)", output_path=output_plot_path)
    print(f"Plot wurde in {output_plot_path} gespeichert.")

file_path = "./data_price/combined_day_ahead_prices_all_countries_2020_2025.csv"
# CSV einlesen und Zeitstempel parsen
df = pd.read_csv(file_path, parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Duplikate aggregieren (Mittelwert je Zeitstempel)
df = df.groupby(df.index).mean()

# Auf Viertelstundenauflösung interpolieren
df_15min = df.resample("15min").interpolate(method="linear")

#Speichere die interpolierten Daten in eine CSV-Datei
df_15min.to_csv(output_csv_path.replace(".csv", "_15min.csv"), index=True)