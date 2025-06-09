import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
from functools import reduce
import matplotlib.pyplot as plt

# XML-Namespace gemäss ENTSO-E Standard
NAMESPACE = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}

def parse_file(filepath, column_name):
    """
    Liest eine XML-Datei mit Zeitreihendaten im ENTSO-E Format ein und gibt ein DataFrame mit timestamp und Werten zurück.
    """
    data = []

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for ts in root.findall("ns:TimeSeries", NAMESPACE):
            for period in ts.findall("ns:Period", NAMESPACE):
                start_time = period.find("ns:timeInterval/ns:start", NAMESPACE).text
                resolution = period.find("ns:resolution", NAMESPACE).text

                for point in period.findall("ns:Point", NAMESPACE):
                    pos = point.find("ns:position", NAMESPACE)
                    qty = point.find("ns:quantity", NAMESPACE)

                    if pos is not None and qty is not None:
                        pos = int(pos.text)
                        qty = float(qty.text)

                        if resolution == "PT60M":
                            timestamp = pd.to_datetime(start_time) + pd.to_timedelta(pos - 1, unit='h')
                        elif resolution == "PT15M":
                            timestamp = pd.to_datetime(start_time) + pd.to_timedelta((pos - 1) * 15, unit='m')
                        else:
                            raise ValueError(f"Nicht unterstützte Auflösung: {resolution}")

                        data.append({"timestamp": timestamp, column_name: qty})
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {filepath}: {e}")

    return pd.DataFrame(data)


def combine_all_xml_to_df(data_directory):
    """
    Sammelt alle XML-Dateien aus dem Verzeichnis und kombiniert die Zeitreihen in einem DataFrame.
    """
    dataframes = []

    for filename in sorted(os.listdir(data_directory)):
        if filename.endswith(".xml"):
            filepath = os.path.join(data_directory, filename)
            # Füge die Einheit [MWh] zum Spaltennamen hinzu
            column_name = f"{filename.replace('.xml', '')} [MWh]"
            print(f"[+] Verarbeite Datei: {filename}")
            df = parse_file(filepath, column_name)
            if not df.empty:
                dataframes.append(df)

    if not dataframes:
        print("[!] Keine gültigen XML-Dateien oder Daten gefunden.")
        return pd.DataFrame()

    # Kombiniere alle DataFrames anhand des 'timestamp'
    combined_df = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dataframes)
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    return combined_df


def plot_combined_timeseries(df, output_path):
    """
    Erstellt einen Zeitreihenplot aller Spalten außer 'timestamp' und speichert ihn.
    """
    plt.figure(figsize=(16, 6))
    for col in df.columns:
        if col != "timestamp":
            plt.plot(df["timestamp"], df[col], label=col, linewidth=0.8)

    plt.xlabel("Zeit")
    plt.ylabel("Wert")
    plt.title("Zeitverläufe der XML-Dateien")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[📈] Plot gespeichert unter: {output_path}")


if __name__ == "__main__":
    data_directory = "./data"  # ← Pfad zu deinen XML-Dateien
    output_csv = os.path.join(data_directory, "combined_all_time_series.csv")
    output_plot = os.path.join(data_directory, "combined_all_time_series_plot.png")

    combined_df = combine_all_xml_to_df(data_directory)

    if not combined_df.empty:
        combined_df.to_csv(output_csv, index=False)
        print(f"[✔] Zeitreihen gespeichert in: {output_csv}")
        plot_combined_timeseries(combined_df, output_plot)
    else:
        print("Kein DataFrame erzeugt.")
