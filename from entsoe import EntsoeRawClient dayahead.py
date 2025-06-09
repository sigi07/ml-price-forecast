from entsoe import EntsoeRawClient
import pandas as pd
import os

# API-Schlüssel
api_key = '0b5a8f2d-c618-47a2-8679-f21cf36ec231'  # Ersetzen Sie dies durch Ihren API-Schlüssel
client = EntsoeRawClient(api_key=api_key)

# Liste der Länder, die abgerufen werden sollen
countries = ["IT_NORD"]

def save_xml_to_file(xml_data, filename):
    
    """
    Speichert die XML-Daten in einer Datei im Verzeichnis 'data_price'.

    Args:
        xml_data (str): Die XML-Daten als String.
        filename (str): Der Dateiname, unter dem die XML-Daten gespeichert werden.
    """
    try:
        # Sicherstellen, dass das Verzeichnis 'data_price' existiert
        output_dir = 'data_price'
        os.makedirs(output_dir, exist_ok=True)

        # Speichern der Datei im Verzeichnis 'data_price'
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(xml_data)
        print(f"XML-Daten wurden in {output_file} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der XML-Daten in {filename}: {e}")

def fetch_day_ahead_prices_for_week(client, year, week, country_code):
    """
    Fragt die Day-Ahead-Preise für eine bestimmte Kalenderwoche ab und speichert sie als XML-Datei.

    Args:
        client (EntsoeRawClient): ENTSO-E Client-Instanz.
        year (int): Jahr der Kalenderwoche.
        week (int): Kalenderwoche.
        country_code (str): Ländercode (z. B. 'AT' für Österreich).
    """
    # Berechne den Start- und Endzeitpunkt der Kalenderwoche
    start = pd.Timestamp.fromisocalendar(year, week, 1).tz_localize('Europe/Brussels')  # Montag der KW
    end = (start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59))  # Sonntag der KW

    try:
        print(f"Abfrage der Day-Ahead-Preise für {country_code} von {start} bis {end}...")
        xml_data = client.query_day_ahead_prices(country_code, start, end)
        filename = f"day_ahead_prices_{country_code}_KW{week}_{year}.xml"
        save_xml_to_file(xml_data, filename)
        return xml_data
    except Exception as e:
        print(f"Fehler bei der Abfrage der Day-Ahead-Preise von KW {week} ({start.date()} bis {end.date()}): {e}")
        return None

def combine_all_xml_files(output_filename):
    """
    Kombiniert alle XML-Dateien im Verzeichnis 'data_price' in eine einzige Datei.

    Args:
        output_filename (str): Der Name der kombinierten XML-Datei.
    """
    try:
        output_dir = 'data_price'
        combined_data = "<CombinedData>\n"

        # Iteriere über alle XML-Dateien im Verzeichnis
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith(".xml"):
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    combined_data += file.read() + "\n"

        combined_data += "</CombinedData>"

        # Speichere die kombinierte Datei
        combined_file_path = os.path.join(output_dir, output_filename)
        with open(combined_file_path, 'w', encoding='utf-8') as combined_file:
            combined_file.write(combined_data)
        print(f"Kombinierte XML-Daten wurden in {combined_file_path} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Kombinieren der XML-Dateien: {e}")

# Hauptprogramm
if __name__ == "__main__":
    # Eingabe des Start- und Endjahres
    start_year = int(input("Geben Sie das Startjahr ein (z. B. 2020): "))
    end_year = int(input("Geben Sie das Endjahr ein (z. B. 2022): "))

    # Schleife über alle Länder
    for country_code in countries:
        print(f"Starte Abfragen für {country_code}...")
        # Schleife über alle Jahre und Kalenderwochen
        for year in range(start_year, end_year + 1):
            for week in range(1, 53):  # Maximal 52 Wochen in einem Jahr
                try:
                    fetch_day_ahead_prices_for_week(client, year, week, country_code)
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von KW {week} in Jahr {year} für {country_code}: {e}")

    # Kombiniere alle XML-Dateien in eine Datei
    combined_filename = f"combined_day_ahead_prices_{start_year}_{end_year}.xml"
    combine_all_xml_files(combined_filename)