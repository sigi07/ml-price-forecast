from entsoe import EntsoeRawClient
import pandas as pd
import os  # Import for directory handling

api_key = '0b5a8f2d-c618-47a2-8679-f21cf36ec231'  # Ersetzen Sie dies durch Ihren API-Schlüssel
client = EntsoeRawClient(api_key=api_key)

startyear = 2020  # Startjahr für die Datenabfrage
endyear = 2025  # Endjahr für die Datenabfrage
country_code = 'CH'  # Ersetzen Sie dies durch den gewünschten Ländercode

# --- Neighbor-Mapping nach Bidding Zone ---
NEIGHBOR_MAP = {
    "CH": ["FR", "DE_LU", "IT", "AT"],
    "FR": ["CH"],
    "DE_LU": ["CH"],
    "IT": ["CH"],
    "AT": ["CH"],
}

def resolve_neighbors(country_code: str, neighbors: list | None) -> list:
    """
    Wenn neighbors angegeben ist und nicht leer -> diese verwenden.
    Sonst aus NEIGHBOR_MAP ableiten. Entfernt ggf. Selbst-Referenz.
    """
    if neighbors:
        chosen = neighbors[:]
    else:
        chosen = NEIGHBOR_MAP.get(country_code, []).copy()

    # Sicherheit: keine Selbst-Nachbarschaft
    chosen = [n for n in chosen if n != country_code]
    return chosen

def save_xml_to_file(xml_data, filename):
    """
    Saves the XML data to a file in the 'data' subdirectory.

    Args:
        xml_data (str): The XML data as a string.
        filename (str): The name of the file to save the XML data.
    """  
    try:
        # Ensure the 'data' directory exists
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)

        # Save the file in the 'data' directory
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(xml_data)
        print(f"XML data saved to {output_file}")
    except Exception as e:
        print(f"Failed to save XML data to {filename}: {e}")

def fetch_yearly_data_for_all(client, start_year, end_year, country_code, neighbors=None):
    """
    Fetches data for each year in the range for all queries and saves them to XML files.
    If neighbors is None or empty, it auto-resolves neighbors from NEIGHBOR_MAP.
    """
    # Queries ohne Nachbar (bei Bedarf wieder aktivieren)
    queries = [
        # ("day_ahead_prices", client.query_day_ahead_prices, [country_code]),
        # ("load", client.query_load, [country_code]),
        # ("load_forecast", client.query_load_forecast, [country_code]),
        # ("wind_and_solar_forecast", client.query_wind_and_solar_forecast, [country_code], {"psr_type": None}),
        # ("generation_forecast", client.query_generation_forecast, [country_code]),
        # ("generation", client.query_generation, [country_code], {"psr_type": None}),
        # ("installed_generation_capacity", client.query_installed_generation_capacity, [country_code], {"psr_type": None}),
        # ("aggregate_water_reservoirs_and_hydro_storage", client.query_aggregate_water_reservoirs_and_hydro_storage, [country_code]),
    ]

    # Nachbarn bestimmen
    resolved_neighbors = resolve_neighbors(country_code, neighbors)
    if resolved_neighbors:
        print(f"[INFO] Verwende Nachbarn für {country_code}: {', '.join(resolved_neighbors)}")
    else:
        print(f"[WARN] Keine Nachbarn für {country_code} gefunden/angegeben – Crossborder-Queries werden übersprungen.")

    # Crossborder-Queries je Nachbar anhängen
    for neighbor in resolved_neighbors:
        queries.extend([
            ("crossborder_flows", client.query_crossborder_flows, [country_code, neighbor], {}, neighbor),
            # ("scheduled_exchanges", client.query_scheduled_exchanges, [country_code, neighbor], {"dayahead": False}, neighbor),
            # ("net_transfer_capacity_dayahead", client.query_net_transfer_capacity_dayahead, [country_code, neighbor], {}, neighbor),
        ])

    # Einheitlich iterieren – neighbor optional anhängen
    for query in queries:
        # Entpacken mit optionalem kwargs und neighbor
        if len(query) == 5:
            query_name, query_function, args, kwargs, neighbor = query
        else:
            query_name, query_function, args = query[:3]
            kwargs = query[3] if len(query) > 3 else {}
            neighbor = None

        # Wenn es sich um eine Query handelt, die explizit einen Nachbarn braucht, aber keiner gesetzt ist: skip
        if query_name in {"crossborder_flows", "scheduled_exchanges", "net_transfer_capacity_dayahead"} and not neighbor:
            continue

        for year in range(start_year, end_year + 1):
            start = pd.Timestamp(f"{year}-01-01", tz="Europe/Brussels")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Brussels")
            try:
                tag = f"{country_code}-{neighbor}" if neighbor else country_code
                print(f"Fetching {query_name} for {tag} {year}...")
                xml_data = query_function(*args, start, end, **kwargs)

                # Dateiname
                if neighbor:
                    filename = f"{query_name}_{country_code}-{neighbor}_{year}.xml"
                else:
                    filename = f"{query_name}_{country_code}_{year}.xml"

                save_xml_to_file(xml_data, filename)
            except Exception as e:
                print(f"Failed to fetch {query_name} ({neighbor or '-'}) for {year}: {e}")

# Fetch yearly data for all queries from 2020 to 2025
try:
    fetch_yearly_data_for_all(client, startyear, endyear, country_code, neighbors=None)
except Exception as e:
    print(f"Error: {e}")
