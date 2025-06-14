from entsoe import EntsoeRawClient
import pandas as pd
import os  # Import for directory handling

api_key = '0b5a8f2d-c618-47a2-8679-f21cf36ec231'  # Ersetzen Sie dies durch Ihren API-Schlüssel
client = EntsoeRawClient(api_key=api_key)

startyear = 2021  # Startjahr für die Datenabfrage
endyear = 2021  # Endjahr für die Datenabfrage
country_code = 'FR'  # Ersetzen Sie dies durch den gewünschten Ländercode

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


def fetch_yearly_data_for_all(client, start_year, end_year, country_code, neighbors):
    """
    Fetches data for each year in the range for all queries and combines them into separate XML files.

    Args:
        client (EntsoeRawClient): ENTSO-E client instance.
        start_year (int): Start year for the data retrieval.
        end_year (int): End year for the data retrieval.
        country_code (str): Country code (e.g., 'FR' for France).
        neighbors (list): List of neighboring country codes for cross-border queries.
    """
    queries = [
        #("day_ahead_prices", client.query_day_ahead_prices, [country_code]),
        #("load", client.query_load, [country_code]),
        #("load_forecast", client.query_load_forecast, [country_code]),
        #("wind_and_solar_forecast", client.query_wind_and_solar_forecast, [country_code], {"psr_type": None}),
        #("intraday_wind_and_solar_forecast", client.query_intraday_wind_and_solar_forecast, [country_code], {"psr_type": None}),
        #("generation_forecast", client.query_generation_forecast, [country_code]),
        ("generation", client.query_generation, [country_code], {"psr_type": None}),
        #("generation", client.query_generation, [country_code], {"psr_type": None})
        #("installed_generation_capacity", client.query_installed_generation_capacity, [country_code], {"psr_type": None}),
        #("aggregate_water_reservoirs_and_hydro_storage", client.query_aggregate_water_reservoirs_and_hydro_storage, [country_code]),
    ]

    # Add cross-border queries for each neighbor
    for neighbor in neighbors:
        queries.extend([
            #("crossborder_flows", client.query_crossborder_flows, [country_code, neighbor]),
            #("scheduled_exchanges", client.query_scheduled_exchanges, [country_code, neighbor], {"dayahead": False}),
            #("net_transfer_capacity_dayahead", client.query_net_transfer_capacity_dayahead, [country_code, neighbor]),
        ])

    for query_name, query_function, args, kwargs in [(q[0], q[1], q[2], q[3] if len(q) > 3 else {}) for q in queries]:
        yearly_data = []
        for year in range(start_year, end_year + 1):
            start = pd.Timestamp(f'{year}-01-01', tz='Europe/Brussels')
            end = pd.Timestamp(f'{year + 1}-01-01', tz='Europe/Brussels')
            try:
                print(f"Fetching {query_name} for {year}...")
                xml_data = query_function(*args, start, end, **kwargs)
                yearly_data.append(xml_data)
                save_xml_to_file(xml_data, f'{query_name}_{country_code}_{year}.xml')
            except Exception as e:
                print(f"Failed to fetch {query_name} for {year}: {e}")


# Define neighbors
neighbors = []  # For FR
#neighbors = ["FR", "DE_LU", "IT", "AT"]  # for CH
#neighbors = ["BE", "DE_LU", "IT", "ES", "CH"]  # for FR
#neighbors = ["BE", "NL", "FR", "CH", "AT", "CZ", "PL", "DK_1", "DK_2", "NO_2", "SE_4"]  # For DE_LU
#neighbors = ["CH", "CZ", "DE_LU", "HU", "IT", "SI", ]  # for AT
#neighbors = ["FR","CH", "AT", "SI", "GR", "MT"]  # For IT

#neighbors = ["FR","DE_LU"]  # For BE Done
#neighbors = ["FR"]  # For ES Done
#neighbors = ["DE_LU"]  # For NL Done
#neighbors = ["DE_LU"]  # For PL Done
#neighbors = ["DE_LU", "AT"]  # For CZ Done
#neighbors = ["DE_LU"]  # For DK_1 Done
#neighbors = ["DE_LU"]  # For No_2 Done
#neighbors = ["DE_LU"]  # For SE_4 Done
#neighbors = ["AT"]  # For HU
#neighbors = ["AT", "IT"]  # For SI
#neighbors = ["IT"]  # For GR
#neighbors = ["IT"]  # For MT

# Fetch yearly data for all queries from 2020 to 2025
try:
    fetch_yearly_data_for_all(client, startyear, endyear, country_code, neighbors)
except Exception as e:
    print(f"Error: {e}")
