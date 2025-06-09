import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def parse_xml_file(filepath):
    import xml.etree.ElementTree as ET
    import pandas as pd

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

        time_interval = root.find(".//ns:time_Period.timeInterval", ns)
        start_time = pd.to_datetime(time_interval.find("ns:start", ns).text)

        records = []

        for ts in root.findall(".//ns:TimeSeries", ns):
            psr_type_elem = ts.find(".//ns:MktPSRType/ns:psrType", ns)
            psr_type = psr_type_elem.text if psr_type_elem is not None else "UNKNOWN"

            for period in ts.findall(".//ns:Period", ns):
                resolution = period.find("ns:resolution", ns).text
                for point in period.findall("ns:Point", ns):
                    position = int(point.find("ns:position", ns).text)
                    quantity = float(point.find("ns:quantity", ns).text)

                    if resolution == "PT60M":
                        delta = pd.to_timedelta(position - 1, unit='h')
                    elif resolution == "PT30M":
                        delta = pd.to_timedelta((position - 1) * 30, unit='min')
                    elif resolution == "PT15M":
                        delta = pd.to_timedelta((position - 1) * 15, unit='min')
                    else:
                        continue

                    timestamp = start_time + delta
                    records.append({"timestamp": timestamp, "value": quantity, "psrType": psr_type})

        return pd.DataFrame(records)

    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()


        df = pd.DataFrame(records)
        if df.empty:
            print(f"Keine Daten in {filepath} gefunden.")
        return df

    except Exception as e:
        print(f"Fehler beim Parsen von {filepath}: {e}")
        return pd.DataFrame()

def combine_files_by_country_and_type(filepaths_by_country_and_type):
    import os
    import pandas as pd

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

            if not combined_data.empty:
                pivot_df = combined_data.pivot_table(
                    index="timestamp", columns="psrType", values="value", aggfunc="first"
                ).reset_index()

                pivot_df.columns = [
                    "timestamp" if col == "timestamp"
                    else f"{data_type}_{PSRTYPE_MAPPING.get(col, col)}"
                    for col in pivot_df.columns
                ]
                pivot_df = pivot_df.sort_values("timestamp").reset_index(drop=True)
                
                pivot_df = pivot_df.set_index("timestamp").asfreq("H")  # lückenloser Stundenindex
                pivot_df = pivot_df.reset_index()
                # Add a total column that sums all psrType columns
                sum_column_name = f"{data_type}_Total [MWh]"
                pivot_df[sum_column_name] = pivot_df.drop(columns=["timestamp"]).sum(axis=1)


                output_dir = os.path.dirname(filepaths[0])
                output_path = os.path.join(output_dir, f"data_combined_{country}_{data_type}.csv")
                pivot_df.to_csv(output_path, index=False)
                print(f" --> Gespeichert: {output_path}")

                combined_data_by_type[data_type] = pivot_df

        combined_data_by_country_and_type[country] = combined_data_by_type

    return combined_data_by_country_and_type

def combine_all_countries_to_single_file(combined_data_by_country_and_type, output_path):
    import pandas as pd

    combined_all_countries = pd.DataFrame()

    for country, combined_data_by_type in combined_data_by_country_and_type.items():
        for data_type, data in combined_data_by_type.items():
            if not data.empty:
                if "timestamp" not in data.columns:
                    raise ValueError("Spalte 'timestamp' fehlt in einem DataFrame")

                data = data.set_index("timestamp")
                data = data.rename(columns=lambda col: f"{country}_{col}" if not col.startswith("timestamp") else col)

                combined_all_countries = (
                    data if combined_all_countries.empty
                    else combined_all_countries.join(data, how="outer")
                )

    combined_all_countries = combined_all_countries.sort_index()
    combined_all_countries.to_csv(output_path)
    print(f"Alle kombinierten Daten wurden in {output_path} gespeichert.")

def plot_combined_data_by_country_and_type(combined_data_by_country_and_type):
    """
    Plots the combined time series data for each country and type.
    Bei gepivoteten Daten werden alle psrType-Spalten separat geplottet.
    """
    for country, combined_data_by_type in combined_data_by_country_and_type.items():
        for data_type, data in combined_data_by_type.items():
            if not data.empty:
                plt.figure(figsize=(14, 6))
                for col in data.columns:
                    if col != "timestamp":
                        plt.plot(data["timestamp"], data[col], label=col)
                plt.xlabel("Zeit")
                plt.ylabel("Produktion [MWh]")
                plt.title(f"{country} - {data_type}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    # Define file paths for each country and type of data
    filepaths_by_country_and_type = {
        "CH": {
            "generation": [
                "./data/generation_CH_2020.xml",
                "./data/generation_CH_2021.xml",
                "./data/generation_CH_2022.xml",
                "./data/generation_CH_2023.xml",
                "./data/generation_CH_2024.xml",
                "./data/generation_CH_2025.xml"
            ],
        },
        "AT": {
            "generation": [
                "./data/generation_AT_2020.xml",
                "./data/generation_AT_2021.xml",
                "./data/generation_AT_2022.xml",
                "./data/generation_AT_2023.xml",
                "./data/generation_AT_2024.xml",
                "./data/generation_AT_2025.xml"
            ],
        },
        "FR": {
            "generation": [
                "./data/generation_FR_2020.xml",
                "./data/generation_FR_2021.xml",
                "./data/generation_FR_2022.xml",
                "./data/generation_FR_2023.xml",
                "./data/generation_FR_2024.xml",
                "./data/generation_FR_2025.xml"
            ],
        },
        "DE_LU": {
            "generation": [
                "./data/generation_DE_LU_2020.xml",
                "./data/generation_DE_LU_2021.xml",
                "./data/generation_DE_LU_2022.xml",
                "./data/generation_DE_LU_2023.xml",
                "./data/generation_DE_LU_2024.xml",
                "./data/generation_DE_LU_2025.xml"
            ],
        },
        "IT": {
            "generation": [
                "./data/generation_IT_2020.xml",
                "./data/generation_IT_2021.xml",
                "./data/generation_IT_2022.xml",
                "./data/generation_IT_2023.xml",
                "./data/generation_IT_2024.xml",
                "./data/generation_IT_2025.xml"
            ],
        },
    }
    PSRTYPE_MAPPING = {
        "A03": "Mixed [MWh]",
        "A04": "Generation [MWh]",
        "A05": "Load [MWh]",
        "B01": "Biomass [MWh]",
        "B02": "Fossil_Brown_coal_Lignite [MWh]",
        "B03": "Fossil_Coal_derived_gas [MWh]",
        "B04": "Fossil_Gas [MWh]",
        "B05": "Fossil_Hard_coal [MWh]",
        "B06": "Fossil_Oil [MWh]",
        "B07": "Fossil_Oil_shale [MWh]",
        "B08": "Fossil_Peat [MWh]",
        "B09": "Geothermal [MWh]",
        "B10": "Hydro_Pumped_Storage [MWh]",
        "B11": "Hydro_Run_of_river_and_poundage [MWh]",
        "B12": "Hydro_Water_Reservoir [MWh]",
        "B13": "Marine [MWh]",
        "B14": "Nuclear [MWh]",
        "B15": "Other_renewablev [MWh]",
        "B16": "Solar [MWh]",
        "B17": "Waste [MWh]",
        "B18": "Wind_Offshore [MWh]",
        "B19": "Wind_Onshore [MWh]",
        "B20": "Other [MWh]",
        "B21": "AC_Link [MWh]",
        "B22": "DC_Link [MWh]",
        "B23": "Substation [MWh]",
        "B24": "Transformer [MWh]",
        "B25": "Energy_storage [MWh]"
    }

    # Combine files by country and type
    combined_data_by_country_and_type = combine_files_by_country_and_type(filepaths_by_country_and_type)

    # Speichere alle Länder in einer einzigen Datei
    output_csv_path = "./data/combined_generation_all_countries.csv"
    combine_all_countries_to_single_file(combined_data_by_country_and_type, output_csv_path)

    # Plot the combined data
    plot_combined_data_by_country_and_type(combined_data_by_country_and_type)
