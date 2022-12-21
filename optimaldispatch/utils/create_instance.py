import csv
import json


# Exports
__all__ = ["create_instance"]


def create_instance(output_file, demand_file, generators_file, battery_file, fuel_file, solar_file):
    """
    Create an instance in JSON format from CSV files.
    :param output_file: File in which JSON data will be written.
    :param demand_file: CSV file with demand data.
    :param generators_file: CSV file with generators data.
    :param battery_file: CSV file with battery data.
    :param fuel_file: CSV file with fuel data.
    :param solar_file: CSV file with solar data.
    """
    data = dict()
    data["n_intervals"] = 0
    data["n_generators"] = 0
    data["business_hours"] = {
        "start": 8,  # 8:00 AM
        "end": 17    # 5:00 PM
    }
    data["demand"] = []
    data["selling_price"] = []
    data["buying_price"] = []
    data["generators"] = []
    data["battery"] = dict()
    data["biogas"] = dict()
    data["ethanol"] = dict()
    data["biomethane"] = dict()
    data["gnv"] = dict()
    data["buses_demand"] = None
    data["solar_energy"] = []

    # Read demand data
    demand_reader = csv.DictReader(demand_file, delimiter=",")
    for row in demand_reader:
        data["n_intervals"] += 1
        data["demand"].append(float(row["DEMAND"]))
        data["selling_price"].append(float(row["SELLING.PRICE"]))
        data["buying_price"].append(float(row["BUYING.PRICE"]))

    # Read generators data
    generators_reader = csv.DictReader(generators_file, delimiter=",")
    for row in generators_reader:
        generator_data = dict()
        generator_data["initial_state"] = int(row["INITIAL.STATE"])
        generator_data["upper_rate"] = float(row["UPPER.RATE"])
        generator_data["lower_rate"] = float(row["LOWER.RATE"])
        generator_data["ramp_rate_limit"] = float(row["RAMP.RATE.LIMIT"])
        generator_data["efficiency_1"] = (float(row["EFFICIENCY1.A"]), float(row["EFFICIENCY1.B"]))
        generator_data["efficiency_2"] = (float(row["EFFICIENCY2.A"]), float(row["EFFICIENCY2.B"]))
        generator_data["efficiency_3"] = (float(row["EFFICIENCY3.A"]), float(row["EFFICIENCY2.B"]))
        generator_data["up_cost"] = float(row["UP.COST"])
        generator_data["down_cost"] = float(row["DOWN.COST"])
        generator_data["up_down_window"] = int(row["UP.DOWN.WINDOW"])

        data["n_generators"] += 1
        data["generators"].append(generator_data)

    # Read battery data
    battery_reader = csv.DictReader(battery_file, delimiter=",")
    for row in battery_reader:
        data["battery"]["initial_load"] = float(row["INITIAL.LOAD"])
        data["battery"]["eff_charge"] = float(row["EFFICIENCY.CHARGE"])
        data["battery"]["eff_discharge"] = float(row["EFFICIENCY.DISCHARGE"])
        data["battery"]["max_load"] = float(row["MAX.LOAD"])
        data["battery"]["max_flow"] = float(row["MAX.FLOW"])
        data["battery"]["dod"] = float(row["MAX.DOD"])
        data["battery"]["cost"] = float(row["COST"])
        break

    # Read fuel data
    fuel_reader = csv.DictReader(fuel_file, delimiter=",")
    for row in fuel_reader:
        data["biogas"]["mean_production"] = float(row["BIOGAS.MEAN.PRODUCTION"])
        data["biogas"]["deviation_production"] = float(row["BIOGAS.DEVIATION.PRODUCTION"])
        data["biogas"]["input_cost"] = float(row["INPUT.COST"])
        data["biogas"]["maintenance_cost"] = (float(row["BG.MAINTENANCE.COST.A"]), float(row["BG.MAINTENANCE.COST.B"]))
        data["biogas"]["transport_cost"] = float(row["TRANSPORT.COST"])
        data["biogas"]["consumption"] = (float(row["BG.CONSUMPTION.A"]), float(row["BG.CONSUMPTION.B"]))
        data["ethanol"]["cost"] = float(row["ETHANOL.COST"]) / 5.93  # convert from $/liter to $/KWh
        data["ethanol"]["disponibility"] = float(row["ETHANOL.DISPONIBILITY"])
        data["biomethane"]["efficiency"] = float(row["BIOMETHANE.EFFICIENCY"])
        data["biomethane"]["maintenance_cost"] = (float(row["BM.MAINTENANCE.COST.A"]), float(row["BM.MAINTENANCE.COST.B"]))
        data["biomethane"]["energy_consumption"] = float(row["BM.CONSUMPTION.A"])
        data["gnv"]["cost"] = float(row["GNV.COST"])
        data["buses_demand"] = float(row["BUSES.DEMAND"])
        break

    data["biomethane"]["compression_capacity"] = 60  # m3/h

    # Read solar data
    solar_reader = csv.DictReader(solar_file, delimiter=",")
    for row in solar_reader:
        data["solar_energy"].append(float(row["SOLAR"]))

    # Export data to JSON file
    json.dump(data, output_file, indent=2)
