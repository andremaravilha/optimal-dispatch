import copy
import math
import json
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.lines as mlines
from scipy.stats import norm

# Exports
__all__ = ["OptimalDispatch"]


class OptimalDispatch(object):
    """
    Class that defines the Optimal Dispatch problem.
    """

    def __init__(self, instance_file, cvar_epsilon=None):
        """
        Constructor.
        :param instance_file: JSON file with instance data
        """

        # Parse JSON file
        data = json.loads(instance_file.read())

        # Time horizon (24 hours divided into intervals)
        self.n_intervals = data["n_intervals"]
        self.business_hours = data["business_hours"]

        # Demand
        self.demand = data["demand"]

        # Energy price
        self.selling_price = data["selling_price"]
        self.buying_price = data["buying_price"]

        # Generators
        self.n_generators = data["n_generators"]
        self.generators = data["generators"]

        # Battery
        self.battery = data["battery"]

        # Fuel
        self.biogas = data["biogas"]
        self.ethanol = data["ethanol"]
        self.biomethane = data["biomethane"]
        self.gnv = data["gnv"]

        # Bus fleet
        self.buses_demand = data["buses_demand"]

        # Solar energy
        self.solar_energy = data["solar_energy"]

        # CVaR e-constraint
        self.enable_cvar = cvar_epsilon is not None
        self.epsilon = cvar_epsilon

    def evaluate(self, solution):
        """
        Calculate objective function of the given solution.
        :param solution: Solution to evaluate.
        :return: Objective function value.
        """

        # Expand solution
        solution = self.__expand_solution(solution)

        # Return total cost and constraints
        return solution["data"]["total_cost"], solution["constraints"]["inequality"], solution["constraints"]["equality"]

    def calculate_infeasibility(self, g, h, eps=1e-5):
        """
        Calculate infeasibility for all constraints.
        :param g: inequality constraints.
        :param h: equality constraints.
        :param eps: precision.
        :return: two lists: the first regarding to the inequality constraints and the
        second regarding to the equality constraints.
        """
        inf_g = [max(0, value - eps) for value in g]
        inf_h = [abs(value) > eps for value in h]
        return inf_g, inf_h

    def is_feasible(self, solution, eps=1e-5):
        """
        Check if a give solution is feasible.
        :param solution: Solution to check feasibility.
        :return: True if the solution is feasible, or False otherwise.
        """
        _, g, h = self.evaluate(solution)
        inf_g, inf_h = self.calculate_infeasibility(g, h, eps)
        return (sum(inf_g) + sum(inf_h)) > 0

    def draw_solution(self, solution, label="optimaldispatch", block=False, interactive=False, show=True):
        """
        Draw a solution.
        """

        # Expand solution
        solution = self.__expand_solution(solution)
        data = solution["data"]

        # Computes infeasibility
        inf_g, inf_h = self.calculate_infeasibility(solution["constraints"]["inequality"],
                                                    solution["constraints"]["equality"])

        infeasibility = sum(inf_g) + sum(inf_h)

        # Set interactivity on plot
        if interactive:
            plt.ion()
        else:
            plt.ioff()

        # Intervals (timeline: x-axis)
        intervals = list(range(0, self.n_intervals))
        xtick_label = [dates.num2date(1 + (i * (1.0 / self.n_intervals))).strftime('%H:%M') for i in intervals]
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)

        # Create a figure (or set it as the current one if already exists) and set its size
        plt.figure(label, figsize=(15, 10), clear=True)

        # Plot electrical demands
        plt_demands = plt.plot(intervals, data["electric_energy_demand"], color='black')
        previous = [0] * self.n_intervals

        # Plot solar energy
        plt_solar_energy = plt.bar(intervals, self.solar_energy, color='tab:orange', bottom=previous)
        previous = [previous[j] + self.solar_energy[j] for j in range(0, self.n_intervals)]

        # Plot generators (biogas and ethanol)
        rate_biogas = [0] * self.n_intervals
        rate_ethanol = [0] * self.n_intervals
        for i in range(0, self.n_generators):
            biogas = data["generators"]["biogas_rate"][i]
            ethanol = data["generators"]["ethanol_rate"][i]
            for j in range(0, self.n_intervals):
                rate_biogas[j] += biogas[j]
                rate_ethanol[j] += ethanol[j]

        plt_engine_biogas = plt.bar(intervals, rate_biogas, color='limegreen', bottom=previous)
        previous = [previous[j] + rate_biogas[j] for j in range(0, self.n_intervals)]

        plt_engine_ethanol = plt.bar(intervals, rate_ethanol, color='tab:green', bottom=previous)
        previous = [previous[j] + rate_ethanol[j] for j in range(0, self.n_intervals)]

        # Plot battery energy
        battery_energy_use = [max(0, -x) for x in data["battery"]["energy"]]
        plt_battery_energy = plt.bar(intervals, battery_energy_use, color='tab:blue', bottom=previous)
        previous = [previous[j] + battery_energy_use[j] for j in range(0, self.n_intervals)]

        # Purchased electricity
        purchased_electric_energy = data["commercialization"]["electric_energy_buying"]
        plt_purchased = plt.bar(intervals, purchased_electric_energy, color='gold', bottom=previous)
        previous = [previous[j] + purchased_electric_energy[j] for j in range(0, self.n_intervals)]

        # Defines the basis for plotting energy use beyond demand
        previous_excess = data["electric_energy_demand"].copy()

        # Battery charging
        battery_charging_energy = [max(0, x) for x in data["battery"]["energy"]]
        battery_charging_values = [battery_charging_energy[j] for j in range(0, self.n_intervals) if battery_charging_energy[j] > 0]
        battery_charging_intervals = [intervals[j] for j in range(0, self.n_intervals) if battery_charging_energy[j] > 0]
        battery_charging_previous = [previous_excess[j] for j in range(0, self.n_intervals) if battery_charging_energy[j] > 0]

        plt_battery_charging = plt.bar(battery_charging_intervals, battery_charging_values, facecolor=None,
                                       edgecolor="tab:red", hatch="///", fill=False, bottom=battery_charging_previous)

        previous_excess = [previous_excess[j] + battery_charging_energy[j] for j in range(0, self.n_intervals)]

        # Sold electricity
        sold_electric_energy = data["commercialization"]["electric_energy_selling"]
        sold_electric_values = [sold_electric_energy[j] for j in range(0, self.n_intervals) if sold_electric_energy[j] > 0]
        sold_electric_intervals = [intervals[j] for j in range(0, self.n_intervals) if sold_electric_energy[j] > 0]
        sold_electric_previous = [previous_excess[j] for j in range(0, self.n_intervals) if sold_electric_energy[j] > 0]

        plt_sold = plt.bar(sold_electric_intervals, sold_electric_values, facecolor=None, edgecolor="black",
                           hatch="...", fill=False, bottom=sold_electric_previous)

        previous_excess = [previous_excess[j] + sold_electric_energy[j] for j in range(0, self.n_intervals)]

        # Legend
        plt_demands_proxy = mlines.Line2D([], [], color='black', marker=None)
        plt_objects = [plt_demands_proxy, plt_solar_energy, plt_engine_biogas, plt_engine_ethanol, plt_battery_energy,
                       plt_purchased, plt_sold, plt_battery_charging]
        legends = ["Electrical demands", "PV", "Engine (biogas)", "Engine (ethanol)", "Battery use",
                   "Purchased electricity", "Sold electricity", "Battery charging"]

        plt.legend(plt_objects, legends, loc="upper left", ncol=2, fontsize=14)

        # Other attributes
        plt.xlabel("Time (HH:MM)", fontsize=14)
        plt.ylabel("Energy (kWh/2)", fontsize=14)
        #plt.xticks(intervals, xtick_label, rotation='vertical')
        plt.xticks(intervals, xtick_label, rotation=45)

        # Solution details
        battery_final_load = data["battery"]["load"][-1]
        biogas_produced = data["biogas"]["production"]
        biogas_for_engine = data["biogas"]["used_for_engines"]
        biomethane_produced = data["biomethane"]["production"]

        details = ""
        details += "Total cost: {:.2f} {}\n".format(data["total_cost"], "(infeasible solution)" if infeasibility > 1E-6 else "")
        details += "Biogas production: {:.2f} kWh\n".format(biogas_produced)
        details += "Engine biogas consumption: {:.2f} kWh ({:.2f}%)\n".format(biogas_for_engine, (biogas_for_engine / biogas_produced) * 100)
        details += "Bus fleet biomethane: {:.2f} kWh\n".format(biomethane_produced)
        details += "Purchased VNG: {:.2f} m³\n".format(data["bus_fleet"]["gnv_m3"])
        details += "Battery final load: {:.2f} kWh\n".format(battery_final_load)

        if self.enable_cvar:
            details += "CVaR: {:.2f} (Probability level: {:.2f})".format(data["cvar"]["cvar"], data["cvar"]["alpha"])

        #plt.figtext(0.0, 0.0, details, horizontalalignment='left', color='black', fontsize=16)
        #plt.text(0.0, -100, details, ha="left", fontsize=16, wrap=True)
        plt.gcf().text(0.05, 0.005, details, fontsize=14)
        plt.subplots_adjust(bottom=0.26, top=0.99, left=0.05, right=0.99)

        # Show
        if show:
            plt.show(block=block)

            # Allow time to draw
            if not block:
                plt.pause(1e-3)

    def solution_to_json(self, solution, file):

        # Get solution attributes
        generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production = solution

        cost, g, h = self.evaluate(solution)
        inf_g, inf_h = self.calculate_infeasibility(g, h)
        infeasibility = sum(inf_g) + sum(inf_h)

        data = dict()

        # cost and infeasibility
        data["cost"] = cost
        data["infeasibility"] = infeasibility

        # CVaR data
        data["cvar"] = {
            "enabled": self.enable_cvar,
            "epsilon": self.epsilon
        }

        # Generators' data
        data["generators"] = []
        for i in range(0, self.n_generators):
            data_generator = dict()
            data_generator["status"] = generators_status[i]
            data_generator["rate"] = generators_rate[i]
            data_generator["fuel_composition"] = generators_fuel_composition[i]
            data["generators"].append(data_generator)

        # Battery and biogas data
        data["battery_energy"] = battery_energy
        data["biogas_production"] = biogas_production

        json.dump(data, file, indent=2)

    def __expand_solution(self, solution):
        """
        Computes a set of auxiliary data from a given solution, returning them into a dictionary. Besides the auxiliary
        data computed from the solution, the dictionary contains the attributes of the solution itself. An expanded
        solution is used to evaluates its cost, evaluates its constraints, and draw it.
        The dictionary returned has the following structure:
        TODO
        :param solution: Solution to compute auxiliary data.
        """

        # Dictionary to store the all solution attributes and its auxiliary data
        expanded_solution = {
            "data": {
                "total_cost": None,
                "electric_energy_demand": [],
                "battery": {
                    "energy": [],
                    "load": [],
                    "cost": None
                },
                "generators": {
                    "status": [[] for _ in range(0, self.n_generators)],
                    "rate": [[] for _ in range(0, self.n_generators)],
                    "fuel_composition": [[] for _ in range(0, self.n_generators)],
                    "efficiency": [[] for _ in range(0, self.n_generators)],
                    "biogas_consumption": [[] for _ in range(0, self.n_generators)],
                    "ethanol_consumption": [[] for _ in range(0, self.n_generators)],
                    "biogas_rate": [[] for _ in range(0, self.n_generators)],
                    "ethanol_rate": [[] for _ in range(0, self.n_generators)],
                    "status_changed": [[] for _ in range(0, self.n_generators)],
                    "fuel_cost": [None for _ in range(0, self.n_generators)],
                    "up_down_cost": [None for _ in range(0, self.n_generators)],
                },
                "bus_fleet": {
                    "biomethane_m3": None,
                    "gnv_m3": None,
                    "cost": None
                },
                "commercialization": {
                    "electric_energy_buying": [],
                    "electric_energy_selling": [],
                    "cost": None
                },
                "biogas": {
                    "production": None,
                    "production_cost": None,
                    "used_for_engines": None,
                    "used_for_biomethane": None
                },
                "biomethane": {
                    "production": None,
                    "production_cost": None
                },
                "cvar": {
                    "cvar": None,
                    "alpha": None
                }
            },
            "constraints": {
                "equality": [],
                "inequality": []
            }
        }

        # Get solution attributes
        generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production = solution

        # Interval slots per hour
        intervals_per_hour = int(self.n_intervals / 24)

        # Calculate biogas cost per KW
        biogas_cost = ((self.biogas["maintenance_cost"][0] * biogas_production + self.biogas["maintenance_cost"][1] +
                        self.biogas["input_cost"] + self.biogas["transport_cost"]) / biogas_production)

        # Battery
        battery_energy = battery_energy.copy()
        battery_load = ([0.0] * self.n_intervals) + [self.battery["initial_load"]]
        battery_cost = [0.0] * self.n_intervals
        for j in range(0, self.n_intervals):

            # Battery load
            if battery_energy[j] > 0:
                battery_load[j] = battery_load[j - 1] + battery_energy[j] * self.battery["eff_charge"]
            else:
                battery_load[j] = battery_load[j - 1] + battery_energy[j] / self.battery["eff_discharge"]

            # Battery use cost
            battery_cost[j] = abs(battery_energy[j] * self.battery["cost"])

        expanded_solution["data"]["battery"]["energy"] = battery_energy
        expanded_solution["data"]["battery"]["load"] = battery_load[0:(len(battery_load) - 1)]
        expanded_solution["data"]["battery"]["cost"] = sum(battery_cost)
        # end: battery

        # Generators
        generators_status = copy.deepcopy(generators_status)
        generators_rate = copy.deepcopy(generators_rate)
        generators_fuel_composition = copy.deepcopy(generators_fuel_composition)
        generators_status_changed = [[] for _ in range(0, self.n_generators)]
        generators_efficiency = [[] for _ in range(0, self.n_generators)]
        generators_consumption = [[] for _ in range(0, self.n_generators)]
        generators_biogas_consumption = [[] for _ in range(0, self.n_generators)]
        generators_ethanol_consumption = [[] for _ in range(0, self.n_generators)]
        generators_biogas_rate = [[] for _ in range(0, self.n_generators)]
        generators_ethanol_rate = [[] for _ in range(0, self.n_generators)]
        generators_fuel_cost = [0 for _ in range(0, self.n_generators)]
        generators_up_down_cost = [0 for _ in range(0, self.n_generators)]

        for i in range(0, self.n_generators):

            # Status changing (Up / Down)
            status_aux = generators_status[i] + [self.generators[i]["initial_state"]]
            generators_status_changed[i] = [status_aux[j] != status_aux[j - 1] for j in range(0, self.n_intervals)]

            # Makes sure generator rate is zero if it is down (status equal to zero)
            generators_rate[i] = [generators_rate[i][j] * generators_status[i][j] for j in range(0, self.n_intervals)]

            # Efficiency
            label = [None, "efficiency_1", "efficiency_2", "efficiency_3"]
            a = [self.generators[i][label[generators_fuel_composition[i][j]]][0] for j in range(0, self.n_intervals)]
            b = [self.generators[i][label[generators_fuel_composition[i][j]]][1] for j in range(0, self.n_intervals)]
            generators_efficiency[i] = [a[j] * (generators_rate[i][j] / self.generators[i]["upper_rate"]) + b[j]
                                        for j in range(0, self.n_intervals)]

            # Fuel consumption
            generators_consumption[i] = [(generators_rate[i][j] / generators_efficiency[i][j])
                                         for j in range(0, self.n_intervals)]

            # Biogas consumption
            generators_biogas_consumption[i] = [(generators_consumption[i][j] *
                                                 (0.20 * 2 ** (generators_fuel_composition[i][j] - 1) +
                                                  0.10 * (generators_fuel_composition[i][j] - 1)))
                                                for j in range(0, self.n_intervals)]

            # Ethanol rate
            generators_ethanol_consumption[i] = [(generators_consumption[i][j] - generators_biogas_consumption[i][j])
                                                 for j in range(0, self.n_intervals)]

            # Biogas rate
            generators_biogas_rate[i] = [(generators_rate[i][j] *
                                          (0.20 * 2 ** (generators_fuel_composition[i][j] - 1) +
                                           0.10 * (generators_fuel_composition[i][j] - 1)))
                                         for j in range(0, self.n_intervals)]

            # Ethanol consumption
            generators_ethanol_rate[i] = [(generators_rate[i][j] - generators_biogas_rate[i][j])
                                          for j in range(0, self.n_intervals)]

            # Calculates fuel cost and up/down cost
            for j in range(0, self.n_intervals):

                # Fuel cost
                generators_fuel_cost[i] += (generators_biogas_consumption[i][j] * biogas_cost +
                                            generators_ethanol_consumption[i][j] * self.ethanol["cost"])

                # Up/Down cost
                if generators_status_changed[i][j] and generators_status[i][j] == 1:
                    generators_up_down_cost[i] += self.generators[i]["up_cost"]
                elif generators_status_changed[i][j] and generators_status[i][j] == 0:
                    generators_up_down_cost[i] += self.generators[i]["down_cost"]
            # end: generators cost

        expanded_solution["data"]["generators"]["stats"] = generators_status
        expanded_solution["data"]["generators"]["rate"] = generators_rate
        expanded_solution["data"]["generators"]["fuel_composition"] = generators_fuel_composition
        expanded_solution["data"]["generators"]["efficiency"] = generators_efficiency
        expanded_solution["data"]["generators"]["biogas_consumption"] = generators_biogas_consumption
        expanded_solution["data"]["generators"]["ethanol_consumption"] = generators_ethanol_consumption
        expanded_solution["data"]["generators"]["biogas_rate"] = generators_biogas_rate
        expanded_solution["data"]["generators"]["ethanol_rate"] = generators_ethanol_rate
        expanded_solution["data"]["generators"]["status_changed"] = generators_status_changed
        expanded_solution["data"]["generators"]["fuel_cost"] = generators_fuel_cost
        expanded_solution["data"]["generators"]["up_down_cost"] = generators_up_down_cost
        #end: generators

        # Calculates the amount of biogas remaining
        used_biogas = sum([sum(generators_biogas_consumption[i]) for i in range(0, self.n_generators)])
        remaining_biogas = max(0, biogas_production - used_biogas)

        expanded_solution["data"]["biogas"]["production"] = biogas_production
        expanded_solution["data"]["biogas"]["production_cost"] = biogas_cost
        expanded_solution["data"]["biogas"]["used_for_engines"] = used_biogas
        expanded_solution["data"]["biogas"]["used_for_biomethane"] = remaining_biogas

        # Calculates available biomethane and its cost
        biomethane_available = remaining_biogas * self.biomethane["efficiency"]
        biomethane_cost = (biogas_cost * remaining_biogas +
                           self.biomethane["maintenance_cost"][0] * biomethane_available +
                           self.biomethane["maintenance_cost"][1])

        expanded_solution["data"]["biomethane"]["production"] = biomethane_available
        expanded_solution["data"]["biomethane"]["production_cost"] = biomethane_cost

        # Convert available biomethane from KWh to m^3
        biomethane_available = biomethane_available / 10.92

        # Updates demand with energy required by biogas production and biomethane production
        demand = self.demand.copy()

        # Business hours (in hours and intervals of the time horizon)
        working_hours = self.business_hours["end"] - self.business_hours["start"]
        start_interval = self.business_hours["start"] * intervals_per_hour
        end_interval = self.business_hours["end"] * intervals_per_hour

        # Biogas energy consumption
        biogas_energy_consumption = ((self.biogas["consumption"][0] * biogas_production + self.biogas["consumption"][1]) /
                                     (end_interval - start_interval))

        for j in range(start_interval, end_interval):
            demand[j] = demand[j] + biogas_energy_consumption

        # Biomethane energy consumption
        total_compression_capacity = self.biomethane["compression_capacity"] * working_hours

        if biomethane_available <= total_compression_capacity:
            compression_time_start = start_interval
            compression_time_end = end_interval
            compression_intervals = compression_time_end - compression_time_start
        else:
            compression_intervals = math.ceil((biomethane_available / self.biomethane["compression_capacity"]) *
                                              intervals_per_hour)
            compression_time_start = start_interval
            compression_time_end = compression_time_start + compression_intervals
            if compression_time_end > self.n_intervals:
                compression_time_start = max(0, compression_time_start - (compression_time_end - self.n_intervals))
                compression_time_end = self.n_intervals

        biomethane_energy_consumption = ((self.biomethane["energy_consumption"] * biomethane_available) /
                                         compression_intervals)

        for j in range(compression_time_start, compression_time_end):
            demand[j] = demand[j] + biomethane_energy_consumption

        # Bus fleet
        buses_gnv_required = max(0, self.buses_demand - biomethane_available)
        buses_fuel_cost = biomethane_cost + (self.gnv["cost"] * buses_gnv_required)

        expanded_solution["data"]["bus_fleet"]["biomethane_m3"] = biomethane_available
        expanded_solution["data"]["bus_fleet"]["gnv_m3"] = buses_gnv_required
        expanded_solution["data"]["bus_fleet"]["cost"] = buses_fuel_cost

        # Electrical Energy Commercialization (buy / sell)
        commercialization_electric_energy = demand.copy()
        commercialization_electric_energy_buying = [0.0] * self.n_intervals
        commercialization_electric_energy_selling = [0.0] * self.n_intervals
        commercialization_electric_energy_cost = [0.0] * self.n_intervals
        for j in range(0, self.n_intervals):

            # Energy commercialized
            commercialization_electric_energy[j] -= self.solar_energy[j]
            commercialization_electric_energy[j] += battery_energy[j]
            for i in range(0, self.n_generators):
                commercialization_electric_energy[j] -= generators_status[i][j] * generators_rate[i][j]

            # Cost of energy commercialized
            if commercialization_electric_energy[j] > 0:
                commercialization_electric_energy_buying[j] = abs(commercialization_electric_energy[j])
                commercialization_electric_energy_cost[j] = commercialization_electric_energy[j] * self.buying_price[j]
            else:
                commercialization_electric_energy_selling[j] = abs(commercialization_electric_energy[j])
                commercialization_electric_energy_cost[j] = commercialization_electric_energy[j] * self.selling_price[j]

        expanded_solution["data"]["commercialization"]["electric_energy_buying"] = commercialization_electric_energy_buying
        expanded_solution["data"]["commercialization"]["electric_energy_selling"] = commercialization_electric_energy_selling
        expanded_solution["data"]["commercialization"]["cost"] = sum(commercialization_electric_energy_cost)
        # end: electrical energy commercialization

        # Calculates total cost
        total_cost = (sum(commercialization_electric_energy_cost) + sum(battery_cost) + buses_fuel_cost +
                      sum(generators_up_down_cost) + sum(generators_fuel_cost))

        expanded_solution["data"]["total_cost"] = total_cost
        expanded_solution["data"]["electric_energy_demand"] = demand

        # -----------------------------------------------------------
        # CONSTRAINTS

        expanded_solution["constraints"]["equality"] = []
        expanded_solution["constraints"]["inequality"] = []

        # Constraint: ramp rate limit
        for i in range(0, self.n_generators):
            rate = generators_rate[i] + [self.generators[i]["initial_state"] * self.generators[i]["lower_rate"]]
            status = generators_status[i] + [self.generators[i]["initial_state"]]
            diff = [abs((rate[j] * status[j]) - (rate[j - 1] * status[j - 1])) for j in range(0, self.n_intervals)]

            # Ramp rate constraints: (diff - limit) * <= 0, if generator is up
            constraints = [(diff[j] - self.generators[i]["ramp_rate_limit"]) * int(status[j])
                           for j in range(0, self.n_intervals)]

            expanded_solution["constraints"]["inequality"] += constraints
        # end: ramp rate limit

        # Constraint: minimum window length without generator status changing
        constraints_generator_window = []
        for i in range(0, self.n_generators):
            status_changed = generators_status_changed[i]

            for j in range(0, (len(status_changed) - self.generators[i]["up_down_window"] + 1)):
                start_window = j
                end_window = j + self.generators[i]["up_down_window"]

                # Window constraint: changes - 1 <= 0
                constraint = sum(status_changed[start_window:end_window]) - 1
                expanded_solution["constraints"]["inequality"].append(constraint)
        # end: windows minimum length

        # Constraint: fuel available (used - available <= 0)
        total_used_ethanol = sum([sum(generators_ethanol_consumption[i]) for i in range(0, self.n_generators)])
        expanded_solution["constraints"]["inequality"].append(used_biogas - biogas_production)
        expanded_solution["constraints"]["inequality"].append(total_used_ethanol - self.ethanol["disponibility"])
        # end: fuel available

        # Constraint: maximum and minimum battery load
        battery_free_load = [(self.battery["max_load"] - battery_load[i - 1]) / self.battery["eff_charge"]
                             for i in range(0, self.n_intervals)]
        battery_available_load = [battery_load[i - 1] * self.battery["eff_discharge"]
                                  for i in range(0, self.n_intervals)]

        constraint_battery1 = 0
        constraint_battery2 = 0
        for i in range(0, self.n_intervals):
            if battery_energy[i] < 0:
                constraint_battery1 += max((abs(battery_energy[i]) - battery_available_load[i]), 0)
            else:
                constraint_battery1 += max((battery_energy[i] - battery_free_load[i]), 0)

            constraint_battery2 += sum(
                [(abs(battery_load[i]) if (battery_load[i] < 0 or battery_load[i] > self.battery["max_load"]) else 0)
                 for i in range(0, self.n_intervals)])

        expanded_solution["constraints"]["inequality"].append(constraint_battery1)
        expanded_solution["constraints"]["inequality"].append(constraint_battery2)
        # end: battery limit

        # Constraints: CVaR (if enabled)
        if self.enable_cvar:
            z = (biogas_production - self.biogas["mean_production"]) / self.biogas["deviation_production"]
            alpha = norm.cdf(z)
            biogas_cvar = abs(alpha ** -1 * norm.pdf(z) * self.biogas["deviation_production"] - self.biogas["mean_production"])
            diff_biogas = self.biogas["mean_production"] - biogas_cvar
            cvar = diff_biogas * 0.27 * self.buying_price[40]

            expanded_solution["data"]["cvar"]["cvar"] = cvar
            expanded_solution["data"]["cvar"]["alpha"] = alpha

            constraint_cvar = min((cvar - self.epsilon * 82), 0)
            expanded_solution["constraints"]["inequality"].append(constraint_cvar)
        # end: CVaR

        # Return the expanded solution
        return expanded_solution
