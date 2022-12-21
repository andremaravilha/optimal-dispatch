import argparse
import json
import matplotlib.pyplot as plt
import optimaldispatch as opt


def _optimize(args):
    """
    Callback for "optimize" command.
    """

    # Get general command line parameters
    instance_file = args.instance
    params_file = args.params_file
    solution_path = args.solution
    chart_path = args.chart
    verbosity = args.verbosity
    draw_frequency = args.draw_frequency
    epsilon = args.cvar_epsilon
    seed = args.seed

    # Load the Optimal Dispatch problem from instance file
    problem = opt.OptimalDispatch(instance_file, cvar_epsilon=epsilon)

    # Encode the problem to be solved with the Differential Evolution
    f, x_lb, x_ub = opt.utils.encode_problem(problem)

    # Create a drawing function, if enabled
    draw_function = None
    if draw_frequency > 0:

        # Draw best solution
        draw_function = lambda best, population: opt.utils.draw_solution(problem, opt.utils.decode_solution(problem, best),
                                                                         block=False, show=True, interactive=True)

        # # Draw variables convergence
        # draw_function = lambda best, population: opt.utils.draw_convergence(best, population, x_lb, x_ub, block=False,
        #                                                                     show=True, interactive=True)

    # Algorithm's parameters
    params = opt.utils.Parameters()

    # Load parameters from file, if any
    if params_file is not None:
        for line in params_file:
            key, value = [token.strip() for token in line.strip().split("=")]
            params[key] = value

    # Load parameters from command line
    if args.params is not None:
        for key, value in args.params:
            params[key] = value

    # Solve the problem
    x_best, f_best, pop_x, pop_f, iterations, runtime = \
        opt.algorithms.SaNSDE(f, x_lb, x_ub, params=params, seed=seed, verbosity=verbosity,
                              draw_function=draw_function, draw_frequency=draw_frequency)

    solution = opt.utils.decode_solution(problem, x_best)
    f_val, g_val, h_val = problem.evaluate(solution)
    g_inf, h_inf = problem.calculate_infeasibility(g_val, h_val)

    # Print result
    print("Cost: {:.4f}".format(f_val))
    print("Iterations: {}".format(iterations))
    print("Runtime (seconds): {:.4f}".format(runtime))
    print("Infeasibility (inequality constraints): {:.4f}".format(sum(g_inf)))
    print("Infeasibility (equality constraints): {:.4f}".format(sum(h_inf)))

    # Write best solution to JSON file
    with open(solution_path, "w") as solution_file:
        problem.solution_to_json(solution, solution_file)

    # Save figure in a file
    if chart_path is not None:
        problem.draw_solution(solution, block=False, show=False, interactive=False)
        plt.savefig(chart_path, dpi=600)

    # Plot the final best solution found
    if draw_function:
        problem.draw_solution(solution, block=True, show=True, interactive=True)


def _draw(args):
    """
    Callback for "draw" command.
    """

    # Get parameters
    instance_file = args.instance
    solution_file = args.solution
    chart_file = args.chart_file

    # Load solution data from JSON
    solution_json = json.loads(solution_file.read())

    if solution_json["cvar"]["enabled"]:
        epsilon = solution_json["cvar"]["epsilon"]
    else:
        epsilon = None

    generators_status = [generator["status"] for generator in solution_json["generators"]]
    generators_rate = [generator["rate"] for generator in solution_json["generators"]]
    generators_fuel_composition = [generator["fuel_composition"] for generator in solution_json["generators"]]
    battery_energy = solution_json["battery_energy"]
    biogas_production = solution_json["biogas_production"]

    # Build problem instance and solution
    problem = opt.OptimalDispatch(instance_file, cvar_epsilon=epsilon)
    solution = (generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production)

    # Save as figure in a file
    if chart_file is not None:
        problem.draw_solution(solution, block=False, show=False, interactive=False)
        plt.savefig(chart_file, dpi=600)

    # Just show the figure on screen
    else:
        problem.draw_solution(solution, block=True, show=True, interactive=False)


def _create_instance(args):
    """
    Callback for "create-instance" command .
    """
    output_file = args.output
    demand_file = args.demand
    generators_file = args.generators
    battery_file = args.battery
    fuel_file = args.fuel
    solar_file = args.solar
    opt.utils.create_instance(output_file, demand_file, generators_file, battery_file, fuel_file, solar_file)


def _main():
    """
    Main function.
    """

    # Create command line argument parser
    parser = argparse.ArgumentParser(prog="optimaldispatch", description="Optimal dispatch in a microgrid")
    subparser = parser.add_subparsers(title="subcommands")

    # Sub-parser: optimize
    parser_optimize = subparser.add_parser("optimize", help="solve an instance of the problem")
    parser_optimize.set_defaults(func=_optimize)
    parser_optimize.add_argument("--instance", metavar="FILE", dest="instance", type=argparse.FileType('r'),
                                 help="Instance file in .json format.", required=True)
    parser_optimize.add_argument("--solution", metavar="FILE", dest="solution", type=str,
                                 help="JSON file in which the final solution will be written.", required=True)
    parser_optimize.add_argument("--chart", metavar="FILE", dest="chart", type=str, default=None,
                                 help="Image file in which the final solution will be drawn. Supported format are JPG, "
                                      "PNG, PDF and EPS.")
    parser_optimize.add_argument("--cvar", metavar="EPSILON", dest="cvar_epsilon", type=float, default=None,
                                 help="If set, CVaR constraints are enabled as an e-constraint with the specified "
                                      "value for epsilon.")
    parser_optimize.add_argument("--verbosity", metavar="VALUE", dest="verbosity", type=int, default=1,
                                 help="Set the frequency (interval of iterations) in which the optimization progress "
                                      "is displayed on screen. If set to 0 (zero), only the final result is displayed.")
    parser_optimize.add_argument("--draw-frequency", metavar="VALUE", dest="draw_frequency", type=int, default=0,
                                 help="Set the frequency (interval of iterations) the chart from the current best "
                                      "solution is updated. If set to 0 (zero), no chart is displayed.")
    parser_optimize.add_argument("--seed", metavar="VALUE", dest="seed", type=int, default=0,
                                 help="Seed to initialize the random number generator.")
    parser_optimize.add_argument("--param", metavar="VALUE", dest="params", nargs=2, action="append",
                                 help="Set algorithm parameters. It expects two values, the first is the name of the "
                                      "parameter and the second is the value. This option can be used more than once. "
                                      "If --param-file is set, parameters set with --param overwrite the values set on "
                                      "the file.")
    parser_optimize.add_argument("--param-file", metavar="FILE", dest="params_file", type=argparse.FileType('r'),
                                 default=None,
                                 help="Text file with algorithm parameters.")

    # Sub-parser: draw
    parser_plot = subparser.add_parser("draw", help="draw a solution")
    parser_plot.set_defaults(func=_draw)
    parser_plot.add_argument("--instance", metavar="FILE", dest="instance", type=argparse.FileType('r'),
                             help="Instance file in .json format.", required=True)
    parser_plot.add_argument("--solution", metavar="FILE", dest="solution", type=argparse.FileType('r'),
                             help="Solution file in .json format.", required=True)
    parser_plot.add_argument("--chart", metavar="FILE", dest="chart_file", type=str, default=None,
                             help="Image file in which the final solution will be drawn. Supported format are JPG, "
                                  "PNG, PDF and EPS. If not set, the final solution is only shown on screen.")

    # Sub-parser: instance creation
    parser_instance = subparser.add_parser("create-instance", help="create .json instance from .csv files")
    parser_instance.set_defaults(func=_create_instance)
    parser_instance.add_argument("--output", metavar="FILE", dest="output", type=argparse.FileType('w'),
                                 help="Output file to write instance data in JSON format.", required=True)
    parser_instance.add_argument("--demand", metavar="FILE", dest="demand", type=argparse.FileType('r'),
                                 help="Demand file in CSV format.", required=True)
    parser_instance.add_argument("--generators", metavar="FILE", dest="generators", type=argparse.FileType('r'),
                                 help="Generators file in CSV format.", required=True)
    parser_instance.add_argument("--battery", metavar="FILE", dest="battery", type=argparse.FileType('r'),
                                 help="Battery file in CSV format.", required=True)
    parser_instance.add_argument("--fuel", metavar="FILE", dest="fuel", type=argparse.FileType('r'),
                                 help="Fuel file in CSV format.", required=True)
    parser_instance.add_argument("--solar", metavar="FILE", dest="solar", type=argparse.FileType('r'),
                                 help="Solar file in CSV format.", required=True)

    # Parse command line arguments
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
