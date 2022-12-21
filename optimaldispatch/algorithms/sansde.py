import signal
import math
import random
import time
import numpy as np
from ..utils.parameters import Parameters


# Exports
__all__ = ["SaNSDE"]


########################################################################################################################
# Global variables and functions to handle system events (signals)

keyboard_interrupt = False

keyboard_interrupt_handler_SIGBREAK = signal.getsignal(signal.SIGBREAK)
keyboard_interrupt_handler_SIGINT = signal.getsignal(signal.SIGINT)


def on_system_signal(signum, stack):
    global keyboard_interrupt
    if signum == signal.SIGBREAK or signum == signal.SIGINT:
        print("Interrupt signal received... wait while the program ends.")
        print("To force it to end, press CTRL+C again.")
        keyboard_interrupt = True
        signal.signal(signal.SIGBREAK, keyboard_interrupt_handler_SIGBREAK)
        signal.signal(signal.SIGINT, keyboard_interrupt_handler_SIGINT)


########################################################################################################################
def SaNSDE(feval, var_lb, var_ub, params=Parameters(), seed=None, verbosity=0, draw_function=None, draw_frequency=0):
    """
    Implements the Self-adaptive Differential Evolution with Neighborhood Search [1].

    [1] YANG, Z.; TANG, K. ; YAO, X. Self-adaptive Differential Evolution with Neighborhood Search. IEEE Congress on
        Evolutionary Computation (IEEE World Congress on Computational Intelligence), 2008.
    """

    # Set signal handler
    global keyboard_interrupt
    signal.signal(signal.SIGBREAK, on_system_signal)
    signal.signal(signal.SIGINT, on_system_signal)

    # Verbosity parameters
    verbose = (verbosity > 0)
    verbose_frequency = verbosity

    # Stop criteria parameters
    iteration_limit = params.get("iteration-limit", 100, float)
    time_limit = params.get("time-limit", math.inf, float)

    # Algorithm parameters
    population_size = params.get("population-size", 30, int)
    mutation_learning_period = params.get("mutation:learning-period", 50, int)
    crossover_learning_period = params.get("crossover:learning-period", 25, int)
    crossover_update_frequency = params.get("crossover:update-frequency", 5, int)
    selection_strategy = params.get("selection:strategy", "dynamic", str)  # dynamic, fitness, stochastic-rank
    exploitation_frequency = params.get("selection:exploitation:frequency", 500, int)
    exploitation_iterations = params.get("selection:exploitation:iterations", 50, int)
    stochastic_rank_threshold = params.get("selection:stochastic-rank:threshold", 0.3, float)
    fitness_g = params.get("selection:fitness:g", 30, float)
    fitness_h = params.get("selection:fitness:h", 50, float)

    # Initialize the random number generators
    random.seed(seed)
    np.random.seed(seed)

    # Self-adaptive variables for scaling factor parameter
    fp = 0.5
    ns1_fp = 0
    ns2_fp = 0
    nf1_fp = 0
    nf2_fp = 0

    # Self-adaptive variables for mutation step
    p = 0.5
    ns1_p = 0
    ns2_p = 0
    nf1_p = 0
    nf2_p = 0

    # Self-adaptive variables for crossover step
    CR_rec = []
    f_rec = []
    CRm = 0.5
    CR = [random.gauss(CRm, 0.1) for i in range(0, population_size)]

    # Start the timer
    time_begin = time.perf_counter()

    # Initialize population
    population_x = [_create_random_solution(var_lb, var_ub) for _ in range(0, population_size)]
    population_f = [feval(x) for x in population_x]
    best_idx, _, _, _, _ = _find_best(population_f)

    # Log: initial population
    runtime = time.perf_counter() - time_begin
    _log_header(verbose)
    _log_status(0, runtime, population_x, population_f, False, verbose)

    # Drawing
    if draw_function is not None and draw_frequency > 0:
        draw_function(population_x[best_idx], population_x)

    # Counters
    iteration = 0
    exploration_counter = 0
    exploitation_counter = 0
    perform_exploitation = False

    # Main loop
    while iteration < iteration_limit and time.perf_counter() - time_begin < time_limit and not keyboard_interrupt:

        # Increment the iteration counter
        iteration += 1

        if perform_exploitation:
            exploitation_counter += 1
        else:
            exploration_counter += 1

        # Create the offspring population
        offspring_x = [[]] * population_size
        offspring_f = [[]] * population_size

        for i in range(0, population_size):

            # Select three solutions from the population for mutation step
            idx1, idx2, idx3 = random.choices(range(0, population_size), k=3)

            # Scaling factor
            used1_fp = (random.random() < fp)
            scaling = random.gauss(0.5, 0.3) if used1_fp else float(np.random.standard_cauchy(1))

            # Mutation
            used1_p = random.random() < p
            if used1_p:
                diff_x = _mutation_DE_rand_1(population_x[idx1], population_x[idx2], population_x[idx3],
                                             scaling, var_lb, var_ub)
            else:
                diff_x = _mutation_DE_best_2(population_x[best_idx], population_x[i], population_x[idx1],
                                             population_x[idx2], scaling, var_lb, var_ub)

            # Crossover
            trial_x = _crossover_binomial(population_x[i], diff_x, CR[i])

            # Evaluate solution obtained from crossover
            trial_f = feval(trial_x)

            # Selection
            if selection_strategy == "dynamic":
                if perform_exploitation:
                    accepted = _acceptance_by_fitness(trial_f, population_f[i], fitness_g, fitness_h)
                else:
                    accepted = _acceptance_by_stochastic_rank(trial_f, population_f[i], stochastic_rank_threshold)

            elif selection_strategy == "fitness":
                accepted = _acceptance_by_fitness(trial_f, population_f[i], fitness_g, fitness_h)

            elif selection_strategy == "stochastic-rank":
                accepted = _acceptance_by_stochastic_rank(trial_f, population_f[i], stochastic_rank_threshold)
            else:
                raise ValueError("Invalid value for 'selection:strategy' parameter")

            # Update offspring population and learning variables
            if accepted:
                offspring_x[i] = trial_x
                offspring_f[i] = trial_f

                if used1_fp:
                    ns1_fp += 1
                else:
                    ns2_fp += 1

                if used1_p:
                    ns1_p += 1
                else:
                    ns2_p += 1

                CR_rec.append(CR[i])
                f_rec.append(population_f[i][0] - trial_f[0])

            else:
                offspring_x[i] = population_x[i]
                offspring_f[i] = population_f[i]

                if used1_fp:
                    nf1_fp += 1
                else:
                    nf2_fp += 1

                if used1_p:
                    nf1_p += 1
                else:
                    nf2_p += 1

        # Make sure that the best solution won't be lost
        best_idx, _, _, _, _ = _find_best(population_f)
        worst_idx, _, _, _, _ = _find_worst(offspring_f)

        offspring_x[worst_idx] = population_x[best_idx]
        offspring_f[worst_idx] = population_f[best_idx]

        # Update main population and best solution
        population_x = offspring_x
        population_f = offspring_f
        best_idx, _, _, _, _ = _find_best(population_f)

        # Mutation learning period
        if iteration % mutation_learning_period == 0:

            # Update self-adaptive parameters
            try:
                fp = (ns1_fp * (ns2_fp + nf2_fp)) / (ns2_fp * (ns1_fp + nf1_fp) + ns1_fp * (ns2_fp + nf2_fp))
            except ZeroDivisionError:
                fp = 0.5

            try:
                p = (ns1_p * (ns2_p + nf2_p)) / (ns2_p * (ns1_p + nf1_p) + ns1_p * (ns2_p + nf2_p))
            except ZeroDivisionError:
                p = 0.5

            # Reset learning variables
            ns1_fp = 0
            ns2_fp = 0
            nf1_fp = 0
            nf2_fp = 0
            ns1_p = 0
            ns2_p = 0
            nf1_p = 0
            nf2_p = 0

        # Crossover leaning period
        if iteration % crossover_learning_period == 0:
            if len(CR_rec) > 0:
                sum_f_rec = sum(f_rec)
                CRm = sum([(f_rec[i] / sum_f_rec) * CR_rec[i] for i in range(0, len(CR_rec))])
                CR_rec.clear()
                f_rec.clear()

        # Update CR values
        if iteration % crossover_update_frequency == 0:
            CR = [random.gauss(CRm, 0.1) for _ in range(0, population_size)]

        # Log: current iteration
        if verbose and (iteration % verbose_frequency == 0 or iteration == iteration_limit):
            runtime = time.perf_counter() - time_begin
            _log_status(iteration, runtime, population_x, population_f, perform_exploitation, verbose)

        # Plot
        if draw_function is not None and draw_frequency > 0:
            if iteration % draw_frequency == 0:
                draw_function(population_x[best_idx], population_x)

        # Change between exploration and exploitation phases, if dynamic selection is set
        if selection_strategy == "dynamic":
            if perform_exploitation:
                if exploitation_counter >= exploitation_iterations:
                    perform_exploitation = False
                    exploration_counter = 0
            else:
                if exploration_counter >= exploitation_frequency:
                    perform_exploitation = True
                    exploitation_counter = 0

    # end: main loop

    # Log: footer
    _log_footer(verbose)

    # Find best solution
    best_idx, _, _, _, _ = _find_best(population_f)

    # Compute runtime
    runtime = time.perf_counter() - time_begin

    # Return the best solution found, the final population, number of iterations and runtime
    return population_x[best_idx], population_f[best_idx], population_x, population_f, iteration, runtime


########################################################################################################################
def _create_random_solution(var_lb, var_ub):
    nvars = len(var_ub)
    solution = [random.uniform(var_lb[i], var_ub[i]) for i in range(0, nvars)]
    return solution


########################################################################################################################
def _fitness(f, g, h, g_weight, h_weight, eps=1e-5):
    return f + g_weight * sum([max(0, x - eps) for x in g]) + h_weight * sum([max(0, abs(x) - eps) for x in h])


########################################################################################################################
def _mutation_DE_rand_1(x1, x2, x3, scaling, var_lb, var_ub):
    nvar = len(var_lb)
    x_diff = [max(var_lb[j], min(var_ub[j], x1[j] + scaling * (x2[j] - x3[j]))) for j in range(0, nvar)]
    return x_diff


########################################################################################################################
def _mutation_DE_best_2(xbest, xi, x1, x2, scaling, var_lb, var_ub):
    nvar = len(var_lb)
    x_diff = [max(var_lb[j], min(var_ub[j], xi[j] + scaling * ((xbest[j] - xi[j]) + (x1[j] - x2[j]))))
              for j in range(0, nvar)]
    return x_diff


########################################################################################################################
def _crossover_binomial(base, diff, crossover_constant):
    nvar = len(base)
    l = random.randint(0, nvar - 1)
    cross = [diff[j] if random.random() <= crossover_constant or j == l else base[j] for j in range(0, nvar)]
    return cross


########################################################################################################################
def _crossover_arithmetic(base, diff):
    nvar = len(base)
    lambda_val = [random.random() for _ in range(0, nvar)]
    cross = [(1 - lambda_val[j]) * base[j] + lambda_val[j] * diff[j] for j in range(0, nvar)]
    return cross


########################################################################################################################
def _crossover_sbx(base, diff, eta):
    nvar = len(base)
    lambda_val = [random.random() for j in range(0, nvar)]
    beta = [0] * nvar

    for j in range(0, nvar):
        if lambda_val[j] <= 0.5:
            beta[j] = (2 * lambda_val[j]) ** (1 / (eta + 1))
        else:
            beta[j] = (2 * (1 - lambda_val[j])) ** (1 / (eta + 1))

    a = [0] * nvar
    b = [0] * nvar
    for j in range(0, nvar):
        if random.random() <= 0.5:
            a[j] = base[j]
            b[j] = diff[j]
        else:
            a[j] = diff[j]
            b[j] = base[j]

    cross = [0] * nvar
    for j in range(0, nvar):
        cross[j] = ((1 - beta[j]) * a[j] + (1 - beta[j]) * b[j]) / 2

    # Return the solution
    return cross


########################################################################################################################
def _acceptance_by_fitness(trial, current, g_weight, h_weight, eps=1e-5):
    return _fitness(*trial, g_weight, h_weight, eps) < _fitness(*current, g_weight, h_weight, eps)


########################################################################################################################
def _acceptance_by_stochastic_rank(trial, current, threshold, eps=1e-5):

    # Calculate the infeasibility
    trial_inf = sum([max(0, value - eps) for value in trial[1]]) + sum([max(0, abs(value) - eps) for value in trial[2]])
    current_inf = sum([max(0, value - eps) for value in current[1]]) + sum([max(0, abs(value) - eps) for value in current[2]])

    if random.random() <= threshold:
        return (trial_inf, trial[0]) < (current_inf, current[0])
    else:
        return (trial[0], trial_inf) < (current[0], current_inf)


########################################################################################################################
def _find_best(population_f, eps=1e-5):

    best_idx = 0
    best_obj, best_g, best_h = population_f[0]
    best_inf = sum([max(0, value - eps) for value in best_g]) + sum([max(0, abs(value) - eps) for value in best_h])

    for i in range(1, len(population_f)):

        obj, g, h = population_f[i]
        inf = sum([max(0, value - eps) for value in g]) + sum([max(0, abs(value) - eps) for value in h])
        if (inf, obj) < (best_inf, best_obj):
            best_idx = i
            best_obj = obj
            best_inf = inf
            best_g = g
            best_h = h

    return best_idx, best_obj, best_g, best_h, best_inf


########################################################################################################################
def _find_worst(population_f, eps=1e-5):

    worst_idx = 0
    worst_obj, worst_g, worst_h = population_f[0]
    worst_inf = sum([max(0, value - eps) for value in worst_g]) + sum([max(0, abs(value) - eps) for value in worst_h])

    for i in range(1, len(population_f)):

        obj, g, h = population_f[i]
        inf = sum([max(0, value - eps) for value in g]) + sum([max(0, abs(value) - eps) for value in h])
        if (inf, obj) > (worst_inf, worst_obj):
            worst_idx = i
            worst_obj = obj
            worst_inf = inf
            worst_g = g
            worst_h = h

    return worst_idx, worst_obj, worst_g, worst_h, worst_inf


########################################################################################################################
def _log_header(verbose):
    if verbose:
        print("------------------------------------------------------------------------------------------------")
        print("|           |           BEST SOLUTION           |          AVG. POPULATION          |          |")
        print("| ITERATION |-----------------------------------|-----------------------------------| TIME (S) |")
        print("|           |    OBJECTIVE    |   VIOLATIONS    |    OBJECTIVE    |   VIOLATIONS    |          |")
        print("------------------------------------------------------------------------------------------------")


########################################################################################################################
def _log_footer(verbose):
    if verbose:
        print("------------------------------------------------------------------------------------------------\n")


########################################################################################################################
def _log_status(iteration, runtime, population_x, population_f, from_exploitation, verbose):
    if verbose:

        # Population size
        npop = len(population_x)

        # Find the best solution
        best_idx, best_obj, best_g, best_h, best_inf = _find_best(population_f)

        # Sum of objectives and violations
        avg_obj = 0.0
        avg_inf = 0.0

        for i in range(0, npop):

            obj, g, h = population_f[i]
            inf = sum([max(0, value - 1e-5) for value in g]) + sum([max(0, abs(value) - 1e-5) for value in h])

            avg_obj += obj
            avg_inf += inf

        # Compute average objectives and violations
        avg_obj /= npop
        avg_inf /= npop

        # Print log
        print("| {}{:8} | {:15.4f} | {:15.4f} | {:15.4f} | {:15.4f} | {:8.2f} |".format(
            ">" if from_exploitation else " ", iteration, best_obj, best_inf, avg_obj, avg_inf, runtime))
