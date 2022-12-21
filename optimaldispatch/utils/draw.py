import matplotlib.pyplot as plt
from .radar_chart import radar_factory


# Exports
__all__ = ["draw_convergence", "draw_solution"]


def draw_convergence(best, population, lb, ub, block=False, show=True, interactive=True):

    nvar = len(lb)
    nsol = len(population)

    data_pop = [[population[j][i] for j in range(0, nsol)] for i in range(0, nvar)]
    data = [(max(x) - min(x)) / ((u - l) if (u - l) > 0 else 1) for x, l, u in zip(data_pop, lb, ub)]

    if interactive:
        plt.ion()
    else:
        plt.ioff()

    theta = radar_factory(nvar, frame="polygon")
    fig, axes = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'), num="foo", clear=True)
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    ax = fig.axes[0]

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title("Convergence", weight="bold", size="medium", position=(0.5, 1.1), horizontalalignment="center", verticalalignment="center")
    ax.plot(theta, data, color="tab:blue")
    ax.fill(theta, data, alpha=0.25)
    #ax.plot(theta, [(best[i] - lb[i]) / ((ub[i] - lb[i]) if (ub[i] - lb[i]) > 0 else 1) for i in range(0, nvar)], color="tab:red")

    if show:
        plt.show(block=block)
        if not block:
            plt.pause(0.1)


def draw_solution(problem, solution, block=False, show=True, interactive=True):
    """
    Draw the best solution in the population.
    :param problem: an instance of the Optimal Dispatch problem.
    :param solution: solution to draw
    """
    problem.draw_solution(solution, block=block, show=show, interactive=interactive)
