"""Implementation of the Multiswarm Particle Swarm Optimization algorithm as
presented in *Blackwell, Branke, and Li, 2008, Particle Swarms for Dynamic
Optimization Problems.*
"""
import itertools
import datetime
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

COLOR = 3
#plt.rcParams["figure.figsize"] = (20, 8)
if(COLOR == 1):
    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = "cornsilk"
    plt.rcParams["savefig.facecolor"] = "#1c1c1c"
elif(COLOR == 2):
    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = "#1c1c1c"
    plt.rcParams["savefig.facecolor"] = "#1c1c1c"
elif(COLOR == 3):
    plt.rcParams["axes.facecolor"] = "white"



try:
    from itertools import imap
except:
    # Python 3 nothing to do
    pass
else:
    map = imap

from deap import base
from deap.benchmarks import movingpeaks
from deap import creator
from deap import tools

NDIM = 5
scenario = movingpeaks.SCENARIO_1
scenario["period"] = 1000000
scenario["npeaks"] = 1
scenario["uniform_height"] = 20
scenario["move_severity"] = 2
scenario["min_height"] = 1
scenario["max_height"] = 42
BOUNDS = [scenario["min_coord"], scenario["max_coord"]]
mpb = movingpeaks.MovingPeaks(dim=NDIM, **scenario)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    best=None, bestfit=creator.FitnessMax)
creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

def generate(pclass, dim, pmin, pmax, smin, smax):
    part = pclass(random.uniform(pmin, pmax) for _ in range(dim))
    part.speed = [random.uniform(smin, smax) for _ in range(dim)]
    return part

def convertQuantum(swarm, rcloud, centre, dist):
    dim = len(swarm[0])
    for part in swarm:
        position = [random.gauss(0, 1) for _ in range(dim)]
        dist = math.sqrt(sum(x**2 for x in position))

        if dist == "gaussian":
            u = abs(random.gauss(0, 1.0/3.0))
            part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

        elif dist == "uvd":
            u = random.random()
            part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

        elif dist == "nuvd":
            u = abs(random.gauss(0, 1.0/3.0))
            part[:] = [(rcloud * x * u / dist) + c for x, c in zip(position, centre)]

        del part.fitness.values
        del part.bestfit.values
        part.best = None

    return swarm

def updateParticle(part, best, chi, c):
    ce1 = (c * random.uniform(0, 1) for _ in range(len(part)))
    ce2 = (c * random.uniform(0, 1) for _ in range(len(part)))
    ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
    ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
    a = map(operator.sub,
                      map(operator.mul,
                                    itertools.repeat(chi),
                                    map(operator.add, ce1_p, ce2_g)),
                      map(operator.mul,
                                     itertools.repeat(1 - chi),
                                     part.speed))
    part.speed = list(map(operator.add, part.speed, a))
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, creator.Particle, dim=NDIM,
    pmin=BOUNDS[0], pmax=BOUNDS[1], smin=-(BOUNDS[1] - BOUNDS[0])/2.0,
    smax=(BOUNDS[1] - BOUNDS[0])/2.0)
toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
toolbox.register("update", updateParticle, chi=0.729843788, c=2.05)
toolbox.register("convert", convertQuantum, dist="nuvd")

def main(verbose=True):
    NEXCESS = 3
    RCLOUD = 0.5    # 0.5 times the move severity
    POPSIZE = 50
    GEN = int(input("Generations: "))
    ITER = int(input("Iterations: "))
    POPSIZE=50
    verbose=True

    SWARMSIZE = 1
    bestFitness = [ [] for i in range(ITER)]
    algorithmName = "MPSO"
    benchmark = mpb
    benchmarkName = "Moving Peak"


    toolbox.register("evaluate", benchmark)
    for it in range(ITER):


        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "nswarm", "evals", "error", "offline_error", "avg", "max"


        # Generate the initial population
        population = [toolbox.swarm(n=POPSIZE) for _ in range(SWARMSIZE)]
        # Evaluate each particle
        for swarm in population:
            for part in swarm:
                part.fitness.values = toolbox.evaluate(part)

                # Update swarm's attractors personal best and global best
                if not part.best or part.fitness > part.bestfit:
                    part.best = toolbox.clone(part[:])          # Get the position
                    part.bestfit.values = part.fitness.values   # Get the fitness
                if not swarm.best or part.fitness > swarm.bestfit:
                    swarm.best = toolbox.clone(part[:])         # Get the position
                    swarm.bestfit.values = part.fitness.values  # Get the fitness

        record = stats.compile(itertools.chain(*population))
        logbook.record(gen=0, evals=mpb.nevals, nswarm=len(population),
                       error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

        if verbose:
            print(logbook.stream)

        generation = 1
        for g in range(GEN):
            # Check for convergence
            rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * len(population)**(1.0/NDIM))

            not_converged = 0
            worst_swarm_idx = None
            worst_swarm = None
            for i, swarm in enumerate(population):
                # Compute the diameter of the swarm
                for p1, p2 in itertools.combinations(swarm, 2):
                    d = math.sqrt(sum((x1 - x2)**2. for x1, x2 in zip(p1, p2)))
                    if d > 2*rexcl:
                        not_converged += 1
                        # Search for the worst swarm according to its global best
                        if not worst_swarm or swarm.bestfit < worst_swarm.bestfit:
                            worst_swarm_idx = i
                            worst_swarm = swarm
                        break

            # If all swarms have converged, add a swarm
            if not_converged == 0:
                population.append(toolbox.swarm(n=POPSIZE))
            # If too many swarms are roaming, remove the worst swarm
            elif not_converged > NEXCESS:
                population.pop(worst_swarm_idx)

            # Update and evaluate the swarm
            for swarm in population:
                # Check for change
                if swarm.best and toolbox.evaluate(swarm.best) != swarm.bestfit.values:
                    # Convert particles to quantum particles
                    swarm[:] = toolbox.convert(swarm, rcloud=RCLOUD, centre=swarm.best)
                    swarm.best = None
                    del swarm.bestfit.values

                for part in swarm:
                    # Not necessary to update if it is a new swarm
                    # or a swarm just converted to quantum
                    if swarm.best and part.best:
                        toolbox.update(part, swarm.best)
                    part.fitness.values = toolbox.evaluate(part)

                    # Update swarm's attractors personal best and global best
                    if not part.best or part.fitness > part.bestfit:
                        part.best = toolbox.clone(part[:])
                        part.bestfit.values = part.fitness.values
                    if not swarm.best or part.fitness > swarm.bestfit:
                        swarm.best = toolbox.clone(part[:])
                        swarm.bestfit.values = part.fitness.values

            record = stats.compile(itertools.chain(*population))
            logbook.record(gen=g, evals=mpb.nevals, nswarm=len(population),
                           error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

            if verbose:
                print(logbook.stream)

            # Apply exclusion
            reinit_swarms = set()
            for s1, s2 in itertools.combinations(range(len(population)), 2):
                # Swarms must have a best and not already be set to reinitialize
                if population[s1].best and population[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
                    dist = 0
                    for x1, x2 in zip(population[s1].best, population[s2].best):
                        dist += (x1 - x2)**2.
                    dist = math.sqrt(dist)
                    if dist < rexcl:
                        if population[s1].bestfit <= population[s2].bestfit:
                            reinit_swarms.add(s1)
                        else:
                            reinit_swarms.add(s2)

            # Reinitialize and evaluate swarms
            for s in reinit_swarms:
                population[s] = toolbox.swarm(n=POPSIZE)
                for part in population[s]:
                    part.fitness.values = toolbox.evaluate(part)

                    # Update swarm's attractors personal best and global best
                    if not part.best or part.fitness > part.bestfit:
                        part.best = toolbox.clone(part[:])
                        part.bestfit.values = part.fitness.values
                    if not population[s].best or part.fitness > population[s].bestfit:
                        population[s].best = toolbox.clone(part[:])
                        population[s].bestfit.values = part.fitness.values
            generation += 1


        bestFitness[it] = logbook.select("error")

    bMean = [sum(sub_list) / len(sub_list) for sub_list in zip(*bestFitness)]
    bStd = np.std(bMean)
    gen = logbook.select("gen")

    cDate = datetime.datetime.now()
    year = cDate.year
    month = cDate.month
    day = cDate.day
    hour = cDate.hour
    fig, ax = plt.subplots(1)
    ax.plot(gen, bMean, color="green", label="MPSO - SP")
    ax.fill_between(gen, bMean - bStd, bMean + bStd, color="darkgreen", alpha=0.1)
    #ax.fill_between(gen, bMean - 2*bStd, bMean + 2*bStd, color='#888888', alpha=0.2)
    #for it in range(ITER):
    #    ax.plot(gen, bestFitness[it])
    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Error", fontsize=15)
    #ax.set_ylim(0, 50)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, len(gen))
    ax.grid(1)
    plt.grid(which="major", color="dimgrey", linewidth=0.8)
    plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    plt.legend()
    for text in plt.legend().get_texts():
        text.set_color("black")
        #text.set_color("white")
        text.set_fontsize(24)
    ax.set_title(f"{algorithmName} on {benchmarkName} \n\nPop size: {POPSIZE}  Gen: {GEN}  Iter:{ITER}", fontsize=20)
    plt.savefig(f"./figs/{algorithmName}-{benchmarkName}-{POPSIZE}-{GEN}-{ITER}-{year}{month}{day}{hour}.png", format="png")
    plt.show()



if __name__ == "__main__":
    main()

