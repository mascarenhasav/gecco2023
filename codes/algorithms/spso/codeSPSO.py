"""Implementation of the Speciation Particle Swarm Optimization algorithm as
presented in *Li, Blackwell, and Branke, 2006, Particle Swarm with Speciation
and Adaptation in a Dynamic Environment.*
"""
import itertools
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

plt.rcParams["figure.figsize"] = (20, 8)
plt.style.use("dark_background")
plt.rcParams["axes.facecolor"] = "cornsilk"
plt.rcParams["savefig.facecolor"] = "#1c1c1c"

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

scenario = movingpeaks.SCENARIO_1
scenario["period"] = 1000
NDIM = 5
BOUNDS = [scenario["min_coord"], scenario["max_coord"]]
mpb = movingpeaks.MovingPeaks(dim=NDIM, **scenario)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    best=None, bestfit=creator.FitnessMax)

def generate(pclass, dim, pmin, pmax, smin, smax):
    part = pclass(random.uniform(pmin, pmax) for _ in range(dim)) 
    part.speed = [random.uniform(smin, smax) for _ in range(dim)]
    return part

def convert_quantum(swarm, rcloud, centre):
    dim = len(swarm[0])
    for part in swarm:
        position = [random.gauss(0, 1) for _ in range(dim)]
        dist = math.sqrt(sum(x**2 for x in position))

        # Gaussian distribution
        # u = abs(random.gauss(0, 1.0/3.0))
        # part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

        # UVD distribution
        # u = random.random()
        # part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

        # NUVD distribution
        u = abs(random.gauss(0, 1.0/3.0))
        part[:] = [(rcloud * x * u / dist) + c for x, c in zip(position, centre)]

        del part.fitness.values
        del part.bestfit.values
        part.best = None

    return swarm

def updateParticle(part, best, chi, c):
    ce1 = (c*random.uniform(0, 1) for _ in range(len(part)))
    ce2 = (c*random.uniform(0, 1) for _ in range(len(part)))
    ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
    ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
    a = map(operator.sub,
                      map(operator.mul,
                                    itertools.repeat(chi),
                                    map(operator.add, ce1_p, ce2_g)),
                      map(operator.mul,
                                     itertools.repeat(1-chi),
                                     part.speed))
    part.speed = list(map(operator.add, part.speed, a))
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, creator.Particle, dim=NDIM,
    pmin=BOUNDS[0], pmax=BOUNDS[1], smin=-(BOUNDS[1] - BOUNDS[0])/2.0,
    smax=(BOUNDS[1] - BOUNDS[0])/2.0)
toolbox.register("swarm", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, chi=0.729843788, c=2.05)
toolbox.register("convert", convert_quantum)


def main(verbose=True):
    NEXCESS = 3
    RCLOUD = 0.5    # 0.5 times the move severity
    GEN = 1000
    ITER = 2
    POPSIZE = 50
    SWARMSIZE = 1
    RS = (BOUNDS[1] - BOUNDS[0]) / (50**(1.0/NDIM))    # between 1/20 and 1/10 of the domain's range
    PMAX = 10
    bestFitness = [ [] for i in range(ITER)]
    algorithmName = "SPSO"
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


        swarm = toolbox.swarm(n=POPSIZE)

        for g in range(GEN):
            # Evaluate each particle in the swarm
            for part in swarm:
                part.fitness.values = toolbox.evaluate(part)
                if not part.best or part.bestfit < part.fitness:
                    part.best = toolbox.clone(part[:])         # Get the position
                    part.bestfit.values = part.fitness.values  # Get the fitness


            # Sort swarm into species, best individual comes first
            sorted_swarm = sorted(swarm, key=lambda ind: ind.bestfit, reverse=True)
            species = []
            while sorted_swarm:
                found = False
                for s in species:
                    dist = math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(sorted_swarm[0].best, s[0].best)))
                    if dist <= RS:
                        found = True
                        s.append(sorted_swarm[0])
                        break
                if not found:
                    species.append([sorted_swarm[0]])
                sorted_swarm.pop(0)

            record = stats.compile(swarm)
            logbook.record(gen=g, evals=mpb.nevals, nswarm=len(species),
                           error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

            if verbose:
                print(logbook.stream)

            # Detect change
            if any(s[0].bestfit.values != toolbox.evaluate(s[0].best) for s in species):
                # Convert particles to quantum particles
                for s in species:
                    s[:] = toolbox.convert(s, rcloud=RCLOUD, centre=s[0].best)

            else:
                # Replace exceeding particles in a species with new particles
                for s in species:
                    if len(s) > PMAX:
                        n = len(s) - PMAX
                        del s[PMAX:]
                        s.extend(toolbox.swarm(n=n))

                # Update particles that have not been reinitialized
                for s in species[:-1]:
                    for part in s[:PMAX]:
                        toolbox.update(part, s[0].best)
                        del part.fitness.values

            # Return all but the worst species' updated particles to the swarm
            # The worst species is replaced by new particles
            swarm = list(itertools.chain(toolbox.swarm(n=len(species[-1])), *species[:-1]))


        bestFitness[it] = logbook.select("error")

    bMean = [sum(sub_list) / len(sub_list) for sub_list in zip(*bestFitness)]
    bStd = np.std(bMean)
    gen = logbook.select("gen")
    fig, ax = plt.subplots(1)
    ax.plot(gen, bMean, color="green")
    ax.fill_between(gen, bMean - bStd, bMean + bStd, color='#888888', alpha=0.3)
    #ax.fill_between(gen, bMean - 2*bStd, bMean + 2*bStd, color='#888888', alpha=0.2)
    #for it in range(ITER):
    #    ax.plot(gen, bestFitness[it])
    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Mean of the best individual per generation", fontsize=15)
    ax.set_ylim(0, 50)
    ax.grid(1)
    plt.grid(which="major", color="dimgrey", linewidth=0.8)
    plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    ax.set_title(f"{algorithmName} on {benchmarkName} \n\nPop size: {POPSIZE}  Gen: {GEN}  Iter:{ITER}", fontsize=20)
    plt.savefig(f"./figs/{algorithmName}-{benchmarkName}-{POPSIZE}-{GEN}-{ITER}.png", format="png")
    plt.show()





if __name__ == '__main__':
    main()

