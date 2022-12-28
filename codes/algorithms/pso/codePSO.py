import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import os
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.benchmarks import movingpeaks
from deap import creator
from deap import tools


NDIM = 2
MAXIMIZATION = 1
GOPT = 2
COLOR = 2


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
           smin=None, smax=None, best=None)
# Setup of MPB
scenario = movingpeaks.SCENARIO_1
scenario["period"] = 1000
BOUNDS = [scenario["min_coord"], scenario["max_coord"]]
mpb = movingpeaks.MovingPeaks(dim=NDIM, **scenario)


cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))



def registerStats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats

def createLogbook(stats):
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "best", "error"] + stats.fields
    return logbook

def createToolbox(bounds, size):
    # Creating the deap objects
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=NDIM,
        pmin=BOUNDS[0], pmax=BOUNDS[1], smin=-(BOUNDS[1] - BOUNDS[0])/2.0,
        smax=(BOUNDS[1] - BOUNDS[0])/2.0)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
    return toolbox


def pso(GEN, POPSIZE, ITER, BOUNDS, benchmark):
    toolbox = createToolbox(BOUNDS, NDIM)
    toolbox.register("evaluate", benchmark)
    bestFitness = [ [] for i in range(ITER)]
    for it in range(ITER):
        pop = toolbox.population(n=POPSIZE)
        stats = registerStats()
        logbook = createLogbook(stats)
        best = None

        for g in range(GEN):
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            #logbook.record(gen=g, evals=len(pop), best=best, error=mpb.currentError(), **stats.compile(pop))
            logbook.record(gen=g, evals=len(pop), best=best, error=best.fitness.values[0], **stats.compile(pop))
            print(logbook.stream)

            bestFitness[it] = logbook.select("error")

    if(MAXIMIZATION):
        for i in range(len(bestFitness)):
            for j in range(len(bestFitness[i])):
                bestFitness[i][j] = GOPT - bestFitness[i][j]

    bMean = [sum(sub_list) / len(sub_list) for sub_list in zip(*bestFitness)]
    bStd = np.std(bMean)
    gen = logbook.select("gen")
    return gen, bMean, bStd, logbook.select("best")


def plot(data):
    plt.rcParams["figure.figsize"] = (20, 8)
    plt.style.use("dark_background")
    if(COLOR == 1):
        plt.rcParams["axes.facecolor"] = "cornsilk"
    elif(COLOR == 2):
        plt.rcParams["axes.facecolor"] = "#1c1c1c"
    plt.rcParams["savefig.facecolor"] = "#1c1c1c"
    fig, ax = plt.subplots(1)
    ax.plot(data[0], data[1], color="red", label="PSO")
    ax.fill_between(data[0], data[1] - data[2], data[1] + data[2], color="darkred", alpha=0.1)
    #for it in range(ITER):
    #    ax.plot(gen, bestFitness[it])
    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Error", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 1000)
    ax.grid(1)
    plt.grid(which="major", color="dimgrey", linewidth=0.8)
    plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    plt.legend()
    for text in plt.legend().get_texts():
        if(COLOR == 1):
            text.set_color("black")
        elif(COLOR == 2):
            text.set_color("white")
        text.set_fontsize(24)
    return fig, ax


def printa(best, bestFit, std):
    print(f"PSO:")
    print(f"--- Best solution: {best} ")
    print(f"--- Error: {bestFit} ({std})")
    print(f"-------------------------------")

def writeFile(best, bestFit, std, path, fileName):
    file = open(f"{path}{fileName}.txt","w")
    L = ["PSO:",
        f"\n--- Best solution: {best} ",
        f"\n--- Error: {bestFit} ({std})",
        "\n-------------------------------\n",
    ]
    file.writelines(L)
    file.close()



def main():
    GEN = int(input("Generations: "))
    POPSIZE = int(input("Population size: "))
    ITER = int(input("Runnings: "))
    BOUNDS = [-5, 5]
    algorithmName = "PSO"
    benchmark = benchmarks.h1
    benchmarkName = "H1"
    fileName = f"{algorithmName}-{benchmarkName}-{POPSIZE}-{GEN}-{ITER}-{year}{month}{day}{hour}{minute}"
    path = f"./files/{fileName}/"
    if( os.path.isdir(path) == False ):
        os.mkdir(path)

    # Run PSO algortihm
    psoValues = pso(GEN, POPSIZE, ITER, BOUNDS, benchmark)
    bestPSO = psoValues[3][psoValues[1].index(min(psoValues[1]))]
    bestPSOfit = min(psoValues[1])
    stdPSO = psoValues[2]

    # Output of the results
    printa(bestPSO, bestPSOfit, stdPSO)
    writeFile(bestPSO, bestPSOfit, stdPSO, path, fileName)
    fig, ax = plot(psoValues)

    ax.set_title(f"{algorithmName} on {benchmarkName} \n\nPop size: {POPSIZE}  Gen: {GEN}  Iter:{ITER}", fontsize=20)
    plt.savefig(f"{path}{fileName}.png", format="png")



    plt.show()


if __name__ == "__main__":
    main()

