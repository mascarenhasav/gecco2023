import itertools
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


COLOR = 2

verbose=True

EVALF = 1

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
           smin=None, smax=None, best=None, bestfit=creator.FitnessMax)
creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

# Setup of MPB
scenario = movingpeaks.SCENARIO_1
NDIM = config["NDIM"]
scenario["period"] = config["PERIOD_MPB"]
scenario["npeaks"] = config["NPEAKS_MPB"]
scenario["uniform_height"] = config["UNIFORM_HEIGHT_MPB"]
scenario["move_severity"] = config["MOVE_SEVERITY__MPB"]
scenario["min_height"] = config["MIN_HEIGHT_MPB"]
scenario["max_height"] = config["MAX_HEIGHT_MPB"]
BOUNDS = [scenario["min_coord"], scenario["max_coord"]]
mpb = movingpeaks.MovingPeaks(dim=NDIM, **scenario)


cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute


def evaluate(x):
    fitness = [math.sqrt( (mpb(x)[0] - mpb.maximums()[0][0])**2 )]
    #fitness = benchmarks.himmelblau(x)
    #fitness = benchmarks.ackley(x)
    #fitness = abs(benchmarks.shekel(x, [[3.33, 3.33]], [2])[0] - benchmarks.shekel([3.33,3.33], [[3.33, 3.33]], [2])[0])
    #fitness = benchmarks.h1(x)
    return fitness

def configPlot():
    plt.rcParams["figure.figsize"] = (20, 8)
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

    fig, ax = plt.subplots(1)
    ax.grid(1)
    plt.grid(which="major", color="dimgrey", linewidth=0.8)
    plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    return fig, ax

def plot(ax, data, color, label):
    ax.plot(data[0], data[1], color=color, label=label)
    ax.fill_between(data[0], data[1] - data[2], data[1] + data[2], color="dark"+color, alpha=0.1)
    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Error", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, len(data[0]))
    return ax

def printa(best, bestFit, std, nameAlg):
    print(f"{nameAlg}:")
    print(f"--- Best solution: {best} ")
    print(f"--- Error: {bestFit} ({std})")
    print(f"-------------------------------")

def writeFile(best, bestFit, std, path, fileName, nameAlg):
    file = open(f"{path}{fileName}.txt","w")
    L = [f"{nameAlg}:",
        f"\n--- Best solution: {best} ",
        f"\n--- Error: {bestFit} ({std})",
        "\n-------------------------------\n",
    ]
    file.writelines(L)
    file.close()

def showPlots(fig, ax, title, name):
    plt.legend()
    for text in plt.legend().get_texts():
        if(COLOR == 1):
            text.set_color("black")
        elif(COLOR == 2):
            text.set_color("white")
        text.set_fontsize(18)
    ax.set_title(title, fontsize=20)
    plt.savefig(name, format="png")
    plt.show()

def ale(seed, min, max):
    random.seed(seed)
    return random.uniform(min, max)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(ale(i, pmin, pmax) for i in range(size))
    part.speed = [ale(i, smin, smax) for i in range(size)]
    part.smin = smin
    part.smax = smax
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

def updateParticle(part, best, phi1, phi2, mode):
    if(mode == 1):
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
    else:
        ce1 = (phi2*random.uniform(0, 1) for _ in range(len(part)))
        ce2 = (phi2*random.uniform(0, 1) for _ in range(len(part)))
        ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
        ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
        a = map(operator.sub,
                          map(operator.mul,
                                        itertools.repeat(phi1),
                                        map(operator.add, ce1_p, ce2_g)),
                          map(operator.mul,
                                         itertools.repeat(1 - phi1),
                                         part.speed))
        part.speed = list(map(operator.add, part.speed, a))
        part[:] = list(map(operator.add, part, part.speed))



def registerStats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats

def createLogbook(header):
    logbook = tools.Logbook()
    logbook.header = header
    return logbook

def createToolbox(bounds, size):
    # Creating the deap objects
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=NDIM,
        pmin=BOUNDS[0], pmax=BOUNDS[1], smin=-(BOUNDS[1] - BOUNDS[0])/2.0,
        smax=(BOUNDS[1] - BOUNDS[0])/2.0)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=0.729843788, phi2=2.05, mode=2)
    toolbox.register("convert", convertQuantum, dist="nuvd")
    toolbox.register("evaluate", evaluate)
    return toolbox


def pso(GEN, POPSIZE, ITER, BOUNDS):
    toolbox = createToolbox(BOUNDS, NDIM)
    bestFitness = [ [] for i in range(ITER)]

    for it in range(ITER):
        pop = toolbox.population(n=POPSIZE)
        stats = registerStats()
        logbook = createLogbook(["gen", "evals", "best", "error"] + stats.fields)
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

            best.fitness.values = evaluate(best)
            # Gather all the fitnesses in one list and print the stats
            #logbook.record(gen=g, evals=len(pop), best=best, error=mpb.currentError(), **stats.compile(pop))
            error = best.fitness.values[0]
            logbook.record(gen=g, evals=len(pop), best=best, error=error, **stats.compile(pop))
            #print(logbook.stream)
            print(f"Gen: {logbook.select('gen')[-1]}")
            #for i in range(NPEAKS):
             #   print(f"     -GLOBAL: {mpb.maximums()[i][0]} {mpb.maximums()[i][1]}")
             #   print(f"     -GLOBAL: {evaluate(mpb.maximums()[i][1])}")

            gBest = mpb.maximums()[0]

            print(f"     -GLOBAL Best: {gBest[0]} {gBest[1]}")
            print(f"     -ATUALA: {best.fitness.values[0]} {best}")
            print(f"     -ATUALB: {evaluate(best)} {best}")


        bestFitness[it] = logbook.select("error")

    if(MAXIMIZATION):
        for i in range(len(bestFitness)):
            for j in range(len(bestFitness[i])):
                bestFitness[i][j] = GOPT - bestFitness[i][j]

    bMean = [sum(sub_list) / len(sub_list) for sub_list in zip(*bestFitness)]
    bStd = np.std(bMean)
    gen = logbook.select("gen")
    return gen, bMean, bStd, logbook.select("best")


def main():
    GEN = int(input("Generations: "))
    POPSIZE = int(input("Population size: "))
    ITER = int(input("Runnings: "))
    #BOUNDS = [-15, 30]
    benchmarkName = "MPB"
    global mpb
    algorithmName = "PSO"
    fileName = f"{algorithmName}-{benchmarkName}-{POPSIZE}-{GEN}-{ITER}-{year}{month}{day}{hour}{minute}"
    path = f"./files/{fileName}/"
    if( os.path.isdir(path) == False ):
        os.mkdir(path)
    # Run PSO algortihm
    psoValues = pso(GEN, POPSIZE, ITER, BOUNDS)
    bestPSO = psoValues[3][psoValues[1].index(min(psoValues[1]))]
    bestPSOfit = min(psoValues[1])
    stdPSO = psoValues[2]
    # Output of the results
    writeFile(bestPSO, bestPSOfit, stdPSO, path, fileName, algorithmName)
    # Plot
    fig, ax = configPlot()
    ax = plot(ax, psoValues, color="red", label="PSO - SP") 
    # Run PSO algortihm
    scenario["period"] = 2000
    mpb = movingpeaks.MovingPeaks(dim=NDIM, **scenario)
    psoValues = pso(GEN, POPSIZE, ITER, BOUNDS)
    bestPSO = psoValues[3][psoValues[1].index(min(psoValues[1]))]
    bestPSOfit = min(psoValues[1])
    stdPSO = psoValues[2]
    # Output of the results
    writeFile(bestPSO, bestPSOfit, stdPSO, path, fileName, algorithmName)
    # Plot
    ax = plot(ax, psoValues, color="green", label="PSO - DP")

    printa(bestPSO, bestPSOfit, stdPSO, "PSO")
    title = f"{algorithmName} on {benchmarkName} \n\nPop size: {POPSIZE}  Gen: {GEN}"
    name = f"{path}{fileName}.png"
    showPlots(fig, ax, title, name)



if __name__ == "__main__":
    main()


