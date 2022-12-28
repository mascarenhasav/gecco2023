'''
Code for PSO algorithm using DEAP library

Alexandre Mascarenhas
'''

import json
import shutil
import itertools
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import os
import csv
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.benchmarks import movingpeaks
from deap import creator
from deap import tools

# datetime variables
cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

nevals = 0
it = 0

def ale(seed, min, max):
    random.seed(seed+it)
    return random.uniform(min, max)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(ale(i, pmin, pmax) for i in range(size))
    part.speed = [ale(i, smin, smax) for i in range(size)]
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

def createLogbook(header):
    logbook = tools.Logbook()
    logbook.header = header
    return logbook

def createToolbox(parameters):
    BOUNDS = parameters["BOUNDS"]
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=parameters["NDIM"],
        pmin=BOUNDS[0], pmax=BOUNDS[1], smin=-(BOUNDS[1] - BOUNDS[0])/2.0,
        smax=(BOUNDS[1] - BOUNDS[0])/2.0)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=parameters["phi1"], phi2=parameters["phi2"])
    toolbox.register("evaluate", evaluate)
    return toolbox

def writeLog(mode, filename, header, data=None):
    if(mode==0):
        # Create csv file
        with open(filename, "w") as file:
            csvwriter = csv.DictWriter(file, fieldnames=header)
            csvwriter.writeheader()
    elif(mode==1):
        # Writing in csv file
        with open(filename, mode="a") as file:
            csvwriter = csv.DictWriter(file, fieldnames=header)
            csvwriter.writerows(data)

def evaluate(x, function):
    global nevals
    fitness = [math.sqrt( (function(x)[0] - function.maximums()[0][0])**2 )]
    #fitness = benchmarks.himmelblau(x)
    #fitness = benchmarks.ackley(x)
    #fitness = abs(benchmarks.shekel(x, [[3.33, 3.33]], [2])[0] - benchmarks.shekel([3.33,3.33], [[3.33, 3.33]], [2])[0])
    #fitness = benchmarks.h1(x)
    nevals += 1
    return fitness

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
           smin=None, smax=None, best=None, bestfit=creator.FitnessMax)
creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

def pso(parameters):

    GEN = parameters["GEN"]
    POPSIZE = parameters["POPSIZE"]
    ITER = parameters["RUNS"]

    path = f"{parameters['PATH']}/{parameters['ALGORITHM']}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    path += f"/{year}-{month}-{day}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    path += f"/{hour}-{minute}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    filename = f"{path}/{parameters['FILENAME']}"
    debug = parameters["DEBUG"]

    global nevals
    global it

    # Setup of MPB
    scenario = movingpeaks.SCENARIO_1
    scenario["period"] = parameters["PERIOD_MPB"]
    scenario["npeaks"] = parameters["NPEAKS_MPB"]
    scenario["uniform_height"] = parameters["UNIFORM_HEIGHT_MPB"]
    scenario["move_severity"] = parameters["MOVE_SEVERITY_MPB"]
    scenario["min_height"] = parameters["MIN_HEIGHT_MPB"]
    scenario["max_height"] = parameters["MAX_HEIGHT_MPB"]
    mpb = movingpeaks.MovingPeaks(dim=parameters["NDIM"], **scenario)


    header = ["run", "gen", "nevals", "partN", "part", "partError", "best", "bestError"]
    writeLog(mode=0, filename=filename, header=header)
    toolbox = createToolbox(parameters)
    bestFitness = [ [] for i in range(ITER)]

    for it in range(ITER):
        pop = toolbox.population(n=POPSIZE)
        best = None
        nevals = 0

        for g in range(GEN):
            for part, partN in zip(pop, range(len(pop))):
                part.fitness.values = toolbox.evaluate(part, mpb)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                log = [{"run": it, "gen": g, "nevals":nevals, "partN": partN, "part":part, "partError": part.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0]}]
                writeLog(mode=1, filename=filename, header=header, data=log)
                if(debug):
                    print(log)
            for part in pop:
                toolbox.update(part, best)

    shutil.copyfile("config.ini", f"{path}/config.ini")
    print(f"File generated: {path}/data.csv \nThx!")



def main():
    # reading the parameters from the config file
    with open("./config.ini") as f:
        parameters = json.loads(f.read())
    debug = parameters["DEBUG"]
    if(debug):
        print("Parameters:")
        print(parameters)

    pso(parameters)


if __name__ == "__main__":
    main()


