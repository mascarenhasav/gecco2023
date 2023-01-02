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
import ast
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

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
           smin=None, smax=None, best=None, bestfit=creator.FitnessMax)
creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)


def ale(seed, min, max):
    #random.seed(int(seed+it))
    return random.uniform(min, max)

def generate(ndim, pmin, pmax, smin, smax):
    #pmin = 0
    #pmax = 100
    part = creator.Particle(ale(i, pmin, pmax) for i in range(ndim))
    part.speed = [ale(i, smin, smax) for i in range(ndim)]
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

def createToolbox(parameters):
    BOUNDS_POS = parameters["BOUNDS_POS"]
    BOUNDS_VEL = parameters["BOUNDS_VEL"]
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, ndim=parameters["NDIM"],\
    pmin=BOUNDS_POS[0], pmax=BOUNDS_POS[1], smin=BOUNDS_VEL[0],smax=BOUNDS_VEL[1])
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

def evaluate(x, function, evalInc=1):
    global nevals
    fitInd = function(x)[0]
    globalOP = function.maximums()[0][0]
    fitness = [abs( fitInd - globalOP )]
    #print(f"Ind: {fitInd}   GOP: {globalOP}   Fitness: {fitness}")
    if(evalInc):
        nevals += 1
    else:
        function.nevals -= 1    # Dont increment the evals in the lib
    return fitness


def evaluateAll(pop, best, toolbox, mpb):
    for part in pop:
        part.fitness.values = toolbox.evaluate(part, mpb, 0)
        if not part.best or part.best.fitness > part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values

    best.fitness.values = toolbox.evaluate(best, mpb, 0)
    return pop, best

def changeDetection(pop, toolbox, best):
    now = toolbox.evaluate(best, mpb, 0)  # Check if ocurred a change
    if(now[0] != best.fitness.values[0]):
        print(f"gen:{g} n:{nevals} best:{best.fitness.values[0]} now:{now[0]}")
        change = 1
        best.fitness.values = now
        pop = evaluateAll(pop, toolbox, mpb)
    return pop, best



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

    peaks = 0

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

    header = ["run", "gen", "nevals", "partN", "part", "partError", "best", "bestError", "env", "gop", "gopFitness", "lop", "lopFitness"]
    writeLog(mode=0, filename=filename, header=header)
    headerOPT = [f"opt{i}" for i in range(parameters["NPEAKS_MPB"])]
    writeLog(mode=0, filename=f"{path}/optima.csv", header=headerOPT)
    toolbox = createToolbox(parameters)
    bestFitness = [ [] for i in range(ITER)]

    if(parameters["RANDOM_CHANGES"]):
        changesGen = [random.randint(parameters["RANGE_GEN_CHANGES"][0], parameters["RANGE_GEN_CHANGES"][1]) for _ in range(parameters["NCHANGES"])]
    else:
        changesGen = parameters["CHANGES_GEN"]

    for it in range(1, ITER+1):
        rndMPB = random.Random()
        rnd = random.Random()
        random.seed(it**5)
        rnd.seed(it**5)
        rndMPB.seed(minute**5)
        mpb = movingpeaks.MovingPeaks(dim=parameters["NDIM"], random=rndMPB, **scenario)
        pop = toolbox.population(n=POPSIZE)
        best = None
        nevals = 0
        env = 0
        opt = [0 for _ in range(parameters["NPEAKS_MPB"])]

        if(peaks < parameters["NCHANGES"]):
            for i in range(parameters["NPEAKS_MPB"]):
                opt[i] = mpb.maximums()[i]
            with open(f"{path}/optima.csv", "a") as f:
                write = csv.writer(f)
                write.writerow(opt)
            peaks += 1


        for g in range(1, GEN+1):

            # Change detection
            if(parameters["CHANGE"]):
                if(g in changesGen):
                    env += 1
                    mpb.changePeaks()
                    pop, best = evaluateAll(pop, best, toolbox, mpb)
                    if(peaks < parameters["NCHANGES"]):
                        for i in range(parameters["NPEAKS_MPB"]):
                            opt[i] = mpb.maximums()[i]
                        with open(f"{path}/optima.csv", "a") as f:
                            write = csv.writer(f)
                            write.writerow(opt)
                        peaks += 1

            # PSO
            for part, partN in zip(pop, range(1, len(pop)+1)):
                part.fitness.values = toolbox.evaluate(part, mpb)

                if(parameters["CD"] and best):
                    changeDetection(pop, toolbox, best)

                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

                log = [{"run": it, "gen": g, "nevals":nevals, "partN": partN, "part":part, "partError": part.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0], "env": env}]
                writeLog(mode=1, filename=filename, header=header, data=log)

                if(debug):
                    print(log)

            for part in pop:
                toolbox.update(part, best) # Update the particles position

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

    if(parameters["PLOT"]):
        os.system(f"python3 ../../PLOTcode.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")


if __name__ == "__main__":
    main()


