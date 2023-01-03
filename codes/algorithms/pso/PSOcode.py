'''
Code for PSO algorithm using DEAP library

Alexandre Mascarenhas

2023/1
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


# datetime variables
cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

# Global variables
nevals = 0 # Number of evaluations
it = 0 # Current running

'''
Create the particle, with its initial position and speed
being randomly generated
'''
def ale(seed, min, max):
    return random.uniform(min, max)
def generate(ndim, pmin, pmax, smin, smax):
    part = creator.Particle(ale(i, pmin, pmax) for i in range(ndim))
    part.speed = [ale(i, smin, smax) for i in range(ndim)]
    part.smin = smin
    part.smax = smax
    return part

'''
Update the position of the particles
'''
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


'''
Define the functions responsibles to create the objects of the algorithm
particles, swarms, the update and evaluate function also
'''
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

'''
Write the log of the algorithm over the generations on a csv file
'''
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

'''
Fitness function. Returns the error between the fitness of the particle
and the global optimum
'''
def evaluate(x, function, evalInc=1):
    global nevals
    fitInd = function(x)[0]
    globalOP = function.maximums()[0][0]
    fitness = [abs( fitInd - globalOP )]
    if(evalInc):
        nevals += 1
    else:
        function.nevals -= 1    # Dont increment the evals in the lib
    return fitness


'''
Update the evaluation of all particles after a change occurred
'''
def evaluateAll(pop, best, toolbox, mpb):
    for part in pop:
        part.fitness.values = toolbox.evaluate(part, mpb, 0)
        if not part.best or part.best.fitness > part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values

    best.fitness.values = toolbox.evaluate(best, mpb, 0)
    return pop, best

'''
This is to write the position and fitness of the peaks on
the 'optima.csv' file. The number of the peaks will be
NPEAKS_MPB*NCHANGES
'''
def saveOptima(parameters, mpb, path):
    opt = [0 for _ in range(parameters["NPEAKS_MPB"])]
    for i in range(parameters["NPEAKS_MPB"]):
        opt[i] = mpb.maximums()[i]
    with open(f"{path}/optima.csv", "a") as f:
        write = csv.writer(f)
        write.writerow(opt)

'''
Check if the dirs already exist, and if not, create them
Returns the path
'''
def checkDirs(path):
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    path += f"/{year}-{month}-{day}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    path += f"/{hour}-{minute}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    return path


'''
Algorithm
'''
def pso(parameters):
    # Create the DEAP creators
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None, bestfit=creator.FitnessMax)
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

    # Set the general parameters
    GEN = parameters["GEN"]
    POPSIZE = parameters["POPSIZE"]
    ITER = parameters["RUNS"]
    debug = parameters["DEBUG"]
    path = f"{parameters['PATH']}/{parameters['ALGORITHM']}"

    global nevals
    global it
    bestFitness = [ [] for i in range(ITER)]
    peaks = 0

    # Check if the dirs already exist otherwise create them
    path = checkDirs(path)
    filename = f"{path}/{parameters['FILENAME']}"

    # Setup of MPB
    scenario = movingpeaks.SCENARIO_1
    scenario["period"] = parameters["PERIOD_MPB"]
    scenario["npeaks"] = parameters["NPEAKS_MPB"]
    scenario["uniform_height"] = parameters["UNIFORM_HEIGHT_MPB"]
    scenario["move_severity"] = parameters["MOVE_SEVERITY_MPB"]
    scenario["min_height"] = parameters["MIN_HEIGHT_MPB"]
    scenario["max_height"] = parameters["MAX_HEIGHT_MPB"]
    scenario["min_coord"] = parameters["MIN_COORD_MPB"]
    scenario["max_coord"] = parameters["MAX_COORD_MPB"]

    # Headers of the log files
    header = ["run", "gen", "nevals", "partN", "part", "partError", "best", "bestError", "env"]
    writeLog(mode=0, filename=filename, header=header)
    headerOPT = [f"opt{i}" for i in range(parameters["NPEAKS_MPB"])]
    writeLog(mode=0, filename=f"{path}/optima.csv", header=headerOPT)

    # Create the toolbox functions
    toolbox = createToolbox(parameters)

    # Check if the changes should be random or pre defined
    if(parameters["RANDOM_CHANGES"]):
        changesGen = [random.randint(parameters["RANGE_GEN_CHANGES"][0], parameters["RANGE_GEN_CHANGES"][1]) for _ in range(parameters["NCHANGES"])]
    else:
        changesGen = parameters["CHANGES_GEN"]

    # Main loop of ITER runs
    for it in range(1, ITER+1):
        random.seed(it**5)
        best = None
        nevals = 0
        env = 0

        # Initialize the benchmark for each run with seed being the minute
        rndMPB = random.Random()
        rndMPB.seed(minute**5)
        mpb = movingpeaks.MovingPeaks(dim=parameters["NDIM"], random=rndMPB, **scenario)

        # Create the population with size POPSIZE
        pop = toolbox.population(n=POPSIZE)

        # Save the optima values
        if(peaks < parameters["NCHANGES"]):
            saveOptima(parameters, mpb, path)
            peaks += 1

        # Loop for each generation
        for g in range(1, GEN+1):

            # Change detection
            if(parameters["CHANGE"]):
                if(g in changesGen):
                    env += 1
                    mpb.changePeaks() # Change the environment
                    pop, best = evaluateAll(pop, best, toolbox, mpb) # Re-evaluate all particles
                    if(peaks < parameters["NCHANGES"]): # Save the optima values
                        saveOptima(parameters, mpb, path)
                        peaks += 1

            # PSO
            for part, partN in zip(pop, range(1, len(pop)+1)):

                # Evaluate the particles
                part.fitness.values = toolbox.evaluate(part, mpb)

                # Check if the particles are the best of itself and best at all
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

                # Save the log
                log = [{"run": it, "gen": g, "nevals":nevals, "partN": partN, "part":part, "partError": part.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0], "env": env}]
                writeLog(mode=1, filename=filename, header=header, data=log)

                if(debug):
                    print(log)

            # Update the particles position
            for part in pop:
                toolbox.update(part, best)

    # Copy the config.ini file to the experiment dir
    shutil.copyfile("config.ini", f"{path}/config.ini")
    print(f"File generated: {path}/data.csv \nThx!")



def main():
    # Read the parameters from the config file
    with open("./config.ini") as f:
        parameters = json.loads(f.read())
    debug = parameters["DEBUG"]
    if(debug):
        print("Parameters:")
        print(parameters)

    # Call the algorithm
    pso(parameters)

    # For automatic calling of the plot functions
    if(parameters["PLOT"]):
        os.system(f"python3 ../../PLOTcode.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")
        os.system(f"python3 ../../PLOT2code.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")


if __name__ == "__main__":
    main()


