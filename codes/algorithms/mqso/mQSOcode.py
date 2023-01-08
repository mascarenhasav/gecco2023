'''
Code for mQSO algorithm using DEAP library.

As proposed in:
" T. Blackwell, J. Branke, Multiswarms, exclusion, and anti-convergence
in dynamic environments, IEEE Transactions on Evolutionary Computation
10 (2006) 459–472. "

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
import sys
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
run = 0 # Current running
peaks = 0
env = 0
changesEnv = [0 for _ in range(100)]

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
Convert particles to probabilistic particle
'''
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
    toolbox.register("convert", convertQuantum, dist="nuvd")
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
def evaluate(x, function, parameters=None, evalInc=1):
    global nevals
    fitInd = function(x)[0]
    globalOP = function.maximums()[0][0]
    fitness = [abs( fitInd - globalOP )]
    if(evalInc):
        nevals += 1
    else:
        function.nevals -= 1    # Dont increment the evals in the lib
    changeEnvironment(function, parameters)
    return fitness


'''
Update the evaluation of all particles after a change occurred
'''
def evaluateAll(pop, best, toolbox, mpb):
    for swarm in pop:
        for part, partId in zip(swarm, range(1, len(swarm)+1)):
            part.fitness.values = toolbox.evaluate(part, mpb, 0)
            # Check if the particles are the best of itself and best at all
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

        swarm.best.fitness.values = toolbox.evaluate(swarm.best, mpb, 0)

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
Check if the dirs already exist, and if not, create them
Returns the path
'''
def antiConvergence(pop, parameters, randomInit):
    nconv = 0
    if(parameters["RCONV"]):
        rconv = parameters["RCONV"]
    else:
        rconv  = 0.75*(parameters["BOUNDS_POS"][1] - parameters["BOUNDS_POS"][0]) \
                 / (2 * len(pop)**(1.0/parameters["NDIM"]))
    wswarmId = None
    wswarm = None
    for i, swarm in enumerate(pop):
        # Compute the diameter of the swarm
        for p1, p2 in itertools.combinations(swarm, 2):
            d = math.sqrt(sum((x1 - x2)**2. for x1, x2 in zip(p1, p2)))
            if d > 2*rconv:
                nconv += 1
            # Search for the worst swarm according to its global best
            if not wswarm or swarm.best.fitness < wswarm.best.fitness:
                wswarmId = i
                wswarm = swarm
            break

    # If all swarms have converged, remember to randomize the worst
    if nconv == 0:
        randomInit[wswarmId] = 1

    return randomInit

'''
Check if the dirs already exist, and if not, create them
Returns the path
'''
def exclusion(pop, parameters, randomInit):
    if(parameters["REXCL"]):
        rexcl = parameters["REXCL"]
    else:
        rexcl  = (parameters["BOUNDS_POS"][1] - parameters["BOUNDS_POS"][0]) \
                 / (2 * len(pop)**(1.0/parameters["NDIM"]))

    reinit_swarms = set()
    for s1, s2 in itertools.combinations(range(len(pop)), 2):
        # Swarms must have a best and not already be set to reinitialize
        if pop[s1].best and pop[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
            dist = 0
            for x1, x2 in zip(pop[s1].best, pop[s2].best):
                dist += (x1 - x2)**2.
            dist = math.sqrt(dist)
            if dist < rexcl:
                if pop[s1].best.fitness <= pop[s2].best.fitness:
                    randomInit[s1] = 1
                else:
                    randomInit[s2] = 1

    return randomInit


'''
Check if a change occurred in the environment
'''
def changeDetection(swarm, toolbox, mpb, parameters):
    change = 0
    sensor = toolbox.evaluate(swarm.best, mpb, parameters=parameters)
    if(sensor[0] != swarm.best.fitness.values[0]):
        print(f"[CHANGE] nevals: {nevals}  sensor: {sensor}  sbest:{swarm.best.fitness.values[0]}")
        swarm.best.fitness.values = sensor
        change = 1
    return change

def reevaluateSwarm(swarm, best, toolbox, mpb, parameters):
    for part in swarm:
        part.best.fitness.values = toolbox.evaluate(part.best, mpb, parameters=parameters)
        if not swarm.best or swarm.best.fitness < part.best.fitness:
            swarm.best = creator.Particle(part.best)
            swarm.best.fitness.values = part.best.fitness.values

    return swarm, best

def changeEnvironment(mpb, parameters):
    # Change environment
    global changesEnv
    global nevals
    global peaks
    global env
    global path
    if(nevals in changesEnv):
        mpb.changePeaks() # Change the environment
        env += 1
        if(peaks <= parameters["NCHANGES"]): # Save the optima values
            saveOptima(parameters, mpb, path)
            peaks += 1
        print(f"MUDOU nevals: {nevals}")


'''
Algorithm
'''
def mQSO(parameters):
    # Create the DEAP creators
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None, bestfit=creator.FitnessMax)
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

    global nevals
    global run
    global peaks
    global changesEnv
    global path
    global env

    # Set the general parameters
    NEVALS = parameters["NEVALS"]
    POPSIZE = parameters["POPSIZE"]
    NSWARMS = parameters["NSWARMS"]
    SWARMSIZE = int(POPSIZE/NSWARMS)
    RUNS = parameters["RUNS"]
    debug = parameters["DEBUG"]
    path = f"{parameters['PATH']}/{parameters['ALGORITHM']}"

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
    header = ["run", "gen", "nevals", "swarmId", "partId", "part", "partError", "sbest", "sbestError", "best", "bestError", "env"]
    writeLog(mode=0, filename=filename, header=header)
    headerOPT = [f"opt{i}" for i in range(parameters["NPEAKS_MPB"])]
    writeLog(mode=0, filename=f"{path}/optima.csv", header=headerOPT)

    # Create the toolbox functions
    toolbox = createToolbox(parameters)

    # Check if the changes should be random or pre defined
    if(parameters["RANDOM_CHANGES"]):
        changesEnv = [random.randint(parameters["RANGE_NEVALS_CHANGES"][0], parameters["RANGE_NEVALS_CHANGES"][1]) for _ in range(parameters["NCHANGES"])]
    else:
        changesEnv = parameters["CHANGES_NEVALS"]

    # Main loop of ITER runs
    for run in range(1, RUNS+1):
        random.seed(run**5)
        best = None
        nevals = 0
        env = 1
        change = 0
        gen = 1
        genControl = 0
        randomInit = [0 for _ in range(1, NSWARMS+2)]
        ES_particles = [i for i in range(1, int(SWARMSIZE/2)+1)]

        # Initialize the benchmark for each run with seed being the minute
        rndMPB = random.Random()
        try:
            rndMPB.seed(int(sys.argv[1])**5)
        except IndexError:
            rndMPB.seed(minute**5)

        mpb = movingpeaks.MovingPeaks(dim=parameters["NDIM"], random=rndMPB, **scenario)

        # Create the population with size POPSIZE
        pop = [toolbox.swarm(n=int(POPSIZE/NSWARMS)) for _ in range(NSWARMS)]

        # Save the optima values
        if(peaks <= parameters["NCHANGES"]):
            saveOptima(parameters, mpb, path)
            peaks += 1

        # Repeat until reach the number of evals
        while nevals < NEVALS+1:

            # Anti-convergence operator
            if(parameters["ANTI_CONVERGENCE_OP"] and gen > 2):
                randomInit = antiConvergence(pop, parameters, randomInit)

            # Exclusion operator
            if(parameters["EXCLUSION_OP"] and gen > 2):
                randomInit = exclusion(pop, parameters, randomInit)

            # PSO
            for swarmId, swarm in enumerate(pop, 1):

                # Change detection
                if(parameters["CHANGE_DETECTION_OP"] and swarm.best):
                    change = changeDetection(swarm, toolbox, mpb, parameters=parameters)

                if(change and swarm):
                    swarm, best = reevaluateSwarm(swarm, best, toolbox, mpb, parameters=parameters)
                    best = None
                    print(f"[CHANGE] sbest:{swarm.best.fitness.values[0]}")
                    randomInit[swarmId] = 0

                for partId, part in enumerate(swarm, 1):
                    # If convergence or exclusion, randomize the particle
                    if(randomInit[swarmId]):
                        part = toolbox.particle()
                    else:
                        if(parameters["ES_PARTICLE_OP"] and partId in ES_particles):
                            part = toolbox.particle()
                        elif swarm.best and part.best:
                            toolbox.update(part, swarm.best)

                    # Evaluate the particle
                    part.fitness.values = toolbox.evaluate(part, mpb, parameters=parameters)

                    # Check if the particles are the best of itself and best at all
                    if not part.best or part.best.fitness < part.fitness:
                        part.best = creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                    if not swarm.best or swarm.best.fitness < part.fitness:
                        swarm.best = creator.Particle(part)
                        swarm.best.fitness.values = part.fitness.values
                    if not best or best.fitness < part.fitness:
                        best = creator.Particle(part)
                        best.fitness.values = part.fitness.values

                    # Save the log
                    log = [{"run": run, "gen": gen, "nevals":nevals, "swarmId": swarmId, "partId": partId, "part":part, "partError": part.best.fitness.values[0], "sbest": swarm.best, "sbestError": swarm.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0], "env": env}]
                    writeLog(mode=1, filename=filename, header=header, data=log)

                    if(debug):
                        print(log)

                # Randomization complete
                randomInit[swarmId] = 0


                #pop[wswarmId] = toolbox.swarm(n=(parameters["POPSIZE"]/parameters["NSWARMS"]))

            gen += 1


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
    mQSO(parameters)

    # For automatic calling of the plot functions
    if(parameters["PLOT"]):
        os.system(f"python3 ../../PLOTcode.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")
        os.system(f"python3 ../../PLOT2code.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")


if __name__ == "__main__":
    main()


