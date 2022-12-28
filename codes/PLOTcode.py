'''
Code to plot graph using data file

Alexandre Mascarenhas
'''
import json
import shutil
import itertools
import operator
import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
import os
import csv
import sys
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.benchmarks import movingpeaks
from deap import creator
from deap import tools

cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

def configPlot(parameters):
    THEME = parameters["THEME"]
    plt.rcParams["figure.figsize"] = (20, 8)
    if(THEME == 1):
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "cornsilk"
        plt.rcParams["savefig.facecolor"] = "#1c1c1c"
    elif(THEME == 2):
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "#1c1c1c"
        plt.rcParams["savefig.facecolor"] = "#1c1c1c"
    elif(THEME == 3):
        plt.rcParams["axes.facecolor"] = "white"

    fig, ax = plt.subplots(1)
    ax.grid(1)
    plt.grid(which="major", color="dimgrey", linewidth=0.8)
    plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    return fig, ax



def plot(ax, data, label, fStd=0, color="orange"):
    ax.plot(data["gen"], data["bestError"], color=color, label=label)
    if(fStd):
        ax.fill_between(data["gen"], data["bestError"] - data["std"], data["bestError"] + data["std"], color="dark"+color, alpha=0.1)
    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Error", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, len(data["gen"]))
    return ax




def showPlots(fig, ax, parameters):
    path = f"{parameters['PATH']}/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}"
    THEME = parameters["THEME"]
    plt.legend()
    for text in plt.legend().get_texts():
        if(THEME == 1):
            text.set_color("black")
        elif(THEME == 2):
            text.set_color("white")
        text.set_fontsize(18)
    title = parameters["TITLE"]
    if(parameters["TITLE"] == 0):
        title = f"{parameters['ALGORITHM']} on {parameters['BENCHMARK']}"
    ax.set_title(title, fontsize=20)
    plt.savefig(f"{path}/{parameters['NAME']}", format="png")
    plt.show()



def mean(data):
    bMean = [0 for i in range( len(data[0]["bestError"]) )]
    bStd = [0 for i in range( len(data[0]["bestError"]) )]
    std = [0 for i in range( len(data) )]
    sum = 0

    for i in range (len(data[0]["bestError"])):
        for j in range(len(data)):
            sum += data[j]["bestError"][i]
            std[j] = data[j]["bestError"][i]
        bMean[i] = sum/len(data)
        bStd[i] = np.std(std)
        sum = 0
        std = [0 for i in range( len(data) )]

    zipped = list(zip(data[0]["gen"], bMean, bStd))
    bestMean = pd.DataFrame(zipped, columns=["gen", "bestError", "std"])
    return bestMean



def main():
    # reading the parameters from the config file
    with open("./config.ini") as f:
        parameters = json.loads(f.read())
    debug = parameters["DEBUG"]
    if(debug):
        print("Parameters:")
        print(parameters)

    THEME = parameters["THEME"]

    path = f"{parameters['PATH']}/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}"
    df = pd.read_csv(f"{path}/data.csv")

    fig, ax = configPlot(parameters)

    data = [[] for i in range( len(pd.unique(df["run"])) )]
    for i in range( len(pd.unique(df["run"])) ):
        data[i] = df[df["run"] == i]
        data[i] = data[i].drop_duplicates(subset=["gen"])[["gen", "bestError"]]
        data[i].reset_index(inplace=True)
        #ax = plot(ax, data=data[i], label=i)

    bestMean = mean(data)
    ax = plot(ax, data=bestMean, label=parameters["ALGORITHM"], color="green", fStd=1)
    showPlots(fig, ax, parameters)


if __name__ == "__main__":
    main()
