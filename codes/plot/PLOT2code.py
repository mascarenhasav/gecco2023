
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

colors = ["red", "b", "gray", "orange", "lime", "yellow"]
colors2 = ["r", "gray"]

def configPlot(parameters):
    THEME = parameters["THEME"]
    #plt.rcParams["figure.figsize"] = (20, 8)
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
    if(parameters["GRID"] == 1):
        ax.grid(True)
        plt.grid(which="major", color="dimgrey", linewidth=0.8)
        plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    else:
        ax.grid(False)
    return fig, ax


def plot(ax, data, label=None, fStd=0, color="orange", s=1, marker="o", alpha=1, conn=False):
    #print(data)
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    if(conn):
        ax.plot(x,y, color=color, label=label, marker=marker, alpha=alpha)
    else:
        ax.scatter(x,y, color=color, label=label, s=s, marker=marker, alpha=alpha)
    if(fStd):
        ax.fill_between(data["gen"], data["bestError"] - data["std"], data["bestError"] + data["std"], color="dark"+color, alpha=0.1)
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    #ax.set_ylim(bottom=0)
    ax.set_ylim(0, 100)
    #print(len(data["gen"]))
    ax.set_xlim(0, 100)
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
        text.set_fontsize(12)
    title = parameters["TITLE"]
    if(parameters["TITLE"] == 0):
        title = f"{parameters['ALGORITHM']} on {parameters['BENCHMARK']}"
    ax.set_title(title, fontsize=20)
    plt.savefig(f"{path}/scatter.png", format="png")
    plt.show()


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
    gop = pd.read_csv(f"{path}/optima.csv")

    fig, ax = configPlot(parameters)

    print( len(pd.unique(df["run"])) )
    data = [[] for i in range( len(pd.unique(df["run"])) )]
    best = [[] for i in range( len(pd.unique(df["run"])) )]
    part = [[] for i in range( len(pd.unique(df["run"])) )]
    for i in range(len(pd.unique(df["run"])) ):
        data[i] = df[df["run"] == i+1]
        best[i] = data[i].drop_duplicates(subset=["gen"])[["best"]].iloc[1:]
        best[i].reset_index(inplace=True)
        best[i] = best[i]["best"].values.tolist()
        part[i] = data[i]["part"].values.tolist()
        temp = [json.loads(item)[0:2] for item in part[i]]
        ax = plot(ax, data=temp, label=f"Run {i+1}", color=colors[i], s=5, alpha=0.3)
        temp = [json.loads(item)[0:2] for item in best[i]]
        ax = plot(ax, data=temp, color=colors[i], s=40, marker="X", conn=True)
        '''
        temp = [json.loads(item)[0:2] for item in gop[i]]
        if(i == len(pd.unique(df["run"]))-1):
            ax = plot(ax, data=temp, label=f"GOP", color="white", s=60, marker="s", conn=True)
        else:
            ax = plot(ax, data=temp, color="white", s=60, marker="s", conn=True)
        '''

    for k in range(gop.shape[1]):
        temp = gop[f"opt{k}"].values.tolist()
        for j in range(len(temp)):
            temp[j] = list(temp[j].split(", "))
            for i in range(len(temp[j])):
                temp[j][i] =  temp[j][i].replace("[", "")
                temp[j][i] =  temp[j][i].replace("(", "")
                temp[j][i] =  temp[j][i].replace(")", "")
                temp[j][i] =  temp[j][i].replace("]", "")
                temp[j][i] =  float(temp[j][i])

        print(temp)
        x = [[x[1], x[2]] for x in temp]
        if(k == 0):
            ax = plot(ax, data=x, label=f"GOP", color="white", s=60, marker="*", conn=True)
        else:
            ax = plot(ax, data=x, color="brown", s=60, marker="s", conn=True, alpha=0.5)


    showPlots(fig, ax, parameters)


if __name__ == "__main__":
    main()
