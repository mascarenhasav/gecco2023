# Framework for Dynamic Particle Swarm Optimization-Evolutionary Strategy (DPSO-ES)


## Description
The DPSO-ES framework was developed in Python using the DEAP library to test different components present in the literature on Dynamic Evolutionary Algorithms (DEA) that are used in optimization of dynamic problems (DOP). It allows the components to be turned on/off in order to test the effectiveness of each one of them independently in a given problem.
Another characteristic is the possibility of configuring the parameters of both the optimizers (PSO, ES) and the characteristics of the Benchmark, which until now consists of the Moving Peak Benchmark (MPB).

## Contents

This repository contains both the framework code in the "codes/algorithms/adpso" folder and codes related to the analysis of experimental data:

In "codes/plot" are the codes responsible for generating the performance graphs of the algorithms;

In "codes/analysis" are the codes responsible for calculating the metrics of the execution of the algorithms.

### General parameters:
- RUNS: Number of runs;
- NEVALS: Number of Evaluations of each run;
- POPSIZE: Population size (integer);
- NDIM: Number of dimensions of the problem (integer);
- BOUNDS: Problem boundaries (list of integers);

### PSO the possible configurations are:
- phi1 (PSO): Parameter referring to the weight of the individual's contribution (float);
- phi2 (PSO): Parameter referring to the contribution weight of the best individual in the flock (float);

### Benchmark (Moving Peak Benchmark) parameters:
- CHANGE: If there will be changes in the environment (bool);
- RANDOM_CHANGES: Whether the changes will be random or not (bool);
- RANGE_GEN_CHANGES: Range of allowed values for random changes to occur (list of integers);
- NCHANGES: Number of random changes (integer);
- CHANGES_GEN: If the changes are manual, these are the values of the generations in which they will occur (list of integers);
- NPEAKS_MPB: Number of benchmark peaks (integer);
- UNIFORM_HEIGHT_MPB: Initial value of the peaks (integer);
- MAX_HEIGHT_MPB: Maximum value for peaks (integer);
- MIN_HEIGHT_MPB: Minimum value for peaks (integer);
- MOVE_SEVERITY_MPB: Intensity of the change in the position of the peaks when there is a change of environment (float);


## Work to do:
1. <del>Implementacao do benchmark dinamico (periodo variavel opcional)</del>
2. Definicao dos criterios de avaliacao dos resultados dos algoritmos
3. Programar um sistema de PSO que possa "ligar/desligar" os componentes daquela tabela que o alexandre apresentou semana passada atravez de um arquivo de configuracao.
4. Rodar testes dos algoritmos basicos contra o benchmark basico
5. Rodar um iRace no sistema configuravel acima para encontrar a melhor configuracao.
6. Analisar os resultados
