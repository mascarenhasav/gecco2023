# Auto-Dynamic Particle Swarm Optimization (ADPSO)

## Work to do:
1. <del>Implementacao do benchmark dinamico (periodo variavel opcional)</del>
2. Definicao dos criterios de avaliacao dos resultados dos algoritmos
3. Programar um sistema de PSO que possa "ligar/desligar" os componentes daquela tabela que o alexandre apresentou semana passada atravez de um arquivo de configuracao.
4. Rodar testes dos algoritmos basicos contra o benchmark basico
5. Rodar um iRace no sistema configuravel acima para encontrar a melhor configuracao.
6. Analisar os resultados


## Features
It is possible to configure whether the Benchmark will have changes in the environment in fixed periods, or random periods. Periods are based on population generations.

### PSO the possible configurations are:
- GEN: Number of generations (integer);
- POPSIZE: Population size (integer);
- RUNS: Number of times the algorithm will run (integer);
- phi1: Parameter referring to the weight of the individual's contribution (float);
- phi2: Parameter referring to the contribution weight of the best individual in the flock (float);
- NDIM: Number of dimensions of the problem (integer);
- BOUNDS: Problem boundaries (list of integers);

### Benchmark (Moving Peak Benchmark) the configurable parameters are the following:
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
