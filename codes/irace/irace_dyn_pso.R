rm(list = ls(all = TRUE))

library(jsonlite)

setwd("~/pso_dynamic/gecco2023/codes/algorithms/adpso/")

options(scipen = 999)

suppressPackageStartupMessages(library(irace))
suppressPackageStartupMessages(library(reticulate))
suppressPackageStartupMessages(library(parallel))

#use_python("/Users/yurilavinas/miniforge3/bin/python")




scenario                <- irace::defaultScenario()
scenario$seed           <- 165214 # Seed for the experiment
scenario$targetRunner   <-
  "target.runner" # Runner function (def. below)
scenario$forbiddenFile  <-
  "../../irace/forbidden.txt" # forbidden configs
scenario$debugLevel     <- 0
scenario$maxExperiments <- 20000 # Tuning budget
scenario$testNbElites   <-
  1     # test all final elite configurations
scenario$parallel       <- 8
scenario$digits       <- 2


source_python("ADPSO.py")
source_python("../../analysis/metrics.py")


# Read tun-able parameter list from file
parameters <- readParameters("../../irace/parameters.txt")

#===============
### Build training instances
fname1 <- (paste0("mpb_", c(8,10))) #mpb + num of peaks
dimensions1 = c(8,10) # dimensions
severity1 = c(1,3,5)
allfuns1          <-
  expand.grid(fname1, dimensions1, severity1, stringsAsFactors = FALSE)

scenario$instances <- paste0(allfuns1[, 1], "_", allfuns1[, 2], "_", allfuns1[, 3])

### Build test instances
fname2 <- (paste0("mpb_", c(7,9))) #mpb + num of peaks
dimensions2 = c(7,9) # dimensions
severity1 = c(2,4,6)
allfuns2            <-
  expand.grid(fname2, dimensions2, stringsAsFactors = FALSE)

scenario$testInstances <- paste0(allfuns2[, 1], "_", allfuns2[, 2])

target.runner <- function(experiment, scenario) {
  force(experiment)
  
  conf <- experiment$configuration
  inst <- experiment$instance
  
  ### pso par
  runs = 1
  nevals = 500000
  
  ### mpb par
  change.evals = c(5000, 15000, 25000, 30000, 40000)
  bounds.pos = c(0, 100)
  bounds.vel = c(-5, 5)
  change.dec = 1
  change = 1
  random.changes = 1
  range.changes = c(20, 50000)
  period = 0
  uni.height = 0
  min.height = 30
  max.height = 70
  min.coord = 0
  max.coord = 100
  
  #getting peaks and dimensions
  problem.info = strsplit(inst, "mpb_")[[1]][2]
  # creating config.ini list to convert to json so that the algo can have an input
  conf.list = list(
    CHANGES_NEVALS = change.evals,
    POPSIZE = as.integer(conf$POPSIZE),
    NSWARMS = as.integer(conf$NSWARMS),
    ES_PARTICLE_OP = as.integer(conf$ES_PARTICLE_OP),
    EXCLUSION_OP = as.integer(conf$EXCLUSION_OP),
    ANTI_CONVERGENCE_OP = as.integer(conf$ANTI_CONVERGENCE_OP),
    MOVE_SEVERITY_MPB = as.numeric(strsplit(problem.info, "_")[[1]][3]),
    RCLOUD = as.numeric(conf$RCLOUD),
    REXCL = as.numeric(conf$REXCL),
    RCONV = as.numeric(conf$RCONV),
    NDIM = as.numeric(strsplit(problem.info, "_")[[1]][2]),
    NPEAKS_MPB = as.numeric(strsplit(problem.info, "_")[[1]][1]),
    phi1 = as.numeric(conf$phi1),
    phi2 = as.numeric(conf$phi2),
    LOCAL_SEARCH_OP = as.integer(conf$LOCAL_SEARCH_OP),
    ES_PARTICLE_PERC = as.numeric(conf$ES_PARTICLE_PERC),
    RLS = as.numeric(conf$RLS),
    ETRY = as.integer(conf$ETRY),
    ALGORITHM = "ADPSO",
    BENCHMARK = "MPB",
    NEVALS = nevals,
    BOUNDS_POS = bounds.pos,
    BOUNDS_VEL = bounds.vel,
    CHANGE_DETECTION_OP = change.dec,
    CHANGE = change,
    RANDOM_CHANGES = random.changes, # don't change
    RANGE_NEVALS_CHANGES = range.changes,
    NCHANGES = length(change.evals),
    PERIOD_MPB = period,
    UNIFORM_HEIGHT_MPB = uni.height,
    MIN_HEIGHT_MPB = min.height,
    MAX_HEIGHT_MPB = max.height,
    MIN_COORD_MPB = min.coord,
    MAX_COORD_MPB = max.coord,
    PATH = "../../../experiments", # don't change
    FILENAME = "data.csv", # don't change
    PLOT = 0, # don't change
    DEBUG = 0, # don't change
    RUNS = runs, # don't change
    ES_CHANGE_OP = 0 # don't change
  )

  # create the json
  config.file = toJSON(conf.list,pretty=TRUE,auto_unbox=TRUE)
  
  
  
  
  # creating unique folder based on time for parallel execs
  seed = gsub("[: -]", "" , Sys.time(), perl=TRUE)
  path = paste0("../../../experiments/irace/",seed)
  while(dir.create(path)==FALSE){
    Sys.sleep(1) #for parallel execs!
    seed = gsub("[: -]", "" , Sys.time(), perl=TRUE)
    path = paste0("../../../experiments/irace/",seed)
  }
  
  #saving the config.ini in the right folder
  write(config.file,paste0(path, '/config.ini'))
  
  # running the algorithm
  call_adpso(path)
  
  ## get metric value
  out = offlineError(path, std = 0)
  
  print(out)
  
  ## remove folder after exp for memory reasons
  unlink(path, recursive=TRUE)
  
  ### return metric
  return(list(cost = out))
  
}

dynamic.results <- irace(scenario, parameters)
saveRDS(dynamic.results, "../../irace/results/irace-tuning_adpso.rds")
file.copy(from = "./irace.Rdata", to = "../../irace/results/irace-training_adpso.Rdata")

testing.main(logFile = "../../irace/results/irace-training_adpso.Rdata")
file.copy(from = "./irace.Rdata", to = "../../irace/results/irace-testing_adpso.Rdata")
