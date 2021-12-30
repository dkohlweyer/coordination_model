# Coordination Model

Version: January 2021

This is the source code of the model used in the paper *"Consumer Learning in a Market with Capacity Constraints"*. The model is used to analyze a scenario, where a number of consumers has to decide which supplier of a certain good he wants to visit in every iteration. Suppliers may be capacity constraint and consumers may be rationed. In order to make their choice, agents employ a meta-heuristic and learn the parameterization of the heuristic using one of three learning methods. (1) An imitation-based learning algorithm, (2) a genetic algorithm (GA) at the population leven, (3) a GA at the individual level. The model can be used to analyze and compare the outcome of the different learning algorithms.

## Getting Started

These instructions will allow you to run the model on your system.

### System Requirements and Installation

To run the code you need to install **[Python 3](https://www.python.org/downloads/)** and **[SciPy](https://scipy.org/install/)** (including numpy and matplotlib).

### Running The Model

The model has to be confiugured by writing a configuration file in the *config* folder. A configuration file specifies the model parameters and experiments. A set of pre-configured experiments can be found in the *config* folder. Important parameters are:

* *WORKSPACE* - Path where simulation data is stored.
* *n_c* - Number of consumers.
* *n_s* - Number of suppliers.
* *capacity* - Capacity per firm.
* *u_mu* - Mean utility per firm.
* *learning_rate* - Learning rate.
* *min* - Lower bound for gamma.
* *max* - Upper bound for gamma.
* *pl_mutation_rate* - mutation rate imitation-based learning (random exploration).
* *pl_n* - Number of observed agent in imitation-based learning.
* *pl_creep_factor* - Creep factor in imitation-based learning.
* *ga_pop_size* - Number of GA rules.
* *ga_new_rules* - Number of bew rules when learning.
* *ga_crossover_rate* - GA crossover rate
* *ga_mutation_rate* - GA mutation rate
* *ga_creep_factor* - GA creep factor
* *T* - number of iterations

To run one simulation, use the command

```
python launch.py <config_file> <task> <no_of_tasks>
```

*config_file* denotes the name of the configuration file in the *config* folder and has to be specified without the file extension (.py). In order to parralize execution, you can split execution in different chunks by specifying *<task>* *<no_of_tasks>* and launching several instances of the executable.

The full command for one of the pre-configured experiments without parralelization would be

```
python launch.py capacity_10 1 1
```

## Replication

The experiments presented in the paper can be replicated by executing all experiments from the *config* folder.

### Plotting

Plots can be created using the following command

```
julia ploy.py <config_file>
```

## Author

Dirk Kohlweyer