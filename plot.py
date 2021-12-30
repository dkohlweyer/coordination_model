import sys
import os
import importlib
import time
from abpy import management
import numpy

if __name__ == "__main__":
    config_file = sys.argv[1]

    c = importlib.import_module("config." + config_file)

    print(time.strftime("%I:%M:%S", time.localtime()) + " PLOTTING: " + config_file, flush=True)

    em = management.ExecutionManager(c.config, 1, True)

    em.config.agent_aggregator_functions = {"AVG": lambda v: numpy.mean(v), "MED": lambda v: numpy.median(v)}
    em.config.run_aggregator_functions = {"AVG": lambda v: numpy.mean(v), "MED": lambda v: numpy.median(v)}

    em.generate_runs()

    em.plot_scenario_comparison()
    #em.plot_timeseries()
    #em.plot_param_var()
    #em.plot_histograms()



    print(time.strftime("%I:%M:%S", time.localtime()) + " TERMINATED: " + config_file, flush=True)




