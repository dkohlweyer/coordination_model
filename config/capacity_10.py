from abpy import management
from simple_coordination import model
from simple_coordination import agents

config = None

WORKSPACE = "./data/capacity-10"
RUNS = 100

n_c = 100
n_s = 5
capacity = 10
u_mu = 10
u_sigma = 2.0
learning_rate = 0.01

learning_alg = "PL"

min = 0
max = 25

pl_mutation_rate = 0.01
pl_n = 5
pl_z = 50
pl_creep_factor = 0.1

ga_pop_size = 40
ga_new_rules = 10
ga_crossover_rate = 0.95
ga_mutation_rate = 0.025
ga_creep_factor = 0.5

T = 20000

settingSGA = {"N_c": n_c, "N_s": n_s, "capacity": capacity, "u_mu": u_mu, "u_sigma": u_sigma, "gamma_min": min,
               "gamma_max": max, "learning_rate": learning_rate/n_c, "pl_mutation_rate": pl_mutation_rate, "pl_n": pl_n,
               "pl_z": pl_z, "pl_creep_factor": pl_creep_factor, "ga_pop_size": ga_pop_size,
               "ga_new_rules": ga_new_rules, "ga_crossover_rate":ga_crossover_rate, "ga_mutation_rate": ga_mutation_rate, "ga_creep_factor": ga_creep_factor,
               "learning_alg": "SGA", "redraw_utilities": True}

settingIGA = {"N_c": n_c, "N_s": n_s, "capacity": capacity, "u_mu": u_mu, "u_sigma": u_sigma, "gamma_min": min,
            "gamma_max": max, "learning_rate": learning_rate, "pl_mutation_rate": pl_mutation_rate, "pl_n": pl_n,
            "pl_z": pl_z, "pl_creep_factor": pl_creep_factor, "ga_pop_size": ga_pop_size,
            "ga_new_rules": ga_new_rules, "ga_crossover_rate": ga_crossover_rate, "ga_mutation_rate": ga_mutation_rate,
            "ga_creep_factor": ga_creep_factor,
            "learning_alg": "IGA", "redraw_utilities": True}

settingIMIT = {"N_c": n_c, "N_s": n_s, "capacity": capacity, "u_mu": u_mu, "u_sigma": u_sigma, "gamma_min": min,
            "gamma_max": max, "learning_rate": 0.01, "pl_mutation_rate": 0.01, "pl_n": pl_n,
            "pl_z": pl_z, "pl_creep_factor": pl_creep_factor, "ga_pop_size": ga_pop_size,
            "ga_new_rules": ga_new_rules, "ga_crossover_rate": ga_crossover_rate, "ga_mutation_rate": ga_mutation_rate,
            "ga_creep_factor": ga_creep_factor,
            "learning_alg": "PL", "redraw_utilities": True}

config = management.Configuration(WORKSPACE)

m_learning = model.LearningModel()
m_learning_ffw = model.LearningModel(ffw=4)

sSGA = management.Scenario('SGA', m_learning, settingSGA, T, RUNS)
sIGA = management.Scenario('IGA', m_learning, settingIGA, T, RUNS)
sIMIT = management.Scenario('IMIT', m_learning, settingIMIT, T, RUNS)

config.add_scenario(sSGA)
config.add_scenario(sIGA)
config.add_scenario(sIMIT)

# Variable Captures
config.add_variable_capture(management.AgentVariableCapture("gamma_series", lambda a: a.gamma, lambda a: type(a) is agents.Consumer))
config.add_variable_capture(management.AgentVariableCapture("utility_series", lambda a: a.utility, lambda a: type(a) is agents.Consumer))
config.add_variable_capture(management.AgentVariableCapture("rationing_series", lambda a: a.rationed, lambda a: type(a) is agents.Consumer))
config.add_variable_capture(management.AgentVariableCapture("prob_best_series", lambda a: a.prob_best, lambda a: type(a) is agents.Consumer,plot_single_ts=False, plot_single_agents=False, plot_hist_single=False, hist_steps=10000, hist_bins=100))
config.add_variable_capture(management.ModelVariableCapture("share_best_series", lambda m: m.share_best, plot_single_ts=False))
