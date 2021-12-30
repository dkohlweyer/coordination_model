from simple_coordination import agents
from simple_coordination import learning
import random
from abpy import modelling
from simple_coordination import learning as learning
import numpy
from scipy.stats import norm
import math
from simple_coordination import analytical

class LearningModel(modelling.Model, modelling.ModelParameterization):

    def __init__(self, ffw=1):
        super().__init__()
        modelling.ModelParameterization.__init__(self)
        self.consumers = []
        self.suppliers = []

        self.ffw = ffw

        self.n_c = 0
        self.n_s = 0
        self.capacity = 0
        self.u_mu = 0
        self.u_sigma = 0
        self.gamma_min = 0
        self.gamma_max = 0
        self.learning_rate = 0
        self.pl_mutation_rate = 0
        self.pl_n = 0
        self.pl_z = 0
        self.pl_creep_factor = 0
        self.ga_pop_size = 0
        self.ga_new_rules = 0
        self.ga_crossover_rate = 0
        self.ga_mutation_rate = 0
        self.ga_creep_factor = 0
        self.learning_alg = 0
        self.redraw_utilities = False

        self.share_best = 0

    def initialize(self):
        self.n_c = self.params["N_c"]
        self.n_s = self.params["N_s"]
        self.capacity = self.params["capacity"]
        self.u_mu = self.params["u_mu"]
        self.u_sigma = self.params["u_sigma"]
        self.gamma_min = self.params["gamma_min"]
        self.gamma_max = self.params["gamma_max"]
        self.learning_rate = self.params["learning_rate"]
        self.pl_mutation_rate = self.params["pl_mutation_rate"]
        self.pl_n = self.params["pl_n"]
        self.pl_z = self.params["pl_z"]
        self.pl_creep_factor = self.params["pl_creep_factor"]
        self.ga_pop_size = self.params["ga_pop_size"]
        self.ga_new_rules = self.params["ga_new_rules"]
        self.ga_crossover_rate = self.params["ga_crossover_rate"]
        self.ga_mutation_rate = self.params["ga_mutation_rate"]
        self.ga_creep_factor = self.params["ga_creep_factor"]
        self.learning_alg = self.params["learning_alg"]
        self.redraw_utilities = self.params["redraw_utilities"]


    def populate(self):

        if self.learning_alg == "PL":
            learner = learning.PeerLearner(self.pl_mutation_rate, self.gamma_min, self.gamma_max, self.pl_n, self.pl_z, self.pl_creep_factor)
        if self.learning_alg == "IGA":
            learner = learning.IndividualGALearner(self.gamma_min, self.gamma_max, self.ga_pop_size, self.ga_new_rules, self.ga_crossover_rate, self.ga_mutation_rate, self.ga_creep_factor)
        if self.learning_alg == "SGA":
            learner = learning.SocialGALearner(self.gamma_min, self.gamma_max, self.ga_pop_size, self.ga_new_rules, self.ga_crossover_rate, self.ga_mutation_rate, self.ga_creep_factor)
        if self.learning_alg == "NO":
            learner = learning.NotLearner()

        # Create Supplier
        for i in range(0, self.n_s):
            s = agents.Supplier("S" + str(i), self,  self.capacity)
            self.suppliers.append(s)
            self.add_agent(s)

        self.set_utilities()

        # Create Consumer
        for j in range(0, self.n_c):
            g = numpy.random.uniform(self.gamma_min, self.gamma_max)
            c = agents.Consumer("C" + str(j), self, g, learner)
            self.consumers.append(c)
            self.add_agent(c)

    def step(self, tick):
        for i in range(0, self.ffw):
            # 1. Decide prices
            # SKIPPED

            # 1. Set utilities
            if self.redraw_utilities:
                self.set_utilities()

            # 2. Consumer: Choose Supplier
            for c in self.consumers:
                c.choose_supplier()

            # 3. Go!
            c_shuffled = self.consumers
            random.shuffle(c_shuffled)
            for c in c_shuffled:
                c.go()

            self.observe_share_best()

            # 4. Learning
            for c in self.consumers:
                c.learner.update(c)
                if random.random() < self.learning_rate:
                    c.learner.learn(c)
                c.learner.change_rule(c)

            for s in self.suppliers:
                s.reset()

    def set_utilities(self):
        utilities=[11.9595, 11.9962, 14.5368, 15.1159, 8.17484]
        c=0

        for s in self.suppliers:
            if not self.redraw_utilities:
                s.utility = utilities[c]
                c+=1
            else:
                s.utility = numpy.random.normal(self.u_mu, self.u_sigma)

    def observe_share_best(self):
        current_best_util = 0
        current_best_supplier = None

        for s in self.suppliers:
            if s.utility > current_best_util:
                current_best_supplier = s
                current_best_util = s.utility

        self.share_best = current_best_supplier.demand / len(self.consumers)

    def reset(self):
        self.consumers = []
        self.suppliers = []

    def init_random(self, seed=None):
        random.seed(seed)
        numpy.random.seed(seed)


class AnalyticalModel(modelling.Model, modelling.ModelParameterization):
    def __init__(self, mean_of=1):
        super().__init__()
        modelling.ModelParameterization.__init__(self)

        self.mean_of = mean_of
        self.agent = None

    def initialize(self):
        self.n_c = self.params["N_c"]
        self.n_s = self.params["N_s"]
        self.capacity = self.params["capacity"]
        self.u_mu = self.params["u_mu"]
        self.u_sigma = self.params["u_sigma"]
        self.gamma_max = self.params["gamma_max"]
        self.regime = self.params["regime"]

        gammaL=[]
        rationingL=[]
        utilityL=[]
        prob_bestL=[]

        for i in range(0, self.mean_of):
            self.utils = []

            for i in range(0,self.n_s):
                self.utils.append(numpy.random.normal(self.u_mu, self.u_sigma))

            anal = analytical.AnalyticalSolution(self.n_c, self.n_s, self.capacity, self.gamma_max, 0.00001)

            if self.regime == "NASH":
                gamma1 = anal.calc_opt_gamma(self.utils)
            if self.regime == "SOCIAL":
                gamma1 = anal.calc_soc_gamma(self.utils)

            rationing1 = anal.calc_rationing(self.utils, gamma1)
            utility1 = anal.calc_utility(self.utils, gamma1)
            prob_best1 = anal.calc_prob_best(self.utils, gamma1)

            gammaL.append(gamma1)
            rationingL.append(rationing1)
            utilityL.append(utility1)
            prob_bestL.append(prob_best1)

        self.gamma = numpy.mean(gammaL)
        self.gamma_var = numpy.var(gammaL)
        self.rationing = numpy.mean(rationingL)
        self.utility=numpy.mean(utilityL)
        self.share_best = numpy.mean(prob_bestL)

        self.gamma_lower_bound = self.gamma - norm.ppf(0.975)*math.sqrt(self.gamma/len(gammaL))
        self.gamma_upper_bound = self.gamma + norm.ppf(0.975)*math.sqrt(self.gamma/len(gammaL))

        print(self.gamma)
        print(self.gamma_var)
        print("[" + str(self.gamma_lower_bound) + ", " + str(self.gamma_upper_bound) + "]")

    def populate(self):
        self.agent = agents.Consumer("JohnNash", self, self.gamma, None)
        self.agent.utility = self.utility
        self.agent.rationed = self.rationing
        self.add_agent(self.agent)

    def step(self):
        pass

    def reset(self):
        self.agent = None

    def init_random(self, seed=None):
        random.seed(seed)
        numpy.random.seed(seed)



