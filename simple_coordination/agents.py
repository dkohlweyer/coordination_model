from abpy import modelling
import math
import numpy
import random


class Consumer(modelling.Agent):
    def __init__(self, unique_id, model, gamma_init, learner):
        modelling.Agent.__init__(self, unique_id, model)
        self.utility_series = []
        self.gamma_series = [gamma_init]

        self.learner = learner

        self.gamma = gamma_init
        self.utility = 0
        self.prob_best = 0
        self.rationed = 0

        self.chosen_supplier = None

    def step(self):
        pass

    def choose_supplier(self):
        utilities = []
        for s in self.model.suppliers:
            utilities.append(s.utility)

        exp_values = list(map(lambda x: math.exp(x * self.gamma), utilities))
        sum_exp_values = sum(exp_values)
        probs = list(map(lambda x: x / sum_exp_values, exp_values))

        self.chosen_supplier = numpy.random.choice(self.model.suppliers, 1, False, probs)[0]
        self.prob_best = max(probs)

    def go(self):
        if self.chosen_supplier.ask():
            self.utility = self.chosen_supplier.utility
            self.rationed = 0
        else:
            self.utility = 0
            self.rationed = 1
        self.utility_series.append(self.utility)
        self.gamma_series.append(self.gamma)

    def learn(self):
        self.learner.learn(self)


class Supplier(modelling.Agent):

    def __init__(self, unique_id, model, capacity):
        modelling.Agent.__init__(self, unique_id, model)
        self.demand = 0
        self.price = 0
        self.utility = 0
        self.capacity = capacity

    def decide_price(self, mu, sigma):
        self.price = 0

    def ask(self):
        self.demand += 1
        if self.demand <= self.capacity:
            return True
        else:
            return False

    def reset(self):
        self.demand = 0

