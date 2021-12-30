import math
import numpy
from scipy import special
from decimal import *


class AnalyticalSolution:
    def __init__(self, n_c, n_s, capacity, gamma_max, precision):
        self.n_c = n_c
        self.n_s = n_s
        self.capacity = capacity
        self.gamma_max = gamma_max
        self.precision = precision

        self.use_decimal = True
        getcontext().prec = 64

    def calc_max_possible_utility(self, utilities):
        sorted_utilities = sorted(utilities, reverse=True)
        remaining_n = self.n_c

        i = 0
        u = 0
        while remaining_n > 0 and i < len(utilities):
            if remaining_n > self.capacity:
                u += self.capacity * sorted_utilities[i]
                remaining_n -= self.capacity
            else:
                u += remaining_n * sorted_utilities[i]
                remaining_n = 0
            i += 1

        return u

    def make_dec(self, utilities):
        return list(map(lambda u : Decimal(u), utilities))

    def calc_min_possible_utility(self, utilities):
        return min(utilities) * self.capacity

    def calc_opt_gamma(self, utilities):
        try:
            return self.bisection(lambda g: self.foc_optimized(Decimal(g), self.make_dec(utilities)), 0, self.gamma_max, self.precision)
        except RuntimeWarning:
            return self.gamma_max

    def calc_soc_gamma(self, utilities):
        try:
            return self.bisection(lambda g: self.social_foc_optimized(Decimal(g), self.make_dec(utilities)), 0, self.gamma_max, self.precision)
        except RuntimeWarning:
            return self.gamma_max

    def calc_utility(self, utilities, gamma):
        return self.utility(self.transform(gamma), self.make_dec(utilities))

    def calc_rationing(self, utilities, gamma):
        return self.rationing(self.transform(gamma), self.make_dec(utilities))

    def calc_prob_best(self, utilities, gamma):
        return self.prob_best(self.transform(gamma), self.make_dec(utilities))

    def sm(self, gamma, utils):
        exp_values = map(lambda u: self.exp((gamma * u)), utils)
        return sum(exp_values)

    def x(self, gamma, utils, util):
        return self.exp(gamma * util) / self.sm(gamma, utils)

    def q(self, gamma, utils, util):
        x = self.x(gamma, utils, util)

        return sum(map(lambda l: self.transform(special.binom(self.n_c - 1, l)) * x ** l * (1 - x) ** (
        self.n_c - l - 1) * self.transform(min(1, self.capacity / (1 + l))), range(0, self.n_c)))

    def q_dash(self, gamma, utils, util):
        x = self.x(gamma, utils, util)

        l = 0
        l_0 = (1 - x) ** (self.n_c - l - 1) * self.transform(min(1, self.capacity / (1 + l)))

        l = self.n_c - 1
        l_last = x ** l * self.transform(min(1, self.capacity / (1 + l)))

        return sum(map(lambda l: self.transform(special.binom(self.n_c - 1, l)) * x ** l * (1 - x) ** (
            self.n_c - l - 1) * self.transform(min(1, self.capacity / (1 + l))), range(1, self.n_c - 1))) + l_0 + l_last

    def foc(self, gamma, utils):
        return sum(map(lambda u: self.exp(gamma * u) * sum(
            map(lambda uu: (u - uu) * self.exp(gamma * uu) / self.sm(gamma, utils) ** 2, utils)) * self.q(gamma, utils,
                                                                                                          u) * u,
                       utils))

    def foc_optimized(self, gamma, utils):
        return sum(map(
            lambda u: self.exp(gamma * u) * sum(map(lambda uu: (u - uu) * self.exp(gamma * uu), utils)) * self.q_dash(
                gamma, utils, u) * u, utils))

    def dx(self, gamma, utils, util):
        sm = self.sm(gamma, utils)

        return (util * self.exp(gamma * util) * sm - self.exp(gamma * util) * sum(
            map(lambda u: u * self.exp(gamma * u), utils))) / sm ** 2

    def dq(self, gamma, prices, price):

        x = self.x(gamma, prices, price)
        dx = self.dx(gamma, prices, price)

        return sum(map(lambda l: self.transform(min(1, self.capacity / (1 + l)) * special.binom(self.n_c - 1, l)) * (
        l * x ** (l - 1) * dx * (1 - x) ** (self.n_c - 1 - l) + x ** l * (self.n_c - 1 - l) * (1 - x) ** (
        self.n_c - l - 2) * (-dx)), range(0, self.n_c)))

    def dq_dash(self, gamma, utils, util):
        x = self.x(gamma, utils, util)
        dx = self.dx(gamma, utils, util)

        l = 0
        l_0 = self.transform(min(1, self.capacity / (1 + l))) * (
        (self.n_c - 1 - l) * (1 - x) ** (self.n_c - l - 2) * (-dx))

        l = 1
        l_1 = self.transform(min(1, self.capacity / (1 + l)) * special.binom(self.n_c - 1, l)) * (
        l * dx * (1 - x) ** (self.n_c - 1 - l) + x ** l * (self.n_c - 1 - l) * (1 - x) ** (self.n_c - l - 2) * (-dx))

        l = self.n_c - 1
        l_last = self.transform(min(1, self.capacity / (1 + l)) * special.binom(self.n_c - 1, l)) * (
        l * x ** (l - 1) * dx)

        l = self.n_c - 2
        l_2last = self.transform(min(1, self.capacity / (1 + l)) * special.binom(self.n_c - 1, l)) * (
        l * x ** (l - 1) * dx * (1 - x) ** (self.n_c - 1 - l) + x ** l * (self.n_c - 1 - l) * (-dx))

        return sum(map(lambda l: self.transform(min(1, self.capacity / (1 + l)) * special.binom(self.n_c - 1, l)) * (
        l * x ** (l - 1) * dx * (1 - x) ** (self.n_c - 1 - l) + x ** l * (self.n_c - 1 - l) * (1 - x) ** (
        self.n_c - l - 2) * (-dx)), range(2, self.n_c - 2))) + l_last + l_0 + l_1 + l_2last

    def social_foc(self, gamma, utils):
        return self.foc(gamma, utils) + sum(
            map(lambda u: self.x(gamma, utils, u) * self.dq_dash(gamma, utils, u) * u, utils))

    def social_foc_optimized(self, gamma, utils):
        return sum(map(lambda u: (self.dx(gamma, utils, u) * self.q_dash(gamma, utils, u) + self.x(gamma, utils,
                                                                                                   u) * self.dq_dash(
            gamma, utils, u)) * u, utils))

    def utility(self, gamma, utils):
        return sum(map(lambda u: self.x(gamma, utils, u) * self.q_dash(gamma, utils, u) * u, utils))

    def prob_best(self, gamma, utils):
        max_util = max(utils)
        return self.x(gamma, utils, max_util)

    def rationing(self, gamma, utils):
        return 1 - sum(map(lambda u: self.x(gamma, utils, u) * self.q_dash(gamma, utils, u), utils))

    def bisection(self, f, min, max, precision):
        mid = min + (max - min) / 2

        if max - min < 2 * precision:
            return mid

        v_min = f(min)
        v_mid = f(mid)
        v_max = f(max)

        if self.sign(v_min) != self.sign(v_mid):
            return self.bisection(f, min, mid, precision)
        else:
            if self.sign(v_mid) != self.sign(v_max):
                return self.bisection(f, mid, max, precision)
            else:
                raise RuntimeWarning("No root found!")

    def reset(self):
        pass

    def exp(self, a):
        if not self.use_decimal:
            return math.exp(a)
        else:
            if type(a) is Decimal:
                return a.exp()
            else:
                raise AssertionError("Not supported " + str(type(a)))

    def sign(self, a):
        if type(a) is float or type(a) is numpy.float64 or type(a) is int:
            return numpy.sign(a)
        else:
            if type(a) is Decimal:
                return Decimal.copy_sign(Decimal(1), a)
            else:
                raise AssertionError("Not supported " + str(type(a)))

    def transform(self, a):
        if self.use_decimal:
            return Decimal(a)
        else:
            return a