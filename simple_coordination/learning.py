import random
import numpy
import copy


class AbstractLearner:

    def update(self, agent):
        pass

    def learn(self, agent):
        pass

    def change_rule(self, agent):
        pass

class NotLearner(AbstractLearner):

    def __init__(self):
        pass


class PeerLearner(AbstractLearner):

    def __init__(self, mutation_rate, min, max, n, z, creep_factor):
        self.mutation_rate = mutation_rate
        self.min = min
        self.max = max
        self.n = n
        self.z = z
        self.creep_factor = creep_factor
        self.utils_with_current_rule = []

    def change_rule(self, agent):
        pass

    def learn(self, agent):
        if random.random() < self.mutation_rate:
            new_gamma = self.mutate(agent)
        else:
            new_gamma = self.learn_by_peers(agent)

        agent.gamma = new_gamma

    def mutate(self, agent):
        r = numpy.random.uniform(-1, 1)
        s = self.creep_factor

        agent.utility_series = []
        new_gamma = min(self.max, max(self.min, agent.gamma + r * s * (self.max - self.min)))

        return new_gamma

    def learn_by_peers(self, agent):
        observed = random.sample(agent.model.consumers, self.n)

        my_util = numpy.mean(agent.utility_series)

        other_utils = []
        for c in observed:
            if len(c.utility_series) > 0:
                other_utils.append(numpy.mean(c.utility_series))

        if len(other_utils) > 0:
            max_util = max(other_utils)
            if max_util > my_util:
                i = other_utils.index(max_util)
                best = observed[i]
                new_gamma = best.gamma
                agent.utility_series = []
                return new_gamma
            else:
                return agent.gamma
        else:
            return agent.gamma


class GARatedRule:
    def __init__(self, rule):
        self.rule = rule
        self.fitness = []
        self.average_fitness = 1

    def get_avg_fitness(self):
        if len(self.fitness)==0:
            return 1
        if sum(self.fitness) / len(self.fitness)==0:
            return 1e-30
        return sum(self.fitness) / len(self.fitness)

    def update_fitness(self, fitness):
        self.fitness.append(fitness)
        self.average_fitness = self.get_avg_fitness()


class GALearning:
    def __init__(self, g_min, g_max, pop_size, no_new_rules, crossover_rate, mutation_rate, creep_factor):
        self.g_min = g_min
        self.g_max = g_max
        self.pop_size = pop_size
        self.no_new_rules = no_new_rules
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.creep_factor = creep_factor
        self.rated_rules = []

    def init_random_rules(self):
        while len(self.rated_rules) < self.pop_size:
            rule = numpy.random.uniform(self.g_min, self.g_max)
            self.rated_rules.append(GARatedRule(rule))

    def select(self, n=1):
        sum_fitness = sum(map(lambda r: r.average_fitness, self.rated_rules))

        probs = list(map(lambda r: r.average_fitness / sum_fitness, self.rated_rules))

        if n==1:
            id = numpy.random.choice(len(self.rated_rules), 1, False, probs)[0]
            return id, self.rated_rules[id]
        else:
            ids = numpy.random.choice(len(self.rated_rules), n, False, probs)
            return ids, numpy.array(self.rated_rules)[ids]

    def create_new_rules(self):
        no_created = 0
        while no_created < self.no_new_rules:
            self.rated_rules = sorted(self.rated_rules, key=lambda r: r.average_fitness)

            parent_ids, parent_rules = self.select(n=2)
            r1 = parent_rules[0]
            r2 = parent_rules[1]

            if random.random() < self.crossover_rate:
                new_rule1, new_rule2 = self.crossover(r1, r2)
            else:
                new_rule1 = r1
                new_rule2 = r2

            if random.random() < 0.5:
                new_rule = new_rule1
            else:
                new_rule = new_rule2

            if random.random() < self.mutation_rate:
                new_rule = self.mutate(new_rule)

            if not self.rule_exists(new_rule):
                replace = random.randint(0, self.no_new_rules)
                self.rated_rules[replace] = new_rule
                no_created += 1

    def crossover(self, r1, r2):
        a = random.random()

        new_rule1 = a*r1.rule + (1-a)*r2.rule
        new_rule2 = (1-a) * r1.rule + a * r2.rule
        new_fitness = (r1.average_fitness + r2.average_fitness)/2

        new_rated_rule1 = GARatedRule(new_rule1)
        new_rated_rule2 = GARatedRule(new_rule2)
        new_rated_rule1.update_fitness(new_fitness)
        new_rated_rule2.update_fitness(new_fitness)

        return new_rated_rule1, new_rated_rule2

    def mutate(self, rule):
        r = numpy.random.normal(loc=0,scale=0.1)

        new_rule = min(self.g_max,max(self.g_min, rule.rule + r*(self.g_max-self.g_min)))
        new_fitness = rule.average_fitness

        new_rated_rule = GARatedRule(new_rule)
        new_rated_rule.update_fitness(new_fitness)

        return new_rated_rule

    def get_rated_rule(self, rule):
        for rr in self.rated_rules:
            if rr.rule == rule:
                return rr
        raise ValueError()

    def rule_exists(self, rule):
        for rr in self.rated_rules:
            if rr.rule == rule.rule:
                return True
        return False

    def update_fitness(self, rule, fitness):
        rr = self.get_rated_rule(rule)
        rr.update_fitness(fitness)


class IndividualGALearner(AbstractLearner):

    def __init__(self, g_min, g_max, pop_size, no_new_rules, crossover_rate, mutation_rate, creep_factor):
        self.proto_ga = GALearning(g_min, g_max, pop_size, no_new_rules, crossover_rate, mutation_rate, creep_factor)
        self.crossover_rate = crossover_rate

        self.learners = {}

    def update(self, agent):
        if agent in self.learners:
            gamma = agent.gamma
            utility = agent.utility

            try:
                self.learners[agent].update_fitness(gamma, utility)
            except ValueError:
                pass

        else:
            self.learners[agent] = copy.deepcopy(self.proto_ga)
            self.learners[agent].init_random_rules()

    def learn(self, agent):
        self.learners[agent].create_new_rules()

    def change_rule(self, agent):
        id, rule = self.learners[agent].select()
        agent.gamma = rule.rule


class SocialGALearner:
    def __init__(self, g_min, g_max, pop_size, no_new_rules, crossover_rate, mutation_rate, creep_factor):
        self.social_ga = GALearning(g_min, g_max, pop_size, no_new_rules, crossover_rate, mutation_rate, creep_factor)
        self.crossover_rate = crossover_rate

        self.social_ga.init_random_rules()

    def update(self, agent):
        gamma = agent.gamma
        utility = agent.utility

        try:
            self.social_ga.update_fitness(gamma, utility)
        except ValueError:
            pass

    def learn(self, agent):
        self.social_ga.create_new_rules()

    def change_rule(self, agent):
        id, rule = self.social_ga.select()
        agent.gamma = rule.rule


















