import copy
import itertools
import time


class ModelRun:

    def __init__(self, model, t):
        self.model = model
        self.collectors = []
        self.t = t

    def add_data_collector(self, collector):
        self.collectors.append(collector)

    def run_model(self):
        self.model.initialize()
        self.model.populate()

        time_model = 0
        time_data = 0


        for tick in range(0, self.t):
            a = time.time()
            self.model.step(tick)
            b = time.time()
            time_model += (b-a)
            a = time.time()
            for c in self.collectors:
                if tick >= c.offset:
                    if (tick - c.offset) % c.interval == 0:
                        c.collect_data()
            b = time.time()
            time_data += (b-a)

    def reset(self):
        self.model.reset()
        for c in self.collectors:
            c.reset()


class Batch:

    def __init__(self, prototype_run, no_of_runs):
        self.runs = []
        self.no_of_runs = no_of_runs
        self.prototype_run = prototype_run

        for i in range(0, no_of_runs):
            self.runs.append(copy.deepcopy(self.prototype_run))

    def do_complete_batch(self):
        for r in self.runs:
            r.run_model()

    def do_specific_run(self, no_of_run):
        self.runs[no_of_run].run_model()


class ParameterSweep:
    def __init__(self, baseline_run, no_of_runs):
        self.baseline_run = baseline_run
        self.no_of_runs = no_of_runs
        self.values = {}
        self.access_functions = {}
        self.batches = {}

    def add_parameter_variation(self, name, access_function, values):
        self.access_functions[name] = access_function
        self.values[name] = values

    def generate_batches(self):
        keys = []
        l = []
        for k in self.values:
            keys.append(k)
            l.append(list(self.values[k]))
        perm = list(itertools.product(*l))

        for p in perm:
            self.batches[p] = []
            proto = copy.deepcopy(self.baseline_run)
            for i in range(0, len(p)):
                self.access_functions[keys[i]](proto.model, p[i])
            self.batches[p].append(Batch(proto, self.no_of_runs))








