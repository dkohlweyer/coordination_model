from abpy import persistence
from abpy import running
from abpy import datacollection
from abpy import plotting
import copy
import itertools
import numpy
import random
import time


class Configuration:
    def __init__(self, workspace, data_folder="DATA", plot_folder="PLOTS", single_run_folder="SINGLE",
                 aggregate_folder="AGGREGATED", hist_folder="HIST", combined_folder="ALL", final_values_folder="FINAL",
                 plot_ts=True, plot_hist=True, plot_param_var=True, plot_scenario_comparison=True):
        self.workspace = workspace
        self.single_run_folder = single_run_folder
        self.aggregate_folder = aggregate_folder
        self.data_folder = data_folder
        self.plot_folder = plot_folder
        self.plot_ts = plot_ts
        self.plot_hist = plot_hist
        self.hist_folder = hist_folder
        self.combined_folder = combined_folder
        self.final_values_folder = final_values_folder
        self.plot_param_var = plot_param_var
        self.plot_scenario_comparison = plot_scenario_comparison

        persistence.ensure_dir(workspace)
        persistence.ensure_dir(workspace + "/" + data_folder)
        persistence.ensure_dir(workspace + "/" + plot_folder)
        persistence.ensure_dir(workspace + "/" + data_folder + "/" + single_run_folder)
        persistence.ensure_dir(workspace + "/" + plot_folder + "/" + aggregate_folder)

        self.scenarios = []
        self.captures = []

        self.agent_aggregator_functions = {"MIN": lambda v: min(v), "MAX": lambda v: max(v),
                                           "AVG": lambda v: numpy.mean(v), "MED": lambda v: numpy.median(v),
                                           "VAR": lambda v: numpy.var(v)}
        self.run_aggregator_functions = self.agent_aggregator_functions#{"AVG": lambda v: numpy.mean(v)}

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def add_variable_capture(self, capture):
        self.captures.append(capture)

    def add_parameter_variations(self, name, values):
        for s in self.scenarios:
            s.add_parameter_variation(name, values)


class ModelVariableCapture:
    def __init__(self, name, collector_function, interval=1, offset=0, run_aggregation=True, plot_single_ts=True, plot_aggregated_ts=True, plot_param_var=True, plot_scenario_comparison=True, plot_final_values=True, avgs_of_final_values=[1,10,50]):
        self.name = name
        self.collector_function = collector_function
        self.interval = interval
        self.offset = offset
        self.run_aggregation = run_aggregation
        self.plot_single_ts = plot_single_ts
        self.plot_aggregated_ts = plot_aggregated_ts
        self.plot_final_values = plot_final_values
        self.plot_param_var = plot_param_var
        self.plot_scenario_comparison = plot_scenario_comparison
        self.avgs_of_final_values = avgs_of_final_values


class AgentVariableCapture(ModelVariableCapture):
    def __init__(self, name, collector_function, filter_function, interval=1, offset=0, run_aggregation=True,
                 agent_aggregation=True, plot_single_ts=True, plot_aggregated_ts=True, plot_param_var=True, plot_scenario_comparison=True, plot_final_values=True, avgs_of_final_values=[1,10,50], plot_single_agents=True,
                 plot_aggregated_agents=True, plot_hist_single=True, plot_hist_aggregated=True, hist_bins=10, hist_steps=1):
        super().__init__(name, collector_function, interval, offset, run_aggregation, plot_single_ts, plot_aggregated_ts,plot_param_var, plot_scenario_comparison, plot_final_values, avgs_of_final_values)
        self.agent_aggregation=agent_aggregation
        self.filter_function = filter_function
        self.plot_single_agents = plot_single_agents
        self.plot_aggregated_agents = plot_aggregated_agents
        self.plot_hist_aggregated = plot_hist_aggregated
        self.plot_hist_single = plot_hist_single
        self.hist_bins = hist_bins
        self.hist_steps = hist_steps


class Scenario:
    def __init__(self, name, model_prototype, parameter_setting, no_steps, no_runs):
        self.name = name
        self.model = model_prototype
        self.parameters = parameter_setting
        self.no_steps = no_steps
        self.no_runs = no_runs
        self.variations = {}

    def add_parameter_variation(self, name, values):
        self.variations[name] = values


class ExecutionManager:
    def __init__(self, configuration, no_threads, skip_existing_runs):
        self.config = configuration
        self.no_threads = no_threads
        self.skip_existing_runs = skip_existing_runs
        self.runs = []
        self.base_dirs = {}

    def generate_runs(self):
        for s in self.config.scenarios:
            if len(s.variations) == 0:
                self.generate_runs_basic_scenarios(s)
            else:
                self.generate_runs_param_variations(s)

    def build_collectors(self, model):
        collectors = []
        for c in self.config.captures:
            collectors.extend(self.build_collectors_by_capture(c, model))

        return collectors

    def build_collectors_by_capture(self, capture, model):
        collectors = []

        if type(capture) is ModelVariableCapture:
            collectors.append(datacollection.ModelDataCollector(capture.name, model, capture.collector_function, capture.interval, capture.offset))

        if type(capture) is AgentVariableCapture:
            collectors.append(datacollection.FullAgentDataCollector(capture.name,model,capture.collector_function, capture.filter_function, capture.interval, capture.offset))
            if capture.agent_aggregation:
                for a in self.config.agent_aggregator_functions.keys():
                    collectors.append(datacollection.AggregateAgentDataCollector(capture.name + "_" + a,model,capture.collector_function, self.config.agent_aggregator_functions[a], capture.filter_function, capture.interval, capture.offset))

        return collectors

    def build_runs(self, base_dir, scenario, params):
        for i in range(0, scenario.no_runs):
            model = copy.deepcopy(scenario.model)
            model.params = params
            self.runs.append(ManagedRun(self.config.workspace + '/' + self.config.data_folder + "/" + self.config.single_run_folder + "/" + base_dir, i, model, scenario.no_steps, self.build_collectors(model)))

    def generate_runs_basic_scenarios(self, scenario):
        base_dir = scenario.name
        self.build_runs(base_dir,scenario,scenario.parameters)
        self.add_base_dir(scenario, base_dir)

    def generate_runs_param_variations(self, scenario):
        params = []
        values = []
        for param in sorted(scenario.variations.keys()):
            params.append(param)
            values.append(list(scenario.variations[param]))
        permutations = list(itertools.product(*values))

        for p in permutations:
            base_dir = scenario.name + '/'
            current_params = copy.deepcopy(scenario.parameters)

            for x in range(0, len(params)):
                current_params[params[x]] = p[x]
                base_dir = base_dir + params[x] + str(p[x])

            self.build_runs(base_dir,scenario,current_params)
            self.add_base_dir(scenario, base_dir)

    def add_base_dir(self, scenario, base_dir):
        if scenario in self.base_dirs:
            self.base_dirs[scenario].append(base_dir)
        else:
            self.base_dirs[scenario] = [base_dir]

    def execute_complete(self):
        for r in self.runs:
            self.execute_single_run(r)

    def execute_share(self, no, no_shares, shuffle=False):
        sorted_runs = list(sorted(self.runs, key=lambda r: r.base_dir + str(r.i)))
        if shuffle:
            random.seed(0)
            random.shuffle(sorted_runs)

        shares = {}
        s_id=0

        for run in sorted_runs:
            if s_id not in shares:
                shares[s_id] = []
            shares[s_id].append(run)

            s_id+=1
            if s_id==no_shares:
                s_id=0

        for run in shares[no]:
            self.execute_single_run(run)

    def run_completed(self, run):
        return persistence.check_dir_exists(run.dir)

    def execute_single_run(self, run):
        if not (self.skip_existing_runs and persistence.check_dir_exists(run.base_dir + "/" + str(run.i))):
            print(time.strftime("%I:%M:%S", time.localtime()) + " PROCESSING: " + run.base_dir + "/" + str(run.i), flush=True)
            a = time.time()
            run.execute()
            b=time.time()
            print(time.strftime("%I:%M:%S", time.localtime()) + " DONE: " + run.base_dir + "/" + str(run.i) + " [" + str(int(b-a)) + "s]",
              flush=True)

        else:
            print(time.strftime("%I:%M:%S", time.localtime()) + " SKIPPED: " + run.base_dir + "/" + str(run.i), flush=True)

    def aggregate_share(self, no, no_shares):
        steps = []
        for s in self.config.scenarios:
            for dir in self.base_dirs[s]:
                for c in self.config.captures:
                    steps.append([c, s, dir])

        sorted_steps = list(sorted(steps, key=lambda s: s[0].name + s[1].name + s[2]))

        shares = {}
        s_id = 0

        for s in sorted_steps:
            if s_id not in shares:
                shares[s_id] = []
            shares[s_id].append(s)

            s_id += 1
            if s_id == no_shares:
                s_id = 0

        for s in shares[no]:
            print(time.strftime("%I:%M:%S", time.localtime()) + " PROCESSING: " + s[0].name + ", "+ s[2], flush=True)
            a = time.time()
            self.aggregate_capture(s[0], s[1], s[2])
            b = time.time()
            print(time.strftime("%I:%M:%S", time.localtime()) + " DONE: " + s[0].name + ", "+ s[2] + " [" + str(int(b - a)) + "s]", flush=True)

    def aggregate(self):
        for s in self.config.scenarios:
            for dir in self.base_dirs[s]:
                for c in self.config.captures:
                    if c.run_aggregation:
                        print(" --- " + dir, flush=True)
                        self.aggregate_capture(c, s, dir)

    def aggregate_capture(self, capture, scenario, dir):
        self.aggregate_datacollector(dir, capture.name, scenario.no_runs)
        if type(capture) is AgentVariableCapture and capture.agent_aggregation:
            for a in self.config.agent_aggregator_functions.keys():
                self.aggregate_datacollector(dir, capture.name + "_" + a, scenario.no_runs)

    def aggregate_datacollector(self, dir, name, no_runs):
        dcs = []
        for i in range(0, no_runs):
            dcs.append(persistence.create_datacollector_from_csv_file(
                self.config.workspace + '/' + self.config.data_folder + "/" + self.config.single_run_folder + "/" + dir + '/' + str(
                    i) + '/' + name + ".csv"))

        for a in self.config.run_aggregator_functions.keys():
            data_aggregator = datacollection.DataAggregator(name + "_" + a, self.config.run_aggregator_functions[a])
            aggregated_dc = data_aggregator.aggregate(dcs)
            persistence.write_datacollector_to_csv_file(
                self.config.workspace + '/' + self.config.data_folder + "/" + self.config.aggregate_folder + "/" + dir + '/' + a + '/' + name + ".csv",
                aggregated_dc)

    def plot(self):
        if self.config.plot_ts:
            self.plot_timeseries()
        if self.config.plot_hist:
            self.plot_histograms()
        if self.config.plot_param_var:
            self.plot_param_var()
        if self.config.plot_scenario_comparison:
            self.plot_scenario_comparison()

    def plot_timeseries(self):
        for c in self.config.captures:
            print("PLOTTING TS: " + c.name, flush=True)
            for s in self.config.scenarios:
                for dir in self.base_dirs[s]:
                    print(" --- " + dir, flush=True)
                    if type(c) is ModelVariableCapture:
                        self.plot_model_timeseries(c, dir, s.no_runs)
                    if type(c) is AgentVariableCapture:
                        self.plot_agent_timeseries(c, dir, s.no_runs)

    def plot_model_timeseries(self, capture, dir, no_runs):
        if capture.plot_single_ts:
            self.plot_timeseries_single(capture.name, dir, no_runs)
        if capture.plot_aggregated_ts:
            self.plot_timeseries_aggregate(capture.name, dir)

    def plot_agent_timeseries(self, capture, dir, no_runs):
        if capture.plot_single_ts:
            if capture.plot_single_agents:
                self.plot_timeseries_single(capture.name, dir, no_runs)
            if capture.plot_aggregated_agents:
                dict_names = {}
                for a in self.config.agent_aggregator_functions.keys():
                    self.plot_timeseries_single(capture.name + "_" + a, dir, no_runs)
                    dict_names[a] = capture.name + "_" + a
                if capture.plot_aggregated_ts:
                    self.plot_timeseries_single_combined(capture.name + "_ALL", dict_names, dir, no_runs)
        if capture.plot_aggregated_ts:
            if capture.plot_single_agents:
                self.plot_timeseries_aggregate(capture.name, dir, legend=False)
            if capture.plot_aggregated_agents:
                dict_names = {}
                for a in self.config.agent_aggregator_functions.keys():
                    self.plot_timeseries_aggregate(capture.name + "_" + a, dir)
                    dict_names[a] = capture.name + "_" + a
                #if capture.plot_aggregated_ts:
                    #self.plot_timeseries_aggregate_combined(capture.name + "_ALL", dict_names, dir)

    def plot_timeseries_single(self, name, dir, no_runs, legend=True):
        plot_all = plotting.TimeSeriesPlot()
        for i in range(0, no_runs):
            dc = persistence.create_datacollector_from_csv_file(
                self.config.workspace + '/' + self.config.data_folder + "/" + self.config.single_run_folder + "/" + dir + '/' + str(
                    i) + '/' + name + ".csv")
            plot = plotting.TimeSeriesPlot()
            plot.add_series(dc)
            plot_all.add_series(dc)
            plot.save_plot(
                self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.single_run_folder + "/" + dir + '/' + str(
                    i) + '/' + name + ".pdf", legend=legend)

        plot_all.save_plot(
            self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.single_run_folder + "/" + dir + '/' + self.config.combined_folder + '/' + name + ".pdf", legend=False)

    def plot_timeseries_aggregate(self, name, dir, legend=True):
        for a in self.config.run_aggregator_functions.keys():
            dc = persistence.create_datacollector_from_csv_file(
                self.config.workspace + '/' + self.config.data_folder + "/" + self.config.aggregate_folder + "/" + dir + '/' + a + '/' + name + ".csv")
            plot = plotting.TimeSeriesPlot()
            plot.add_series(dc, a)
            plot.save_plot(
                self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.aggregate_folder + "/" + dir + '/' + a + '/' + name + ".pdf", legend=legend)

    def plot_histograms(self):
        for c in self.config.captures:
            print("PLOTTING HIST: " + c.name, flush=True)
            for s in self.config.scenarios:
                for dir in self.base_dirs[s]:
                    print(" --- " + dir, flush=True)
                    if type(c) is AgentVariableCapture:
                        if c.plot_hist_single or c.plot_hist_aggregated:
                            for i in range(0,s.no_steps,c.hist_steps):
                                self.plot_histogram(c.name, i, dir, s.no_runs, c.plot_hist_single, c.plot_hist_aggregated, c.hist_bins)
                            self.plot_histogram(c.name, s.no_steps-1, dir, s.no_runs, c.plot_hist_single, c.plot_hist_aggregated, c.hist_bins)

    def plot_histogram(self, name, i, dir, no_runs, single, aggregated, bins):
        plot_all = plotting.HistogramPlot(bins=bins)
        for j in range(0, no_runs):
            dc = persistence.create_datacollector_from_csv_file(
                self.config.workspace + '/' + self.config.data_folder + "/" + self.config.single_run_folder + "/" + dir + '/' + str(
                    j) + '/' + name + ".csv")
            plot_all.add_data(dc, i=i)
            if single:
                plot = plotting.HistogramPlot(bins=bins)
                plot.add_data(dc, i=i)
                plot.save_plot(
                    self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.single_run_folder + "/" + dir + '/' + str(
                        j) + "/" + self.config.hist_folder +"/" + name + "_" + str(i) + ".pdf")
        if aggregated:
            plot_all.save_plot(
                self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.aggregate_folder + "/" + dir + '/' + self.config.hist_folder +"/"+ name + "_" + str(i) + ".pdf")

    def plot_param_var(self):
        for s in self.config.scenarios:
            if len(s.variations) == 1:
                key = list(s.variations.keys())[0]
                for c in self.config.captures:
                    print("PLOTTING PARAMVAR: " + c.name, flush=True)
                    if type(c) is ModelVariableCapture:
                        if c.plot_param_var:
                            self.plot_variation_series(key,c.name,s.name,s.variations[key])
                            if c.plot_final_values:
                                self.plot_variation_final_values(key,c.name,s.name,s.variations[key])
                    if type(c) is AgentVariableCapture:
                        if c.plot_param_var:
                            for a in self.config.run_aggregator_functions.keys():
                                self.plot_variation_series(key, c.name + "_" + a, s.name, s.variations[key])
                                if c.plot_final_values:
                                    for x in c.avgs_of_final_values:
                                        self.plot_variation_final_values(key, c.name + "_" + a, s.name, s.variations[key], x)

    def plot_variation_series(self, param, name, dir, variation):
        for a in self.config.run_aggregator_functions.keys():
            plot = plotting.TimeSeriesPlot()
            for v in variation:
                dc = persistence.create_datacollector_from_csv_file(
                    self.config.workspace + '/' + self.config.data_folder + "/" + self.config.aggregate_folder + "/" + dir + '/' + param + str(v) + "/" + a + '/' + name + ".csv")
                plot.add_series(dc, v)
            plot.save_plot(
                self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.aggregate_folder + "/" + dir + "/" + self.config.combined_folder + "/" + a + "/" + name + ".pdf", legend=True)

    def plot_variation_final_values(self, param, name, dir, variation, avg_of=1):
        for a in self.config.run_aggregator_functions.keys():
            plot = plotting.XYPlot()
            for v in variation:
                dc = persistence.create_datacollector_from_csv_file(
                    self.config.workspace + '/' + self.config.data_folder + "/" + self.config.aggregate_folder + "/" + dir + '/' + param + str(
                        v) + "/" + a + '/' + name + ".csv")
                key = list(dc.data.keys())[0]
                value = numpy.mean(dc.data[key][-avg_of:])
                plot.add_data_point(v,value,id=a)
            plot.save_plot(
                self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.aggregate_folder + "/" + dir + "/" + self.config.final_values_folder +"/" + a + "/" + name + "_BY_" + param + "_" + str(avg_of) + ".pdf", legend=True)

    def plot_scenario_comparison(self):
        scenario_names = [s.name for s in self.config.scenarios]
        variations_combined = self.get_all_variations_combined(self.config.scenarios)

        for c in self.config.captures:
            print("PLOTTING SCENCOMP: " + c.name, flush=True)
            if c.plot_param_var:
                if type(c) is ModelVariableCapture:
                    self.plot_timeseries_scenarios_combined(scenario_names, variations_combined, c.name)
                    if self.plot_variation_final_values:
                        for x in c.avgs_of_final_values:
                            self.plot_final_values_scenarios_combined(scenario_names, variations_combined, c.name, x)
                if type(c) is AgentVariableCapture:
                    if c.plot_param_var:
                        for a in self.config.agent_aggregator_functions.keys():
                            self.plot_timeseries_scenarios_combined(scenario_names, variations_combined, c.name + "_" + a)
                            if self.plot_variation_final_values:
                                for x in c.avgs_of_final_values:
                                    self.plot_final_values_scenarios_combined(scenario_names, variations_combined, c.name + "_" + a,x)

    def get_all_variations_combined(self, scenarios):
        variations = {}
        for s in scenarios:
            for p in s.variations:
                if p not in variations:
                    variations[p]= []
                for v in s.variations[p]:
                    if v not in variations[p]:
                        variations[p].append(v)

        return variations

    def plot_timeseries_scenarios_combined(self, scenario_names, variations_combined, name):
        for p in variations_combined:
            for v in variations_combined[p]:
                for a in self.config.run_aggregator_functions:
                    plot = plotting.TimeSeriesPlot()
                    for s in scenario_names:
                        full_dir = self.config.workspace + '/' + self.config.data_folder + "/" + self.config.aggregate_folder + "/" + s + '/' + p + str(v) + "/" + a + '/' + name + ".csv"
                        print(" --- " + full_dir, flush=True)
                        if persistence.check_dir_exists(full_dir):
                            dc = persistence.create_datacollector_from_csv_file(full_dir)
                            plot.add_series(dc, s)
                    plot.save_plot(
                        self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.aggregate_folder + "/" + self.config.combined_folder + '/' + p + str(v) + "/" + a + "/" + name + ".pdf",
                        legend=True)

    def plot_final_values_scenarios_combined(self, scenario_names, variations_combined, name, avg_of=1):
        for a in self.config.run_aggregator_functions.keys():
            plot = plotting.XYPlot()
            for p in variations_combined:
                for v in variations_combined[p]:
                    for s in scenario_names:
                        full_dir = self.config.workspace + '/' + self.config.data_folder + "/" + self.config.aggregate_folder + "/" + s + '/' + p + str(
                                v) + "/" + a + '/' + name + ".csv"
                        print(" --- " + full_dir, flush=True)
                        if persistence.check_dir_exists(full_dir):
                            dc = persistence.create_datacollector_from_csv_file(full_dir)
                            key = list(dc.data.keys())[0]
                            value = numpy.mean(dc.data[key][-avg_of:])
                            plot.add_data_point(v, value, id=s)
                    plot.save_plot(
                        self.config.workspace + '/' + self.config.plot_folder + "/" + self.config.aggregate_folder + "/" + self.config.combined_folder +"/" + self.config.final_values_folder +"/"+ a + "/" + name + "_BY_" + p + "_" + str(avg_of) + ".pdf",
                        legend=True)


class ManagedRun:
    def __init__(self, base_dir, i, model, no_steps, collectors):
        self.base_dir = base_dir
        self.no_steps = no_steps
        self.model = model
        self.collectors = collectors
        self.i = i

    def execute(self):
        self.model.reset()

        r = running.ModelRun(self.model, self.no_steps)

        for c in self.collectors:
            c.reset()
            r.add_data_collector(c)

        r.model.init_random(None)
        r.run_model()

        persistence.write_modelrun_to_csv_files(self.base_dir + "/" + str(self.i), r)
        r = None
