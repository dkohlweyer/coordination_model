import copy

class AbstractDataCollector:

    def __init__(self, model, name, interval=1, offset=0):
        self.model = model
        self.name = name
        self.data = {}
        self.interval = interval
        self.offset = offset

    def collect_data(self):
        pass

    def reset(self):
        self.data = {}


class FullAgentDataCollector(AbstractDataCollector):

    def __init__(self, name, model, collector_function, interval=1, offset=0):
        self.__init__(name, model, collector_function, lambda a: True, interval, offset)

    def __init__(self, name, model, collector_function, agent_filter, interval=1, offset=0):
        super().__init__(model, name, interval, offset)
        self.collector_function = collector_function
        self.agent_filter = agent_filter

    def collect_data(self):
        for a in self.model.agents:
            if self.agent_filter(a):
                if not (a.unique_id in self.data):
                    self.data[a.unique_id] = []
                self.data[a.unique_id].append(self.collector_function(a))


class AggregateAgentDataCollector(AbstractDataCollector):

    def __init__(self, name, model, collector_function, aggregator_function, agent_filter, interval=1, offset=0):
        super().__init__(model, name, interval, offset)
        self.collector_function = collector_function
        self.aggregator_function = aggregator_function
        self.agent_filter = agent_filter
        self.data[name] = []

        if self.agent_filter is None:
            self.agent_filter = lambda a : True

    def collect_data(self):
        values = []
        for a in self.model.agents:
            if self.agent_filter(a):
                values.append(self.collector_function(a))
        self.data[self.name].append(self.aggregator_function(values))

    def reset(self):
        self.data[self.name] = []


class ModelDataCollector(AbstractDataCollector):

    def __init__(self, name, model, collector_function, interval=1, offset=0):
        super().__init__(model, name, interval, offset)
        self.data[name] = []
        self.collector_function = collector_function

    def collect_data(self):
        self.data[self.name].append(self.collector_function(self.model))

    def reset(self):
        self.data[self.name] = []


class DataAggregator:

    def __init__(self, name, aggregator_function):
        self.name = name
        self.aggregator_function = aggregator_function

    def aggregate(self, list_of_datacollectors):
        aggregated_dc = copy.deepcopy(list_of_datacollectors[0])
        aggregated_dc.data = {}

        for id in list_of_datacollectors[0].data:
            data = []
            for dc in list_of_datacollectors:
                data.append(dc.data[id])

            transposed = list(map(list, (zip(*data))))

            aggregated = []
            for p in transposed:
                aggregated.append(self.aggregator_function(p))

            aggregated_dc.data[id] = aggregated

        return aggregated_dc



