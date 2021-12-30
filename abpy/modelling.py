import copy

class Agent:

    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

    def step(self, stage):
        pass

    def advance(self):
        pass


class Model:

    def __init__(self):
        self.agents = []
        pass

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def populate(self):
        pass

    def initialize(self):
        pass

    def step(self):
        pass

    def reset(self):
        pass

    def init_random(self, seed=None):
        pass


class ModelParameterization():
    def __init__(self):
        self.params = {}

    def set_param(self, identifier, value):
        self.params[identifier] = value

    def get_param(self, identifier):
        return self.params[identifier]

    def set_param_setting(self, setting):
        self.params = copy.deepcopy(setting)

    def get_param_setting(self):
        return self.params

    def get_required_params(self):
        raise NotImplementedError("This should have been implemented!")

    def check_params(self):
        for p in self.get_required_params():
            if p not in self.params.keys():
                return False
        return True
