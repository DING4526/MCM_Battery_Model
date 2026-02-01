# usage/control.py
import random


def sample_dwell_time(state_name):
    dwell_time_ranges = {
        "DeepIdle": (300, 1800),
        "Social": (60, 300),
        "Video": (300, 1800),
        "Gaming": (600, 3600),
        "Navigation": (300, 5400),
    }
    t_min, t_max = dwell_time_ranges[state_name]
    return random.uniform(t_min, t_max)


class ScenarioController:
    """
    统一的使用行为控制器
    """

    def __init__(self, scenario):
        self.scenario = scenario
        self.type = scenario["type"]

        if self.type == "markov":
            self.current_state = scenario["initial_state"]
        else:
            self.current_state = None

        self.remaining_time = 0.0

    def _next_state(self):
        if self.type == "mixed":
            states = list(self.scenario["state_ratio"].keys())
            probs = list(self.scenario["state_ratio"].values())
            return random.choices(states, probs)[0]

        elif self.type == "markov":
            trans = self.scenario["transition_matrix"]
            next_states = list(trans[self.current_state].keys())
            probs = list(trans[self.current_state].values())
            return random.choices(next_states, probs)[0]

        else:
            raise ValueError("Unknown scenario type")

    def step(self, dt):
        """
        返回当前状态名
        """
        if self.remaining_time <= 0:
            self.current_state = self._next_state()
            self.remaining_time = sample_dwell_time(self.current_state)

        self.remaining_time -= dt
        return self.current_state
