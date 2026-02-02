# usage/control.py
import random


DEFAULT_DWELL_SEC = 15 * 60  # 900s

def sample_dwell_time(state_name, dwell_mode="fixed", fixed_sec=DEFAULT_DWELL_SEC):
    if dwell_mode == "fixed":
        return float(fixed_sec)

    # 兼容旧模式（你以后要升级马尔可夫/半马尔可夫时可用）
    dwell_time_ranges = {
        "IDLE": (300, 1800),
        "SOCIAL": (60, 300),
        "VIDEO": (300, 1800),
        "GAME": (600, 3600),
    }
    t_min, t_max = dwell_time_ranges[state_name]
    return random.uniform(t_min, t_max)

class ScenarioController:
    """
    统一的使用行为控制器
    """

    def __init__(self, scenario, dwell_mode="fixed", fixed_dwell_sec=DEFAULT_DWELL_SEC):
        self.scenario = scenario
        self.type = scenario["type"]
        self.dwell_mode = dwell_mode
        self.fixed_dwell_sec = fixed_dwell_sec

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
            self.remaining_time = sample_dwell_time(
                self.current_state,
                dwell_mode=self.dwell_mode,
                fixed_sec=self.fixed_dwell_sec,
            )
        self.remaining_time -= dt
        return self.current_state
