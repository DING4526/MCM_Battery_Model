# usage/control.py
import random
import math

DEFAULT_DWELL_SEC = 15 * 60  # 900s


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def sample_dwell_time(state_name, dwell_mode="fixed", fixed_sec=DEFAULT_DWELL_SEC, dwell_params=None):
    """
    dwell_mode:
      - fixed: fixed_sec
      - random: uniform ranges (legacy)
      - hsmm: sample per-state duration distribution (gamma/lognormal/exponential)
    """
    if dwell_mode == "fixed":
        return float(fixed_sec)

    if dwell_mode == "random":
        dwell_time_ranges = {
            "IDLE": (300, 1800),
            "SOCIAL": (60, 300),
            "VIDEO": (300, 1800),
            "GAME": (600, 3600),
        }
        t_min, t_max = dwell_time_ranges[state_name]
        return random.uniform(t_min, t_max)

    if dwell_mode == "hsmm":
        # default per-state mean durations (seconds) if not provided
        # you SHOULD tune these to look realistic; online correction will still match ratios.
        default_mean = {
            "IDLE": 18 * 60,
            "SOCIAL": 6 * 60,
            "VIDEO": 20 * 60,
            "GAME": 15 * 60,
        }
        default_shape_k = {
            "IDLE": 2.0,
            "SOCIAL": 1.6,
            "VIDEO": 2.2,
            "GAME": 2.0,
        }
        dp = dwell_params or {}

        dist = (dp.get(state_name, {}) or {}).get("dist", "gamma")
        mean = float((dp.get(state_name, {}) or {}).get("mean_sec", default_mean.get(state_name, fixed_sec)))
        mean = max(1.0, mean)

        if dist == "exponential":
            # Exp(mean): random.expovariate(lambda) where lambda=1/mean
            return random.expovariate(1.0 / mean)

        if dist == "lognormal":
            # lognormal with given mean, choose sigma, solve mu
            sigma = float((dp.get(state_name, {}) or {}).get("sigma", 0.55))
            sigma = max(0.05, sigma)
            mu = math.log(mean) - 0.5 * sigma * sigma
            return random.lognormvariate(mu, sigma)

        # default gamma(k, theta) with mean = k*theta
        k = float((dp.get(state_name, {}) or {}).get("shape_k", default_shape_k.get(state_name, 2.0)))
        k = max(0.2, k)
        theta = mean / k
        return random.gammavariate(k, theta)

    raise ValueError(f"Unknown dwell_mode: {dwell_mode}")


class ScenarioController:
    """
    Semi-Markov (HSMM) upgrade:
      scenario["type"] can be:
        - "mixed"  : legacy ratio-based next-state draw
        - "markov" : legacy Markov transitions
        - "hsmm"   : HSMM with dwell distributions + online ratio matching

    For hsmm:
      scenario = {
        "type": "hsmm",
        "state_ratio": {...},              # target time ratios
        "dwell_params": {state: {...}},    # optional per-state dwell distribution config
        "initial_state": "SOCIAL"          # optional
      }
    """

    def __init__(self, scenario, dwell_mode="fixed", fixed_dwell_sec=DEFAULT_DWELL_SEC):
        self.scenario = scenario
        self.type = scenario["type"]
        self.dwell_mode = dwell_mode
        self.fixed_dwell_sec = fixed_dwell_sec

        self.remaining_time = 0.0

        if self.type == "markov":
            self.current_state = scenario["initial_state"]
        elif self.type in ("mixed", "hsmm"):
            self.current_state = scenario.get("initial_state")
        else:
            raise ValueError("Unknown scenario type")

        # For hsmm online ratio matching
        self._spent = {}   # seconds spent per state
        self._t = 0.0      # total time
        self._eps = float(scenario.get("ratio_match_eps", 1e-9))

        # hsmm dwell params
        self._dwell_params = scenario.get("dwell_params", None)

    def _next_state_markov(self):
        trans = self.scenario["transition_matrix"]
        next_states = list(trans[self.current_state].keys())
        probs = list(trans[self.current_state].values())
        return random.choices(next_states, probs)[0]

    def _next_state_mixed(self):
        states = list(self.scenario["state_ratio"].keys())
        probs = list(self.scenario["state_ratio"].values())
        return random.choices(states, probs)[0]

    def _next_state_hsmm_ratio_match(self):
        """
        Online ratio matching:
          weight_i = max(target_i * t - spent_i, 0)
        If all weights zero (very early), fallback to target ratios.
        """
        ratio = self.scenario["state_ratio"]
        states = list(ratio.keys())

        # initialize trackers
        for s in states:
            self._spent.setdefault(s, 0.0)

        # "debt" weights
        weights = []
        for s in states:
            target = float(ratio[s])
            debt = target * max(self._t, 1.0) - self._spent[s]
            weights.append(max(debt, 0.0))

        if sum(weights) <= self._eps:
            # fallback to target ratios
            probs = [float(ratio[s]) for s in states]
            return random.choices(states, probs)[0]

        return random.choices(states, weights)[0]

    def _choose_next_state(self):
        if self.type == "mixed":
            return self._next_state_mixed()
        if self.type == "markov":
            return self._next_state_markov()
        if self.type == "hsmm":
            return self._next_state_hsmm_ratio_match()
        raise ValueError("Unknown scenario type")

    def step(self, dt):
        """
        Return current state name.
        """
        dt = float(dt)
        if dt <= 0:
            return self.current_state

        if self.remaining_time <= 0.0:
            if self.current_state is None:
                self.current_state = self._choose_next_state()
            else:
                self.current_state = self._choose_next_state()

            # dwell sampling
            if self.type == "hsmm":
                self.remaining_time = sample_dwell_time(
                    self.current_state,
                    dwell_mode="hsmm",
                    fixed_sec=self.fixed_dwell_sec,
                    dwell_params=self._dwell_params,
                )
            else:
                self.remaining_time = sample_dwell_time(
                    self.current_state,
                    dwell_mode=self.dwell_mode,
                    fixed_sec=self.fixed_dwell_sec,
                )

        # update time accounting (for hsmm ratio matching)
        consume = min(dt, self.remaining_time)
        self.remaining_time -= dt

        if self.type == "hsmm":
            self._t += consume
            self._spent[self.current_state] = self._spent.get(self.current_state, 0.0) + consume

        return self.current_state
