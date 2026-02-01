# usage/scenario.py

SCENARIO_STUDENT_MIXED = {
    "type": "mixed",
    "state_ratio": {
        "DeepIdle": 0.35,
        "Social": 0.30,
        "Video": 0.20,
        "Gaming": 0.10,
        "Navigation": 0.05,
    }
}

SCENARIO_COMMUTE_MARKOV = {
    "type": "markov",
    "initial_state": "Navigation",
    "transition_matrix": {
        "Navigation": {"Navigation": 0.6, "Social": 0.3, "DeepIdle": 0.1},
        "Social": {"Social": 0.6, "Video": 0.3, "DeepIdle": 0.1},
        "Video": {"Video": 0.7, "Social": 0.2, "DeepIdle": 0.1},
        "DeepIdle": {"DeepIdle": 0.7, "Social": 0.3},
    }
}
