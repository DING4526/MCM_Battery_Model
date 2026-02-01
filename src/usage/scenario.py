# usage/scenario.py

# =====================================================
# 1. STUDENT DAILY
# =====================================================

SCENARIO_STUDENT_DAILY_MIXED = {
    "type": "mixed",
    "state_ratio": {
        "DeepIdle": 0.30,
        "Social": 0.30,
        "Video": 0.20,
        "Gaming": 0.15,
        "Navigation": 0.05,
    }
}

SCENARIO_STUDENT_DAILY_MARKOV = {
    "type": "markov",
    "initial_state": "Social",
    "transition_matrix": {
        "Social": {"Social": 0.5, "Video": 0.25, "Gaming": 0.15, "DeepIdle": 0.10},
        "Video": {"Video": 0.6, "Social": 0.25, "DeepIdle": 0.15},
        "Gaming": {"Gaming": 0.6, "Social": 0.2, "DeepIdle": 0.2},
        "DeepIdle": {"DeepIdle": 0.6, "Social": 0.3, "Video": 0.1},
    }
}

# =====================================================
# 2. WEEKDAY COMMUTE
# =====================================================

SCENARIO_COMMUTE_MIXED = {
    "type": "mixed",
    "state_ratio": {
        "Navigation": 0.35,
        "Social": 0.25,
        "DeepIdle": 0.25,
        "Video": 0.10,
        "Gaming": 0.05,
    }
}

SCENARIO_COMMUTE_MARKOV = {
    "type": "markov",
    "initial_state": "Navigation",
    "transition_matrix": {
        "Navigation": {"Navigation": 0.6, "Social": 0.25, "DeepIdle": 0.15},
        "Social": {"Social": 0.5, "Navigation": 0.3, "DeepIdle": 0.2},
        "DeepIdle": {"DeepIdle": 0.6, "Navigation": 0.3, "Social": 0.1},
    }
}

# =====================================================
# 3. WEEKEND HEAVY ENTERTAINMENT
# =====================================================

SCENARIO_WEEKEND_MIXED = {
    "type": "mixed",
    "state_ratio": {
        "Gaming": 0.35,
        "Video": 0.30,
        "Social": 0.20,
        "DeepIdle": 0.10,
        "Navigation": 0.05,
    }
}

SCENARIO_WEEKEND_MARKOV = {
    "type": "markov",
    "initial_state": "Video",
    "transition_matrix": {
        "Video": {"Video": 0.6, "Gaming": 0.25, "Social": 0.15},
        "Gaming": {"Gaming": 0.6, "Video": 0.25, "Social": 0.15},
        "Social": {"Social": 0.4, "Video": 0.4, "Gaming": 0.2},
    }
}

# =====================================================
# 4. TRAVEL / OUTDOOR TRIP
# =====================================================

SCENARIO_TRAVEL_MIXED = {
    "type": "mixed",
    "state_ratio": {
        "Navigation": 0.40,
        "Social": 0.25,
        "Video": 0.15,
        "DeepIdle": 0.10,
        "Gaming": 0.10,
    }
}

SCENARIO_TRAVEL_MARKOV = {
    "type": "markov",
    "initial_state": "Navigation",
    "transition_matrix": {
        "Navigation": {"Navigation": 0.65, "Social": 0.25, "Video": 0.10},
        "Social": {"Social": 0.45, "Navigation": 0.35, "Video": 0.20},
        "Video": {"Video": 0.5, "Social": 0.3, "Navigation": 0.2},
    }
}

# =====================================================
# 5. PURE STATES FOR TESTING
# =====================================================
PURE_DEEPIDLE = {
    "type": "mixed", 
    "state_ratio": {
        "DeepIdle": 1.0,
    }
}
PURE_SOCIAL = {
    "type": "mixed", 
    "state_ratio": {
        "Social": 1.0,
    }
}
PURE_VIDEO = {
    "type": "mixed", 
    "state_ratio": {
        "Video": 1.0,
    }
}
PURE_GAMING = {
    "type": "mixed", 
    "state_ratio": {
        "Gaming": 1.0,
    }
}
PURE_NAVIGATION = {
    "type": "mixed",   
    "state_ratio": {
        "Navigation": 1.0,
    }
}