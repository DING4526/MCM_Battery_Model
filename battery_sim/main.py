from battery_sim.analysis.montecarlo import run_mc
from battery_sim.analysis.attribution import energy_breakdown, state_ablation
from battery_sim.analysis.sensitivity import oat_sensitivity


def main():
    results = run_mc(50, "Average", user="Heavy", system="battery_saver")
    energy_breakdown(results)
    state_ablation(results)
    oat_sensitivity(results, key="E0")


if __name__ == "__main__":
    main()

