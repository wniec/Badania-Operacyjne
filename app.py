import sys
import json
import numpy as np

from bf import bf
from ga import ga
from utils import CheckStatus, check_solution

PARAMS = ["work", "time", "cost", "effi"]
HYPER_PARAMS = {
    "bf": [
        "solver",
    ],
    "ga": [
        "max_iter",
        "pop_size",
        "rep_ratio",
        "penalty_c",
    ],
}

# TODO: Add sanity checks of the passed config file + handling of errors
if __name__ == "__main__":
    argc, argv = len(sys.argv), sys.argv

    if argc != 2:
        print("Usage: '$ python3 <path_to_json_config_file>'")
        exit(0)

    with open(argv[1], "r") as file:
        args: dict = json.load(file)

    algo = {"bf": bf, "ga": ga}

    for algo_name in args.keys():
        print("=" * 60)
        print(algo_name)
        print("-" * 60)

        params = {k: np.array(v) for k, v in args[algo_name].items() if k in PARAMS}
        hyper_params = {k: v for k, v in args[algo_name].items() if k in HYPER_PARAMS[algo_name]}

        best_cost, best_assignment = algo[algo_name](**params, **hyper_params)

        if best_assignment is None:
            print("Problem infeasible!")
        elif CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, **params)):
            print(f"Solution incorrect: {status_code}")
        else:
            print("Solution found!")
            print("\n".join([f"Best cost = {best_cost:.3f}", f"Best assignment = {best_assignment}"]))
        print("=" * 60)
        print()
