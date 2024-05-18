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

class ParamsException(Exception):
    pass


class BaseParams():
    def __init__(self) -> None:
        self.params = {}
    
    def load(self, args :dict):
        try:
            self.params["work"] = np.array(args["work"])
        except Exception:
            raise ParamsException("Missing parameter: work")
            
        try:
            self.params["time"] = np.array(args["time"])
        except Exception:
            raise ParamsException("Missing parameter: time")
            
        try:
            self.params["cost"] = np.array(args["cost"])
        except Exception:
            raise ParamsException("Missing parameter: cost")
            
        try:
            self.params["effi"] = np.array(args["effi"])
        except Exception:
            raise ParamsException("Missing parameter: effi")
        
    def getParams(self):
        return self.params
        
class GeneticParams(BaseParams):
    def __init__(self) -> None:
        super().__init__()
        self.max_iter = None
        self.pop_size = None
        self.rep_ratio = None
        self.penalty_c = None
        
    # def load(self, args :dict):
    #     super().load(args)
        
    def getHiperParams(self):
        return {}
    

# TODO: Add sanity checks of the passed config file + handling of errors
if __name__ == "__main__":
    argc, argv = len(sys.argv), sys.argv

    if argc != 2:
        print("Usage: '$ python3 <path_to_json_config_file>'")
        exit(0)
    
    try:
        with open(argv[1], "r") as file:
            args: dict = json.load(file)
    except Exception as e:
        print("Error when reading confing file")
        print(e)
        exit(1)
        

    algo = {"bf": bf, "ga": ga}

    for algo_name in args.keys():
        if( algo_name not in algo.keys() ):
            print("Error when reading confing file")
            print("wrong alorithm name: " + algo_name)
            exit(0)
        print("=" * 60)
        print(algo_name)
        print("-" * 60)
        
        if algo_name == "bf":
            params = BaseParams()
        if algo_name == "ga":
            params = GeneticParams()
        params.load(args[algo_name])
        print(params)
        print(params.getParams())
        print(args[algo_name])

        # params = {k: np.array(v) for k, v in args[algo_name].items() if k in PARAMS}
        hyper_params = {k: v for k, v in args[algo_name].items() if k in HYPER_PARAMS[algo_name]}

        best_cost, best_assignment = algo[algo_name](**params.getParams(), **hyper_params)

        if best_assignment is None:
            print("Problem infeasible!")
        elif CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, **params.getParams())):
            print(f"Solution incorrect: {status_code}")
        else:
            print("Solution found!")
            print("\n".join([f"Best cost = {best_cost:.3f}", f"Best assignment = {best_assignment}"]))
        print("=" * 60)
        print()
