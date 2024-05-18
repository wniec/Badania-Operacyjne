import sys
import json
import numpy as np
import time
import os.path

from bf import bf
from ga import ga
from ba import ba
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
        for arg in args:
            if arg not in {"work", "time", "cost", "effi"}:
                raise ParamsException("Unknown parameter: params." + arg)
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
    
class HiperParams():
    def load(self, args :dict):
        pass
    def getHiperParams(self):
        return {}
        
class GeneticParams(HiperParams):
    def __init__(self) -> None:
        self.hiperParams = {}
        
    def load(self, args :dict):
        if args is None:
            return
        for arg in args:
            if arg not in {"max_iter", "pop_size", "rep_ratio", "penalty_c"}:
                raise ParamsException("Unknown parameter: ga." + arg)
        if "max_iter" in args:
            self.hiperParams["max_iter"] = args["max_iter"]
            
        if "pop_size" in args:
            self.hiperParams["pop_size"] = args["pop_size"]
            
        if "rep_ratio" in args:
            self.hiperParams["rep_ratio"] = args["rep_ratio"]
            
        if "penalty_c" in args:
            self.hiperParams["penalty_c"] = args["penalty_c"]
            
    def getHiperParams(self):
        return self.hiperParams
    
class BeesParams(HiperParams):
    def __init__(self) -> None:
        self.hiperParams = {}
        
    def load(self, args :dict):
        if args is None:
            return
        for arg in args:
            if arg not in {"max_iter", "sol_size", "top_frac", "top_attempts", "rest_attempts", "penalty_c"}:
                raise ParamsException("Unknown parameter: ba." + arg)
            
        if "max_iter" in args:
            self.hiperParams["max_iter"] = args["max_iter"]
            
        if "sol_size" in args:
            self.hiperParams["sol_size"] = args["sol_size"]
            
        if "top_frac" in args:
            self.hiperParams["top_frac"] = args["top_frac"]
            
        if "top_attempts" in args:
            self.hiperParams["top_attempts"] = args["top_attempts"]
            
        if "rest_attempts" in args:
            self.hiperParams["rest_attempts"] = args["rest_attempts"]
            
        if "penalty_c" in args:
            self.hiperParams["penalty_c"] = args["penalty_c"]
            
            
    def getHiperParams(self):
        return self.hiperParams
    

if __name__ == "__main__":
    argc, argv = len(sys.argv), sys.argv

    if argc != 2:
        print("Usage: '$ python3 <path_to_json_config_file>'")
        exit(1)
    
    if not os.path.isfile(argv[1]):
        print("Invalid config path")
        exit(1)
    try:
        with open(argv[1], "r") as file:
            args: dict = json.load(file)
    except Exception as e:
        print("Error when reading confing file")
        print(e)
        exit(1)
        
    if "params" not in args.keys():
        print("Missing 'params' atribute")
        exit(1)
        
    params = BaseParams()
    try:
        params.load(args["params"])
    except Exception as e:
        print("Invalid configuartion: ")
        print(e)
        exit(1)
    
    if "algorithms" not in args.keys():
        print("Missing 'algorithms' atribute")
        exit(1)

    algo = {"bf": bf, "ga": ga, "ba": ba}

    for algo_name in args["algorithms"].keys():
        if( algo_name not in algo.keys() ):
            print("Error when reading confing file")
            print("wrong alorithm name: " + algo_name)
            exit(0)
        print("=" * 60)
        print(algo_name)
        print("-" * 60)
        
        
        hyper_params = HiperParams()
        if algo_name == "ga":
            hyper_params = GeneticParams()
            hyper_params.load(args["algorithms"][algo_name])
        if algo_name == "ba":
            hyper_params = BeesParams()
            hyper_params.load(args["algorithms"][algo_name])
            
        start = time.time()

        best_cost, best_assignment = algo[algo_name](**params.getParams(), **hyper_params.getHiperParams())
        
        end = time.time()
        
        if best_assignment is None:
            print("Problem infeasible!")
        elif CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, **params.getParams())):
            print(f"Solution incorrect: {status_code}")
        else:
            print("Solution found!")
            print("\n".join([f"Best cost = {best_cost:.3f}", f"Best assignment = {best_assignment}"]))
        print(f"time: {end-start:.3f} s")
        print("=" * 60)
        print()
