import numpy as np
import cvxpy as cp

from tqdm import tqdm
from scipy.special import stirling2
from more_itertools import set_partitions


def bf(
    work: np.ndarray[float],
    time: np.ndarray[float],
    cost: np.ndarray[float],
    effi: np.ndarray[float],
    solver: str = "GUROBI",
) -> tuple[float, list[list[int]]]:
    """Implements brute-force algorithm to solve the following combinatorial optimization problem

    minimize sum_{i=1,..,n} work[i] * (sum_{j=1,..,m} cost[j] * x[i,j]) / (sum_{j=1,..,m} effi[j] * x[i,j])
    s.t.     ∀ i=1,..,n; j=1,..,m : x[i,j] ∈ {0,1}
             ∀ i=1,..,n : sum_{j=1,..,m} x[i,j] >= 1
             ∀ j=1,..,m : sum_{i=1,..,n} x[i,j] == 1
             ∀ i=1,..,n : work[i] / (sum_{j=1,..,m} effi[j] * x[i,j]) <= time[i]

    where x[i,j] = 1 if j-th worker is assigned to i-th job else 0, n is the number of jobs and m is
    the number of workers. We assume that m >= n. The algorithm exhaustively searches over all
    n-partitions of the set of all workers and for each generated partition solves the corresponding
    Generalized Assignment Problem (GAP) using ILP solver.

    Args:
        work: Array of work values for each job. Shape (n_jobs,).
        time: Array of max times for each job. Shape (n_jobs,).
        cost: Array of costs per unit of time for each worker. Shape (n_workers,).
        effi: Array of efficacies (i.e. amount of work a worker can do in unit time) for each
              worker. Shape (n_workers,).
        solver: Solver used to solve the ILP. Should be one of the: "CPLEX", "GUROBI", "GLPK_MI".

    Returns:
        Minimum value of objective and the optimal assignment of workers to jobs as a list of lists
        where each i-th list contains workers assigned to i-th job.
    """
    assert all(isinstance(arr, np.ndarray) for arr in (work, time, cost, effi)), "Expected Numpy arrays"
    assert all(len(arr.shape) == 1 for arr in (work, time, cost, effi)), "Expected 1-D arrays"
    assert work.shape == time.shape, "Expected `work` and `time` arrays to have the same shape"
    assert cost.shape == effi.shape, "Expected `cost` and `effi` arrays to have the same shape"
    assert (n_jobs := len(work)) <= (n_workers := len(cost)), "Expected n_workers >= n_jobs"

    best_cost, best_assignment = float("inf"), None

    for partition in tqdm(set_partitions(range(n_workers), n_jobs), total=stirling2(n_workers, n_jobs)):
        cost_matrix = np.outer(work, [cost[p].sum() / effi[p].sum() for p in partition])
        time_matrix = np.outer(work, [1 / effi[p].sum() for p in partition])

        X = cp.Variable((n_jobs, n_jobs), boolean=True)
        constraints = [
            X.sum(axis=0) == 1,
            X.sum(axis=1) == 1,
            cp.multiply(X, time_matrix).sum(axis=1) <= time,
        ]

        problem = cp.Problem(cp.Minimize(cp.multiply(cost_matrix, X).sum()), constraints)
        partition_cost = problem.solve(verbose=False, solver=solver)

        if partition_cost < best_cost:
            best_cost = partition_cost
            best_assignment = [partition[X.value[i].tolist().index(1)] for i in range(n_jobs)]

    return best_cost, best_assignment


# Example usage
if __name__ == "__main__":
    from utils import check_solution, CheckStatus

    work = np.array([0.0564, 0.5310, 0.8676])
    time = np.array([0.5367, 0.4316, 0.9047])
    cost = np.array([0.4241, 0.7391, 0.3058, 0.6060, 0.6486])
    effi = np.array([0.5549, 0.3572, 0.7561, 0.1787, 0.9503])

    best_cost, best_assignment = bf(work, time, cost, effi)

    if best_assignment is None:
        print("Problem infeasible!")
    elif CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, work, time, cost, effi)):
        print(f"Solution incorrect: {status_code}")
    else:
        print("\n".join(["Solution found!", f"Best cost = {best_cost:.3f}", f"Best assignment = {best_assignment}"]))
