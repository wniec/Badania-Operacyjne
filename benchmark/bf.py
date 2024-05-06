import numpy as np
import cvxpy as cp

from tqdm import tqdm
from scipy.special import stirling2
from more_itertools import set_partitions


def bf(
    work: np.ndarray[float],
    time: np.ndarray[float],
    cost: np.ndarray[float],
    eta: np.ndarray[float],
    solver: str = "GUROBI",
) -> tuple[float, list[list[int]]]:
    """
    Implements brute-force algorithm to solve the following combinatorial optimization problem

    minimize sum_{i=1,..,n} work[i] * (sum_{j=1,..,m} cost[j] * x[i,j]) / (sum_{j=1,..,m} eta[j] * x[i,j])
    s.t.     ∀ i=1,..,n; j=1,..,m : x[i,j] ∈ {0,1}
             ∀ i=1,..,n : sum_{j=1,..,m} x[i,j] >= 1
             ∀ j=1,..,m : sum_{i=1,..,n} x[i,j] = 1
             ∀ i=1,..,n : work[i] / (sum_{j=1,..,m} eta[j] * x[i,j]) <= time[i]

    where x[i,j] = 1 if j-th worker is assigned to i-th job else 0, n is the number of jobs and m is
    the number of workers. We assume that m >= n. The algorithm exhaustively searches over all
    n-partitions of the set of all workers and for each generated partition solves the corresponding
    Generalized Assignment Problem (GAP) using ILP.

    Args:
        work: Array of work values for each job. Shape (n_jobs,).
        time: Array of max times for each job. Shape (n_jobs,).
        cost: Array of costs per unit of time for each worker. Shape (n_workers,).
        eta : Array of efficacies (i.e. amount of work a worker can do in unit time) for each
              worker. Shape (n_workers,)
        solver: Solver used to solve the ILP. Should be one of the: "CPLEX", "GUROBI", "GLPK_MI".
                Defaults to "CPLEX".

    Returns:
        Minimum value of objective and the optimal assignment of workers to jobs as a list of lists
        where each i-th list contains workers assigned to i-th job.
    """
    assert all((isinstance(work, np.ndarray), isinstance(cost, np.ndarray), isinstance(eta, np.ndarray)))
    assert len(work.shape) == len(cost.shape) == len(eta.shape) == 1, "Expected 1-D arrays"
    assert cost.shape == eta.shape, "Expected cost and eta arrays to have the same shape"
    assert (n_jobs := len(work)) <= (n_workers := len(cost)), "Expected n_workers >= n_jobs"

    best_cost, best_assignment = float("inf"), None

    for partition in tqdm(set_partitions(range(n_workers), n_jobs), total=stirling2(n_workers, n_jobs)):
        cost_matrix = np.outer(work, [cost[p].sum() / eta[p].sum() for p in partition])
        time_matrix = np.outer(work, [1 / eta[p].sum() for p in partition])

        X = cp.Variable((n_jobs, n_jobs), boolean=True)
        constraints = [
            X.sum(axis=0) == 1,
            X.sum(axis=1) == 1,
            cp.multiply(X, time_matrix).sum(axis=1) <= time,
        ]

        problem = cp.Problem(cp.Minimize(cp.multiply(cost_matrix, X).sum()), constraints)
        partition_cost = problem.solve(verbose=False, solver=solver)
        print(partition_cost)
        if partition_cost < best_cost:
            best_cost = partition_cost
            best_assignment = [partition[X.value[i].tolist().index(1)] for i in range(n_jobs)]

    return best_cost, best_assignment


def check_solution(
    cost: float,
    work: np.ndarray[float],
    time: np.ndarray[float],
    eta: np.ndarray[float],
    assignment: list[list[int]],
    n_workers: int,
):
    assert all(len(group) > 0 for group in assignment), "There is at least 1 job with no workers"
    assert all(
        sum(1 if worker in group else 0 for group in assignment) == 1 for worker in range(n_workers)
    ), "At least one of the workers is assigned to more than 1 job"
    assert all(
        work[i] / sum(eta[worker] for worker in group) <= time[i] for i, group in enumerate(assignment)
    ), "At least one job takes more time than available"
    assert cost == sum(
        w * cost[p].sum() / eta[p].sum() for w, p in zip(work, assignment)
    ), "Cost is incorretct for the given assignment"


# Example usage
if __name__ == "__main__":
    n_jobs, n_workers = 5, 8

    # JOBS
    work = np.random.random(n_jobs)
    time = 0.8 + np.random.random(n_jobs)

    work = np.array([])
    # WORKERS
    cost = np.random.random(n_workers)
    eta = 0.1 + np.random.random(n_workers)

    n_jobs = 3
    n_workers = 5

    work = np.array([0.05649429, 0.53109867, 0.86764481])
    time = np.array([0.53677527, 0.43160818, 0.9047183])
    cost = np.array([0.42415955, 0.73918663, 0.3058235, 0.6060437, 0.64862661])
    eta = np.array([0.55492947, 0.35726281, 0.75614558, 0.17878197, 0.95030017])

    print("\nJOBS")
    print(f"Work = {work}")
    print(f"Time = {time}")

    print("\nWORKERS")
    print(f"Cost = {cost}")
    print(f"Eta  = {eta}")

    best_cost, best_assignment = bf(work, time, cost, eta)
    check_solution(work, time, eta, best_assignment, n_workers) if best_cost < float("inf") else None

    print("\nSOLUTION")
    print(f"Best cost = {best_cost:.3f}")
    print(f"Best assignment = {best_assignment}")

    print()
