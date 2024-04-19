import numpy as np
import cvxpy as cp

from more_itertools import set_partitions


def bf(
    work: np.ndarray[float],
    time: np.ndarray[float],
    cost: np.ndarray[float],
    eta: np.ndarray[float],
    solver: str = "CPLEX",
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

    for partition in set_partitions(range(n_workers), n_jobs):
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

        if partition_cost < float("inf"):
            print([partition[np.where(X.value[i] == 1)[0][0]] for i in range(n_jobs)], partition_cost)

        if partition_cost < best_cost:
            best_cost = partition_cost
            best_assignment = [partition[np.where(X.value[i] == 1)[0][0]] for i in range(n_jobs)]

    return best_cost, best_assignment


def check_solution(
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


# Example usage
if __name__ == "__main__":
    n_jobs, n_workers = 5, 6

    # JOBS
    work = np.array([10.0, 10.0, 10.0, 1.0, 1.0])
    time = np.array([4.0, 4.0, 4.0, 3.0, 3.0])
    # WORKERS
    cost = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    eta = np.array([5.0, 3.0, 3.0, 3.0, 3.0, 10.0])

    print("\nJOBS")
    print(f"Work = {work}")
    print(f"Time = {time}")

    print("\nWORKERS")
    print(f"Cost = {cost}")
    print(f"Eta  = {eta}")

    best_cost, best_assignment = bf(work, time, cost, eta)
    check_solution(work, time, eta, best_assignment, n_workers)

    print("\nSOLUTION")
    print(f"Best cost = {best_cost:.3f}")
    print(f"Best assignment = {best_assignment}")
