import numpy as np

from typing import Iterable
from scipy.optimize import linear_sum_assignment as lap


def set_partitions(L: Iterable, k: int):
    """
    Implements a generator which yields partitions of the set L into k non-empty sets. Based on the
    implementation found in https://github.com/more-itertools/more-itertools

    Args:
        L: iterable representing the set for which we generate partitions.
        k: number of non-empty partitions.

    Yields:
        A list of k iterables representing partitions of L.
    """
    n = len(L)
    if k == 1:
        yield [L]
    elif n == k:
        yield [[s] for s in L]
    else:
        e, *M = L
        for p in set_partitions(M, k - 1):
            yield [[e], *p]
        for p in set_partitions(M, k):
            for i in range(len(p)):
                yield p[:i] + [[e] + p[i]] + p[i + 1 :]


def bf(
    work: np.ndarray[float],
    cost: np.ndarray[float],
    eta: np.ndarray[float],
) -> tuple[float, list[list]]:
    """
    Implements brute-force algorithm to solve the following optimization problem

    minimize sum_{i=1,..,n} work[i] * (sum_{j=1,..,m} cost[j] * x[i,j]) / (sum_{j=1,..,m} eta[j] * x[i,j])
    s.t.     ∀ i=1,..,n; j=1,..,m : x[i,j] ∈ {0,1}
             ∀ i=1,..,n : sum_{j=1,..,m} x[i,j] >= 1
             ∀ j=1,..,m : sum_{i=1,..,n} x[i,j] = 1

    where x[i,j] = 1 if j-th worker is assigned to i-th job else 0, n is the number of jobs and m is
    the number of workers. We assume that m >= n. The algorithm exhaustively searches over all
    n-partitions of the set of all workers and for each generated partition solves the corresponding
    Linear Assignment Problem (LAP) to optimally assign each group of workers in a partition to a
    job. Time complexity of this algorithm is O( S(m,n) * n**3), where S(m,n) denotes the Stirling
    number of the 2nd kind.

    Args:
        work: Array of work values for each job. Shape (n_jobs,).
        cost: Array of costs per unit of time for each worker. Shape (n_workers,).
        eta: Array of efficacies (i.e. amount of work a worker can do in unit time) for each worker.
             Shape (n_workers,).

    Returns:
        Minimum value of function and the optimal assignment of workers to jobs as a list of lists
        where each i-th list contains workers assigned to i-th job.
    """
    assert all((isinstance(work, np.ndarray), isinstance(cost, np.ndarray), isinstance(eta, np.ndarray)))
    assert len(work.shape) == len(cost.shape) == len(eta.shape) == 1, "Expected 1-D arrays"
    assert cost.shape == eta.shape, "Expected cost and eta arrays to have the same shape"
    assert (n_jobs := len(work)) <= (n_workers := len(cost)), "Expected n_workers >= n_jobs"

    best_cost, best_assignment = float("inf"), None
    for partition in set_partitions(range(n_workers), n_jobs):
        cost_matrix = np.outer(work, [cost[p].sum() / eta[p].sum() for p in partition])

        row_ind, col_ind = lap(cost_matrix)
        partition_cost = cost_matrix[row_ind, col_ind].sum()

        if partition_cost < best_cost:
            best_cost = partition_cost
            best_assignment = [partition[i] for i in col_ind]

    return best_cost, best_assignment


# Example usage
if __name__ == "__main__":
    n_jobs, n_workers = 4, 6
    work = np.random.rand(n_jobs)
    cost = np.random.rand(n_workers)
    eta = np.random.rand(n_workers)

    print(f"Work = {work}")
    print(f"Cost = {cost}")
    print(f"Eta  = {eta}")

    best_cost, best_assignment = bf(work, cost, eta)
    print(f"Best cost = {best_cost:.3f}")
    print(f"Best assignment =\n{best_assignment}")
