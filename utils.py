import numpy as np

from enum import Enum, auto
from typing import Iterable, Generator
from scipy.special import stirling2


class CheckStatus(Enum):
    Correct = auto()
    TimeExceeded = auto()
    CostMismatch = auto()
    JobNoneAssignment = auto()
    WorkerWrongAssignment = auto()

    def __str__(self) -> str:
        s = f"[{self.name}] "
        match self:
            case CheckStatus.Correct:
                s += "Solution correct."
            case CheckStatus.TimeExceeded:
                s += "At least one job takes more time than available."
            case CheckStatus.CostMismatch:
                s += "Total cost does not equal the cost resulting from assignment."
            case CheckStatus.JobNoneAssignment:
                s += "There is at least 1 job with no workers."
            case CheckStatus.WorkerWrongAssignment:
                s += "At least one of the workers is assigned to more than 1 job or is not assigned to any."
            case _:
                pass
        return s


def check_solution(
    assignment: list[list[int]],
    total_cost: float,
    work: np.ndarray[float],
    time: np.ndarray[float],
    cost: np.ndarray[float],
    effi: np.ndarray[float],
    eps: float = 1e-5,
) -> CheckStatus:
    assert all(isinstance(arr, np.ndarray) for arr in (work, time, cost, effi)), "Expected Numpy arrays"
    assert all(len(arr.shape) == 1 for arr in (work, time, cost, effi)), "Expected 1-D arrays"
    assert work.shape == time.shape, "Expected `work` and `time` arrays to have the same shape"
    assert cost.shape == effi.shape, "Expected `cost` and `effi` arrays to have the same shape"
    assert (n_jobs := len(work)) <= (n_workers := len(cost)), "Expected n_workers >= n_jobs"

    if not all(len(group) > 0 for group in assignment):
        return CheckStatus.JobNoneAssignment

    if not all(sum(1 if worker in group else 0 for group in assignment) == 1 for worker in range(n_workers)):
        return CheckStatus.WorkerWrongAssignment

    if not all(w / effi[group].sum() <= t for w, t, group in zip(work, time, assignment)):
        return CheckStatus.TimeExceeded

    if abs(total_cost - sum(w * cost[group].sum() / effi[group].sum() for w, group in zip(work, assignment))) > eps:
        return CheckStatus.CostMismatch

    return CheckStatus.Correct


def set_partitions(L: Iterable, k: int):
    """Yields partitions of L into k non-empty sets represented as a list of lists. This
    implementation was taken from https://github.com/more-itertools/more-itertools.

    Args:
        L: Any finite iterable of length n >= k representing a set (i.e. order doesn't matter).
        k: Positive integer defining the number of non-empty partitions of L.

    Yields:
        Partition of L represented as a list of lists.
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


def sample_partition(n: int, k: int) -> list[list[int]]:
    """Implements a simple sampling scheme to sample uniformly a partition of set {0,..,n-1} into k
    non-empty sets. An explanation of why this scheme works can be found here:
    https://mathoverflow.net/questions/141999/how-to-efficiently-sample-uniformly-from-the-set-of-p-partitions-of-an-n-set.

    Args:
        n: Number of elements of the set.
        k: Number of partitions.

    Returns:
        Random partition of a set {0,..,n-1} into k non-empty sets represented as a list of lists.
    """
    assert n >= k >= 1, "Expected n >= k >= 1"

    if k == 1:
        return [[i for i in range(n)]]

    if k == n:
        return [[i] for i in range(n)]

    if np.random.random() <= (stirling2(n - 1, k - 1) / stirling2(n, k)):
        return [[n - 1], *sample_partition(n - 1, k - 1)]

    partition = sample_partition(n - 1, k)
    partition[np.random.randint(k)].append(n - 1)

    return partition
