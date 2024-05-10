import numpy as np
from enum import Enum, auto


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

    if not all(work[i] / sum(effi[worker] for worker in group) <= time[i] for i, group in enumerate(assignment)):
        return CheckStatus.TimeExceeded

    if abs(total_cost - sum(w * cost[group].sum() / effi[group].sum() for w, group in zip(work, assignment))) > eps:
        return CheckStatus.CostMismatch

    return CheckStatus.Correct
