import numpy as np

from tqdm import tqdm
from utils import sample_partition


def ba(
    work: np.ndarray[float],
    time: np.ndarray[float],
    cost: np.ndarray[float],
    effi: np.ndarray[float],
    max_iter: int = 100,
    sol_size: int = 100,
    top_frac: float = 0.2,
    top_attempts: int = 20,
    rest_attempts: int = 10,
    penalty_c: float = 100.0,
) -> tuple[float, list[list[int]]]:
    """TODO: Write a docstring"""

    assert all(isinstance(arr, np.ndarray) for arr in (work, time, cost, effi)), "Expected Numpy arrays"
    assert all(len(arr.shape) == 1 for arr in (work, time, cost, effi)), "Expected 1-D arrays"
    assert work.shape == time.shape, "Expected `work` and `time` arrays to have the same shape"
    assert cost.shape == effi.shape, "Expected `cost` and `effi` arrays to have the same shape"
    assert (n_jobs := len(work)) <= (n_workers := len(cost)), "Expected n_workers >= n_jobs"

    def encode(partition: list[list[int]]) -> list[int]:
        """Encodes the solution represented as a list of lists where each i-th list contains workers
        assigned to i-th job into a list of `n_workers` values where each value is an int between 0
        and `n_jobs - 1` which determines the job assigned to a given worker.
        """
        encoding = [None for _ in range(n_workers)]
        for i, job in enumerate(partition):
            for worker in job:
                encoding[worker] = i
        return encoding

    def decode(solution: list[int]) -> list[list[int]]:
        """Decodes the solution represented as a list of `n_workers` values where each value is an
        int between 0 and `n_jobs - 1` which determines the job assigned to a given worker into a
        list of lists where each i-th list contains workers assigned to i-th job.
        """
        partition = [[] for _ in range(n_jobs)]
        for worker, job in enumerate(solution):
            partition[job].append(worker)
        return partition

    def objective(solution: list[int]) -> float:
        """Computes the value of the objective function for the given solution solution defined as
        the negated value of the cost function for the defined above combinatorial optimization
        problem.
        """
        partition = decode(solution)
        return -sum(w * cost[p].sum() / effi[p].sum() for w, p in zip(work, partition))

    def penalty(solution: list[int]) -> float:
        """Computes the penalty for the given solution solution if it violates the 'max time to
        finish given job'-constraint. The penalty is computed as a sum of differences between the
        time it takes to finish the job and max allowed time for this job (if the time is smaller
        the summand is 0). This sum is then multiplied by a suitably chosen penalty coefficient.
        """
        partition = decode(solution)
        return penalty_c * sum(max(0, w / effi[p].sum() - t) for w, t, p in zip(work, time, partition))

    def alter(solution: list[int], p: float = 0.5) -> list[int]:
        """TODO: Write a docstring"""

        _solution = solution.copy()

        if np.random.random() < p:
            i, j = np.random.choice(n_workers, size=2, replace=False)
            _solution[i], _solution[j] = _solution[j], _solution[i]
        else:
            _solution[np.random.randint(n_workers)] = np.random.randint(n_jobs)

            while len(unassigned := (set(range(n_jobs)) - set(_solution))) != 0:
                for job in unassigned:
                    _solution[np.random.randint(n_workers)] = job

        return _solution

    fitness = lambda solution: objective(solution) - penalty(solution)
    best_fitness, best_assignment = -float("inf"), None

    # Sample initial solutions, encode them as linear arrays and sort according do fitness values
    solutions = [sample_partition(n_workers, n_jobs) for _ in tqdm(range(sol_size), desc="Sampling...")]
    solutions = [encode(partition) for partition in solutions]
    solutions = sorted(solutions, key=fitness, reverse=True)

    # NOTE: Is this really the whole algorithm ???
    for _ in (pbar := tqdm(range(max_iter))):
        # Update the best assignment and fitness value found so far
        if (best_iter_fitness := fitness(solutions[0])) > best_fitness:
            best_fitness, best_assignment = best_iter_fitness, solutions[0]

        # Update progress bar
        pbar.set_description(desc=f"Best fitness: {best_fitness:.3f}")

        new_solutions = []
        for i, sol in enumerate(solutions):
            for _ in range(top_attempts if i < top_frac * sol_size else rest_attempts):
                new_solutions.append(alter(sol))

        solutions = sorted(solutions + new_solutions, key=fitness, reverse=True)[:sol_size]

    return -best_fitness, decode(best_assignment)


if __name__ == "__main__":
    from utils import CheckStatus, check_solution

    work = np.array([0.0564, 0.5310, 0.8676])
    time = np.array([0.5367, 0.4316, 0.9047])
    cost = np.array([0.4241, 0.7391, 0.3058, 0.6060, 0.6486])
    effi = np.array([0.5549, 0.3572, 0.7561, 0.1787, 0.9503])

    best_cost, best_assignment = ba(work, time, cost, effi)

    if CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, work, time, cost, effi)):
        print(f"Solution incorrect: {status_code}")
    else:
        print("\n".join(["Solution found!", f"Best cost = {best_cost:.3f}", f"Best assignment = {best_assignment}"]))
