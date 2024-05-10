import numpy as np

from tqdm import tqdm
from utils import check_solution, CheckStatus
from more_itertools import sample, set_partitions


def ga(
    work: np.ndarray[float],
    time: np.ndarray[float],
    cost: np.ndarray[float],
    effi: np.ndarray[float],
    max_iter: int = 100,
    pop_size: int = 100,
    rep_ratio: float = 0.1,
    penalty_c: float = 100.0,
) -> tuple[float, list[list[int]]]:
    """Implements Genetic Algorithm (GA) to solve the following combinatorial optimization problem

    minimize sum_{i=1,..,n} work[i] * (sum_{j=1,..,m} cost[j] * x[i,j]) / (sum_{j=1,..,m} effi[j] * x[i,j])
    s.t.     ∀ i=1,..,n; j=1,..,m : x[i,j] ∈ {0,1}
             ∀ i=1,..,n : sum_{j=1,..,m} x[i,j] >= 1
             ∀ j=1,..,m : sum_{i=1,..,n} x[i,j] == 1
             ∀ i=1,..,n : work[i] / (sum_{j=1,..,m} effi[j] * x[i,j]) <= time[i]

    where x[i,j] = 1 if j-th worker is assigned to i-th job else 0, n is the number of jobs and m is
    the number of workers. We assume that m >= n.

    A candidate solution in the population is represented as a list of `n_workers` values where each
    value is an int between 0 and `n_jobs - 1` which determines the job assigned to a given worker.
    Fitness value of a candidate is calculated as a difference between the value of the objective
    function (negation of the cost function) and a penalty value for violating the 'max time to
    finish given job'-constraint. To select parents we use a simple deterministic binary tournament
    scheme where firstly we repeatedly sample a pair of candidates and add the one with higher
    fitness value to the set of `parents` and then sample a pair of parents from the `parents` set.
    A new candidate is created by applying the `cross` operator to the pair of parents, followed by
    a `mutate` operator. Cross-over operator is implemented as a simple one-point crossover followed
    by a random resolution of 'at least one worker for every job'-constraint violations. Mutation
    operator is implemented as a single swap of two random unique elements of the candidate
    solution. Newly generated candidate solutions then replace the candidates in the population
    having the lowest fitness values.

    Args:
        work: Array of work values for each job. Shape (n_jobs,).
        time: Array of max times for each job. Shape (n_jobs,).
        cost: Array of costs per unit of time for each worker. Shape (n_workers,).
        effi: Array of efficacies (i.e. amount of work a worker can do in unit time) for each
              worker. Shape (n_workers,).
        max_iter: Max number of iterations.
        pop_size: Fixed size of the population of candidate solutions.
        rep_ratio: Fraction of the population that gets replaced by the new candidates in every
                   iteration.
        penalty_c: Penalty coefficient.

    Returns:
        Value of the cost function including the penalty term (negation of the fitness value) and
        the corresponding assignment of workers to jobs as a list of lists where each i-th list
        contains workers assigned to i-th job found using the Genetic Algorithm.
    """
    assert all(isinstance(arr, np.ndarray) for arr in (work, time, cost, effi)), "Expected Numpy arrays"
    assert all(len(arr.shape) == 1 for arr in (work, time, cost, effi)), "Expected 1-D arrays"
    assert work.shape == time.shape, "Expected `work` and `time` arrays to have the same shape"
    assert cost.shape == effi.shape, "Expected `cost` and `effi` arrays to have the same shape"
    assert (n_jobs := len(work)) <= (n_workers := len(cost)), "Expected n_workers >= n_jobs"
    assert rep_ratio < 1, "Expected `rep_ratio` to be smaller than 1"

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

    def decode(candidate: list[int]) -> list[list[int]]:
        """Decodes the solution represented as a list of `n_workers` values where each value is an
        int between 0 and `n_jobs - 1` which determines the job assigned to a given worker into a
        list of lists where each i-th list contains workers assigned to i-th job.
        """
        partition = [[] for _ in range(n_jobs)]
        for worker, job in enumerate(candidate):
            partition[job].append(worker)
        return partition

    def objective(candidate: list[int]) -> float:
        """Computes the value of the objective function for the given candidate solution defined as
        the negated value of the cost function for the defined above combinatorial optimization
        problem.
        """
        partition = decode(candidate)
        return -sum(w * cost[p].sum() / effi[p].sum() for w, p in zip(work, partition))

    def penalty(candidate: list[int]) -> float:
        """Computes the penalty for the given candidate solution if it violates the 'max time to
        finish given job'-constraint. The penalty is computed as a sum of differences between the
        time it takes to finish the job and max allowed time for this job (if the time is smaller
        the summand is 0). This sum is then multiplied by a suitably chosen penalty coefficient.
        """
        partition = decode(candidate)
        return penalty_c * sum(max(0, w / effi[p].sum() - t) for w, t, p in zip(work, time, partition))

    def cross(parent_x: list[int], parent_y: list[int]) -> list[int]:
        """Implements simple one-point cross-over operator followed by a random resolution of
        'at least one worker for every job'-constraint violations.
        """
        crosspoint = np.random.randint(n_workers)
        child = parent_x[:crosspoint] + parent_y[crosspoint:]

        while len(unassigned := (set(range(n_jobs)) - set(child))) != 0:
            for job in unassigned:
                child[np.random.randint(n_workers)] = job

        return child

    def mutate(candidate: list[int]) -> list[int]:
        """Implements simple mutation operator which swaps two random unique elements of the
        candidate solution.
        """
        i, j = np.random.choice(n_workers, size=2, replace=False)
        _candidate = candidate.copy()
        _candidate[i], _candidate[j] = _candidate[j], _candidate[i]
        return _candidate

    best_fitness, best_assignment = -float("inf"), None

    # Sample initial population and encode candidates as linear arrays
    population = [sample(set_partitions(range(n_workers), n_jobs), k=1)[0] for _ in range(pop_size)]
    population = [encode(candidate) for candidate in population]

    for _ in (pbar := tqdm(range(max_iter))):
        # Calculate fitness values for every candidate in population
        fitness = [objective(candidate) - penalty(candidate) for candidate in population]

        # Update the best assignment and fitness value found so far
        best_iter_assignment, best_iter_fitness = population[np.argmax(fitness)], np.max(fitness)
        if best_iter_fitness > best_fitness:
            best_fitness, best_assignment = best_iter_fitness, best_iter_assignment

        # Update progress bar
        pbar.set_description(f"Best fitness: {best_fitness:.3f} | Best iter. fitness: {best_iter_fitness:.3f}")

        # Deterministic Binary Tournament selection
        parents = []
        for _ in range(int(rep_ratio * pop_size)):
            i, j = np.random.choice(pop_size, size=2, replace=False)
            parents.append(i if fitness[i] > fitness[j] else j)

        # Apply cross-over and mutation operators
        children = []
        for _ in range(int(rep_ratio * pop_size)):
            i, j = np.random.choice(parents, size=2, replace=False)
            children.append(mutate(cross(population[i], population[j])))

        # Replace the len(children) worst candidates in population with newly created candidates to
        # maintain constant population size
        for i, child in zip(np.argsort(fitness)[: len(children)].tolist(), children):
            population[i] = child

    return -best_fitness, decode(best_assignment)


# Example usage
if __name__ == "__main__":
    work = np.array([0.0564, 0.5310, 0.8676])
    time = np.array([0.5367, 0.4316, 0.9047])
    cost = np.array([0.4241, 0.7391, 0.3058, 0.6060, 0.6486])
    effi = np.array([0.5549, 0.3572, 0.7561, 0.1787, 0.9503])

    best_cost, best_assignment = ga(work, time, cost, effi)

    if CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, work, time, cost, effi)):
        print(f"Solution incorrect: {status_code}")
    else:
        print("\n".join(["Solution found!", f"Best cost = {best_cost:.3f}", f"Best assignment = {best_assignment}"]))
