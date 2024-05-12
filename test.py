import numpy as np

from bf import bf
from ga import ga
from ba import ba
from utils import CheckStatus, check_solution


def log_sol(best_assignment, best_cost, work, time, cost, effi):
    if best_assignment is None:
        print("Problem infeasible")
    elif CheckStatus.Correct != (status_code := check_solution(best_assignment, best_cost, work, time, cost, effi)):
        print(f"Solution incorrect: {status_code}")
    else:
        print("Solution found!")
        print(f"Best cost: {best_cost:.3f}")
        print(f"Best assignment: {best_assignment}")
    print("=" * 80)


n_jobs, n_workers = 4, 12

work = 0.1 + np.random.random(n_jobs)
time = 0.3 + np.random.random(n_jobs)
cost = np.random.random(n_workers)
effi = 0.2 + np.random.random(n_workers)

print(f"work = np.array({work.tolist()})")
print(f"time = np.array({time.tolist()})")
print(f"cost = np.array({cost.tolist()})")
print(f"effi = np.array({effi.tolist()})")

print("\n" + "\n".join(["=" * 80, "Brute force", "-" * 80]))
best_cost, best_assignment = bf(work, time, cost, effi)
log_sol(best_assignment, best_cost, work, time, cost, effi)

print("\n" + "\n".join(["=" * 80, "Genetic Algorithm", "-" * 80]))
best_cost, best_assignment = ga(work, time, cost, effi, max_iter=500, pop_size=1_000)
log_sol(best_assignment, best_cost, work, time, cost, effi)

print("\n" + "\n".join(["=" * 80, "Bees Algorithm", "-" * 80]))
best_cost, best_assignment = ba(work, time, cost, effi, max_iter=30, sol_size=1_000)
log_sol(best_assignment, best_cost, work, time, cost, effi)


"""
#0 Simple, small problem with sparse solution space and non-trivial optimal solution (i.e. workers
are not partitioned like 1-1-3 but 1-2-2). Good for checking if the implementation even works.
```
work = np.array([0.0564, 0.5310, 0.8676])
time = np.array([0.5367, 0.4316, 0.9047])
cost = np.array([0.4241, 0.7391, 0.3058, 0.6060, 0.6486])
effi = np.array([0.5549, 0.3572, 0.7561, 0.1787, 0.9503])
```

#1 Just a non-trivial solution. Both GA and BA find optimal solution.
```
work = np.array([0.9555409465881047, 0.6269190124729263, 0.5479486960426908, 0.6992022687514542, 0.06917410978120286])
time = np.array([0.5384266901736443, 0.7238977381794904, 1.0599886440199735, 0.779270226487863, 0.9028674080871559])
cost = np.array([0.6602065571078749, 0.2807850642877451, 0.9284766051964652, 0.19150579268042478, 0.6765320559588606, 0.5271867220475578, 0.6700141100582238, 0.03643346165331485, 0.853182488220499, 0.5948784275950318])
effi = np.array([0.9257154338706128, 0.7063183477273554, 1.182687016021314, 0.4269195541294259, 1.0573736190025793, 0.6487006350861948, 1.1924149689289327, 0.7197553562217296, 0.7901538060662321, 0.3201802362309358])
```

#2 Interesting example showing that penalty strength does matter. GA and BA (with default penalty
coeff. = 100.0) find a solution which just slightly violates the 'max time to finish job'-constraint
but is much cheaper than optimal.
```
work = np.array([0.5383893030276895, 0.7623445150296647, 0.7329372276566987, 0.24336025087201055])
time = np.array([1.2844650507547961, 0.4314535572183091, 0.6121457224341011, 1.02327118101561])
cost = np.array([0.08008510126545654, 0.17988264509534468, 0.3226554777028282, 0.4401550523248945, 0.3673397647739226, 0.5661463404816119, 0.13021553785315654, 0.9625780700394736, 0.589570173828238, 0.619210385512805])
effi = np.array([1.195619110148109, 0.984277276061462, 1.07789586166882, 0.7470223687503699, 0.6946223275619035, 1.137267936132908, 0.6073026291259944, 1.0084775128235917, 0.5838963297129256, 0.9934167367165119])
```
"""
