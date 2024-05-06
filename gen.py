from lib import *
from functools import cmp_to_key
from copy import deepcopy
import random


def newSolution():
    rows, cols = genSolution(NT, NM, NM)
    X = coo_matrix((np.ones(NM), (cols, rows)), shape=(NT, NM)).todense()
    return X

def cost(X, E, D, eta):
    cond_1 = np.sum(np.sum(X, axis=0) > 1)**2
    cond_2 = np.sum(np.sum(X, axis=1) < 1)**2
    cond_3 = E / (np.sum(np.multiply(eta, X), axis=1) + 1e-8) - D  # deadlines not met
    cond_3_sum = np.sum(np.where(cond_3 > 0, 10 * cond_3, 0.1 * cond_3))
    return cond_3_sum + cond_2 + cond_1

def metric(X):
    return cost(X, E, D, eta)

def cmp(X1, X2):
    return metric(X1) - metric(X2)

def cross(X1, X2):
    X = deepcopy(X1)
    rng = np.random.choice(list(range(X.shape[1])), size=X.shape[1]//2, replace=False)
    for i in rng:
        X[:,i] = deepcopy(X2[:,i])
    return X

def mutate(X1):
    X = deepcopy(X1)
    for x in range(X.shape[1]):
        fate = random.randint(1, 50)
        if fate == 1 :
            X[:,x] = np.zeros((X.shape[0],1))
        if fate == 2 or fate == 3:
            X[:,x] = np.zeros((X.shape[0],1))
            X[random.randint(0, X.shape[0]-1),x] = 1.
    return X
    
    

def genetic(E, D, eta):
    group_size = 200
    cross_size = 10
    solutions = [ newSolution() for _ in range(group_size) ]
    solutions.sort(key=cmp_to_key(cmp))
    for i in range(20):
        for x in solutions:
            print(metric(x))
        print("------")
        for i in range(cross_size):
            for j in range(i+1,cross_size):
                solutions.append(cross(solutions[i],solutions[j]))
        mutations = []
        for S in solutions:
            mutations.append(mutate(S))
        # for x in solutions:
        #     print(metric(x))
        # print("+------")
        solutions.sort(key=cmp_to_key(cmp))
        solutions = solutions[:group_size]


if __name__ == "__main__":
    NT = 10
    NM = 50

    E = np.random.random(NT)
    D = np.random.random(NT)
    eta = np.random.random((NT, NM))
    rows, cols = genSolution(NT, NM, NM)
    X = coo_matrix((np.ones(NM), (cols, rows)), shape=(NT, NM)).todense()
    print(rows)
    print(cols)
    print(X)
    genetic(E,D,eta)