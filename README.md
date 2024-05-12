## Brief description

This repository contains implementations of Genetic Algorithm (GA) and Bees Algorithm (BA)
metaheuristics which solve the following combinatorial optimization problem (which is a variation of
Generalized Assignment Problem)
```
min_{x} sum_{i=1,..,n} w[i] * ( sum_{j=1,..,m} c[j] * x[i,j] ) / ( sum_{j=1,..,m} e[j] * x[i,j] )
s.t.    forall i=1,..,n : sum_{j=1,..,m} x[i,j] >= 1
        forall j=1,..,m : sum_{i=1,..,n} x[i,j] == 1
        forall i=1,..,n : w[i] / ( sum_{j=1,..,m} e[j] * x[i,j] ) <= t[i]
```
This problem is a natural generalization of GAP for situations where we need to assign workers to
tasks in such a way that:
* every task has at least one worker
* every worker has exactly one task
* a worker is characterized by two positive numbers: cost per time $c$ and efficacy $\eta$
* a task is characterized by two positive numbers: work $w$ and max time $t$
* the efficacy of a group of workers is equal to the sum of efficacies of the workers
* the time it takes to finish a task with work $w$ is equal $w / \eta$, where $\eta$ is the efficacy
  of the group of workers
* the total cost of the group of workers is equal to the sum of costs per time of the workers times
  the time it takes to finish a task 
* we want to minimize the total cost of all workers

## Installation

Clone this repository using
```
$ git clone https://github.com/barhanc/or-proj.git
```
then create virtual environment and install required dependencies using
```
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```