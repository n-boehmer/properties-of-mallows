import numpy as np
from scipy.optimize import linear_sum_assignment
import copy

import gurobipy as gp
from gurobipy import GRB


def election_to_matrix(m,n, election):
    P = np.zeros((m, m))
    for v in election:
        for i in range(m):
            P[v[i], i] += 1/n
    return P




def l1(vector_1, vector_2):
    return np.linalg.norm(vector_1 - vector_2, ord=1)


def emd(vector_1, vector_2):
    vector_1 = copy.deepcopy(vector_1)
    dirt = 0.
    for i in range(len(vector_1) - 1):
        surplus = vector_1[i] - vector_2[i]
        dirt += abs(surplus)
        vector_1[i + 1] += surplus
    return dirt


def solve_matching_vectors(m,cost_table):
    """ Return: objective value, optimal matching """
    cost_table = np.array(cost_table)
    row_ind, col_ind = linear_sum_assignment(cost_table)
    return cost_table[row_ind, col_ind].sum(), list(col_ind)
    #return lp_matching(m,cost_table)


def get_matching_cost_positionwise(pos_election_1, pos_election_2,m,inner_distance):
    """ Return: Cost table """
    return [[inner_distance(pos_election_1[i], pos_election_2[j]) for i in range(m)] for j in range(m)]

#returned list says in position i to which candidate i from position 2 is matched to
def compute_positionwise_distance(pos_election_1, pos_election_2,m,inner_distance):
    """ Compute Positionwise distance between ordinal elections """
    cost_table = get_matching_cost_positionwise(pos_election_1, pos_election_2,m, inner_distance)
    #print(cost_table)
    return solve_matching_vectors(m,cost_table)


'''
def lp_matching(m,cost):
    mod = gp.Model("mip1")
    mod.setParam('OutputFlag', False)
    x = mod.addVars(m, m, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    opt = mod.addVar(vtype=GRB.CONTINUOUS)
    for j in range(m):
        mod.addConstr(gp.quicksum(x[i, j] for i in range(m)) == 1)
        mod.addConstr(gp.quicksum(x[j, i] for i in range(m)) == 1)
    mod.addConstr(gp.quicksum(cost[i, j] * x[i, j] for i in range(m) for j in range(m)) == opt)
    mod.setObjective(opt, GRB.MINIMIZE)
    mod.optimize()
    return mod.objVal
'''

def kemeny(pos_election,m):
    pos_id=np.zeros((m,m))
    for i in range(m):
        pos_id[i,i]=1
    cost_table = get_matching_cost_positionwise(pos_election,pos_id, m, emd)
    return solve_matching_vectors(m,cost_table)[0]
