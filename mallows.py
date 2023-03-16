import numpy as np
import random


def computeInsertionProbas(i, phi):
    probas = (i + 1) * [0]
    for j in range(i + 1):
        probas[j] = pow(phi, (i + 1) - (j + 1))
    return probas


def weighted_choice(choices):
    total = 0
    for w in choices:
        total = total + w
    r = np.random.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(choices):
        if upto + w >= r:
            return i
        upto = upto + w
    assert False, "Shouldn't get here"


def mallowsVote(m, insertion_probabilites_list):
    vote = [0]
    for i in range(1, m):
        index = weighted_choice(insertion_probabilites_list[i - 1])
        vote.insert(index, i)
    return vote

def generate_mallows(n,m,phi):
    insertion_probabilites_list = []
    for i in range(1, m):
        insertion_probabilites_list.append(computeInsertionProbas(i, phi))
    V = []
    for i in range(n):
       vote = mallowsVote(m, insertion_probabilites_list)
       V += [vote]
    return V


def generate_mallows_mixture(n,m,phi1,phi2):
    weight=0.5
    c1= list(range(m))
    random.shuffle(c1)
    c2= list(reversed(c1))
    print(c1,c2)


    insertion_probabilites_list1 = []
    insertion_probabilites_list2 = []
    for i in range(1, m):
        insertion_probabilites_list1.append(computeInsertionProbas(i, phi1))
        insertion_probabilites_list2.append(computeInsertionProbas(i, phi2))

    V = []
    for i in range(n):
        probability = np.random.random()
        if probability <= weight:
            cur_center = c1
            insertion_probabilites_list=insertion_probabilites_list1
        else:
            cur_center = c2
            insertion_probabilites_list = insertion_probabilites_list2


        vote = mallowsVote(m, insertion_probabilites_list)
        mapped_vote=[]
        for v in vote:
            mapped_vote+=[cur_center[v]]
        V += [mapped_vote]
    return V
