import math
#Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps in a vote sampled from Mallows model
def calculateExpectedNumberSwaps(m,phi):
    if phi==1:
        return (m * (m - 1)) / 4
    res= phi*m/(1-phi)
    for j in range(1,m+1):
        res = res + (j*(phi**j))/((phi**j)-1)
    return res

#Given the number m of candidates and a value of norm_phi\in [0,1], this function returns a value of phi such that in a vote sampled from Mallows model with this parameter the expected number of swaps is norm_phi*(m * (m - 1)) / 4
def binary_search_phi(m,norm_phi):
    if norm_phi==0:
        return 0
    if norm_phi==1:
        return 1
    exp_abs=norm_phi * (m * (m - 1)) / 4
    low=0
    high=1
    while low <= high:
        #print(low)
        #print(high)
        mid = (high + low) / 2
        cur=calculateExpectedNumberSwaps(m, mid)
        if abs(cur-exp_abs)<1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1

def pos_can1(m,phi):
    #print(m,phi)
    if phi==1:
        return (m+1)/2
    return 1/(1-phi)-m*math.pow(phi,m)/(1-math.pow(phi,m))

def phi_from_relphi_positions(num_candidates,relphi=None):
    if relphi is None:
        relphi = rand.random()
    if relphi==1:
        return 1
    exp_abs=1+relphi*(num_candidates-1)/2
    #
    low=0
    high=1
    while low <= high:
       #print(exp_abs)
        mid = (high + low) / 2
        cur=  pos_can1(num_candidates,mid)
        #print(cur)
        if abs(cur-exp_abs)<1e-8:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid
        #print(high)
        #exit()

    # If we reach here, then the element was not present
    return -1

def score_one(m,phi):
    deno=1
    for j in range(1,m):
        deno+=phi**j
    return 1/deno

def phi_from_relphi_score(num_candidates,relphi=None):
    if relphi is None:
        relphi = rand.random()
    if relphi==1:
        return 1
    exp_abs=1/(relphi*(num_candidates-1)+1)
    #print(exp_abs)
    #exit()
    low=0
    high=1
    while low <= high:
       #print(exp_abs)
        mid = (high + low) / 2
        cur=  score_one(num_candidates,mid)
        #print(low,high)
        #print(cur)
        if abs(cur-exp_abs)<1e-5:
            return mid
        # If x is greater, ignore left half
        if cur > exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur < exp_abs:
            high = mid
        #print(high)
        #exit()

    # If we reach here, then the element was not present
    return -1