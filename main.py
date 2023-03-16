import numpy as np
from mallows import *
from matrix import *
from phi_from_relphi import *
from positionwise_utils import *
from os import listdir
from os.path import isfile, join
import time
import random as rand
import math
import os
from matplotlib import pyplot
import tikzplotlib
import multiprocessing

from functools import partial
import statistics

def borda(matrix,m):
    borda_score=[0 for _ in range(m)]
    for i in range(m):
        for j in range(m):
            borda_score[i]+=(m-j)*matrix[i,j]
    return borda_score.index(max(borda_score))

def plural(m,phi):
    summed=0
    for i in range(m):
        summed+=phi**i
    return 1/summed

def plural_1(election):
    c=0
    for v in election:
        if v[0]==0:
            c+=1
    return c/len(election)


def max_plural(election,m):
    scores=[0 for _ in range(m)]
    for v in election:
        scores[v[0]]+=1
    return max(scores)/len(election)

def plurality_winner(election,m):
    scores=[0 for _ in range(m)]
    for v in election:
        scores[v[0]]+=1
    return scores.index(max(scores))

def borda_winner(election,m):
    scores=[0 for _ in range(m)]
    for v in election:
        for i in range(m):
            scores[v[i]]+=m-i
    return scores.index(max(scores))


def condorcet_winner_1(election,m,n,id):
    comparison=[0 for _ in range(m)]
    for v in election:
        for i in range(1,m):
            if v.index(id) < v.index(i):
                comparison[i] += 1
    for i in range(1,m):
        if comparison[i]<(n/2):
            return False
    return True


def condorcet_winner(election,m,n):
    comparision=np.zeros((m,m))
    for v in election:
        #v_index=[0 for _ in range(m)]


        for i,t in enumerate(v):
            for tt in (v[i:]):
                comparision[t,tt]+=1


    for i in range(m):
        winner=True
        for j in range(m):
            if comparision[i,j]<n/2:
                winner=False
        if winner==True:
            return i
    return -1

def average_position(election,m):
    position=0
    for v in election:
        position+=v.index(0)
    return position/(m*len(election))


def average_position_max(election,m):
    w=plurality_winner(election, m)
    position=0
    for v in election:
        position+=v.index(w)
    return position/(m*len(election))


def normalized_plur_score(election,m):
    c = 0
    for v in election:
        if v[0] == 0:
            c += 1/len(election)
    if c==0:
        return 0
    else:
        return ((1-c)/(m-1))/c

def normalized_plur_score_max(election,m):
    c= max_plural(election, m)
    return ((1-c)/(m-1))/c

def swap_distance(v1,v2):
    d=0
    m=len(v1)
    for i in range(m):
        for j in range(i,m):
            if not (v1.index(i)<v1.index(j))==(v2.index(i)<v2.index(j)):
                d+=1
    return d

def normalized_swap(election,m):
    summed_dis=0
    for v in election:
        summed_dis+=swap_distance(v,list(range(m)))
    return summed_dis/(len(election)*m*(m-1)/4)

def delete_candidate(election,m,num):
    election=copy.deepcopy(election)
    to_delete=random.sample(list(range(m)),num)
    for v in election:
        for i in to_delete:
            v.remove(i)
    new_map=[0 for _ in range(m)]
    c=0
    for i in range(m):
        if not i in to_delete:
            new_map[i]=c
            c+=1
    for v in election:
        for i in range(len(v)):
            v[i]=new_map[v[i]]
    return election

def read_real_world(folder,selec=20,shift=True,m=1000000):
    onlyfiles = [f for f in listdir("./"+folder) if isfile(join("./"+folder, f))]
    print(onlyfiles,m)
    onlyfiles.sort()
    elections=[]

    for name in onlyfiles:
        election = []
        start = False
        with open("./"+folder+"/"+name, "r") as a_file:
            for line in a_file:
                stripped_line = line.strip()
                li = list(stripped_line.split(","))
                if start:
                    numbers = [int(x) for x in li]
                    tmpvote = numbers[1:]
                    vote = tmpvote
                    if shift:
                        vote = [x-1 for x in vote if x < m]
                    else:
                        vote = [x for x in vote if x < m]

                    for j in range(numbers[0]):
                        election.append(copy.deepcopy(vote))
                if len(li) == 3:
                    start = True
        elections.append([name,election])
    return elections

def read_real_world_complete(folder,thresh=10,selec=20,shift=True,m=1000000):
    print(folder)
    onlyfiles = [f for f in listdir("./"+folder) if isfile(join("./"+folder, f))]
    onlyfiles.sort()
    elections=[]

    for name in onlyfiles:
        print(name)
        election = []
        start = False
        with open("./"+folder+"/"+name, "r") as a_file:
            for line in a_file:
                stripped_line = line.strip()
                li = list(stripped_line.split(","))
                if start:
                    numbers = [int(x) for x in li]
                    tmpvote = numbers[1:]
                    vote = tmpvote

                    #dirty
                    if shift:
                        vote = [x-1 for x in vote if x < m]
                    else:
                        vote = [x for x in vote if x < m]

                    for j in range(numbers[0]):
                        election.append(copy.deepcopy(vote))
                if len(li) == 3:
                    start = True

        canidates = set([item for sublist in election for item in sublist])


        for c in canidates:
            delete=False
            for v in election:
                if c not in v:
                    delete=True
            if delete:
                for v in election:
                    if c in v:
                        v.remove(c)
        print(len(election[0]))
        new_map = {}
        c = 0
        for i in election[0]:
            new_map[i] = c
            c += 1
        for j in range(len(election)):
            for i in range(len(election[j])):
                election[j][i] = new_map[election[j][i]]
        if len(election[0])>thresh and len(election)>10:
            elections.append([name,election])
    return elections



def read_real_world_football_week(big=True,selec=20,shift=True,m=1000000):
    onlyfiles = [f for f in listdir("./football week") if isfile(join("./football week", f))]
    print(onlyfiles,m)
    onlyfiles.sort()
    elections=[]

    for name in onlyfiles:
        election = []
        start = False
        with open("./football week/"+name, "r") as a_file:
            #print(name)
            for line in a_file:
                stripped_line = line.strip()
                li = list(stripped_line.split(","))
                if start:
                    # print(li)
                    numbers = [int(x) for x in li]
                    tmpvote = numbers[1:]
                    vote = tmpvote

                    #dirty
                    if shift:
                        vote = [x-1 for x in vote if x < m]
                    else:
                        vote = [x for x in vote if x < m]

                    for j in range(numbers[0]):
                        election.append(copy.deepcopy(vote))
                if len(li) == 3:
                    start = True
        lengths_big=[len(v) for v in election if len(v)>150]
        lengths_small = [len(v) for v in election if len(v) < 150]
        if len(lengths_big)==0:
            lengths_big=[-1]
        election_small=[]
        election_big=[]
        for v in election:
            if len(v)== max(lengths_big):
                election_big.append(v)
            if len(v)== max(lengths_small):
                election_small.append(v)

        if len(election_small) > 1:
            new_map = {}
            c = 0
            for i in election_small[0]:
                new_map[i] = c
                c += 1
            for v in election_small:
                for i in range(len(v)):
                    v[i] = new_map[v[i]]

        if len(election_big)>1:
            new_map = {}
            c = 0
            for i in election_big[0]:
                new_map[i] = c
                c += 1
            for v in election_big:
                for i in range(len(v)):
                    v[i] = new_map[v[i]]

        if len(election_big)>5 and len(election_small)>5 and max(lengths_small)>100:
            if big:
                elections.append([name+" Big",election_big])
            elections.append([name+" Sma",election_small])
    return elections








def parwise_comparision(phi,i,j):
    if phi==1:
        return 0
    k=j-i+1
    return 2*(1/(1-math.pow(phi,k))-((1-phi)*(k-1)*math.pow(phi,k-1))/((1-math.pow(phi,k))*(1-math.pow(phi,k-1)))-0.5)



def properties_exp(max_m,phi_type,e):
    m_list = range(10, max_m)
    for phi in [0.4,0.6,0.8,0.9,0.95,1]:
        results = []
        for m in m_list:
            if phi_type==1:
                lphi=phi
            if phi_type==2:
                lphi=binary_search_phi(m,phi)
            if e==1:
                results.append((m/(m-1))*(score_one(m, lphi)-1/(m)))
            if e==4:
                results.append((pos_can1(m, lphi)-1)/((m-1)/2))
            if e==6:
                results.append(calculateExpectedNumberSwaps(m,lphi)/((m * (m - 1)) / 4))
            if e==8:
                results.append(parwise_comparision(lphi,1,m))
        pyplot.plot(m_list, results, label="Phi " + str(phi))
    pyplot.xlabel("m")
    if phi_type == 1:
        pyplot.title("classical phi")
    if phi_type == 2:
        pyplot.title("swap phi")

    if e == 1:
        pyplot.ylabel("Probability that c_1 is ranked first in sampled vote")
    if e == 4:
        pyplot.ylabel("Expected normalized position of c_1 in sampled vote")
    if e == 6:
        pyplot.ylabel("Expected normalized swap distance of sampled vote")
    if e == 8:
        pyplot.ylabel("Probability that c_1 is ranked before c_m")
    pyplot.legend()
    if not os.path.exists("./results_properties_exp/" + str(e) + "/"):
        os.makedirs("./results_properties_exp/" + str(e) + "/")
    if phi_type==1:
        pyplot.savefig("./results_properties_exp/" + str(e) + "/classic"  + "_" + str(max_m) + ".png")
        tikzplotlib.save("./results_properties_exp/" + str(e) + "/clasic" + "_" + str(max_m) + ".tex")
    if phi_type==2:
        pyplot.savefig("./results_properties_exp/" + str(e) + "/norm" + "_" + str(max_m) + ".png")
        tikzplotlib.save("./results_properties_exp/" + str(e) + "/norm" + "_" + str(max_m) + ".tex")
    pyplot.close()





def check_propertie(e,m,n,itt,election):
    if e == 1:
        result = max_plural(election, m) / itt
    if e == 2:
        if plurality_winner(election, m) == borda_winner(election, m):
            result = 1 / itt
        else:
            result=0
    if e == 8:
        if plurality_winner(election, m) == 0:
            result = 1 / itt
        else:
            result=0
    if e==9:
        if condorcet_winner_1(election, m,n,0):
            result = 1 / itt
        else:
            result=0
    if e==11:
        if borda_winner(election, m)==0:
            result = 1 / itt
        else:
            result = 0
    if e==12:
        if condorcet_winner_1(election, m,n,plurality_winner(election, m)):
            result = 1 / itt
        else:
            result = 0
    if e == 4:
        result = average_position_max(election, m) / itt
    if e == 6:
        matrix = election_to_matrix(m, len(election), election)
        result = 3 * kemeny(matrix, m) / ((m * m - 1) * itt)
    return result

def properties(n,max_m,e,phi_type):
    itt = 1000
    m_list = range(10, max_m)
    for phi in [0.4,0.6,0.8,0.9, 0.95, 1]:
        results = []
        for m in m_list:
            print(m)
            if phi_type==1:
                lphi=phi
            if phi_type==2:
                lphi=binary_search_phi(m,phi)
            votes = generate_mallows(n * itt, m, lphi)
            elections = [votes[i:i + n] for i in range(0, len(votes), n)]
            pool = multiprocessing.Pool(8)
            func = partial(check_propertie, e,m,n,itt)
            result=pool.map(func, elections)
            pool.close()
            pool.join()
            results.append(sum(result))

        pyplot.plot(m_list, results, label="Phi " + str(phi))
    pyplot.xlabel("m")
    if phi_type==1:
        pyplot.title("classical phi")
    if phi_type==2:
        pyplot.title("swap phi")

    if e == 1:
        pyplot.ylabel("normalized maximum Plurality score")
    if e == 2:
        pyplot.ylabel("probability that Plurality winner is Borda winner")
    if e == 4:
        pyplot.ylabel("Average position of Plurality winner")
    if e == 6:
        pyplot.ylabel("Normalized distance from ID")
    if e == 8:
        pyplot.ylabel("Probability that c_1 is Plurality winner")
    if e == 9:
        pyplot.ylabel("Probability that c_1 is Condorcet winner")
    if e == 11:
        pyplot.ylabel("Probability that c_1 is Borda winner")
    if e == 12:
        pyplot.ylabel("Probability that Plurality winner is Condorcet winner")

    pyplot.legend()
    if not os.path.exists("./results_properties/"+str(e)+"/"):
        os.makedirs("./results_properties/"+str(e)+"/")
    if phi_type==1:
        pyplot.savefig("./results_properties/"+str(e)+"/"+  "classic_"+ str(n) + "_" + str(max_m) + ".png")
        tikzplotlib.save("./results_properties/"+str(e)+"/"+ "classic_"+ str(n) + "_" + str(max_m) + ".tex")
    if phi_type==2:
        pyplot.savefig("./results_properties/"+str(e)+"/"+  "norm_"+ str(n) + "_" + str(max_m) + ".png")
        tikzplotlib.save("./results_properties/"+str(e)+"/"+ "norm_"+ str(n) + "_" + str(max_m) + ".tex")
    pyplot.close()






def deleted_elections_properties_classic(n,max_m,e):
    m_list=range(10,max_m)
    itt=1000
    for phi in [0.4,0.6,0.8,0.9,0.95]:
        votes = generate_mallows(n*itt, max_m, phi)
        elections=[votes[i:i + n] for i in range(0, len(votes), n)]
        results=[]
        for m in m_list:
            del_elections=[delete_candidate(elections[i], max_m, max_m - m) for i in range(itt)]
            pool = multiprocessing.Pool(8)
            func = partial(check_propertie, e, m, n, itt)
            result = pool.map(func, del_elections)
            pool.close()
            pool.join()
            results.append(sum(result))
        pyplot.plot(m_list,results,label="Phi "+str(phi))
    pyplot.xlabel("m")
    if e == 1:
        pyplot.ylabel("normalized maximum Plurality score")
    if e == 2:
        pyplot.ylabel("probability that Plurality winner is Borda winner")
    if e == 4:
        pyplot.ylabel("Average position of Plurality winner")
    if e == 6:
        pyplot.ylabel("Normalized distance from ID")
    if e == 8:
        pyplot.ylabel("Probability that c_1 is Plurality winner")
    if e == 9:
        pyplot.ylabel("Probability that c_1 is Condorcet winner")
    if e == 11:
        pyplot.ylabel("Probability that c_1 is Borda winner")
    if e == 12:
        pyplot.ylabel("Probability that Plurality winner is Condorcet winner")
    pyplot.legend()
    if not os.path.exists("./syn_deletions_classic/"):
        os.makedirs("./syn_deletions_classic/")
    pyplot.savefig('./syn_deletions_classic/' + str(e) + "_" + str(n) + "_" + str(max_m) + "_" + str(phi) + ".png")
    tikzplotlib.save('./syn_deletions_classic/' + str(e) + "_" + str(n) + "_" + str(max_m) + "_" + str(phi) + ".tex")

    pyplot.close()





def deleted_election_properties_norm(n,max_m,e):
    m_list=range(10,max_m)
    itt=1000
    for lphi in [0.4,0.6,0.8,0.9,0.95]:
        phi = binary_search_phi(max_m, lphi)
        votes = generate_mallows(n*itt, max_m, phi)
        elections=[votes[i:i + n] for i in range(0, len(votes), n)]
        results=[]
        for m in m_list:
            print(m)
            del_elections = [delete_candidate(elections[i], max_m, max_m - m) for i in range(itt)]
            pool = multiprocessing.Pool(8)
            func = partial(check_propertie, e, m, n, itt)
            result = pool.map(func, del_elections)
            pool.close()
            pool.join()
            results.append(sum(result))
        with open("final_results.txt", "a") as output:
            output.write(str(results)+"\n")
        pyplot.plot(m_list,results,label="Deleted Phi "+str(lphi))

    pyplot.xlabel("m")
    if e == 1:
        pyplot.ylabel("normalized maximum Plurality score")
    if e == 2:
        pyplot.ylabel("probability that Plurality winner is Borda winner")
    if e == 4:
        pyplot.ylabel("Average position of Plurality winner")
    if e == 6:
        pyplot.ylabel("Normalized distance from ID")
    pyplot.legend()
    if not os.path.exists("./syn_deletions_normphi/"):
        os.makedirs("./syn_deletions_normphi/")
    pyplot.savefig('./syn_deletions_normphi/' + str(e) + "_" + str(n) + "_" + str(max_m) + "_" + str(phi) + ".png")
    tikzplotlib.save('./syn_deletions_normphi/' + str(e) + "_" + str(n) + "_" + str(max_m) + "_" + str(phi) + ".tex")
    pyplot.close()





def plot_size_football(e,week=False):
    xx=[]
    yy=[]
    y_map={}
    y_map_years_small={}
    y_map_years_big = {}
    year_to_m_small={}
    year_to_m_big={}
    if week:
        elections=read_real_world_football_week()
    else:
        elections=read_real_world_football()
    for electionn in elections:
        name=electionn[0]
        elec=electionn[1]
        m = len(elec[0])

        if e == 1:
            score = max_plural(elec, len(elec[0]))
        if e == 2:
            if plurality_winner(elec, len(elec[0])) == borda_winner(elec, len(elec[0])):
                score = 1
            else:
                score=0
        if e == 3:
            if condorcet_winner(elec, len(elec[0]), len(elec)) == plurality_winner(elec, len(elec[0])):
                score=1
            else:
                score=0
        if e == 4:
            score = average_position_max(elec, len(elec[0]))
        if e == 6:
            matrix = election_to_matrix(len(elec[0]), len(elec), elec)
            score= 3*kemeny(matrix,len(elec[0])) / (m*m-1)


        if m in xx:
            y_map[m]=y_map[m]+[score]
        else:
            y_map[m]=[score]

        if name[-3:]=="Big":
            year_to_m_big[name[2:6]]=m
            if name[2:6] in y_map_years_big:
                y_map_years_big[name[2:6]] = y_map_years_big[name[2:6]] + [score]
            else:
                y_map_years_big[name[2:6]] = [score]
        else:
            year_to_m_small[name[2:6]] = m
            if name[2:6] in y_map_years_small:
                y_map_years_small[name[2:6]] = y_map_years_small[name[2:6]] + [score]
            else:
                y_map_years_small[name[2:6]] = [score]
        xx.append(m)
        yy.append(score)

    if week:
        id="week"
    else:
        id="season"
    pyplot.scatter(xx, yy, marker='o')

    if e == 1:
        pyplot.ylabel("normalized Plurality score of Plurality winner")
    if e == 2:
        pyplot.ylabel("probability that Plurality winner is Borda winner")
    if e == 3:
        pyplot.ylabel("probability that Plurality winner is  Condorcet winner")
    if e == 4:
        pyplot.ylabel("Average position of Plurality winner")
    if e == 6:
        pyplot.ylabel("positionwise distance from ID")

    if not os.path.exists('./datset_comparison/'+id):
        os.makedirs('./datset_comparison/'+id)

    pyplot.xlabel("m")
    pyplot.savefig('./datset_comparison/'+id+'/football-points'+str(e) + ".png")
    tikzplotlib.save('./datset_comparison/'+id+'/football-points'+str(e) + ".tex")
    pyplot.close()

    x=[]
    y=[]
    small_set=[]
    large_set=[]
    for k, v in y_map.items():
        x.append(k)
        y.append(statistics.mean(v))
        if k>150:
            large_set=large_set+v
        else:
            small_set=small_set+v
    if e == 1:
        pyplot.ylabel("normalized Plurality score of Plurality winner")
    if e == 2:
        pyplot.ylabel("probability that Plurality winner is Borda winner")
    if e == 3:
        pyplot.ylabel("probability that Plurality winner is  Condorcet winner")
    if e == 4:
        pyplot.ylabel("Average position of Plurality winner")
    if e == 6:
        pyplot.ylabel("positionwise distance from ID")



    with open('res_week.txt', 'a') as f:
        f.write("SMALL "+str(e)+" "+str(statistics.mean(small_set))+'\n')
        f.write("LARGE " + str(e) +" "+ str(statistics.mean(large_set)) + '\n')

    pyplot.scatter(x, y, marker='o')

    pyplot.xlabel("m")
    pyplot.savefig('./datset_comparison/'+id+'/football-avg'+ str(e) + ".png")
    tikzplotlib.save('./datset_comparison/'+id+'/football-avg'+ str(e) + ".tex")
    pyplot.close()
    if week:
        NUM_COLORS = 20

        cm = pyplot.get_cmap('gist_rainbow')
        LINE_STYLES = ['v', 'o', '*', 'D']
        NUM_STYLES = len(LINE_STYLES)

        x = []
        y = []
        map_year_to_color={}
        map_year_to_marker = {}
        col=0
        for k, v in y_map_years_big.items():
            print(k)
            x.append(year_to_m_big[k])
            y.append(statistics.mean(v))
            pyplot.scatter(year_to_m_big[k], statistics.mean(v), label=k,color=cm(col//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS),marker=LINE_STYLES[col%NUM_STYLES])
            map_year_to_color[k]=cm(col//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS)
            map_year_to_marker[k] = LINE_STYLES[col%NUM_STYLES]
            col+=1
        x2 = []
        y2 = []
        for k, v in y_map_years_small.items():
            x2.append(year_to_m_small[k])
            y2.append(statistics.mean(v))
            pyplot.scatter(year_to_m_small[k], statistics.mean(v), label=k,color=map_year_to_color[k],marker=map_year_to_marker[k])
        pyplot.xlabel("m")
        pyplot.ylim([0, 1])
        if e == 1:
            pyplot.ylabel("normalized Plurality score of Plurality winner")
        if e == 2:
            pyplot.ylabel("probability that Plurality winner is Borda winner")
        if e == 3:
            pyplot.ylabel("probability that Plurality winner is  Condorcet winner")
        if e == 4:
            pyplot.ylabel("Average position of Plurality winner")
        if e == 6:
            pyplot.ylabel("positionwise distance from ID")
        pyplot.legend()
        pyplot.savefig('./datset_comparison/' + id + '/football-year' + str(e) + ".png")
        tikzplotlib.save('./datset_comparison/' + id + '/football-year' + str(e) + ".tex")

        pyplot.close()



def plot_size_kemeny(folder,e,complete=False,tresh=10,tdf=False):
    xx=[]
    yy=[]
    colors_tdf=[]
    if complete:
        elections=read_real_world_complete(folder,thresh=tresh)
    else:
        elections=read_real_world(folder)
    for electionn in elections:
        name=electionn[0]
        if tdf:
            tdf_name=int(name[4:8])
        else:
            tdf_name=1
        tdf_color_scheme=(tdf_name-1910)/110
        colors_tdf.append(tdf_name)
        print(tdf_name,tdf_color_scheme)
        elec=electionn[1]
        m = len(elec[0])
        print(len(elec[0]))
        if e == 6:
            matrix = election_to_matrix(len(elec[0]), len(elec), elec)
            score = 3 * kemeny(matrix, len(elec[0])) / (m * m - 1)
        xx.append(m)
        yy.append(score)
    pyplot.scatter(xx, yy, s=10, marker='o')

    os.makedirs('./datset_comparison/' + folder + '/',exist_ok=True)

    pyplot.xlabel("m")
    pyplot.ylim([0, 1])
    if e == 6:
        pyplot.ylabel("distance from ID")
    pyplot.legend()
    pyplot.savefig('./datset_comparison/' + folder + '/' + str(e) + ".png")
    tikzplotlib.save('./datset_comparison/' + folder + '/' + str(e) + ".tex")
    pyplot.close()











#Generates figures comparing exactly computable properties of elections and votes sampled from the classic and normalized Mallows model.
#Generates Figures 1, 3a+3b+3c. Figures can be found in folder "results_properties_exp"
for e in [1,4,6,8]:
    m=201
    for phi_type in [1,2]:
        properties_exp(m,phi_type,e)




#Generates figures comparing properties computed by sampling of elections and votes sampled from the classic and normalized Mallows model.
#Generates Figures 3d, Figure 4, Figure 6 and Figure 7. Figures can be found in folder "results_properties"
for e in [1,2,4,6,8,9,11,12]:
    n=100
    m=201
    for phi_type in [1,2]:
        properties(n,m,e,phi_type)



# Generates figures on properties of elections and votes sampled from the classic and normalized Mallows model in case some candidates are deleted uniformly at random from the election.
#Generates Figures 2, Figures 8, Figures 9. Figures can be found in folder "syn_deletions_classic" (for the classic Mallows model) and "syn_deletions_normphi" (for the normalized Mallows model)
for e in [1,2,4,6]:
    n = 100
    m = 201
    deleted_election_properties_norm(n,m,e)
    deleted_elections_properties_classic(n,m,e)



# Generates figures showing how the normalized positionwise distance from ID depends on the number of alternatives in real-world elections.
# Generates Figures 5b+c in folder "data_comparision"
for e in [6]:
    plot_size_kemeny("spotify month raw",e,complete=True,tresh=50)
    plot_size_kemeny("Tour de France raw", e, complete=True)



#Generates figures showing how different properties depend on the number of alternatives in the football elections, also outputs average values for the different types of data.
#Generates Figure 5a in folder "data_comparison"
#Generates information needed for Table 1 in file "res_week"
for e in [1,2,3,4,6]:
    plot_size_football(e,week=True)
