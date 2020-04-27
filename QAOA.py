#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:57:14 2020

@author: YXZhang

A project to simulate a QAOA algorithm, inspired by lattice QED, to solve 
network flow problems.

Specifically, Gauss's law is comparable to the "hard constraints" in flow problems. Thus,
it's natural to borrow the Hamiltionian from lattice QED as a mixer to solve the 
problem in QAOA: given an initial state, it only evolves in the feasible solution subspace

However, the Hamiltionian can still create some "plausible but unfeasible" configurations:
those with isolate loops. We can modify to get the "constrainted" version of QED mixer.
We compare all three mixers here.

Qubits(qutrits for undirected graph) are defined on the edges for each sink/source pair
"""
from matplotlib import (
    pyplot)
import scipy
from math import pi
import numpy as np
import networkx as nx
from scipy.linalg import expm
import random
import time
""" Part I
This part concerns purely about graph properties: finding and encoding emelemtary cycles 
on a planar graph, which is necessary for preparing the mixers.

I'm using a brutal method here: find all cycles in a graph and then only select elememtary
ones by adding a cut to their length. This might fail in some more complecated graphs, but 
works fine for the triangle graphs we're testing.
"""
#a naive program that outputs all cycles in a undirected graph

def loopfinding():
    global Graph_edges
    global cycles
    for edge in Graph_edges:
        for node in edge:
            findNewCycles([node])

def findNewCycles(path):
    start_node = path[0]
    next_node= None
    sub = []

    #visit each edge and each node of each edge
    for edge in Graph_edges:
        node1, node2 = edge
        if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
                if not visited(next_node, path):
                        # neighbor node not on path yet
                        sub = [next_node]
                        sub.extend(path)
                        # explore extended path
                        findNewCycles(sub);
                elif len(path) > 2  and next_node == path[-1]:
                        # cycle found
                        p = rotate_to_smallest(path);
                        inv = invert(p)
                        if isNew(p) and isNew(inv):
                            cycles.append(p)

def invert(path):
    return rotate_to_smallest(path[::-1])

#  rotate cycle path such that it begins with the smallest node
def rotate_to_smallest(path):
    n = path.index(min(path))
    return path[n:]+path[:n]

def isNew(path):
    return not path in cycles

def visited(node, path):
    return node in path

#returns the set of all undirected loop; alldloop gives the set of edges of all simple loops;
# parity means the direction of an edge comparing to the defult (as sorted by python) order.
# for instance,  [(1,2,3)] returns [[1,2],[1,3],[2,3]] and [1,-1,1].
def Dloop(cycles):
    alldloop = [] 
    loopparity = []
    for cycle in cycles :      
        dloop = [] 
        parity = []
        if len(cycle) <4:
            for i in range (len(cycle)):
                if cycle[i-1] < cycle[i]:
                    dloop.append([cycle[i-1], cycle[i]])
                    parity.append(1)
                else:
                    dloop.append([cycle[i], cycle[i-1]])
                    parity.append(-1)
            alldloop.append(dloop)
            loopparity.append(parity)
    return alldloop, loopparity

""" Part II
This section is for Hamiltonians and all used functions. The dimension for the Hilbert 
space for directed graph is simply (k+1)^e;
we use the convention that first k rows represent the flow of the first edge, etc.
Currently, the cost function is for min weight path finding
"""
# the cost function is just for 
def prepare_cost(k, e):
    C = complex(1,0)*np.zeros((2*k+1)**e)
    for i in range ((2*k+1)**e):
        tem = 0
        for m in range (e):
            if m in [1,4]:
                flow = abs(int ((i%(2*k+1)**(m+1))/((2*k+1)**(m)))-1) +1 #absolute value of flow
            else:
                flow = abs (int ((i%(2*k+1)**(m+1))/((2*k+1)**(m)))-1)                 
            tem += flow *graph_weight[m]       
        C[i] = tem 
    return C

# x cost is only 
def prepare_xcost(k, Graph_edges, e, s, t, n, pen):
    C = complex(1,0)*np.zeros((2*k+1)**e)
    penalty = pen
    for i in range ((2*k+1)**e):
        tem = 0
        C[i] += penalty * violation(i, Graph_edges, s, t, n)
        for m in range (e):
            flow = abs (int ((i%(2*k+1)**(m+1))/((2*k+1)**(m)))-1)                   
            if flow > 1 :
                tem +=  flow * graph_weight[m]
            elif flow == 1:
                tem += graph_weight[m]   
        C[i] = tem+C[i]             
    return C

def convertflow(i, k, e): #convert a encoding into a list of flows
    fl = []
    for m in range (e):
        flow = (int ((i%(2*k+1)**(m+1))/((2*k+1)**(m)))-1) #directed in this case
        fl.append(flow)
    return fl

#convert a list of flows in to state #
def convertstate(flow):
    state = 0
    for i in range (len(flow)):
        state += (flow[i] + 1)*(2*k + 1)**i
    return int(state)

#voilation in x mixer
def violation(i, edges, s, t, n):
    flows = convertflow(i, k, e)
    violate = 0
    for x in range (n):
        temf = 0
        for i in range(len(edges)):
            if edges[i][0] == int(x):
                temf += flows[i]
            elif edges[i][1] == int(x):
                temf = temf - flows[i]
        if int(x) in s and int(temf) != int(1):
            violate += 1
        elif int(x) in t and int(temf) != int(-1) :
            violate += 1
        elif int(x) not in t and int(x) not in s and int(temf) != int(x):
            violate += 1
    return violate

#x-mixer as difined in qutrit (3-level) state:
#    tensor product of [(0,1,1),(1,0,1),(1,1,0)] on each local site
def prepare_xmixer(k, e):
    M = np.zeros(((2*k+1)**e, (2*k+1)**e))
    for i in range ((2*k+1)**e):
        for j in range (e):
            flow = int ((i%((2*k+1)**(j+1)))/((2*k+1)**(j)))
            f1 = ((flow + 1)%(2*k+1)-flow) *(2*k+1)**(j) +i
            f2 = ((flow + 2)%(2*k+1)-flow) *(2*k+1)**(j) + i
            M[f1, i] = 1/np.sqrt(2)
            M[i, f1] = 1/np.sqrt(2)
            M[f2, i] = 1/np.sqrt(2)
            M[i, f2] = 1/np.sqrt(2)
    return M

#naive QED mixer
def prepare_mixer_naive(alldloop, loopparity, k, e):
    M = np.zeros(((2*k+1)**e, (2*k+1)**e))
    for p in range (len(loopparity)):
        index = []# the indices of a loop's edges
        for q in range (len(loopparity[p])):
            x = Graph_edges.index(alldloop[p][q])
            index.append(x)
        parity = loopparity[p]
        for i in range ((2*k+1)**e):
            if check_unitary_naive(parity, index, i)[0] == 1:
                ff = check_unitary(parity, index, i)[1]
                M[ff, i] = 1
                M[i, ff] = 1
    return M

#constrained QED-mixer
def prepare_mixer(alldloop, loopparity, k, e):
    M = np.zeros(((2*k+1)**e, (2*k+1)**e))
    for p in range (len(loopparity)):
        index = []# the indices of a loop's edges
        for q in range (len(loopparity[p])):
            x = Graph_edges.index(alldloop[p][q])
            index.append(x)
        parity = loopparity[p]
        for i in range ((2*k+1)**e):
            if check_unitary(parity, index, i)[0] == 1:
                ff = check_unitary(parity, index, i)[1]
                M[ff, i] = 1
                M[i, ff] = 1
    return M

#to check whether applying a loop operator on some state would return 0; adding flow 
#to a edge with max flow would kill the state.
def check_unitary_naive(parity, index, number):
    ff = number
    for i in range (len(parity)):        
        flow = int ((number%((2*k+1)**(index[i]+1)))/((2*k+1)**(index[i]))) -1
        if flow + parity[i] > k:            
            return 0, 0 
        elif flow + parity[i] < -k:
            return 0, 0
        else:
            ff += parity[i]*(2*k+1)**(index[i])
    return 1, ff

#to check whether applying a loop operator on some state would return 0; adding flow 
#to a edge with max flow would kill the state; furthermore, isolated loops are not allowed.
def check_unitary(parity, index, number):  
    ff = number
    for i in range (len(parity)):        
        flow = int ((number%((2*k+1)**(index[i]+1)))/((2*k+1)**(index[i]))) -1
        if flow + parity[i] > k:            
            return 0, 0 
        elif flow + parity[i] < -k:
            return 0, 0
        else:
            ff += parity[i]*(2*k+1)**(index[i])
    x = int((isolated_loop(number) + isolated_loop(ff))/2)
    return x, ff

def isolated_loop(ff): # this checks whether a config contains isolated loop
    input_state = np.asarray(convertflow(ff, k, e))
    GG = nx.DiGraph()
    for i in range (e):
        if input_state[i] > 0:
            GG.add_edge(Graph_edges[i][0], Graph_edges[i][1])
        elif input_state[i] < 0:
            GG.add_edge(Graph_edges[i][1], Graph_edges[i][0])
    try:
        nx.find_cycle(GG)
        x = 0
    except:
        x = 1
    return x

def phase_separator(state, C, gamma):
    e_c = np.exp(np.complex(0,-1)*gamma*C)
    return np.multiply(e_c, state)

def mixer(state, M, beta):
    e_m = expm(np.complex(0,-1)*beta*M)
    return np.matmul(e_m, state)

def evolution(istate, M, C, gamma, beta, steps):
    state2 = istate
    for i in range (steps):        
        state1 = phase_separator(state2, C, gamma)
        state2 = mixer(state1, M, beta)
    return state2

def benchmarking(result, edges, s, t, n ,k, c):
    cost =[]
    feasible_config = feasible_solution(s, t)
    for i in range (len(feasible_config)):
        cost.append(c[feasible_config[i]])
    MaxCost = max(cost)
    MinCost = min(cost)
    app = 0
    for o in feasible_config:
        app += result[o] * (MaxCost - c[o])
    appratio = app/(MaxCost - MinCost)
    return appratio


#returns all feasible_solution between s and t
def feasible_solution(s, t):
    feas = []
    paths = list(nx.all_simple_paths(G, 0, 4))
    for i in range(len(paths)):
        p = np.zeros(e)
        for j in range(len(paths[i])):
            if [paths[i][j-1], paths[i][j]] in A:
                p[A.index([paths[i][j-1], paths[i][j]])] += 1
            elif [paths[i][j], paths[i][j-1]] in A:
                p[A.index([paths[i][j], paths[i][j-1]])] += -1
        feas.append(convertstate(p))
    return feas

#returns expectation value, as defined by (Cmax - Cminimized)/(Cmax - Cmin); 
#where Cminimized  = C_i(cost of a state i) * P_i (probability of a state i) 
#only feasible solutions count
def expectation(angles, state, C, w):
    gamma = angles[0]
    beta = angles[1]
    v = complex(1,0) * np.zeros((size,size))
    for l in range(size):
        v[l][l] = np.exp(np.complex(0,-1)*beta* w[0][l])
    u_mixing = np.matmul(w[1], np.matmul(v, (np.linalg.inv(w[1]))))
    state = phase_separator(state, C, gamma)
    state = np.matmul(u_mixing, state)
    result = abs(np.multiply(state, np.conj(state)))
    return -1* abs(benchmarking(result, Graph_edges, s, t, n ,k, C))

    
''' Part III
This is a part of the code one actually call; includes comparison 
3 different mixers: x-mixer, naive lattice QED, constrainted lattice QED.
To guarantee accuracy, we first perform a grid search in the parameter space;
then a gradient decent method is applied for optimization
'''         
#before we run the the actual, QAOA ipr_test is for a test about the time evolution given 
def ipr_test():
    istate = np.zeros(size)
    istate[random.choice(feasible_solution(s, t))] = 1
    M = prepare_mixer(alldloop, loopparity, k, e)
    w = np.linalg.eigh(M)
    result=[]
    for i in range (30):
        v = complex(1,0) * np.zeros((size,size))
        for j in range(size):
            v[j][j] = np.exp(np.complex(0,-1)*i/20 * (2 * pi)* w[0][j])
        Q = np.matmul(w[1], np.matmul(v, (np.linalg.inv(w[1]))))
        fstate1 = np.matmul(Q, istate)
        x1 = np.multiply(fstate1, np.conj(fstate1)).round(10)
        result.append(x1)
    return result

def x_main():
    state = np.sqrt(1/(2*k+1)**e)*np.ones((2*k+1)**e)
    M = prepare_xmixer(k, e)
    C = prepare_xcost(k, Graph_edges, e, s, t, n, penalty)
    w = np.linalg.eigh(M)
    numpy_parameter_grid = np.mgrid[0 : granularity, 0 : granularity].astype(float);
    numpy_parameter_grid[0] = numpy_parameter_grid[0] / granularity * pi; # Beta.
    numpy_parameter_grid[1] = numpy_parameter_grid[1] / granularity * (2 * pi); # Gamma.
    expectations = np.zeros((granularity, granularity))
    maxex = [0, 0, 0]
    for i in range (granularity):
        for j in range (granularity):
            gamma = numpy_parameter_grid[1][i][j]
            beta = numpy_parameter_grid[0][i][j]
            angles = np.array([gamma, beta])
            expectations[i, j] = -1*expectation(angles, state, C, w)
            if expectations[i, j] > maxex [0]:
                maxex [0] = expectations[i, j]
                maxex [1] = gamma
                maxex [2] = beta
    arg = (state, C, w)
    x0 = np.array([maxex [1],maxex [2]])
    output = scipy.optimize.minimize(expectation, x0, arg, method='BFGS',  options={'gtol': 1e-05,  'eps': 1.4901161193847656e-08, 'maxiter': 20, 'disp': False, 'return_all': False})
    return  -1 * list(output.values())[0]

def qed_main_naive():
    state = np.zeros(size)
    state[random.choice(feasible_solution(s, t))] = 1
    M = prepare_mixer_naive(alldloop, loopparity, k, e)
    C = prepare_cost(k, e)
    w = np.linalg.eigh(M)
    numpy_parameter_grid = np.mgrid[0 : granularity, 0 : granularity].astype(float);
    numpy_parameter_grid[0] = numpy_parameter_grid[0] / granularity * pi; # Beta.
    numpy_parameter_grid[1] = numpy_parameter_grid[1] / granularity * (2 * pi); # Gamma.
    expectations = np.zeros((granularity, granularity))
    maxex = [0, 0, 0]
    m = complex(1,0) * np.zeros((size,size))
    for l in range(size):
        m[l][l] = np.exp(np.complex(0,-1)*np.pi/2* w[0][l])
    prep = np.matmul(w[1], np.matmul(m, (np.linalg.inv(w[1]))))
    state = np.matmul(prep, state)
    for i in range (granularity):
        for j in range (granularity):
            gamma = numpy_parameter_grid[0][i][j]
            beta = numpy_parameter_grid[1][i][j]
            angles = np.array([gamma, beta])
            expectations[i, j] = -1*expectation(angles, state, C, w)
            if expectations[i, j] > maxex [0]:
                maxex [0] = expectations[i, j]
                maxex [1] = i/ granularity * pi
                maxex [2] = j/ granularity * (2 * pi)
    arg = (state, C, w)
    x0 = np.array([maxex [1],maxex [2]])
    output = scipy.optimize.minimize(expectation, x0, arg, method='BFGS',  options={'gtol': 1e-05,  'eps': 1.4901161193847656e-08, 'maxiter': 20, 'disp': False, 'return_all': False})
    return  -1 * list(output.values())[0]


def qed_main():
    state = np.zeros(size)
    state[random.choice(feasible_solution(s, t))] = 1
    M = prepare_mixer(alldloop, loopparity, k, e)
    C = prepare_cost(k, e)
    w = np.linalg.eigh(M)
    numpy_parameter_grid = np.mgrid[0 : granularity, 0 : granularity].astype(float);
    numpy_parameter_grid[0] = numpy_parameter_grid[0] / granularity * pi; # Beta.
    numpy_parameter_grid[1] = numpy_parameter_grid[1] / granularity * (2 * pi); # Gamma.
    expectations = np.zeros((granularity, granularity))
    maxex = [0, 0, 0]
    m = complex(1,0) * np.zeros((size,size))
    for l in range(size):
        m[l][l] = np.exp(np.complex(0,-1)*np.pi/2* w[0][l])
    prep = np.matmul(w[1], np.matmul(m, (np.linalg.inv(w[1]))))
    state = np.matmul(prep, state)
    for i in range (granularity):
        for j in range (granularity):
            gamma = numpy_parameter_grid[0][i][j]
            beta = numpy_parameter_grid[1][i][j]
            angles = np.array([gamma, beta])
            expectations[i, j] = -1*expectation(angles, state, C, w)
            if expectations[i, j] > maxex [0]:
                maxex [0] = expectations[i, j]
                maxex [1] = i/ granularity * pi
                maxex [2] = j/ granularity * (2 * pi)
    arg = (state, C, w)
    x0 = np.array([maxex [1],maxex [2]])
    output = scipy.optimize.minimize(expectation, x0, arg, method='BFGS',  options={'gtol': 1e-05,  'eps': 1.4901161193847656e-08, 'maxiter': 10, 'disp': False, 'return_all': False})
    return  -1 * list(output.values())[0]

'''Part IV
Global variables; execute lines
as an sample, the following lines is a parallel code for outputing a 
4 loop triangle graph. The lines perform a comparison between 3 mixers 
with number of times same as the numer of cores one is using.
To change to other graphs one should change A, s, t, n as well as the condition in Dloop:
    "if len(cycle) <4"
e.g. if one uses sq lattice this should be changed to if len(cycle) < 5.
You're certainly encouraged to improve this naive condition once you get a understanding of it.
'''
#MPI import; ignore this part if not using MPI
mpi = True
if mpi == True:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() #labels your MPI tasks; in other words, the serial # of the core you're calling
    N_cores = comm.Get_size() #total # of cores
else:
    rank = 0
    N_cores = 1

t0 = time.time()
A = [[0,1],[0,2],[0,3],[1,2],[1,3],[1,4],[2,4],[3,4]]
A.sort()
Graph_edges = A
G = nx.Graph() # undirected
graph_weight = []
for i in range (len(A)):
    graph_weight.append(random.random())
    G.add_edge(A[i][0],A[i][1])
s = [0] # set of sources
t = [4] # set of sinks
k = len(s) # number of s-t pairs
e  = len(A)
n = 5# number of nodes
granularity = 15#for the 
penalty = 1 #for each x mixer violation
size = int((2*k+1)**e)
cycles = []
loopfinding()
loopparity = Dloop(cycles)[1]
alldloop = Dloop(cycles)[0]
x_result = x_main()
q_result = qed_main()
n_result = qed_main_naive()
print("AI server:#"+str(rank), time.time()-t0)

total_x = np.asarray(comm.gather(x_result, root = 0))
total_q = np.asarray(comm.gather(q_result, root = 0))
total_n = np.asarray(comm.gather(n_result, root = 0))
# allows the first core to write data to file
if rank == 0:
    np.savetxt('/home1/07017/yxzhang/QAOA/data/total_x_4loops.out',total_x)
    np.savetxt('/home1/07017/yxzhang/QAOA/data/total_q_4loops.out',total_q)
    np.savetxt('/home1/07017/yxzhang/QAOA/data/total_n_4loops.out',total_n)
    print("4loops, avg ="+str(np.avg(total_x))+", std = "+str(np.std(total_x)/np.sqrt(N_cores)))
    print("4loops, avg ="+str(np.avg(total_q))+", std = "+str(np.std(total_q)/np.sqrt(N_cores)))
    print("4loops, avg ="+str(np.avg(total_n))+", std = "+str(np.std(total_n)/np.sqrt(N_cores)))
