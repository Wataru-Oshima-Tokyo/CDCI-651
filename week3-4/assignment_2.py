import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from  collections import Counter 

def _stublist(degSeq):
    L =[]
    for i in range(len(degSeq)):
        for j in range(degSeq[i]):
            L.append(i)
    return L

def configModel(degSeq):
    G = nx.Graph()
    nodes =[]
    for a in range(len(degSeq)):
        nodes.append(a)
    G.add_nodes_from(nodes)
    edges =[]
    stublist = _stublist(degSeq)
    np.random.shuffle(stublist)
    n = len(degSeq)
    half = n//2
    _out, _in = stublist[:half], stublist[half:]
    G.add_edges_from(zip(_out, _in))
    return G


def degPresRand(G):
    # keys, degrees = zip(*G.degree())  # keys, degree
    # cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    # discrete_sequence = nx.utils.discrete_sequence
    # (ui, xi) = discrete_sequence(2, cdistribution=cdf)
    # if ui == xi:
    #     return G  # same source, skip
    u = np.random.choice(G.nodes())
    x = np.random.choice(G.nodes())
    # choose target uniformly from neighbors
    v = np.random.choice(G.nodes())
    y = np.random.choice(G.nodes())
    if v == y or u==x:
        return G  # same target, skip
    if (x not in G[u]) and (y not in G[v]):  # avoid creating parallel edge
        G.add_edge(u, x)
        G.add_edge(v, y)
        G.remove_edge(u, v)
        G.remove_edge(x, y)
    return G

if __name__ == "__main__":
    n = 10000 # the number of nodes
    # n =10
    max_p = 1
    step_size = 0.05
    p =0
    
    # from 0 to 1 but since the step size should be innteger 
    # so 20 = 1 /0.05 and multipy i by 0.05 inside
    for i in range(max_p*20): 
        p1 = (i+1)*0.05
        print(p1, end=" ")
        Graphs=[]
        Gccs =[]
        for t in range(10):
            degreeSeq =[] 
            for j in range(n):
                if np.random.rand() <p1:
                    degreeSeq.append(1)
                else:
                    degreeSeq.append(3)
            # print(degreeSeq)
            G = configModel(degreeSeq)
            Gccs.append(len(max(nx.connected_components(G), key=len)))
            Graphs.append(G)
        print(np.mean(Gccs))
        #it does not desapper even thogh it reaches 1
    #swap edges here
    

    
        
 