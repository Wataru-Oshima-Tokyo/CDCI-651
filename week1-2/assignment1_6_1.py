from functools import total_ordering
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

# erdos_renyi graph generator
def _erdos_renyi_graph(n, p):
    # make a graph
    G = nx.Graph()
    # adding nodes according to n
    nodes =[]
    for a in range(n):
        nodes.append(a)
    G.add_nodes_from(nodes)
    #number of pairs
    # pairs = math.comb(n,2) # n*(n-1)/2
    # print(pairs)
    # adding edges
    edges =[]
    for i in range(n):
        degree = 0
        for j in range(i+1,n): # from the node itself +1 to the last node
            if np.random.rand() <p: # if the random number is bigger than input p then
                edges.append((i,j)) # make an edge between them
    aveDegree = 2*len(edges)/n #<k>
    # print("average of degree is:", aveDegree)
    G.add_edges_from(edges)
    #the size of giantcomponent
    Gcc = len(max(nx.connected_components(G), key=len))
    # print(Gcc)
    return G,Gcc,aveDegree

def question6_1():
    # US airport networks (2010)
    n = 1574 # number of nodes
    E = 28236 # number of edges
    p = E/math.comb(n,2) # probability
    G = nx.gnp_random_graph(n,p, directed=False)
    degrees = [G.degree(i) for i in G.nodes()]
    print(degrees)
    print(G.nodes())
    # plot with various axes scales
    plt.figure()    
    plt.subplot(221)
    plt.scatter(degrees, G.nodes())
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('linear')

    plt.subplot(224)
    plt.scatter(degrees, G.nodes())
    plt.yscale("log")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('log')
    #Showing the result for each graph
    plt.show()

    