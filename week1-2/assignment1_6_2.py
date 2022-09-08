import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from  collections import Counter 

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

def question6_2():
    # US airport networks (2010)
    directory_path = os.getcwd()
    print("My current directory is : " + directory_path)
    filename = directory_path + "\edgelist.xlsx"
    print("My file name is : " + filename)
    data = pd.read_excel(filename)
    df = pd.DataFrame(data)
    numOfGraphs = 10 # number of graphs



    print(df.max(numeric_only=True).max())
    L = list(zip(df["source"], df["target"]))
    N = (int)(df.max(numeric_only=True).max()+1)  # number of nodes
    E = len(L) # number of edges
    p = E/math.comb(N,2) # probability
    Graphs =[]
    for a in range(numOfGraphs):
        G, Gcc, avgD = _erdos_renyi_graph(N,p)
        Graphs.append(G)
        print(".", end="")
    print("")
    

    # get all the degrees for each graph
    ex = dict(Counter([G.degree(j) for j in G.nodes() for G in Graphs]))
    degrees = list(ex.keys())
    node = []
    #take the average of each degree in 10 graphs such as (degree[1][0] + degree[2][0]) /2 ...
    for key in degrees:
        node.append(ex[key]/(int)(df.max(numeric_only=True).max()))

    Ep =0
    # expected and average degree distribution
    # Since a degree is equaly treated, there is not weight to consider
    for i in degrees:
        Ep += i/len(degrees)
    print("Expected and approximate degree distribution is ", Ep) 
    # plot with various axes scales
    plt.figure()    
    plt.subplot(221)
    plt.scatter(degrees, node)
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('num nodes')
    plt.title('Coauthorships Degree Distribution')

    plt.subplot(224)
    plt.scatter(degrees, node)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Log Degree')
    plt.ylabel('log num nodes')
    plt.title('log Coauthorships Degree Distribution')
    #Showing the result for each graph
    plt.show()

    #Showing the result for each graph
    plt.show()


question6_2()