from functools import total_ordering
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import os

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
    # Network scientist coauthorships (2019)
    #loading the data set
    #get the current directory
    directory_path = os.getcwd()
    print("My current directory is : " + directory_path)
    filename = directory_path + "\edgelist.xlsx"
    print("My file name is : " + filename)
    data = pd.read_excel(filename)
    df = pd.DataFrame(data)

    print(df.max(numeric_only=True).max())
    L = list(zip(df["source"], df["target"]))
    N = (int)(df.max(numeric_only=True).max()+1)  # number of nodes
    E = len(L) # number of edges
    p = E/math.comb(N,2) # probability
    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    G.add_edges_from(L)
    # print(G.degree())

    # G = nx.gnp_random_graph(n,p, directed=False)
    degrees = [G.degree(i) for i in G.nodes()]
    # print(degrees)
    # print(G.nodes())
    # # plot with various axes scales
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
    plt.xscale("log")
    plt.xlabel('Log Degree')
    plt.ylabel('index of nodes')
    plt.title('log')
    #Showing the result for each graph
    plt.show()

question6_1()