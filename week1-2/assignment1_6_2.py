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

def question6_2():
    # US airport networks (2010)
    n = 1574 # number of nodes
    E = 28236 # number of edges
    p = E/math.comb(n,2) # probability
    # G = nx.gnp_random_graph(n,p, directed=False)
    numOfGraphs = 10 # number of graphs
    x =[]
    y =[]
    k =[]
    c =[]
    Graphs =[]
    #generate 10 ER graphs with n and e
    print(p)
    for a in range(numOfGraphs):
        G, Gcc, avgD = _erdos_renyi_graph(n,p)
        x.append(avgD)
        y.append(Gcc/n)
        Graphs.append(G)
    degrees =[]

    # get all the degrees for each graph
    for i in range(len(Graphs)):
        degrees.append([Graphs[i].degree(j) for j in Graphs[i].nodes()])
    
    avgDegrees =[]
    #take the average of each degree in 10 graphs such as (degree[1][0] + degree[2][0]) /2 ...
    for i in range(len(Graphs[0].nodes())):
        total =0
        for j in range(len(Graphs)):
            total += degrees[j][i]
        avgDegrees.append(total/len(Graphs))

    # expected and average degree distribution
    #Since a degree is equaly treated, there is not weight to consider
    Ep =0
    for i in avgDegrees:
        Ep += i/len(Graphs[0].nodes())
    # Ep += [i/len(Graphs[0].nodes()) for i in avgDegrees]
    print("Expected and approximate degree distribution is ", Ep) 
    # plot with various axes scales
    plt.figure()    
    plt.subplot(221)
    plt.scatter(avgDegrees, Graphs[0].nodes())
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('linear')

    plt.subplot(224)
    plt.scatter(avgDegrees, Graphs[0].nodes())
    plt.yscale("log")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('log')
    #Showing the result for each graph
    plt.show()

    