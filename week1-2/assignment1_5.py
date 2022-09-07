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

def question5():
    n = 1000 # number of nodes
    numOfGraphs =10 # number of graphs
    # _p = 1/(n-1) # reasonable probability 
    x =[]
    y =[]
    k =[]
    
    c =[]
    counter =0
    for a in range(500):
        a *=0.05
        if a != 0:
            k.append(a)
    for _k in k:
        print(_k)
        p = _k/(n-1)
        _y =[]
        for a in range(numOfGraphs):
            G, Gcc, avgD = _erdos_renyi_graph(n,p)
            _y.append(Gcc/n)
        x.append(_k)
        avg = sum(_y)/len(_y)
        y.append(avg)
        print("avg: ", avg)
    print(x,y)
    plt.plot(x,y)
    # plt.yscale("log")
    #obtain m (slope) and b(intercept) of linear regression line
    # m, b = np.polyfit(x, y, 1)
    # line = [m*_x +b for _x in x]
    # #use red as color for regression line
    # plt.plot(x, line, color='red')
    plt.xlabel('<k>')
    plt.ylabel('(size of giant comp)/number of vertices')
    #Showing the result for each graph
    # for i in range(numOfGraphs):
    #     print("The graph ", end="")
    #     print(i+1)
    #     print(k[i])
    #     if c[i]:
    #         print("connected regime\n")
    #     else:
    #         print("not connected regime\n")
    plt.show()

question5()