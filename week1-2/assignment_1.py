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


    
# making a graph

def question4():
    # G = nx.erdos_renyi_graph(8, 1/2)
    G1 = _erdos_renyi_graph(10,0.5)
    G2 = _erdos_renyi_graph(200,0.05)
    G3 = _erdos_renyi_graph(500,0.05)
    position1 = nx.circular_layout(G1)
    position2 = nx.circular_layout(G2)
    position3 = nx.circular_layout(G3)
    nx.draw(G1, position1,with_labels = True)
    plt.show()
    nx.draw(G2, position2,with_labels = True)
    plt.show()
    nx.draw(G3, position3,with_labels = True)
    plt.show()
    # visualizing the graph
    # nx.draw(G, with_labels = True) 
    # plt.show()

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


if __name__ == "__main__":
    question4()
    question5()
    question6_1()
    question6_2()