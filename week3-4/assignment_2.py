import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from  collections import Counter 
# from __future__ import print_function

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
    n = len(stublist)
    half = n//2
    _out, _in = stublist[:half], stublist[half:]
    G.add_edges_from(zip(_out, _in))
    return G


def degPresRand(G):
    # return G  # same source, skip
    m = len(G.edges()) # get m which is the number of edges
    index = m/2*math.log(math.pow(10,7))
    while 1: #iterate m/2*ln(10^7) times
        rng = np.random.default_rng()
        e = rng.choice(G.edges(),2, replace=False)
        u,x,v,y = e[0][0],e[1][0], e[0][1],e[1][1]
        #avoid having  self loop and multi edges
        # print(u,x,v,y)
        # print(np.random.choice(list(G[u])))
        # print(G[u].,x)
        if v == y or u==x:
            return G  # same target, skip
        # ex = dict(j for j in G[u])
        x_nodes=[]
        y_nodes=[]
        #converting dict to just array
        for i in G.edges(u):
            x_nodes.append(i[1])
        for i in G.edges(v):
            y_nodes.append(i[1])
        if (x not in x_nodes) and (y not in y_nodes):  # avoid creating parallel edge
            #rewirings
            G.add_edge(u, x)
            G.add_edge(v, y)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
        index -=1
        if index <0:
            break
    return G

def question2_a():
    n = int(math.pow(10,4)) # the number of nodes
    print(n)
    # n =10
    max_p = 1
    step_size = 0.05
    p =[]
    Gccs_mean =[]
    # from 0 to 1 but since the step size should be innteger 
    # so 20 = 1 /0.05 and multipy i by 0.05 inside
    for i in range(int(max_p*((max_p/step_size)))): 
        p1 = (i+1)*step_size
        print(p1, end=" ")
        Graphs=[]
        Gccs = []
        for t in range(10):
            degreeSeq =[] 
            for j in range(n):
                r = np.random.rand()
                if r <p1:
                    degreeSeq.append(1)
                else:
                    degreeSeq.append(3)
            # print(degreeSeq)
            G = configModel(degreeSeq)
            # G = degPresRand(G)

            Gccs.append(len(max(nx.connected_components(G), key=len)))
            Graphs.append(G)
            
        print(".")
        p.append(p1)
        Gccs_mean.append(np.mean(Gccs)/n)
        print(np.mean(Gccs)/n)
    #swap edges here
    print("")
    plt.scatter(Gccs_mean, p)
    plt.yscale("linear")
    plt.xlabel('probability of degree 1')
    plt.ylabel('Size of giant component/n')
    plt.title('linear')
    # position1 = nx.circular_layout(Graphs[0])
    # nx.draw(Graphs[0], position1,with_labels = True)
    plt.show()

def question2_c():
    directory_path = os.getcwd()
    print("My current directory is : " + directory_path)
    filename = directory_path + "\Openflights_airport_network_2016.xlsx"
    print("My file name is : " + filename)
    data = pd.read_excel(filename)
    df = pd.DataFrame(data)

    print(df.max(numeric_only=True).max())
    L = list(zip(df["source"], df["target"]))
    N = (int)(df.max(numeric_only=True).max()+1)  # number of nodes
    E = len(L) # number of edges
    p = E/math.comb(N,2) # probability

    #Actual graph
    actual_G = nx.Graph()
    actual_G.add_nodes_from([i for i in range(N)])
    actual_G.add_edges_from(L)

    #get the degree seq and distribution
    ex = dict(Counter([actual_G.degree(j) for j in actual_G.nodes()]))
    node = []
    degreeSeq= list(ex.keys())
    for key in degreeSeq:
        node.append(ex[key]/(int)(df.max(numeric_only=True).max()))


    path = dict(nx.all_pairs_shortest_path(actual_G))
    print(path)
    for i in path:
        print(len(i))
    # all_paths =[]
    # for key in paths:
    #     all_paths.append(paths[key])
    # print(all_paths)


    #config model
    config_G = nx.Graph()
    config_G = configModel([actual_G.degree(j) for j in actual_G.nodes()])
    ex = dict(Counter([config_G.degree(j) for j in config_G.nodes()]))
    node_config = []
    degreeSeq_config= list(ex.keys())

    for key in degreeSeq_config:
        node_config.append(ex[key]/(int)(df.max(numeric_only=True).max()))

    #randamized graph
    randamize_G = nx.Graph()
    randamize_G = degPresRand(config_G)
    ex = dict(Counter([randamize_G.degree(j) for j in randamize_G.nodes()]))
    degreeSeq_random= list(ex.keys())
    # print(degrees)
    node_random =[]
    for key in degreeSeq_random:
        node_random.append(ex[key]/(int)(df.max(numeric_only=True).max()))
    #visualzie graphs
    position1 = nx.circular_layout(actual_G)
    position2 = nx.circular_layout(config_G)
    position3 = nx.circular_layout(randamize_G)


    # degrees = [G.degree(i) for i in G.nodes()]
    plt.figure()    
    plt.subplot(221)
    plt.scatter(degreeSeq, node)
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('num nodes')
    plt.title('Actual Graph')
   
    plt.subplot(222)
    plt.scatter(degreeSeq_config, node_config)
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('num nodes')
    plt.title('config model Graph')

    
 
    plt.subplot(223)
    plt.scatter(degreeSeq_random, node_random)
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('num nodes')
    plt.title('randamized Graph')

    plt.show()

def preferntal_attachment(G,m0,t,a,i,tk):
    Ki =G.degree(i)
    if tk ==0:
        tk =1
    p = a*(1/(m0+t-1))+(1-a)*(Ki/tk)
    return p

def preferential_attachment_mixture_model(n,m0,m,a):
    G = nx.Graph() #make a graph
    #start node with m0 nodes
    for i in range(m0):
        G.add_node(i)
    edges=[]
    for i in range(m):
        degree = 0
        for j in range(i+1,m): # from the node itself +1 to the last node
                if np.random.rand() <a: # if the random number is bigger than input p then
                    edges.append((i,j))
    G.add_edges_from(edges)
    
    # from here we use the probability of preferential attachment
    t= 1
    preferential_edges =[]
    total_k = G.degree(1)
    for i in range(m0,n):
        G.add_node(i)
        for j in range(0,i-1):
            if np.random.rand() < preferntal_attachment(G,m0,t,a,j,total_k):
                preferential_edges.append((i,j))
        total_k += G.degree(m0+t-1)
        t+=1 # increment the time
        print("Elapsed time is ", t)
    #get the probability to add an egge from v to i
    G.add_edges_from(preferential_edges)
    return G


def example():
    directory_path = os.getcwd()
    print("My current directory is : " + directory_path)
    filename = directory_path + "\Openflights_airport_network_2016.xlsx"
    print("My file name is : " + filename)
    data = pd.read_excel(filename)
    df = pd.DataFrame(data)

    print(df.max(numeric_only=True).max())
    L = list(zip(df["source"], df["target"]))
    N = (int)(df.max(numeric_only=True).max()+1)  # number of nodes
    E = len(L) # number of edges
    p = E/math.comb(N,2) # probability

    #Actual graph
    actual_G = nx.Graph()
    actual_G.add_nodes_from([i for i in range(N)])
    actual_G.add_edges_from(L)

    #get the degree seq and distribution
    ex = dict(Counter([actual_G.degree(j) for j in actual_G.nodes()]))
    node = []
    degreeSeq= list(ex.keys())
    for key in degreeSeq:
        node.append(ex[key]/(int)(df.max(numeric_only=True).max()))


    path = dict(nx.all_pairs_shortest_path(actual_G))
    # print(path)
    keys = list(path.keys())
    # print(keys)
    ditance_dist = []
    edges =[]
    for i in keys:
        ditance_dist.append(len(path[i]))
        edges.append(i)
    
    plt.scatter(edges, ditance_dist)
    plt.yscale("linear")
    plt.xlabel('indx of edges')
    plt.ylabel('distance')
    plt.title('actual Graph')

    plt.show()


if __name__ == "__main__":
    # question2_c()
    # example()
    n = int(math.pow(10,4))
    arpha = [0,1/2,1]
    m0,m, =4,4
    subplotNum =220
    plt.figure()
    for i in range(len(arpha)):
        G1 = preferential_attachment_mixture_model(n,m0,m,arpha[i])
        ex = dict(Counter([G1.degree(j) for j in G1.nodes()]))
        node = []
        degreeSeq= list(ex.keys())
        for key in degreeSeq:
            node.append(ex[key]/n)
        subplotNum += 1
        title = "arpha: " + str(arpha[i])
        plt.subplot(subplotNum)
        plt.scatter(degreeSeq, node)
        plt.yscale("linear")
        plt.xlabel('Degree')
        plt.ylabel('num nodes')
        plt.title(title)

    plt.show()
    
        
 