from cProfile import label
import colorsys
from functools import total_ordering
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from  collections import Counter 
import time
from multiprocessing import Pool
import platform


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
    os_name = platform.system()
    print(os_name)
    directory_path = os.getcwd()
    print("My current directory is : " + directory_path)
    if os_name =="Windows":
        file_names =[directory_path + "\german_highway.xlsx", directory_path + "\Openflights_airport_network_2016.xlsx"]
    else:
        file_names =[directory_path + "/german_highway.xlsx", directory_path + "/Openflights_airport_network_2016.xlsx"]
    for d in range(2):
        filename = file_names[d]
        print("My file name is : " + filename)
        data = pd.read_excel(filename)
        df = pd.DataFrame(data)
        L = []
        _max =0
        if d ==1:
            _max =(int)(df.max(numeric_only=True).max()+1)  
            L = list(zip(df["source"], df["target"]))
        else:
            _max = (int)(len(df)+1)
            for v in range(_max):
                for u in range(len(df.columns)):
                    try:
                        if df[v][u] ==1:
                            L.append((v,u))
                    except:
                        pass
        print(_max)
        N = _max # number of nodes
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


        path_actual = dict(nx.all_pairs_shortest_path_length(actual_G))
        # print(max(path))
        distances= list(path_actual.keys())
        distance_actual =[0]* (max(path_actual)+1)
        for key in distances:
            for j in path_actual[key]:
                distance_actual[j] +=1


        #config model
        config_G = nx.Graph()
        config_G = configModel([actual_G.degree(j) for j in actual_G.nodes()])
        ex = dict(Counter([config_G.degree(j) for j in config_G.nodes()]))
        node_config = []
        degreeSeq_config= list(ex.keys())

        for key in degreeSeq_config:
            node_config.append(ex[key]/(int)(df.max(numeric_only=True).max()))

        path_config = dict(nx.all_pairs_shortest_path_length(config_G))
        # print(max(path))
        distances= list(path_config.keys())
        distance_config =[0]* (max(path_config)+1)
        for key in distances:
            for j in path_config[key]:
                distance_config[j] +=1

        #randamized graph
        randamize_G = nx.Graph()
        randamize_G = degPresRand(config_G)
        ex = dict(Counter([randamize_G.degree(j) for j in randamize_G.nodes()]))
        degreeSeq_random= list(ex.keys())
        # print(degrees)
        node_random =[]
        for key in degreeSeq_random:
            node_random.append(ex[key]/(int)(df.max(numeric_only=True).max()))

        
        path_radnom = dict(nx.all_pairs_shortest_path_length(randamize_G))
        # print(max(path))
        distances= list(path_radnom.keys())
        distance_random =[0]* (max(path_radnom)+1)
        for key in distances:
            for j in path_radnom[key]:
                distance_random[j] +=1

        #visualzie graphs
        position1 = nx.circular_layout(actual_G)
        position2 = nx.circular_layout(config_G)
        position3 = nx.circular_layout(randamize_G)


        # degrees = [G.degree(i) for i in G.nodes()]
        plt.figure()    
        plt.subplot(221)
        plt.scatter(degreeSeq, node,color="red",label='Actual Grpah',alpha=0.3, edgecolors='none')
        plt.scatter(degreeSeq_config, node_config, color="blue", label="Config Model",alpha=0.3, edgecolors='none')
        plt.scatter(degreeSeq_random, node_random, color="green", label="Random Model",alpha=0.3, edgecolors='none')
        plt.yscale("log")
        plt.xlabel('Degree')
        plt.ylabel('num nodes')
        plt.title('Degree distribution')
        plt.legend()
        plt.subplot(224)
    
        plt.yscale("linear")
        plt.xlabel('Distance')
        plt.ylabel('num edges')
        plt.scatter(distance_actual, [i for i in range(len(distance_actual))], color="red",label='Actual Grpah',alpha=0.3, edgecolors='none')
        plt.scatter(distance_config, [i for i in range(len(distance_config))], color="blue",label='Config Model',alpha=0.3, edgecolors='none')
        plt.scatter(distance_random, [i for i in range(len(distance_random))], color="green",label='Random Model',alpha=0.3, edgecolors='none')
        plt.title('Distance distribution')
        plt.legend()
        plt.show()
    
 
    # # plt.subplot(223)
    
    # plt.yscale("linear")
    # plt.xlabel('Degree')
    # plt.ylabel('num nodes')
    # plt.title('randamized Graph')


def preferntal_attachment(G,m0,t,a,i,tk):
    Ki =G.degree(i)
    if tk ==0:
        tk =1
    if Ki ==0:
        Ki =1
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
    total_k =0
    for i in G.nodes():
        total_k += G.degree(i) 

    for i in range(m0,n):
        G.add_node(i) # addding a node
        for j in range(0,i-1): #decide whcih node should be conneceted to node i
            p = preferntal_attachment(G,m0,t,a,j,total_k) # get the prob for this time
            if np.random.rand() < p:
                # preferential_edges.append((i,j))
                G.add_edge(i,j)

        total_k += G.degree(m0+t-1)
        # print("total_k:", total_k)
        t+=1 # increment the time
        print("Elapsed time is ", t)
        # print("p is ", p)
    #get the probability to add an egge from v to i
    # G.add_edges_from(preferential_edges)
    return G



def quesiton3():
    n = int(math.pow(10,4))
    arpha = [0,1/2,1]
    colors = ["red","blue", "green"]
    titles = ["a=0","a=0.5","a=1"]
    m0,m, =4,4
    subplotNum =220
    pool = Pool() # make a multiprocessing 
    Graphs = pool.starmap(preferential_attachment_mixture_model, [(n,m0,m,arpha[0]),(n,m0,m,arpha[1]),(n,m0,m,arpha[2])]) # make a graph simultenously
    pool.close()

    for i in range(len(Graphs)):
        ex = dict(Counter([Graphs[i].degree(j) for j in Graphs[i].nodes()])) #ge the degree counts 
        node = []
        degreeSeq= list(ex.keys()) # get the degree distribution
        for key in degreeSeq:
            node.append(ex[key]/n)
        plt.scatter(degreeSeq, node, color=colors[i], label=titles[i],alpha=0.3, edgecolors='none')
    
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Degree')
    plt.ylabel('num nodes')
    plt.title("pref_attach_mixture")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    question2_c()
    quesiton3()    os_name = os.platform.system()
    print(os_name)


    
        
 