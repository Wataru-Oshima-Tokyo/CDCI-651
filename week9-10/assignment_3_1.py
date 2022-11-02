import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def get_dict_inOrder(dict):
    keys = list(dict.keys())
    temp = []
    for i in keys:
        temp.append(int(i))
    temp.sort()
    keys =[]
    for i in temp:
        keys.append(str(i))
    return ([dict[i] for i in keys])

def readFile(file_name):
    G = nx.Graph()
    labels = []
    with open(file_name, 'r') as f:
        # id name, neighbors
        w =0
        for line in f:
            line = line.strip().split() #no arg = split by space
            label = line[1].split(",")[0]
            print(label)
            labels.append(label)
            G.add_node(line[0], label=label)
            for i in range(len(line)-2):
                G.add_edge(line[0],line[i+2])
            w+=1

    return G, labels
if __name__ == "__main__":
    file_path = "/Users/wataruoshima/CSCI651/week9-10/medici_network.adjlist.txt"
    G, labels = readFile(file_name=file_path)
    position1 = nx.circular_layout(G)
    # nx.draw(G,pos=position1, with_labels = True)
    # nx.draw_networkx_labels(G, pos=position1,labels=labels)
    # plt.show()
    #show the degree centrality for each vertex
    degree_centrality = nx.degree_centrality(G)
    # harmonic centrality
    harmonic_centrality = nx.harmonic_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    for i in G.nodes():
        print("Node: %d (%s) has %lf degree centrality, %lf harmonic centrality, %lf eigenvector centrality, %lf betweenness_centrality" %(int(i), labels[int(i)], degree_centrality[i], harmonic_centrality[i],eigenvector_centrality[i], betweenness_centrality[i]))
    #For each measure, make a table with the families ranked by importance. Show these tables side by side
    real_harmonic_centrality = get_dict_inOrder(harmonic_centrality)
    """
    a.
    1. How can I defince the importance?
    2. what does the table look like?
    """
    """
    b.How important was the Medici family with respect to this network? What family was the second most important?
    Node: 8 (Medici) has 0.400000 degree centrality, 9.500000 harmonic centrality, 0.430315 eigenvector centrality, 0.452381 betweenness_centrality

    Answer. Sicne Medici famili has the higest degree centrality, it can be thought the family has the largest number of edges connected to other families and the second most important family is Guadagni family since it has 
    the second higest number for degree centrality 
    """

    """
    c.Are our measurements for harmonic centrality unusual given the degree distribution of this network? 
    Using the degree sequence of this network, generate 10,000 random networks using the configuration model. 
    Make a figure with families on the x-axis and harmonic centrality minus mean harmonic centrality on the y-axis. 

     h- h_bar
        |
        |
        |
        |
        |
        |
        - - -- -- - - -- -- - - -- -- -  
                familyt name

    Include error bars representing one standard deviation for each point. Discuss what you find.

    Answer. What I found is that the acutal data and data from the same degree sequence are similar but not actually the same and also the 
    harmonic centrality for each node is slightly above the value from degree sequence one.
    """
    #get the degree seaquence
    _degreeSeq ={}
    for i in G.nodes():
        _degreeSeq[i] = G.degree[i] 
    degreeSeq = get_dict_inOrder(_degreeSeq)
    random_harmonic_centrality = {}
    total_harmonic_centrality =0
    counter=0
    for i in range(10000):
        G = configModel(degSeq=degreeSeq)
        harmonic_centrality = nx.harmonic_centrality(G)
        for i in G.nodes():
            if i in random_harmonic_centrality:
                random_harmonic_centrality[i].append(harmonic_centrality[i])
            else:
                random_harmonic_centrality[i] = [harmonic_centrality[i]]
            total_harmonic_centrality += harmonic_centrality[i]
            counter+=1
    
    mean_harmonic_centrality = total_harmonic_centrality/counter
    stds =[]
    for i in random_harmonic_centrality:
        stds.append(np.std(random_harmonic_centrality[i]))
        random_harmonic_centrality[i] = real_harmonic_centrality[i] - np.mean(random_harmonic_centrality[i])
    print(random_harmonic_centrality)
    values = list(random_harmonic_centrality.values())
    print(mean_harmonic_centrality)

    print(labels)
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(labels, values, color="red",label='Values from Degree seq.',alpha=0.3, edgecolors='none')
    plt.errorbar(labels, values, yerr=stds, fmt="o", color="red")
    # plt.scatter(labels, real_harmonic_centrality,color="green",label='Actual data',alpha=0.3, edgecolors='none')
    # plt.yscale("log")
    plt.xlabel('Families')
    plt.ylabel('Harmonic centrality - mean of Harmonic centrality')
    plt.title('Harmonic centrality distribution')
    plt.legend()
    plt.show()


    
