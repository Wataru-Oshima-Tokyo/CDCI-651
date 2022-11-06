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
import copy
"""
Q3.
(25 pts) 
Given a network in which nodes have different attributes, we might be interested in
predicting node labels. For example, in a social network some users might specify their gender
while others may not. We would like to use the network structure to predict the gender of nodes
without labels. One of the most straightforward approaches to addressing this problem is known
as the “guilt by association” heuristic. Here we predict the label of a node based on the most
common label of its neighbors where ties are broken randomly. This tends to work well when the
network structure is assortative with respect to the given label.
Consider the undirected version of the PubMed Diabetes network where nodes are classified as 1
(Diabetes Mellitus, Experimental), 2 (Diabetes Mellitus Type 1), or 3 (Diabetes Mellitus Type
2). For a given p between 0 and 1, pick a random fraction p of the nodes for which to observe
labels. Predict labels for the remaining nodes using the guilt by association heuristic. Repeat this
procedure 10 times for values of p ranging from 0.1 to 0.9 in increments of 0.1 and keep track of
the average fraction of correct guesses for each p. Make a figure with p on the x-axis and average
fraction of correct labels on the y-axis.
Note that the format of the network here may be a little difficult to deal with. You will likely
need to write your own parser to use the information in the files to create a graph in NetworkX.
"""

def _erdos_renyi_graph(n, p):
    # make a graph
    labels =["1", "2", "3"]
    prob_list = [7/8,1/16,1/16]
    G = nx.Graph()
    # adding nodes according to n
    nodes =[]
    for a in range(n):
        G.add_node(a, label=np.random.choice(labels, p =prob_list))
    # G.add_nodes_from(nodes)
    #number of pairs
    pairs = math.comb(n,2) # n*(n-1)/2
    print(pairs)
    # adding edges
    edges =[]
    for i in range(n):
        for j in range(i+1,n): # from the node itself +1 to the last node
            if np.random.rand() <p: # if the random number is bigger than input p then
                edges.append((i,j)) # make an edge between them
    G.add_edges_from(edges)
    return G


def readDiabetesdata(edgeFile, labelFile):
    G = readDiabetesLabels(labelFile)

    with open(edgeFile, 'r') as f:
        #skip the first two lines
        f.readline()
        f.readline()
        for line in f:
            line = line.strip().split() #no arg = split by space
            u = line[1].split(":")[1]
            v = line[-1].split(":")[1]
            if u ==v: #avoid self loop
                continue
            nodes=[]
            for i in G.edges(u):
                nodes.append(i[1])
            #erase self loop and duplicatd edges
            if v not in nodes:
                G.add_edge(u,v)
            
    return G

def readDiabetesLabels(labelFile):
    G = nx.Graph()

    with open(labelFile, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            line = line.strip().split()
            u = line[0]
            label = line[1].split("=")[1]
            # print(label)
            G.add_node(u,label=label)

    return G
    

if __name__ == "__main__":
    edgeFile = "./pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"
    labelFile = "./pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab"
    G = readDiabetesdata(edgeFile=edgeFile, labelFile=labelFile)
    # G =_erdos_renyi_graph(1000, 0.001)
    print(len(G.nodes()))
    # position1 = nx.circular_layout(G)
    # nx.draw(G,pos=position1, with_labels = True)
    # plt.show()
    estimatiion =[]
    std_error = []
    ps=[]
    for p in np.arange(0.1,1.0,0.1):
        temp_estimate =[]
        for _ in range(10):
            Gp = copy.deepcopy(G)
            label_nodes = {}
            unobserved_nodes =[]
            observed_nodes = np.random.choice(Gp.nodes(), size=int(p*len(Gp)), replace=False)
            
            #initialize the label of all the nodes except the randomly chosen nodes
            for v in Gp.nodes():
                if v in observed_nodes:
                    # print("here")
                    continue
                else:
                    label_nodes[v] = int(Gp.nodes[v]["label"]) #store the previous label so that we can compare if the predicted label is the same the actual one
                    Gp.nodes[v]["label"] = "0"
                    unobserved_nodes.append(v)
            success_counter=0
            total_counter=0
            print("observed list: ", len(observed_nodes))
            print("unobserved list: ", len(unobserved_nodes))
            start = time.time()
            for v in unobserved_nodes:
                total_counter+=1
                labels = [0,0,0]
                # print(v, "neighbors:")
                for u in Gp.neighbors(v): #check all the neighbors of v to use guild by association heuristic
                    _label = nx.get_node_attributes(Gp, "label")[u]
                    if int(_label) == 1:
                        labels[0] +=1
                    elif int(_label) == 2:
                        labels[1] +=1
                    elif int(_label) == 3:
                        labels[2] +=1
                _max =[]
                index = labels.index(max(labels))
                _max.append(index+1)
                for i in range(len(labels)):
                    if index == i:
                        continue
                    elif labels[i] == labels[index]:
                        _max.append(i+1)
                prob = 1.0/len(_max)
                prob_list = [prob] * len(_max)
                node_label = np.random.choice(_max, p =prob_list)
                # print(node_label)
                if node_label == label_nodes[v]:
                    # print("same!")
                    success_counter+=1
                else:
                    # print("different...")
                    pass
                # time.sleep(3)
            print(p, "prob:", float(success_counter/total_counter))
            temp_estimate.append(float(success_counter/total_counter))
            end = time.time()
            print(i, " total time:",(end-start))
        estimatiion.append(np.mean(temp_estimate))
        std_error.append(np.std(temp_estimate))
        ps.append(p)
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(ps, estimatiion, color="red",label='Induced subgraph',alpha=0.3, edgecolors='none')
    plt.errorbar(ps, estimatiion, yerr=std_error, fmt="o", color="red")
    # plt.yscale("log")
    plt.xlabel('Probabilities')
    plt.ylabel('Average fraction of correct guesses')
    plt.title('Guilt by association heuristic')
    plt.legend()
    plt.show()




