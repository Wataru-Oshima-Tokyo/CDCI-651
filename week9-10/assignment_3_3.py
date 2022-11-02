
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

"""
 (25 pts) Consider the largest connected component of the Facebook wall posts (2009) network and
imagine a piece of fake news spreading across the nodes. In this problem, we want to simulate
this process under different conditions and observe the effect of “immunizing” nodes in different
ways.
For R0 = 3, simulate the spread 10 times and keep track of the average fraction of infected nodes
over time with no immunization. Repeat the experiment with 10%, 30%, 50%, 70%, and 90% of
the nodes immunized following three different strategies: random immunization, immunization of
high degree nodes first, and neighbor immunization as described in class.
With the data collected from these experiments, generate three figures, one for each immunization
strategy. Each figure will have time on the x-axis and It/n (the fraction of infected nodes) on the
y-axis and five separate curves associated with the fraction of nodes immunized in each experiment.
What do you observe? Does one immunization strategy seem more effective than the others?
Why or why not?
"""

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
            #erase self loop and duplicatd edges
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
            G.add_node(u,label=label)

    return G
    
def simSIR(G, immunization, percent_imunized):
    N =len(G)
if __name__ == "__main___":
    G = readDiabetesdata
    pass