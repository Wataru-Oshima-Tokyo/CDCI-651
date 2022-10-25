
from cProfile import label
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