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


    
# making a graph

if __name__ == "__main__":
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