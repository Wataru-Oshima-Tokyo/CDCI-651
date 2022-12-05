import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import combinations_with_replacement, product
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from collections import defaultdict
from networkx.utils.mapped_queue import MappedQueue
from multiprocessing import Pool
import random


def question4():
    n = 1000
    m = 3
    m0 =5
    attempts = 5
    p = 2*m/(n-1)
    nodes = [i for i in range(n)]
    betweeness_centrality_mean_barabasi = [0 for i in range(n)]
    betweeness_centrality_mean_erdos = [0 for i in range(n)]
    # print(betweeness_centrality_mean)
    for i in range(attempts):
        G0 = nx.complete_graph(m0)
        G1 = nx.barabasi_albert_graph(n,m,initial_graph=G0)
        G2 = nx.erdos_renyi_graph(n,p)
        betweeness_centrality_barabasi = nx.betweenness_centrality(G1)
        betweeness_centrality_erdos = nx.betweenness_centrality(G2)
        for j in range(n):
            betweeness_centrality_mean_barabasi[j] += betweeness_centrality_barabasi[j]
            betweeness_centrality_mean_erdos[j] += betweeness_centrality_erdos[j]
        # print(betweeness_centrality_mean_barabasi[0]/(i+1))    

    # for i in range(len(nodes)):
    #     betweeness_centrality_mean_barabasi[i] /= attempts
    #     betweeness_centrality_mean_erdos[i] /= attempts
        # betweeness_centrality_mean_barabasi[i] /= np.mean(betweeness_centrality_mean_barabasi)
        # betweeness_centrality_mean_erdos[i] /= np.mean(betweeness_centrality_mean_erdos)

    # plt.scatter(nodes, betweeness_centrality_mean, color="red",label='Values from Degree seq.',alpha=0.3, edgecolors='none')
    plt.figure()    
    plt.subplot(221)
    plt.hist(betweeness_centrality_mean_barabasi, bins=100, density=True)
    # plt.xlim(0.0,5)
    plt.title("Betweeness centrality Histogram")
    plt.ylabel("Distribution")
    plt.xlabel("The Fraction of Betweeness Centrality")

    plt.subplot(222)
    plt.hist(betweeness_centrality_mean_erdos, density=True)
    plt.title("Betweeness centrality Distribution")
    plt.ylabel("The Fraction of Betweeness Centrality")
    plt.xlabel("Nodes")
    plt.legend()
    plt.show()

def question5():
    file_path = "/Users/wataruoshima/CSCI651/exam/power/power.gml"
    G = nx.read_gml(file_path, label=None)
    print("Number of nodes: ",len(G.nodes()))
    print("Number of edges: ",len(G.edges()))
    print("The diameter: ",nx.diameter(G))

"""
Generates and returns a second-order random walk starting from the node v with param p and q as defined in node2vec
of length n
"""
def randomWalk(G,v,n,p,q):
    #1. Get the current state according to the given v
    
    current_state = v
    previous_state = None
    randomWalk_list =[]
    randomWalk_list.append(v)
    
    for _ in range(n):
        neighbors = G.neighbors(current_state)
        probability =[]
        nodes =[]
        #2. Get the neigbors of v
        for u in neighbors:
            nodes.append(u)
            #3  set a probability for each node to visit next according to p and q
            if u != previous_state:
                probability.append(1/p)
            else:
                probability.append(1/q)
        next_node = random.choices(nodes, weights=probability,k=1)
        #4. set the current node as a previous node
        previous_state = current_state
        #5 go to the next node
        current_state = next_node[0]
        randomWalk_list.append(current_state)
        #6 iterate this procedure n times
    # finally return the random walk list
    return randomWalk_list
    
    
    
    
     

def question6():
    p = np.random.rand()
    q = np.random.rand()
    n = 100
    
    G = nx.erdos_renyi_graph(n,p)
    source = np.random.choice(G.nodes())
    print("The number of nodes is %d" %n)
    print("start from %s" %str(source))
    print("p = %lf and q = %lf" %(p,q))
    random_walk_list = randomWalk(G,source,n,p,q)
    print(random_walk_list)
    # nx.draw(G,with_labels = True)
    # plt.show()

def question7():
    G = nx.karate_club_graph()
    p= [1,1]
    q = [0.5,2]
    k = [6,3]
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**    
    for i in range(2):
        node2vec = Node2Vec(G,dimensions=20, walk_length=15, num_walks=500, p=p[i],q=q[i], workers=1, seed=1234)
        model = node2vec.fit(window=15, min_count=1)
        embedding = [model.wv[str(v)] for v in sorted(G.nodes())]
        kmeans_model = KMeans(n_clusters=k[i]).fit(embedding)
        yHat =kmeans_model.labels_
        print("With k=%d, p=%d, and q=%.1f, the number of groups is %d" %(k[i],p[i],q[i],len(np.unique(yHat))))

        pos = nx.spring_layout(G)
        # nx.draw_networkx(G, pos=pos, node_size=100, font_size=5, width=0.5)
        file_name = 'node2vec_karate_club_' + str(i) + '.png'
        # plt.savefig(file_name, bbox_inches="tight")
        # plt.clf()
        print('silhouette coefficient:', metrics.silhouette_score(embedding,yHat))
        tsne = TSNE(n_components=2, init='pca')
        pos = tsne.fit_transform(np.array(embedding))
        pos = {v:pos[i] for i,v in enumerate(sorted(G.nodes()))}
        nx.draw_networkx(G, pos=pos, edgelist=[], alpha=0.5, node_size=100, font_size=5,
                        nodelist=sorted(G.nodes()), node_color=yHat)
        plt.savefig(file_name, bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    question4()
    