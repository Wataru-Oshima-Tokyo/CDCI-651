import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import combinations_with_replacement, product
import matplotlib.pyplot as plt
from node2vec import Node2Vec
def _modularity(G, C):
    """
    (5 pts) Write a function modularity(G, C) to calculate the modularity of the graph G where
    C is a dictionary with nodes as keys and community number as values.
    """
    numOfGroups = len(nx.unique(C['values']))
    comm = [None] * numOfGroups
    for v in G.nodes():
        for i in range(numOfGroups):
            if v['values'] == i:
                comm[i].append(v)
    
    m = G.edges()
    links = [0]* numOfGroups
    for v in G.nodes():
        for u in G.neighbors(v):
            for n in range(len(comm)):
                if u in comm[n]:
                    links[n] +=1
                    

    
    def modularity_value():
        

        pass

def _greedyModularity(G):



if __name__ == "__main__":
    file_path = "/Users/wataruoshima/CSCI651/week12-13/lesmis/lesmis.gml"
    G = nx.read_gml(file_path)
    p= [1,1]
    q = [0.5,2]
    k = [6,3]
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**    
    for i in range(2):
        node2vec = Node2Vec(G,dimensions=20, walk_length=15, num_walks=500, p=p[i],q=q[i], workers=1, seed=1234)
        model = node2vec.fit(window=15, min_count=1)
        embedding = [model.wv[str(v)] for v in sorted(G.nodes())]
        # numClusters = len(set(nx.get_node_attributes(G, 'label').values()))
        kmeans_model = KMeans(n_clusters=k[i]).fit(embedding)
        yHat =kmeans_model.labels_
        print("With k=%d, p=%d, and q=%.1f, the number of groups is %d" %(k[i],p[i],q[i],len(np.unique(yHat))))

        pos = nx.spring_layout(G)
        # nx.draw_networkx(G, pos=pos, node_size=100, font_size=5, width=0.5)
        file_name = 'node2vec_figure1_' + str(i) + '.png'
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
    
    # numClusters = len(set(nx.get_node_attributes(G, 'value').values()))
    # kmeans = KMeans(numClusters)
    # kmeans.fit(embedding)
    # yHat =kmeans.labels_
    # y = [G.nodes[v]['value'] for v in sorted(G.nodes() )]

    # print('adjustd ran index:', metrics.adjusted_rand_score(y, yHat))
    

    
    
    # position = nx.circular_layout(G)
    # nx.draw(G, position,with_labels = True)
    # plt.show()