import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import combinations_with_replacement, product
import matplotlib.pyplot as plt


if __name__ == "__main__":
	if True:
		G = nx.read_gml('football.gml')
		print(G.number_of_nodes(), G.number_of_edges())

		node2vec = Node2Vec(G,dimensions=20, walk_length=15, num_walks=500, p=1,q=0.5, workers=1, seed=1234)
		model = node2vec.fit(window=15, min_count=1)
		embedding = [model.wv[str(v)] for v in sorted(G.nodes())]
		pos = nx.spring_layout(G)
		nx.draw_networkx(G, pos=pos, node_size=100, font_size=5, width=0.5)
		plt.savefig('temp_node2vec.png', bbox_inches="tight")
		plt.clf()

		tsne = TSNE(n_components=2, init='pca')
		pos = tsne.fit_transform(np.array(embedding))
		pos = {v:pos[i] for i,v in enumerate(sorted(G.nodes()))}
		nx.draw_networkx(G, pos=pos, edgelist=[], alpha=0.5, node_size=100, font_size=5,
						nodelist=sorted(G.nodes()), node_color=[G.nodes[v]['value'] for v in sorted(G.nodes())])
		plt.savefig('temp_node2vec_2.png', bbox_inches='tight')
		plt.clf()
		
		numClusters = len(set(nx.get_node_attributes(G, 'value').values()))
		kmeans = KMeans(numClusters)
		kmeans.fit(embedding)
		yHat =kmeans.labels_
		y = [G.nodes[v]['value'] for v in sorted(G.nodes() )]

		print('adjustd ran index:', metrics.adjusted_rand_score(y, yHat))
		print('silhouette coefficient:', metrics.silhouette_score(embedding,yHat))

