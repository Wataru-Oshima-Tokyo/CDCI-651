import matplotlib.pyplot as plt
import networkx as nx
from  collections import Counter


G = nx.karate_club_graph()
G = nx.gnp_random_graph(500, 0.02)
bet_centrality = nx.betweenness_centrality(G, normalized = True, 
                                              endpoints = False)
# get all the degrees for each graph
ex = [bet_centrality[i] for i in bet_centrality]
degrees = list(ex.keys())
print(degrees)
# print(degrees)
node = []
# #take the average of each degree in 10 graphs such as (degree[1][0] + degree[2][0]) /2 ...
for key in bet_centrality:
    degrees.append(bet_centrality[key]/500)
# G is the Karate Social Graph, parameters normalized
# and endpoints ensure whether we normalize the value
# and consider the endpoints respectively.
# print(node)
# print(bet_centrality)
plt.figure()    
plt.subplot(221)
plt.scatter(degrees, node)
plt.yscale("linear")
plt.xlabel('Degree')
plt.ylabel('num nodes')
plt.title('Coauthorships Degree Distribution')



# plt.figure(figsize =(15, 15))
# nx.draw_networkx(G, with_labels = True)
plt.show()