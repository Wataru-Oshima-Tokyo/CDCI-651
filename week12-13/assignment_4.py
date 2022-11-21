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
def _modularity(G, C):
    """
    (5 pts) Write a function modularity(G, C) to calculate the modularity of the graph G where
    C is a dictionary with nodes as keys and community number as values.
    """
    g = 1
    numOfGroups = len(nx.unique(C['values'])) # check how many communities in the graph
    comm = [None] * numOfGroups
    for v in G.nodes():
        for i in range(numOfGroups):
            if v['values'] == i:
                comm[i].append(v) #splitting the nodes to each group
    
    m = G.edges()
    visited =[]
    links = [0] * numOfGroups
    sumOfDegrees = [0] * numOfGroups
    for v in G.nodes():
        for u in G.neighbors(v): # check all the neighbors 
            for n in range(len(comm)): # if it is in the community
                if u in comm[n]:
                    sumOfDegrees[n] += 1
                    if u not in visited: # if the node is not visited, whcih means avoiding the same egge multiple times
                        links[n] += 1    
        visited.append(v)
    Q =[]
    for i in range(numOfGroups):
        Q += (links[i]/m -g*pow(sumOfDegrees[i]/(2*m),2))
    return Q


def _greedyModularity(G):
    """
    Copied and pasted from the source code of networkx
    """
    N = G.number_of_nodes()
    weight =None
    resolution = 1
    # Count edges (or the sum of edge-weights for weighted graphs)
    # m = G.size(weight)
    m = G.edges()
    q0 = 1 / m

    # Calculate degrees (notation from the papers)
    # a : the fraction of (weighted) out-degree for each node
    # b : the fraction of (weighted) in-degree for each node
    # a = b = {node: deg * q0 * 0.5 for node, deg in G.degree(weight=weight)}
    # {a: b+c for ... } means its making a dictionary
    a = b = {node: degree * q0 * 0.5 for node, degree in G.degree()}

    
    # this preliminary step collects the edge weights for each node pair
    # It handles multigraph and digraph and works fine for graph.
    # the reason why I used default dict is to avoid getting keyerror when the dictionary does not have a key
    dq_dict = defaultdict(lambda: defaultdict(float))
    # for u, v, wt in G.edges(data=weight, default=1):
    #     if u == v:
    #         continue
    #     dq_dict[u][v] += wt
    #     dq_dict[v][u] += wt
    for u, v, wt in G.edges():
        if u == v:
            continue
        dq_dict[u][v] += wt
        dq_dict[v][u] += wt

    # now scale and subtract the expected edge-weights term
    for u, nbrdict in dq_dict.items():
        for v, wt in nbrdict.items():
            dq_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])

    # Use -dq to get a max_heap instead of a min_heap
    # dq_heap holds a heap for each node's neighbors
    dq_heap = {u: MappedQueue({(u, v): -dq for v, dq in dq_dict[u].items()}) for u in G}
    # H -> all_dq_heap holds a heap with the best items for each node
    H = MappedQueue([dq_heap[n].heap[0] for n in G if len(dq_heap[n]) > 0])

    # Initialize single-node communities
    communities = {n: frozenset([n]) for n in G}
    yield communities.values()

    # Merge the two communities that lead to the largest modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            negdq, u, v = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        # Remove best merge from row u heap
        dq_heap[u].pop()
        # Push new row max onto H
        if len(dq_heap[u]) > 0:
            H.push(dq_heap[u].heap[0])
        # If this element was also at the root of row v, we need to remove the
        # duplicate entry from H
        if dq_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            # Remove best merge from row v heap
            dq_heap[v].remove((v, u))
            # Push new row max onto H
            if len(dq_heap[v]) > 0:
                H.push(dq_heap[v].heap[0])
        else:
            # Duplicate wasn't in H, just remove from row v heap
            dq_heap[v].remove((v, u))

        # Perform merge
        # communities[v] = frozenset(communities[u] | communities[v])
        communities[v] = set(communities[u] | communities[v])
        del communities[u]

        # Get neighbor communities connected to the merged communities
        u_nbrs = set(dq_dict[u])
        v_nbrs = set(dq_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        # Update dq for merge of u into v
        for w in all_nbrs:
            # Calculate new dq value
            if w in both_nbrs:
                dq_vw = dq_dict[v][w] + dq_dict[u][w]
            elif w in v_nbrs:
                dq_vw = dq_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:  # w in u_nbrs
                dq_vw = dq_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            # Update rows v and w
            for row, col in [(v, w), (w, v)]:
                dq_heap_row = dq_heap[row]
                # Update dict for v,w only (u is removed below)
                dq_dict[row][col] = dq_vw
                # Save old max of per-row heap
                if len(dq_heap_row) > 0:
                    d_oldmax = dq_heap_row.heap[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (row, col)
                d_negdq = -dq_vw
                # Save old value for finding heap index
                if w in v_nbrs:
                    # Update existing element in per-row heap
                    dq_heap_row.update(d, d, priority=d_negdq)
                else:
                    # We're creating a new nonzero element, add to heap
                    dq_heap_row.push(d, priority=d_negdq)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d, priority=d_negdq)
                else:
                    # We've updated an entry in this row, has the max changed?
                    row_max = dq_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)

        # Remove row/col u from dq_dict matrix
        for w in dq_dict[u]:
            # Remove from dict
            dq_old = dq_dict[w][u]
            del dq_dict[w][u]
            # Remove from heaps if we haven't already
            if w != v:
                # Remove both row and column
                for row, col in [(w, u), (u, w)]:
                    dq_heap_row = dq_heap[row]
                    # Check if replaced dq is row max
                    d_old = (row, col)
                    if dq_heap_row.heap[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap_row.remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap_row) > 0:
                            H.push(dq_heap_row.heap[0])
                    else:
                        # Only update per-row heap
                        dq_heap_row.remove(d_old)

        del dq_dict[u]
        # Mark row u as deleted, but keep placeholder
        dq_heap[u] = MappedQueue()
        # Merge u into v and update a
        a[v] += a[u]
        a[u] = 0

        yield communities.values()


def question5():
    attempts = 100
    prob_total =0
    for i in range(attempts):
        total =0
        G = nx.erdos_renyi_graph(1000,0.1)
        for i in G.nodes():
            if G.degree(i)  == 95:
                total +=1
        prob = (total/1000) *100
        print("The probability of the nodes having 95 degrees is %lf percent" %(prob))
        prob_total += prob
    print("The mean probability of the nodes having 95 degrees in 100 attempts is %lf pecent" %(prob_total/attempts))

def question7_helper(G, mean_harmonic_centrality, index):
    print("Index %d started!" %(index))
    meanofHarmonic_centrality =0
    
    harmonic_centrality = nx.harmonic_centrality(G)
    for i in G.nodes():
        meanofHarmonic_centrality += harmonic_centrality[i]
    mean_harmonic_centrality = meanofHarmonic_centrality/1000
    print("Index %d Finished!" %(index))
    return mean_harmonic_centrality
    

def question7():
    ps =[p for p in np.arange(0.05,0.95,0.05)]
    harmonic_centrality_array = [0] * len(ps)
    data_list =[]
    for j,p in enumerate(ps):
            G = nx.erdos_renyi_graph(1000, p)
            data = [G, harmonic_centrality_array[j], j]
            data_list.append(data)
    pool = Pool()
    returned_data = pool.starmap(question7_helper, data_list)
    pool.close()
    values =[]
    for i in range(len(returned_data)):
        values.append(returned_data[i]/np.mean(returned_data))
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(ps, values, color="red",label='harmonic centrality',alpha=0.3, edgecolors='none')
    # plt.errorbar(ps, harmonic_centrality_total, yerr=std_error, fmt="o", color="red")
    # plt.yscale("log")
    plt.xlabel('Probabilities')
    plt.ylabel('Average fraction of harmonic centrality')
    plt.title('Question 7')
    plt.legend()
    plt.show()

def question1():
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

if __name__ == "__main__":
    question7()
    
    # numClusters = len(set(nx.get_node_attributes(G, 'value').values()))
    # kmeans = KMeans(numClusters)
    # kmeans.fit(embedding)
    # yHat =kmeans.labels_
    # y = [G.nodes[v]['value'] for v in sorted(G.nodes() )]

    # print('adjustd ran index:', metrics.adjusted_rand_score(y, yHat))
    

    
    
    # position = nx.circular_layout(G)
    # nx.draw(G, position,with_labels = True)
    # plt.show()